// src/mimo/mimo_window_quality.cpp - real-model teacher-forced layout/window-CPU check.
//
// NOT a quality or performance evaluation: synthetic fixed token ids through the existing
// run_verify_logits seam (ctx 128, chunk 8, no T1 calibration, no peer, cache forced zero
// for deterministic placement).  Four arms run as FRESH CHILD PROCESSES of this binary so
// every process-static kernel flag is the same explicit value in all arms; the ONLY deltas
// are STRATA_MIMO_BAND8 (storage: native source vs the shared packed cache) and
// STRATA_MIMO_WINDOW_CPU (0 vs 1, groups 2 min 1 - the same CPU selection in both layouts).
//
//   A (storage, gate):  native/wcpu0 vs band8/wcpu0 logits must be BITWISE identical.
//   B (window CPU, gate): native/wcpu1 vs band8/wcpu1 logits must be BITWISE identical
//      (identical CPU selection and CPU arithmetic across layouts).
//   Cross (report only): native wcpu0 vs wcpu1 MAY differ (CPU int8 vs GPU half on the
//      CPU-selected rows) - diff count, max abs, rel RMS and per-row top1/margin are
//      REPORTED, never waived as model quality.
//
// Evidence per arm (stderr logs): window_cpu forward jobs/groups/rows and the CPU tier's
// repack_fills/cache_bytes/native_reads - the packed+wcpu1 arm must show jobs/rows > 0 AND
// fills=0/cache=0/native_reads=0 (canonical reads off the payload, no runtime copies).
// The cache generation is REUSED: build_or_open matches the fingerprint the CPU helper
// verified (0.2 s); if the engine ever started a new ~80 GiB generation here that is a
// STOP condition, not a fallback.
//
// Usage: mimo_window_quality --model <dir> --cache <root> --out <dir> [--widths 2,4]
//   Child contract: --child <arm> <model> <out-file> <width>... (exit 0, writes F32 logits).
#include "strata/mimo/engine.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
    if (ok) { std::printf("PASS %s\n", what); return; }
    ++g_failures;
    std::printf("FAIL %s\n", what);
}

const char* arm_env(const char* arm) {
    if (std::strcmp(arm, "native-wcpu0") == 0)
        return "STRATA_MIMO_BAND8=0 STRATA_MIMO_WINDOW_CPU=0";
    if (std::strcmp(arm, "band8-wcpu0") == 0)
        return "STRATA_MIMO_BAND8=1 STRATA_MIMO_WINDOW_CPU=0";
    if (std::strcmp(arm, "native-wcpu1") == 0)
        return "STRATA_MIMO_BAND8=0 STRATA_MIMO_WINDOW_CPU=1 "
               "STRATA_MIMO_WINDOW_CPU_MAX_GROUPS=2 STRATA_MIMO_WINDOW_CPU_MIN_GROUPS=1";
    return "STRATA_MIMO_BAND8=1 STRATA_MIMO_WINDOW_CPU=1 "
           "STRATA_MIMO_WINDOW_CPU_MAX_GROUPS=2 STRATA_MIMO_WINDOW_CPU_MIN_GROUPS=1";
}

// Every process-static flag, identical in all four arms (the common kernel configuration
// the current launcher uses: RDNA=1, QFUSE=3, native grouped=1, peer padded/multirow,
// attention split, QKV share, q8 prefill columns 8, dense q8, small-M, primary multirow 2).
// LD_LIBRARY_PATH and the rest of the parent environment carry through unchanged.
const char* common_env() {
    return "STRATA_MIMO_ATTN_SPLIT=1 STRATA_MIMO_QKV_SHARE=1 STRATA_MIMO_Q8_PREFILL_COLS=8 "
           "STRATA_MIMO_DENSE_Q8=1 STRATA_MIMO_DECODE_DEVICE=1 STRATA_MIMO_DECODE_HANDOFF=1 "
           "STRATA_MIMO_VERIFY_HOST=1 STRATA_MIMO_WINDOW_HANDOFF=1 "
           "STRATA_EXL3_SMALL_M=1 STRATA_EXL3_PRIMARY_MULTIROW=2 STRATA_EXL3_PEER_MULTIROW=4 "
           "STRATA_EXL3_PEER_PADDED=1 STRATA_EXL3_RDNA=1 STRATA_EXL3_QFUSE=3 "
           "STRATA_EXL3_NATIVE_GROUPED=1 "
           "STRATA_EXL3_CPU_TIER=1 STRATA_EXL3_CPU_CACHE_MIB=0 STRATA_MIMO_CPU_WORKERS=8 "
           "STRATA_MIMO_TEST_EXPERT_CACHE_ZERO=1 STRATA_MIMO_WINDOW_CPU_STATS=1";
}

int child_main(int argc, char** argv) {   // --child <arm> <model> <out-file> <width>...
    const char* arm = argc > 2 ? argv[2] : "";
    const char* model = argc > 3 ? argv[3] : "";
    const char* out = argc > 4 ? argv[4] : "";
    std::vector<int> widths;
    for (int i = 5; i < argc; ++i) widths.push_back(std::atoi(argv[i]));
    std::vector<float> logits;
    std::string err;
    const int rc = strata::mimo::run_verify_logits(model, widths, logits, err);   // no T1 calib
    if (rc != 0 || logits.empty()) {
        std::printf("CHILD_FAIL rc=%d err=%s\n", rc, err.c_str());
        return 1;
    }
    FILE* f = std::fopen(out, "wb");
    if (f == nullptr) return 1;
    const uint64_t n = logits.size();
    std::fwrite(&n, 8, 1, f);
    std::fwrite(logits.data(), 4, logits.size(), f);
    std::fclose(f);
    std::printf("CHILD_OK arm=%s floats=%llu\n", arm, (unsigned long long) n);
    return 0;
}

bool read_logits(const std::string& path, std::vector<float>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    uint64_t n = 0;
    if (std::fread(&n, 8, 1, f) != 1 || n == 0 || n > (1ull << 28)) { std::fclose(f); return false; }
    out.resize((size_t) n);
    const bool ok = std::fread(out.data(), 4, out.size(), f) == out.size();
    std::fclose(f);
    return ok;
}

struct Diff {
    size_t count = 0, n = 0;
    double max_abs = 0, rel_rms = 0;
    std::vector<size_t> top1_changes;
    std::vector<double> margin_a, margin_b;
};
Diff compare(const std::vector<float>& a, const std::vector<float>& b, size_t vocab, size_t rows) {
    Diff d;
    d.n = a.size();
    double e2 = 0, r2 = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double dd = (double) a[i] - b[i];
        if (dd != 0) ++d.count;
        e2 += dd * dd; r2 += (double) b[i] * b[i];
        d.max_abs = std::max(d.max_abs, std::abs(dd));
    }
    d.rel_rms = std::sqrt(e2 / std::max(r2, 1e-30));
    for (size_t r = 0; r < rows; ++r) {
        size_t t1a = 0, t1b = 0;
        double m1a = -1e30, m2a = -1e30, m1b = -1e30, m2b = -1e30;
        for (size_t j = 0; j < vocab; ++j) {
            const double va = a[r * vocab + j], vb = b[r * vocab + j];
            if (va > m1a) { m2a = m1a; m1a = va; t1a = j; } else if (va > m2a) m2a = va;
            if (vb > m1b) { m2b = m1b; m1b = vb; t1b = j; } else if (vb > m2b) m2b = vb;
        }
        if (t1a != t1b) d.top1_changes.push_back(r);
        d.margin_a.push_back(m1a - m2a);
        d.margin_b.push_back(m1b - m2b);
    }
    return d;
}

void report(const char* name, const Diff& d) {
    std::printf("%s: floats=%zu differing=%zu max_abs=%.6g rel_rms=%.6g top1_changes=%zu\n",
                name, d.n, d.count, d.max_abs, d.rel_rms, d.top1_changes.size());
    for (size_t r = 0; r < d.margin_a.size() && r < 6; ++r)
        std::printf("  row %zu margin cpu-off=%.6g cpu-on=%.6g%s\n", r, d.margin_a[r], d.margin_b[r],
                    std::find(d.top1_changes.begin(), d.top1_changes.end(), r) != d.top1_changes.end()
                        ? " TOP1-CHANGED" : "");
}

bool stats_ok(const std::string& log, bool packed, bool want_jobs, std::string& line) {
    // The strongest evidence line: any forward with jobs>0; for the packed arm it (and every
    // other forward line) must carry repack_fills=0 cache_bytes=0 native_reads=0.
    auto key = [](const char* text, const char* name, long long& out) {
        const char* p = std::strstr(text, name);
        out = p != nullptr ? std::atoll(p + std::strlen(name)) : -1;
    };
    bool any_jobs = false, all_canonical = true;
    std::FILE* f = std::fopen(log.c_str(), "r");
    if (f == nullptr) return false;
    char buf[1024];
    while (std::fgets(buf, sizeof buf, f) != nullptr) {
        if (std::strstr(buf, "window_cpu forward") == nullptr) continue;
        long long jobs = 0, fills = 0, cache = 0, nat = 0;
        key(buf, "jobs=", jobs);
        key(buf, "repack_fills=", fills);
        key(buf, "cache_bytes=", cache);
        key(buf, "native_reads=", nat);
        if (jobs > 0) { any_jobs = true; line = buf; }
        if (packed && (fills != 0 || cache != 0 || nat != 0)) all_canonical = false;
    }
    std::fclose(f);
    return any_jobs && all_canonical && (!want_jobs || any_jobs);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--child") == 0) return child_main(argc, argv);
    std::string model, cache, out = "notes/results/2026-11-06-canonical-band8-model/quality";
    std::vector<int> widths{2, 4};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--model") model = next();
        else if (a == "--cache") cache = next();
        else if (a == "--out") out = next();
        else if (a == "--widths") {
            widths.clear();
            const std::string v = next();
            for (size_t p = 0, q; p < v.size(); p = q + 1) {
                q = v.find(',', p);
                widths.push_back(std::atoi(v.substr(p, q - p).c_str()));
                if (q == std::string::npos) break;
            }
        }
    }
    if (model.empty() || cache.empty()) {
        std::fprintf(stderr, "usage: %s --model <dir> --cache <root> [--out <dir>] [--widths 2,4]\n",
                     argv[0]);
        return 2;
    }
    fs::create_directories(out);
    const char* arms[4] = {"native-wcpu0", "band8-wcpu0", "native-wcpu1", "band8-wcpu1"};
    const std::string self = fs::read_symlink("/proc/self/exe").string();
    for (const char* arm : arms) {
        std::string cmd = "timeout 170 env ";
        cmd += common_env();
        cmd += " STRATA_EXL3_BAND8_CACHE='";
        cmd += cache;
        cmd += "' ";
        cmd += arm_env(arm);
        cmd += " '" + self + "' --child " + std::string(arm) + " '" + model + "' '";
        cmd += out + "/logits-" + arm + ".f32'";
        for (int w : widths) cmd += " " + std::to_string(w);
        cmd += " > '" + out + "/" + arm + ".log' 2>&1";
        const int rc = std::system(cmd.c_str());
        const int status = rc == -1 ? -1 : WEXITSTATUS(rc);
        std::printf("arm %s exit=%d log=%s/%s.log\n", arm, status, out.c_str(), arm);
        if (status != 0) {
            ++g_failures;
            std::printf("FAIL arm %s (exit %d; see its log)\n", arm, status);
        }
    }
    std::vector<float> a, b, c, d;
    const bool ok_read = read_logits(out + "/logits-native-wcpu0.f32", a) &&
                         read_logits(out + "/logits-band8-wcpu0.f32", b) &&
                         read_logits(out + "/logits-native-wcpu1.f32", c) &&
                         read_logits(out + "/logits-band8-wcpu1.f32", d);
    check(ok_read && a.size() == b.size() && b.size() == c.size() && c.size() == d.size() &&
          !a.empty(), "four arms produced equal-length logits vectors");
    if (ok_read && a.size() == b.size() && !a.empty()) {
        const size_t rows = widths.size();
        const size_t vocab = a.size() / rows;
        check(std::memcmp(a.data(), b.data(), a.size() * 4) == 0,
              "A storage: native vs band8, CPU window OFF - bitwise identical");
        check(std::memcmp(c.data(), d.data(), c.size() * 4) == 0,
              "B window CPU: native vs band8, CPU window ON (groups 2) - bitwise identical");
        const Diff cross = compare(a, c, vocab, rows);
        report("cross native cpu-off vs cpu-on (reported, not a gate)", cross);
        std::string jobs_line;
        check(stats_ok(out + "/band8-wcpu1.log", true, true, jobs_line),
              "packed + window CPU: jobs/rows > 0 and canonical fills=0 cache=0 native_reads=0");
        std::printf("evidence: %s", jobs_line.c_str());
        check(stats_ok(out + "/native-wcpu1.log", false, true, jobs_line),
              "native + window CPU: jobs/rows > 0 (native tier may read native bytes)");
        std::printf("evidence: %s", jobs_line.c_str());
        std::string off_line;
        stats_ok(out + "/native-wcpu0.log", false, false, off_line);
        std::printf("cpu-off arm window_cpu lines (expect none/jobs0): %s",
                    off_line.empty() ? "(no jobs)\n" : off_line.c_str());
    }
    std::printf("%s\n", g_failures ? "WINDOW QUALITY CHECKS FAILED" : "WINDOW QUALITY CHECKS PASSED");
    return g_failures ? 1 : 0;
}
