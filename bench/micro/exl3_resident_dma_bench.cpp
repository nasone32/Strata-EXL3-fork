// bench/micro/exl3_resident_dma_bench.cpp - R1 prototype's static route-replay microbench.
//
// Measures EXACTLY the staging-transport delta the opt-in STRATA_EXL3_RESIDENT_DIRECT_DMA
// changes, on the REAL model's bytes and geometry, with no engine, server or drafter in
// the loop: one resident backing (real cudaHostRegister, the exl3_resident_host_options
// registrar, the machine's REAL available-RAM guard) is loaded for a layer SUBSET of the
// real packed cache, then two Exl3ExpertStreams over that SAME backing replay one fixed
// miss schedule each - one constructed with the flag unset (today's gather transport:
// memcpy arena -> pinned slab -> one DMA), one with STRATA_EXL3_RESIDENT_DIRECT_DMA=1
// (one DMA straight from the registered arena).  Passes alternate AB, BA, AB, BA (armed
// arms warmed first) so a monotonic drift cancels in the paired means; every pass stages
// the same experts through the same destination layouts, and EVERY call's output rows are
// compared BITWISE between the arms (memcmp over finite floats) - a work-count mismatch
// is a hard failure, not a print.
//
// Reported per pass: wall ms, staged experts/MiB (work parity), the engine's own staging
// JOIN ms (STRATA_EXL3_CACHE_STATS stage_host_ms deltas), and the direct arm's counters
// (jobs/bytes/fallback - the fast path must not be vacuous).  The header lines record the
// exact command, the binary's sha256, the STRATA_* environment and MemAvailable, so a log
// is self-describing.
//
// STRICTLY READ-ONLY on the packed cache: the store is opened with allow_build=false, so a
// missing/mismatched generation REFUSES instead of becoming an implicit ~80 GiB pack job -
// the artifact-side test exl3_band8_store_test covers that refusal; this bench never
// writes.  Needs one GPU.  Not a benchmark of the engine: no router, drafter, peer or CPU
// tier runs here, and NO end-to-end engine speed claim may be derived from its numbers
// (the staging here is deliberately the dominant term).

#include "strata/artifact/exl3_band8_store.hpp"
#include "strata/artifact/exl3_model.hpp"
#include "strata/artifact/exl3_resident.hpp"
#include "strata/artifact/mimo_pack.hpp"
#include "strata/core/mimo_geometry.hpp"
#include "strata/kernels/exl3_experts.hpp"
#include "strata/kernels/exl3_resident_host.hpp"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

extern "C" {
extern char** environ;
}

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// The REAL sha256 of THIS binary.  /proc/self/exe must be resolved in the PARENT: passing
// the literal path through a shell hands the CHILD its own /proc/self/exe (= sha256sum
// itself), which is exactly the bug this replaced.  Resolution happens here first, then
// fork/execvp feeds the absolute path to sha256sum as a single argv entry - no shell, no
// quoting, nothing injectable.  Falls back to size+mtime when the tool is unavailable.
std::string binary_identity() {
    std::error_code ec;
    const auto self = std::filesystem::canonical("/proc/self/exe", ec);
    if (ec || self.empty()) return "identity:unresolvable";
    int pipefd[2];
    if (::pipe(pipefd) != 0) return "identity:no-pipe";
    const pid_t pid = ::fork();
    if (pid == 0) {   // child: sha256sum <resolved path>, stdout into the pipe
        ::close(pipefd[0]);
        ::dup2(pipefd[1], STDOUT_FILENO);
        ::close(pipefd[1]);
        const char* argv[3] = {"sha256sum", self.c_str(), nullptr};
        ::execvp("sha256sum", const_cast<char* const*>(argv));
        _exit(127);
    }
    ::close(pipefd[1]);
    std::string out;
    char buf[512];
    ssize_t n;
    while ((n = ::read(pipefd[0], buf, sizeof buf)) > 0) out.append(buf, (size_t) n);
    ::close(pipefd[0]);
    int status = 0;
    ::waitpid(pid, &status, 0);
    const size_t space = out.find(' ');
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0 && space >= 64)
        return "sha256:" + out.substr(0, space);
    const auto size = std::filesystem::file_size(self, ec);
    const auto mtime = std::filesystem::last_write_time(self, ec).time_since_epoch().count();
    char identity[160];
    std::snprintf(identity, sizeof identity, "size+mtime:%llu:%lld",
                  (unsigned long long) size, (long long) mtime);
    return identity;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <model_dir> <band8_cache_root> <layer_lo> <layer_hi> "
                             "[passes=6] [experts_per_call=8] [slots=4]\n", argv[0]);
        return 2;
    }
    const std::string model = argv[1], cache = argv[2];
    const int layer_lo = std::atoi(argv[3]), layer_hi = std::atoi(argv[4]);
    const int passes = argc > 5 ? std::atoi(argv[5]) : 6;
    const int per_call = argc > 6 ? std::atoi(argv[6]) : 8;
    const int slots = argc > 7 ? std::atoi(argv[7]) : 4;
    if (layer_lo < 1 || layer_hi <= layer_lo || passes <= 0 || per_call <= 0 || slots <= 0 ||
        passes > 64 || per_call > 256 || layer_hi - layer_lo > 48) {
        std::fprintf(stderr, "bench: invalid range/counts\n");
        return 2;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("# bench: %s %s %d %d %d %d %d\n", model.c_str(), cache.c_str(),
                layer_lo, layer_hi, passes, per_call, slots);
    std::printf("# binary: %s\n", binary_identity().c_str());
    for (int i = 1; i < argc; ++i) std::printf("# argv[%d]=%s\n", i, argv[i]);
    if (environ != nullptr)
        for (char* const* e = environ; *e != nullptr; ++e)
            if (std::strncmp(*e, "STRATA_", 7) == 0 || std::strncmp(*e, "HIP_", 4) == 0 ||
                std::strncmp(*e, "LD_", 3) == 0)
                std::printf("# env %s\n", *e);
    { std::ifstream mi("/proc/meminfo"); std::string line;
      while (std::getline(mi, line)) if (line.rfind("MemAvailable:", 0) == 0) std::printf("# %s\n", line.c_str()); }

    // The engine's exact packed-cache request list (mimo_engine.cu): every MoE layer's
    // every expert, three projections, in that order - the SAME list, and the open is
    // READ-ONLY (allow_build=false): the generation must already exist or the bench fails
    // loudly; it never creates, locks or partials anything under the cache root.
    std::ifstream config_in(model + "/config.json", std::ios::binary);
    std::string config((std::istreambuf_iterator<char>(config_in)), std::istreambuf_iterator<char>());
    strata::core::MimoGeometry geo = strata::core::mimo_geometry_from_config(config);
    if ((int) geo.n_expert < per_call || layer_hi > (int) geo.n_layers) {
        std::fprintf(stderr, "bench: per_call/experts or layer range outside this model\n");
        return 2;
    }
    std::vector<std::string> prefixes;
    {
        int64_t moe = 0;
        for (int64_t l = 0; l < geo.n_layers; ++l)
            if (geo.is_moe(l)) ++moe;
        prefixes.reserve((size_t) moe * (size_t) geo.n_expert * 3);
        for (int64_t l = 0; l < geo.n_layers; ++l) {
            if (!geo.is_moe(l)) continue;
            for (int64_t e = 0; e < geo.n_expert; ++e)
                for (const char* proj : {"gate", "up", "down"})
                    prefixes.push_back(strata::mimo_expert_tensor((int) l, (int) e, proj));
        }
    }
    const auto t_open = Clock::now();
    strata::Exl3Band8BuildOptions ro;
    ro.allow_build = false;   // READ-ONLY: a missing generation refuses, never rebuilds
    auto store = strata::Exl3Band8Store::build_or_open(model, cache, prefixes, ro);
    std::error_code fec;
    const auto payload = std::filesystem::file_size(store->payload_path(), fec);
    std::printf("store: %zu records, %.2f GiB payload, fingerprint %016llx (%.1f s)\n",
                store->record_count(), (double) payload / 1073741824.0,
                (unsigned long long) store->fingerprint(), ms_since(t_open) / 1000.0);

    // The resident subset: only the benchmarked layers load into pinned RAM.  The RAM guard
    // is the REAL one (the registrar's own available-memory check, no override), so a
    // machine without headroom for the subset refuses instead of overcommitting.
    std::vector<strata::Exl3ResidentBacking::Group> groups;
    for (int l = layer_lo; l < layer_hi; ++l) {
        if (!geo.is_moe(l)) continue;
        for (int e = 0; e < (int) geo.n_expert; ++e)
            groups.push_back({l, e, {strata::mimo_expert_tensor(l, e, "gate"),
                                     strata::mimo_expert_tensor(l, e, "up"),
                                     strata::mimo_expert_tensor(l, e, "down")}});
    }
    auto options = strata::kernels::exl3_resident_host_options();
    options.progress = nullptr;   // keep the default RAM/headroom discipline
    const auto t_ram = Clock::now();
    auto ram = strata::Exl3ResidentBacking::create(model, groups, store);
    ram->activate({}, options);
    const auto st = ram->stats();
    std::printf("resident: %zu experts, %.2f GiB locked/committed (%.1f s)\n",
                groups.size(), (double) st.locked_bytes / 1073741824.0, ms_since(t_ram) / 1000.0);

    const int embd = (int) geo.n_embd;
    std::vector<float> x((size_t) embd);
    for (int i = 0; i < embd; ++i) x[(size_t) i] = 0.02f * (float) std::sin((double) i);
    std::vector<float> out((size_t) per_call * embd, 0.0f);
    std::vector<int32_t> ids((size_t) per_call);

    auto make = [&]() {
        return std::make_unique<strata::kernels::Exl3ExpertStream>(
            model, embd, (int) geo.n_layers, slots, nullptr, 0, (size_t) 512 << 20,
            layer_lo, layer_hi, 8, "model.layers.", ram);
    };
    // Arm A: the flag unset (the default gather transport).  Arm B: the opt-in, constructed
    // under the flag.  (The flag is a CONSTRUCTION-time read.)
    unsetenv("STRATA_EXL3_RESIDENT_DIRECT_DMA");
    auto a = make();
    setenv("STRATA_EXL3_RESIDENT_DIRECT_DMA", "1", 1);
    auto b = make();
    unsetenv("STRATA_EXL3_RESIDENT_DIRECT_DMA");
    bool on_flag = true;
    b->resident_direct_dma_debug(nullptr, nullptr, nullptr, &on_flag);
    bool off_flag = false;
    a->resident_direct_dma_debug(nullptr, nullptr, nullptr, &off_flag);
    if (!on_flag || off_flag) { std::fprintf(stderr, "bench: arms did not arm as intended\n"); return 1; }

    // One pass = every benchmarked layer stages `per_call` DISTINCT experts (a window
    // complement's shape).  Pass p reads experts [(p*per_call) % window, +per_call): all
    // misses at `slots` slots/layer, identical for both arms.  NOTE the WARM pass below
    // pre-admits `slots` experts per layer, so measured pass 0 stages slightly fewer
    // experts than later passes - identically in both arms, which the parity checks verify.
    const int n_expert = (int) geo.n_expert;
    const int calls_per_pass = layer_hi - layer_lo;
    auto one_pass = [&](strata::kernels::Exl3ExpertStream& s, int pass,
                        std::vector<std::vector<float>>* rows) {
        for (int l = layer_lo; l < layer_hi; ++l) {
            if (!geo.is_moe(l)) continue;
            const int window = std::max(1, n_expert - per_call);
            const int base = (pass * per_call) % window;
            for (int i = 0; i < per_call; ++i) ids[(size_t) i] = (base + i) % n_expert;
            if (!s.run_multi(l, x.data(), ids.data(), 1, per_call, out.data(), nullptr)) {
                std::fprintf(stderr, "bench: run_multi failed at layer %d pass %d\n", l, pass);
                return false;
            }
            if (rows) rows->push_back(out);
        }
        return true;
    };
    struct Sample { double wall_ms, join_ms; uint64_t experts, bytes; };
    auto measure = [&](strata::kernels::Exl3ExpertStream& s, int pass, Sample& out_sample,
                       std::vector<std::vector<float>>* rows) {
        const auto before = s.cache_stats();
        const auto t0 = Clock::now();
        if (!one_pass(s, pass, rows)) return false;
        out_sample.wall_ms = ms_since(t0);
        const auto after = s.cache_stats();
        out_sample.join_ms = after.stage_host_ms - before.stage_host_ms;
        out_sample.experts = after.staged_experts - before.staged_experts;
        out_sample.bytes = after.staged_bytes - before.staged_bytes;
        return true;
    };
    // Warm both arms (first-touch, JIT, worker pools) outside the measurement.  The warm
    // pass also leaves each arm's cache holding the same `slots` experts per layer, so the
    // measured passes see identical miss patterns.
    {
        Sample warm{};
        if (!measure(*a, 0, warm, nullptr) || !measure(*b, 0, warm, nullptr)) return 1;
    }
    std::vector<Sample> sa((size_t) passes), sb((size_t) passes);
    std::vector<std::vector<float>> rows_a, rows_b;
    // Strict pairwise AB, BA, AB, BA: pass p runs the SAME expert window in both arms,
    // with the arm ORDER alternating so a monotonic machine drift cancels in the means.
    for (int p = 0; p < passes; ++p) {
        const bool direct_first = p % 2 == 1;
        const size_t mark_a = rows_a.size(), mark_b = rows_b.size();
        const bool ran = direct_first
                ? (measure(*b, p, sb[(size_t) p], &rows_b) && measure(*a, p, sa[(size_t) p], &rows_a))
                : (measure(*a, p, sa[(size_t) p], &rows_a) && measure(*b, p, sb[(size_t) p], &rows_b));
        if (!ran) return 1;
        // BITWISE output parity for every call of this pass (memcmp over finite floats):
        // the transports may differ, the computed rows may not.
        if (rows_a.size() - mark_a != rows_b.size() - mark_b) {
            std::fprintf(stderr, "bench: pass %d call-count mismatch %zu vs %zu\n", p,
                         rows_a.size() - mark_a, rows_b.size() - mark_b);
            return 1;
        }
        for (size_t r = 0; r < rows_a.size() - mark_a; ++r) {
            const auto& u = rows_a[mark_a + r];
            const auto& v = rows_b[mark_b + r];
            bool finite = true;
            for (float f : u) finite = finite && std::isfinite(f);
            if (!finite || u.size() != v.size() ||
                    std::memcmp(u.data(), v.data(), u.size() * sizeof(float)) != 0) {
                std::fprintf(stderr, "bench: pass %d call %zu rows differ (or non-finite)\n", p, r);
                return 1;
            }
        }
        // Work parity is a REQUIREMENT, not an observation.
        if (sa[(size_t) p].experts != sb[(size_t) p].experts || sa[(size_t) p].bytes != sb[(size_t) p].bytes) {
            std::fprintf(stderr, "bench: pass %d work mismatch %llu/%llu experts, "
                                 "%llu/%llu bytes\n", p,
                         (unsigned long long) sa[(size_t) p].experts,
                         (unsigned long long) sb[(size_t) p].experts,
                         (unsigned long long) sa[(size_t) p].bytes,
                         (unsigned long long) sb[(size_t) p].bytes);
            return 1;
        }
    }
    long long jobs = 0, bytes = 0, fb = 0;
    b->resident_direct_dma_debug(&jobs, &bytes, &fb);
    auto sum = [](const std::vector<Sample>& v, double Sample::* f) {
        double t = 0; for (const auto& s : v) t += s.*f; return t;
    };
    auto total = [](const std::vector<Sample>& v, uint64_t Sample::* f) {
        uint64_t t = 0; for (const auto& s : v) t += s.*f; return t;
    };
    for (int p = 0; p < passes; ++p)
        std::printf("pass %d: gather wall %8.2f ms join %8.2f ms | direct wall %8.2f ms "
                    "join %8.2f ms | staged %llu/%llu experts\n", p,
                    sa[(size_t) p].wall_ms, sa[(size_t) p].join_ms,
                    sb[(size_t) p].wall_ms, sb[(size_t) p].join_ms,
                    (unsigned long long) sa[(size_t) p].experts,
                    (unsigned long long) sb[(size_t) p].experts);
    const double wa = sum(sa, &Sample::wall_ms), wb = sum(sb, &Sample::wall_ms);
    const double ja = sum(sa, &Sample::join_ms), jb = sum(sb, &Sample::join_ms);
    const uint64_t ea = total(sa, &Sample::experts), eb = total(sb, &Sample::experts);
    const uint64_t ba = total(sa, &Sample::bytes), bb = total(sb, &Sample::bytes);
    std::printf("TOTAL %d passes: gather wall %.1f ms join %.1f ms | direct wall %.1f ms "
                "join %.1f ms | delta wall %+.1f ms join %+.1f ms\n",
                passes, wa, ja, wb, jb, wa - wb, ja - jb);
    std::printf("work parity (HARD): staged experts gather=%llu direct=%llu bytes gather=%llu MiB "
                "direct=%llu MiB\n",
                (unsigned long long) ea, (unsigned long long) eb,
                (unsigned long long) (ba >> 20), (unsigned long long) (bb >> 20));
    std::printf("rows: %zu calls per arm compared BITWISE, all equal and finite\n",
                rows_a.size());
    std::printf("direct arm counters (lifetime, warm pass included): jobs=%lld "
                "bytes=%.2f MiB fallback=%lld\n",
                jobs, (double) bytes / 1048576.0, fb);
    std::printf("# measured on the layer subset only; NO engine end-to-end speed claim "
                "is derivable from this bench\n");
    return (ea == eb && ba == bb) ? 0 : 1;
}
