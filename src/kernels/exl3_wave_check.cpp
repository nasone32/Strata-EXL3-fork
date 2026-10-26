// src/kernels/exl3_wave_check.cpp - the real expert-stream prefill path (run_device) under
// STRATA_EXL3_WAVESYNC=1 and STRATA_EXL3_PIPE=1 against the default per-tile barrier, on a
// synthetic model directory - plus the STREAM LOAN battery: STRATA_EXL3_PREFILL_BORROW's arena
// lends its tail ranks (lend_tail), the borrower overwrites them (0x5A read back as ownership
// evidence, then 0xFF - an fp16 NaN pattern), runs continue on the active prefix and restore_tail()
// refills, everything against an original-backing oracle running the same calls (cold/partial/full
// cache at lend, saved-tail experts re-admitted into the prefix during the loan, eviction inside
// the loan, growth/idempotent/shrink/out-of-bounds requests, the maximal one-active-rank loan,
// run/run_multi/run_device all inside loans, a second full turn, cache 0 and the original backing
// reporting no loan, and a nonzero owned layer range with a different K/FFN geometry).
//
// All arms run the same kernels in the same order; only stream-barrier granularity, route-table
// upload batching and staging/compute scheduling differ, so the outputs must be BYTE-identical.
// The fixture carries 80 experts with DISTINCT weights and scales per (expert, projection), so a
// slot/expert mixup or a staging overwrite before the slot's last reader (the pipeline's
// slot-reuse ordering) cannot hide behind identical weights.  Shape coverage: several waves with
// slot eviction (16 experts through a 4-slot cache), group counts >= 16 (the reconstruct+BLAS and
// WMMA arms), counts < 16 (the indexed GEMV arm), one group split across successive tiles (5000
// routes over the 4096-route tile bound), a larger group queued after a smaller one inside one
// wave, a partial final wave (10 groups over a 4-slot cache: waves 4+4+2), scratch mode (cache 0)
// with two concurrent waves whose second one reuses the scratch slots of the first (80 groups
// through the 64-slot scratch wave size), warm second calls, and a prefill -> single-token run() ->
// decode-window phase sequence.  The STRATA_EXL3_WMMA and STRATA_EXL3_PIPE flags are set
// explicitly per arm so an inherited environment cannot silently change one, and both kernel
// regimes (reconstruct fallback and the gfx11 matrix-core arm where available) are compared.
// Exits 77 when no CUDA device is present (ctest skip); every other failure fails.
//
// Usage: exl3_wave_check <model_dir>   (fixture written by tools/test_exl3_loader.py)
#include "strata/kernels/exl3_experts.hpp"
#include "strata/kernels/exl3_arena_layout.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kEmb = 2560, kExperts = 80, kSlots = 4;

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    (void) _putenv_s(name, value);
#else
    (void) setenv(name, value, 1);
#endif
}

bool run_ids(strata::kernels::Exl3ExpertStream& stream, int tokens, int k, const std::vector<int32_t>& ids,
             const std::vector<float>& x, std::vector<float>& out, int layer = 0) {
    out.assign((size_t) tokens * k * kEmb, -1.0f);
    float* d_x = nullptr;
    float* d_out = nullptr;
    if (cudaMalloc(&d_x, (size_t) tokens * kEmb * 4) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(d_x) failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    if (cudaMalloc(&d_out, out.size() * 4) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(d_out) failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        (void) cudaFree(d_x);   // do not leak the first allocation on the partial failure
        return false;
    }
    bool ok = cudaMemcpy(d_x, x.data(), (size_t) tokens * kEmb * 4, cudaMemcpyHostToDevice) == cudaSuccess;
    if (!ok) std::fprintf(stderr, "input upload failed: %s\n", cudaGetErrorString(cudaGetLastError()));
    ok = ok && stream.run_device(layer, d_x, ids.data(), tokens, k, d_out, nullptr);
    const cudaError_t se = cudaDeviceSynchronize();
    ok = ok && se == cudaSuccess;
    if (!ok) std::fprintf(stderr, "run_device failed (sync=%s)\n", cudaGetErrorString(se));
    ok = ok && cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
    (void) cudaFree(d_x);
    (void) cudaFree(d_out);
    return ok;
}

bool run_shape(strata::kernels::Exl3ExpertStream& stream, int tokens, int k, int seed, int span,
               const std::vector<float>& x, std::vector<float>& out) {
    std::vector<int32_t> ids((size_t) tokens * k);
    for (int t = 0; t < tokens; ++t)
        for (int j = 0; j < k; ++j)
            ids[(size_t) t * k + j] = (t * 7 + j * 3 + seed) % span;
    return run_ids(stream, tokens, k, ids, x, out);
}

// The decode side of the phase seam: host x/ids/out through run_multi (the verifier windows' path).
bool run_window(strata::kernels::Exl3ExpertStream& stream, int tokens, int k, int seed,
                const std::vector<float>& x, std::vector<float>& out) {
    out.assign((size_t) tokens * k * kEmb, -1.0f);
    std::vector<int32_t> ids((size_t) tokens * k);
    for (int t = 0; t < tokens; ++t)
        for (int j = 0; j < k; ++j)
            ids[(size_t) t * k + j] = (t * 7 + j * 3 + seed) % 16;
    return stream.run_multi(0, x.data(), ids.data(), tokens, k, out.data(), nullptr);
}

// The single-token decode path with k=2 distinct experts (k <= the smallest active prefix the
// battery lends, so it exercises the cache arm; compare with compare_rel - atomic accumulation).
bool run_pair(strata::kernels::Exl3ExpertStream& stream, int seed, const std::vector<float>& x,
              std::vector<float>& out) {
    constexpr int k = 2;
    std::vector<int> ids((size_t) k);
    for (int j = 0; j < k; ++j) ids[(size_t) j] = (seed * 5 + j * 3) % 16;
    out.assign((size_t) k * kEmb, -1.0f);
    return stream.run(0, x.data(), ids.data(), k, out.data(), nullptr);
}

// The single-token decode path (run): k distinct experts per token, host in/out.
bool run_single(strata::kernels::Exl3ExpertStream& stream, int seed, const std::vector<float>& x,
                std::vector<float>& out) {
    constexpr int k = 4;
    std::vector<int> ids((size_t) k);
    for (int j = 0; j < k; ++j) ids[(size_t) j] = (seed * 5 + j * 3) % 16;
    out.assign((size_t) k * kEmb, -1.0f);
    return stream.run(0, x.data(), ids.data(), k, out.data(), nullptr);
}

// run() goes through the no-input_rows GEMV, whose K partitions accumulate with atomics: two runs
// of the SAME call differ in the last ULPs (elements near zero see large RELATIVE wobble, so the
// comparison is normalized by the output's scale, not per element).  A slot/expert mixup changes
// whole rows (distinct weights), orders of magnitude above the tolerance.
// STRATA_EXL3_WAVE_CHECK_HASH=1: a running FNV-1a over every output the checker compares, in
// comparison order.  Two PROCESSES (e.g. STRATA_EXL3_NO_PREAD unset vs present - that flag is a
// once-per-process static) must print the same SEQUENCEHASH when every output is byte-identical.
// The compare_rel path folds only lengths (its tolerance is not byte-identity by design).
uint64_t g_seq_hash = 1469598103934665603ull;
void hash_fold(const void* data, size_t bytes) {
    const uint8_t* p = (const uint8_t*) data;
    for (size_t i = 0; i < bytes; ++i) { g_seq_hash ^= p[i]; g_seq_hash *= 1099511628211ull; }
}
void hash_fold_len(size_t n) { hash_fold(&n, sizeof n); }

bool compare_rel(const std::vector<float>& a, const std::vector<float>& b, double tol, const char* what) {
    if (a.size() != b.size() || a.empty()) {
        std::printf("%s: size mismatch (%zu vs %zu) FAIL\n", what, a.size(), b.size());
        return false;
    }
    double scale = 0, worst = 0;
    size_t at = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            std::printf("%s: non-finite output at %zu FAIL\n", what, i);
            return false;
        }
        scale = std::max(scale, (double) std::fabs(a[i]));
        const double d = std::fabs((double) a[i] - (double) b[i]);
        if (d > worst) { worst = d; at = i; }
    }
    const double norm = worst / std::max(1e-6, scale);
    if (norm > tol) {
        std::printf("%s: FAIL (max |diff| %.3e at %zu, scale %.3e, norm %.3e > tol %.1e)\n",
                    what, worst, at, scale, norm, tol);
        return false;
    }
    std::printf("%s: max |diff| %.3e at %zu, scale %.3e, norm %.3e (tol %.1e) PASS\n",
                what, worst, at, scale, norm, tol);
    hash_fold_len(a.size());   // lengths only: this path is tolerance-based by design
    return true;
}

// Byte-identity plus sanity: finite values and a non-degenerate spread, so a slot/expert mixup in
// the fixture cannot hide behind identical or infinite outputs.
bool compare(const std::vector<float>& a, const std::vector<float>& b, const char* what) {
    if (a.size() != b.size() || a.empty()) {
        std::printf("%s: size mismatch (%zu vs %zu) FAIL\n", what, a.size(), b.size());
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            std::printf("%s: non-finite output at %zu FAIL\n", what, i);
            return false;
        }
    if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0) {
        const uint32_t first = reinterpret_cast<const uint32_t*>(a.data())[0];
        size_t distinct = 0;
        for (size_t i = 0; i < a.size() && distinct < 2; ++i)
            if (reinterpret_cast<const uint32_t*>(a.data())[i] != first) ++distinct;
        if (distinct < 1) { std::printf("%s: degenerate constant output FAIL\n", what); return false; }
        std::printf("%s: byte-identical over %zu floats, non-constant PASS\n", what, a.size());
        hash_fold(a.data(), a.size() * sizeof(float));   // read-arm equivalence evidence
        return true;
    }
    size_t differ = 0, first = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            if (!differ) first = i;
            ++differ;
        }
    std::printf("%s: FAIL (%zu of %zu floats differ, first at %zu)\n", what, differ, a.size(), first);
    return false;
}

// The shapes' outputs for one arm, on the 4-slot cache stream and on a scratch (cache 0) stream.
struct ArmOut {
    std::vector<float> a, b, c, d, g, warm_a;
    std::vector<float> h, i, warm_h;
};

bool run_arm(const std::string& dir, ArmOut& o, const std::vector<float>& x, const std::vector<float>& x_c,
             const std::vector<float>& x_d, const std::vector<float>& x_h,
             const std::vector<int32_t>& ids_c, const std::vector<int32_t>& ids_d,
             const std::vector<int32_t>& ids_g, const std::vector<int32_t>& ids_h,
             const std::vector<int32_t>& ids_i, size_t ram_bytes = 0) {
    bool ok = true;
    {
        strata::kernels::Exl3ExpertStream s(dir, kEmb, 1, kSlots, nullptr, ram_bytes);
        ok = ok && run_shape(s, 96, 8, 0, 16, x, o.a);          // >=16 rows/group, 4 waves with eviction
        ok = ok && run_shape(s, 24, 4, 1, 16, x, o.b);          // <16 rows/group, the indexed GEMV arm
        ok = ok && run_ids(s, 5000, 1, ids_c, x_c, o.c);        // one group over two tiles (> 4096)
        ok = ok && run_ids(s, 616, 1, ids_d, x_d, o.d);         // 16-row group then 600-row group, one wave
        ok = ok && run_ids(s, 200, 1, ids_g, x_d, o.g);         // 10 groups x 20 rows, waves 4+4+2 (partial)
        ok = ok && run_shape(s, 96, 8, 0, 16, x, o.warm_a);     // warm cache, repeated call
    }
    {
        strata::kernels::Exl3ExpertStream s(dir, kEmb, 1, 0, nullptr, ram_bytes);   // scratch: every use re-uploads
        ok = ok && run_ids(s, 320, 4, ids_h, x_h, o.h);         // 80 groups x 16 rows: waves 64+16, the
                                                                // second wave's scratch slots 0..15 are
                                                                // reused while the first wave's readers
                                                                // may still be queued (slot-reuse order)
        ok = ok && run_ids(s, 80, 2, ids_i, x, o.i);            // the same reuse on the indexed path
        ok = ok && run_ids(s, 320, 4, ids_h, x_h, o.warm_h);    // warm repeat
    }
    return ok;
}

// ================================ the stream loan battery ================================
//
// The prefill loan primitives against an original-backing oracle: the arena stream lends tail
// ranks, the borrowed range is OVERWRITTEN the way the prompt path would (a recognizable byte
// first, read back so the borrower provably owns the exact addresses, then 0xFF - an fp16 NaN
// pattern - so any read of a lent or unrestored slot surfaces as a non-finite output), runs
// continue on the active prefix, and restore_tail() refills.  The oracle stream
// (STRATA_EXL3_PREFILL_BORROW=0) runs the SAME calls in the same order; outputs must match
// because residency never changes the math.  Phases:
//   - cold cache lend (empty saved tail, but an ACTIVE loan), run()/restore;
//   - full-cache lend with a NON-duplicate refill (no run during the loan: every saved record is
//     re-read from the model; the next call must hit those slots' fresh bytes, not the 0xFF);
//   - a run_multi window DURING a loan (admissions and evictions inside the prefix; restore sees a
//     mix of duplicates re-admitted into the prefix and surviving records);
//   - the critical duplicate case: a loan whose saved experts ALL get re-admitted while lent
//     (restore must keep the hot prefix copy and never publish the poisoned tail bytes);
//   - growth (the address moves DOWN by rank_bytes, a superset range), idempotent repeat, shrink
//     and out-of-bounds rejections that change nothing (epoch/active/outputs);
//   - the maximal loan (one active rank): run_device on one-slot waves and run() with k above the
//     active prefix falling back to scratch;
//   - a second full lend/run/restore turn, then the no-loan arms: original backing and cache 0
//     report no loan, reject lend/restore, and still run;
//   - a nonzero owned layer range with a different K/FFN geometry (its own smaller arena).
bool poison_range(void* address, size_t bytes, int byte, const char* what) {
    if (cudaMemset(address, byte, bytes) != cudaSuccess) {
        std::printf("loan %s: cudaMemset(%d) failed: %s\n", what, byte, cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    const size_t n = std::min<size_t>(bytes, 256);
    std::vector<uint8_t> head(n), tail(n);
    if (cudaMemcpy(head.data(), address, n, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(tail.data(), (uint8_t*) address + bytes - n, n, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::printf("loan %s: pattern readback failed: %s\n", what, cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    for (size_t i = 0; i < n; ++i)
        if (head[i] != (uint8_t) byte || tail[i] != (uint8_t) byte) {
            std::printf("loan %s: the borrowed range did not take byte %d (readback %d/%d at %zu) FAIL\n",
                        what, byte, head[i], tail[i], i);
            return false;
        }
    std::printf("loan %s: %zu bytes at %p take byte %d (readback verified) PASS\n", what, bytes, address, byte);
    return true;
}

// The loan battery's own oracle outputs (the S1..S9 sequence both arms run) and loan-arm outputs.
// The battery's call recorder: the loan arm runs first and records every (layer, ids, kind) call
// it makes - including calls whose ids were captured from the loan seams (saved/restored experts) -
// then the original-backing oracle replays the same list and every output is compared.  Outputs
// are residency-independent, so a replay on a stream that never lends is a valid oracle.
struct LoanCall {
    int layer = 0;
    int tokens = 0, k = 0;
    bool single = false;              // run() (single-token decode) rather than run_device
    std::vector<int32_t> ids;         // run_device/run_multi ids, or run()'s k ids when single
    std::vector<float> out;
    const char* what = "";
};

// ids covering exactly `who` (rows cycle the set): every group the call forms is one of `who`, so
// right after a restore it reads exactly the published slots and admits nothing else.
std::vector<int32_t> ids_only(const std::vector<int>& who, int tokens, int k) {
    std::vector<int32_t> ids;
    if (who.empty()) return ids;              // callers assert non-empty; never divide by zero
    ids.resize((size_t) tokens * k);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = (int32_t) who[i % who.size()];
    return ids;
}

// Span-16 churn ids (the fixture's distinct-expert space), same shape as run_shape's seeds.
std::vector<int32_t> churn_ids(int seed, int tokens, int k) {
    std::vector<int32_t> ids((size_t) tokens * k);
    for (int t = 0; t < tokens; ++t)
        for (int j = 0; j < k; ++j)
            ids[(size_t) t * k + j] = (t * 7 + j * 3 + seed) % 16;
    return ids;
}

struct LoanBattery {
    strata::kernels::Exl3ExpertStream& s;
    const std::vector<float>& x;
    std::vector<LoanCall>& calls;   // the battery records into the caller's list
    bool ok = true;

    void expect(bool got, const char* what) {
        if (!got) { std::printf("%s FAIL\n", what); ok = false; }
    }
    LoanCall& device(int layer, const std::vector<int32_t>& ids, int tokens, int k, const char* what) {
        if (ids.empty()) {
            std::printf("%s: empty id set FAIL\n", what);
            ok = false;
        }
        LoanCall c; c.layer = layer; c.tokens = tokens; c.k = k; c.ids = ids; c.what = what;
        calls.push_back(std::move(c));
        LoanCall& r = calls.back();
        ok = ok && run_ids(s, tokens, k, r.ids, x, r.out, layer);
        return r;
    }
    void single(int seed, const char* what) {
        LoanCall c; c.layer = 0; c.single = true; c.tokens = 1;
        std::vector<int> ids((size_t) 2);
        for (int j = 0; j < 2; ++j) ids[(size_t) j] = (seed * 5 + j * 3) % 16;
        c.k = (int) ids.size();
        c.ids.assign(ids.begin(), ids.end());
        c.what = what;
        calls.push_back(std::move(c));
        LoanCall& r = calls.back();
        ok = ok && run_pair(s, seed, x, r.out);
    }
};

// Replay a recorded list on a fresh stream (kind per call), returning the outputs.
bool loan_replay(const std::string& dir, const std::vector<float>& x, int n_layers, int slots,
                 const std::vector<LoanCall>& calls, std::vector<std::vector<float>>& outs) {
    strata::kernels::Exl3ExpertStream s(dir, kEmb, n_layers, slots, nullptr);
    for (const LoanCall& c : calls) {
        std::vector<float> out;
        if (c.single) {
            std::vector<int> ids(c.ids.begin(), c.ids.end());
            out.assign((size_t) c.k * kEmb, -1.0f);
            if (!s.run(c.layer, x.data(), ids.data(), c.k, out.data(), nullptr)) return false;
        } else {
            if (!run_ids(s, c.tokens, c.k, c.ids, x, out, c.layer)) return false;
        }
        if (!s.loan_dense_invariant()) {
            std::printf("oracle replay lost the dense invariant at \"%s\" FAIL\n", c.what);
            return false;
        }
        outs.push_back(std::move(out));
    }
    return true;
}

// The single-layer battery (see the file header).  `tag` names the arm in every assertion line.
bool loan_battery(const std::string& dir, const char* pipe, const char* batch, const char* wmma,
                  const std::vector<float>& x) {
    using strata::kernels::Exl3ExpertStream;
    using Loan = Exl3ExpertStream::Loan;
    const std::string tag = std::string("loan PIPE=") + pipe + ",BATCH=" + batch + ",WMMA=" + wmma;
    set_env("STRATA_EXL3_PIPE", pipe);
    set_env("STRATA_EXL3_EXPERT_BATCH", batch);
    set_env("STRATA_EXL3_WMMA", wmma);
    set_env("STRATA_EXL3_WAVESYNC", "1");
    bool ok = true;
    std::vector<std::vector<float>> oracle_out;
    uint64_t epoch = 0;
    std::vector<LoanCall> calls;
    {
        set_env("STRATA_EXL3_PREFILL_BORROW", "1");
        Exl3ExpertStream s(dir, kEmb, 1, kSlots, nullptr);
        LoanBattery b{s, x, calls};
        const size_t rb = s.loan_rank_bytes();
        b.expect(s.loan_available() && rb > 0 && rb % 256 == 0 && s.loan_max_ranks() == kSlots - 1 &&
                 s.loan_active_ranks() == kSlots && !s.loan_active() && s.loan_epoch() == 0 &&
                 s.loan_dense_invariant(), "cold stream reports its loan capacity, invariant holds");
        for (int bad : {0, -2, kSlots, kSlots + 5})
            b.expect(s.lend_tail(bad).address == nullptr, "rejected lend changes nothing");
        b.expect(!s.restore_tail() && s.loan_epoch() == 0 && s.loan_active_ranks() == kSlots &&
                 s.loan_dense_invariant(), "restore without a loan is rejected");
        {   // cold lend: nothing resident, the saved tail is empty - still an active loan
            const Loan l = s.lend_tail(2);
            std::vector<int> saved;
            s.loan_debug_saved(0, saved);
            b.expect(l.address != nullptr && l.bytes == 2 * rb && l.ranks == 2 && l.epoch == ++epoch &&
                     (reinterpret_cast<uintptr_t>(l.address) % 256) == 0 && s.loan_active() &&
                     s.loan_active_ranks() == 2 && saved.empty() && s.loan_dense_invariant(),
                     "cold lend of two ranks, empty saved tail");
            b.ok = b.ok && poison_range(l.address, l.bytes, 0x5A, "borrower owns the range (cold)");
            b.ok = b.ok && poison_range(l.address, l.bytes, 0xFF, "NaN poison (cold)");
            b.single(2, "run() inside the cold loan");
            std::vector<int> restored;
            b.expect(s.restore_tail() && s.loan_epoch() == ++epoch && !s.loan_active() &&
                     s.loan_active_ranks() == kSlots && (s.loan_debug_restored(0, restored), restored.empty()) &&
                     s.loan_dense_invariant(), "cold loan restored (empty saved tail)");
        }
        {   // partially filled prefix (2 of 4 slots occupied), tail empty
            const Loan l = s.lend_tail(2);
            std::vector<int> saved;
            s.loan_debug_saved(0, saved);
            b.expect(l.address != nullptr && l.epoch == ++epoch && saved.empty() &&
                     s.loan_active_ranks() == 2 && s.loan_dense_invariant(),
                     "partial-prefix lend, empty saved tail");
            b.ok = b.ok && poison_range(l.address, l.bytes, 0xFF, "NaN poison (partial prefix)");
            b.expect(s.restore_tail() && s.loan_epoch() == ++epoch && s.loan_dense_invariant(),
                     "partial-prefix loan restored");
        }
        b.device(0, churn_ids(0, 96, 8), 96, 8, "churn: fill the cache");   // filled == 4 now
        {   // the PURE REFILL: no run during the loan, so every saved record is a survivor.
            // restore must re-read them from the model into the tail, and the next call hits those
            // slots' fresh bytes BEFORE anything else can evict or repair them.
            const Loan l = s.lend_tail(2);
            std::vector<int> saved;
            s.loan_debug_saved(0, saved);
            b.expect(l.address != nullptr && l.epoch == ++epoch && !saved.empty() && saved.size() <= 2 &&
                     s.loan_dense_invariant(), "full-cache lend, saved ids captured");
            b.ok = b.ok && poison_range(l.address, l.bytes, 0xFF, "NaN poison (pure refill)");
            std::vector<int> restored;
            b.expect(s.restore_tail() && s.loan_epoch() == ++epoch, "pure-refill restore");
            s.loan_debug_restored(0, restored);
            b.expect(restored == saved && restored.size() <= 2 && s.loan_dense_invariant(),
                     "pure refill restored exactly the saved ids, within the tail bound");
            b.device(0, ids_only(restored, 96, 4), 96, 4, "hit-only read of every restored expert");
        }
        {   // a run_multi window inside a loan: admissions and evictions in the prefix, restore sees
            // a mix of re-admitted duplicates and survivors
            const Loan l = s.lend_tail(2);
            std::vector<int> saved;
            s.loan_debug_saved(0, saved);
            b.expect(l.address != nullptr && l.epoch == ++epoch && s.loan_dense_invariant(),
                     "lend around a window");
            b.ok = b.ok && poison_range(l.address, l.bytes, 0xFF, "NaN poison (window)");
            {
                LoanCall c; c.layer = 0; c.tokens = 24; c.k = 8; c.what = "run_multi window inside a loan";
                for (int t = 0; t < 24; ++t)
                    for (int j = 0; j < 8; ++j)
                        c.ids.push_back((t * 7 + j * 3 + 5) % 16);
                calls.push_back(std::move(c));
                std::vector<float>& out = calls.back().out;
                out.assign((size_t) 24 * 8 * kEmb, -1.0f);
                b.ok = b.ok && s.run_multi(0, x.data(), calls.back().ids.data(), 24, 8, out.data(), nullptr);
            }
            std::vector<int> restored;
            b.expect(s.restore_tail() && s.loan_epoch() == ++epoch, "window loan restored");
            s.loan_debug_restored(0, restored);
            b.expect(restored.size() <= saved.size() && s.loan_dense_invariant(),
                     "window restore placed at most the saved ids (mixed duplicates/survivors)");
            if (!restored.empty())
                b.device(0, ids_only(restored, 96, 4), 96, 4, "hit-only read of the window's restored experts");
        }
        b.device(0, churn_ids(0, 96, 8), 96, 8, "churn: refill the cache for the duplicate case");
        {   // the ALL-DUPLICATES case, explicit: route exactly the saved ids during the loan so
            // every one of them is resident in the prefix at restore; restore must publish NONE of
            // the poisoned tail bytes, and the immediate hit-only read of the same ids (all prefix
            // hits, zero admissions) proves it.
            const Loan l = s.lend_tail(2);
            std::vector<int> saved;
            s.loan_debug_saved(0, saved);
            b.expect(l.address != nullptr && l.epoch == ++epoch && !saved.empty() && saved.size() <= 2,
                     "duplicate-case lend, saved ids captured");
            b.ok = b.ok && poison_range(l.address, l.bytes, 0xFF, "NaN poison (duplicates)");
            b.device(0, ids_only(saved, 96, 4), 96, 4, "route exactly the saved ids during the loan");
            std::vector<int> restored;
            b.expect(s.restore_tail() && s.loan_epoch() == ++epoch, "duplicate-case restore");
            s.loan_debug_restored(0, restored);
            b.expect(restored.empty() && s.loan_dense_invariant(),
                     "all-duplicates restore placed nothing in the tail");
            b.device(0, ids_only(saved, 96, 4), 96, 4, "hit-only read of the retained prefix copies");
        }
        {   // growth: churn during the loan, then lend MORE while it is open (the extra rank held a
            // prefix expert - possibly one already saved: restore must place it once), the maximal
            // one-active-rank loan, and run()'s scratch fallback above the prefix.
            const Loan two = s.lend_tail(2);
            std::vector<int> saved2;
            s.loan_debug_saved(0, saved2);
            b.expect(two.address != nullptr && two.epoch == ++epoch && s.loan_dense_invariant(),
                     "growth-phase lend");
            b.ok = b.ok && poison_range(two.address, two.bytes, 0xFF, "NaN poison (growth)");
            b.device(0, churn_ids(3, 96, 8), 96, 8, "churn inside the growth-phase loan");
            const Loan three = s.lend_tail(3);
            std::vector<int> saved3;
            s.loan_debug_saved(0, saved3);
            b.expect(three.address != nullptr && three.bytes == 3 * rb && three.epoch == ++epoch &&
                     (const uint8_t*) three.address == (const uint8_t*) two.address - rb &&
                     s.loan_active_ranks() == 1 && saved3.size() <= 3 && s.loan_dense_invariant(),
                     "loan growth to three ranks, saved ids unique within the bound");
            const Loan again = s.lend_tail(3);
            b.expect(again.address == three.address && again.epoch == three.epoch,
                     "idempotent repeat of the current size");
            b.expect(s.lend_tail(2).address == nullptr && s.lend_tail(kSlots).address == nullptr,
                     "shrink and out-of-bounds growth rejected mid-loan");
            b.ok = b.ok && poison_range(three.address, three.bytes, 0xFF, "NaN poison (grown range)");
            b.single(9, "run() scratch fallback above the active prefix");   // k=2 > active=1
            std::vector<int> restored;
            b.expect(s.restore_tail() && s.loan_epoch() == ++epoch, "growth-phase restore");
            s.loan_debug_restored(0, restored);
            b.expect(restored.size() <= 3 && s.loan_dense_invariant(),
                     "growth restore placed each survivor once, within the tail bound");
            if (!restored.empty())
                b.device(0, ids_only(restored, 96, 4), 96, 4, "hit-only read of the growth-phase survivors");
        }
        b.device(0, churn_ids(0, 96, 8), 96, 8, "warm churn after all restores");
        {   // a second full turn: lend, churn during the loan, restore, hit-only read, then churn
            const Loan l = s.lend_tail(2);
            std::vector<int> saved;
            s.loan_debug_saved(0, saved);
            b.expect(l.address != nullptr && l.epoch == ++epoch && s.loan_dense_invariant(),
                     "second-turn lend");
            b.ok = b.ok && poison_range(l.address, l.bytes, 0xFF, "NaN poison (second turn)");
            b.device(0, churn_ids(3, 96, 8), 96, 8, "second-turn churn inside the loan");
            std::vector<int> restored;
            b.expect(s.restore_tail() && s.loan_epoch() == ++epoch, "second-turn restore");
            s.loan_debug_restored(0, restored);
            b.expect(restored.size() <= 2 && s.loan_dense_invariant(), "second-turn restore bounded");
            if (!restored.empty())
                b.device(0, ids_only(restored, 96, 4), 96, 4, "hit-only read after the second turn");
            b.device(0, churn_ids(0, 96, 8), 96, 8, "broad churn only after the direct hit read");
        }
        b.expect(s.loan_dense_invariant() && !s.loan_active(), "battery ends invariant-clean");
        if (!b.ok) return false;
    }
    {   // the oracle: original backing, no loan, replaying the recorded list
        set_env("STRATA_EXL3_PREFILL_BORROW", "0");
        if (!loan_replay(dir, x, 1, kSlots, calls, oracle_out) || oracle_out.size() != calls.size()) {
            std::printf("%s, oracle replay failed\n", tag.c_str());
            return false;
        }
        for (size_t i = 0; i < calls.size(); ++i) {
            const std::string what = tag + ", " + calls[i].what;
            if (calls[i].single)
                ok = ok && compare_rel(oracle_out[i], calls[i].out, 5e-3, what.c_str());
            else
                ok = ok && compare(oracle_out[i], calls[i].out, what.c_str());
        }
        if (!ok) return false;
    }
    {   // cache 0 under the borrow flag: no loan at all, and the runs are untouched
        set_env("STRATA_EXL3_PREFILL_BORROW", "1");
        Exl3ExpertStream s(dir, kEmb, 1, 0, nullptr);
        std::vector<float> y;
        ok = ok && !s.loan_available() && s.loan_rank_bytes() == 0 && s.loan_max_ranks() == 0 &&
               s.lend_tail(1).address == nullptr && !s.restore_tail() && s.loan_epoch() == 0;
        ok = ok && run_ids(s, 96, 8, calls[1].ids, x, y, 0);   // call [1]: the fill churn, same shape
        ok = ok && compare(oracle_out[1], y, (tag + ", cache 0 unchanged run").c_str());
    }
    return ok;
}

// Repeated growth with re-admission cycles on a SIX-slot cache (two growth steps), with an
// old-saved expert deliberately kept OUT of the final prefix: the union of one loan's saved
// records is bounded by the lent ranks (records come from the disjoint rank increments of its
// lends), so valid growths must never overflow the tail - the restore succeeds - and the
// old-saved expert absent at the final prefix must come back as a placed survivor, read
// hit-only before anything else can churn it.
bool loan_repeated_growth(const std::string& dir, const std::vector<float>& x) {
    using strata::kernels::Exl3ExpertStream;
    using Loan = Exl3ExpertStream::Loan;
    constexpr int kWide = 6;
    const std::string tag = "loan repeated-growth";
    set_env("STRATA_EXL3_PIPE", "1");
    set_env("STRATA_EXL3_EXPERT_BATCH", "1");
    set_env("STRATA_EXL3_WMMA", "1");
    set_env("STRATA_EXL3_WAVESYNC", "1");
    bool ok = true;
    uint64_t epoch = 0;
    std::vector<LoanCall> calls;
    std::vector<std::vector<float>> oracle_out;
    {
        set_env("STRATA_EXL3_PREFILL_BORROW", "1");
        Exl3ExpertStream s(dir, kEmb, 1, kWide, nullptr);
        LoanBattery b{s, x, calls};
        const size_t rb = s.loan_rank_bytes();
        b.expect(s.loan_available() && s.loan_max_ranks() == kWide - 1 && s.loan_dense_invariant(),
                 "cache-6 stream reports its loan capacity");
        b.device(0, churn_ids(0, 96, 8), 96, 8, "fill all six slots");
        const Loan two = s.lend_tail(2);
        std::vector<int> s1;
        s.loan_debug_saved(0, s1);
        b.expect(two.address != nullptr && s1.size() == 2 && two.epoch == ++epoch && s.loan_dense_invariant(),
                 "initial lend saves exactly the two tail experts");
        b.ok = b.ok && poison_range(two.address, two.bytes, 0xFF, "NaN poison (repeated growth, 2 ranks)");
        const int keep = s1[0], drop = s1[1];   // keep is re-admitted during the loan; drop never again
        int other = 0;
        while (other == keep || other == drop) ++other;
        b.device(0, ids_only({keep, other}, 96, 4), 96, 4, "readmit one saved expert during the loan");
        const Loan three = s.lend_tail(3);
        std::vector<int> s3;
        s.loan_debug_saved(0, s3);
        b.expect(three.address != nullptr && (const uint8_t*) three.address == (const uint8_t*) two.address - rb &&
                 three.epoch == ++epoch && s3.size() <= 3 && s.loan_dense_invariant(),
                 "first growth, saved union bounded by the lent ranks");
        b.ok = b.ok && poison_range(three.address, three.bytes, 0xFF, "NaN poison (growth to 3)");
        b.device(0, ids_only({keep, other}, 96, 4), 96, 4, "readmit churn under the 3-rank loan");
        const Loan five = s.lend_tail(kWide - 1);
        std::vector<int> s5;
        s.loan_debug_saved(0, s5);
        b.expect(five.address != nullptr && five.bytes == (size_t) (kWide - 1) * rb && s.loan_active_ranks() == 1 &&
                 five.epoch == ++epoch && s5.size() <= kWide - 1 && s.loan_dense_invariant(),
                 "second growth to the maximum, saved union still bounded");
        b.ok = b.ok && poison_range(five.address, five.bytes, 0xFF, "NaN poison (maximal loan)");
        b.device(0, ids_only({keep}, 48, 2), 48, 2, "single-expert waves under the maximal loan");
        std::vector<int> restored;
        b.expect(s.restore_tail() && s.loan_dense_invariant(),
                 "repeated-growth restore succeeds (no overflow on valid growths)");
        s.loan_debug_restored(0, restored);
        b.expect(std::find(restored.begin(), restored.end(), drop) != restored.end() &&
                 restored.size() <= (size_t) (kWide - 1),
                 "old-saved expert absent at the final prefix was restored, within the bound");
        b.device(0, ids_only(restored, 96, 4), 96, 4, "hit-only read of the repeated-growth survivors");
        b.device(0, churn_ids(5, 96, 8), 96, 8, "broad churn after the direct read");
        b.expect(s.loan_dense_invariant() && !s.loan_active(), "repeated-growth battery ends invariant-clean");
        if (!b.ok) return false;
    }
    {
        set_env("STRATA_EXL3_PREFILL_BORROW", "0");
        if (!loan_replay(dir, x, 1, kWide, calls, oracle_out) || oracle_out.size() != calls.size()) {
            std::printf("%s, oracle replay failed\n", tag.c_str());
            return false;
        }
        for (size_t i = 0; i < calls.size(); ++i)
            ok = ok && compare(oracle_out[i], calls[i].out, (tag + ", " + calls[i].what).c_str());
    }
    return ok;
}

// ================================ the host-IO overlap battery ================================
//
// STRATA_EXL3_OVERLAP's split of a host-IO wave into a hit pass and a miss pass against the
// original single-pass order, on identically-constructed streams (overlap off = the oracle)
// running the SAME call sequence - residency and routing are deterministic, so every output must
// be BYTE-identical (only scheduling differs).  Phases on the 4-slot cache: an all-miss window
// (the original order, a counted fallback), its all-hit repeat (fallback again), a mixed window
// (the split), duplicate expert ids within one token, tokens 1/2/4/8 (the verifier window
// shapes), a 512-route window over all 80 experts (20 waves, eviction pressure, and a larger
// workspace than any earlier call - cross-call growth under the split), then a cache-0 stream
// (no cache: the original path), a nonzero owned layer range with the fixture's other K/FFN
// geometry, and the loan interplay: fill, lend_tail, a mixed window INSIDE the loan (the active
// prefix shrinks to two slots), restore_tail, then a window reading the restored slots.  The
// split/fallback counters (overlap_debug) prove the split actually ran on the mixed phases and
// the oracle never counted one - no silent fallback.
bool overlap_battery(const std::string& dir, const char* pipe, const char* batch, const char* wmma,
                     const std::vector<float>& x) {
    using strata::kernels::Exl3ExpertStream;
    const std::string tag = std::string("overlap PIPE=") + pipe + ",BATCH=" + batch + ",WMMA=" + wmma;
    set_env("STRATA_EXL3_PIPE", pipe);
    set_env("STRATA_EXL3_EXPERT_BATCH", batch);
    set_env("STRATA_EXL3_WMMA", wmma);
    set_env("STRATA_EXL3_WAVESYNC", "0");
    set_env("STRATA_EXL3_PREFILL_BORROW", "0");
    bool ok = true;
    auto ids_span = [](int tokens, int k, int seed, int span) {
        std::vector<int32_t> ids((size_t) tokens * k);
        for (int t = 0; t < tokens; ++t)
            for (int j = 0; j < k; ++j)
                ids[(size_t) t * k + j] = (t * 3 + j * 7 + seed) % span;
        return ids;
    };
    enum Phase { kAllMiss, kAllHit, kMixed, kDupIds, kTok1, kTok2, kTok4, kTok8, kBig, kCache0, kLayer1, kLoanIn, kLoanWarm, kPhases };
    static const char* phase_name[kPhases] = { "all-miss window", "all-hit window", "mixed window (the split)",
        "duplicate ids in one token", "tokens=1", "tokens=2", "tokens=4", "tokens=8", "512 routes / 80 experts",
        "cache 0 (scratch)", "nonzero layer range (other geometry)", "window inside a loan", "window after restore" };
    std::vector<float> out[2][kPhases];
    long long splits_after_mixed[2] = {-1, -1}, fallbacks_after_mixed[2] = {-1, -1};
    long long splits_all[2] = {}, fallbacks_all[2] = {};
    bool lend_ok[2] = {}, restore_ok[2] = {};
    for (int arm = 0; arm < 2; ++arm) {
        set_env("STRATA_EXL3_OVERLAP", arm ? "1" : "0");
        {
            Exl3ExpertStream s(dir, kEmb, 1, kSlots, nullptr);
            auto win = [&](const std::vector<int32_t>& ids, int tokens, int k, int phase) {
                out[arm][phase].assign((size_t) tokens * k * kEmb, -1.0f);
                return s.run_multi(0, x.data(), ids.data(), tokens, k, out[arm][phase].data(), nullptr);
            };
            // exactly four distinct experts: the first call misses all four (fallback), the repeat
            // hits all four (fallback again).  The third window routes three of the four residents
            // PLUS two cold experts: with the cache's 4-group waves, wave 0 then holds three hits
            // and one miss in the SAME wave - the split the battery exists to check (a wave of only
            // residents or only misses keeps the original order by design).
            ok = ok && win(ids_span(4, 4, 0, 4), 4, 4, kAllMiss);
            ok = ok && win(ids_span(4, 4, 0, 4), 4, 4, kAllHit);
            {
                const std::vector<int32_t> mixed = {0, 1, 2, 4, 5, 0, 1, 2, 4, 5, 0, 1, 2, 4, 5, 0};
                ok = ok && win(mixed, 4, 4, kMixed);
            }
            s.overlap_debug(&splits_after_mixed[arm], &fallbacks_after_mixed[arm]);
            {   // one token, the same expert twice per pair: duplicate ids inside one token
                const std::vector<int32_t> dup = {3, 3, 7, 7, 3, 7, 11, 11};
                ok = ok && win(dup, 1, 8, kDupIds);
            }
            ok = ok && win(ids_span(1, 6, 3, 16), 1, 6, kTok1);
            ok = ok && win(ids_span(2, 8, 5, 16), 2, 8, kTok2);
            ok = ok && win(ids_span(4, 8, 7, 16), 4, 8, kTok4);
            ok = ok && win(ids_span(8, 8, 9, 16), 8, 8, kTok8);
            ok = ok && win(ids_span(64, 8, 11, 80), 64, 8, kBig);
            s.overlap_debug(&splits_all[arm], &fallbacks_all[arm]);
        }
        {
            Exl3ExpertStream s0(dir, kEmb, 1, 0, nullptr);   // cache 0: every use re-stages
            out[arm][kCache0].assign((size_t) 8 * 8 * kEmb, -1.0f);
            const std::vector<int32_t> ids = ids_span(8, 8, 13, 16);
            ok = ok && s0.run_multi(0, x.data(), ids.data(), 8, 8, out[arm][kCache0].data(), nullptr);
        }
        {
            Exl3ExpertStream s1(dir, kEmb, 2, kSlots, nullptr, 0, (size_t) 512 << 20, 1, 2);  // layers [1,2)
            out[arm][kLayer1].assign((size_t) 8 * 8 * kEmb, -1.0f);
            const std::vector<int32_t> ids = ids_span(8, 8, 15, 16);
            ok = ok && s1.run_multi(1, x.data(), ids.data(), 8, 8, out[arm][kLayer1].data(), nullptr);
        }
        {
            set_env("STRATA_EXL3_PREFILL_BORROW", "1");   // the loan interplay on the arena backing
            Exl3ExpertStream s(dir, kEmb, 1, kSlots, nullptr);
            {   // fill all four slots, then lend two: the routable prefix shrinks to two slots
                const std::vector<int32_t> fill = ids_span(4, 4, 0, 4);
                std::vector<float> ignored;
                ignored.assign((size_t) 4 * 4 * kEmb, -1.0f);
                ok = ok && s.run_multi(0, x.data(), fill.data(), 4, 4, ignored.data(), nullptr);
                const auto loan = s.lend_tail(2);
                lend_ok[arm] = loan.address != nullptr && s.loan_active() && s.loan_active_ranks() == 2;
                const std::vector<int32_t> ids = ids_span(8, 8, 21, 16);   // mixed window inside the loan
                out[arm][kLoanIn].assign((size_t) 8 * 8 * kEmb, -1.0f);
                ok = ok && s.run_multi(0, x.data(), ids.data(), 8, 8, out[arm][kLoanIn].data(), nullptr);
                ok = ok && s.loan_dense_invariant();
                restore_ok[arm] = s.restore_tail() && !s.loan_active() && s.loan_dense_invariant();
                const std::vector<int32_t> warm = ids_span(4, 4, 23, 16);  // reads restored + resident slots
                out[arm][kLoanWarm].assign((size_t) 4 * 4 * kEmb, -1.0f);
                ok = ok && s.run_multi(0, x.data(), warm.data(), 4, 4, out[arm][kLoanWarm].data(), nullptr);
            }
            set_env("STRATA_EXL3_PREFILL_BORROW", "0");
        }
    }
    set_env("STRATA_EXL3_OVERLAP", "0");
    for (int p = 0; p < kPhases; ++p)
        ok = compare(out[0][p], out[1][p], (tag + ", " + phase_name[p]).c_str()) && ok;
    // split evidence: the arm split the mixed phases and never silently fell back; the oracle
    // (env off) counted nothing at all
    const bool counts = splits_after_mixed[1] >= 1 && fallbacks_after_mixed[1] >= 2 &&
                        splits_after_mixed[0] == 0 && fallbacks_after_mixed[0] == 0 &&
                        splits_all[1] > splits_after_mixed[1] && lend_ok[0] && lend_ok[1] &&
                        restore_ok[0] && restore_ok[1];
    std::printf("%s: split evidence splits %lld->%lld (after the mixed phase) / %lld (all), fallbacks %lld, "
                "lend %d/%d restore %d/%d %s\n", tag.c_str(), splits_after_mixed[0], splits_after_mixed[1],
                splits_all[1], fallbacks_all[1], lend_ok[0], lend_ok[1], restore_ok[0], restore_ok[1],
                counts ? "PASS" : "FAIL");
    ok = ok && counts;
    return ok;
}

// The multilayer battery: BOTH fixture layers owned (K4 2560->640 and K3 ->128 geometries), each
// with its own full 4-slot cache.  A lend takes the tail of EVERY layer at once; restore places
// each layer's survivors into THAT layer's tail (per-layer slot spaces), so this is the regression
// for cross-layer placement and bounds.  Full-cache lend, poison, churn on both layers during the
// loan, growth to the maximal loan, restore with per-layer saved/restored bounds, an invariant
// check after every step, hit-only reads per layer immediately after restore, and a second turn -
// all capture-replayed against an original-backing oracle.
bool loan_multilayer(const std::string& dir, const std::vector<float>& x) {
    using strata::kernels::Exl3ExpertStream;
    using Loan = Exl3ExpertStream::Loan;
    const std::string tag = "loan multi-layer";
    set_env("STRATA_EXL3_PIPE", "1");
    set_env("STRATA_EXL3_EXPERT_BATCH", "1");
    set_env("STRATA_EXL3_WMMA", "1");
    set_env("STRATA_EXL3_WAVESYNC", "1");
    bool ok = true;
    std::vector<LoanCall> calls;
    std::vector<std::vector<float>> oracle_out;
    {
        set_env("STRATA_EXL3_PREFILL_BORROW", "1");
        Exl3ExpertStream s(dir, kEmb, 2, kSlots, nullptr);
        LoanBattery b{s, x, calls};
        const size_t rb = s.loan_rank_bytes();
        b.expect(s.loan_available() && rb > 0 && s.loan_active_ranks() == kSlots && s.loan_dense_invariant(),
                 "two-layer stream reports its loan capacity");
        b.device(0, churn_ids(0, 96, 4), 96, 4, "layer0 fill");
        b.device(1, churn_ids(1, 96, 4), 96, 4, "layer1 fill (K3 geometry)");
        b.expect(s.loan_dense_invariant(), "both layers full, invariant holds");
        {
            const Loan l = s.lend_tail(2);
            std::vector<int> s0, s1;
            s.loan_debug_saved(0, s0);
            s.loan_debug_saved(1, s1);
            b.expect(l.address != nullptr && !s0.empty() && s0.size() <= 2 && !s1.empty() && s1.size() <= 2 &&
                     s.loan_active_ranks() == 2 && s.loan_dense_invariant(),
                     "both layers' tails captured at lend, each within the rank bound");
            b.ok = b.ok && poison_range(l.address, l.bytes, 0xFF, "NaN poison (both layers)");
            b.device(0, churn_ids(2, 96, 4), 96, 4, "layer0 churn inside the loan");
            b.device(1, churn_ids(3, 96, 4), 96, 4, "layer1 churn inside the loan");
            b.expect(s.loan_dense_invariant(), "invariant during the loan, both layers");
            const Loan three = s.lend_tail(3);
            std::vector<int> g0, g1;
            s.loan_debug_saved(0, g0);
            s.loan_debug_saved(1, g1);
            b.expect(three.address != nullptr && three.bytes == 3 * rb && s.loan_active_ranks() == 1 &&
                     g0.size() <= 3 && g1.size() <= 3 && s.loan_dense_invariant(),
                     "growth takes both layers to the maximal loan, saved ids bounded per layer");
            b.ok = b.ok && poison_range(three.address, three.bytes, 0xFF, "NaN poison (grown, both layers)");
            b.device(1, churn_ids(5, 48, 4), 48, 4, "layer1 run under the maximal loan");
            std::vector<int> r0, r1;
            b.expect(s.restore_tail() && s.loan_dense_invariant(), "multilayer restore");
            s.loan_debug_restored(0, r0);
            s.loan_debug_restored(1, r1);
            b.expect(r0.size() <= 3 && r1.size() <= 3, "each layer's survivors within its own tail bound");
            if (!r0.empty())
                b.device(0, ids_only(r0, 96, 4), 96, 4, "layer0 hit-only read of its restored experts");
            if (!r1.empty())
                b.device(1, ids_only(r1, 96, 4), 96, 4, "layer1 hit-only read of its restored experts");
        }
        b.device(0, churn_ids(6, 96, 4), 96, 4, "layer0 broad churn after the direct reads");
        b.device(1, churn_ids(7, 96, 4), 96, 4, "layer1 broad churn after the direct reads");
        {
            const Loan l = s.lend_tail(2);
            std::vector<int> s0, s1;
            s.loan_debug_saved(0, s0);
            s.loan_debug_saved(1, s1);
            b.expect(l.address != nullptr && s0.size() <= 2 && s1.size() <= 2 && s.loan_dense_invariant(),
                     "second-turn lend, both layers");
            b.ok = b.ok && poison_range(l.address, l.bytes, 0xFF, "NaN poison (second turn, both layers)");
            b.device(0, churn_ids(8, 96, 4), 96, 4, "second-turn layer0 churn inside the loan");
            b.device(1, churn_ids(9, 96, 4), 96, 4, "second-turn layer1 churn inside the loan");
            std::vector<int> r0, r1;
            b.expect(s.restore_tail() && s.loan_dense_invariant(), "second-turn multilayer restore");
            s.loan_debug_restored(0, r0);
            s.loan_debug_restored(1, r1);
            b.expect(r0.size() <= 2 && r1.size() <= 2, "second-turn survivors bounded per layer");
            if (!r0.empty())
                b.device(0, ids_only(r0, 96, 4), 96, 4, "second-turn layer0 hit-only read");
            if (!r1.empty())
                b.device(1, ids_only(r1, 96, 4), 96, 4, "second-turn layer1 hit-only read");
            b.device(0, churn_ids(0, 96, 4), 96, 4, "layer0 broad churn last");
            b.device(1, churn_ids(1, 96, 4), 96, 4, "layer1 broad churn last");
        }
        b.expect(s.loan_dense_invariant() && !s.loan_active(), "multilayer battery ends invariant-clean");
        if (!b.ok) return false;
    }
    {
        set_env("STRATA_EXL3_PREFILL_BORROW", "0");
        if (!loan_replay(dir, x, 2, kSlots, calls, oracle_out) || oracle_out.size() != calls.size()) {
            std::printf("%s, oracle replay failed\n", tag.c_str());
            return false;
        }
        for (size_t i = 0; i < calls.size(); ++i)
            ok = ok && compare(oracle_out[i], calls[i].out, (tag + ", " + calls[i].what).c_str());
    }
    return ok;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s <model_dir>\n", argv[0]); return 2; }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    const std::string dir = argv[1];
    std::vector<int32_t> ids_c((size_t) 5000, 2);
    std::vector<int32_t> ids_d(616);
    for (int t = 0; t < 616; ++t) ids_d[(size_t) t] = t < 16 ? 0 : 1;
    std::vector<int32_t> ids_g(200);
    for (int t = 0; t < 200; ++t) ids_g[(size_t) t] = t % 10;
    std::vector<int32_t> ids_h(320 * 4), ids_i(80 * 2);
    for (int t = 0; t < 320; ++t)
        for (int j = 0; j < 4; ++j) ids_h[(size_t) t * 4 + j] = (t * 4 + j) % kExperts;
    for (int t = 0; t < 80; ++t)
        for (int j = 0; j < 2; ++j) ids_i[(size_t) t * 2 + j] = (t * 2 + j) % kExperts;
    std::vector<float> x(96 * kEmb);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = std::sin(0.031f * (float) i) * 0.25f + 0.05f * (float) ((i * 13) % 97);
    std::vector<float> x_c(5000 * kEmb);
    for (size_t i = 0; i < x_c.size(); ++i)
        x_c[i] = std::cos(0.017f * (float) i) * 0.2f - 0.03f * (float) ((i * 29) % 89);
    std::vector<float> x_d(616 * kEmb);
    for (size_t i = 0; i < x_d.size(); ++i)
        x_d[i] = std::sin(0.023f * (float) i) * 0.22f + 0.04f * (float) ((i * 37) % 83);
    std::vector<float> x_h(320 * kEmb);
    for (size_t i = 0; i < x_h.size(); ++i)
        x_h[i] = std::cos(0.011f * (float) i) * 0.18f + 0.06f * (float) ((i * 17) % 79);
    bool ok = true;
    try {
        for (const char* wmma : {"0", "1"}) {
            set_env("STRATA_EXL3_WMMA", wmma);
            set_env("STRATA_EXL3_EXPERT_BATCH", "0");
            set_env("STRATA_EXL3_PREFILL_BORROW", "0");
            ArmOut def, wav, pip, pipw, ram, bat, bat_serial, arena;
            {
                set_env("STRATA_EXL3_WAVESYNC", "0");   // explicit: an inherited env must not pick the arm
                set_env("STRATA_EXL3_PIPE", "0");
                ok = ok && run_arm(dir, def, x, x_c, x_d, x_h, ids_c, ids_d, ids_g, ids_h, ids_i);
            }
            {
                set_env("STRATA_EXL3_WAVESYNC", "1");
                set_env("STRATA_EXL3_PIPE", "0");
                ok = ok && run_arm(dir, wav, x, x_c, x_d, x_h, ids_c, ids_d, ids_g, ids_h, ids_i);
            }
            {
                set_env("STRATA_EXL3_WAVESYNC", "0");   // PIPE implies the wave batching on its own
                set_env("STRATA_EXL3_PIPE", "1");
                ok = ok && run_arm(dir, pip, x, x_c, x_d, x_h, ids_c, ids_d, ids_g, ids_h, ids_i);
            }
            {
                set_env("STRATA_EXL3_WAVESYNC", "1");
                set_env("STRATA_EXL3_PIPE", "1");       // both opt-ins together: identical to PIPE alone
                ok = ok && run_arm(dir, pipw, x, x_c, x_d, x_h, ids_c, ids_d, ids_g, ids_h, ids_i);
            }
            // Eight RAM slots: partial RAM hits, cross-wave eviction, and warm source reuse.
            // stage_parallel joins the workers before RAM pins reset; DMA only owns pinned buffers.
            ok = ok && run_arm(dir, ram, x, x_c, x_d, x_h, ids_c, ids_d, ids_g, ids_h, ids_i,
                               (size_t) 20 << 20);
            set_env("STRATA_EXL3_EXPERT_BATCH", "1");
            ok = ok && run_arm(dir, bat, x, x_c, x_d, x_h, ids_c, ids_d, ids_g, ids_h, ids_i);
            set_env("STRATA_EXL3_PIPE", "0");
            ok = ok && run_arm(dir, bat_serial, x, x_c, x_d, x_h, ids_c, ids_d, ids_g, ids_h, ids_i);
            set_env("STRATA_EXL3_PIPE", "1");
            set_env("STRATA_EXL3_PREFILL_BORROW", "1");
            ok = ok && run_arm(dir, arena, x, x_c, x_d, x_h, ids_c, ids_d, ids_g, ids_h, ids_i);
            set_env("STRATA_EXL3_PREFILL_BORROW", "0");
            set_env("STRATA_EXL3_EXPERT_BATCH", "0");
            if (!ok) return 1;
            const std::string tag = std::string("WMMA=") + wmma;
            struct { const ArmOut* arm; const char* name; } arms[] = {
                { &wav, "wave" }, { &pip, "pipe" }, { &pipw, "pipe+wave" }, { &ram, "pipe+RAM" }, { &bat, "expert-batch" }, { &bat_serial, "expert-batch-no-PIPE" }, { &arena, "rank-major-arena" },
            };
            struct { const std::vector<float> ArmOut::* field; const char* what; } shapes[] = {
                { &ArmOut::a, "group path (>=16 rows), 4 waves with eviction" },
                { &ArmOut::b, "indexed path (<16 rows)" },
                { &ArmOut::c, "one group over two tiles" },
                { &ArmOut::d, "16-row group then 600-row group in one wave" },
                { &ArmOut::g, "partial final wave (waves 4+4+2)" },
                { &ArmOut::warm_a, "warm cache, repeated call" },
                { &ArmOut::h, "scratch: 80 groups over two waves, slot reuse in flight" },
                { &ArmOut::i, "scratch: indexed path, slot reuse in flight" },
                { &ArmOut::warm_h, "scratch: warm repeat" },
            };
            for (const auto& arm : arms)
                for (const auto& sh : shapes) {
                    const std::string what = tag + ", " + arm.name + " vs default, " + sh.what;
                    ok = ok && compare(def.*(sh.field), (arm.arm)->*(sh.field), what.c_str());
                }
            if (!ok) return 1;
            ok = ok && compare(pip.a, pip.warm_a, (tag + ", pipe arm, cold vs warm cache").c_str());
            ok = ok && compare(pip.h, pip.warm_h, (tag + ", pipe arm scratch, cold vs warm").c_str());
            if (!ok) return 1;
        }
        // Rank-major arithmetic and overflow checks need no device allocations.
        {
            using Layout = strata::kernels::Exl3ArenaLayout;
            std::vector<Layout::Planes> sizes(2);
            sizes[0].fill(512); sizes[1].fill(768);
            Layout layout(sizes, 4);
            ok = ok && layout.bytes == 4 * layout.rank_bytes;
            size_t previous_end = 0;
            for (size_t l = 0; l < sizes.size(); ++l)
                for (size_t p = 0; p < 9; ++p) {
                    const size_t off = layout.offsets[l][p];
                    const size_t end = off + sizes[l][p];
                    ok = ok && off % 256 == 0 && off >= previous_end && end <= layout.rank_bytes;
                    previous_end = end;
                    for (int rank = 0; rank < 4; ++rank)
                        ok = ok && rank * layout.rank_bytes + end <= (rank + 1) * layout.rank_bytes;
                }
            Layout zero(sizes, 0);
            ok = ok && zero.bytes == 0 && zero.rank_bytes == layout.rank_bytes;
            for (int invalid : {-1, 513}) {
                bool rejected = false;
                try { Layout bad(sizes, invalid); } catch (const std::invalid_argument&) { rejected = true; }
                ok = ok && rejected;
            }
            bool rejected = false;
            try { Layout bad({}, 4); } catch (const std::invalid_argument&) { rejected = true; }
            ok = ok && rejected;
            auto zero_plane = sizes; zero_plane[1][4] = 0;
            rejected = false;
            try { Layout bad(zero_plane, 4); } catch (const std::invalid_argument&) { rejected = true; }
            ok = ok && rejected;
            rejected = false;
            try { Layout bad(sizes, 513); } catch (const std::invalid_argument&) { rejected = true; }
            ok = ok && rejected;
            rejected = false; sizes[0][0] = SIZE_MAX;
            try { Layout bad(sizes, 4); } catch (const std::overflow_error&) { rejected = true; }
            ok = ok && rejected;
        }
        // Two layers, differing K/hidden width, repeated eviction and a nonzero stage range.
        std::vector<float> mixed[2][3];
        std::vector<int32_t> mixed_ids(96 * 4);
        for (size_t i = 0; i < mixed_ids.size(); ++i) mixed_ids[i] = (int32_t) ((i * 7 + 3) % 16);
        for (int arm = 0; arm < 2; ++arm) {
            set_env("STRATA_EXL3_PREFILL_BORROW", arm ? "1" : "0");
            set_env("STRATA_EXL3_PIPE", "1");
            set_env("STRATA_EXL3_WMMA", "1");
            set_env("STRATA_EXL3_EXPERT_BATCH", "1");
            strata::kernels::Exl3ExpertStream s(dir, kEmb, 2, kSlots, nullptr);
            ok = ok && run_ids(s, 96, 4, mixed_ids, x, mixed[arm][0], 0);
            ok = ok && run_ids(s, 96, 4, mixed_ids, x, mixed[arm][1], 1);
            ok = ok && run_ids(s, 96, 4, mixed_ids, x, mixed[arm][2], 0);
        }
        for (int i = 0; i < 3; ++i)
            ok = compare(mixed[0][i], mixed[1][i], "arena mixed-layer geometry/repeated eviction") && ok;
        {
            int previous = -1; cudaGetDevice(&previous);
            strata::kernels::Exl3ExpertStream stage(dir, kEmb, 2, kSlots, nullptr, 0, (size_t) 512 << 20, 1, 2);
            std::vector<float> y;
            ok = ok && run_ids(stage, 96, 4, mixed_ids, x, y, 1);
            ok = compare(mixed[1][1], y, "arena nonzero layer range") && ok;
            int current = -1; cudaGetDevice(&current); ok = ok && current == previous;
        }
        {
            set_env("STRATA_EXL3_PREFILL_BORROW", "1");
            int before = -1; cudaGetDevice(&before);
            bool rejected = false;
            try { strata::kernels::Exl3ExpertStream missing(dir, kEmb, 3, kSlots, nullptr); }
            catch (const std::exception&) { rejected = true; }
            int after = -1; cudaGetDevice(&after); ok = ok && rejected && before == after;
            auto owned = std::make_unique<strata::kernels::Exl3ExpertStream>(dir, kEmb, 2, kSlots, nullptr);
            if (devices > 1) cudaSetDevice((before + 1) % devices);
            int caller = -1; cudaGetDevice(&caller);
            owned.reset();
            cudaGetDevice(&after); ok = ok && caller == after;
            cudaSetDevice(before);
        }
        set_env("STRATA_EXL3_PREFILL_BORROW", "0");
        set_env("STRATA_EXL3_EXPERT_BATCH", "0");
        // The stream loan battery: the arena stream lends tail ranks, the borrower overwrites them
        // (0x5A read back, then 0xFF - an fp16 NaN pattern that makes any read of a lent or
        // unrestored byte non-finite), runs continue on the active prefix and restore_tail()
        // refills.  The loan arm records every call - including calls whose ids were captured from
        // the saved/restored seams - and an original-backing oracle replays the list.  Once with
        // the PIPE+EXPERT_BATCH arm on and once with every opt-in off; then the two-layer battery
        // (both fixture geometries owned, per-layer tails and bounds).
        ok = ok && loan_battery(dir, "1", "1", "1", x);
        ok = ok && loan_battery(dir, "0", "0", "0", x);
        ok = ok && loan_multilayer(dir, x);
        ok = ok && loan_repeated_growth(dir, x);
        if (!ok) return 1;
        // The host-IO overlap battery: STRATA_EXL3_OVERLAP's hit/miss split vs the original order,
        // once with PIPE+BATCH+WMMA on and once with every opt-in off (see its header comment).
        ok = ok && overlap_battery(dir, "1", "1", "1", x);
        ok = ok && overlap_battery(dir, "0", "0", "0", x);
        if (!ok) return 1;
        // Phase-sequence leg (kept from the frozen-cache experiment): prefill through run_device,
        // the single-token run() path, then decode windows through the host-IO run_multi path - the
        // verifier windows' entry - across two turns.  Two identically-constructed streams (one with
        // STRATA_EXL3_PIPE, one without) must produce byte-identical outputs at every step (cache
        // fill, eviction and admission are deterministic): the pipeline changes only scheduling.
        // Arms 4/5 add STRATA_EXL3_OVERLAP on top (default and PIPE): the overlap arm must also be
        // byte-identical through the full prefill -> run() -> window -> prefill -> window sequence.
        std::vector<float> win_out[6][3], pre2_out[6], single_out[6];
        for (int arm = 0; arm < 6; ++arm) {
            set_env("STRATA_EXL3_WMMA", "1");
            set_env("STRATA_EXL3_WAVESYNC", "0");
            set_env("STRATA_EXL3_PIPE", arm == 1 || arm == 3 || arm == 5 ? "1" : "0");
            set_env("STRATA_EXL3_PREFILL_BORROW", arm == 3 ? "1" : "0"); // arena vs PIPE arm
            set_env("STRATA_EXL3_OVERLAP", arm >= 4 ? "1" : "0");
            strata::kernels::Exl3ExpertStream s(dir, kEmb, 1, kSlots, nullptr);
            std::vector<float> first;
            ok = ok && run_shape(s, 96, 8, 0, 16, x, first);             // turn-1 prefill (fills/evicts)
            ok = ok && run_single(s, 2, x, single_out[arm]);             // single-token run() after prefill
            ok = ok && run_window(s, 24, 8, 1, x, win_out[arm][0]);      // decode window after prefill
            ok = ok && run_window(s, 24, 8, 5, x, win_out[arm][1]);      // second window (admissions hit)
            ok = ok && run_shape(s, 96, 8, 3, 16, x, pre2_out[arm]);     // turn-2 prefill
            ok = ok && run_window(s, 24, 8, 7, x, win_out[arm][2]);      // window after the new prefill
        }
        set_env("STRATA_EXL3_OVERLAP", "0");
        if (!ok) return 1;
        ok = compare_rel(single_out[0], single_out[2], 5e-3, "single-token run(), default vs default (atomic-accumulation noise)") && ok;
        ok = compare(win_out[0][0], win_out[2][0], "window after prefill, default vs default (determinism)") && ok;
        ok = compare_rel(single_out[0], single_out[1], 5e-3, "single-token run() after prefill, pipe vs default") && ok;
        ok = compare(win_out[0][0], win_out[1][0], "window after prefill, pipe vs default") && ok;
        ok = compare(win_out[0][1], win_out[1][1], "second window, pipe vs default") && ok;
        ok = compare(pre2_out[0], pre2_out[1], "second-turn prefill, pipe vs default") && ok;
        ok = compare(win_out[0][2], win_out[1][2], "window after second prefill, pipe vs default") && ok;
        ok = compare_rel(single_out[1], single_out[3], 5e-3, "arena run() vs original backing") && ok;
        for (int i = 0; i < 3; ++i)
            ok = compare(win_out[1][i], win_out[3][i], "arena host window phase parity") && ok;
        ok = compare(pre2_out[1], pre2_out[3], "arena second-turn prefill parity") && ok;
        ok = compare_rel(single_out[0], single_out[4], 5e-3, "single-token run() after prefill, overlap vs default") && ok;
        for (int i = 0; i < 3; ++i) {
            const char* step[3] = {"window after prefill", "second window", "window after second prefill"};
            ok = compare(win_out[0][i], win_out[4][i], (std::string(step[i]) + ", overlap vs default").c_str()) && ok;
            ok = compare(win_out[0][i], win_out[5][i], (std::string(step[i]) + ", overlap+pipe vs default").c_str()) && ok;
        }
        ok = compare(pre2_out[0], pre2_out[4], "second-turn prefill, overlap vs default (prefill untouched)") && ok;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        return 1;
    }
    if (std::getenv("STRATA_EXL3_WAVE_CHECK_HASH") != nullptr)
        std::printf("SEQUENCEHASH %016llx\n", (unsigned long long) g_seq_hash);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
