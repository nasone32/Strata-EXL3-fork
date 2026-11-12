// src/kernels/mimo_router_parity.cpp - the MiMo v2 router (src/kernels/cuda/mimo_router.cu) against an independent CPU
// reference: the F16 gemv, sigmoid scoring, noaux_tc biased top-8 selection (descending biased score, ties to
// the lower expert id) and the unbiased renormalised weights (notes/MIMO-V2-PORT.md §1).  The reference is
// written from the spec, in double, sharing no code with the kernel; ids match EXACTLY (the random inputs have
// comfortable gaps between selection scores, so a rounding difference cannot flip a pick - the deliberate tie
// cases are exact in F16 by construction), the weights within FP32-order slack, and every row's weights sum to
// 1.  Cases: random batches (64 and 7 rows, two hidden widths), an all-tie row (zero input), duplicated router
// rows (exact score ties inside the top-8), a bias that flips the top-8 against the raw sigmoid order (one
// expert pushed in, the raw winner pushed out), a NaN router row (score clamped to 0, never selected) and a
// saturated row (sigmoid exactly 0/1 in FP32).  The WINDOW arm (mimo_router_route_window, the
// T=2..8 verify windows' batched split kernels) is gated BYTE-EXACT against the one-block
// kernel on the same device buffers - per row count 1/2/3/4/5/8, the real geometry 256x4096
// top-8, the fixture's 4/2, a 33-expert/k-5 and a k-32 shape, with tie/bias/NaN inputs, tail
// guards on every buffer, a scores-scratch full-write check and the pre-launch validation
// battery (bad shapes and aliased buffers refused, outputs untouched).  `--bench [reps batch]`
// times the one-block kernel against the window arm at the real geometry with CUDA events
// (median/quartiles per row count; launch-count facts only, no sub-us claims).  GPU-gated:
// exits 77 (skip) with no device, the convention of qsa_topk_active_parity.
#include <cuda_runtime.h>
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/mimo_router.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <bit>
#include <functional>
#include <array>
#include <algorithm>
#include <limits>
#include <stdexcept>

#define CHECK(call)                                                                                                  \
    do {                                                                                                             \
        const cudaError_t e = (call);                                                                                \
        if (e != cudaSuccess) {                                                                                      \
            std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e));                                          \
            std::exit(2);                                                                                            \
        }                                                                                                            \
    } while (0)

namespace k = strata::kernels;

static constexpr int NE = 256, K = 8;

/// The spec in double: unbiased sigmoid scores, biased selection (ties: the lower expert id), unbiased
/// renormalised weights.  A NaN dot reads as score 0 and a NaN biased score is never selected, mirroring the
/// kernel's documented clamps.
static void reference(const std::vector<uint16_t>& router, const std::vector<uint16_t>& x, const std::vector<float>& bias,
                      int rows, int n_embd, std::vector<int32_t>& ids, std::vector<float>& weights) {
    ids.assign((size_t) rows * K, -1);
    weights.assign((size_t) rows * K, 0.0f);
    std::vector<double> s(NE);
    for (int r = 0; r < rows; ++r) {
        for (int e = 0; e < NE; ++e) {
            double dot = 0.0;
            for (int i = 0; i < n_embd; ++i)
                dot += (double) k::f32_from_f16(router[(size_t) e * n_embd + i]) *
                       (double) k::f32_from_f16(x[(size_t) r * n_embd + i]);
            const double sig = 1.0 / (1.0 + std::exp(-dot));
            s[e] = std::isnan(sig) ? 0.0 : sig;
        }
        std::vector<char> taken(NE, 0);
        std::vector<double> sel(K, 0.0);
        double sum = 0.0;
        for (int rank = 0; rank < K; ++rank) {
            int best_e = -1;
            double best_c = 0.0;
            for (int e = 0; e < NE; ++e) {   // the first pass wins ties: the LOWER expert id
                if (taken[e]) continue;
                const double c = s[e] + (double) bias[e];
                if (std::isnan(c)) continue;
                if (best_e < 0 || c > best_c) { best_c = c; best_e = e; }
            }
            taken[best_e] = 1;
            ids[(size_t) r * K + rank] = best_e;
            sel[rank] = s[best_e];
            sum += s[best_e];
        }
        for (int i = 0; i < K; ++i) weights[(size_t) r * K + i] = (float) (sel[i] / (sum + 1e-20));
    }
}

/// GPU run on the host arrays, copied back.  Fills `ids`/`weights`.
static void run_gpu(const std::vector<uint16_t>& router, const std::vector<uint16_t>& x, const std::vector<float>& bias,
                    int rows, int n_embd, std::vector<int32_t>& ids, std::vector<float>& weights) {
    uint16_t* dw = nullptr;
    uint16_t* dx = nullptr;
    float* db = nullptr;
    int32_t* di = nullptr;
    float* dw8 = nullptr;
    CHECK(cudaMalloc(&dw, router.size() * 2));
    CHECK(cudaMalloc(&dx, x.size() * 2));
    CHECK(cudaMalloc(&db, bias.size() * 4));
    CHECK(cudaMalloc(&di, (size_t) rows * K * 4));
    CHECK(cudaMalloc(&dw8, (size_t) rows * K * 4));
    CHECK(cudaMemcpy(dw, router.data(), router.size() * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(di, 0x7f, (size_t) rows * K * 4));   // poison: an unwritten rank shows up as INT32_MAX
    CHECK(cudaMemset(dw8, 0x7f, (size_t) rows * K * 4));
    k::mimo_router_route(dw, dx, db, di, dw8, rows, n_embd, NE, K, nullptr);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    ids.resize((size_t) rows * K);
    weights.resize((size_t) rows * K);
    CHECK(cudaMemcpy(ids.data(), di, ids.size() * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(weights.data(), dw8, weights.size() * 4, cudaMemcpyDeviceToHost));
    (void) cudaFree(dw);
    (void) cudaFree(dx);
    (void) cudaFree(db);
    (void) cudaFree(di);
    (void) cudaFree(dw8);
}

static int row_of(const std::vector<int32_t>& ids, int r, int expert) {
    for (int i = 0; i < K; ++i)
        if (ids[(size_t) r * K + i] == expert) return i;
    return -1;
}

/// The decode arm on one row: the same host arrays through `mimo_router_route_decode`, plus the
/// one-block kernel's output on the same inputs so a case can demand bit-identity between arms.
static void run_gpu_decode(const std::vector<uint16_t>& router, const std::vector<uint16_t>& x,
                           const std::vector<float>& bias, int n_embd, std::vector<int32_t>& ids,
                           std::vector<float>& weights, std::vector<int32_t>* mono_ids = nullptr,
                           std::vector<float>* mono_w = nullptr) {
    uint16_t* dw = nullptr;
    uint16_t* dx = nullptr;
    float* db = nullptr;
    float* ds = nullptr;
    int32_t* di = nullptr;
    float* dw8 = nullptr;
    int32_t* mi = nullptr;
    float* mw = nullptr;
    CHECK(cudaMalloc(&dw, router.size() * 2));
    CHECK(cudaMalloc(&dx, x.size() * 2));
    CHECK(cudaMalloc(&db, bias.size() * 4));
    CHECK(cudaMalloc(&ds, (size_t) NE * 4));
    CHECK(cudaMalloc(&di, (size_t) K * 4));
    CHECK(cudaMalloc(&dw8, (size_t) K * 4));
    CHECK(cudaMemcpy(dw, router.data(), router.size() * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(di, 0x7f, (size_t) K * 4));
    CHECK(cudaMemset(dw8, 0x7f, (size_t) K * 4));
    k::mimo_router_route_decode(dw, dx, db, ds, di, dw8, n_embd, NE, K, nullptr);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    ids.resize(K);
    weights.resize(K);
    CHECK(cudaMemcpy(ids.data(), di, K * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(weights.data(), dw8, K * 4, cudaMemcpyDeviceToHost));
    if (mono_ids != nullptr) {   // the one-block kernel on the same buffers
        CHECK(cudaMalloc(&mi, (size_t) K * 4));
        CHECK(cudaMalloc(&mw, (size_t) K * 4));
        CHECK(cudaMemset(mi, 0x7f, (size_t) K * 4));
        CHECK(cudaMemset(mw, 0x7f, (size_t) K * 4));
        k::mimo_router_route(dw, dx, db, mi, mw, 1, n_embd, NE, K, nullptr);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        mono_ids->resize(K);
        mono_w->resize(K);
        CHECK(cudaMemcpy(mono_ids->data(), mi, K * 4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(mono_w->data(), mw, K * 4, cudaMemcpyDeviceToHost));
    }
    (void) cudaFree(dw);
    (void) cudaFree(dx);
    (void) cudaFree(db);
    (void) cudaFree(ds);
    (void) cudaFree(di);
    (void) cudaFree(dw8);
    (void) cudaFree(mi);
    (void) cudaFree(mw);
}

/// The window arm (rows tokens through `mimo_router_route_window`) against the one-block
/// kernel on the SAME device buffers: ids and weights must be BIT-IDENTICAL (the split
/// kernels share the one-block kernel's per-warp accumulation and warp-0 selection, so any
/// difference is a real bug, not rounding).  Every device buffer carries a poison TAIL guard
/// (a row-stride overrun writes poison-visible bytes past the logical end) and the scores
/// scratch must be fully rewritten (a sigmoid score is always in [0,1]; the 0x7f poison
/// pattern is ~3.4e38).  Returns 0 on pass; fills this arm's host ids/weights either way.
static int run_gpu_window(const char* name, const std::vector<uint16_t>& router,
                          const std::vector<uint16_t>& x, const std::vector<float>& bias,
                          int rows, int n_embd, int ne, int k, std::vector<int32_t>& ids,
                          std::vector<float>& weights) {
    constexpr size_t kGuard = 64;   // BYTES of poison past every guarded buffer's end
    const size_t out_floats = (size_t) rows * k;
    uint16_t* dw = nullptr;
    uint16_t* dx = nullptr;
    float* db = nullptr;
    float* ds = nullptr;
    int32_t* di = nullptr;
    float* dw8 = nullptr;
    int32_t* mi = nullptr;
    float* mw = nullptr;
    CHECK(cudaMalloc(&dw, router.size() * 2 + kGuard));
    CHECK(cudaMalloc(&dx, x.size() * 2 + kGuard));
    CHECK(cudaMalloc(&db, bias.size() * 4 + kGuard));
    CHECK(cudaMalloc(&ds, (size_t) rows * ne * 4 + kGuard));
    CHECK(cudaMalloc(&di, out_floats * 4 + kGuard));
    CHECK(cudaMalloc(&dw8, out_floats * 4 + kGuard));
    CHECK(cudaMalloc(&mi, out_floats * 4 + kGuard));
    CHECK(cudaMalloc(&mw, out_floats * 4 + kGuard));
    CHECK(cudaMemcpy(dw, router.data(), router.size() * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemset((uint8_t*) dw + router.size() * 2, 0x7f, kGuard));
    CHECK(cudaMemset((uint8_t*) dx + x.size() * 2, 0x7f, kGuard));
    CHECK(cudaMemset((uint8_t*) db + bias.size() * 4, 0x7f, kGuard));
    CHECK(cudaMemset(ds, 0x7f, (size_t) rows * ne * 4 + kGuard));
    CHECK(cudaMemset(di, 0x7f, out_floats * 4 + kGuard));
    CHECK(cudaMemset(dw8, 0x7f, out_floats * 4 + kGuard));
    CHECK(cudaMemset(mi, 0x7f, out_floats * 4 + kGuard));
    CHECK(cudaMemset(mw, 0x7f, out_floats * 4 + kGuard));
    k::mimo_router_route_window(dw, dx, db, ds, di, dw8, rows, n_embd, ne, k, nullptr);
    CHECK(cudaGetLastError());
    k::mimo_router_route(dw, dx, db, mi, mw, rows, n_embd, ne, k, nullptr);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    ids.resize(out_floats);
    weights.resize(out_floats);
    std::vector<int32_t> fused_ids(out_floats);
    std::vector<float> fused_w(out_floats);
    CHECK(cudaMemcpy(ids.data(), di, ids.size() * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(weights.data(), dw8, weights.size() * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(fused_ids.data(), mi, fused_ids.size() * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(fused_w.data(), mw, fused_w.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<float> scores((size_t) rows * ne + kGuard / 4);
    std::vector<uint32_t> guards(5 * kGuard / 4);
    CHECK(cudaMemcpy(scores.data(), ds, scores.size() * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&guards[0], (uint8_t*) db + bias.size() * 4, kGuard, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&guards[kGuard / 4], (uint8_t*) dx + x.size() * 2, kGuard, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&guards[2 * (kGuard / 4)], (uint8_t*) dw + router.size() * 2, kGuard, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&guards[3 * (kGuard / 4)], (uint8_t*) di + out_floats * 4, kGuard, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&guards[4 * (kGuard / 4)], (uint8_t*) dw8 + out_floats * 4, kGuard, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int r = 0; r < rows; ++r) {   // the gate: BYTE EXACT against the one-block kernel
        if (std::memcmp(&ids[(size_t) r * k], &fused_ids[(size_t) r * k], (size_t) k * 4) != 0 ||
            std::memcmp(&weights[(size_t) r * k], &fused_w[(size_t) r * k], (size_t) k * 4) != 0) {
            if (bad < 2) {
                std::printf("  row %d: split ids", r);
                for (int i = 0; i < k; ++i) std::printf(" %d/%d", ids[(size_t) r * k + i], fused_ids[(size_t) r * k + i]);
                std::printf("  w");
                for (int i = 0; i < k; ++i)
                    std::printf(" %.7f/%.7f", weights[(size_t) r * k + i], fused_w[(size_t) r * k + i]);
                std::printf("\n");
            }
            ++bad;
        }
    }
    bool scores_full = true;   // every scratch score rewritten: all in [0,1], no poison left
    for (int i = 0; i < rows * ne; ++i)
        if (!(scores[(size_t) i] >= 0.0f && scores[(size_t) i] <= 1.0f)) scores_full = false;
    bool tail_ok = true;
    for (size_t i = (size_t) rows * ne; i < scores.size(); ++i)
        if (scores[i] != std::bit_cast<float>(0x7f7f7f7fu)) tail_ok = false;
    for (uint32_t g : guards) tail_ok = tail_ok && g == 0x7f7f7f7fu;
    (void) cudaFree(dw);
    (void) cudaFree(dx);
    (void) cudaFree(db);
    (void) cudaFree(ds);
    (void) cudaFree(di);
    (void) cudaFree(dw8);
    (void) cudaFree(mi);
    (void) cudaFree(mw);
    std::printf("%s: window arm vs one-block kernel (%s) - %d rows, %d experts, k %d, n_embd %d%s\n",
                bad || !scores_full || !tail_ok ? "FAIL" : "PASS", name, rows, ne, k, n_embd,
                bad || !scores_full || !tail_ok ? "" : ", ids/weights bit-identical");
    if (!scores_full) std::printf("  scores scratch not fully rewritten (poison or out-of-range score left)\n");
    if (!tail_ok) std::printf("  a tail guard was clobbered (write past a declared span)\n");
    return bad + (scores_full ? 0 : 1) + (tail_ok ? 0 : 1);
}

/// The window arm's input validation must refuse BEFORE any launch (std::invalid_argument,
/// outputs untouched).  `call` wraps an otherwise-valid setup with one broken argument; when a
/// probe pointer is given, its byte must survive the refusal (nothing ran).
static bool refuses(const char* what, const std::function<void()>& call, float* scratch_probe) {
    if (scratch_probe != nullptr) {
        CHECK(cudaMemset(scratch_probe, 0x5a, 4));
        CHECK(cudaDeviceSynchronize());
    }
    try {
        call();
    } catch (const std::invalid_argument&) {
        if (scratch_probe != nullptr) {
            uint32_t v = 0;
            CHECK(cudaMemcpy(&v, scratch_probe, 4, cudaMemcpyDeviceToHost));
            if (v != 0x5a5a5a5au) {
                std::printf("FAIL: window arm refuses %s - a launch ran before the refusal\n", what);
                return false;
            }
        }
        std::printf("PASS: window arm refuses %s (before launch)\n", what);
        return true;
    } catch (...) {
        std::printf("FAIL: window arm refuses %s - wrong exception type\n", what);
        return false;
    }
    std::printf("FAIL: window arm refuses %s - the call succeeded\n", what);
    return false;
}
static int compare(const char* name, const std::vector<uint16_t>& router, const std::vector<uint16_t>& x,
                   const std::vector<float>& bias, int rows, int n_embd) {
    std::vector<int32_t> ids;
    std::vector<float> weights;
    run_gpu(router, x, bias, rows, n_embd, ids, weights);
    std::vector<int32_t> ref_ids;
    std::vector<float> ref_w;
    reference(router, x, bias, rows, n_embd, ref_ids, ref_w);
    int bad = 0;
    for (int r = 0; r < rows; ++r) {
        bool ok = std::memcmp(&ids[(size_t) r * K], &ref_ids[(size_t) r * K], K * 4) == 0;
        double sum = 0.0;
        for (int i = 0; i < K; ++i) {
            const double got = weights[(size_t) r * K + i], want = ref_w[(size_t) r * K + i];
            if (!(std::fabs(got - want) <= 5e-4 + 1e-3 * std::fabs(want))) ok = false;
            sum += got;
        }
        if (std::fabs(sum - 1.0) > 5e-4) ok = false;   // the renormalised weights sum to 1 (sum s >> 1e-20 here)
        if (!ok) {
            if (bad < 2) {
                std::printf("  row %d: ids", r);
                for (int i = 0; i < K; ++i) std::printf(" %d/%d", ids[(size_t) r * K + i], ref_ids[(size_t) r * K + i]);
                std::printf("  w");
                for (int i = 0; i < K; ++i)
                    std::printf(" %.6f/%.6f", weights[(size_t) r * K + i], ref_w[(size_t) r * K + i]);
                std::printf("  sum %.6f\n", sum);
            }
            ++bad;
        }
    }
    std::printf("%s: %s - %d rows, n_embd %d%s\n", bad ? "FAIL" : "PASS", name, rows, n_embd,
                bad ? "" : ", ids exact, weights within tolerance, sums 1.0");
    return bad;
}

static std::vector<uint16_t> random_f16(std::mt19937& rng, size_t n, double scale) {
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<uint16_t> v(n);
    for (auto& e : v) e = k::f16_from_f32((float) (nd(rng) * scale));
    return v;
}

/// The launch-count + CUDA-event microbench at the target model's router geometry (256
/// experts x 4096 hidden, top-8): the one-block kernel (`mimo_router_route`, the T-window
/// status quo) vs the window arm (`mimo_router_route_window`) per row count T in
/// {1,2,3,4,5,8}.  Protocol: per row count, warmup both arms, then `reps` samples per arm of
/// `batch` consecutive calls timed between two CUDA events on one stream, arms INTERLEAVED
/// with the starting arm swapped every sample (order counterbalanced within and across row
/// counts); the reported number is the per-call mean of a sample.  Structural launch counts
/// (1 vs 2 kernels per call) are facts from the code, printed beside the times.  Event
/// resolution rules: no claim finer than 0.1 us, and medians/quartiles over samples only.
static int bench(int reps, int batch) {
    if (reps < 5 || batch < 1) {
        std::fprintf(stderr, "bench: reps >= 5 and batch >= 1\n");
        return 2;
    }
    int device = 0;
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("bench: device %d (%s), %d experts x %d hidden, k %d; %d samples/arm of %d calls\n",
                device, prop.name, NE, 4096, K, reps, batch);
    std::printf("bench: launches per call: one-block 1, window arm 2 (scores + pick)\n");
    std::mt19937 rng(4701);
    const int n_embd = 4096;
    const std::vector<uint16_t> w = random_f16(rng, (size_t) NE * n_embd, 1.0 / 64.0);
    const std::vector<float> bias(NE, 0.0f);
    uint16_t* dw = nullptr;
    uint16_t* dx = nullptr;
    float* db = nullptr;
    float* ds = nullptr;
    int32_t* di = nullptr;
    float* dw8 = nullptr;
    int32_t* mi = nullptr;
    float* mw = nullptr;
    cudaEvent_t ev0, ev1;
    cudaStream_t stream;
    CHECK(cudaMalloc(&dw, (size_t) NE * n_embd * 2));
    CHECK(cudaMalloc(&db, (size_t) NE * 4));
    CHECK(cudaMalloc(&ds, (size_t) 8 * NE * 4));
    CHECK(cudaMalloc(&di, (size_t) 8 * K * 4));
    CHECK(cudaMalloc(&dw8, (size_t) 8 * K * 4));
    CHECK(cudaMalloc(&mi, (size_t) 8 * K * 4));
    CHECK(cudaMalloc(&mw, (size_t) 8 * K * 4));
    CHECK(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(db, 0, (size_t) NE * 4));
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));
    CHECK(cudaStreamCreate(&stream));
    auto percentile = [](std::vector<double> v, double q) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        const size_t i = (size_t) (q * (double) (v.size() - 1) + 0.5);
        return v[std::min(i, v.size() - 1)];
    };
    int failures = 0;
    for (int rows : {1, 2, 3, 4, 5, 8}) {
        CHECK(cudaMalloc(&dx, (size_t) rows * n_embd * 2));
        const std::vector<uint16_t> x = random_f16(rng, (size_t) rows * n_embd, 1.0);
        CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
        std::vector<double> fused, split;
        for (int warm = 0; warm < 5; ++warm) {   // both arms compiled in and warm
            k::mimo_router_route(dw, dx, db, mi, mw, rows, n_embd, NE, K, stream);
            k::mimo_router_route_window(dw, dx, db, ds, di, dw8, rows, n_embd, NE, K, stream);
        }
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaGetLastError());
        for (int s = 0; s < reps; ++s) {
            for (int arm = 0; arm < 2; ++arm) {   // interleave; the starting arm swaps per sample
                const bool fused_first = ((s + rows) & 1) == 0;
                const bool run_fused = arm == 0 ? fused_first : !fused_first;
                CHECK(cudaEventRecord(ev0, stream));
                for (int c = 0; c < batch; ++c) {
                    if (run_fused)
                        k::mimo_router_route(dw, dx, db, mi, mw, rows, n_embd, NE, K, stream);
                    else
                        k::mimo_router_route_window(dw, dx, db, ds, di, dw8, rows, n_embd, NE, K, stream);
                }
                CHECK(cudaEventRecord(ev1, stream));
                CHECK(cudaEventSynchronize(ev1));
                float ms = 0;
                CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
                (run_fused ? fused : split).push_back((double) ms / batch);
            }
        }
        (void) cudaFree(dx);
        dx = nullptr;
        const double fm = percentile(fused, 0.5), fq1 = percentile(fused, 0.25), fq3 = percentile(fused, 0.75);
        const double sm = percentile(split, 0.5), sq1 = percentile(split, 0.25), sq3 = percentile(split, 0.75);
        std::printf("bench: T=%d one-block median %.1f us (q25 %.1f q75 %.1f) | window arm median %.1f us "
                    "(q25 %.1f q75 %.1f) | ratio %.2fx\n",
                    rows, fm * 1000.0, fq1 * 1000.0, fq3 * 1000.0, sm * 1000.0, sq1 * 1000.0, sq3 * 1000.0,
                    fm / std::max(sm, 1e-9));
        // the bench never gates correctness; a NaN time would still be reported loudly
        if (!(fm >= 0.0 && sm >= 0.0)) ++failures;
    }
    CHECK(cudaEventDestroy(ev0));
    CHECK(cudaEventDestroy(ev1));
    CHECK(cudaStreamDestroy(stream));
    (void) cudaFree(dw);
    (void) cudaFree(dx);
    (void) cudaFree(db);
    (void) cudaFree(ds);
    (void) cudaFree(di);
    (void) cudaFree(dw8);
    (void) cudaFree(mi);
    (void) cudaFree(mw);
    return failures ? 1 : 0;
}

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        const int reps = argc > 2 ? std::atoi(argv[2]) : 40;
        const int batch = argc > 3 ? std::atoi(argv[3]) : 32;
        return bench(reps, batch);
    }
    if (!k::mimo_router_supported(NE, K)) {
        std::fprintf(stderr, "mimo_router_supported(256, 8) is false\n");
        return 1;
    }
    int fails = 0;

    // random batches: two hidden widths, a large batch and an odd small one
    {
        std::mt19937 rng(4001);
        const int rows = 64, n_embd = 4096;
        const std::vector<uint16_t> w = random_f16(rng, (size_t) NE * n_embd, 1.0 / 64.0);
        const std::vector<uint16_t> x = random_f16(rng, (size_t) rows * n_embd, 1.0);
        const std::vector<float> bias(NE, 0.0f);
        fails += compare("random batch", w, x, bias, rows, n_embd) ? 1 : 0;
    }
    // the decode arm (one token): the reference agrees, and the picks are BIT-IDENTICAL to the
    // one-block kernel's on the same inputs (the scores kernel copies the per-warp accumulation
    // order; the selection kernel is the same warp-0 code) - a drift here would make a prompt
    // chunk and its decode tokens route differently
    {
        std::mt19937 rng(4101);
        const int n_embd = 4096;
        const std::vector<uint16_t> w = random_f16(rng, (size_t) NE * n_embd, 1.0 / 64.0);
        const std::vector<uint16_t> x = random_f16(rng, (size_t) n_embd, 1.0);
        std::vector<float> bias(NE, 0.0f);
        std::mt19937 brng(4102);
        std::normal_distribution<float> bd(0.0f, 0.02f);
        for (auto& b : bias) b = bd(brng);
        std::vector<int32_t> ids, mono_ids;
        std::vector<float> weights, mono_w;
        run_gpu_decode(w, x, bias, n_embd, ids, weights, &mono_ids, &mono_w);
        std::vector<int32_t> ref_ids;
        std::vector<float> ref_w;
        reference(w, x, bias, 1, n_embd, ref_ids, ref_w);
        bool ok = std::memcmp(ids.data(), ref_ids.data(), K * 4) == 0;
        double sum = 0.0;
        for (int i = 0; i < K; ++i) {
            const double got = weights[(size_t) i], want = ref_w[(size_t) i];
            if (!(std::fabs(got - want) <= 5e-4 + 1e-3 * std::fabs(want))) ok = false;
            sum += got;
        }
        if (std::fabs(sum - 1.0) > 5e-4) ok = false;
        std::printf("%s: decode arm vs reference - 1 row, n_embd %d, random bias%s\n", ok ? "PASS" : "FAIL",
                    n_embd, ok ? ", ids exact, weights within tolerance, sums 1.0" : "");
        fails += ok ? 0 : 1;
        ok = std::memcmp(ids.data(), mono_ids.data(), K * 4) == 0 &&
             std::memcmp(weights.data(), mono_w.data(), K * 4) == 0;
        std::printf("%s: decode arm vs one-block kernel - ids and weights bit-identical\n", ok ? "PASS" : "FAIL");
        if (!ok)
            for (int i = 0; i < K; ++i)
                std::printf("  rank %d: decode %d/%.7f mono %d/%.7f\n", i, ids[(size_t) i], weights[(size_t) i],
                            mono_ids[(size_t) i], mono_w[(size_t) i]);
        fails += ok ? 0 : 1;
    }
    {
        std::mt19937 rng(4002);
        const int rows = 7, n_embd = 512;
        const std::vector<uint16_t> w = random_f16(rng, (size_t) NE * n_embd, 1.0 / std::sqrt(512.0));
        const std::vector<uint16_t> x = random_f16(rng, (size_t) rows * n_embd, 1.0);
        std::vector<float> bias(NE, 0.0f);
        std::mt19937 brng(4003);
        std::normal_distribution<float> bd(0.0f, 0.02f);   // small random biases on top of the random batch
        for (auto& b : bias) b = bd(brng);
        fails += compare("random batch, random bias, n_embd 512", w, x, bias, rows, n_embd) ? 1 : 0;
    }

    // the pure tie rule: a zero input row makes every sigmoid exactly 0.5, every biased score exactly 0.5 -
    // the top-8 must be experts 0..7 in order, each weight exactly 0.125
    {
        std::mt19937 rng(4004);
        const int n_embd = 4096;
        const std::vector<uint16_t> w = random_f16(rng, (size_t) NE * n_embd, 1.0 / 64.0);
        const std::vector<uint16_t> x((size_t) n_embd, 0);
        const std::vector<float> bias(NE, 0.0f);
        std::vector<int32_t> ids;
        std::vector<float> weights;
        run_gpu(w, x, bias, 1, n_embd, ids, weights);
        bool ok = true;
        for (int i = 0; i < K; ++i) ok = ok && ids[i] == i && weights[i] == 0.125f;
        std::printf("%s: all-tie row - ids 0..7 ascending, weights exactly 0.125\n", ok ? "PASS" : "FAIL");
        fails += ok ? 0 : 1;
    }

    // exact score ties from duplicated router rows: {10, 200} share the strongest row, {11, 12, 13} the
    // second strongest; both members of a tie must be selected, the LOWER id at the better rank.  The strong
    // rows are built as a scaled copy of x, so their dots land at a chosen logit (7.0 and 6.3) - clearly above
    // every base row (~N(0,1)) and below sigmoid saturation, so the two groups tie within, not across
    {
        std::mt19937 rng(4005);
        const int n_embd = 4096;
        std::vector<uint16_t> w = random_f16(rng, (size_t) NE * n_embd, 1.0 / 64.0);
        const std::vector<uint16_t> x = random_f16(rng, (size_t) n_embd, 1.0);
        for (int i = 0; i < n_embd; ++i) {
            const float xi = k::f32_from_f16(x[(size_t) i]);
            const uint16_t a = k::f16_from_f32(xi * (float) (7.0 / n_embd));
            w[(size_t) 10 * n_embd + i] = a;
            w[(size_t) 200 * n_embd + i] = a;   // bitwise the same row: the same dot, the same score
            const uint16_t b = k::f16_from_f32(xi * (float) (6.3 / n_embd));
            w[(size_t) 11 * n_embd + i] = b;
            w[(size_t) 12 * n_embd + i] = b;
            w[(size_t) 13 * n_embd + i] = b;
        }
        const std::vector<float> bias(NE, 0.0f);
        std::vector<int32_t> ids;
        std::vector<float> weights;
        run_gpu(w, x, bias, 1, n_embd, ids, weights);
        const int r10 = row_of(ids, 0, 10), r200 = row_of(ids, 0, 200);
        const int r11 = row_of(ids, 0, 11), r12 = row_of(ids, 0, 12), r13 = row_of(ids, 0, 13);
        const bool ok = r10 >= 0 && r200 >= 0 && r10 < r200 && r11 >= 0 && r12 >= 0 && r13 >= 0 &&
                        (r11 < r12 && r12 < r13) && r200 < r11;
        std::printf("%s: duplicated rows - {10,200} and {11,12,13} selected, lower id first inside each tie\n",
                    ok ? "PASS" : "FAIL");
        if (!ok)
            std::printf("  ranks: 10@%d 200@%d 11@%d 12@%d 13@%d (ids %d %d %d %d %d %d %d %d)\n", r10, r200, r11, r12,
                        r13, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7]);
        fails += ok ? 0 : 1;
        fails += compare("duplicated rows vs reference", w, x, bias, 1, n_embd) ? 1 : 0;
    }

    // the bias flips the top-8 against the raw sigmoid order: +8 pushes a mid-score expert in at rank 0,
    // -8 drops the raw winner out entirely, and the pushed-in expert's WEIGHT is its small unbiased score
    {
        std::mt19937 rng(4006);
        const int n_embd = 4096;
        const std::vector<uint16_t> w = random_f16(rng, (size_t) NE * n_embd, 1.0 / 64.0);
        const std::vector<uint16_t> x = random_f16(rng, (size_t) n_embd, 1.0);
        const std::vector<float> zero(NE, 0.0f);
        std::vector<int32_t> raw_ids;
        std::vector<float> raw_w;
        reference(w, x, zero, 1, n_embd, raw_ids, raw_w);
        const int32_t raw_top = raw_ids[0];
        std::vector<float> bias(NE, 0.0f);
        bias[77] = 8.0f;
        bias[raw_top] = -8.0f;
        std::vector<int32_t> ids;
        std::vector<float> weights;
        run_gpu(w, x, bias, 1, n_embd, ids, weights);
        const int r77 = row_of(ids, 0, 77), rtop = row_of(ids, 0, raw_top), rraw77 = row_of(raw_ids, 0, 77);
        bool ok = r77 == 0 && rtop < 0;             // in at rank 0, the raw winner gone
        ok = ok && (rraw77 < 0);                    // ...and 77 was NOT in the raw top-8
        ok = ok && weights[0] < 0.5f;               // its weight is the UNBIASED score (sigmoid stays mid)
        fails += compare("bias-flip vs reference", w, x, bias, 1, n_embd) ? 1 : 0;
        std::printf("%s: bias flip - expert 77 out of the raw top-8 (rank %d) to rank %d, raw winner %d dropped; "
                    "weight %.4f is the unbiased score\n",
                    ok ? "PASS" : "FAIL", rraw77, r77, raw_top, weights[0]);
        fails += ok ? 0 : 1;
    }

    // a NaN router row: the dot is NaN, the score clamps to 0, the expert is never selected; the rest match
    {
        std::mt19937 rng(4007);
        const int n_embd = 4096;
        std::vector<uint16_t> w = random_f16(rng, (size_t) NE * n_embd, 1.0 / 64.0);
        for (int i = 0; i < n_embd; ++i) w[(size_t) 5 * n_embd + i] = 0x7e00;   // half NaN
        const std::vector<uint16_t> x = random_f16(rng, (size_t) n_embd, 1.0);
        const std::vector<float> bias(NE, 0.0f);
        std::vector<int32_t> ids;
        std::vector<float> weights;
        run_gpu(w, x, bias, 1, n_embd, ids, weights);
        const bool ok = row_of(ids, 0, 5) < 0;
        std::printf("%s: NaN router row - expert 5 clamped to score 0, not selected\n", ok ? "PASS" : "FAIL");
        fails += ok ? 0 : 1;
        fails += compare("NaN row vs reference", w, x, bias, 1, n_embd) ? 1 : 0;
    }

    // saturation: 60x weights put the top-8 logits near 110, where FP32 sigmoid is exactly 1 (and the negative
    // tail's exactly 0) - the selected weights must come out exactly 0.125 each
    {
        std::mt19937 rng(4008);
        const int n_embd = 4096;
        const std::vector<uint16_t> w = random_f16(rng, (size_t) NE * n_embd, 60.0 / 64.0);
        const std::vector<uint16_t> x = random_f16(rng, (size_t) n_embd, 1.0);
        const std::vector<float> bias(NE, 0.0f);
        std::vector<int32_t> ids;
        std::vector<float> weights;
        run_gpu(w, x, bias, 1, n_embd, ids, weights);
        bool ok = true;
        for (int i = 0; i < K; ++i) ok = ok && weights[i] == 0.125f;
        std::printf("%s: saturated row - sigmoid 0/1, weights exactly 0.125\n", ok ? "PASS" : "FAIL");
        fails += ok ? 0 : 1;
    }

    // ---- the WINDOW arm (mimo_router_route_window): the same math, batched over the rows by
    // the decode arm's two kernels.  The gate is BYTE EXACTNESS against the one-block kernel
    // on the same device buffers (per T row count, both hidden widths, three expert/k shapes),
    // with tail guards on every buffer and the scores-scratch full-write check (see
    // run_gpu_window).  The special-value contracts above are re-demanded through the window
    // arm: the same tie/bias/NaN inputs must route the same rows bit-for-bit.
    {
        std::mt19937 rng(4201);
        const int n_embd = 4096;
        const std::vector<uint16_t> w = random_f16(rng, (size_t) NE * n_embd, 1.0 / 64.0);
        std::vector<float> bias(NE, 0.0f);
        std::mt19937 brng(4202);
        std::normal_distribution<float> bd(0.0f, 0.02f);
        for (auto& b : bias) b = bd(brng);
        for (int rows : {1, 2, 3, 4, 5, 8}) {   // the verify-window envelope's edges and the real widths
            const std::vector<uint16_t> x = random_f16(rng, (size_t) rows * n_embd, 1.0);
            std::vector<int32_t> ids;
            std::vector<float> weights;
            fails += run_gpu_window("random windows", w, x, bias, rows, n_embd, NE, K, ids, weights) ? 1 : 0;
            if (rows == 4) {   // the batch also agrees with the double reference (both arms, one shot)
                std::vector<int32_t> ref_ids;
                std::vector<float> ref_w;
                reference(w, x, bias, rows, n_embd, ref_ids, ref_w);
                bool ok = true;
                for (int r = 0; r < rows; ++r) {
                    ok = ok && std::memcmp(&ids[(size_t) r * K], &ref_ids[(size_t) r * K], K * 4) == 0;
                    double sum = 0;
                    for (int i = 0; i < K; ++i) {
                        const double got = weights[(size_t) r * K + i], want = ref_w[(size_t) r * K + i];
                        ok = ok && std::fabs(got - want) <= 5e-4 + 1e-3 * std::fabs(want);
                        sum += got;
                    }
                    ok = ok && std::fabs(sum - 1.0) <= 5e-4;
                }
                std::printf("%s: random T4 window vs double reference - ids exact, weights within tolerance, "
                            "sums 1.0\n", ok ? "PASS" : "FAIL");
                fails += ok ? 0 : 1;
            }
        }
        // ties and special values INSIDE one multi-row batch, both contracts of the single-row
        // cases: row 1 is the pure-tie row (zero input), a NaN router row clamps to score 0, a
        // NaN bias is never selected, and a +8 bias wins rank 0 with its UNBIASED weight
        {
            const int rows = 5;
            std::vector<uint16_t> x = random_f16(rng, (size_t) rows * n_embd, 1.0);
            std::vector<uint16_t> wt = w;
            for (int i = 0; i < n_embd; ++i) x[(size_t) 1 * n_embd + i] = 0;   // row 1 ties everything
            for (int i = 0; i < n_embd; ++i) wt[(size_t) 5 * n_embd + i] = 0x7e00;   // NaN expert 5
            std::vector<float> bt(NE, 0.0f);
            bt[77] = 8.0f;                                        // the bias-flip contract
            bt[200] = std::numeric_limits<float>::quiet_NaN();    // a NaN bias never poisons the argmax
            std::vector<int32_t> ids;
            std::vector<float> weights;
            fails += run_gpu_window("tie/bias/NaN windows", wt, x, bt, rows, n_embd, NE, K, ids, weights) ? 1 : 0;
            bool ok = ids[(size_t) 1 * K] == 77;   // row 1: everything ties at 0.5, the +8 bias wins rank 0
            for (int r = 0; r < rows; ++r)
                ok = ok && row_of(ids, r, 5) < 0 && row_of(ids, r, 200) < 0;   // NaN row/bias never selected
            std::printf("%s: window arm's tie/NaN rows - pure-tie row biased winner at rank 0, NaN expert %d, "
                        "NaN-biased expert %d never picked\n",
                        ok ? "PASS" : "FAIL", 5, 200);
            fails += ok ? 0 : 1;
        }
        // exact score ties from duplicated router rows on the batch's first row (the single-row
        // case's construction): {10,200} and {11,12,13} tie within, lower id at the better rank
        {
            const int rows = 4;
            std::vector<uint16_t> x = random_f16(rng, (size_t) rows * n_embd, 1.0);
            std::vector<uint16_t> wt = w;
            for (int i = 0; i < n_embd; ++i) {
                const float xi = k::f32_from_f16(x[(size_t) i]);
                const uint16_t a = k::f16_from_f32(xi * (float) (7.0 / n_embd));
                wt[(size_t) 10 * n_embd + i] = a;
                wt[(size_t) 200 * n_embd + i] = a;
                const uint16_t b = k::f16_from_f32(xi * (float) (6.3 / n_embd));
                wt[(size_t) 11 * n_embd + i] = b;
                wt[(size_t) 12 * n_embd + i] = b;
                wt[(size_t) 13 * n_embd + i] = b;
            }
            std::vector<float> bt(NE, 0.0f);   // zero bias: the duplicated-row ties must stay EXACT
            // (near-equal biased scores are already covered by the random-window batches above)
            std::vector<int32_t> ids;
            std::vector<float> weights;
            fails += run_gpu_window("duplicated-row windows", wt, x, bt, rows, n_embd, NE, K, ids, weights) ? 1 : 0;
            const int r10 = row_of(ids, 0, 10), r200 = row_of(ids, 0, 200);
            const int r11 = row_of(ids, 0, 11), r12 = row_of(ids, 0, 12), r13 = row_of(ids, 0, 13);
            const bool ok = r10 >= 0 && r200 >= 0 && r10 < r200 && r11 >= 0 && r12 >= 0 && r13 >= 0 &&
                            r11 < r12 && r12 < r13 && r200 < r11;
            std::printf("%s: window arm's duplicated rows - {10,200} and {11,12,13} selected, lower id first "
                        "inside each tie\n", ok ? "PASS" : "FAIL");
            fails += ok ? 0 : 1;
        }
    }
    // the window arm at the engine fixture's other shapes: the mini engine's 4 experts/k 2,
    // a 33-expert/k 5 count (lanes past n_expert in the values[8] tail), k 32 (the bound) and
    // the second hidden width - all bitwise against the one-block kernel
    for (auto shape : std::vector<std::array<int, 4>>{{4, 2, 64, 8}, {33, 5, 512, 3}, {256, 32, 1024, 2}}) {
        const int ne = shape[0], k = shape[1], n_embd = shape[2], rows = shape[3];
        if (!k::mimo_router_supported(ne, k)) {
            std::printf("FAIL: shape %d/%d unexpectedly unsupported\n", ne, k);
            ++fails;
            continue;
        }
        std::mt19937 rng(4300 + ne * 7 + k);
        const std::vector<uint16_t> w = random_f16(rng, (size_t) ne * n_embd, 1.0 / std::sqrt((double) n_embd));
        const std::vector<uint16_t> x = random_f16(rng, (size_t) rows * n_embd, 1.0);
        std::vector<float> bias((size_t) ne, 0.0f);
        std::mt19937 brng(4400 + ne);
        std::normal_distribution<float> bd(0.0f, 0.05f);
        for (auto& b : bias) b = bd(brng);
        std::vector<int32_t> ids;
        std::vector<float> weights;
        fails += run_gpu_window("shape sweep", w, x, bias, rows, n_embd, ne, k, ids, weights) ? 1 : 0;
    }
    // the window arm's validation: every broken argument is refused BEFORE a launch, and a
    // refused call leaves every buffer untouched (the probe byte survives)
    {
        const int rows = 4, n_embd = 512, ne = 64, k = 4;
        std::mt19937 rng(4501);
        const std::vector<uint16_t> w = random_f16(rng, (size_t) ne * n_embd, 1.0 / 32.0);
        const std::vector<uint16_t> x = random_f16(rng, (size_t) rows * n_embd, 1.0);
        const std::vector<float> bias(ne, 0.0f);
        uint16_t* dw = nullptr;
        uint16_t* dx = nullptr;
        float* db = nullptr;
        float* ds = nullptr;
        int32_t* di = nullptr;
        float* dw8 = nullptr;
        CHECK(cudaMalloc(&dw, (size_t) ne * n_embd * 2));
        CHECK(cudaMalloc(&dx, (size_t) rows * n_embd * 2));
        CHECK(cudaMalloc(&db, (size_t) ne * 4));
        CHECK(cudaMalloc(&ds, (size_t) rows * ne * 4));
        CHECK(cudaMalloc(&di, (size_t) rows * k * 4));
        CHECK(cudaMalloc(&dw8, (size_t) rows * k * 4));
        CHECK(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(db, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice));
        auto call = [&](int r, int emb, int experts, int kk, void* rr, void* xx, void* bb, void* ss, void* ii,
                        void* ww) {
            k::mimo_router_route_window(rr, xx, bb, static_cast<float*>(ss), static_cast<int32_t*>(ii),
                                        static_cast<float*>(ww), r, emb, experts, kk, nullptr);
        };
        bool ok = true;
        ok = ok && refuses("rows 0", [&] { call(0, n_embd, ne, k, dw, dx, db, ds, di, dw8); }, ds);
        ok = ok && refuses("rows 9 (past the window envelope)", [&] { call(9, n_embd, ne, k, dw, dx, db, ds, di, dw8); }, ds);
        ok = ok && refuses("n_embd not a multiple of 8", [&] { call(rows, n_embd + 3, ne, k, dw, dx, db, ds, di, dw8); }, ds);
        ok = ok && refuses("k 33 (past 32)", [&] { call(rows, n_embd, ne, 33, dw, dx, db, ds, di, dw8); }, ds);
        ok = ok && refuses("k beyond n_expert", [&] { call(rows, n_embd, 4, 5, dw, dx, db, ds, di, dw8); }, ds);
        ok = ok && refuses("n_expert 257 (past 256)", [&] { call(rows, n_embd, 257, k, dw, dx, db, ds, di, dw8); }, ds);
        ok = ok && refuses("misaligned router", [&] { call(rows, n_embd, ne, k, (char*) dw + 8, dx, db, ds, di, dw8); }, ds);
        ok = ok && refuses("scores aliasing the input rows", [&] { call(rows, n_embd, ne, k, dw, dx, db, dx, di, dw8); }, ds);
        ok = ok && refuses("scores aliasing the router", [&] { call(rows, n_embd, ne, k, dw, dx, db, dw, di, dw8); }, ds);
        ok = ok && refuses("scores aliasing the bias", [&] { call(rows, n_embd, ne, k, dw, dx, db, db, di, dw8); }, ds);
        ok = ok && refuses("scores aliasing the ids", [&] { call(rows, n_embd, ne, k, dw, dx, db, di, di, dw8); }, ds);
        ok = ok && refuses("scores aliasing the weights", [&] { call(rows, n_embd, ne, k, dw, dx, db, dw8, di, dw8); }, ds);
        ok = ok && refuses("ids aliasing the weights", [&] { call(rows, n_embd, ne, k, dw, dx, db, ds, dw8, dw8); }, ds);
        fails += ok ? 0 : 1;
        (void) cudaFree(dw);
        (void) cudaFree(dx);
        (void) cudaFree(db);
        (void) cudaFree(ds);
        (void) cudaFree(di);
        (void) cudaFree(dw8);
    }

    std::printf("FAILURES: %d\n", fails);
    return fails ? 1 : 0;
}
