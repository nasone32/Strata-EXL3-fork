// bench/micro/mimo_qkv_bench.cpp - the P1/P2 dense-QKV decisions on REAL MiMo shapes, same input
// bytes for every variant (2026-11-05 design: notes/MIMO-DENSE-QKV-PREFILL-2026-11-05.md).
//
//     build-hip-exl3/mimo_qkv_bench [--reps R] [--warmup W] [--tsv]
//
// WHAT IT TIMES.  The engine's own loop shapes, nothing idealized: per T (a prompt chunk's rows) the
// exact launch sequence mimo_engine.cu would issue for one layer's projections -
//   old    : per <= 8-row tile, q|k|v: 3x (native_quantize_q8_1 + native_q8_0_mmvq)   (today's path)
//   p1     : per <= 8-row tile, 1x native_quantize_q8_1 + 1x native_mmvq_bunch        (P1, cols 8)
//   p2-16/.: wide tiles (native_quantize_q8_1_wide + 3x native_q8_0_mmvq_wide) + the <= 8 tail as old
//   p12-16/: wide tiles (1x wide quantize + 1x native_mmvq_bunch_wide) + the tail as p1
//   o-*    : the o_proj (n_in 8192 -> n_out 4096) single-matrix forms: old (native_q8_0_f32 per tile)
//            vs p2 (native_q8_0_f32_wide per wide tile + tail)
// Same weights, same activation bytes for every variant.  Reps ALTERNATE variants (A B C A B C...),
// warm and odd by default; the median is reported.  Launch counts are printed from the same loop
// arithmetic the engine runs.  Actual compiled VGPR/shared usage comes from
// native_q8_0_wide_resources (cudaFuncGetAttributes on the loaded code object), not a prediction.
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

struct Buf {   // one shape's weights (random Q8_0 blocks, finite fp16 scales) + activation rows
    void* w[3] = {nullptr, nullptr, nullptr};
    float* x = nullptr;          // [T][n_in]
    float* y[3] = {nullptr, nullptr, nullptr};
    void* q8 = nullptr;          // the <= 8-column image
    void* qw = nullptr;          // the wide image (32 columns)
    int n_in = 0, n_out[3] = {0, 0, 0}, m = 0, T = 0;
    long long launches = 0;      // counted per variant run

    void alloc(int n_in_, const int* n_out_, int m_, int T_) {
        n_in = n_in_; m = m_; T = T_;
        for (int i = 0; i < 3; ++i) n_out[i] = n_out_[i];
        if (!ck(cudaMalloc(&x, (size_t) T_ * n_in_ * 4), "x")) std::exit(1);
        for (int i = 0; i < m; ++i) {
            if (!ck(cudaMalloc(&y[i], (size_t) T_ * n_out_[i] * 4), "y")) std::exit(1);
            const size_t wb = K::native_mmvq_weight_bytes(8, n_in_, n_out_[i]);
            std::mt19937 rng(3131u + (unsigned) n_out_[i]);
            std::vector<uint8_t> h(wb);
            for (auto& b : h) b = (uint8_t) (rng() & 0xff);
            for (size_t k = 0; k < (size_t) n_out_[i] * (n_in_ / 32); ++k) {
                const uint16_t s = sane_half(rng);
                std::memcpy(&h[k * 34], &s, 2);
            }
            if (!ck(cudaMalloc(&w[i], wb), "w") || !ck(cudaMemcpy(w[i], h.data(), wb, cudaMemcpyHostToDevice), "w up"))
                std::exit(1);
        }
        if (!ck(cudaMalloc(&q8, K::native_q8_1_bytes(n_in_, 8)), "q8")) std::exit(1);
        if (!ck(cudaMalloc(&qw, K::native_q8_1_wide_bytes(n_in_, 32)), "qw")) std::exit(1);
        std::mt19937 rx(4242u + (unsigned) n_in_ + (unsigned) T_);
        std::vector<float> hx((size_t) T_ * n_in_);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : hx) v = nd(rx);
        if (!ck(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x up")) std::exit(1);
    }
};

// the engine's loop shapes, verbatim -----------------------------------------------------------

void run_old(Buf& b, cudaStream_t s) {   // today: per projection per <= 8 tile, quantize + mmvq
    for (int r = 0; r < b.T; r += 8) {
        const int nc = std::min(8, b.T - r);
        for (int i = 0; i < b.m; ++i) {
            K::native_q8_0_f32(b.w[i], b.x + (int64_t) r * b.n_in, b.q8, b.y[i] + (int64_t) r * b.n_out[i],
                               b.n_in, b.n_out[i], nc, s);
            b.launches += 2;
        }
    }
}

void run_p1(Buf& b, cudaStream_t s) {   // P1: shared quantize + bunch per <= 8 tile
    for (int r = 0; r < b.T; r += 8) {
        const int nc = std::min(8, b.T - r);
        K::native_quantize_q8_1(b.x + (int64_t) r * b.n_in, b.q8, b.n_in, nc, s);
        if (nc >= 2 && K::native_mmvq_bunch(8, b.w[0], b.w[1], b.m > 2 ? b.w[2] : nullptr, b.q8,
                                             b.y[0] + (int64_t) r * b.n_out[0], b.y[1] + (int64_t) r * b.n_out[1],
                                             b.m > 2 ? b.y[2] + (int64_t) r * b.n_out[2] : nullptr, b.n_in,
                                             b.n_out[0], b.n_out[1], b.m > 2 ? b.n_out[2] : 0, nc, s))
            b.launches += 2;
        else {
            for (int i = 0; i < b.m; ++i)
                K::native_q8_0_mmvq(b.w[i], b.q8, b.y[i] + (int64_t) r * b.n_out[i], b.n_in, b.n_out[i], nc, s);
            b.launches += 1 + b.m;
        }
    }
}

void run_wide(Buf& b, cudaStream_t s, int cols, bool with_p1) {   // P2 (and P1+P2) wide tiles + tail
    int r = 0;
    for (; r + cols <= b.T;) {
        K::native_quantize_q8_1_wide(b.x + (int64_t) r * b.n_in, b.qw, b.n_in, cols, s);
        bool done = false;
        if (b.m == 3 && with_p1 &&
            K::native_mmvq_bunch_wide(8, b.w[0], b.w[1], b.w[2], b.qw, b.y[0] + (int64_t) r * b.n_out[0],
                                      b.y[1] + (int64_t) r * b.n_out[1], b.y[2] + (int64_t) r * b.n_out[2],
                                      b.n_in, b.n_out[0], b.n_out[1], b.n_out[2], cols, s))
            done = true;
        else {
            bool all = true;
            for (int i = 0; i < b.m; ++i)
                all = K::native_q8_0_mmvq_wide(b.w[i], b.qw, b.y[i] + (int64_t) r * b.n_out[i], b.n_in,
                                               b.n_out[i], cols, s) &&
                      all;
            done = all;
        }
        if (!done) break;   // refused: the tail loop below recomputes this range (as the engine does)
        b.launches += with_p1 && b.m == 3 ? 2 : 1 + b.m;
        r += cols;
    }
    for (; r < b.T; r += 8) {
        const int nc = std::min(8, b.T - r);
        if (b.m == 3 && with_p1) {
            K::native_quantize_q8_1(b.x + (int64_t) r * b.n_in, b.q8, b.n_in, nc, s);
            if (nc >= 2 && K::native_mmvq_bunch(8, b.w[0], b.w[1], b.w[2], b.q8,
                                                 b.y[0] + (int64_t) r * b.n_out[0],
                                                 b.y[1] + (int64_t) r * b.n_out[1],
                                                 b.y[2] + (int64_t) r * b.n_out[2], b.n_in, b.n_out[0],
                                                 b.n_out[1], b.n_out[2], nc, s)) {
                b.launches += 2;
                continue;
            }
            for (int i = 0; i < b.m; ++i)
                K::native_q8_0_mmvq(b.w[i], b.q8, b.y[i] + (int64_t) r * b.n_out[i], b.n_in, b.n_out[i], nc, s);
            b.launches += 1 + b.m;
        } else {
            for (int i = 0; i < b.m; ++i) {
                K::native_q8_0_f32(b.w[i], b.x + (int64_t) r * b.n_in, b.q8, b.y[i] + (int64_t) r * b.n_out[i],
                                   b.n_in, b.n_out[i], nc, s);
                b.launches += 2;
            }
        }
    }
}

float time_variant(Buf& b, cudaStream_t s, const char* which) {
    cudaEvent_t a, z;
    (void) cudaEventCreate(&a);
    (void) cudaEventCreate(&z);
    b.launches = 0;
    (void) cudaEventRecord(a, s);
    if (std::strcmp(which, "old") == 0) run_old(b, s);
    else if (std::strcmp(which, "p1") == 0) run_p1(b, s);
    else if (std::strcmp(which, "p2-16") == 0) run_wide(b, s, 16, false);
    else if (std::strcmp(which, "p2-32") == 0) run_wide(b, s, 32, false);
    else if (std::strcmp(which, "p12-16") == 0) run_wide(b, s, 16, true);
    else run_wide(b, s, 32, true);
    (void) cudaEventRecord(z, s);
    if (!ck(cudaEventSynchronize(z), "sync")) std::exit(1);
    float ms = 0;
    (void) cudaEventElapsedTime(&ms, a, z);
    (void) cudaEventDestroy(a);
    (void) cudaEventDestroy(z);
    return ms;
}

}  // namespace

int main(int argc, char** argv) {
    int reps = 5, warmup = 3;
    bool tsv = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--tsv") == 0) tsv = true;
    }
    if (reps % 2 == 0) ++reps;   // odd: a true median
    cudaStream_t s;
    if (!ck(cudaStreamCreate(&s), "stream")) return 1;

    std::printf("mimo_qkv_bench: gfx1100-class, %d reps (alternating, median), %d warmup\n", reps, warmup);
    for (int nc : {8, 16, 32}) {
        int vg = 0, sm = 0, mt = 0;
        K::native_q8_0_wide_resources(nc, &vg, &sm, &mt);
        std::printf("resources nc=%2d: vgprs=%d shared=%d B/block maxThreads=%d\n", nc, vg, sm, mt);
    }

    const struct { const char* what; int n_in; int n_out[3]; int m; } shapes[] = {
        {"qkv-full", 4096, {12288, 768, 768}, 3},
        {"qkv-swa", 4096, {12288, 1536, 1536}, 3},
        {"o", 8192, {4096, 0, 0}, 1},
    };
    const int Ts[] = {1, 4, 8, 16, 32, 512};
    const char* qkv_variants[] = {"old", "p1", "p2-16", "p2-32", "p12-16", "p12-32"};
    const char* o_variants[] = {"old", "p2-16", "p2-32"};

    std::printf(tsv ? "shape\tT\tvariant\tmedian_ms\tlaunches\n"
                    : "%-9s %4s %7s %10s %8s\n", "shape", "T", "variant", "median_ms", "launches");
    for (const auto& sh : shapes) {
        const bool is_qkv = sh.m == 3;
        const char*const* vars = is_qkv ? qkv_variants : o_variants;
        const int nv = is_qkv ? 6 : 3;
        for (int T : Ts) {
            if (!is_qkv && T > 8 && T < 16) continue;
            Buf b;
            b.alloc(sh.n_in, sh.n_out, sh.m, T);
            std::vector<std::vector<float>> ms(nv, std::vector<float>(reps));
            std::vector<long long> counts(nv, 0);
            for (int w = 0; w < warmup; ++w)
                for (int v = 0; v < nv; ++v) (void) time_variant(b, s, vars[v]);
            for (int rep = 0; rep < reps; ++rep)   // alternate within every rep: drift cancels
                for (int v = 0; v < nv; ++v) {
                    ms[v][rep] = time_variant(b, s, vars[v]);
                    if (rep == reps / 2) counts[v] = b.launches;
                }
            for (int v = 0; v < nv; ++v) {
                std::sort(ms[v].begin(), ms[v].end());
                std::printf(tsv ? "%s\t%d\t%s\t%.4f\t%lld\n" : "%-9s %4d %7s %10.4f %8lld\n", sh.what, T,
                            vars[v], ms[v][reps / 2], counts[v]);
            }
            for (int i = 0; i < sh.m; ++i) {
                cudaFree(b.w[i]);
                cudaFree(b.y[i]);
            }
            cudaFree(b.x);
            cudaFree(b.q8);
            cudaFree(b.qw);
        }
    }
    (void) cudaStreamDestroy(s);
    return 0;
}
