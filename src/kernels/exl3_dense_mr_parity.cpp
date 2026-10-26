// src/kernels/exl3_dense_mr_parity.cpp - the dense multirow trellis GEMV (A9) against references.
//
// Three checks per fixture:
//   1. every row of an m-row call against the CPU reconstruct reference (rel error),
//   2. row r of an m-row call BIT-IDENTICAL to a separate m == 1 call (the design invariant the
//      window's consistency rests on: same values whatever the row count of the step),
//   3. the m == 1 call against the old exl3_gemv_f32 pipeline (rel error; not bit-identical - the old
//      path rounds the output through fp16 and splits k differently).
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "strata/kernels/exl3.hpp"
#include "strata/kernels/cpu/exl3.hpp"

namespace {

uint16_t to_h(float v) {
    unsigned f; std::memcpy(&f, &v, 4);
    unsigned sign = (f >> 16) & 0x8000u, m = f & 0x7FFFFFu; int e = (int)((f >> 23) & 0xFF) - 127 + 15;
    if (e <= 0) return (uint16_t)sign;
    if (e >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | (e << 10) | (m >> 13));
}
float h2f(uint16_t h) {
    unsigned sign = (unsigned)(h >> 15) << 31, exp = (h >> 10) & 0x1F, man = h & 0x3FF, f;
    if (exp == 0) { if (!man) f = sign; else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; --exp; } man &= 0x3FF; f = sign | (exp << 23) | (man << 13); } }
    else if (exp == 0x1F) f = sign | 0x7F800000u | (man << 13);
    else f = sign | ((exp + 127 - 15) << 23) | (man << 13);
    float o; std::memcpy(&o, &f, 4); return o;
}

int run(int ki, int nj, int bits, int cb, int rows, uint32_t seed) {
    const int k = ki * 16, n = nj * 16, words = 256 * bits / 16;
    std::mt19937 rng(seed);
    std::vector<uint16_t> trellis((size_t)ki * nj * words);
    for (auto& x : trellis) x = (uint16_t)(rng() & 0xFFFF);
    std::vector<uint16_t> suh(k), svh(n);
    std::vector<float> x((size_t)rows * k);
    std::uniform_real_distribution<float> sc(-1.0f, 1.0f);
    for (auto& v : suh) v = to_h(sc(rng) * 0.5f);
    for (auto& v : svh) v = to_h(sc(rng) * 0.5f);
    for (auto& v : x) v = h2f(to_h(sc(rng) * 0.3f));   // fp16-valued, like the engine's activations

    // reference: x @ reconstruct_weight, per row
    std::vector<uint16_t> w((size_t)k * n);
    strata::kernels::cpu::exl3_reconstruct_weight(trellis.data(), ki, nj, bits,
                                                 (strata::kernels::cpu::Exl3Codebook)cb,
                                                 suh.data(), svh.data(), w.data());

    uint16_t *d_suh = nullptr, *d_svh = nullptr, *d_tr = nullptr;
    float *d_x = nullptr, *d_ym = nullptr, *d_yold = nullptr;
    bool ok = true;
    auto chk = [&](cudaError_t e, const char* what) { if (e != cudaSuccess) { std::printf("CUDA %s: %s\n", what, cudaGetErrorString(e)); ok = false; } };
    chk(cudaMalloc(&d_x, x.size() * 4), "x");
    chk(cudaMalloc(&d_suh, (size_t)k * 2), "suh");
    chk(cudaMalloc(&d_svh, (size_t)n * 2), "svh");
    chk(cudaMalloc(&d_tr, trellis.size() * 2), "tr");
    chk(cudaMalloc(&d_ym, (size_t)rows * n * 4), "ym");
    chk(cudaMalloc(&d_yold, (size_t)n * 4), "yold");
    chk(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "cx");
    chk(cudaMemcpy(d_suh, suh.data(), (size_t)k * 2, cudaMemcpyHostToDevice), "csuh");
    chk(cudaMemcpy(d_svh, svh.data(), (size_t)n * 2, cudaMemcpyHostToDevice), "csvh");
    chk(cudaMemcpy(d_tr, trellis.data(), trellis.size() * 2, cudaMemcpyHostToDevice), "ctr");

    // one m-row call, plus the old m == 1 pipeline on row 0
    strata::kernels::exl3_dense_gemv_multi(0, d_x, k, d_tr, d_suh, d_svh, ki, nj, bits, cb, rows, d_ym, n, nullptr);
    chk(cudaGetLastError(), "launch");
    strata::kernels::exl3_gemv_f32(d_x, d_suh, d_svh, d_tr, ki, nj, bits, cb, d_yold, nullptr);
    chk(cudaGetLastError(), "launch-old");

    std::vector<float> ym((size_t)rows * n), yold(n);
    chk(cudaMemcpy(ym.data(), d_ym, ym.size() * 4, cudaMemcpyDeviceToHost), "cym");
    chk(cudaMemcpy(yold.data(), d_yold, (size_t)n * 4, cudaMemcpyDeviceToHost), "cyold");
    (void)cudaFree(d_x); (void)cudaFree(d_suh); (void)cudaFree(d_svh); (void)cudaFree(d_tr);
    (void)cudaFree(d_ym); (void)cudaFree(d_yold);
    if (!ok) return 1;

    // 1. every row vs the CPU reference
    double worst = 0;
    for (int r = 0; r < rows; ++r) {
        double num = 0, den = 0;
        for (int j = 0; j < n; ++j) {
            double acc = 0;
            for (int i = 0; i < k; ++i) acc += (double)x[(size_t)r * k + i] * h2f(w[(size_t)i * n + j]);
            const double d = (double)ym[(size_t)r * n + j] - acc;
            num += d * d; den += acc * acc;
        }
        worst = std::max(worst, std::sqrt(num) / (std::sqrt(den) + 1e-30));
    }
    // 3. m == 1 vs the old pipeline
    double num = 0, den = 0;
    for (int j = 0; j < n; ++j) {
        double acc = 0;
        for (int i = 0; i < k; ++i) acc += (double)x[i] * h2f(w[(size_t)i * n + j]);
        const double d = (double)yold[j] - acc;
        num += d * d; den += acc * acc;
    }
    const double rel_old = std::sqrt(num) / (std::sqrt(den) + 1e-30);
    std::printf("dense_mr cb=%d %dx%d K=%d rows=%d: rel_ref=%.3e rel_old_path=%.3e\n", cb, k, n, bits, rows,
                worst, rel_old);
    return (worst < 1e-2 && rel_old < 1e-2) ? 0 : 1;
}

// Row r of an m-row call vs a separate m == 1 call: BIT-identical fp32 outputs.
int run_bits(int ki, int nj, int bits, int cb, int rows, uint32_t seed) {
    const int k = ki * 16, n = nj * 16, words = 256 * bits / 16;
    std::mt19937 rng(seed);
    std::vector<uint16_t> trellis((size_t)ki * nj * words);
    for (auto& x : trellis) x = (uint16_t)(rng() & 0xFFFF);
    std::vector<uint16_t> suh(k), svh(n);
    std::vector<float> x((size_t)rows * k);
    std::uniform_real_distribution<float> sc(-1.0f, 1.0f);
    for (auto& v : suh) v = to_h(sc(rng) * 0.5f);
    for (auto& v : svh) v = to_h(sc(rng) * 0.5f);
    for (auto& v : x) v = sc(rng) * 0.3f;

    uint16_t *d_suh = nullptr, *d_svh = nullptr, *d_tr = nullptr;
    float *d_x = nullptr, *d_y1 = nullptr, *d_ym = nullptr;
    bool ok = true;
    auto chk = [&](cudaError_t e, const char* what) { if (e != cudaSuccess) { std::printf("CUDA %s: %s\n", what, cudaGetErrorString(e)); ok = false; } };
    chk(cudaMalloc(&d_x, x.size() * 4), "x");
    chk(cudaMalloc(&d_suh, (size_t)k * 2), "suh");
    chk(cudaMalloc(&d_svh, (size_t)n * 2), "svh");
    chk(cudaMalloc(&d_tr, trellis.size() * 2), "tr");
    chk(cudaMalloc(&d_ym, (size_t)rows * n * 4), "ym");
    chk(cudaMalloc(&d_y1, (size_t)n * 4), "y1");
    chk(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "cx");
    chk(cudaMemcpy(d_suh, suh.data(), (size_t)k * 2, cudaMemcpyHostToDevice), "csuh");
    chk(cudaMemcpy(d_svh, svh.data(), (size_t)n * 2, cudaMemcpyHostToDevice), "csvh");
    chk(cudaMemcpy(d_tr, trellis.data(), trellis.size() * 2, cudaMemcpyHostToDevice), "ctr");

    strata::kernels::exl3_dense_gemv_multi(0, d_x, k, d_tr, d_suh, d_svh, ki, nj, bits, cb, rows, d_ym, n, nullptr);
    chk(cudaGetLastError(), "launch-m");
    int bad = 0;
    std::vector<float> ym(n), y1(n);
    for (int r = 0; r < rows && ok; ++r) {
        strata::kernels::exl3_dense_gemv_multi(0, d_x + (size_t)r * k, k, d_tr, d_suh, d_svh, ki, nj, bits, cb,
                                               1, d_y1, n, nullptr);
        chk(cudaGetLastError(), "launch-1");
        chk(cudaMemcpy(ym.data(), (char*)d_ym + (size_t)r * n * 4, (size_t)n * 4, cudaMemcpyDeviceToHost), "cym");
        chk(cudaMemcpy(y1.data(), d_y1, (size_t)n * 4, cudaMemcpyDeviceToHost), "cy1");
        for (int j = 0; j < n; ++j)
            if (std::memcmp(&ym[(size_t)j], &y1[(size_t)j], 4) != 0) { ++bad; if (bad == 1) std::printf("  row %d col %d: %g vs %g\n", r, j, ym[j], y1[j]); }
    }
    (void)cudaFree(d_x); (void)cudaFree(d_suh); (void)cudaFree(d_svh); (void)cudaFree(d_tr);
    (void)cudaFree(d_y1); (void)cudaFree(d_ym);
    if (!ok) return 1;
    std::printf("dense_mr bits cb=%d %dx%d K=%d rows=%d: %d mismatched fp32 values\n", cb, k, n, bits, rows, bad);
    return bad == 0 ? 0 : 1;
}

}  // namespace

int main() {
    if (!strata::kernels::exl3_dense_mr_available()) {
        std::printf("dense_mr unavailable on this build SKIP\n");
        return 0;
    }
    int fails = 0;
    // the real dense shapes of the 3bpw model (mul1, K=5) plus codebook/bitrate coverage
    fails += run(160, 640, 5, 2, 1, 21);    // attn in_proj_qkv 2560x10240, m=1
    fails += run(160, 640, 5, 2, 5, 22);    // same, a 5-row window step
    fails += run(160, 640, 5, 2, 8, 23);    // a full row tile
    fails += run(160, 160, 5, 2, 3, 24);    // attn o_proj 2560x2560, odd tail rows
    fails += run(40, 40, 5, 2, 6, 25);      // shared expert down 640x640, parts > 1 shape
    fails += run(160, 8, 5, 2, 4, 26);      // narrow out (k_proj-like), k-split path
    fails += run(8, 8, 3, 2, 7, 27);        // small mul1
    fails += run(16, 24, 4, 1, 5, 28);      // mcg fallback
    fails += run(16, 16, 2, 0, 5, 29);      // 3inst fallback
    fails += run(8, 40, 6, 2, 5, 30);       // wide (K>5) two-funnel path
    // bit-identity: rows of a multi call vs singles, every tile size and tail
    fails += run_bits(160, 640, 5, 2, 2, 31);
    fails += run_bits(160, 640, 5, 2, 3, 32);
    fails += run_bits(160, 640, 5, 2, 4, 33);
    fails += run_bits(160, 640, 5, 2, 5, 34);
    fails += run_bits(160, 640, 5, 2, 8, 35);
    fails += run_bits(40, 40, 5, 2, 7, 36);
    fails += run_bits(160, 8, 5, 2, 6, 37);
    fails += run_bits(8, 40, 6, 2, 5, 38);
    std::printf("%s\n", fails ? "FAIL" : "OK");
    return fails ? 1 : 0;
}
