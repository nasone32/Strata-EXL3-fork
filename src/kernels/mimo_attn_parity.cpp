// src/kernels/mimo_attn_parity.cpp - GPU MiMo v2 attention vs the CPU reference
// (notes/MIMO-V2-PORT.md, include/strata/kernels/mimo_attn.hpp).
//
// Synthetic F16 tensors, no model.  The CPU reference below is deliberately independent of the
// kernels: rope angles in fp64 (fmod + std::cos/std::sin), attention accumulated in fp64, so
// the printed errors measure the GPU's fp32 path alone.  Cache rows past each row's seq_len are
// filled with F16 NaNs - a kernel that reads past the mask disagrees with the reference (a NaN
// score reaches the softmax denominator, whose guard writes a zero row).  Cases: full-attn
// layer (kv 4, group 16, no window/sink, theta 1e7), SWA
// layer (kv 8, group 8, window 128, sinks, theta 1e4), a 256-row prefill chunk crossing the
// window edge from both sides, a verify-shaped decode batch with distinct seq_lens (including
// a row whose window is entirely past its history and a row capped by seq_len), capacity >
// seq_len, and rope at positions > 1e6.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/mimo_attn.hpp"

namespace {

constexpr uint16_t kF16NaN = 0x7E88;

bool set_split_gate(const char* value) {
#ifdef _WIN32
    return _putenv_s("STRATA_MIMO_ATTN_SPLIT", value) == 0;
#else
    return setenv("STRATA_MIMO_ATTN_SPLIT", value, 1) == 0;
#endif
}

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

// ---------------------------------------------------------------- CPU reference (fp64 oracle)

// NEOX partial rope, the header's exact semantics: within each head row, pair (i, i + n_rot/2)
// of the first n_rot channels rotates by pos * theta^(-2i/n_rot); fp64 angle (the whole reason
// positions > 1e6 stay accurate), fp64 trig, F16 store.  Channels at or past n_rot untouched.
void ref_rope(uint16_t* qk, const int64_t* positions, int rows, int n_heads, int head_dim_qk,
              int n_rot, double theta) {
    const int half = n_rot / 2;
    for (int r = 0; r < rows; ++r)
        for (int h = 0; h < n_heads; ++h) {
            uint16_t* row = qk + ((size_t) r * n_heads + h) * head_dim_qk;
            for (int i = 0; i < half; ++i) {
                const double inv_freq = std::pow(theta, -2.0 * i / n_rot);
                const double angle = std::fmod((double) positions[r] * inv_freq, 2.0 * 3.14159265358979323846);
                const double c = std::cos(angle), s = std::sin(angle);
                const double a = f32_from_f16(row[i]), b = f32_from_f16(row[i + half]);
                row[i] = f16_from_f32((float) (a * c - b * s));
                row[i + half] = f16_from_f32((float) (a * s + b * c));
            }
        }
}

// Attention for one batch of rows.  Row r sits at position pos[r] with pos_len[r] keys in the
// cache; valid keys are [lo, hi] with lo = max(0, p - window + 1) (window <= 0: causal) and
// hi = min(p, len - 1).  The sink logit joins the softmax denominator raw, as a zero-value
// virtual key.  Scores and softmax in fp64; output F32.
void ref_attn(const uint16_t* q, const uint16_t* k, const uint16_t* v, float* out,
              const int64_t* pos, const int* len, int rows, int n_q_head, int n_kv_head,
              int head_dim_qk, int head_dim_v, int window, const float* sinks, double sm_scale) {
    const int g = n_q_head / n_kv_head;
    for (int r = 0; r < rows; ++r) {
        const int64_t p = pos[r];
        const int n = len[r];
        const int hi = (int) std::min<int64_t>(p, (int64_t) n - 1);
        const int lo = window > 0 ? (int) std::max<int64_t>(0, p - (window - 1)) : 0;
        for (int h = 0; h < n_q_head; ++h) {
            const int kvh = h / g;
            const uint16_t* qrow = q + ((size_t) r * n_q_head + h) * head_dim_qk;
            double m = -std::numeric_limits<double>::infinity();
            std::vector<double> s;
            if (lo <= hi) {
                s.reserve((size_t) (hi - lo + 1));
                for (int j = lo; j <= hi; ++j) {
                    const uint16_t* krow = k + ((size_t) j * n_kv_head + kvh) * head_dim_qk;
                    double dot = 0.0;
                    for (int d = 0; d < head_dim_qk; ++d)
                        dot += (double) f32_from_f16(qrow[d]) * f32_from_f16(krow[d]);
                    dot *= sm_scale;
                    s.push_back(dot);
                    m = std::max(m, dot);
                }
            }
            const double sink = sinks ? (double) sinks[h]
                                      : -std::numeric_limits<double>::infinity();
            m = std::max(m, sink);
            float* o = out + ((size_t) r * n_q_head + h) * head_dim_v;
            if (m == -std::numeric_limits<double>::infinity()) {
                std::memset(o, 0, sizeof(float) * head_dim_v);
                continue;
            }
            double den = sinks ? std::exp(sink - m) : 0.0;
            std::vector<double> num(head_dim_v, 0.0);
            for (int j = lo; j <= hi; ++j) {
                const double w = std::exp(s[(size_t) (j - lo)] - m);
                den += w;
                const uint16_t* vrow = v + ((size_t) j * n_kv_head + kvh) * head_dim_v;
                for (int d = 0; d < head_dim_v; ++d) num[d] += w * f32_from_f16(vrow[d]);
            }
            for (int d = 0; d < head_dim_v; ++d) o[d] = (float) (num[d] / den);
        }
    }
}

// ------------------------------------------------------------------------ comparison helpers

struct Compare {
    double max_abs = 0.0, max_rel = 0.0, norm_rel = 0.0;
    size_t bad = 0;   // non-finite GPU outputs
};

Compare compare_f32(const std::vector<float>& gpu, const std::vector<float>& ref) {
    Compare c;
    double num = 0.0, den = 0.0, ref_max = 0.0;
    for (float v : ref) ref_max = std::max(ref_max, std::fabs((double) v));
    const double floor_den = 0.05 * ref_max;   // outputs below 5% of the peak are F16-noise
    for (size_t i = 0; i < ref.size(); ++i) {
        const double g = gpu[i], b = ref[i];
        if (!std::isfinite(g)) { ++c.bad; continue; }
        const double d = std::fabs(g - b);
        num += d * d;
        den += b * b;
        c.max_abs = std::max(c.max_abs, d);
        c.max_rel = std::max(c.max_rel, d / std::max(std::fabs(b), floor_den));
    }
    c.norm_rel = std::sqrt(num) / (std::sqrt(den) + 1e-30);
    return c;
}

// --------------------------------------------------------------------------- GPU case drivers

uint16_t rand_f16(std::mt19937& rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    return f16_from_f32(u(rng));
}

#define CHK(e, what)                                                                              \
    do {                                                                                          \
        const cudaError_t _e = (e);                                                               \
        if (_e != cudaSuccess) {                                                                  \
            std::printf("CUDA %s: %s\n", what, cudaGetErrorString(_e));                           \
            return 1;                                                                             \
        }                                                                                         \
    } while (0)

// Rope-only parity: the GPU kernel vs ref_rope on the same F16 input.  Untouched channels
// (>= n_rot) must stay bit-identical; rotated channels may differ only in F16 rounding.
int run_rope(const char* name, int rows, int n_heads, int head_dim_qk, int n_rot, double theta,
             const std::vector<int64_t>& positions, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint16_t> host((size_t) rows * n_heads * head_dim_qk);
    for (auto& x : host) x = rand_f16(rng);
    std::vector<uint16_t> ref = host;

    uint16_t* d_qk = nullptr;
    int64_t* d_pos = nullptr;
    CHK(cudaMalloc(&d_qk, host.size() * 2), "rope malloc");
    CHK(cudaMalloc(&d_pos, positions.size() * 8), "rope pos malloc");
    CHK(cudaMemcpy(d_qk, host.data(), host.size() * 2, cudaMemcpyHostToDevice), "rope cpy");
    CHK(cudaMemcpy(d_pos, positions.data(), positions.size() * 8, cudaMemcpyHostToDevice), "rope pos cpy");
    strata::kernels::mimo_rope_f16(d_qk, d_pos, rows, n_heads, head_dim_qk, n_rot, theta, nullptr);
    CHK(cudaGetLastError(), "rope launch");
    CHK(cudaMemcpy(host.data(), d_qk, host.size() * 2, cudaMemcpyDeviceToHost), "rope back");
    CHK(cudaFree(d_qk), "rope free");
    CHK(cudaFree(d_pos), "rope pos free");

    ref_rope(ref.data(), positions.data(), rows, n_heads, head_dim_qk, n_rot, theta);

    size_t untouched_mism = 0, rotated_diff = 0, nonfinite = 0;
    double max_abs = 0.0;
    for (size_t i = 0; i < host.size(); ++i) {
        const int d = (int) (i % (size_t) head_dim_qk);
        if (d >= n_rot) {
            if (host[i] != ref[i]) ++untouched_mism;
            continue;
        }
        const float a = f32_from_f16(host[i]), b = f32_from_f16(ref[i]);
        if (!std::isfinite(a)) { ++nonfinite; continue; }
        if (host[i] != ref[i]) ++rotated_diff;
        max_abs = std::max(max_abs, std::fabs((double) a - b));
    }
    const bool ok = untouched_mism == 0 && nonfinite == 0 && max_abs <= 2e-3;   // ~2 F16 ulps
    std::printf("%-34s max_abs=%.3e differing=%zu/%zu untouched_mismatch=%zu%s\n", name, max_abs,
                rotated_diff, (size_t) rows * n_heads * n_rot, untouched_mism,
                ok ? "" : "  (FAIL)");
    return ok ? 0 : 1;
}

// End-to-end attention parity: GPU rope on q and the whole K cache, then GPU attention, against
// CPU-rope + fp64 attention.  Cache rows past the last real key are F16 NaN.  Decode cases pass
// per-row positions/lengths; prefill cases pass none and this driver synthesizes pos0..pos0+rows
// with key count pos0 + rows on both sides - exactly what the prefill kernel derives in-device.
int run_attn(const char* name, bool prefill, int rows, std::vector<int64_t> pos, std::vector<int> len,
             int64_t pos0, int n_q_head, int n_kv_head, int head_dim_qk, int head_dim_v,
              int capacity, int window, bool has_sinks, double theta, uint32_t seed, int ring = 0) {
    const double sm_scale = 1.0 / std::sqrt((double) head_dim_qk);
    std::mt19937 rng(seed);
    if (prefill) {
        pos.resize(rows);
        len.assign(rows, (int) (pos0 + rows));
        for (int r = 0; r < rows; ++r) pos[r] = pos0 + r;
    }

    int64_t real_n = 0;
    for (int r = 0; r < rows; ++r)
        real_n = std::max<int64_t>(real_n, std::min<int64_t>(pos[r], (int64_t) len[r] - 1) + 1);

    std::vector<uint16_t> k((size_t) capacity * n_kv_head * head_dim_qk, kF16NaN);
    std::vector<uint16_t> v((size_t) capacity * n_kv_head * head_dim_v, kF16NaN);
    for (size_t i = 0; i < (size_t) real_n * n_kv_head * head_dim_qk; ++i) k[i] = rand_f16(rng);
    for (size_t i = 0; i < (size_t) real_n * n_kv_head * head_dim_v; ++i) v[i] = rand_f16(rng);
    std::vector<uint16_t> q((size_t) rows * n_q_head * head_dim_qk);
    for (auto& x : q) x = rand_f16(rng);
    std::vector<float> sinks(n_q_head);
    {
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (int h = 0; h < n_q_head; ++h)
            sinks[h] = (h % 2) == 0 ? u(rng) : 4.0f + 2.0f * u(rng);   // half dominate the max
    }
    std::vector<int64_t> kpos((size_t) capacity);
    for (int64_t i = 0; i < capacity; ++i) kpos[i] = i;

    // CPU side: rope q and the real K rows, then fp64 attention
    std::vector<uint16_t> q_ref = q, k_ref = k;
    ref_rope(q_ref.data(), pos.data(), rows, n_q_head, head_dim_qk, 64, theta);
    ref_rope(k_ref.data(), kpos.data(), (int) real_n, n_kv_head, head_dim_qk, 64, theta);
    std::vector<float> ref((size_t) rows * n_q_head * head_dim_v);
    ref_attn(q_ref.data(), k_ref.data(), v.data(), ref.data(), pos.data(), len.data(), rows,
             n_q_head, n_kv_head, head_dim_qk, head_dim_v, window, has_sinks ? sinks.data() : nullptr,
             sm_scale);

    // GPU side
    uint16_t *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    float *d_out = nullptr, *d_sinks = nullptr;
    int64_t *d_pos = nullptr, *d_kpos = nullptr;
    int* d_len = nullptr;
    CHK(cudaMalloc(&d_q, q.size() * 2), "q");
    CHK(cudaMalloc(&d_k, k.size() * 2), "k");
    CHK(cudaMalloc(&d_v, v.size() * 2), "v");
    CHK(cudaMalloc(&d_out, (size_t) rows * n_q_head * head_dim_v * 4), "out");
    CHK(cudaMalloc(&d_pos, pos.size() * 8), "pos");
    CHK(cudaMalloc(&d_len, len.size() * 4), "len");
    CHK(cudaMalloc(&d_kpos, kpos.size() * 8), "kpos");
    if (has_sinks) CHK(cudaMalloc(&d_sinks, sinks.size() * 4), "sinks");
    CHK(cudaMemcpy(d_q, q.data(), q.size() * 2, cudaMemcpyHostToDevice), "cq");
    CHK(cudaMemcpy(d_k, k.data(), k.size() * 2, cudaMemcpyHostToDevice), "ck");
    CHK(cudaMemcpy(d_v, v.data(), v.size() * 2, cudaMemcpyHostToDevice), "cv");
    if (d_sinks) CHK(cudaMemcpy(d_sinks, sinks.data(), sinks.size() * 4, cudaMemcpyHostToDevice), "cs");
    CHK(cudaMemcpy(d_pos, pos.data(), pos.size() * 8, cudaMemcpyHostToDevice), "cpos");
    CHK(cudaMemcpy(d_len, len.data(), len.size() * 4, cudaMemcpyHostToDevice), "clen");
    CHK(cudaMemcpy(d_kpos, kpos.data(), kpos.size() * 8, cudaMemcpyHostToDevice), "ckpos");
    strata::kernels::mimo_rope_f16(d_q, d_pos, rows, n_q_head, head_dim_qk, 64, theta, nullptr);
    strata::kernels::mimo_rope_f16(d_k, d_kpos, capacity, n_kv_head, head_dim_qk, 64, theta, nullptr);
    if (prefill)
        strata::kernels::mimo_attn_prefill(d_q, d_k, d_v, d_out, pos0, rows, n_q_head, n_kv_head,
                                           head_dim_qk, head_dim_v, capacity, window,
                                           (const void*) d_sinks, sm_scale, nullptr);
    else
        strata::kernels::mimo_attn_decode(d_q, d_k, d_v, d_out, d_pos, d_len, rows, n_q_head,
                                          n_kv_head, head_dim_qk, head_dim_v, capacity, window,
                                          (const void*) d_sinks, sm_scale, nullptr);
    CHK(cudaGetLastError(), "attn launch");
    std::vector<float> gpu((size_t) rows * n_q_head * head_dim_v);
    CHK(cudaMemcpy(gpu.data(), d_out, gpu.size() * 4, cudaMemcpyDeviceToHost), "out back");
    bool ring_exact = true;
    if (ring > 0) {
        // Repack already-rotated flat KV into an exactly sized physical ring. Compare against
        // both the independent CPU oracle below and the FLAT GPU result byte-for-byte. The
        // early chunk queries must still see history after the whole chunk has been appended.
        uint16_t *rk = nullptr, *rv = nullptr;
        const size_t kw = (size_t) n_kv_head * head_dim_qk, vw = (size_t) n_kv_head * head_dim_v;
        CHK(cudaMalloc(&rk, (size_t) ring * kw * 2), "ring k");
        CHK(cudaMalloc(&rv, (size_t) ring * vw * 2), "ring v");
        CHK(cudaMemset(rk, 0xff, (size_t) ring * kw * 2), "ring k poison");
        CHK(cudaMemset(rv, 0xff, (size_t) ring * vw * 2), "ring v poison");
        for (int64_t p = std::max<int64_t>(0, real_n - ring); p < real_n; ++p) {
            CHK(cudaMemcpy(rk + (p % ring) * kw, d_k + p * kw, kw * 2, cudaMemcpyDeviceToDevice), "ring k append");
            CHK(cudaMemcpy(rv + (p % ring) * vw, d_v + p * vw, vw * 2, cudaMemcpyDeviceToDevice), "ring v append");
        }
        if (prefill)
            strata::kernels::mimo_attn_prefill(d_q, rk, rv, d_out, pos0, rows, n_q_head, n_kv_head,
                                               head_dim_qk, head_dim_v, capacity, window, d_sinks,
                                               sm_scale, nullptr, ring);
        else
            strata::kernels::mimo_attn_decode(d_q, rk, rv, d_out, d_pos, d_len, rows, n_q_head, n_kv_head,
                                              head_dim_qk, head_dim_v, capacity, window, d_sinks,
                                              sm_scale, nullptr, ring);
        std::vector<float> ring_out(gpu.size());
        CHK(cudaMemcpy(ring_out.data(), d_out, ring_out.size() * 4, cudaMemcpyDeviceToHost), "ring out");
        ring_exact = std::memcmp(gpu.data(), ring_out.data(), gpu.size() * 4) == 0;
        std::printf("%-34s ring=%d flat byte-exact=%s\n", name, ring, ring_exact ? "yes" : "NO");
        gpu.swap(ring_out);
        CHK(cudaFree(rk), "ring k free");
        CHK(cudaFree(rv), "ring v free");
    }
    CHK(cudaDeviceSynchronize(), "sync");
    CHK(cudaFree(d_q), "q free");
    CHK(cudaFree(d_k), "k free");
    CHK(cudaFree(d_v), "v free");
    CHK(cudaFree(d_out), "out free");
    CHK(cudaFree(d_sinks), "sinks free");
    CHK(cudaFree(d_pos), "pos free");
    CHK(cudaFree(d_len), "len free");
    CHK(cudaFree(d_kpos), "kpos free");

    const Compare c = compare_f32(gpu, ref);
    const bool ok = ring_exact && c.bad == 0 && c.max_rel < 2e-2;
    std::printf("%-34s max_rel=%.3e norm_rel=%.3e max_abs=%.3e nonfinite=%zu%s\n", name, c.max_rel,
                c.norm_rel, c.max_abs, c.bad, ok ? "" : "  (FAIL)");
    return ok ? 0 : 1;
}

// ------------------------------------------------------- split-sequence (opt-in) coverage
//
// The same inputs/oracle/bar as run_attn, but through the *_split entries with the gate ON
// (main sets STRATA_MIMO_ATTN_SPLIT=1 after the gate-off byte-identity checks).  Extra checks
// on top of the oracle comparison:
//   - the K/V caches must come back byte-identical (the split path only ever reads them -
//     "history unaltered", the rollback contract);
//   - ring cases: split-flat vs split-ring outputs byte-identical (same logical key order).
int run_attn_split(const char* name, bool prefill, int rows, std::vector<int64_t> pos,
                   std::vector<int> len, int64_t pos0, int n_q_head, int n_kv_head,
                   int head_dim_qk, int head_dim_v, int capacity, int window, bool has_sinks,
                   double theta, uint32_t seed, int ring = 0, int keys_hint = 0,
                   int splits_hint = 0, bool check_caches = true) {
    const double sm_scale = 1.0 / std::sqrt((double) head_dim_qk);
    std::mt19937 rng(seed);
    if (prefill) {
        pos.resize(rows);
        len.assign(rows, (int) (pos0 + rows));
        for (int r = 0; r < rows; ++r) pos[r] = pos0 + r;
    }

    int64_t real_n = 0;
    for (int r = 0; r < rows; ++r)
        real_n = std::max<int64_t>(real_n, std::min<int64_t>(pos[r], (int64_t) len[r] - 1) + 1);

    std::vector<uint16_t> k((size_t) capacity * n_kv_head * head_dim_qk, kF16NaN);
    std::vector<uint16_t> v((size_t) capacity * n_kv_head * head_dim_v, kF16NaN);
    for (size_t i = 0; i < (size_t) real_n * n_kv_head * head_dim_qk; ++i) k[i] = rand_f16(rng);
    for (size_t i = 0; i < (size_t) real_n * n_kv_head * head_dim_v; ++i) v[i] = rand_f16(rng);
    std::vector<uint16_t> q((size_t) rows * n_q_head * head_dim_qk);
    for (auto& x : q) x = rand_f16(rng);
    std::vector<float> sinks(n_q_head);
    {
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (int h = 0; h < n_q_head; ++h)
            sinks[h] = (h % 2) == 0 ? u(rng) : 4.0f + 2.0f * u(rng);   // half dominate the max
    }
    std::vector<int64_t> kpos((size_t) capacity);
    for (int64_t i = 0; i < capacity; ++i) kpos[i] = i;

    std::vector<uint16_t> q_ref = q, k_ref = k;
    ref_rope(q_ref.data(), pos.data(), rows, n_q_head, head_dim_qk, 64, theta);
    ref_rope(k_ref.data(), kpos.data(), (int) real_n, n_kv_head, head_dim_qk, 64, theta);
    std::vector<float> ref((size_t) rows * n_q_head * head_dim_v);
    ref_attn(q_ref.data(), k_ref.data(), v.data(), ref.data(), pos.data(), len.data(), rows,
             n_q_head, n_kv_head, head_dim_qk, head_dim_v, window, has_sinks ? sinks.data() : nullptr,
             sm_scale);

    uint16_t *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    float *d_out = nullptr, *d_sinks = nullptr;
    int64_t *d_pos = nullptr, *d_kpos = nullptr;
    int* d_len = nullptr;
    CHK(cudaMalloc(&d_q, q.size() * 2), "sq q");
    CHK(cudaMalloc(&d_k, k.size() * 2), "sq k");
    CHK(cudaMalloc(&d_v, v.size() * 2), "sq v");
    CHK(cudaMalloc(&d_out, (size_t) rows * n_q_head * head_dim_v * 4), "sq out");
    CHK(cudaMalloc(&d_pos, pos.size() * 8), "sq pos");
    CHK(cudaMalloc(&d_len, len.size() * 4), "sq len");
    CHK(cudaMalloc(&d_kpos, kpos.size() * 8), "sq kpos");
    if (has_sinks) CHK(cudaMalloc(&d_sinks, sinks.size() * 4), "sq sinks");
    CHK(cudaMemcpy(d_q, q.data(), q.size() * 2, cudaMemcpyHostToDevice), "sq cq");
    CHK(cudaMemcpy(d_k, k.data(), k.size() * 2, cudaMemcpyHostToDevice), "sq ck");
    CHK(cudaMemcpy(d_v, v.data(), v.size() * 2, cudaMemcpyHostToDevice), "sq cv");
    if (d_sinks) CHK(cudaMemcpy(d_sinks, sinks.data(), sinks.size() * 4, cudaMemcpyHostToDevice), "sq cs");
    CHK(cudaMemcpy(d_pos, pos.data(), pos.size() * 8, cudaMemcpyHostToDevice), "sq cpos");
    CHK(cudaMemcpy(d_len, len.data(), len.size() * 4, cudaMemcpyHostToDevice), "sq clen");
    CHK(cudaMemcpy(d_kpos, kpos.data(), kpos.size() * 8, cudaMemcpyHostToDevice), "sq ckpos");
    strata::kernels::mimo_rope_f16(d_q, d_pos, rows, n_q_head, head_dim_qk, 64, theta, nullptr);
    strata::kernels::mimo_rope_f16(d_k, d_kpos, capacity, n_kv_head, head_dim_qk, 64, theta, nullptr);
    CHK(cudaGetLastError(), "sq rope");

    // the pre-reserved workspace: no allocation may happen under the queued attention work
    const uint32_t max_splits = splits_hint > 0 ? (uint32_t) splits_hint : 128;
    strata::kernels::MimoAttnSplitWs ws;
    ws.max_rows = (uint32_t) rows;
    ws.max_splits = max_splits;
    ws.bytes = strata::kernels::mimo_attn_split_ws_bytes(ws.max_rows, (uint32_t) n_kv_head,
                                                         (uint32_t) n_q_head,
                                                         (uint32_t) head_dim_v, max_splits);
    if (ws.bytes == 0) { std::printf("%-34s workspace size == 0 (FAIL)\n", name); return 1; }
    CHK(cudaMalloc(&ws.mem, (size_t) ws.bytes), "sq ws");
    CHK(cudaMemset(ws.mem, 0xff, (size_t) ws.bytes), "sq ws poison");

    // post-rope device cache snapshot: the split call must leave every cache byte untouched
    std::vector<uint16_t> k_before, v_before;
    if (check_caches) {
        k_before.resize(k.size());
        v_before.resize(v.size());
        CHK(cudaMemcpy(k_before.data(), d_k, k.size() * 2, cudaMemcpyDeviceToHost), "sq k snap");
        CHK(cudaMemcpy(v_before.data(), d_v, v.size() * 2, cudaMemcpyDeviceToHost), "sq v snap");
    }

    const int kh = keys_hint > 0 ? keys_hint : (int) std::min<int64_t>(real_n, INT32_MAX);
    if (prefill)
        strata::kernels::mimo_attn_prefill_split(d_q, d_k, d_v, d_out, pos0, rows, n_q_head,
                                                 n_kv_head, head_dim_qk, head_dim_v, capacity,
                                                 window, (const void*) d_sinks, sm_scale, kh,
                                                 splits_hint, &ws, nullptr);
    else
        strata::kernels::mimo_attn_decode_split(d_q, d_k, d_v, d_out, d_pos, d_len, rows, n_q_head,
                                                n_kv_head, head_dim_qk, head_dim_v, capacity, window,
                                                (const void*) d_sinks, sm_scale, kh, splits_hint,
                                                &ws, nullptr);
    CHK(cudaGetLastError(), "sq split attn");
    std::vector<float> gpu((size_t) rows * n_q_head * head_dim_v);
    CHK(cudaMemcpy(gpu.data(), d_out, gpu.size() * 4, cudaMemcpyDeviceToHost), "sq out back");

    // history unaltered: the caches (and only they) must be byte-identical to the post-rope
    // snapshot; the workspace may change (it is scratch by contract)
    bool caches_same = true;
    if (check_caches) {
        std::vector<uint16_t> kd(k.size()), vd(v.size());
        CHK(cudaMemcpy(kd.data(), d_k, k.size() * 2, cudaMemcpyDeviceToHost), "sq k back");
        CHK(cudaMemcpy(vd.data(), d_v, v.size() * 2, cudaMemcpyDeviceToHost), "sq v back");
        caches_same = std::memcmp(kd.data(), k_before.data(), k.size() * 2) == 0 &&
                      std::memcmp(vd.data(), v_before.data(), v.size() * 2) == 0;
    }

    bool ring_exact = true;
    if (ring > 0) {
        uint16_t *rk = nullptr, *rv = nullptr;
        const size_t kw = (size_t) n_kv_head * head_dim_qk, vw = (size_t) n_kv_head * head_dim_v;
        CHK(cudaMalloc(&rk, (size_t) ring * kw * 2), "sq ring k");
        CHK(cudaMalloc(&rv, (size_t) ring * vw * 2), "sq ring v");
        CHK(cudaMemset(rk, 0xff, (size_t) ring * kw * 2), "sq ring k poison");
        CHK(cudaMemset(rv, 0xff, (size_t) ring * vw * 2), "sq ring v poison");
        for (int64_t p = std::max<int64_t>(0, real_n - ring); p < real_n; ++p) {
            CHK(cudaMemcpy(rk + (p % ring) * kw, d_k + p * kw, kw * 2, cudaMemcpyDeviceToDevice), "sq ring k append");
            CHK(cudaMemcpy(rv + (p % ring) * vw, d_v + p * vw, vw * 2, cudaMemcpyDeviceToDevice), "sq ring v append");
        }
        if (prefill)
            strata::kernels::mimo_attn_prefill_split(d_q, rk, rv, d_out, pos0, rows, n_q_head,
                                                     n_kv_head, head_dim_qk, head_dim_v, capacity,
                                                     window, d_sinks, sm_scale, kh, splits_hint, &ws,
                                                     nullptr, ring);
        else
            strata::kernels::mimo_attn_decode_split(d_q, rk, rv, d_out, d_pos, d_len, rows, n_q_head,
                                                    n_kv_head, head_dim_qk, head_dim_v, capacity,
                                                    window, d_sinks, sm_scale, kh, splits_hint, &ws,
                                                    nullptr, ring);
        std::vector<float> ring_out(gpu.size());
        CHK(cudaMemcpy(ring_out.data(), d_out, ring_out.size() * 4, cudaMemcpyDeviceToHost), "sq ring out");
        ring_exact = std::memcmp(gpu.data(), ring_out.data(), gpu.size() * 4) == 0;
        gpu.swap(ring_out);
        CHK(cudaFree(rk), "sq ring k free");
        CHK(cudaFree(rv), "sq ring v free");
    }
    CHK(cudaDeviceSynchronize(), "sq sync");
    CHK(cudaFree(ws.mem), "sq ws free");
    CHK(cudaFree(d_q), "sq q free");
    CHK(cudaFree(d_k), "sq k free");
    CHK(cudaFree(d_v), "sq v free");
    CHK(cudaFree(d_out), "sq out free");
    CHK(cudaFree(d_sinks), "sq sinks free");
    CHK(cudaFree(d_pos), "sq pos free");
    CHK(cudaFree(d_len), "sq len free");
    CHK(cudaFree(d_kpos), "sq kpos free");

    const Compare c = compare_f32(gpu, ref);
    const bool ok = ring_exact && caches_same && c.bad == 0 && c.max_rel < 2e-2;
    std::printf("%-34s max_rel=%.3e norm_rel=%.3e max_abs=%.3e nonfinite=%zu caches_same=%s%s%s\n",
                name, c.max_rel, c.norm_rel, c.max_abs, c.bad, caches_same ? "yes" : "NO",
                ring > 0 ? (ring_exact ? " ring_exact" : " ring_DIFFERS") : "",
                ok ? "" : "  (FAIL)");
    return ok ? 0 : 1;
}

// Gate OFF: the *_split entries must reproduce the plain entries' outputs byte for byte
// (the documented fallback - same dispatch, same kernels).
int run_split_fallback_exact(const char* name, bool prefill, int rows, int64_t pos0, int window,
                             bool has_sinks, int ring, uint32_t seed) {
    const int n_q_head = 64, n_kv_head = has_sinks ? 8 : 4, hd_qk = 192, hd_v = 128;
    const int capacity = ring > 0 ? ring : 2600;
    const double theta = has_sinks ? 1e4 : 1e7;
    const double sm_scale = 1.0 / std::sqrt((double) hd_qk);
    std::mt19937 rng(seed);
    std::vector<int64_t> pos((size_t) rows);
    std::vector<int> len((size_t) rows);
    for (int r = 0; r < rows; ++r) { pos[r] = pos0 + r; len[r] = (int) (pos0 + r + 1); }
    const int64_t real_n = pos0 + rows;
    std::vector<uint16_t> k((size_t) capacity * n_kv_head * hd_qk), v((size_t) capacity * n_kv_head * hd_v);
    for (auto& x : k) x = rand_f16(rng);
    for (auto& x : v) x = rand_f16(rng);
    std::vector<uint16_t> q((size_t) rows * n_q_head * hd_qk);
    for (auto& x : q) x = rand_f16(rng);
    std::vector<float> sinks(n_q_head);
    {
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (int h = 0; h < n_q_head; ++h) sinks[h] = (h % 2) == 0 ? u(rng) : 4.0f + 2.0f * u(rng);
    }
    std::vector<int64_t> kpos((size_t) capacity);
    for (int64_t i = 0; i < capacity; ++i) kpos[i] = i;

    uint16_t *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    float *d_out_old = nullptr, *d_out_split = nullptr, *d_sinks = nullptr;
    int64_t *d_pos = nullptr, *d_kpos = nullptr;
    int* d_len = nullptr;
    CHK(cudaMalloc(&d_q, q.size() * 2), "fb q");
    CHK(cudaMalloc(&d_k, k.size() * 2), "fb k");
    CHK(cudaMalloc(&d_v, v.size() * 2), "fb v");
    CHK(cudaMalloc(&d_out_old, (size_t) rows * n_q_head * hd_v * 4), "fb out old");
    CHK(cudaMalloc(&d_out_split, (size_t) rows * n_q_head * hd_v * 4), "fb out split");
    CHK(cudaMalloc(&d_pos, rows * 8), "fb pos");
    CHK(cudaMalloc(&d_len, rows * 4), "fb len");
    CHK(cudaMalloc(&d_kpos, kpos.size() * 8), "fb kpos");
    if (has_sinks) CHK(cudaMalloc(&d_sinks, sinks.size() * 4), "fb sinks");
    CHK(cudaMemcpy(d_q, q.data(), q.size() * 2, cudaMemcpyHostToDevice), "fb cq");
    CHK(cudaMemcpy(d_k, k.data(), k.size() * 2, cudaMemcpyHostToDevice), "fb ck");
    CHK(cudaMemcpy(d_v, v.data(), v.size() * 2, cudaMemcpyHostToDevice), "fb cv");
    if (d_sinks) CHK(cudaMemcpy(d_sinks, sinks.data(), sinks.size() * 4, cudaMemcpyHostToDevice), "fb cs");
    CHK(cudaMemcpy(d_pos, pos.data(), rows * 8, cudaMemcpyHostToDevice), "fb cpos");
    CHK(cudaMemcpy(d_len, len.data(), rows * 4, cudaMemcpyHostToDevice), "fb clen");
    CHK(cudaMemcpy(d_kpos, kpos.data(), kpos.size() * 8, cudaMemcpyHostToDevice), "fb ckpos");
    strata::kernels::mimo_rope_f16(d_q, d_pos, rows, n_q_head, hd_qk, 64, theta, nullptr);
    strata::kernels::mimo_rope_f16(d_k, d_kpos, capacity, n_kv_head, hd_qk, 64, theta, nullptr);
    CHK(cudaGetLastError(), "fb rope");
    const void* snk = (const void*) d_sinks;
    if (prefill) {
        strata::kernels::mimo_attn_prefill(d_q, d_k, d_v, d_out_old, pos0, rows, n_q_head, n_kv_head,
                                           hd_qk, hd_v, capacity, window, snk, sm_scale, nullptr, ring);
        strata::kernels::mimo_attn_prefill_split(d_q, d_k, d_v, d_out_split, pos0, rows, n_q_head,
                                                 n_kv_head, hd_qk, hd_v, capacity, window, snk,
                                                 sm_scale, 0, 64, nullptr, nullptr, ring);
    } else {
        strata::kernels::mimo_attn_decode(d_q, d_k, d_v, d_out_old, d_pos, d_len, rows, n_q_head,
                                          n_kv_head, hd_qk, hd_v, capacity, window, snk, sm_scale,
                                          nullptr, ring);
        strata::kernels::mimo_attn_decode_split(d_q, d_k, d_v, d_out_split, d_pos, d_len, rows,
                                                n_q_head, n_kv_head, hd_qk, hd_v, capacity, window,
                                                snk, sm_scale, 0, 64, nullptr, nullptr, ring);
    }
    CHK(cudaGetLastError(), "fb attn");
    const size_t bytes = (size_t) rows * n_q_head * hd_v * 4;
    std::vector<float> oldf(bytes / 4), splitf(bytes / 4);
    CHK(cudaMemcpy(oldf.data(), d_out_old, bytes, cudaMemcpyDeviceToHost), "fb old back");
    CHK(cudaMemcpy(splitf.data(), d_out_split, bytes, cudaMemcpyDeviceToHost), "fb split back");
    const bool exact = std::memcmp(oldf.data(), splitf.data(), bytes) == 0;
    std::printf("%-34s byte_exact=%s%s\n", name, exact ? "yes" : "NO", exact ? "" : "  (FAIL)");
    CHK(cudaFree(d_q), "fb q free"); CHK(cudaFree(d_k), "fb k free"); CHK(cudaFree(d_v), "fb v free");
    CHK(cudaFree(d_out_old), "fb old free"); CHK(cudaFree(d_out_split), "fb split free");
    CHK(cudaFree(d_sinks), "fb sinks free"); CHK(cudaFree(d_pos), "fb pos free");
    CHK(cudaFree(d_len), "fb len free"); CHK(cudaFree(d_kpos), "fb kpos free");
    return exact ? 0 : 1;
}

// ------------------------------------------------- workspace-validation unit tests
//
// Pure host, no CUDA calls anywhere (they run before the device check): the negative space
// of MimoAttnSplitWs - byte-short, null, misaligned, out-of-int-range and wrong-G
// reservations - against mimo_attn_split_ws_bytes/ws_check directly.  The entries call the
// same helper and exit(1) on its false; exercising the helper here covers the logic without
// death-test forking, which this harness avoids on purpose (the process holds a live CUDA
// context by the time cases run - forking it is unsupported).
int run_ws_unit_tests() {
    namespace K = strata::kernels;
    int fails = 0;
    auto expect = [&](bool ok, const char* what) {
        if (!ok) {
            ++fails;
            std::printf("ws unit %-42s expected OK, got FAIL\n", what);
        }
    };
    // the engine's reservation shape: 8 rows, worst kv 8, 64 heads, hd_v 128, 64 splits
    const uint64_t engine_need = K::mimo_attn_split_ws_bytes(8, 8, 64, 128, 64);
    expect(engine_need == 8ull * 8 * 64 * 8 * 130 * 4, "bytes exact 8/8/64/128/64");
    // exactly the compiled GQA set; anything else (3, 5, 32) is rejected, not just G > 16
    expect(K::mimo_attn_split_ws_bytes(4, 1, 4, 128, 4) > 0, "G4 admitted");
    expect(K::mimo_attn_split_ws_bytes(4, 1, 3, 128, 4) == 0, "G3 rejected");
    expect(K::mimo_attn_split_ws_bytes(4, 1, 5, 128, 4) == 0, "G5 rejected");
    expect(K::mimo_attn_split_ws_bytes(4, 1, 32, 128, 4) == 0, "G32 rejected");
    expect(K::mimo_attn_split_ws_bytes(4, 1, 33, 128, 4) == 0, "G33 rejected");
    expect(K::mimo_attn_split_ws_bytes(4, 16, 256, 128, 4) > 0, "G16 admitted (256/16)");
    // 64-bit product overflow reports 0, never a wrapped size
    expect(K::mimo_attn_split_ws_bytes(3000000000u, 16, 256, 128, 4294967295u) == 0,
           "product overflow -> 0");
    // a usable descriptor passes and yields the clamped plan
    K::MimoAttnSplitWs ws{};
    ws.mem = (void*) (uintptr_t) 0x10000;   // never dereferenced by the pure-host checks
    ws.bytes = engine_need;
    ws.max_rows = 8;
    ws.max_splits = 64;
    int n = 0;
    // heuristic for 8 rows x 8 kv heads: ceil(8192/128)=64 by keys, but the ~512-block
    // occupancy cap allows 512/(8*8)=8 - the plan is the min, clamped by max_splits anyway
    expect(K::mimo_attn_split_ws_check(&ws, 8, 8, 64, 128, 8192, 0, &n) && n == 8,
           "good ws -> heuristic plan 8 (occupancy-capped)");
    expect(K::mimo_attn_split_ws_check(&ws, 1, 4, 64, 128, 8192, 0, &n) && n == 64,
           "good ws -> heuristic plan 64 (T1, kv4)");
    expect(K::mimo_attn_split_ws_check(&ws, 8, 8, 64, 128, 8192, 100, &n) && n == 64,
           "splits_hint 100 clamped to 64");
    K::MimoAttnSplitWs ws2 = ws;
    ws2.max_splits = 2;
    expect(K::mimo_attn_split_ws_check(&ws2, 8, 8, 64, 128, 8192, 0, &n) && n == 2,
           "max_splits 2 clamps plan to 2");
    expect(K::mimo_attn_split_ws_check(&ws2, 8, 8, 64, 128, 129, 0, &n) && n == 2,
           "bound 129 -> plan 2");
    K::MimoAttnSplitWs ws3 = ws;
    ws3.max_splits = 4;
    ws3.bytes = K::mimo_attn_split_ws_bytes(1, 4, 64, 128, 4);
    expect(K::mimo_attn_split_ws_check(&ws3, 1, 4, 64, 128, 8192, 4, &n) && n == 4,
           "bytes-exact ws passes");
    // one byte short of the requirement
    K::MimoAttnSplitWs shortb = ws3;
    shortb.bytes -= 1;
    expect(!K::mimo_attn_split_ws_check(&shortb, 1, 4, 64, 128, 8192, 4, &n), "bytes=need-1");
    // null descriptor / null mem / misaligned base
    expect(!K::mimo_attn_split_ws_check(nullptr, 1, 4, 64, 128, 8192, 4, &n), "ws null");
    K::MimoAttnSplitWs nullmem = ws3;
    nullmem.mem = nullptr;
    expect(!K::mimo_attn_split_ws_check(&nullmem, 1, 4, 64, 128, 8192, 4, &n), "mem null");
    K::MimoAttnSplitWs misalign = ws3;
    misalign.mem = (void*) (uintptr_t) 0x10008;
    expect(!K::mimo_attn_split_ws_check(&misalign, 1, 4, 64, 128, 8192, 4, &n), "misaligned base");
    // bounds over the reservation, and uint32 values whose int cast would wrap
    expect(!K::mimo_attn_split_ws_check(&ws, 9, 8, 64, 128, 8192, 0, &n), "rows > max_rows");
    K::MimoAttnSplitWs bigrows = ws;
    bigrows.max_rows = 4294967295u;   // > INT32_MAX
    bigrows.bytes = UINT64_MAX;
    expect(!K::mimo_attn_split_ws_check(&bigrows, 8, 8, 64, 128, 8192, 0, &n),
           "max_rows > INT32_MAX");
    K::MimoAttnSplitWs bigsplits = ws;
    bigsplits.max_splits = 2147483648u;   // 2^31: int cast wraps negative
    bigsplits.bytes = UINT64_MAX;
    expect(!K::mimo_attn_split_ws_check(&bigsplits, 8, 8, 64, 128, 8192, 0, &n),
           "max_splits > INT32_MAX");
    // keys_bound < 1 clamps to a plan of 1 (the entry then falls back, documented)
    expect(K::mimo_attn_split_ws_check(&ws, 8, 8, 64, 128, 0, 0, &n) && n == 1,
           "nonsense bound -> plan 1 (fallback)");
    if (fails == 0) std::printf("ws unit tests: %s\n", "OK (21 checks)");
    return fails;
}

}  // namespace

int main() {
    int devices = 0;
    int fails = run_ws_unit_tests();   // pure host: runs even on the GPU-less SKIP path below
    if (fails) { std::printf("FAIL (ws unit tests)\n"); return 1; }
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no CUDA/HIP device (ws unit tests above still ran)\n");
        return 77;
    }

    // ---- rope, including positions > 1e6 (fp64 angle or the rotation is meaningless there)
    fails += run_rope("rope q full-layer theta 1e7", 8, 64, 192, 64, 1e7,
                      {0, 1, 63, 127, 128, 1 << 20, (1 << 20) + 1, 1500000}, 11);
    fails += run_rope("rope k swa-layer theta 1e4", 6, 8, 192, 64, 1e4,
                      {0, 1, 127, 4096, 999999, 2999999}, 12);
    fails += run_rope("rope dflash hd128 theta 1e4", 5, 64, 128, 64, 1e4,
                      {0, 1, 127, 1 << 21, (1 << 21) + 3}, 13);

    // ---- decode: full-attention layer (kv 4, group 16), verify-shaped rows with distinct
    //       seq_lens, capacity > seq_len.  The last row is capped by its own length (p=1008,
    //       len=1007: keys end at 1006) while other rows keep those slots real, so a kernel
    //       that ignores the per-row length produces a finite but wrong answer.
    fails += run_attn("decode full kv4 g16 nowin", false, 8, {999, 1000, 1001, 1002, 1003, 1004, 1010, 1008},
                      {1000, 1001, 1002, 1003, 1004, 1005, 1011, 1007}, 0, 64, 4, 192, 128, 1200, 0,
                      false, 1e7, 21);

    // ---- decode: SWA layer (kv 8, group 8, window 128, sinks): window clamped at 0 (p=30),
    //       first dropped key (p=128), a row capped by seq_len (p=1006, len=1007 ok; p=1000,
    //       len=1001), and a row whose window lies entirely past its history (p=1007, len=3:
    //       sink-only, output exactly 0)
    fails += run_attn("decode swa kv8 g8 w128 sinks", false, 8, {30, 127, 128, 129, 500, 1000, 1006, 1007},
                      {31, 128, 129, 130, 501, 1001, 1007, 3}, 0, 64, 8, 192, 128, 1100, 128, true,
                      1e4, 22);

    // ---- prefill: 256-row chunk over 1000 history crossing the window edge from above
    //      (every query at pos >= 1000 >> window 128) and from below (pos0 = 0, pos < window)
    fails += run_attn("prefill swa pos1000 w128 sinks", true, 256, {}, {}, 1000, 64, 8, 192, 128,
                      1300, 128, true, 1e4, 23);
    fails += run_attn("prefill swa pos0 w128 sinks", true, 256, {}, {}, 0, 64, 8, 192, 128, 300,
                      128, true, 1e4, 24);
    fails += run_attn("prefill full pos1000 nowin", true, 256, {}, {}, 1000, 64, 4, 192, 128, 1300,
                       0, false, 1e7, 25);

    // Minimal safe ring: 127 historical rows + the entire chunk, including multiple wraps,
    // chunk-start/end crossing the physical boundary, and an eight-row verify with length caps.
    fails += run_attn("ring prefill initial", true, 256, {}, {}, 0, 64, 8, 192, 128, 256,
                       128, true, 1e4, 31, 256);
    fails += run_attn("ring prefill wrap", true, 512, {}, {}, 1000, 64, 8, 192, 128, 1600,
                       128, true, 1e4, 32, 639);
    fails += run_attn("ring prefill multiple wraps", true, 512, {}, {}, 1800, 64, 8, 192, 128, 2400,
                       128, true, 1e4, 33, 639);
    fails += run_attn("ring verify length caps", false, 8,
                       {1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004},
                       {1998, 1999, 2000, 2001, 2002, 2003, 2002, 2003},
                       0, 64, 8, 192, 128, 2200, 128, true, 1e4, 34, 135);

    // ---- split-sequence (opt-in), part 1: gate OFF - the *_split entries must be
    //       byte-identical to the plain ones (the documented fallback path)
    if (!set_split_gate("0")) {
        std::puts("setting split gate off failed");
        return 1;
    }
    fails += run_split_fallback_exact("split off full decode fallback", false, 1, 2047, 0, false, 0, 41);
    fails += run_split_fallback_exact("split off swa ring fallback", false, 1, 2047, 128, true, 639, 42);

    // ---- split-sequence, part 2: gate ON.  Contexts 128/512/2048/8192 at rows 1/4/8 across
    //       both API variants, both layer kinds (G16 no-sink full, G8 sink SWA), rings, and
    //       the supported dim tails; the sink must join exactly once at the merge, all-masked
    //       rows stay zero, and the K/V caches must come back untouched.
    if (!set_split_gate("1")) {
        std::puts("setting split gate on failed");
        return 1;
    }
    // ctx 128 (forced splits: the heuristic correctly declines a 128-key range)
    fails += run_attn_split("split full dec p128 T1", false, 1, {127}, {128}, 0, 64, 4, 192, 128,
                            200, 0, false, 1e7, 51, 0, 0, 4);
    // ctx 512, prefill T4 (heuristic splits: 512 keys -> 4)
    fails += run_attn_split("split full pre p512 T4", true, 4, {}, {}, 508, 64, 4, 192, 128, 600, 0,
                            false, 1e7, 52);
    // ctx 2048, prefill T8 (heuristic splits: 16)
    fails += run_attn_split("split full pre p2048 T8", true, 8, {}, {}, 2040, 64, 4, 192, 128, 2200,
                            0, false, 1e7, 53);
    // ctx 8192: decode T1 (heuristic splits: 64) and prefill T4 (32) - the audit's hot shapes
    fails += run_attn_split("split full dec p8192 T1", false, 1, {8191}, {8192}, 0, 64, 4, 192, 128,
                            8300, 0, false, 1e7, 54);
    fails += run_attn_split("split full pre p8192 T4", true, 4, {}, {}, 8188, 64, 4, 192, 128, 8300,
                            0, false, 1e7, 55);
    // causal per-row lengths (verify shape): a row capped by seq_len leaves empty splits
    fails += run_attn_split("split full dec varlen T8", false, 8,
                            {999, 1000, 1001, 1002, 1003, 1004, 1010, 1008},
                            {1000, 1001, 1002, 1003, 1004, 1005, 1011, 1007}, 0, 64, 4, 192, 128,
                            1200, 0, false, 1e7, 56, 0, 0, 8);
    // SWA flat: window + sinks + split merge, incl. the sink-only all-masked row (p=1007,
    // len=3 -> output exactly 0) and window-edge rows (p=30 clamp, p=128 first drop)
    fails += run_attn_split("split swa dec w128 sinks", false, 8,
                            {30, 127, 128, 129, 500, 1000, 1006, 1007},
                            {31, 128, 129, 130, 501, 1001, 1007, 3}, 0, 64, 8, 192, 128, 1100, 128,
                            true, 1e4, 57, 0, 0, 2);
    // SWA ring: decode T1 and a 512-row prefill chunk that wraps the 639 ring (forced splits
    // - the heuristic declines a 128-key window; this exercises the ring+split+sink merge)
    fails += run_attn_split("split swa ring dec T1", false, 1, {2047}, {2048}, 0, 64, 8, 192, 128,
                            2200, 128, true, 1e4, 58, 639, 0, 2);
    fails += run_attn_split("split swa ring pre wrap", true, 512, {}, {}, 1000, 64, 8, 192, 128,
                            1600, 128, true, 1e4, 59, 639, 0, 2);
    // dim tails the API admits beyond the model's 192/128: qk 128 and v 64
    fails += run_attn_split("split full hd128 T1", false, 1, {511}, {512}, 0, 64, 4, 128, 128, 600,
                            0, false, 1e7, 60, 0, 0, 4);
    fails += run_attn_split("split full hdv64 T1", false, 1, {511}, {512}, 0, 64, 4, 192, 64, 600,
                            0, false, 1e7, 61, 0, 0, 4);
    // a second GQA geometry (G4) rides the same template set
    fails += run_attn_split("split full g4 T1", false, 1, {257}, {258}, 0, 64, 16, 192, 128, 400, 0,
                            false, 1e7, 62, 0, 0, 4);
    // keys_hint UNDERESTIMATE: a quarter of the real keys - the last split must absorb the
    // tail so no key is dropped (the hint is a perf knob, never a correctness one)
    fails += run_attn_split("split full hint low T1", false, 1, {2047}, {2048}, 0, 64, 4, 192, 128,
                            2200, 0, false, 1e7, 63, /*ring=*/0, /*keys_hint=*/512,
                            /*splits_hint=*/4);
    // hd_v < 128 with the cache EXACTLY seq-sized (the V-tail guard: lanes >= hd_v must not
    // read past the last (row, kv head); the oracle checks the numerics. No GPU memory
    // sanitizer was available. Covers the tail row at hd_v 1, 31 and 64.
    fails += run_attn_split("split full hdv1 tail T1", false, 1, {255}, {256}, 0, 64, 4, 192, 1,
                            256, 0, false, 1e7, 64, 0, 0, 2);
    fails += run_attn_split("split full hdv31 tail T1", false, 1, {255}, {256}, 0, 64, 4, 192, 31,
                            256, 0, false, 1e7, 65, 0, 0, 2);
    fails += run_attn_split("split full hdv64 tail T1", false, 1, {255}, {256}, 0, 64, 4, 192, 64,
                            256, 0, false, 1e7, 66, 0, 0, 2);
    // the single-scan kernels get the same guard: same shapes through the PLAIN entries
    fails += run_attn("scan full hdv1 tail T1", false, 1, {255}, {256}, 0, 64, 4, 192, 1, 256, 0,
                      false, 1e7, 67);
    fails += run_attn("scan full hdv31 tail T1", false, 1, {255}, {256}, 0, 64, 4, 192, 31, 256, 0,
                      false, 1e7, 68);
    // ring wrap at hd_v 64: the guard holds through the ring modulo path too
    fails += run_attn_split("split swa ring hdv64 T1", false, 1, {899}, {900}, 0, 64, 8, 192, 64,
                            1100, 128, true, 1e4, 69, 256, 0, 2);

    std::printf("%s\n", fails ? "FAIL" : "OK");
    return fails ? 1 : 0;
}
