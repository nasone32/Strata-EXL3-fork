// src/mimo/mimo_engine.cu - strata-mimo: the engine path for Xiaomi MiMo v2 (EXL3 dirs whose config.json says
// model_type "mimo_v2"), the THIRD model family beside Flash-Next (src/program/generate.cpp) and Qwen3.6
// (src/qwen35).  notes/MIMO-V2-PORT.md is the spec of record; the contracts are
// include/strata/core/mimo_geometry.hpp (geometry + config reader), include/strata/artifact/mimo_pack.hpp
// (the dense tensors, reconstructed and uploaded by mimo_pack_build) and include/strata/kernels/mimo_attn.hpp
// (attention), plus this tree's own runtime-width rmsnorm / F16 GEMM / router kernels.
//
// PHASE B (this file): the full greedy forward, correct first - no graphs, no speculation, one stream.
//
//   * the layer driver is a PLAIN pre-norm residual (NO hyper-connections): the residual is n_embd wide in
//     F32,  x += attn(rms(x));  x += mlp(rms(x)),  the norm kernels::native_gr_rms_norm_weighted (runtime
//     width, llama.cpp's arithmetic) with eps and the weights from the pack;
//   * attention per layer kind from the geometry: kv heads 4 (full) / 8 (SWA), rope theta 1e7 / 1e4 on the
//     first n_rot channels, sinks on the SWA layers, window g.sliding_window (0 = full causal), v trimmed to
//     head_dim_v lanes;  K rows land in the flat [capacity][kv][192] F16 cache, V rows in [capacity][kv][128];
//   * layer 0 is a dense SwiGLU (intermediate from the config);  layers 1..47 are MoE: 256 experts, top-8,
//     NO shared expert - kernels::mimo_router_route (sigmoid + noaux_tc + renormalised combine) then the
//     EXL3 expert machinery (Exl3ExpertStream opened with the `model.layers.` prefix, what
//     strata::mimo_expert_tensor spells); decode goes through the stream's host-IO run(), a prompt chunk
//     through run_device() (grouped by expert, one staging per distinct expert per layer);
//   * the projections are the pack's F16 tensors through Gemm::f16 (F16 in, F32 accumulation); the
//     activations are rounded to F16 at exactly the carries the reference in mimo_engine_parity.cpp mirrors
//     (the normed activation, q/k/v into the caches, the attention output into o_proj, silu(g)*u into the
//     down projection);
//   * the lm_head runs on the LAST row of a window only (the pack's F16 lm_head, one Gemm::f16 row), and an
//     in-kernel argmax picks the token;
//   * modes: --tokens IDS [--n N] (greedy, `gen:` line + timings) and --serve (serve/server.py's protocol:
//     READY / GEN / T / DONE / STOP / PP, the qwen35 driver's shape - prefix reuse comes free with a KV
//     cache: a request that extends the held state processes only its fresh tail).
//
// NOT in this build (their modules land separately): MTP drafting and the dflash drafter - their flags error
// cleanly.  A peer-GPU expert tier is a follow-up; the primary GPU only in v1.
#include "strata/artifact/exl3_band8_store.hpp"
#include "strata/artifact/exl3_model.hpp"
#include "strata/artifact/exl3_host_warmup.hpp"
#include "strata/artifact/exl3_resident.hpp"
#include "strata/kernels/exl3_resident_host.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/artifact/mimo_pack.hpp"
#include "strata/core/exl3_flags.hpp"
#include "strata/core/mimo_geometry.hpp"
#include "strata/core/route_digest.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/exl3_experts.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/exl3_peer.hpp"
#include "strata/kernels/exl3_peer_adapt.hpp"
#include "strata/kernels/cpu/exl3_moe.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/mimo_attn.hpp"
#include "strata/kernels/mimo_router.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/mimo/dflash.hpp"
#include "strata/mimo/engine.hpp"
#include "strata/mimo/mtp.hpp"
#include "strata/platform/decode_trace.hpp"
#include "strata/mimo/peer_reserve.hpp"
#include "strata/mimo/window_handoff.hpp"
#include "strata/mimo/window_cpu_plan.hpp"
#include "strata/prefill/gemm.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace K = strata::kernels;
using strata::MimoPack;
inline float k_float_of_f16(uint16_t h) { return K::f32_from_f16(h); }
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "strata-mimo: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__,         \
                         __LINE__);                                                                                    \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

// ------------------------------------------------------------------ the engine's own small kernels
constexpr int kThreads = 256;

/// x[t] = embed[toks[t]] widened to F32 - one block per row.
__global__ void k_embed(const uint16_t* __restrict__ table, const int32_t* __restrict__ toks, float* __restrict__ x,
                        int64_t d) {
    const uint16_t* src = table + (size_t) toks[blockIdx.x] * d;
    float* dst = x + (size_t) blockIdx.x * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) dst[i] = K::f32_from_f16(src[i]);
}

__global__ void k_widen(const uint16_t* input, float* output, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = K::f32_from_f16(input[i]);
}

/// x += y over n F32.
__global__ void k_add(float* __restrict__ x, const float* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

/// h[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * w: the layer norms, out-of-place, runtime width, a SHARED
/// [d] weight (unlike native_gr_rms_norm_weighted, whose gamma is shaped like the data, per row) - one
/// block per row, q36's k_rms shape.
__global__ void k_rmsnorm(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ h,
                          int64_t d, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * d;
    float* hr = h + (size_t) blockIdx.x * d;
    float s = 0.0f;
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) s = fmaf(xr[i], xr[i], s);
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if ((int) threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    const float r = rsqrtf(sh[0] / (float) d + eps);
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) hr[i] = xr[i] * r * w[i];
}

/// h = silu(g) * u in F32 (the trellis down-projection takes the F32 activation; its own f32->f16
/// input conversion is the same rounding the F16 path's carry made).
__global__ void k_silu_mul_f32(const float* __restrict__ g, const float* __restrict__ u,
                               float* __restrict__ h, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    h[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
}

/// h = fp16(silu(g) * u): the dense SwiGLU's carry into the down projection (F32 math, F16 store).
__global__ void k_silu_mul_h(const float* __restrict__ g, const float* __restrict__ u, uint16_t* __restrict__ h,
                             int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = g[i] / (1.0f + expf(-g[i]));
    h[i] = K::f16_from_f32(v * u[i]);
}

/// K cache append: k[pos[r]] = fp16(src[r]) - src rows are [rows][kv*hd_qk] F32.
__global__ void k_scatter_k(const float* __restrict__ src, uint16_t* __restrict__ cache,
                            const int64_t* __restrict__ pos, int kv, int hd, int64_t row_elems) {
    const int r = blockIdx.y, i = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv * hd) return;
    cache[(size_t) pos[r] * row_elems + i] = K::f16_from_f32(src[(size_t) r * kv * hd + i]);
}

/// V cache append: v[pos[r]][h][0..hd_v) = fp16(src[r][h][0..hd_v)) - the padded lanes are dropped here.
__global__ void k_scatter_v(const float* __restrict__ src, uint16_t* __restrict__ cache,
                            const int64_t* __restrict__ pos, int kv, int hd, int hd_v, int64_t row_elems) {
    const int r = blockIdx.y, i = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv * hd_v) return;
    const int h = i / hd_v, lane = i % hd_v;
    cache[(size_t) pos[r] * row_elems + i] = K::f16_from_f32(src[(size_t) r * kv * hd + h * hd + lane]);
}

// SWA ring append. K is already rounded/rotated using ABSOLUTE positions; V drops the
// checkpoint's padded lanes just like k_scatter_v. The ring covers history + the whole chunk.
__global__ void k_scatter_ring_k(const uint16_t* __restrict__ src, uint16_t* __restrict__ cache,
                                const int64_t* __restrict__ pos, int width, int ring) {
    const int r = blockIdx.y, i = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) cache[(size_t) (pos[r] % ring) * width + i] = src[(size_t) r * width + i];
}

__global__ void k_scatter_ring_v(const float* __restrict__ src, uint16_t* __restrict__ cache,
                                const int64_t* __restrict__ pos, int kv, int hd, int hd_v, int ring) {
    const int r = blockIdx.y, i = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv * hd_v)
        cache[(size_t) (pos[r] % ring) * kv * hd_v + i] =
            K::f16_from_f32(src[(size_t) r * kv * hd + (i / hd_v) * hd + i % hd_v]);
}

/// x[t] += sum_j w[t,k+j] * parts[t*k+j] - the MoE combine, F32 accumulation in pick order.
__global__ void k_moe_combine(float* __restrict__ x, const float* __restrict__ parts,
                              const float* w, int rows, int k, int64_t d) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= d || t >= rows) return;
    const float* wr = w + (size_t) t * k;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) acc = fmaf(wr[j], parts[((size_t) t * k + j) * d + i], acc);
    x[(size_t) t * d + i] += acc;
}

/// The peer prompt share's combine: the local rows keep their pick-order accumulation and the
/// peer's ONE WEIGHTED SUM PER TOKEN (computed on its card, router weights already folded)
/// joins once - the masked routes' parts rows are stale and skipped. Same shape as prefill.cpp's
/// moe_combine_peer: NOT bit-identical to the single-GPU prompt's rounding (the exactness class
/// the Qwen peer-prefill documents); decode windows and SHORT chunks (see the T <= 15 arm in
/// forward) never take this path.
__global__ void k_moe_combine_peer(float* __restrict__ x, const float* __restrict__ parts,
                                   const float* __restrict__ sums, const float* __restrict__ w,
                                   const uint8_t* __restrict__ mask, int rows, int k, int64_t d) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= d || t >= rows) return;
    const float* wr = w + (size_t) t * k;
    const uint8_t* mr = mask + (size_t) t * k;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j)
        if (mr[j] == 0) acc = fmaf(wr[j], parts[((size_t) t * k + j) * d + i], acc);
    x[(size_t) t * d + i] += acc + sums[(size_t) t * d + i];
}

/// parts[r] = src[r] for the masked (peer) routes of a SHORT window: the peer's unweighted rows
/// came back to the host (finish()) and land in the device parts slab at their destinations
/// before the ordinary combine - the same kernel family and summation order the single-GPU
/// prompt used, so a short window's peer split is bit-identical to the primary-only run.
__global__ void k_copy_masked_rows(float* __restrict__ parts, const float* __restrict__ src,
                                   const uint8_t* __restrict__ mask, int64_t rows, int64_t d) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * d) return;
    if (mask[i / d] != 0) parts[i] = src[i];
}

__global__ void k_scatter_primary_rows(float* __restrict__ parts, const float* __restrict__ src,
                                      const int32_t* __restrict__ routes, int rows, int64_t d) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t) rows * d) parts[(int64_t) routes[i / d] * d + i % d] = src[i];
}

/// argmax of one F32 row (n <= a few hundred thousand): one block, strided scan, warp + block tree.
__global__ void k_argmax(const float* __restrict__ v, int64_t n, int32_t* __restrict__ out) {
    __shared__ float best[32];
    __shared__ int32_t idx[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    float bv = -INFINITY;
    int64_t bi = -1;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        if (v[i] > bv || (v[i] == bv && i < bi)) {   // ties: the LOWER token id, like the sampler
            bv = v[i];
            bi = i;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int32_t oi = __shfl_xor_sync(0xffffffffu, (int32_t) bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { best[warp] = bv; idx[warp] = (int32_t) bi; }
    __syncthreads();
    if (warp == 0) {
        bv = lane < nw ? best[lane] : -INFINITY;
        bi = lane < nw ? idx[lane] : -1;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int32_t oi = __shfl_xor_sync(0xffffffffu, (int32_t) bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) *out = (int32_t) bi;
    }
}

/// dst[r] = embed[toks[r]] as F16 rows (the diag-embed check's gather, no widening).
__global__ void k_embed_rows16(const uint16_t* __restrict__ table, const int32_t* __restrict__ toks,
                               uint16_t* __restrict__ x, int64_t d) {
    const uint16_t* src = table + (size_t) toks[blockIdx.x] * d;
    uint16_t* dst = x + (size_t) blockIdx.x * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) dst[i] = src[i];
}

/// argmax per row over an [rows][n] F32 block (the diag-embed / diag-tf scans).
__global__ void k_argmax_rows(const float* __restrict__ v, int64_t n, int32_t* __restrict__ out) {
    const int64_t row = blockIdx.y;
    const float* r = v + row * n;
    __shared__ float best[32];
    __shared__ int32_t idx[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    float bv = -INFINITY;
    int64_t bi = -1;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
        if (r[i] > bv || (r[i] == bv && i < bi)) { bv = r[i]; bi = i; }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int32_t oi = __shfl_xor_sync(0xffffffffu, (int32_t)bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { best[warp] = bv; idx[warp] = (int32_t)bi; }
    __syncthreads();
    if (warp == 0) {
        bv = lane < nw ? best[lane] : -INFINITY;
        bi = lane < nw ? idx[lane] : -1;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int32_t oi = __shfl_xor_sync(0xffffffffu, (int32_t)bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) out[row] = (int32_t)bi;
    }
}

/// GPT-J interleaved rope: pairs (2i, 2i+1) within [0, n_rot) - the A/B arm of the pairing question
/// (the default kernels::mimo_rope_f16 is NEOX half-split, pairs (i, i + n_rot/2)).
__global__ void k_rope_gptj(uint16_t* __restrict__ qk, const int64_t* __restrict__ positions, int rows,
                            int n_heads, int head_dim, int n_rot, double theta) {
    const int half = n_rot / 2;
    const long long g = (long long) blockIdx.x * kThreads + threadIdx.x;
    const int i = (int) (g % half);
    const long long rh = g / half;
    if (rh >= (long long) rows * n_heads) return;
    const int64_t pos = positions[rh / n_heads];
    const double freq = ::pow(theta, -(2.0 * i) / (double) n_rot);
    const double reduced = ::fmod((double) pos * freq, 6.283185307179586476925287);
    float sc, cc;
    sincosf((float) reduced, &sc, &cc);
    uint16_t* row = qk + (size_t) rh * head_dim;
    const float a = K::f32_from_f16(row[2 * i]), b = K::f32_from_f16(row[2 * i + 1]);
    row[2 * i] = K::f16_from_f32(a * cc - b * sc);
    row[2 * i + 1] = K::f16_from_f32(a * sc + b * cc);
}

/// dst[i] = src[i] * scale over n F32 (the sink-scaling A/B arm).
__global__ void k_scale(const float* __restrict__ src, float* __restrict__ dst, int64_t n, float scale) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] * scale;
}

/// The F32 twin of the head permutation (the attention OUT rows and the sinks).
__global__ void k_permute_heads_f32(const float* __restrict__ src, float* __restrict__ dst,
                                    const int32_t* __restrict__ perm, int rows, int heads, int hd) {
    const long long g = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long) rows * heads * hd;
    if (g >= total) return;
    const int lane = (int) (g % hd);
    const int h = (int) (g / hd % heads);
    const int t = (int) (g / (hd * (long long) heads));
    dst[((size_t) t * heads + perm[h]) * hd + lane] = src[((size_t) t * heads + h) * hd + lane];
}

/// Mean/std of (x_device - snap_host) over n F32 - the FFN block's contribution to the residual.
void diag_delta(const float* dev, const std::vector<float>& snap, size_t n, double& mean, double& std) {
    if (n == 0 || snap.size() < n) return;
    std::vector<float> v(n);
    CK(cudaMemcpy(v.data(), dev, n * 4, cudaMemcpyDeviceToHost));
    double m = 0;
    for (size_t i = 0; i < n; ++i) v[i] -= snap[i];
    for (float x : v) m += x;
    m /= (double) n;
    double var = 0;
    for (float x : v) var += (double)(x - m) * (x - m);
    mean = m;
    std = var > 0 ? std::sqrt(var / (double) n) : 0.0;
}

/// Mean/std of a device F32 block, into two doubles (the --diag-tf per-layer stats).
void diag_stats(const float* dev, size_t n, double& mean, double& std) {
    if (n == 0) return;
    std::vector<float> v(n);
    CK(cudaMemcpy(v.data(), dev, n * 4, cudaMemcpyDeviceToHost));
    double m = 0;
    for (float x : v) m += x;
    m /= (double) n;
    double var = 0;
    for (float x : v) var += (double)(x - m) * (x - m);
    mean = m;
    std = var > 0 ? std::sqrt(var / (double) n) : 0.0;
}

/// Head permutation for the GQA-mapping A/B: dst[t][perm[h]] = src[t][h] (F16 rows of heads).
__global__ void k_permute_heads(const uint16_t* __restrict__ src, uint16_t* __restrict__ dst,
                                const int32_t* __restrict__ perm, int rows, int heads, int hd) {
    const long long g = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long) rows * heads * hd;
    if (g >= total) return;
    const int lane16 = (int) (g % hd);
    const int h = (int) (g / hd % heads);
    const int t = (int) (g / (hd * (long long) heads));
    dst[((size_t) t * heads + perm[h]) * hd + lane16] = src[((size_t) t * heads + h) * hd + lane16];
}

// ------------------------------------------------------------------ options / sampling / parsing
struct Options {
    std::string model_dir;
    int64_t ctx = 16384;        // notes §2: KV 3.4 GiB at this default
    int chunk = 512;            // the prompt chunk (0 = auto: 512)
    int n_gen = 64, mtp = 0;    // --mtp N: draft with N of the 3 MTP heads (0 = off)
    double expert_vram = -1.0;  // GiB of expert slots; < 0 = none (pure streaming)
    int64_t expert_ram = 0;     // MiB of host expert cache
    bool serve = false, diag_embed = false, diag_tf = false, warmup = false, dense_f16 = false;
    bool dense_q8 = false;   // --dense-q8: the reconstructed dense set quantized to GGUF q8_0 at load
    bool cpu_tier = false;   // --cpu-tier: the CPU-assist tier for cold experts (below)
    int peer_device = -1;
    unsigned long long seed = 1234;   // the benchmark protocol's pin (greedy ignores it; serve's
                                       // GEN seed= key and any sampled A/B take it)
    std::string expert_profile, expert_profile_save;
    std::string dflash_dir;
    int dflash_tokens = 4;
    std::string tokens;
};

struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 1234;   // the benchmark protocol's pinned seed (GEN seed= overrides)
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};

/// serve/server.py's sampler (the qwen35 driver's, kept token-for-token so requests behave the same).
int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize((size_t) kk);
    std::vector<double> p(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
        cum += p[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i)
        if ((r -= p[i]) <= 0) return idx[i];
    return idx[keep - 1];
}

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    while (*s) {
        while (*s == ',' || *s == ' ') ++s;
        if (!*s) break;
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) break;
        v.push_back((int32_t) x);
        s = e;
    }
    return v;
}

bool is_eos(int t) { return t == 151645; }   // <|im_end|>; the tokenizer pack's eos for this family

/// config.json as text (the geometry reader takes the text; mimo_detect_config does its own read).
std::string read_text(const std::string& path, std::string& error) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        error = "cannot open " + path;
        return {};
    }
    std::string text;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
    const bool bad = std::ferror(f) != 0;
    std::fclose(f);
    if (bad) {
        error = "cannot read " + path;
        return {};
    }
    return text;
}

// ------------------------------------------------------------------ the engine
struct Engine {
    strata::core::MimoGeometry g;
    std::unique_ptr<MimoPack> pack;
    /// STRATA_MIMO_BAND8=1 (opt-in; this phase): the shared PACKED routed-expert backing (an
    /// Exl3Band8Store from load_engine) every expert consumer reads - the GPU expert stream,
    /// the CPU tier's model, the peer tier's pmodel and the adapter's refill model.  Declared
    /// BEFORE all of them so, as the engine's own handle, it outlives their destruction (each
    /// consumer ALSO keeps a shared copy through its Exl3Model, so the store's mmap cannot
    /// close under any pointer it handed out).  nullptr (the default) is the byte-identical
    /// native path of every previous build.
    std::shared_ptr<const strata::Exl3ExpertBacking> expert_backing;
    std::shared_ptr<strata::Exl3ResidentBacking> resident_backing;
    bool resident_peer = false;
    /// --peer-device N: the Qwen family's EXL3 "peer mode" (src/kernels/exl3_peer.cpp, the
    /// structure src/program/generate.cpp's Exl3Drive drives end to end) ported to this engine.
    /// An OWNERSHIP table splits every MoE layer's 256 experts between this GPU and GPU N: the
    /// peer holds a RESIDENT subset (its card's free VRAM past its permanent reserve - 4096 MiB
    /// by default, STRATA_MIMO_PEER_RESERVE_MIB overrides; ~2900 slots at this model's
    /// ~6.8 MiB/expert, warm-filled from the route profile - the
    /// durable hot tier), ships its weight planes over the bus exactly once at open() and then
    /// only ever exchanges ACTIVATIONS. Decode (one token): the peer-owned rows LAUNCH first
    /// (enqueue-only, PEER_SPARSE discipline - only routed rows travel), the primary stream
    /// computes the compact complement with its blocking host-IO run() while the peer's card
    /// computes (PEER_CONVERT staging, PEER_PACK_PLAN packing - the default-on machinery of the
    /// Qwen path), finish() lands the peer's unweighted rows at their route destinations and the
    /// unchanged k_moe_combine applies the router weights. Prompt chunks (and MTP verify
    /// windows): the peer's experts run as its grouped-FFN prompt share (launch_prefill) - ONE
    /// WEIGHTED SUM PER TOKEN comes back, sized for the peer's narrow upstream - while
    /// run_device(skip) drops those experts from the primary plan so their weights are never
    /// read; k_moe_combine_peer joins the sums. The primary stream EXCLUDES the peer's experts
    /// (set_owned_elsewhere), so its arena holds only the complement and never double-caches.
    /// Ownership comes from the profile's ranked pairs (a hot-share interleave spreads the top
    /// ranks over both cards, STRATA_PEER_HOT's idea; STRATA_MIMO_PEER_HOT overrides the share,
    /// 0 = the peer takes strictly the top ranks); without a profile a deterministic round-robin
    /// covers every layer. STRATA_MIMO_PEER_ADAPT=1 optionally adapts that subset between
    /// drained windows with bounded uploads. This REPLACES the naive layer-split peer (upper half of the layers on a second
    /// stream) that measured a NET LOSS here (5.7 vs 7.6 tok/s decode: every peer-owned layer
    /// serialized a host-IO round trip while the primary sat idle) - the expert-level split
    /// keeps both cards computing on every layer.
    std::unique_ptr<K::Exl3Peer> peer;
    std::unique_ptr<K::Exl3PeerOutput> peer_output;   // opt-in, stream-ordered peer row collection
    /// [layer * n_expert + expert]: 1 = the peer's. Declared BEFORE the stream it excludes: the
    /// table must outlive it (set_owned_elsewhere's contract).
    std::vector<uint8_t> peer_owned;
    std::unique_ptr<K::Exl3ExpertStream> experts;
    std::unique_ptr<K::Exl3PeerAdapt> peer_adapt;
    bool cache_statistics = false, route_decode = false;
    /// Opt-in experiment boundary evidence (2026-11-01 freeze experiment): when either
    /// switch below is set, every completed serve request closes with an UNTIMED boundary
    /// that (1) drains all in-flight peer refills - in BOTH arms, so the landing is never
    /// part of a measured region and never asymmetric, (2) optionally freezes adaptation
    /// (no further planning/submission; poll/observe continue), and (3) prints one stderr
    /// line with order-invariant residency digests (primary and peer) plus cumulative
    /// adapter counters, so independent warmups can be COMPARED for equality instead of
    /// assumed equal. The serve loop is strictly serial: the boundary completes before the
    /// next GEN is dequeued and before that request's first timestamp.
    bool adapt_boundary = false;
    long long adapt_freeze_after = -1;      // request index (0-based, completed count)
    std::string adapt_dump_path;            // freeze writes the frozen peer set here (never overwritten)
    struct RouteStats { uint64_t primary = 0, peer = 0, nonresident = 0; } cache_routes[2];
    FILE* cache_trace = nullptr;
    bool cache_trace_start = true;
    uint64_t cache_trace_segment = 0;
    int peer_device = -1;
    std::string expert_profile, expert_profile_save;
    strata::prefill::Gemm gemm;
    cudaStream_t st = nullptr;
    int64_t cap = 0, pos = 0;
    int chunk = 512, topk = 0, max_kv = 0;
    double sm_scale = 0;
    std::vector<uint16_t*> kc, vc;
    int swa_ring = 0;                 // STRATA_MIMO_SWA_RING=1: physical SWA rows (0 = flat)
    uint16_t* ring_k16 = nullptr;     // contiguous K chunk for absolute RoPE before ring scatter
    // opt-in split-sequence attention (STRATA_MIMO_ATTN_SPLIT=1): the pre-reserved online-
    // softmax partials workspace + the init-time gate snapshot.  attn_split false (default)
    // keeps every attention call on the original kernels - no getenv in the forward path.
    bool attn_split = false;
    K::MimoAttnSplitWs attn_ws{};

    // the window buffers (sized by the chunk; decode reuses the first rows)
    float *x = nullptr, *h = nullptr, *qf = nullptr, *kf = nullptr, *vf = nullptr, *attn = nullptr, *proj = nullptr;
    float *gg = nullptr, *uu = nullptr, *parts = nullptr, *logits = nullptr;
    uint16_t *h16 = nullptr, *q16 = nullptr, *a16 = nullptr, *hb = nullptr;
    int32_t *tok = nullptr, *ids = nullptr, *arg = nullptr;
    int64_t* dpos = nullptr;
    int* dseq = nullptr;
    float* rw = nullptr;
    // pinned staging for the expert stream's host IO
    float* h_host = nullptr;
    int32_t* ids_host = nullptr;
    float* parts_host = nullptr;
    std::vector<int64_t> pos_stage;

    // ---- the peer tier's transport (allocated by setup_peer; null when the tier is off)
    int primary_slots = 0;               // the primary arena's slots per layer (the hot-share split)
    int64_t moe_layer_count = 0;         // its layer count (the total-capacity product)
    float* peer_comp_host = nullptr;     // pinned [topk][n_embd]: run()'s compact complement rows
    uint16_t* peer_x16_host = nullptr;   // mapped+portable [chunk][n_embd]: the prompt share's F16 image
    float* peer_w_host = nullptr;        // pinned [chunk][topk]: the prompt share's router weights
    float* peer_sums_dev = nullptr;      // device [chunk][n_embd]: the peer's per-token weighted sums
    uint8_t* peer_mask_dev = nullptr;    // device [chunk][topk]: the per-route peer mask (combine)
    float* peer_h_host = nullptr;        // pinned [16][n_embd]: a SHORT window's F32 activations
    float* peer_rows_host = nullptr;     // pinned [16*topk][n_embd]: the peer's unweighted rows back
    float* peer_rows_dev = nullptr;      // device [16*topk][n_embd]: their staging for the gather
    std::vector<int32_t> peer_kind;      // the launch() route classification (decode: topk; a
                                         // short window: 16*topk routes)
    cudaStream_t peer_xs = nullptr;      // the primary's side stream: x16/weights readback
    cudaEvent_t peer_ev = nullptr;       // m.st reached this layer's activations/weights
    std::vector<uint8_t> peer_mask;      // host [chunk][topk]: the split (run_device's skip)
    // the breakdown surface (the Qwen Exl3Drive counters' shape; printed at shutdown): decode
    // rows by tier and window mix, per-stage HOST spans (the peer's launch only enqueues - its
    // GPU work overlaps the primary service and its tail hides in finish's wait), prompt rows
    // by tier, and the bytes that crossed the peer's upstream.
    int64_t peer_rows = 0, peer_win_all = 0, peer_win_mixed = 0, peer_win_zero = 0;
    int64_t peer_windows = 0, peer_calls = 0, peer_fallbacks = 0;
    int64_t peer_primary_rows = 0, peer_primary_miss_rows = 0;
    int64_t peer_pf_layers = 0, peer_pf_rows = 0, peer_pf_local_rows = 0, peer_pf_tokens = 0;
    double peer_launch_ms = 0, peer_primary_ms = 0, peer_finish_ms = 0;
    double peer_pf_launch_ms = 0, peer_pf_primary_ms = 0, peer_pf_finish_ms = 0;
    double peer_bytes_up = 0, peer_bytes_down = 0;

    // STRATA_MIMO_SERVICE_TIME=1 (opt-in diagnostics, 2026-11-01 profiling follow-up):
    // HOST-ELAPSED decomposition of the mapped decode handoff's per-layer service on the
    // servicing thread - the spans whose sum is the producer's decode_wait content. Zero
    // reads of the clock when the env is unset (the plain calls remain). Spans nest inside
    // peer_launch/primary/finish above like this: collect -> [launch] -> class -> [cpu_bg,
    // stage, cpu_join, tail] (the primary bracket) -> [finwait, fintail] (finish) -> done.
    bool svc_time = false;
    long long svc_calls = 0, svc_collect_calls = 0;
    double svc_collect_ms = 0, svc_class_ms = 0, svc_cpu_bg_ms = 0, svc_stage_ms = 0,
           svc_cpu_join_ms = 0, svc_tail_ms = 0, svc_finwait_ms = 0, svc_fintail_ms = 0,
           svc_done_ms = 0;

    // STRATA_MIMO_MTP_TIME=1 (opt-in diagnostics, 2026-11-09 speculation investigation):
    // HOST-ELAPSED per-round MTP phase split, the DFlash request line's shape - draft = the
    // module chain (6 head forwards + 3 lm_head picks + its readback sync), verify =
    // verify_window (the k+1-row target forward through the mapped window handoff + the picks
    // readback sync), commit = the cells upload, the round's second full drain, rewind and
    // catch_up (the module's K/V append for the accepted drafts + the bonus).  Zero clock
    // reads when the env is unset; no code path depends on the flag.
    bool mtp_time = false;
    double mtp_draft_ms = 0, mtp_verify_ms = 0, mtp_commit_ms = 0;
    // STRATA_MIMO_DFTRACE=1 (opt-in diagnostics, 2026-11-09 DFlash verify profiling): ROCTX
    // ranges around each DFlash round phase (draft / verify / commit / adaptive serial
    // fallback / whole round) through decode_trace.hpp's WEAK symbols - inert unless the
    // ROCTX library is preloaded under a marker-aware profiler (rocprofv3 --marker-trace);
    // zero calls, no allocation when the env is unset.
    bool df_trace = false;
    /// One opt-in ROCTX range (STRATA_MIMO_DFTRACE=1).  PORTABILITY: the weak roctx
    /// symbols are declared by decode_trace.hpp ONLY under __linux__ && __GNUC__; this
    /// wrapper carries the same guard so non-Linux/non-GNU builds see a no-op type and
    /// never reference the symbols (a compile error there) even with the flag set.
    /// `active` is tracked separately from the id: roctxRangeStartA may legitimately
    /// return 0, so id != 0 must not be the "is live" test.
    struct DfRange {
        uint64_t id = 0;
        bool active = false;
        DfRange(const char* name, bool on) {
#if defined(__linux__) && defined(__GNUC__)
            if (on && roctxRangeStartA) {
                id = roctxRangeStartA(name);
                active = true;
            }
#else
            (void) name; (void) on;
#endif
        }
        DfRange(const DfRange&) = delete;
        DfRange& operator=(const DfRange&) = delete;
        void stop() {
#if defined(__linux__) && defined(__GNUC__)
            if (active && roctxRangeStop) roctxRangeStop(id);
#endif
            active = false;
        }
        ~DfRange() { stop(); }
    };
    // STRATA_MIMO_MTP_TIME=1 extras: the UNCAPPED per-position match histogram (position j
    // matched because the whole prefix matched - picks[j] == drafts[j], whatever the k-1
    // emission cap does with it), the rounds that ended in a full chain match (cap_bound),
    // and the POTENTIAL extra-bonus opportunities those rounds represent (cap_lost_bonus:
    // a full match emits k tokens - the k-th matched draft lands in the bonus slot - where
    // an uncapped scheme could also emit the picks[k] bonus; an opportunity, not a
    // guaranteed token - the request budget or EOS can suppress it).  The picks and drafts
    // are already on the host in mtp_round; no new GPU work.
    std::vector<int64_t> raw_match_hist;
    int64_t cap_bound_rounds = 0, cap_lost_bonus = 0;

    uint16_t* x16_scratch = nullptr;   // the F16 route's activation image, [chunk][max input width]
    float* silu_f32 = nullptr;         // the trellis MLP's silu(g)*u (F32), [chunk][dense_inter]
    void* q8_scratch = nullptr;        // --dense-q8: the MMVQ q8_1 activation image (8 columns wide)
    void* q8_wide_scratch = nullptr;   // STRATA_MIMO_Q8_PREFILL_COLS=16|32: the wide tiles' q8_1 image
                                       // (the widest input x 32 columns), allocated BEFORE the expert
                                       // arena sizes itself from free VRAM (the split-attention rule);
                                       // null (and the flag inert) unless P2 is on with q8 views

    // P1 STRATA_MIMO_QKV_SHARE=1 (2026-11-05 dense design): the q|k|v projections share ONE q8_1
    // quantization of the same rmsnorm input and, when the library allows it, ONE native_mmvq_bunch
    // launch per <= 8-row tile - bitwise the three sequential calls (the library's own contract;
    // native_mmvq_bunch_supported is the arch/layout gate, e.g. it says no on a gfx906 wave build).
    // P2 STRATA_MIMO_Q8_PREFILL_COLS=16|32 (default/unset 8): prefill tiles T > 8 run the wide
    // exact-layout MMVQ (one weight traversal per 16/32 tokens), the <= 8 remainder/decode/verify
    // keep today's path; every refusal falls back to the <= 8 launches for the SAME rows.
    bool qkv_share = false;
    int q8_wide_cols = 8;
    bool q8_wide_active = false;       // the preflight: cols > 8 AND the scratch AND the library's
                                       // pure eligibility query all said yes (a flag-on-unsupported
                                       // build never enqueues the unused wide quantize at all)

    /// y[rows][n_out] = W[.] . x[rows][n_in]: the F16 pack path (cublas on the reconstructed weights,
    /// x rounded to its F16 image), the Q8 path (--dense-q8: the pack's GGUF q8_0 blocks through
    /// kernels::native_q8_0_f32, the MMVQ dense-Q8_0 GEMV - x quantized to q8_1 in `q8_scratch`,
    /// 1..8 columns per launch so decode and every window up to 8 rows read the weights once) or the
    /// TRELLIS path - exl3_gemv_f32 at one row, the gfx11 multirow kernel in 8-row tiles above (a
    /// tile's row is bit-identical to a 1-row call, so decode and prefill agree).  `w16`/`wq8` null
    /// selects the next view; the q8 view wins over a present F16 pointer (the --mtp F16 lm_head
    /// oracle rides beside it, the engine's own decode stays q8).
    void linear(const void* w16, const void* wq8, const strata::MimoTrellis& t, const float* x, float* y,
              int rows, int64_t n_out, int64_t n_in) {
        if (wq8 != nullptr) {
            // P2 (STRATA_MIMO_Q8_PREFILL_COLS=16|32, prefill only - rows > 8): exact-width wide tiles
            // first, one weight traversal per tile; any refusal (process-stable, never data-dependent)
            // drops this and every later row range onto the <= 8 loop below, which recomputes those
            // rows byte-identically.  Decode (rows 1) and verify windows (rows <= 8) never reach this.
            int r0 = 0;
            if (q8_wide_active && rows > 8)
                for (; r0 + q8_wide_cols <= rows; r0 += q8_wide_cols)
                    if (!K::native_q8_0_f32_wide(wq8, x + (int64_t) r0 * n_in, q8_wide_scratch,
                                                 y + (int64_t) r0 * n_out, (int) n_in, (int) n_out,
                                                 q8_wide_cols, st))
                        break;
            for (int r = r0; r < rows; r += 8) {
                const int nc = std::min(8, rows - r);
                K::native_q8_0_f32(wq8, x + (int64_t) r * n_in, q8_scratch, y + (int64_t) r * n_out,
                                   (int) n_in, (int) n_out, nc, st);
            }
            return;
        }
        if (w16 != nullptr) {
            K::f32_to_f16_bulk(x, x16_scratch, (int64_t) rows * n_in, st);
            gemm.f16(x16_scratch, reinterpret_cast<const uint16_t*>(w16), y, rows, n_out, n_in, 0, 0.0f);
            return;
        }
        if (rows == 1 || t.frac != 0) {
            // one row per call: the M = 1 folded GEMV (frac-aware - the dense multirow kernel takes
            // integer K only, and the real model's dense tensors are all integer K; the fallback
            // keeps fractional-K fixtures on the trellis path too)
            for (int r = 0; r < rows; ++r)
                K::exl3_gemv_f32(x + (int64_t) r * n_in, reinterpret_cast<const uint16_t*>(t.suh),
                                 reinterpret_cast<const uint16_t*>(t.svh),
                                 reinterpret_cast<const uint16_t*>(t.trellis), t.ki, t.nj, t.bits, t.cb,
                                 y + (int64_t) r * n_out, st, t.frac);
            return;
        }
        for (int r = 0; r < rows; r += 8) {
            const int m = std::min(8, rows - r);
            K::exl3_dense_gemv_multi(0, x + (int64_t) r * n_in, n_in,
                                     reinterpret_cast<const uint16_t*>(t.trellis),
                                     reinterpret_cast<const uint16_t*>(t.suh),
                                     reinterpret_cast<const uint16_t*>(t.svh), t.ki, t.nj, t.bits, t.cb,
                                     m, y + (int64_t) r * n_out, n_out, st, t.frac);
        }
    }

    size_t cache_bytes = 0, buffer_bytes = 0;

    // ---- first-light debugging surface (env-read once in init; every knob's default is the port's
    // documented convention, so unset behaviour is exactly the parity-tested engine):
    //   STRATA_MIMO_ROPE    neox (default) | gptj     - the rope pairing inside n_rot
    //   STRATA_MIMO_GQA     block (default) | inter    - q head h -> kv head h/group vs h % n_kv
    //   STRATA_MIMO_THETA   as-config (default) | swap - full/SWA rope thetas swapped
    //   STRATA_MIMO_SINK    raw (default) | scaled     - the sink logit sm_scale'd or not
    bool rope_gptj = false, gqa_interleave = false, theta_swap = false, sink_scaled = false;
    std::vector<int32_t> gqa_perm;              // model head h -> kernel position (interleave arm)
    uint16_t* q_perm_buf = nullptr;             // the permuted q rows (interleave arm)
    float* attn_perm_buf = nullptr;             // the permuted attention out (interleave arm)
    float* gqa_scratch_sinks = nullptr;         // the permuted sinks (interleave arm)
    float* sink_scaled_buf = nullptr;           // per-layer sinks x sm_scale (scaled arm)
    int32_t* diag_ids_host = nullptr;           // pinned: router picks for the diag prints
    float* diag_w_host = nullptr;
    bool diag = false;                          // --diag-tf: collect per-layer stats + all logits
    float* diag_logits = nullptr;               // [T][n_vocab] when diag or a verify window
    int32_t* arg_all = nullptr;                 // per-row argmax of a verify window (device)
    float* route_scores = nullptr;              // [n_expert] F32: the decode router's score scratch
    bool route_mono = false;                    // STRATA_MIMO_ROUTE=mono: the one-block kernel at T=1 too
    // STRATA_MIMO_WINDOW_ROUTER_SPLIT=1: T=2..8 verify windows route through the batched split
    // kernels (bit-identical picks) instead of the one-block-per-row launch.  Default off.
    bool window_router_split = false;
    bool window_router_split_stats_on = false;  // STRATA_MIMO_WINDOW_ROUTER_SPLIT_STATS=1
    float* window_route_scores = nullptr;       // [8, n_expert] F32: the window split's score scratch
    long long window_split_windows = 0;         // forwards whose windows routed through the split
    long long window_split_layers = 0;          // per-window MoE layers routed through the split
    long long win_split_reported_windows = 0;   // the stats line's last-reported marks
    long long win_split_reported_layers = 0;
    bool decode_indexed = false;                // STRATA_MIMO_DECODE_INDEXED=1: packed/fused host-IO at T=1
    bool decode_device = false;                 // STRATA_MIMO_DECODE_DEVICE=1: local activations/results stay on GPU
    bool decode_handoff = false, handoff_ready = false;
    bool window_handoff = false;
    bool handoff_t1_prepared = false;
    bool handoff_cpu = false;
    bool verify_host = false;
    int handoff_cpu_min = 1;
    bool handoff_eager = false, handoff_stop = false, handoff_pending = false, handoff_busy = false;
    uint32_t handoff_ring = 0;
    cudaStream_t handoff_stream = nullptr;
    std::thread handoff_worker;
    std::mutex handoff_mu;
    std::condition_variable handoff_cv;
    std::exception_ptr handoff_error;

    // ---- the CPU-assist tier (--cpu-tier; the Qwen EXL3 CPU tier's exact shape on this engine):
    // with the primary arena FULL (the profile warm-fill leaves it full from token 0), a cold
    // route is REFUSED by the stream - nothing evicted, nothing uploaded - and computed on the
    // host by the ported exllamav3 VNNI kernels straight off the page-cache trellis bytes: no
    // host read -> H2D upload -> wait.  classify() previews the refusal decision, run_rows_bg()
    // publishes the CPU work on the pool's workers, the ordinary run(..., cpu_rows) computes
    // every GPU-served row while they run, and the join lands the CPU rows at their pick-order
    // destinations before the combine stages them.  A CPU failure keeps the rows on the GPU's
    // ordinary evict-and-admit path - the tier must never break generation.  Repeat hits of a
    // refused expert stay on the CPU tier, served from its own repack LRU (RAM speed), the same
    // overflow policy the Qwen tier documented.
    std::unique_ptr<strata::Exl3Model> cpu_model;   // the mmap the raw pointers live in
    std::unique_ptr<K::cpu::Exl3CpuMoe> cpu_moe;
    int64_t cpu_rows_served = 0;
    std::vector<uint8_t> cpu_flag;       // classify()'s per-route refusal preview
    std::vector<int32_t> cpu_ids;        // the refused routes' experts
    std::vector<int> cpu_tok, cpu_row;   // token index (0 at decode) and the dense CPU row
    std::vector<int> cpu_route_row;      // route j -> the CPU row that owes it (-1 = the GPU's)
    std::vector<float> cpu_out;          // [topk][n_embd] F32, pick-order slots

    // ---- the verify-window CPU overflow (STRATA_MIMO_WINDOW_CPU=1, default OFF and
    // independent of STRATA_MIMO_BAND8 - it runs on the native CPU tier and the packed one,
    // so native-vs-band8 attribution stays a clean four-arm comparison).  Planner:
    // include/strata/mimo/window_cpu_plan.hpp; policy and hunks:
    // notes/MIMO-WINDOW-CPU-PLANNER-2026-11-06.md.  Whole cold expert groups of a mapped
    // window (T=2..8, T*k<=64) computed on the CPU tier beside run_multi_complement; the
    // T1 arm above is untouched.  SEPARATE buffers from the T1 arrays on purpose: a window
    // owes up to 64 rows against the T1 tier's topk, and these must never be reused while
    // a background job is in flight.  All allocations happen in init() BEFORE the handoff
    // worker starts; nothing allocates under the service wait.
    bool window_cpu = false;
    int window_cpu_max_groups = 2;       // unique WHOLE groups per window (user 0..4; 2 default)
    int window_cpu_min_groups = 1;       // engagement floor: one cold group beats one upload
    bool window_cpu_stats_on = false;    // STRATA_MIMO_WINDOW_CPU_STATS=1 (every cost branch-gated)
    std::vector<float> win_cpu_out;      // [64][n_embd] F32, dense plan rows (~1 MiB at 4096)
    int32_t win_cpu_ids[64] = {};        // run_rows_bg ids/t/r triple (plan row order)
    int win_cpu_tok[64] = {}, win_cpu_row[64] = {};
    uint8_t win_cpu_skip[64] = {};       // the CPU plan's whole-group mask: written once per
                                         // window before the bg publish, immutable through the
                                         // CPU join and the GPU complement's drain
    std::vector<uint8_t> win_cpu_submit_logged; // once per layer, allocated before worker start
    // ---- the verify-window resident GPU hits (STRATA_MIMO_WINDOW_GPU_HITS=1, default OFF).
    // The mapped window's producer stream computes RESIDENT rows inline from the refresh's
    // descriptor tables (the T1 handoff's mechanism, opt-in extended to T=2..8 windows by
    // Exl3ExpertStream::set_window_graph_hits); the worker resolves only miss/peer/CPU
    // routes. A window whose tables the refresh did NOT declare valid (active cache below
    // T*k) publishes no hits at all - the original all-miss window, byte for byte.
    bool window_gpu_hits = false;
    bool window_gpu_hits_stats_on = false;   // STRATA_MIMO_WINDOW_GPU_HITS_STATS=1
    // ---- STRATA_MIMO_PEER_DEVICE_RESULTS=1 (default OFF; audit R3 prototype, 2026-11-12).
    // The mapped handoff WORKERS (serve_window_routes AND serve_decode_routes) collect the
    // peer tier's unweighted rows through the shared Exl3PeerOutput mapped-F16 frame - the
    // producer-T1 device path's transport, P2P or mapped-host fallback - instead of the
    // established chain (peer F16 rows D2H over x4 -> CPU F16->F32 conversion of every row
    // on the worker thread -> WHOLE-slab F32 H2D, 512 KiB/layer at T=4 with only ~41% of
    // the rows meaningful -> k_copy_masked_rows). The frame's copy kernel converts on the
    // primary GPU and scatters ONLY the peer's rows; the physical GPU1->host->GPU0 hop may
    // be REQUIRED by the x4 chipset topology (P2P not assumed) - the flag removes the CPU
    // conversion and the garbage share of the upload, not the hop. Row writes stay
    // disjoint from the producer's inline hits, the complement's scatter and the CPU
    // overlay's compact upload; the doorbell (enqueued after the copy kernel on the service
    // stream) still orders everything before the producer's combine.
    bool peer_device_results = false;
    // ---- STRATA_MIMO_WINDOW_DEVICE_MISSES=1 (default OFF; audit R4 prototype, 2026-11-12).
    // EVERY mapped verify-window service call (the complement AND the plain no-peer/no-CPU
    // window the host arm served through run_multi) runs DEVICE-input/DEVICE-output
    // (Exl3ExpertStream::run_multi_complement_device): it reads the producer's live `h`
    // activation image in place (no F32 mapped-host publish round trip) and scatters its
    // rows straight into `parts` (no whole-tile D2H + blocking sync + host memcpy scatter +
    // compact re-upload) while the producer spins at decode_wait. FFN arithmetic, plan
    // drops (peer-owned/cpu_skip/graph-hit), admission, staging and the pending-incoming
    // GPU-scratch rule are the host arm's; the pinned/packed plan requirement is refused
    // loudly when absent. On a plain window the drops are provably no-ops (no row is
    // peer-owned or the window would not be plain; no CPU job; graph hits, if armed, drop
    // exactly the rows run_multi's host bracket dropped), so the same call serves it with
    // the same plan arithmetic - only the transport changes.
    bool window_device_misses = false;
    // Engagement/transport counters, always accumulated (printed by peer_summary when the
    // flags armed; the parity harness reads them through the EXPECT test seams):
    int64_t peer_dev_frames = 0;      // frames collected through the device transport
    int64_t peer_dev_rows = 0;        // peer rows converted+scattered on the primary GPU
    double peer_dev_upload_avoided = 0;   // bytes of whole-slab F32 H2D the old path issued
    double peer_dev_convert_avoided = 0;  // rows of CPU F16->F32 conversion the old path ran
    int64_t wdm_calls = 0;            // window complements served by the device arm
    int64_t wdm_routes = 0;           // their GPU rows (misses + pending scratch + hits-none)
    int64_t wdm_cpu_rows = 0;         // CPU-overlay rows kept on the host upload path
    double wdm_bytes_avoided = 0;     // NET host-arm bus bytes removed (see serve_window_routes)
    int64_t peer_dev_zero_windows = 0;   // R3 test seam: window publications that carried ZERO
                                         // peer rows while the flag was on (the no-remote path)
    struct {
        int64_t windows = 0;          // mapped windows served (layers x windows)
        int64_t hit_windows = 0, all_hit_windows = 0, fallback_windows = 0;
        int64_t hit_routes = 0, host_served = 0, peer_routes = 0;
        int64_t simultaneous = 0;     // TEST seam (win_cpu_test_pending set): windows with
                                      // hits>0 AND a live CPU job AND a pending expert the
                                      // GPU complement served - all in the SAME window
    } win_gpu_hits_stats;
    decltype(win_gpu_hits_stats) win_gpu_hits_reported;
    // Test-only controls are set by run_verify_logits, never by the production CLI.
    int win_cpu_test_fault = 0; // 1: before publish; 2: after publish, before primary
                                // service; 3: after publish ONLY on a window that also
                                // carries inline window hits (non-vacuous fault arm)
    bool win_cpu_test_rows = false;   // =1: wiring + retained-GPU + CPU-vs-GPU 3%/12% gate
    int win_test_peer_fault = 0;      // TEST ONLY (STRATA_MIMO_TEST_PEER_FRAME_FAULT, set by
                                      // run_verify_logits): 1 = throw on the worker right
                                      // AFTER a peer launch queued the device frame (in
                                      // flight, no copy yet); 2 = throw right AFTER the
                                      // service-stream wait+copy pair was enqueued (a live
                                      // QUEUED wait kernel through the unwind); 3 = advance
                                      // the frame's expected BETWEEN the launch's signal
                                      // enqueue and the copy enqueue (the queued wait's
                                      // target provably has no signal - no race with the
                                      // kernel's initial read), then enqueue the copy and
                                      // throw. 1-3 are WORKER-ERROR RELEASE tests: the
                                      // unwind through abort+release must drain everything
                                      // without a hang. A stuck frame WITHOUT any worker
                                      // exception (the desync-no-throw case) is
                                      // deliberately NOT a seam: the producer's decode_wait
                                      // has no timeout, so that case is a true mid-forward
                                      // hang - documented as the residual exposure, not
                                      // tested as passable
    bool win_cpu_test_gate = false;   // =2 keeps the two bitwise operator proofs and defers
                                      // the CPU-vs-GPU arithmetic gate to a conditioned,
                                      // real-geometry fixture (the random H128 fixture's
                                      // cancellation makes its relative RMS meaningless -
                                      // measured 4.5% RMS / 7.1% max with wiring bitwise)
    std::vector<float> win_cpu_test_gpu;
    std::vector<float> win_cpu_test_solo;  // TEST ONLY: standalone recompute of the SAME plan
                                           // rows on the SAME tier - a bitwise wiring proof of
                                           // (ids, token, dense-row) that never touches
                                           // win_cpu_out's saved context
    const uint8_t* win_cpu_test_pending = nullptr;
    int win_dump_index = 0;                  // TEST ONLY (STRATA_MIMO_TEST_DUMP_WINDOW): the
                                             // serve_window_routes call counter for the dump seam
    int win_dump_at = -2;                    // TEST ONLY: that env cached on first use (-2 unread,
                                             // -1 off, -3 every window, else the wanted index)
    std::vector<float> win_dump_buf;         // TEST ONLY: its readback scratch  // TEST ONLY (helper-owned table, null in
                                           // production): when set, serve_window_routes
                                           // verifies no selected group is pending-incoming -
                                           // the getter union keeps them OFF the CPU and on
                                           // the GPU's scratch path; the pointer must outlive
                                           // the expert stream (declared before the Engine)
    struct {
        int64_t jobs = 0, groups = 0, rows = 0;  // rows = host-solved rows (coverage proof)
        int64_t bytes_skipped = 0;               // expert bytes kept off the H2D (metadata math)
        double submit_ms = 0.0, join_ms = 0.0;   // submit + residual join wait, NOT compute
    } win_cpu_stats;
    decltype(win_cpu_stats) win_cpu_reported;
    K::cpu::Exl3CpuMoe::Stats win_cpu_reported_runtime{};

    // Main thread only, AFTER the handoff worker is drained. Deltas are labelled per
    // forward (a request can contain multiple forwards), never passed off as request totals.
    void report_window_cpu() {
        if (!window_cpu_stats_on) return;
        const auto s = cpu_moe ? cpu_moe->stats() : K::cpu::Exl3CpuMoe::Stats{};
        std::fprintf(stderr, "strata-mimo: window_cpu forward jobs=%lld groups=%lld rows=%lld "
                             "submit_ms=%.3f residual_join_ms=%.3f runtime_prepare_ms=%.3f "
                             "runtime_join_ms=%.3f repack_fills=%lld cache_bytes=%lld native_reads=%lld "
                             "workers=%d bg_workers=%d\n",
                     (long long)(win_cpu_stats.jobs - win_cpu_reported.jobs),
                     (long long)(win_cpu_stats.groups - win_cpu_reported.groups),
                     (long long)(win_cpu_stats.rows - win_cpu_reported.rows),
                     win_cpu_stats.submit_ms - win_cpu_reported.submit_ms,
                     win_cpu_stats.join_ms - win_cpu_reported.join_ms,
                     s.prepare_ms - win_cpu_reported_runtime.prepare_ms,
                     s.join_ms - win_cpu_reported_runtime.join_ms,
                     (long long)(s.repack_fills - win_cpu_reported_runtime.repack_fills),
                     (long long)s.cache_bytes,
                     (long long)(s.native_reads - win_cpu_reported_runtime.native_reads), s.threads, s.bg_workers);
        win_cpu_reported = win_cpu_stats;
        win_cpu_reported_runtime = s;
    }

    /// STRATA_MIMO_WINDOW_CPU preflight: pure environment validation - no model, no GPU, no
    /// cache writes.  load_engine calls it FIRST, before the pack build and before any
    /// band8 payload byte could be written, so a misconfigured opt-in fails fast and cheap.
    /// The switch itself is exl3_flag semantics ('0' = an explicit OFF that stays out
    /// entirely); the numeric knobs are STRICT whole integers ("2foo" is an error, never
    /// quietly-atoi'd 2).  MAX_GROUPS 0 is the explicit no-op: exact GPU path, no CPU tier
    /// forced (the arm check below skips it), for A/B-ing the branch cost itself.
    bool preflight_window_cpu(std::string& err) {
        if (!strata::core::exl3_flag("STRATA_MIMO_WINDOW_CPU", nullptr, false)) return true;
        window_cpu = true;
        window_cpu_stats_on = strata::core::exl3_flag("STRATA_MIMO_WINDOW_CPU_STATS", nullptr, false);
        auto strict = [&err](const char* name, int lo, int hi, int& out) {
            const char* v = std::getenv(name);
            if (v == nullptr) return true;
            if (*v == '\0') { err = std::string("mimo: ") + name + " must be a whole integer"; return false; }
            for (const char* p = v; *p; ++p)
                if (*p < '0' || *p > '9') {
                    err = std::string("mimo: ") + name + " must be a whole integer";
                    return false;
                }
            char* end = nullptr;
            const long n = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || n < lo || n > hi) {
                err = std::string("mimo: ") + name + " must be a whole integer in [" +
                      std::to_string(lo) + ", " + std::to_string(hi) + "] (got '" + v + "')";
                return false;
            }
            out = (int) n;
            return true;
        };
        if (!strict("STRATA_MIMO_WINDOW_CPU_MAX_GROUPS", 0, 4, window_cpu_max_groups) ||
            !strict("STRATA_MIMO_WINDOW_CPU_MIN_GROUPS", 1, 4, window_cpu_min_groups))
            return false;
        if (window_cpu_max_groups > 0 && window_cpu_min_groups > window_cpu_max_groups) {
            err = "mimo: STRATA_MIMO_WINDOW_CPU_MIN_GROUPS must not exceed STRATA_MIMO_WINDOW_CPU_MAX_GROUPS";
            return false;
        }
        if (window_cpu_max_groups == 0) { window_cpu = false; return true; }
        if (!strata::core::exl3_flag("STRATA_MIMO_WINDOW_HANDOFF", nullptr, false)) {
            err = "mimo: STRATA_MIMO_WINDOW_CPU=1 needs the mapped verify-window handoff: "
                  "set STRATA_MIMO_WINDOW_HANDOFF=1 (an eligible shape T=2..8, k<=64/T is "
                  "checked next, before any request)";
            return false;
        }
        if (strata::core::exl3_flag("STRATA_MIMO_TIME", nullptr, false) ||
            strata::core::exl3_flag("STRATA_MIMO_HANDOFF_EAGER", nullptr, false)) {
            err = "mimo: STRATA_MIMO_WINDOW_CPU=1 needs the async mapped window arm - "
                  "unset STRATA_MIMO_TIME and STRATA_MIMO_HANDOFF_EAGER (the diagnostic "
                  "window branch has no worker to publish CPU work on)";
            return false;
        }
        return true;
    }

    /// STRATA_MIMO_WINDOW_GPU_HITS preflight: pure environment validation, same contract as
    /// preflight_window_cpu (fail fast and cheap, before any pack build or GPU allocation).
    /// The window-hit service rides the SAME persistent worker as the mapped window
    /// handoff, so the switch requires that arm and refuses its diagnostic-only blockers
    /// rather than quietly never engaging.
    bool preflight_window_gpu_hits(std::string& err) {
        if (!strata::core::exl3_flag("STRATA_MIMO_WINDOW_GPU_HITS", nullptr, false)) return true;
        window_gpu_hits = true;
        window_gpu_hits_stats_on =
            strata::core::exl3_flag("STRATA_MIMO_WINDOW_GPU_HITS_STATS", nullptr, false);
        if (!strata::core::exl3_flag("STRATA_MIMO_WINDOW_HANDOFF", nullptr, false)) {
            err = "mimo: STRATA_MIMO_WINDOW_GPU_HITS=1 needs the mapped verify-window "
                  "handoff: set STRATA_MIMO_WINDOW_HANDOFF=1 (an eligible shape T=2..8, "
                  "k<=64/T is checked at init, before any request)";
            return false;
        }
        if (strata::core::exl3_flag("STRATA_MIMO_TIME", nullptr, false) ||
            strata::core::exl3_flag("STRATA_MIMO_HANDOFF_EAGER", nullptr, false)) {
            err = "mimo: STRATA_MIMO_WINDOW_GPU_HITS=1 needs the async mapped window arm - "
                  "unset STRATA_MIMO_TIME and STRATA_MIMO_HANDOFF_EAGER (the diagnostic "
                  "branch publishes no window hits and no worker serves them)";
            return false;
        }
        return true;
    }

    /// STRATA_MIMO_WINDOW_ROUTER_SPLIT preflight: pure environment validation, same contract
    /// as preflight_window_cpu (fail fast and cheap, before any pack build or GPU
    /// allocation).  The arm swaps ONLY the router kernels of the T=2..8 all-rows forwards
    /// (the verify windows) for the decode arm's split kernels batched over the rows -
    /// bit-identical picks by construction - and needs nothing from the window handoff,
    /// peer or CPU machinery, so unlike the window-hit arm there is no dependency to refuse.
    /// Engagement is still never silent: the init banner, the STATS per-forward line, the
    /// destruction total and the STRATA_MIMO_TEST_WINDOW_ROUTER_SPLIT_EXPECT=used test seam
    /// all report the actual split counters.
    bool preflight_window_router_split(std::string& err) {
        (void) err;
        if (!strata::core::exl3_flag("STRATA_MIMO_WINDOW_ROUTER_SPLIT", nullptr, false)) return true;
        window_router_split = true;
        window_router_split_stats_on =
            strata::core::exl3_flag("STRATA_MIMO_WINDOW_ROUTER_SPLIT_STATS", nullptr, false);
        return true;
    }

    /// STRATA_MIMO_PEER_DEVICE_RESULTS preflight: pure environment validation, the same
    /// fail-fast contract as preflight_window_gpu_hits. The arm rides the mapped handoff
    /// WORKER (the T1 arm of serve_decode_routes and/or serve_window_routes), so at least
    /// one handoff must be armed, the diagnostic-only blockers are refused, and the peer
    /// tier must exist (--peer-device; the transport is opened in setup_peer and a failed
    /// open fails the engine loudly - never a silent host fallback while claiming the flag).
    bool preflight_peer_device_results(std::string& err) {
        if (!strata::core::exl3_flag("STRATA_MIMO_PEER_DEVICE_RESULTS", nullptr, false)) return true;
        peer_device_results = true;
        if (peer_device < 0) {
            err = "mimo: STRATA_MIMO_PEER_DEVICE_RESULTS=1 needs the peer tier: pass "
                  "--peer-device <gpu> (the device-row transport collects the peer's "
                  "windows/decode rows through the shared Exl3PeerOutput frame)";
            return false;
        }
        if (!strata::core::exl3_flag("STRATA_MIMO_DECODE_HANDOFF", nullptr, false) &&
            !strata::core::exl3_flag("STRATA_MIMO_WINDOW_HANDOFF", nullptr, false)) {
            err = "mimo: STRATA_MIMO_PEER_DEVICE_RESULTS=1 needs a mapped handoff worker: "
                  "set STRATA_MIMO_DECODE_HANDOFF=1 (T1) and/or STRATA_MIMO_WINDOW_HANDOFF=1 "
                  "(verify windows)";
            return false;
        }
        if (strata::core::exl3_flag("STRATA_MIMO_TIME", nullptr, false) ||
            strata::core::exl3_flag("STRATA_MIMO_HANDOFF_EAGER", nullptr, false)) {
            err = "mimo: STRATA_MIMO_PEER_DEVICE_RESULTS=1 needs the async mapped handoff "
                  "arm - unset STRATA_MIMO_TIME and STRATA_MIMO_HANDOFF_EAGER (the "
                  "diagnostic branch has no worker doorbell to order the frame against)";
            return false;
        }
        return true;
    }

    /// STRATA_MIMO_WINDOW_DEVICE_MISSES preflight: pure environment validation. The arm
    /// lives inside serve_window_routes only, so it needs the mapped window handoff and
    /// refuses the same diagnostic blockers; the packed-plan machinery is checked at init
    /// (the stream exists only there) with a loud refusal, not a silent host fallback.
    bool preflight_window_device_misses(std::string& err) {
        if (!strata::core::exl3_flag("STRATA_MIMO_WINDOW_DEVICE_MISSES", nullptr, false)) return true;
        window_device_misses = true;
        if (!strata::core::exl3_flag("STRATA_MIMO_WINDOW_HANDOFF", nullptr, false)) {
            err = "mimo: STRATA_MIMO_WINDOW_DEVICE_MISSES=1 needs the mapped verify-window "
                  "handoff: set STRATA_MIMO_WINDOW_HANDOFF=1 (an eligible shape T=2..8, "
                  "k<=64/T is checked at init, before any request)";
            return false;
        }
        if (strata::core::exl3_flag("STRATA_MIMO_TIME", nullptr, false) ||
            strata::core::exl3_flag("STRATA_MIMO_HANDOFF_EAGER", nullptr, false)) {
            err = "mimo: STRATA_MIMO_WINDOW_DEVICE_MISSES=1 needs the async mapped window "
                  "arm - unset STRATA_MIMO_TIME and STRATA_MIMO_HANDOFF_EAGER";
            return false;
        }
        return true;
    }

    /// After init_cpu_tier (mimo_main): the window CPU arm must not quietly run without a
    /// CPU.  MAX_GROUPS 0 submits nothing, so it does not force a tier (explicit no-op).
    /// No extra source queries for optional bytes metadata: the planner leaves it at zero.
    bool arm_window_cpu(std::string& err) {
        if (!window_cpu) return true;
        if (window_cpu_max_groups == 0) return true;
        if (cpu_moe == nullptr || cpu_model == nullptr) {
            err = "STRATA_MIMO_WINDOW_CPU=1 (MAX_GROUPS "
                  + std::to_string(window_cpu_max_groups) + ") needs the CPU tier: pass "
                  "--cpu-tier and let it load (the window arm never quietly falls back)";
            return false;
        }
        return true;
    }

    bool activate_resident(std::string& err) {
        if (!resident_backing) return true;
        // The runtime repack (STRATA_MIMO_BAND8=memory) activates EARLY, inside
        // load_engine's resident block and BEFORE E.init's first GPU fill - the arena is
        // then the only packed copy and nothing may load unconverted bytes. This late
        // call site (after setup_peer's fills) must be a no-op there, never a re-activate.
        if (resident_backing->active()) return true;
        try {
            std::vector<std::pair<int,int>> gpu;
            if (resident_peer) {
                if (!peer || !peer->valid()) throw std::runtime_error("resident EXL3: peer is unavailable");
                std::vector<int32_t> slots;
                peer->residency((int)g.n_layers,(int)g.n_expert,slots);
                for (int l = 0; l < g.n_layers; ++l) for (int e = 0; e < g.n_expert; ++e)
                    if (slots[(size_t)l*g.n_expert+e] >= 0) gpu.emplace_back(l,e);
            }
            auto opts = K::exl3_resident_host_options();
            const auto start = Clock::now();
            opts.progress = [start](size_t done,size_t total) {
                std::fprintf(stderr,"strata-mimo: resident load %zu/%zu experts (%.1f s)\n",done,total,ms_since(start)/1000.0);
            };
            std::fprintf(stderr,"strata-mimo: resident EXL3 %s: %.2f GiB required, %.2f GiB available; "
                                "paced pinned RAM, direct startup reads, no file tier after READY\n",
                         resident_peer ? "peer complement" : "full arena",
                         resident_backing->required_bytes(gpu)/1073741824.0,opts.available_bytes/1073741824.0);
            resident_backing->activate(gpu,std::move(opts));
            const auto s = resident_backing->stats();
            std::fprintf(stderr,"strata-mimo: resident EXL3 ready: %.2f GiB locked, %zu GPU-only experts, "
                                "%.1f s; host warmup and file staging bypassed\n",
                         s.locked_bytes/1073741824.0,gpu.size(),ms_since(start)/1000.0);
            return true;
        } catch (const std::exception& ex) { err = ex.what(); return false; }
    }

    bool init_cpu_tier(const std::string& dir, std::string& err) {
        if (!K::cpu::Exl3CpuMoe::supported()) {
            err = "this CPU cannot run the EXL3 kernels";
            return false;
        }
        // The shared backing (when band8 is on): the tier's raw pointers point INTO the backing's
        // immutable mmap - no second native copy exists, and each matrix declares the physical
        // layout its trellis words are in (add_layer refuses a Band8 matrix on a CPU without the
        // band kernels or at an unverified rate - before any byte is decoded).
        cpu_model = std::make_unique<strata::Exl3Model>(dir, expert_backing);
        cpu_moe = std::make_unique<K::cpu::Exl3CpuMoe>();
        std::vector<K::cpu::Exl3CpuMatrix> gs((size_t) g.n_expert), us((size_t) g.n_expert),
            ds((size_t) g.n_expert);
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (!g.is_moe(l)) continue;
            for (int64_t e = 0; e < g.n_expert; ++e) {
                auto fill = [](const strata::Exl3Linear& L, K::cpu::Exl3CpuMatrix& M) {
                    M.trellis = L.trellis; M.suh = L.suh; M.svh = L.svh;
                    M.k = L.in(); M.n = L.out(); M.bits = L.bits; M.hb = L.frac != 0;
                    M.layout = L.layout == strata::Exl3TrellisLayout::Band8
                                   ? K::cpu::Exl3CpuTrellisLayout::Band8
                                   : K::cpu::Exl3CpuTrellisLayout::Native;
                };
                fill(cpu_model->linear(strata::mimo_expert_tensor((int) l, (int) e, "gate")), gs[(size_t) e]);
                fill(cpu_model->linear(strata::mimo_expert_tensor((int) l, (int) e, "up")), us[(size_t) e]);
                fill(cpu_model->linear(strata::mimo_expert_tensor((int) l, (int) e, "down")), ds[(size_t) e]);
            }
            cpu_moe->add_layer((int) l, gs.data(), us.data(), ds.data(), (int) g.n_expert);
        }
        const char* w = std::getenv("STRATA_MIMO_CPU_WORKERS");
        if (w != nullptr && std::atoi(w) > 0) cpu_moe->set_threads(std::atoi(w));
        cpu_flag.assign((size_t) topk, 0);
        cpu_ids.assign((size_t) topk, 0);
        cpu_tok.assign((size_t) topk, 0);
        cpu_row.assign((size_t) topk, 0);
        cpu_route_row.assign((size_t) topk, -1);
        cpu_out.assign((size_t) topk * g.n_embd, 0.0f);
        std::fprintf(stderr, "strata-mimo: cpu tier ready: %lld MoE layer(s) x %lld experts registered, "
                             "workers %s (STRATA_EXL3_CPU_CACHE_MIB bounds the repack cache)\n",
                     (long long) moe_layer_count, (long long) g.n_expert,
                     w != nullptr && std::atoi(w) > 0 ? w : "default (one per physical core)");
        return true;
    }

    /// The primary service for ONE decode token's `k` routes: the ordinary blocking host-IO
    /// run(), with the CPU tier's refused rows overlapped beside it and merged after the join.
    /// `out_host` is [k][n_embd] F32 in the caller's layout (pick order).
    bool serve_primary(int64_t l, const float* x_host, const int32_t* ids, int k, float* out_host) {
        // ExLlama's RDNA decode pipeline is indexed even at T=1. The legacy run() API
        // uses a broadcast FFN, so it misses both the pinned packed plan and FUSE's indexed
        // prologue/epilogue. Reuse our HOST-IO indexed path: same cache/peer ownership, one
        // pinned plan per pass, and original route-order outputs (not run_device's uploads).
        if (decode_indexed && cpu_moe == nullptr)
            return experts->run_multi((int) l, x_host, ids, 1, k, out_host, st);
        if (cpu_moe == nullptr || k > topk || k > 64)
            return experts->run((int) l, x_host, ids, k, out_host, st);
        experts->classify((int) l, ids, 1, k, cpu_flag.data());
        int n = 0;
        for (int j = 0; j < k; ++j) {
            cpu_route_row[(size_t) j] = -1;
            if (cpu_flag[(size_t) j] == 0) continue;
            cpu_ids[(size_t) n] = ids[j];
            cpu_tok[(size_t) n] = 0;
            cpu_row[(size_t) n] = n;
            cpu_route_row[(size_t) j] = n;
            ++n;
        }
        bool bg = false;
        if (n > 0) {
            try {
                cpu_moe->run_rows_bg((int) l, x_host, (int64_t) g.n_embd, cpu_ids.data(), cpu_tok.data(),
                                     cpu_row.data(), n, cpu_out.data(), (int64_t) g.n_embd);
                bg = true;
            } catch (const std::exception& e) {
                std::fprintf(stderr, "strata-mimo: cpu tier: %s (keeping the rows on the GPU)\n", e.what());
            }
        }
        uint8_t refused[64] = {};
        const bool ok = experts->run((int) l, x_host, ids, k, out_host, st, bg ? refused : nullptr);
        if (!ok) return false;
        if (bg) {
            cpu_moe->run_rows_join();
            for (int j = 0; j < k; ++j)
                if (refused[(size_t) j] != 0 && cpu_route_row[(size_t) j] >= 0)
                    std::memcpy(out_host + (size_t) j * g.n_embd,
                                cpu_out.data() + (size_t) cpu_route_row[(size_t) j] * g.n_embd,
                                (size_t) g.n_embd * 4);
            cpu_rows_served += n;
        }
        return true;
    }

    // ---- the MTP drafter (--mtp N, N = 1..3 heads used of the 3): the module owns its KV; the engine
    // feeds it committed cells (token + the target's post-model.norm state, same index) and verifies
    // [anchor, drafts] windows through the ordinary forward (causal inside the window + history - the
    // prefill attention path, which is exactly a verify window's mask).
    strata::mimo::MimoMtp* mtp = nullptr;
    int mtp_heads = 0;

    // ---- the DFlash block-diffusion drafter (--dflash [DIR]): 5-layer bidirectional drafter whose
    // committed ring rows are RECOMPUTED from the target's tap layers {0,11,23,35,47} (cfg order)
    // after every forward; one round drafts a 7-token block and the target verifies all 8 rows.
    // The drafter runs on the DEFAULT stream, so the engine forgoes its own stream when it is on.
    strata::mimo::Dflash* dflash_w = nullptr;            // the loaded weights (owned)
    strata::mimo::DflashDrafter* dflash = nullptr;       // the sequence state (owned)
    uint16_t* tap_bufs[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};   // [chunk][n_embd] F16
    const uint16_t* tap_ptrs[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    int64_t dflash_rounds = 0, dflash_accepted = 0, dflash_offered = 0;
    std::vector<int64_t> dflash_hist;
    int dflash_tokens = 4;
    int32_t* dflash_ids = nullptr;
    double dflash_draft_ms = 0, dflash_verify_ms = 0, dflash_commit_ms = 0;
    double dflash_serial_ms = 0;
    int dflash_calibration = 0, dflash_backoff = 0, dflash_slow = 0;
    int64_t dflash_fallback_tokens = 0;

    /// The adaptive verification window (STRATA_MIMO_DFLASH_AWINDOW=1, default off), after
    /// exllamav3's generator/adaptive_window.py: the drafter still produces its full block,
    /// but each round verifies only the first `w` proposals whose expected tokens/round-cost
    /// is maximal under an online per-position acceptance model.  Verification stays exact -
    /// only the window width changes; accepted output is target-greedy either way.
    /// STATE SCOPE: q/hist/rounds persist for the ENGINE's lifetime (like the reference's
    /// per-generator calibrator) - they are NOT reset per request.  EMA alpha 0.05 + the
    /// explore rounds let the chooser re-adapt when the workload's acceptance drifts (the
    /// bench's CODE-then-PROSE sequence exercises exactly that carry).
    struct Awindow {
        double alpha = 0.05;      // EMA over rounds that OBSERVED the position
        double prior = 0.6;       // q starts here (the reference's measured prior)
        int explore_every = 6;    // every Nth round verifies the max window (stale-q recovery)
        long long rounds = 0;
        std::vector<double> q;    // q[i] = P(draft i+1 accepted | drafts 1..i accepted)
        std::vector<std::pair<int, double>> cost;   // (verified rows r = w+1, relative round cost)
        std::vector<long long> hist;                // chosen windows (index = w)
        void init(int max_w, std::vector<std::pair<int, double>> table) {
            q.assign((size_t) max_w, prior);
            hist.assign((size_t) max_w + 1, 0);
            cost = std::move(table);   // ascending, at least one option <= max_w rows
        }
        /// E[committed tokens] at window w: the carry plus each accepted prefix probability
        double expected(int w) const {
            double e = 1.0, p = 1.0;
            for (int i = 0; i < w; ++i) { p *= q[(size_t) i]; e += p; }
            return e;
        }
        int choose(int max_w) {
            ++rounds;
            if (max_w < 1) return 0;
            if (explore_every > 0 && rounds % explore_every == 0) return max_w;
            int best_w = max_w;
            double best_rate = -1.0;
            for (const auto& [rows, c] : cost) {
                const int w = rows - 1;
                if (w < 1 || w > max_w) continue;
                const double rate = expected(w) / c;
                if (rate > best_rate) { best_rate = rate; best_w = w; }
            }
            return best_w;
        }
        /// `a` of `w` verified drafts accepted; the first mismatch (if any) labels position a
        void update(int w, int a) {
            const int seen = std::min(a, w);
            for (int i = 0; i < seen; ++i) q[(size_t) i] += alpha * (1.0 - q[(size_t) i]);
            if (a < w) q[(size_t) a] += alpha * (0.0 - q[(size_t) a]);
        }
    } awin;
    bool awin_on = false;

    static void dflash_embed(int32_t token, uint16_t* output, void* user) {
        auto& e = *static_cast<Engine*>(user);
        CK(cudaMemcpyAsync(output, (const uint16_t*) e.pack->embed + (size_t) token * e.g.n_embd,
                            (size_t) e.g.n_embd * 2, cudaMemcpyDeviceToDevice));
    }
    static void dflash_argmax(const uint16_t* states, int rows, int32_t* output, void* user) {
        auto& e = *static_cast<Engine*>(user);
        // The standalone drafter owns the default stream; hooks bridge to the target
        // stream only after its states are ready. No producer can be spinning here.
        CK(cudaStreamSynchronize(nullptr));
        k_widen<<<(unsigned) (((int64_t) rows * e.g.n_embd + 255) / 256), 256, 0, e.st>>>(
            states, e.h, (int64_t) rows * e.g.n_embd);
        e.linear(e.pack->lm_head, e.pack->lm_q8.w, e.pack->tlm, e.h, e.diag_logits,
                  rows, e.g.n_vocab, e.g.n_embd);
        k_argmax_rows<<<dim3(1, (unsigned) rows), 1024, 0, e.st>>>(e.diag_logits, e.g.n_vocab, output);
        CK(cudaGetLastError());
        CK(cudaStreamSynchronize(e.st));
    }

    bool prepare_dflash(const std::string& dir, int rows, std::string& error) {
        dflash_w = new strata::mimo::Dflash();
        if (!strata::mimo::dflash_load(dflash_w, dir, &error)) return false;
        const auto& cfg = dflash_w->cfg;
        if (cfg.hidden != g.n_embd || cfg.target_hidden != g.n_embd || cfg.taps.size() > 5 || cfg.block > 8) {
            error = "dflash: incompatible target width/taps/block"; return false;
        }
        for (int layer : cfg.taps) if (layer < 0 || layer >= g.n_layers) {
            error = "dflash: target tap outside layer range"; return false;
        }
        dflash = new strata::mimo::DflashDrafter();
        if (!dflash->init(*dflash_w, dflash_embed, dflash_argmax, this, rows)) {
            error = "dflash: workspace allocation failed"; return false;
        }
        for (size_t i = 0; i < cfg.taps.size(); ++i) {
            CK(cudaMalloc(&tap_bufs[i], (size_t) rows * g.n_embd * 2));
            tap_ptrs[i] = tap_bufs[i];
        }
        CK(cudaMalloc(&dflash_ids, (size_t) cfg.block * 4));
        CK(cudaMalloc(&diag_logits, (size_t) 8 * g.n_vocab * 4));
        CK(cudaMalloc(&arg_all, 8 * 4));
        dflash_hist.assign(cfg.block, 0);
        std::fprintf(stderr, "strata-mimo: DFlash loaded %s, block=%d, target taps=%zu; memory reserved before expert arena\n",
                      dir.c_str(), cfg.block, cfg.taps.size());
        // the adaptive window's relative round-cost table (verified rows -> ms; see awin above).
        // Default measured on this stack (2026-11-01, launcher defaults incl. LFU+adapt+CPU tier
        // + reserve 1024, CODE t1 per-round totals draft+verify+commit: r=4 -> 139.1 ms,
        // r=8 -> 212.3 ms, linear fit 65.9 + 18.3*r; notes/MIMO-DFLASH-WINDOW-2026-11-01.md);
        // override with STRATA_MIMO_DFLASH_AWINDOW_COST="rows:ms,rows:ms" (only listed rows
        // are offered).
        awin_on = strata::core::exl3_flag("STRATA_MIMO_DFLASH_AWINDOW", nullptr, false);
        if (awin_on) {
            // STRATA_MIMO_DFLASH_AWINDOW_COST="rows:ms,rows:ms" - comma list, verified rows (2..8)
            // with finite positive ms. Strictly all-or-defaults: ANY malformed entry (bad syntax,
            // row outside 2..8, non-finite or non-positive cost, empty usable set) is refused with
            // a loud line and the measured defaults below are used; a partial table is never kept.
            std::vector<std::pair<int, double>> table;
            if (const char* e = std::getenv("STRATA_MIMO_DFLASH_AWINDOW_COST"); e && *e) {
                bool ok = true;
                const char* p = e;
                while (*p) {
                    char* end = nullptr;
                    const long r = std::strtol(p, &end, 10);
                    const double ms = (end != p && *end == ':') ? std::strtod(end + 1, &end) : -1.0;
                    if (end == p || r < 2 || r > 8 || !std::isfinite(ms) || ms <= 0) { ok = false; break; }
                    table.emplace_back((int) r, ms);
                    if (*end == '\0') break;                    // clean end after the last pair
                    if (*end != ',') { ok = false; break; }     // garbage where a separator must be
                    p = end + 1;                                // a trailing comma simply ends the list
                }
                if (!ok || table.empty()) {
                    std::fprintf(stderr, "strata-mimo: DFlash awindow: bad COST override '%s'; "
                                         "using measured defaults\n", e);
                    table.clear();
                }
            }
            if (table.empty()) table = {{2, 102.5}, {3, 120.8}, {4, 139.1}, {5, 157.4}, {6, 175.7}, {8, 212.3}};
            if (const char* e = std::getenv("STRATA_MIMO_DFLASH_AWINDOW_EXPLORE"))
                if (int n = std::atoi(e); n >= 0) awin.explore_every = n;
            const int max_w = std::min(dflash_tokens, cfg.block - 1);
            awin.init(max_w, std::move(table));
            std::fprintf(stderr, "strata-mimo: DFlash adaptive window on (opt-in): max %d drafts/round, "
                                 "explore every %d round(s), cost(rows:ms)", max_w, awin.explore_every);
            for (const auto& [r, ms] : awin.cost) std::fprintf(stderr, " %d:%.1f", r, ms);
            std::fprintf(stderr, "\n");
        }
        return true;
    }

    void capture_dflash_tap(int layer, int rows) {
        if (!dflash) return;
        for (size_t i = 0; i < dflash->cfg.taps.size(); ++i) if (dflash->cfg.taps[i] == layer)
            K::f32_to_f16_bulk(x, tap_bufs[i], (int64_t) rows * g.n_embd, st);
    }
    void commit_dflash(int64_t position, int rows) {
        const auto start = Clock::now();
        CK(cudaStreamSynchronize(st));
        if (!dflash->commit_from_taps(tap_ptrs, position, rows)) throw std::runtime_error("dflash: target tap commit failed");
        CK(cudaStreamSynchronize(nullptr));
        dflash_commit_ms += ms_since(start);
    }

    // ---- the per-stage breakdown surface (STRATA_MIMO_TIME=1, printed at shutdown and every 32
    // decode tokens): GPU busy per stage via cudaEvent pairs bracketed on `st`, plus HOST WALL
    // brackets for the serialized blocks (the attention half up to the router, the MoE service
    // block, the lm_head tail) and the peer's launch/primary/finish sub-spans.  wall/token minus
    // the summed GPU busy and block walls is the LAUNCH/SYNC GAP - the overhead target.  All of
    // it is off (zero API calls) unless the env is set: the default path is untouched.
    static constexpr int kTmStages = 12;
    static constexpr const char* kTmNames[kTmStages] = {
        "embed",         "rmsnorm_attn",  "qkv_gemv",    "rope+scatter", "attn",         "o_proj",
        "rmsnorm_mlp",   "router",        "moe_combine", "lm_head+argmax", "attn_half_wall", "moe_block_wall"
    };
    // [0] = decode tokens (T == 1), [1] = prompt chunks; stages 0..9 are event-measured GPU
    // busy, 10..11 are pure host wall brackets, and the peer's launch/primary/finish walls land
    // in the dedicated accumulators below (their spans sit INSIDE the moe block bracket).
    bool tm_on = false;
    double tm_ms[2][kTmStages] = {};
    double tm_wall_ms[2] = {};          // the whole forward() span
    double tm_launch_ms[2] = {}, tm_prim_ms[2] = {}, tm_fin_ms[2] = {}, tm_d2h_ms[2] = {};
    double tm_tail_ms[2] = {};          // past the last layer: final norm + lm_head + argmax + sync
    long long tm_calls[2] = {};
    cudaEvent_t tm_ev[2048] = {};       // the pair ring (one decode token needs < 1000 events)
    int tm_ev_used = 0;
    struct TmRec { int stage, bkt, a, b; };
    std::vector<TmRec> tm_recs;
    int tm_decodes = 0;

    /// record( begin ), enqueue the stage's kernels, record( end ), remember the pair - harvest()
    /// sums.  Off: the plain call, zero API traffic.
    template <class F> void tm_span(int stage, int bkt, F&& enqueue) {
        if (!tm_on) { enqueue(); return; }
        if (tm_ev_used + 2 > (int) (sizeof tm_ev / sizeof tm_ev[0])) { enqueue(); return; }  // ring full
        cudaEvent_t &a = tm_ev[tm_ev_used], &b = tm_ev[tm_ev_used + 1];
        if (a == nullptr) { CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); }
        CK(cudaEventRecord(a, st));
        enqueue();
        tm_recs.push_back({stage, bkt, tm_ev_used, tm_ev_used + 1});
        tm_ev_used += 2;
        CK(cudaEventRecord(b, st));
    }
    /// All events complete (the caller syncs `st` first); fold the pending pairs into the stages.
    void tm_harvest() {
        if (!tm_on) return;
        for (const TmRec& r : tm_recs) {
            float v = 0;
            if (cudaEventElapsedTime(&v, tm_ev[r.a], tm_ev[r.b]) == cudaSuccess) tm_ms[r.bkt][r.stage] += v;
        }
        tm_recs.clear();
        tm_ev_used = 0;
    }
    void tm_print() const {
        if (!tm_on) return;
        for (int bkt = 0; bkt < 2; ++bkt) {
            if (tm_calls[bkt] == 0) continue;
            const double n = (double) tm_calls[bkt];
            const char* what = bkt == 0 ? "decode token" : "prompt chunk";
            std::fprintf(stderr, "strata-mimo: time: %s x%lld, wall %.2f ms each:\n", what,
                         (long long) tm_calls[bkt], tm_wall_ms[bkt] / n);
            double gpu = 0;
            for (int s = 0; s < 10; ++s) gpu += tm_ms[bkt][s];
            for (int s = 0; s < kTmStages; ++s)
                std::fprintf(stderr, "  %-15s %8.3f ms  %5.1f%%%s\n", kTmNames[s], tm_ms[bkt][s] / n,
                             100.0 * tm_ms[bkt][s] / std::max(1e-9, tm_wall_ms[bkt]),
                             s >= 10 ? "  (host wall)" : "");
            std::fprintf(stderr, "  %-15s %8.3f ms          (launch)     %8.3f ms  (primary service)\n",
                         "peer launch", tm_launch_ms[bkt] / n, tm_prim_ms[bkt] / n);
            std::fprintf(stderr, "  %-15s %8.3f ms          (finish)     %8.3f ms  (router d2h+sync)\n",
                         "peer finish", tm_fin_ms[bkt] / n, tm_d2h_ms[bkt] / n);
            std::fprintf(stderr, "  %-15s %8.3f ms  (past the last layer)\n", "tail wall", tm_tail_ms[bkt] / n);
            const double wall = tm_wall_ms[bkt] / n;
            double gpu_notail = 0;
            for (int s = 0; s < 9; ++s) gpu_notail += tm_ms[bkt][s];   // stage 9 sits inside the tail
            std::fprintf(stderr, "  GPU busy (event stages) %.3f ms; attn_half wall %.3f; moe block wall %.3f; "
                                 "gap (wall - stages0..8 - tail) %.3f ms\n",
                         gpu / n, tm_ms[bkt][10] / n, tm_ms[bkt][11] / n,
                         (wall - gpu_notail / n - tm_tail_ms[bkt] / n));
        }
    }

    RouteStats cache_route_snapshot() const {
        return {cache_routes[0].primary + cache_routes[1].primary,
                cache_routes[0].peer + cache_routes[1].peer,
                cache_routes[0].nonresident + cache_routes[1].nonresident};
    }

    /// One boundary evidence line (the freeze experiment): order-invariant residency digests
    /// for both tiers. `set` digests the resident-pair bitmap (equal iff the same experts are
    /// resident); `slot` additionally digests the slot map (equal iff also the same slots) -
    /// independent warmups are compared for equality with these, never assumed equal.
    void boundary_report(const char* tag) const {
        auto fnv_pair = [](uint64_t h, uint64_t v) { return (h ^ v) * 1099511628211ull; };
        auto digest = [&](const std::vector<int32_t>& res, int64_t& n, uint64_t& set, uint64_t& slot) {
            n = 0; set = slot = 1469598103934665603ull;
            for (size_t i = 0; i < res.size(); ++i) {
                const uint64_t s = (uint64_t) (int64_t) res[i];
                if (res[i] >= 0) ++n;
                set = fnv_pair(set, (uint64_t) (res[i] >= 0));   // the resident-pair bitmap
                slot = fnv_pair(slot, s + 0x9e3779b97f4a7c15ull * (uint64_t) i);
            }
        };
        std::vector<int32_t> pres((size_t) g.n_layers * g.n_expert, -1), qres(pres);
        if (experts) experts->residency((int) g.n_layers, (int) g.n_expert, pres);
        int64_t pn = 0, qn = 0; uint64_t pset = 0, pslot = 0, qset = 0, qslot = 0;
        digest(pres, pn, pset, pslot);
        const bool qok = peer && peer->valid();
        if (qok) {
            peer->residency((int) g.n_layers, (int) g.n_expert, qres);
            digest(qres, qn, qset, qslot);
        }
        std::fprintf(stderr, "strata-mimo: boundary %s: primary=%lld set=%016llx slot=%016llx "
                             "peer=%lld set=%016llx slot=%016llx",
                     tag, (long long) pn, (unsigned long long) pset, (unsigned long long) pslot,
                     (long long) (qok ? qn : -1), (unsigned long long) qset, (unsigned long long) qslot);
        if (peer_adapt) {
            const auto c = peer_adapt->counters();
            std::fprintf(stderr, " adapt: windows=%lld rounds=%lld submitted=%lld published=%lld "
                                 "inflight=%lld mib=%.2f submit_ms=%.2f wait_ms=%.2f frozen=%d",
                         (long long) c.windows, (long long) c.rounds, (long long) c.submitted,
                         (long long) c.published, (long long) (c.submitted - c.published),
                         (double) c.bytes / 1048576.0, c.submit_ms, c.wait_ms, peer_adapt->frozen() ? 1 : 0);
        }
        std::fputc('\n', stderr);
    }

    /// The frozen placement as a loadable profile (freeze experiment's replay arm): the
    /// peer's resident pairs, one "layer expert" line each, in pair order. Written once
    /// (wx); a failed create is fatal - a silently missing dump would break the replay.
    void write_frozen_dump() const {
        if (adapt_dump_path.empty() || !peer_adapt || !peer_adapt->frozen()) return;
        FILE* f = std::fopen(adapt_dump_path.c_str(), "wx");
        if (!f) throw std::runtime_error("mimo: cannot create frozen peer dump (never overwritten): " +
                                         adapt_dump_path);
        std::vector<int32_t> qres;
        peer->residency((int) g.n_layers, (int) g.n_expert, qres);
        std::fprintf(f, "# strata-mimo frozen peer residency: layer expert\n");
        int64_t n = 0;
        for (int64_t i = 0; i < (int64_t) qres.size(); ++i)
            if (qres[i] >= 0) {
                std::fprintf(f, "%lld %lld\n", (long long) (i / g.n_expert), (long long) (i % g.n_expert));
                ++n;
            }
        std::fclose(f);
        std::fprintf(stderr, "strata-mimo: frozen peer residency dumped: %lld pairs -> %s\n",
                     (long long) n, adapt_dump_path.c_str());
    }

    void cache_report(const char* phase, const RouteStats& before,
                      const K::Exl3ExpertStream::CacheStats& staged) const {
        if (!cache_statistics) return;
        const auto now = cache_route_snapshot();
        const auto upload = experts->cache_stats();
        std::fprintf(stderr, "strata-mimo: cache stats %s: main=%llu peer=%llu nonresident_routes=%llu "
                             "staged_experts=%llu uploaded_mib=%.2f stage_host_ms=%.2f\n", phase,
            (unsigned long long) (now.primary - before.primary), (unsigned long long) (now.peer - before.peer),
            (unsigned long long) (now.nonresident - before.nonresident),
            (unsigned long long) (upload.staged_experts - staged.staged_experts),
            (upload.staged_bytes - staged.staged_bytes) / 1048576.0, upload.stage_host_ms - staged.stage_host_ms);
    }

    // ---- profiled expert placement: count every routed (layer, expert) during decode and prompt
    // chunks; --expert-profile-save writes them, --expert-profile warm-fills the arena's free slots
    // with the hottest pairs through the stream's own warmup_fill (the Qwen path's --expert-profile
    // idea on this engine's one stream).  Turns the LRU cold start (measured 69% climbing to 81%
    // over 24 tokens) into a hot set from token 0.
    std::vector<int64_t> route_counts;   // [layer * n_expert + expert]
    // STRATA_MIMO_ROUTE_DIGEST=1 (opt-in diagnostics; default off = one cached-bool branch
    // per call, no printing): per FFN call, an FNV-1a digest of the ORDERED routed expert
    // ids plus the exact distinct-expert count over the WHOLE window (route_digest.hpp's
    // tested contract - no sampling or truncation), one stderr line per (call, layer).
    // Used to prove/refute routing divergence between arms at exact (pos, layer)
    // granularity - e.g. whether an epilogue numerics change shifts router selections
    // while the greedy token stream stays identical.
    bool route_digest = false;
    void count_routes(int64_t layer, const int32_t* ids_host_rows, int rows) {
        if (cache_trace && route_decode) {
            std::fprintf(cache_trace, "D %lld %lld", (long long) pos, (long long) layer);
            for (int i = 0; i < rows * topk; ++i) std::fprintf(cache_trace, " %d", ids_host_rows[i]);
            std::fputc('\n', cache_trace);
        }
        if (route_digest) {
            const auto d = strata::core::route_digest(ids_host_rows, (size_t) rows * (size_t) topk);
            std::fprintf(stderr, "strata-mimo: rdigest pos=%lld layer=%lld rows=%d distinct=%d fnv=%016llx\n",
                         (long long) pos, (long long) layer, rows, d.distinct,
                         (unsigned long long) d.fnv);
        }
        if (route_counts.empty()) route_counts.assign((size_t) g.n_layers * g.n_expert, 0);
        for (int r = 0; r < rows; ++r)
            for (int j = 0; j < topk; ++j) {
                const int32_t e = ids_host_rows[(size_t) r * topk + j];
                if (e >= 0 && e < g.n_expert) route_counts[(size_t) layer * g.n_expert + (size_t) e] += 1;
                if (cache_statistics && e >= 0 && e < g.n_expert) {
                    auto& stats = cache_routes[route_decode ? 1 : 0];
                    if (experts->holds((int) layer, e)) ++stats.primary;
                    else if (peer && peer->valid() && peer->has(layer, e)) ++stats.peer;
                    else ++stats.nonresident;
                }
            }
        if (peer_adapt) peer_adapt->observe((int) layer, ids_host_rows, rows * topk, 1.f / rows);
    }
    bool save_profile(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        std::fprintf(f, "# strata-mimo expert routes: layer expert count\n");
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e)
                if (route_counts[(size_t) l * g.n_expert + e] > 0)
                    std::fprintf(f, "%lld %lld %lld\n", (long long) l, (long long) e,
                                 (long long) route_counts[(size_t) l * g.n_expert + e]);
        std::fclose(f);
        return true;
    }
    /// The profile file into (layer, expert) pairs, hottest first (warmup_fill's contract).
    bool parse_profile(const std::string& path, std::vector<std::pair<int32_t, int32_t>>& ranked,
                       std::string& err) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) {
            err = "cannot open " + path;
            return false;
        }
        std::vector<std::tuple<long long, int32_t, int32_t>> order;   // (count, layer, expert)
        char line[256];
        while (std::fgets(line, sizeof line, f) != nullptr) {
            long long l = 0, e = 0, c = 0;
            if (line[0] == '#' || std::sscanf(line, "%lld %lld %lld", &l, &e, &c) != 3) continue;
            if (l < 0 || l >= g.n_layers || e < 0 || e >= g.n_expert || c <= 0) continue;
            order.emplace_back(c, (int32_t) l, (int32_t) e);
        }
        std::fclose(f);
        std::sort(order.begin(), order.end(),
                  [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
        ranked.clear();
        for (const auto& [c, l, e] : order) ranked.emplace_back(l, e);
        if (ranked.empty()) {
            err = "no routes in " + path;
            return false;
        }
        return true;
    }

    /// The primary arena's profile warm-fill (the peer-less arrangement; setup_peer does its own
    /// when the tier is on, after the hot-share reorder).
    bool fill_primary(const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
        int64_t filled = 0, bytes = 0;
        if (experts && !experts->warmup_fill(ranked, &filled, &bytes, err)) return false;
        std::fprintf(stderr, "strata-mimo: expert profile: %zu ranked pairs, %lld slots pre-filled "
                             "(%.2f GiB)\n",
                     ranked.size(), (long long) filled, bytes / 1073741824.0);
        return true;
    }

    /// The peer tier (see the block comment at `peer`): budget -> (optional) the primary's
    /// hot-share warm-fill -> the resident subset -> open() -> the ownership exclusion -> the
    /// prompt share -> the transport. `ranked` is the parsed profile (empty = the cold
    /// round-robin arrangement). Hard-fails when the tier cannot be built (the caller exits);
    /// enable_prefill's refusal is soft (the prompt path stays on the primary, decode keeps it).
    bool setup_peer(const std::string& dir, std::vector<std::pair<int32_t, int32_t>> ranked,
                    std::string& err) {
        if (peer_device < 0 || experts == nullptr) {
            err = "the peer tier is not armed";
            return false;
        }
        int primary_dev = 0;
        CK(cudaGetDevice(&primary_dev));
        if (peer_device == primary_dev) {
            err = "--peer-device " + std::to_string(peer_device) + " is the primary GPU - the peer "
                  "tier needs a second device";
            return false;
        }
        // the peer card's budget: everything free past its permanent reserve (4096 MiB by
        // default) plus a 256 MiB allowance for the tier's own workspace/transport (the
        // Qwen open()'s coarse rule).  STRATA_MIMO_PEER_RESERVE_MIB overrides the reserve
        // in EVERY guard at once - the budget subtraction below, open()'s min_free_bytes
        // and the post-open re-check (both reserve + 1 GiB for the prompt share) - so the
        // knob only moves the tier's free-VRAM floor on the card (MiB of free space left
        // for anything else sharing it; what a desktop already allocated is already absent
        // from "free").  1024..16384 is the knob's deliberate policy range, not a measured
        // safety line.  The measured envelope on this machine is in
        // notes/results/2026-11-01-mimo-peer-reserve/summary.txt.
        size_t peer_reserve_mib = 4096;
        if (const char* v = std::getenv("STRATA_MIMO_PEER_RESERVE_MIB")) {
            long long r = 0;
            if (!strata::mimo::parse_peer_reserve_mib(v, r, err)) return false;
            peer_reserve_mib = (size_t) r;
        }
        size_t budget = 0, peer_free = 0;
        {
            const int cur = primary_dev;
            if (cudaSetDevice(peer_device) != cudaSuccess) {
                err = "cannot switch to the peer device " + std::to_string(peer_device);
                return false;
            }
            size_t tot = 0;
            CK(cudaMemGetInfo(&peer_free, &tot));
            CK(cudaSetDevice(cur));
            const size_t keep = ((size_t) peer_reserve_mib + 256) << 20;
            budget = peer_free > keep ? peer_free - keep : 0;
            std::fprintf(stderr, "strata-mimo: peer GPU %d: %zu MiB free, keeping %zu MiB (the "
                                 "%zu MiB reserve + 256 MiB allowance) -> %zu MiB for experts\n",
                         peer_device, peer_free >> 20, keep >> 20, peer_reserve_mib, budget >> 20);
        }
        if (budget == 0) {
            err = "the peer device has no room past its reserve";
            return false;
        }
        // the hot-share interleave (STRATA_PEER_HOT's mechanics, generate.cpp's arrangement): of
        // the first `at` = the primary's capacity ranks, every pair at a floor((r+1)f) crossing
        // moves behind the primary's fill point, so the peer (which walks the list after the
        // fill, skipping what the primary holds) gets a share of the HOTTEST pairs and both cards
        // compute hot experts every layer instead of the peer owning a cold tail. Default f: the
        // two arenas' capacity ratio, clamped; STRATA_MIMO_PEER_HOT overrides (0 = off).
        const double peer_est = (double) budget / (3.0 * (g.n_embd / 16.0) * (g.moe_intermediate / 16.0) * 40 * 2);
        const double f_def = std::min(0.6, std::max(0.3, peer_est / (peer_est + (double) primary_slots)));
        const char* ph = std::getenv("STRATA_MIMO_PEER_HOT");
        const double f = ph ? std::atof(ph) : f_def;
        const size_t at = (size_t) std::min<int64_t>((int64_t) ranked.size(), primary_slots * moe_layer_count);
        if (!ranked.empty() && f > 0.0 && f < 1.0 && at > 0) {
            std::vector<std::pair<int32_t, int32_t>> keep, moved;
            for (size_t r = 0; r < at; ++r) {
                const bool to_peer = (int64_t) ((double) (r + 1) * f) > (int64_t) ((double) r * f);
                (to_peer ? moved : keep).push_back(ranked[r]);
            }
            const size_t n_moved = moved.size();
            const size_t fill = std::min(ranked.size(), at + n_moved);   // what the primary still takes
            std::vector<std::pair<int32_t, int32_t>> out;
            out.reserve(ranked.size());
            out.insert(out.end(), keep.begin(), keep.end());
            out.insert(out.end(), ranked.begin() + (long) at, ranked.begin() + (long) fill);
            out.insert(out.end(), moved.begin(), moved.end());
            out.insert(out.end(), ranked.begin() + (long) fill, ranked.end());
            ranked.swap(out);
            std::fprintf(stderr, "strata-mimo: peer hot share %.2f: %zu of the first %zu ranked pairs "
                                 "are the peer's (behind the primary's fill point %zu)\n",
                         f, n_moved, at, fill);
        }
        // the primary fills FIRST (the Qwen order: the peer's subset walk then skips what it holds)
        if (!ranked.empty() && !fill_primary(ranked, err)) return false;
        // the peer's resident subset: the ranked pairs it does not hold, else a deterministic
        // round-robin over every MoE layer; byte-bounded by the budget minus a 384 MiB margin
        // (open() allocates ~9 planes per expert as separate cudaMallocs - the granularity waste
        // over ~27k allocations plus the tier's own workspace outweighs the 256 MiB allowance)
        const size_t plane_budget = budget > ((size_t) 384 << 20) ? budget - ((size_t) 384 << 20) : 0;
        // The peer's own model view: the SAME shared backing the primary reads (when band8 is
        // on), so the tier's planes are pread/uploaded from the packed payload - never the
        // native shards - and the entries below carry the packed layout tag.  open() derives
        // ONE tier format from the entries and switches its workspace mode accordingly.
        strata::Exl3Model pmodel(dir, expert_backing);
        std::vector<std::pair<int32_t, int32_t>> subset;
        size_t used = 0;
        auto consider = [&](int32_t l, int32_t e) {
            if (l < 0 || l >= (int32_t) g.n_layers || e < 0 || e >= (int32_t) g.n_expert) return;
            if (g.moe_layer[(size_t) l] == 0) return;   // layer 0 is dense: no expert tensors
            if (experts->holds(l, e)) return;           // never duplicate primary weights
            const strata::Exl3Linear gl = pmodel.linear(strata::mimo_expert_tensor(l, e, "gate"));
            const strata::Exl3Linear ul = pmodel.linear(strata::mimo_expert_tensor(l, e, "up"));
            const strata::Exl3Linear dl = pmodel.linear(strata::mimo_expert_tensor(l, e, "down"));
            size_t bytes = 0;
            for (const strata::Exl3Linear* m : {&gl, &ul, &dl}) {   // trellis bytes via words():
                bytes += (size_t) m->ki * (size_t) m->nj * (size_t) m->words() * 2 +   // frac-aware
                         (size_t) m->in() * 2 + (size_t) m->out() * 2;
            }
            if (bytes == 0 || used > plane_budget || bytes > plane_budget - used) return;   // checked fit
            used += bytes;
            subset.push_back({l, e});
        };
        if (!ranked.empty()) {
            for (const auto& pr : ranked) consider(pr.first, pr.second);
        } else {
            for (int32_t e = 0; e < (int32_t) g.n_expert; ++e)
                for (int32_t l = 1; l < (int32_t) g.n_layers; ++l) consider(l, e);
        }
        if (subset.empty()) {
            err = "no eligible (layer, expert) pair fits the peer budget";
            return false;
        }
        std::fprintf(stderr, "strata-mimo: peer subset: %zu pairs, %zu MiB\n", subset.size(), used >> 20);
        // the entries (host planes straight off the mmap; open() validates every geometry and
        // copies the planes to the peer card - the one time weights cross the bus).  The ~9
        // cudaMallocs per expert carry allocator granularity beyond the summed plane bytes, so
        // an out-of-memory open RETRIES with a 8% smaller subset: open() is transactional (a
        // failed open frees everything it published), and the byte walk above already got within
        // one retry step of the real capacity
        std::vector<K::Exl3Peer::Entry> entries;
        size_t take = subset.size();
        const auto t_fill = Clock::now();
        for (;;) {
            entries.clear();
            peer_owned.assign((size_t) g.n_layers * (size_t) g.n_expert, 0);
            for (size_t i = 0; i < take; ++i) {
                const auto& pr = subset[i];
                K::Exl3Peer::Entry en;
                en.layer = pr.first;
                en.expert = pr.second;
                auto set = [](K::Exl3Mat& m, const strata::Exl3Linear& l) {
                    m.trellis = l.trellis; m.suh = l.suh; m.svh = l.svh;
                    m.ki = l.ki; m.nj = l.nj; m.bits = l.bits; m.frac = l.frac; m.cb = l.cb;
                    m.layout = l.layout;   // the tag says what the source bytes physically are
                };
                set(en.g, pmodel.linear(strata::mimo_expert_tensor(pr.first, pr.second, "gate")));
                set(en.u, pmodel.linear(strata::mimo_expert_tensor(pr.first, pr.second, "up")));
                set(en.d, pmodel.linear(strata::mimo_expert_tensor(pr.first, pr.second, "down")));
                entries.push_back(en);
                peer_owned[(size_t) pr.first * (size_t) g.n_expert + (size_t) pr.second] = 1;
            }
            if (peer == nullptr) peer = std::make_unique<K::Exl3Peer>();
            // max_tokens 16: decode windows are T=1, and a SHORT prompt chunk (T <= 15, where
            // the primary computes every group through the indexed GEMV) rides the decode
            // machinery too - one indexed FFN on the peer instead of ~30 per-group GEMM chains
            if (peer->open(peer_device, (int) g.n_embd, entries, /*max_tokens=*/16, /*max_k=*/topk,
                           err, /*min_free_bytes=*/((size_t) peer_reserve_mib + 1024) << 20)) {
                // the reserve is a hard constraint (the desktop peer keeps its free floor):
                // verify what is REALLY left past the allocator granularity and shrink if it is not
                size_t fr = 0, tot = 0;
                int cur = -1;
                CK(cudaGetDevice(&cur));
                if (cudaSetDevice(peer_device) == cudaSuccess) {
                    CK(cudaMemGetInfo(&fr, &tot));
                    CK(cudaSetDevice(cur));
                    if (fr >= (((size_t) peer_reserve_mib + 1024) << 20)) break;   // reserve + 1 GiB for the prompt share
                } else break;
            }
            const bool oom = err.find("out of memory") != std::string::npos;
            const bool reserve = err.find("reserve") != std::string::npos;
            if (!oom && !reserve && !err.empty()) {
                peer.reset();
                return false;
            }
            if (take < 512) {
                peer.reset();
                err = "the peer device cannot hold a useful subset past its reserve";
                return false;
            }
            std::fprintf(stderr, "strata-mimo: peer open %s at %zu experts - retrying 8%% "
                                 "smaller (the %zu MiB reserve is a hard floor)\n",
                          oom ? "out of memory" : "over the reserve", take, peer_reserve_mib);
            err.clear();
            take = take * 92 / 100;
        }
        experts->set_owned_elsewhere(peer_owned.data(), (int64_t) g.n_layers, g.n_expert);
        std::fprintf(stderr, "strata-mimo: peer tier on GPU %d: %zu static experts (%.2f GiB), "
                             "%.1f/layer, FILLED in %.1f s; the primary excludes their weights\n",
                     peer_device, entries.size(), peer->bytes() / 1073741824.0,
                     (double) entries.size() / std::max<int64_t>(1, moe_layer_count),
                     ms_since(t_fill) / 1000.0);
        // the prompt share: grouped FFN on the peer card, one weighted sum per token back. A
        // refusal is soft - decode keeps the tier, the prompt path stays on the primary. The
        // workspace reserves inside throw on a tight card; that refusal is soft too. (The
        // transport itself - the side stream, the mapped F16 image, the sums/mask buffers - was
        // allocated back in init(), before the arena sized itself from free VRAM.)
        try {
            if (!peer->enable_prefill(chunk, topk, (int64_t) g.n_expert, (int64_t) chunk * topk, err)) {
                std::fprintf(stderr, "strata-mimo: peer prompt share refused (%s) - the prompt path "
                                     "stays on the primary GPU\n", err.c_str());
                err.clear();
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata-mimo: peer prompt share refused (%s) - the prompt path "
                                 "stays on the primary GPU\n", e.what());
            err.clear();
        }
        {   // surface and clear any sticky error from the retry loop's freed attempts now, not at
            // the first forward's kernel check
            const cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess)
                std::fprintf(stderr, "strata-mimo: peer setup left a cuda error state: %s (cleared)\n",
                             cudaGetErrorString(e));
        }
        // STRATA_MIMO_PEER_DEVICE_OUT (the producer-T1 device arm's collector, unchanged) or
        // STRATA_MIMO_PEER_DEVICE_RESULTS (the handoff WORKER's collector, R3) - one shared
        // frame, never simultaneous (the doorbell protocol serializes the producer and
        // worker arms per forward). The worker arm additionally carries WINDOW routes, so
        // its frame capacity covers T*topk for every eligible window (T <= 8, k <= 64/T);
        // the producer-only arm keeps its exact original capacity and bytes.
        const char* device_out_env = std::getenv("STRATA_MIMO_PEER_DEVICE_OUT");
        const bool want_device_out = device_out_env && device_out_env[0] == '1' && decode_device;
        if (want_device_out || peer_device_results) {
            const int capacity = peer_device_results
                ? std::max(topk, std::min(512, 8 * topk))
                : topk;
            peer_output = std::make_unique<K::Exl3PeerOutput>();
            if (!peer_output->open(primary_dev, peer_device, (int) g.n_embd, capacity, err)) return false;
            std::fprintf(stderr, "strata-mimo: peer output: mapped F16 rows collected on the primary "
                                 "stream (capacity %d routes%s)\n", capacity,
                         peer_device_results ? "; worker windows/decode included" : "");
        }
        if (strata::core::exl3_flag("STRATA_MIMO_PEER_ADAPT", nullptr, false)) {
            K::Exl3PeerAdapt::Options opt;
            opt.expert_prefix = "model.layers.";
            opt.external_observer = true;
            opt.every = 16;
            opt.max_swaps = 16;
            opt.byte_cap = (size_t) 128 << 20;
            opt.expert_backing = expert_backing;   // refills read the same packed source (or none)
            if (resident_peer) opt.resident_exchange = resident_backing;
            if (const char* v = std::getenv("STRATA_MIMO_PEER_ADAPT_EVERY")) opt.every = std::atoi(v);
            if (const char* v = std::getenv("STRATA_MIMO_PEER_ADAPT_SWAPS")) opt.max_swaps = std::atoi(v);
            if (const char* v = std::getenv("STRATA_MIMO_PEER_ADAPT_DECAY")) opt.decay = std::strtof(v, nullptr);
            if (const char* v = std::getenv("STRATA_MIMO_PEER_ADAPT_MARGIN")) opt.min_gain = std::strtof(v, nullptr);
            if (const char* v = std::getenv("STRATA_MIMO_PEER_ADAPT_MIB")) {
                const double mib = std::strtod(v, nullptr);
                if (!(mib >= 0 && mib <= 4096)) throw std::invalid_argument("mimo: invalid peer adapt MiB cap");
                opt.byte_cap = (size_t) (mib * 1048576.0);
            }
            peer_adapt = std::make_unique<K::Exl3PeerAdapt>(*peer, *experts, dir,
                (int) g.n_layers, (int) g.n_expert, peer_owned, opt);
        }
        // STRATA_MIMO_OWNERSHIP_CHECK=1 (opt-in live verification, 2026-11-01 ownership audit):
        // one exact-residency scan at setup - primary residency vs the peer tier vs the live
        // ownership table. Steady state must be: table == peer residency, and no (layer,
        // expert) resident on both cards. This is the STATIC invariant only (no refill is in
        // flight at setup); the adapter's own per-window snapshot check carries the runtime
        // one. Reports exact counts and fails the setup loudly on a violation. No cost and no
        // output when the switch is unset. It proves exclusive OWNERSHIP, not that physical
        // bytes are single-copy: a refill's transfer window legitimately holds the incoming
        // expert in pinned staging plus the reused slot, and the primary's bounded scratch may
        // transiently stream an owned expert's planes for fallback compute.
        if (strata::core::exl3_flag("STRATA_MIMO_OWNERSHIP_CHECK", nullptr, false)) {
            std::vector<int32_t> pres, qres;
            experts->residency((int) g.n_layers, (int) g.n_expert, pres);
            peer->residency((int) g.n_layers, (int) g.n_expert, qres);
            int64_t n_primary = 0, n_peer = 0, overlap = 0, table_bad = 0;
            std::vector<std::pair<int64_t, int32_t>> shown;
            const int64_t total = (int64_t) g.n_layers * (int64_t) g.n_expert;
            for (int64_t i = 0; i < total; ++i) {
                n_primary += pres[(size_t) i] >= 0;
                n_peer += qres[(size_t) i] >= 0;
                const bool p = pres[(size_t) i] >= 0, q = qres[(size_t) i] >= 0;
                if (p && q) ++overlap;
                if ((peer_owned[(size_t) i] != 0) != q) ++table_bad;
                if ((p && q || (p && peer_owned[(size_t) i] != 0)) && shown.size() < 8)
                    shown.emplace_back(i / (int64_t) g.n_expert, (int32_t) (i % (int64_t) g.n_expert));
            }
            std::fprintf(stderr, "strata-mimo: ownership check: %lld primary-resident, %lld peer-resident, "
                         "%lld two-card, %lld table/tier mismatches\n",
                         (long long) n_primary, (long long) n_peer,
                         (long long) overlap, (long long) table_bad);
            if (overlap || table_bad) {
                for (const auto& pr : shown)
                    std::fprintf(stderr, "strata-mimo: ownership check: layer %lld expert %d resident "
                                 "on both cards / contradicts the table\n", (long long) pr.first, pr.second);
                err = "the peer tier and the primary expert cache hold overlapping residents";
                return false;
            }
        }
        return true;
    }

    /// The runtime peer failure path (the Qwen fallback): the tier closes (drains), the primary's
    /// exclusion drops, and the FULL primary service covers the layer - nothing partial is ever
    /// published. Later layers take the plain path (peer->valid() is false after close()).
    void peer_fallback(int64_t layer, const char* what, const char* detail) {
        if (resident_backing && resident_backing->active() && resident_backing->has_gpu_only())
            throw std::runtime_error(std::string("resident EXL3: peer failure with GPU-only weights: ") + detail);
        std::fprintf(stderr, "strata-mimo: the peer tier failed at layer %lld (%s: %s) - full "
                             "primary fallback\n", (long long) layer, what, detail);
        ++peer_fallbacks;
        // R3/PEER_DEVICE_OUT: the shared frame may hold an in-flight or half-consumed
        // window whose ready signal the now-failed peer stream will never land - release
        // the watchdog BEFORE close() drains anything, so an already-enqueued wait kernel
        // exits instead of pinning the primary stream. Safe here by construction: after
        // close() the tier advertises nothing, so no later layer can enqueue another
        // frame; the producer-T1 arm's own copy (if queued) exits on the aborted flag.
        if (peer_output) peer_output->release();
        peer->close();
        experts->set_owned_elsewhere(nullptr, 0, 0);
    }

    /// The breakdown surface at shutdown (the Qwen Exl3Drive summary's shape): decode rows by
    /// tier and window mix, per-stage host spans per layer-call, prompt rows by tier, and the
    /// bytes that crossed the peer's narrow upstream. HOST-ELAPSED, not additive GPU time: the
    /// launch only enqueues (the peer's GPU work overlaps the primary service), and finish's
    /// span contains the wait for the peer's remaining work.
    void peer_summary() const {
        if (peer == nullptr && peer_rows == 0 && peer_pf_rows == 0 && wdm_calls == 0 &&
            peer_dev_frames == 0)
            return;
        const int64_t routes = peer_calls * topk;
        if (peer_calls > 0)
            std::fprintf(stderr, "strata-mimo: peer decode: %lld of %lld routed rows on the peer "
                                 "(%.1f%%; windows all/mixed/zero %lld/%lld/%lld of %lld calls), "
                                 "%lld complement rows (%lld from host); per call: launch %.3f ms, "
                                 "primary %.3f ms, finish %.3f ms; %lld fallback(s)\n",
                         (long long) peer_rows, (long long) routes,
                         100.0 * (double) peer_rows / (double) std::max<int64_t>(1, routes),
                         (long long) peer_win_all, (long long) peer_win_mixed,
                         (long long) peer_win_zero, (long long) peer_calls,
                         (long long) peer_primary_rows, (long long) peer_primary_miss_rows,
                         peer_launch_ms / (double) peer_calls, peer_primary_ms / (double) peer_calls,
                         peer_finish_ms / (double) peer_calls, (long long) peer_fallbacks);
        if (peer_pf_layers > 0)
            std::fprintf(stderr, "strata-mimo: peer prompt: %lld layer(s) x %lld token(s): %lld of "
                                 "%lld routed rows on the peer (%.1f%%); per layer: launch %.2f ms, "
                                 "primary %.2f ms, finish %.2f ms\n",
                         (long long) peer_pf_layers, (long long) peer_pf_tokens,
                         (long long) peer_pf_rows,
                         (long long) peer_pf_rows + peer_pf_local_rows,
                         100.0 * (double) peer_pf_rows /
                             (double) std::max<int64_t>(1, peer_pf_rows + peer_pf_local_rows),
                         peer_pf_launch_ms / (double) peer_pf_layers,
                         peer_pf_primary_ms / (double) peer_pf_layers,
                         peer_pf_finish_ms / (double) peer_pf_layers);
        if (svc_time && svc_calls > 0)
            std::fprintf(stderr, "strata-mimo: decode service (host, per layer-call x%lld; "
                                 "collect x%lld): collect %.3f ms, class %.3f ms, cpu_bg %.3f ms, "
                                 "stage %.3f ms, cpu_join %.3f ms, tail %.3f ms, peer_wait %.3f ms, "
                                 "peer_tail %.3f ms, done %.3f ms; sum %.3f ms\n",
                         (long long) svc_calls, (long long) svc_collect_calls,
                         svc_collect_ms / (double) std::max<int64_t>(1, svc_collect_calls),
                         svc_class_ms / (double) svc_calls,
                         svc_cpu_bg_ms / (double) svc_calls, svc_stage_ms / (double) svc_calls,
                         svc_cpu_join_ms / (double) svc_calls, svc_tail_ms / (double) svc_calls,
                         svc_finwait_ms / (double) std::max<int64_t>(1, peer_calls),
                         svc_fintail_ms / (double) std::max<int64_t>(1, peer_calls),
                         svc_done_ms / (double) std::max<int64_t>(1, svc_collect_calls),
                         (svc_collect_ms + svc_class_ms + svc_cpu_bg_ms + svc_stage_ms +
                          svc_cpu_join_ms + svc_tail_ms + svc_finwait_ms + svc_fintail_ms +
                          svc_done_ms + peer_launch_ms) / (double) std::max<int64_t>(1, svc_calls));
        std::fprintf(stderr, "strata-mimo: peer bytes: %.1f MiB up / %.1f MiB down over the peer's "
                             "link (%s)\n",
                     peer_bytes_up / 1048576.0, peer_bytes_down / 1048576.0,
                     resident_backing ? "activations, plans and sums only; weight uploads/copy-backs reported separately"
                                      : "activations, plans and sums only - weight planes never cross after open()");
        if (peer_dev_frames > 0)
            std::fprintf(stderr, "strata-mimo: peer device results: %lld frame(s) / %lld rows "
                                 "converted+scattered on the primary GPU; %.2f MiB whole-slab F32 "
                                 "upload and %lld CPU F16->F32 row conversion(s) avoided\n",
                         (long long) peer_dev_frames, (long long) peer_dev_rows,
                         peer_dev_upload_avoided / 1048576.0, (long long) peer_dev_convert_avoided);
        if (wdm_calls > 0)
            std::fprintf(stderr, "strata-mimo: window device misses: %lld window complement(s) / "
                                 "%lld GPU rows scattered on device, %lld CPU-overlay rows kept on "
                                 "the host upload; %.2f MiB NET host-arm bus bytes removed "
                                 "(activation H2D + result D2H + result H2D - the CPU overlay's "
                                 "compact result H2D, shape-derived)\n",
                         (long long) wdm_calls, (long long) wdm_routes,
                         (long long) wdm_cpu_rows, wdm_bytes_avoided / 1048576.0);
    }
    int32_t* d_anchor_tok = nullptr;            // device anchor token for draft()
    int64_t rounds = 0, accepted_total = 0, offered_total = 0;
    int last_accept = 0;
    std::vector<int64_t> accept_hist;           // accepted drafts per round (0..k-1; the cap is k-1)
    struct DiagLayer {                          // one entry per layer, LAST row of the window
        double attn_mean = 0, attn_std = 0, moe_mean = 0, moe_std = 0;
        std::vector<int32_t> ids;
        std::vector<float> w;
    };
    std::vector<DiagLayer> diag_layers;
    std::vector<float> diag_snap;              // x before the FFN (the delta stats' baseline)

    ~Engine() {
        stop_decode_worker();
        peer_adapt.reset();
        if (cache_trace) std::fclose(cache_trace);
        if (st) CK(cudaStreamSynchronize(st));
        tm_harvest();
        tm_print();
        if (dflash) {
            std::fprintf(stderr, "strata-mimo: DFlash: %lld rounds, %lld/%lld proposals accepted (%.1f%%), "
                                "draft %.1f ms verify %.1f ms commit %.1f ms; histogram",
                (long long) dflash_rounds, (long long) dflash_accepted, (long long) dflash_offered,
                100. * dflash_accepted / std::max<int64_t>(1, dflash_offered),
                dflash_draft_ms, dflash_verify_ms, dflash_commit_ms);
            for (auto n : dflash_hist) std::fprintf(stderr, " %lld", (long long) n);
            std::fprintf(stderr, "\n");
            dflash->free(); delete dflash;
        }
        if (dflash_w) { strata::mimo::dflash_unload(dflash_w); delete dflash_w; }
        for (auto p : tap_bufs) if (p) (void) cudaFree(p);
        if (dflash_ids) (void) cudaFree(dflash_ids);
        if (cpu_moe != nullptr && cpu_rows_served > 0) {
            const auto s = cpu_moe->stats();
            std::fprintf(stderr, "strata-mimo: cpu tier: %lld routed rows computed on the host "
                                 "(%.2f GiB trellis streamed, %lld repack fill(s), %lld evictions); "
                                 "prepare %.1f ms, residual join %.1f ms, service %.1f ms; "
                                 "ISA %d, workers %d, repack cache %.1f MiB\n",
                         (long long) cpu_rows_served, s.trellis_gb, (long long) s.repack_fills,
                         (long long) s.repack_evictions, s.prepare_ms, s.join_ms, s.ms,
                         s.isa, s.threads, s.cache_bytes / 1048576.0);
        }
        peer_summary();
        if (window_router_split)
            std::fprintf(stderr, "strata-mimo: window_router_split total: %lld window forward(s), "
                                 "%lld MoE layer route(s) through the batched split kernels "
                                 "(T2..8 all-rows forwards)\n",
                         (long long) window_split_windows, (long long) window_split_layers);
        if (resident_backing && resident_backing->active()) {
            const auto s = resident_backing->stats();
            std::fprintf(stderr,"strata-mimo: resident EXL3 summary: %.2f GiB RAM / %.2f GiB locked, "
                                "%llu startup file reads (%.2f GiB), %llu GPU->RAM returns, "
                                "%llu RAM->GPU releases; runtime file fallback disabled\n",
                         s.resident_bytes/1073741824.0,s.locked_bytes/1073741824.0,
                         (unsigned long long)s.file_reads,s.file_bytes/1073741824.0,
                         (unsigned long long)s.returned,(unsigned long long)s.released);
        }
        free_all();
        // Lifetime order for the shared packed backing: the decode worker was joined above,
        // the adapter drained, peer/transport freed in free_all(), and the expert stream / CPU
        // tier (unique_ptr members destroyed after this body) each hold their own shared copy
        // through their Exl3Model - so dropping the ENGINE's handle here can never close the
        // payload mmap under a surviving pointer.  Explicit anyway: the last reader out.
        expert_backing.reset();
        // Origin diagnostic (2026-11-12 audit-fix, harness-only value): a sticky
        // cudaErrorInvalidValue SURVIVES this destructor when a peer tier existed (the
        // fixture harness observed it at the NEXT engine's entry with every opt-in off).
        // Report it HERE - at its origin class, before the unique_ptr members finish - so
        // the next engine in the process cannot misattribute it. Nothing is cleared or
        // fataled: production single-engine processes exit anyway, and the parity harness
        // still confines whatever remains to its documented-code-only drains.
        if (peer_device >= 0) {
            const cudaError_t stale = cudaGetLastError();
            if (stale != cudaSuccess)
                std::fprintf(stderr, "strata-mimo: engine teardown left a cuda error state: %s "
                                     "(reported at the destructor; the next engine in this "
                                     "process sees a clean slate)\n", cudaGetErrorString(stale));
        }
    }

    void free_all() {
        if (st) CK(cudaStreamSynchronize(st));
        if (peer_xs) { (void) cudaStreamSynchronize(peer_xs); (void) cudaStreamDestroy(peer_xs); }
        if (peer_ev) (void) cudaEventDestroy(peer_ev);
        peer_xs = nullptr;
        peer_ev = nullptr;
        if (peer_x16_host) (void) cudaFreeHost(peer_x16_host);
        if (peer_w_host) (void) cudaFreeHost(peer_w_host);
        if (peer_comp_host) (void) cudaFreeHost(peer_comp_host);
        if (peer_h_host) (void) cudaFreeHost(peer_h_host);
        if (peer_rows_host) (void) cudaFreeHost(peer_rows_host);
        if (peer_sums_dev) (void) cudaFree(peer_sums_dev);
        if (peer_mask_dev) (void) cudaFree(peer_mask_dev);
        if (peer_rows_dev) (void) cudaFree(peer_rows_dev);
        peer_x16_host = nullptr;
        peer_w_host = peer_comp_host = nullptr;
        peer_h_host = peer_rows_host = nullptr;
        peer_sums_dev = nullptr;
        peer_mask_dev = nullptr;
        peer_rows_dev = nullptr;
        peer.reset();   // drains and frees the tier (its own device guard)
        peer_output.reset();   // primary and peer streams are drained before transport release
        for (uint16_t* p : kc) if (p) (void) cudaFree(p);
        for (uint16_t* p : vc) if (p) (void) cudaFree(p);
        if (ring_k16) (void) cudaFree(ring_k16);
        ring_k16 = nullptr;
        if (attn_ws.mem) (void) cudaFree(attn_ws.mem);   // split-attention workspace (post-sync)
        attn_ws.mem = nullptr;
        attn_ws.bytes = 0;
        for (void* p : { (void*) x, (void*) h, (void*) qf, (void*) kf, (void*) vf, (void*) attn, (void*) proj,
                         (void*) gg, (void*) uu, (void*) parts, (void*) logits })
            if (p) (void) cudaFree(p);
        for (void* p : { (void*) h16, (void*) q16, (void*) a16, (void*) hb, (void*) tok, (void*) ids,
                         (void*) arg, (void*) dpos, (void*) dseq, (void*) rw })
            if (p) (void) cudaFree(p);
        if (q_perm_buf) (void) cudaFree(q_perm_buf);
        if (attn_perm_buf) (void) cudaFree(attn_perm_buf);
        if (q8_scratch) (void) cudaFree(q8_scratch);
        q8_scratch = nullptr;
        if (q8_wide_scratch) (void) cudaFree(q8_wide_scratch);
        q8_wide_scratch = nullptr;
        if (gqa_scratch_sinks) (void) cudaFree(gqa_scratch_sinks);
        if (sink_scaled_buf) (void) cudaFree(sink_scaled_buf);
        if (diag_logits) (void) cudaFree(diag_logits);
        if (arg_all) (void) cudaFree(arg_all);
        if (route_scores) (void) cudaFree(route_scores);
        route_scores = nullptr;
        if (window_route_scores) {
            (void) cudaFree(window_route_scores);
            window_route_scores = nullptr;
        }
        if (diag_ids_host) (void) cudaFreeHost(diag_ids_host);
        if (diag_w_host) (void) cudaFreeHost(diag_w_host);
        if (h_host) (void) cudaFreeHost(h_host);
        if (st) (void) cudaStreamDestroy(st);
        kc.clear();
        vc.clear();
        x = h = qf = kf = vf = attn = proj = gg = uu = parts = logits = nullptr;
        h16 = q16 = a16 = hb = nullptr;
        q_perm_buf = nullptr;
        attn_perm_buf = nullptr;
        gqa_scratch_sinks = sink_scaled_buf = nullptr;
        diag_logits = nullptr;
        tok = ids = arg = nullptr;
        dpos = nullptr;
        dseq = nullptr;
        rw = nullptr;
        h_host = nullptr;
        st = nullptr;
    }

    /// Everything after the pack: the stream, the Gemm, the caches, the window buffers.
    /// `g` (the geometry) is set by the caller before init.
    bool init(const std::string& dir, std::unique_ptr<MimoPack> p, int64_t ctx, int chunk_in,
              double expert_vram, int64_t expert_ram, std::string& err) {
        pack = std::move(p);
        cap = ctx;
        chunk = chunk_in;
        topk = (int) g.n_expert_used;
        window_handoff = strata::mimo::window_handoff_eligible(8, topk, (int) g.n_embd, chunk_in, true) &&
                         strata::core::exl3_flag("STRATA_MIMO_WINDOW_HANDOFF", nullptr, false);
        max_kv = (int) std::max(g.n_head_kv_full, g.n_head_kv_swa);
        sm_scale = 1.0 / std::sqrt((double) g.head_dim_qk);
        CK(cudaStreamCreate(&st));
        if (!gemm.init(st, 0, err)) return false;
        if (peer_device >= 0) {   // the peer transport, BEFORE the arena sizes itself from free
            // VRAM (the tier itself opens later, in setup_peer, once the profile's ranked pairs
            // are known): the prompt share's readback side stream + the mapped F16 image + the
            // per-token sums/mask the combine reads - ~10 MiB the arena must not eat
            CK(cudaStreamCreateWithFlags(&peer_xs, cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&peer_ev, cudaEventDisableTiming));
            CK(cudaHostAlloc((void**) &peer_x16_host, (size_t) chunk_in * g.n_embd * 2,
                             cudaHostAllocMapped | cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &peer_w_host, (size_t) chunk_in * g.n_expert_used * 4,
                             cudaHostAllocDefault));
            CK(cudaHostAlloc((void**) &peer_comp_host, (size_t) g.n_expert_used * g.n_embd * 4,
                             cudaHostAllocDefault));
            CK(cudaMalloc(&peer_sums_dev, (size_t) chunk_in * g.n_embd * 4));
            CK(cudaMalloc(&peer_mask_dev, (size_t) chunk_in * g.n_expert_used));
            CK(cudaHostAlloc((void**) &peer_h_host, (size_t) 16 * g.n_embd * 4, cudaHostAllocDefault));
            CK(cudaHostAlloc((void**) &peer_rows_host,
                             (size_t) 16 * g.n_expert_used * g.n_embd * 4, cudaHostAllocDefault));
            CK(cudaMalloc(&peer_rows_dev, (size_t) 16 * g.n_expert_used * g.n_embd * 4));
            buffer_bytes += (size_t) chunk_in * g.n_embd * 4 + (size_t) chunk_in * g.n_embd * 2 +
                            (size_t) 16 * g.n_embd * 4 + (size_t) 16 * (size_t) g.n_expert_used *
                                                              g.n_embd * 4 * 2;
            peer_mask.assign((size_t) chunk_in * g.n_expert_used, 0);
            peer_kind.assign((size_t) std::max<int64_t>(64, 16 * g.n_expert_used), 0);
        }

        // Opt-in SWA storage: keep the window plus an entire appended prompt chunk. Merely
        // keeping 128 rows would overwrite the history needed by the first query of a chunk.
        // At least eight rows also covers the existing verify-window envelope.
        if (const char* e = std::getenv("STRATA_MIMO_SWA_RING"); e && e[0] == '1') {
            swa_ring = (int) std::min<int64_t>(cap, g.sliding_window + std::max(chunk_in, 8) - 1);
            CK(cudaMalloc(&ring_k16, (size_t) chunk_in * max_kv * g.head_dim_qk * 2));
            buffer_bytes += (size_t) chunk_in * max_kv * g.head_dim_qk * 2;
            std::fprintf(stderr, "strata-mimo: SWA ring: %d physical rows/layer (window %lld + chunk %d - 1), "
                                 "full-attention layers keep %lld rows\n", swa_ring,
                         (long long) g.sliding_window, std::max(chunk_in, 8), (long long) cap);
        }
        // the KV caches: K [physical rows][kv][hd_qk], V [physical rows][kv][hd_v], both F16
        kc.resize((size_t) g.n_layers);
        vc.resize((size_t) g.n_layers);
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const int64_t kv = g.kv_heads(l);
            const int64_t physical = swa_ring && !g.is_full(l) ? swa_ring : cap;
            const size_t kb = (size_t) physical * kv * g.head_dim_qk * 2;
            const size_t vb = (size_t) physical * kv * g.head_dim_v * 2;
            CK(cudaMalloc(&kc[(size_t) l], kb ? kb : 1));
            CK(cudaMalloc(&vc[(size_t) l], vb ? vb : 1));
            cache_bytes += kb + vb;
        }

        // the opt-in split-sequence attention workspace, AFTER the KV caches and BEFORE the
        // expert arena sizes itself from free VRAM (this ~17 MiB is a feature cost the arena
        // must see, not one it eats silently).  One allocation, engine-lifetime ownership,
        // freed in free_all(); decode/verify calls T <= max_rows use it, everything else and
        // the gate-off default keep the original kernels untouched.  rows 8 = the verify-
        // window envelope (decode is T=1); splits 64 covers the heuristic's T1 worst case.
        attn_split = K::mimo_attn_split_enabled();
        if (attn_split) {
            attn_ws.max_rows = 8;
            attn_ws.max_splits = 64;
            attn_ws.bytes = K::mimo_attn_split_ws_bytes(attn_ws.max_rows, (uint32_t) max_kv,
                                                        (uint32_t) g.n_q_head,
                                                        (uint32_t) g.head_dim_v, attn_ws.max_splits);
            CK(cudaMalloc(&attn_ws.mem, (size_t) attn_ws.bytes));
            buffer_bytes += attn_ws.bytes;
            std::fprintf(stderr, "strata-mimo: split attention: workspace %.1f MiB reserved "
                                 "(rows<=%u splits<=%u kv<=%d) - accounted in the buffer budget; "
                                 "arena budget measured after it\n",
                         (double) attn_ws.bytes / 1048576.0, attn_ws.max_rows, attn_ws.max_splits,
                         max_kv);
        }

        // P1/P2 (2026-11-05 dense design): the flags parse here and the WIDE q8_1 image allocates
        // HERE - after the KV caches/split workspace, BEFORE the expert arena sizes itself from free
        // VRAM (the split-attention rule: a feature cost the arena must see, not one it eats).  The
        // image is the widest dense input x 32 columns (~0.6 MiB here); P2 off (the default) costs
        // nothing, and the <= 8 path keeps its own q8_scratch exactly where it was.
        qkv_share = [] { const char* v = std::getenv("STRATA_MIMO_QKV_SHARE"); return v != nullptr && v[0] == '1'; }();
        if (const char* v = std::getenv("STRATA_MIMO_Q8_PREFILL_COLS"); v != nullptr && *v != '\0') {
            if (std::strcmp(v, "8") == 0) q8_wide_cols = 8;         // explicit 8: the default path
            else if (std::strcmp(v, "16") == 0) q8_wide_cols = 16;
            else if (std::strcmp(v, "32") == 0) q8_wide_cols = 32;
            else throw std::invalid_argument("mimo: STRATA_MIMO_Q8_PREFILL_COLS must be 8, 16 or 32 (got '" +
                                             std::string(v) + "')");
        }
        if (qkv_share || q8_wide_cols > 8) {
            const bool any_q8 = !pack->layers.empty() &&
                                (pack->layers[0].q8q.valid() || pack->layers[0].q8o.valid() ||
                                 pack->lm_q8.valid());
            if (q8_wide_cols > 8 && any_q8) {
                const int64_t widest = std::max<int64_t>({g.n_embd, g.v_dim(), g.dense_intermediate});
                CK(cudaMalloc(&q8_wide_scratch, K::native_q8_1_wide_bytes((int) widest, q8_wide_cols)));
                buffer_bytes += K::native_q8_1_wide_bytes((int) widest, q8_wide_cols);
                // the preflight: eligibility is process-stable, so decide it once here - a build where
                // the wide entries would refuse (STRATA_TSUM=1, upstream layout, gfx906 wave) never
                // pays the per-tile wide quantize it would otherwise enqueue before the refusal
                q8_wide_active = K::native_q8_0_wide_supported((int) widest, q8_wide_cols);
            } else if (q8_wide_cols > 8) {
                q8_wide_cols = 8;   // no q8 views: nothing to widen (the flag is inert, not an error)
            }
            std::fprintf(stderr, "strata-mimo: dense q8 P1/P2: qkv_share=%s, prefill_cols=%d%s\n",
                         qkv_share ? "on" : "off", q8_wide_cols,
                         q8_wide_scratch != nullptr ? " (wide q8_1 image reserved before the expert arena)" : "");
        }

        if (window_router_split) {
            // The window split's bounded score scratch, reserved BEFORE the auto arena
            // budget probe below (the split-attention rule: a feature cost the arena must
            // see, not one it eats) - [8, n_expert] F32, 8 KiB on the target model, and
            // nothing at all when the opt-in is off.  Never allocated under a service wait;
            // T1 keeps its own [n_expert] buffer and decode-arm semantics untouched.
            CK(cudaMalloc(&window_route_scores, (size_t) 8 * g.n_expert * 4));
            buffer_bytes += (size_t) 8 * g.n_expert * 4;
            std::fprintf(stderr, "strata-mimo: window router split armed%s: T=2..8 verify "
                                 "windows route through the batched split kernels "
                                 "(bit-identical picks; two launches); T1 decode and prompt "
                                 "chunks keep their established kernels\n",
                         window_router_split_stats_on ? " (stats on)" : "");
        }
        if (window_handoff) {
            // Reserve transport and verify-tail storage before the auto arena budget probe.
            // No allocator is allowed underneath a queued producer wait kernel.
            peer_kind.resize(std::max<size_t>(peer_kind.size(), 64));
            if (!peer_rows_dev) CK(cudaMalloc(&peer_rows_dev, (size_t) 64 * g.n_embd * 4));
            if (!diag_logits) CK(cudaMalloc(&diag_logits, (size_t) 8 * g.n_vocab * 4));
            if (!arg_all) CK(cudaMalloc(&arg_all, 8 * 4));
            buffer_bytes += (size_t) 8 * g.n_vocab * 4 + 8 * 4;
        }

        // the routed experts: the stream reads them from the model dir trellis under the `model.layers.`
        // prefix (what mimo_expert_tensor spells); slots from --expert-vram (a rough 8.8 MB per resident
        // expert: 3 x (D/16)x(moe_inter/16) x 16K words at K~2.5 plus the scale planes).  The stream's
        // layer range is the MoE layers only - a dense layer (layer 0) has no expert tensors to scan.
        int slots = 0;
        int64_t moe_layers = 0;
        int first_moe = (int) g.n_layers;
        for (int64_t l = 0; l < g.n_layers; ++l)
            if (g.moe_layer[(size_t) l] != 0) {
                ++moe_layers;
                first_moe = std::min<int>(first_moe, (int) l);
            }
        {   // the expert arena: --expert-vram GIB caps it; the DEFAULT (auto) fills everything the
            // pack, the KV and the buffers left free (a ~1 GiB safety floor), measured per plane
            // bytes at the model's own K mix (~6.8 MiB/expert here)
            size_t per_expert_bytes = 0;
            if (moe_layers > 0) {
                const double per_expert = 3.0 * (g.n_embd / 16.0) * (g.moe_intermediate / 16.0) * 16 * 2.5 * 2 + (1 << 20);
                size_t free_b = 0, total_b = 0;
                CK(cudaMemGetInfo(&free_b, &total_b));
                double budget = 0;
                if (expert_vram >= 0) budget = expert_vram * (double) (1ull << 30);
                else {
                    budget = std::max(0.0, (double)free_b - (256.0 + 64.0) * 1048576.0);   // stream reserve + floor
                    // The shared handoff workspace is enabled after constructing the stream.
                    // Charge a conservative extra allowance here (MiMo D4096/FF2048: 32 MiB),
                    // including the indexed partials and both layer descriptor tables.
                    if (window_handoff) {
                        const double n = g.n_embd, ff = g.moe_intermediate;
                        const double partials = std::max(std::ceil(n / 256) * ff, std::ceil(ff / 256) * n);
                        budget -= std::max(32. * 1048576., 64. * (6 * ff + 14 * n + 4 * partials) +
                            2. * g.n_layers * 3 * 512 * sizeof(K::Exl3Mat) + 1048576.);
                    }
                    // STRATA_EXL3_RECON_BATCH (C2): the batched weight reconstruction's slab is
                    // charged inside the expert stream's constructor, AFTER this probe - subtract
                    // it here so the arena's auto-fill budget accounts for it (2 x cap F16
                    // gate/up matrices, the down reuses region 0, plus the padded activations)
                    const char* rb = std::getenv("STRATA_EXL3_RECON_BATCH");
                    if (rb && rb[0] && rb[0] != '0') {
                        int cap = 8;
                        if (const char* cm = std::getenv("STRATA_EXL3_RECON_BATCH_MAX"))
                            cap = std::max(2, std::min(16, std::atoi(cm)));
                        const double one = (double) g.n_embd * (double) g.moe_intermediate * 2;
                        budget -= 2.0 * cap * one + 4096.0 * ((double) g.n_embd * 2 + 2.0 * g.moe_intermediate * 4 +
                                                              (double) g.moe_intermediate * 2 + (double) g.n_embd * 4);
                    }
                    budget = std::min(budget, (double)free_b);
                }
                slots = (int) std::min<double>((double) g.n_expert, budget / (per_expert * (double) moe_layers));
                slots = std::max(0, slots);
                per_expert_bytes = (size_t) per_expert;
            }
            (void) per_expert_bytes;
        }
        // TEST ONLY (STRATA_MIMO_TEST_EXPERT_SLOTS, set by the parity fixtures; the
        // production CLI never sets it): pin the arena's slot count for deterministic
        // placement matrices - the window-hit refresh gate is active >= T*k slots, so the
        // fixtures exercise both sides of that gate on purpose. Values above n_expert are
        // legal (the stream clamps residency to the distinct experts it routes; the
        // parity arms use the same headroom) and exist to make wider T windows eligible.
        if (const char* ts = std::getenv("STRATA_MIMO_TEST_EXPERT_SLOTS"); ts != nullptr && *ts != '\0') {
            char* end = nullptr;
            const long n = std::strtol(ts, &end, 10);
            if (end == ts || *end != '\0' || n < 0 || n > 64)
                throw std::invalid_argument("mimo: STRATA_MIMO_TEST_EXPERT_SLOTS must be a whole "
                                            "integer in [0, 64] (this fixture has " +
                                            std::to_string(g.n_expert) + " experts)");
            slots = (int) n;
        }
        experts = std::make_unique<K::Exl3ExpertStream>(dir, (int) g.n_embd, (int) g.n_layers, slots, st,
                                                        (size_t) expert_ram << 20, (size_t) 256 << 20,
                                                        /*layer_lo=*/first_moe, /*layer_hi=*/(int) g.n_layers,
                                                        chunk_in, "model.layers.", expert_backing);
        // the peer tier's split inputs (the tier itself is built by setup_peer, AFTER the optional
        // route profile parsed - its subset comes from the ranked pairs)
        primary_slots = experts->slots_per_layer();
        moe_layer_count = moe_layers;
        std::fprintf(stderr, "strata-mimo: experts: %d slot(s) per layer on the primary GPU, %lld MiB host cache\n",
                     primary_slots, (long long) expert_ram);

        // the window buffers, sized by the chunk (decode reuses the first rows)
        const int64_t T = chunk_in;
        auto fmalloc = [&](float** p, size_t elems) { CK(cudaMalloc(p, elems * 4)); buffer_bytes += elems * 4; };
        auto hmalloc = [&](uint16_t** p, size_t elems) { CK(cudaMalloc(p, elems * 2)); buffer_bytes += elems * 2; };
        fmalloc(&x, (size_t) T * g.n_embd);
        fmalloc(&h, (size_t) T * g.n_embd);
        hmalloc(&h16, (size_t) T * g.n_embd);
        fmalloc(&qf, (size_t) T * g.qk_dim());
        fmalloc(&kf, (size_t) T * max_kv * g.head_dim_qk);
        fmalloc(&vf, (size_t) T * max_kv * g.head_dim_qk);
        hmalloc(&q16, (size_t) T * g.qk_dim());
        fmalloc(&attn, (size_t) T * g.v_dim());
        hmalloc(&a16, (size_t) T * g.v_dim());
        fmalloc(&proj, (size_t) T * g.n_embd);
        fmalloc(&gg, (size_t) T * g.dense_intermediate);
        fmalloc(&uu, (size_t) T * g.dense_intermediate);
        hmalloc(&hb, (size_t) T * g.dense_intermediate);
        fmalloc(&parts, (size_t) T * topk * g.n_embd);
        fmalloc(&logits, (size_t) g.n_vocab);
        CK(cudaMalloc(&tok, T * 4));
        CK(cudaMalloc(&ids, (size_t) T * topk * 4));
        CK(cudaMalloc(&rw, (size_t) T * topk * 4));
        CK(cudaMalloc(&arg, 4));
        CK(cudaMalloc(&dpos, T * 8));
        CK(cudaMalloc(&dseq, T * 4));
        buffer_bytes += ((size_t) T * topk * 4) * 2 + T * 4 + 8 * T + 4 * T + 4;
        CK(cudaMalloc(&x16_scratch, (size_t) T * std::max({g.n_embd, g.v_dim(), g.dense_intermediate}) * 2));
        CK(cudaMalloc(&silu_f32, (size_t) T * g.dense_intermediate * 4));
        buffer_bytes += (size_t) T * std::max({g.n_embd, g.v_dim(), g.dense_intermediate}) * 2 +
                        (size_t) T * g.dense_intermediate * 4;
        {   // --dense-q8: the q8_1 activation image the MMVQ path quantizes into (8 columns, the
            // kernel's widest launch); the pack's own q8 views say whether the mode is on at all
            const bool any_q8 = !pack->layers.empty() &&
                                (pack->layers[0].q8q.valid() || pack->layers[0].q8o.valid() ||
                                 pack->lm_q8.valid());
            if (any_q8) {
                const int64_t widest = std::max<int64_t>({g.n_embd, g.v_dim(), g.dense_intermediate});
                CK(cudaMalloc(&q8_scratch, K::native_q8_1_bytes((int) widest, 8)));
                buffer_bytes += K::native_q8_1_bytes((int) widest, 8);
                std::fprintf(stderr, "strata-mimo: dense q8_0 blocks on the MMVQ path "
                                     "(--dense-q8; activations quantized to q8_1 per 32)\n");
            }
        }
        CK(cudaMallocHost(&h_host, (size_t) 16 * g.n_embd * 4));
        CK(cudaMallocHost(&ids_host, (size_t) T * topk * 4));   // the prefill path stages the WHOLE
                                                                  // chunk's router ids (T = the chunk)
        CK(cudaMallocHost(&parts_host, (size_t) 16 * topk * g.n_embd * 4));
        pos_stage.resize((size_t) T);

        // the first-light A/B knobs (defaults = the port's documented conventions)
        auto env_is = [](const char* name, const char* value) {
            const char* e = std::getenv(name);
            return e != nullptr && std::strcmp(e, value) == 0;
        };
        rope_gptj = env_is("STRATA_MIMO_ROPE", "gptj");
        theta_swap = env_is("STRATA_MIMO_THETA", "swap");
        sink_scaled = env_is("STRATA_MIMO_SINK", "scaled");
        gqa_interleave = env_is("STRATA_MIMO_GQA", "inter");
        if (gqa_interleave) {   // q head h goes to kv head h % n_kv: gather h's block position
            gqa_perm.resize((size_t) g.n_q_head);
            std::vector<int> next((size_t) max_kv, 0);
            for (int h = 0; h < (int) g.n_q_head; ++h) {
                const int kvh_n = (int) ((size_t) h % (size_t) std::max<int64_t>(1, max_kv));
                gqa_perm[(size_t) h] = kvh_n * (int) (g.n_q_head / max_kv) + next[(size_t) kvh_n]++;
            }
            CK(cudaMalloc(&q_perm_buf, (size_t) T * g.qk_dim() * 2));
            CK(cudaMalloc(&attn_perm_buf, (size_t) T * g.v_dim() * 4));
            buffer_bytes += (size_t) T * g.qk_dim() * 2 + (size_t) T * g.v_dim() * 4;
        }
        if (rope_gptj || gqa_interleave || theta_swap || sink_scaled)
            std::fprintf(stderr, "strata-mimo: A/B knobs active: rope=%s gqa=%s theta=%s sink=%s\n",
                         rope_gptj ? "gptj" : "neox", gqa_interleave ? "inter" : "block",
                         theta_swap ? "swap" : "as-config", sink_scaled ? "scaled" : "raw");
        if (const char* tv = std::getenv("STRATA_MIMO_TIME")) tm_on = std::strcmp(tv, "1") == 0;
        svc_time = env_is("STRATA_MIMO_SERVICE_TIME", "1");
        mtp_time = env_is("STRATA_MIMO_MTP_TIME", "1");
        df_trace = env_is("STRATA_MIMO_DFTRACE", "1");
        route_mono = env_is("STRATA_MIMO_ROUTE", "mono");
        decode_indexed = env_is("STRATA_MIMO_DECODE_INDEXED", "1");
        decode_device = env_is("STRATA_MIMO_DECODE_DEVICE", "1") &&
                         strata::core::exl3_flag("STRATA_EXL3_PRIMARY_PACK_PLAN", nullptr, true);
        decode_handoff = decode_device && env_is("STRATA_MIMO_DECODE_HANDOFF", "1");
        handoff_cpu = env_is("STRATA_MIMO_HANDOFF_CPU", "1");
        verify_host = env_is("STRATA_MIMO_VERIFY_HOST", "1");
        if (const char* v = std::getenv("STRATA_MIMO_CPU_MIN_MISSES")) {
            handoff_cpu_min = std::atoi(v);
            if (handoff_cpu_min < 1 || handoff_cpu_min > 64)
                throw std::invalid_argument("mimo: CPU_MIN_MISSES must be 1..64");
        }
        cache_statistics = env_is("STRATA_EXL3_CACHE_STATS", "1");
        route_digest = env_is("STRATA_MIMO_ROUTE_DIGEST", "1");
        if (const char* path = std::getenv("STRATA_MIMO_CACHE_TRACE")) {
            if (strata::core::exl3_flag("STRATA_MIMO_PEER_ADAPT", nullptr, false))
                throw std::invalid_argument("mimo: fixed-route cache trace requires static peer ownership");
            cache_trace = std::fopen(path, "wx");
            if (!cache_trace) throw std::runtime_error("mimo: cannot create cache trace (never overwritten)");
        }
        handoff_eager = env_is("STRATA_MIMO_HANDOFF_EAGER", "1");
        if (decode_device)
            std::fprintf(stderr, "strata-mimo: device decode: local FFN input/output stay on GPU; "
                                 "pinned indexed plans, cache admission and peer ownership retained\n");
        if (decode_indexed)
            std::fprintf(stderr, "strata-mimo: indexed decode: packed host-IO FFN at T=1 "
                                 "(FUSE applies; CPU tier keeps its established service)\n");
        CK(cudaMalloc(&route_scores, (size_t) g.n_expert * 4));
        buffer_bytes += (size_t) g.n_expert * 4;
        CK(cudaMallocHost(&diag_ids_host, (size_t) 16 * topk * 4));
        CK(cudaMallocHost(&diag_w_host, (size_t) 16 * topk * 4));
        if (window_handoff) {
            // Startup only: reserve the shared frame/workspaces before any request queues work.
            const bool log_memory = env_is("STRATA_MIMO_WINDOW_HANDOFF_LOG", "1");
            size_t before = 0, after = 0, total = 0;
            if (log_memory) CK(cudaMemGetInfo(&before, &total));
            experts->enable_decode_handoff(topk, 8);
            if (log_memory) {
                CK(cudaMemGetInfo(&after, &total));
                std::fprintf(stderr, "strata-mimo: window frame/workspaces enable device delta %lld bytes "
                                     "(cudaMemGetInfo; concurrent external allocations may affect it)\n",
                             (long long) before - (long long) after);
            }
            // The verify-window CPU overflow's buffers (STRATA_MIMO_WINDOW_CPU=1, preflighted
            // in load_engine): allocated HERE, before the worker thread starts - nothing in
            // the window arm may allocate under the service wait.  ~1 MiB host F32 beside
            // the canonical mmap; the CPU tier itself registers later (arm_window_cpu).
            if (window_cpu) {
                win_cpu_out.assign((size_t) strata::mimo::kWindowCpuMaxRoutes * g.n_embd, 0.0f);
                win_cpu_submit_logged.assign((size_t) g.n_layers, 0);
                std::fprintf(stderr, "strata-mimo: window CPU overflow armed%s: whole groups "
                                     "<= %d (min %d), cold non-resident misses only - peer-held, "
                                     "pending-incoming and resident groups never taken; works on "
                                     "the native and packed CPU tiers; T1 decode unchanged\n",
                             window_cpu_stats_on ? " (stats on)" : "",
                             window_cpu_max_groups, window_cpu_min_groups);
            }
            if (win_cpu_test_rows) {
                win_cpu_test_gpu.resize((size_t) strata::mimo::kWindowCpuMaxRoutes * g.n_embd);
                win_cpu_test_solo.resize((size_t) strata::mimo::kWindowCpuMaxRoutes * g.n_embd);
            }
            // The resident window-hit opt-in (STRATA_MIMO_WINDOW_GPU_HITS=1, preflighted in
            // load_engine). Geometry was proven by this arm (T=2..8, k <= 64/T, chunk >= 8)
            // and enable_decode_handoff(topk, 8) above reserved the 8-row frame, its 64-route
            // graph workspace and the miss-side runtime scratch - the hit arm adds NO new
            // allocation under the service wait. Tables are (re)declared per forward by
            // refresh_verify_graph(st, T); an active cache below T*k falls back to the
            // original all-miss window, byte for byte.
            if (window_gpu_hits) {
                experts->set_window_graph_hits(true);
                std::fprintf(stderr, "strata-mimo: window GPU hits armed%s: resident rows "
                                     "compute inline on the producer stream; the worker serves "
                                     "only miss/peer/CPU routes; active slots < T*k windows "
                                     "fall back to the all-miss handoff\n",
                             window_gpu_hits_stats_on ? " (stats on)" : "");
            }
            if (window_device_misses) {
                // The device complement rides the SAME pinned/packed dispatch slabs the T1
                // device arm requires; without them the arm would upload pageable plans -
                // the exact 2026-10-28 regression it exists to avoid - so refuse loudly
                // (the engine never silently claims engagement).
                if (!experts->window_complement_device_ready(8, topk))
                    throw std::invalid_argument(
                        "mimo: STRATA_MIMO_WINDOW_DEVICE_MISSES=1 needs the packed pinned-plan "
                        "dispatch (STRATA_EXL3_PRIMARY_PACK_PLAN / STRATA_EXL3_PINNED_PLAN on, "
                        "window routes within the reserved capacity) - refusing instead of "
                        "running the device arm on pageable plans");
                std::fprintf(stderr, "strata-mimo: window device misses armed: EVERY mapped "
                                     "verify-window service call (the complement AND the plain "
                                     "no-peer/no-CPU window) reads the producer's device "
                                     "activation and scatters into parts on device (pinned "
                                     "packed plans; host arithmetic kept)\n");
            }
            start_decode_worker();
            handoff_ready = true;
            std::fprintf(stderr, "strata-mimo: mapped verify-window handoff enabled (T=2..8, empty GPU hits)\n");
        } else if (window_cpu) {
            // preflight_window_cpu proved the env switch; eligibility is geometry (topk and
            // chunk are known only here).  Refuse loudly rather than silently not using CPU.
            throw std::invalid_argument(
                "mimo: STRATA_MIMO_WINDOW_CPU=1 needs an eligible mapped window shape: "
                "T=2..8, k <= 64/T and chunk >= 8 (this model's topk/chunk do not qualify)");
        } else if (window_gpu_hits) {
            // Same refusal rule for the hit arm: the env switch was proven, the shape was not.
            throw std::invalid_argument(
                "mimo: STRATA_MIMO_WINDOW_GPU_HITS=1 needs an eligible mapped window shape: "
                "T=2..8, k <= 64/T and chunk >= 8 (this model's topk/chunk do not qualify)");
        }
        if (peer_device_results)
            std::fprintf(stderr, "strata-mimo: peer device results armed: the handoff worker "
                                 "collects the peer's rows through the mapped F16 frame "
                                 "(converted+scattered on the primary GPU; no CPU conversion, "
                                 "no whole-slab F32 upload)\n");
        return true;
    }


    void reset() {
        if (dflash) dflash->reset();
        dflash_serial_ms = 0; dflash_calibration = dflash_backoff = dflash_slow = 0;
        pos = 0;
        if (mtp != nullptr) mtp->rewind(0);
    }

    /// Per-forward window-hit participation (STRATA_MIMO_WINDOW_GPU_HITS_STATS=1): windows
    /// served, how many actually carried hits, all-hit windows that skipped host service
    /// entirely, structural fallbacks (refresh declared no tables for that T) and the
    /// route split. Main thread only, after the worker drained - like report_window_cpu.
    void report_window_gpu_hits() {
        if (!window_gpu_hits_stats_on) return;
        const auto& s = win_gpu_hits_stats;
        const auto& r = win_gpu_hits_reported;
        std::fprintf(stderr, "strata-mimo: window_gpu_hits forward windows=%lld hit_windows=%lld "
                             "all_hit=%lld fallback=%lld routes hit=%lld host_served=%lld peer=%lld\n",
                     (long long)(s.windows - r.windows), (long long)(s.hit_windows - r.hit_windows),
                     (long long)(s.all_hit_windows - r.all_hit_windows),
                     (long long)(s.fallback_windows - r.fallback_windows),
                     (long long)(s.hit_routes - r.hit_routes),
                     (long long)(s.host_served - r.host_served),
                     (long long)(s.peer_routes - r.peer_routes));
        win_gpu_hits_reported = s;
    }

    /// Per-forward window-router-split participation (STRATA_MIMO_WINDOW_ROUTER_SPLIT_STATS=1):
    /// how many of THIS forward's windows routed through the split kernels and how many MoE
    /// layers that was - the non-vacuous proof the opt-in actually engaged (with the arm off
    /// the line is absent, and the counters stay zero).  Main thread only, like the hits report.
    void report_window_router_split() {
        if (!window_router_split_stats_on) return;
        std::fprintf(stderr, "strata-mimo: window_router_split forward windows=%lld layers=%lld "
                             "(T2..8 all-rows forwards only)\n",
                     (long long)(window_split_windows - win_split_reported_windows),
                     (long long)(window_split_layers - win_split_reported_layers));
        win_split_reported_windows = window_split_windows;
        win_split_reported_layers = window_split_layers;
    }

    /// One persistent host service thread, armed once per decode forward. The submitting
    /// thread queues the layer loop; the worker services only missing/peer routes.
    void start_decode_worker() {
        int device = 0;
        CK(cudaGetDevice(&device));
        CK(cudaStreamCreateWithFlags(&handoff_stream, cudaStreamNonBlocking));
        handoff_worker = std::thread([this, device] {
            CK(cudaSetDevice(device));
            std::unique_lock<std::mutex> lock(handoff_mu);
            for (;;) {
                handoff_cv.wait(lock, [&] { return handoff_pending || handoff_stop; });
                if (handoff_stop) break;
                const uint32_t ring = handoff_ring;
                handoff_pending = false;
                lock.unlock();
                std::exception_ptr error;
                try {
                    for (int layer = 0; layer < g.n_layers; ++layer) if (g.is_moe(layer)) {
                        const int32_t* routes = nullptr;
                        const float* activation = nullptr;
                        const uint8_t* hits = nullptr;
                        int tokens = 1;
                        // Never query the producer here: a pageable token readback can hold
                        // its HIP stream lock while waiting for our completion flag.
                        const auto svc_c0 = svc_time ? Clock::now() : Clock::time_point{};
                        experts->collect_decode_handoff(layer, ring, handoff_stream, routes, activation, hits, &tokens);
                        const auto svc_c1 = svc_time ? Clock::now() : Clock::time_point{};
                        if (svc_time) {
                            svc_collect_ms += ms_since(svc_c0);
                            ++svc_collect_calls;
                        }
                        if (tokens > 1) serve_window_routes(layer, routes, activation, tokens, handoff_stream, hits);
                        else serve_decode_routes(layer, routes, activation, hits, handoff_stream);
                        // Publish after the service kernels, without a per-layer host sync.
                        // The NEXT routing publication proves both streams consumed this
                        // layer and makes the shared pinned plans/scratch safe to reuse.
                        bool misses = false;
                        for (int j = 0; j < tokens * topk; ++j) misses |= !hits[j];
                        const auto svc_d0 = svc_time ? Clock::now() : Clock::time_point{};
                        experts->complete_decode_handoff(layer, ring, misses ? handoff_stream : nullptr);
                        if (svc_time) svc_done_ms += ms_since(svc_d0);
                    }
                } catch (...) {
                    error = std::current_exception();
                    // Failfast FIRST (the producer's wait must release), then - on THIS
                    // owner thread, after the unwind - close the host-service bracket the
                    // aborted frame left open. abort_decode_handoff itself never touches
                    // host_active (it would race this thread's own bracket).
                    experts->abort_decode_handoff();
                    experts->recover_host_service();
                    // R3 (STRATA_MIMO_PEER_DEVICE_RESULTS): a peer device-results frame
                    // may be IN FLIGHT - the launch already wrote rank[]/expected and
                    // queued the transfer + ready signal on the peer stream, and this
                    // exception fired before (or around) the service-stream wait+copy.
                    // The frame's wait kernel exits only on ready==expected or the
                    // aborted flag, and a failed peer stream may never signal: release
                    // the watchdog NOW, host stores only, BEFORE any later sync of the
                    // service stream (stop_decode_worker's drain, teardown) could block
                    // on that kernel forever. Idempotent and permanent by design - the
                    // worker error is rethrown at the next forward boundary, so nothing
                    // may enqueue another frame afterwards anyway.
                    if (peer_output) peer_output->release();
                }
                lock.lock();
                handoff_error = error;
                handoff_busy = false;
                handoff_cv.notify_all();
            }
        });
    }

    void stop_decode_worker() {
        if (handoff_worker.joinable()) {
            {
                std::lock_guard<std::mutex> lock(handoff_mu);
                handoff_stop = true;
                experts->abort_decode_handoff();
                // R3: publish the frame watchdog BEFORE the join. Nothing in the CURRENT
                // worker flow host-syncs the service stream behind a queued wait kernel
                // (every sync - the complement's D2H, the device arm's sync_compute -
                // precedes that layer's copy enqueue, and collect's doorbell spin is
                // bounded at 30 s), so the join returns today even with a stuck frame.
                // But the ordering is load-bearing and silent: any future reordering that
                // puts a host sync behind an unsignaled wait kernel would turn this join
                // into a permanent deadlock - the release was previously AFTER it. The
                // watchdog is host-stores only and idempotent, safe to publish while the
                // worker runs (an in-flight copy kernel then skips its writes - the
                // producer's wait was aborted above, so those rows are discarded anyway),
                // and nothing can enqueue another frame after the abort + stop.
                if (peer_output) peer_output->release();
            }
            handoff_cv.notify_all();
            handoff_worker.join();
        }
        // Idempotent repeat (the worker's catch may have released already): keep the
        // invariant explicit - the stream drain below NEVER runs with an armed frame.
        if (peer_output) peer_output->release();
        if (handoff_stream) {
            (void) cudaStreamSynchronize(handoff_stream);
            (void) cudaStreamDestroy(handoff_stream);
            handoff_stream = nullptr;
        }
    }

    void serve_window_routes(int layer, const int32_t* routes, const float* activation,
                             int tokens, cudaStream_t service, const uint8_t* hits) {
        if (!strata::mimo::window_handoff_eligible(tokens, topk, (int) g.n_embd, chunk, true))
            throw std::runtime_error("mimo: invalid mapped window shape");
        const int count = tokens * topk;
        count_routes(layer, routes, tokens);
        uint8_t mask[64] = {};
        int remote = 0;
        // The opt-in runs its branch ONLY when the feature is armed AND this publication
        // ACTUALLY carries hits. Every other window - feature off, structural fallback,
        // zero-resident - takes the ORIGINAL branch below, control flow and bytes alike.
        int hit_routes = 0;
        if (window_gpu_hits)
            for (int r = 0; r < count; ++r) hit_routes += hits[r] != 0;
        const bool hit_arm = window_gpu_hits && hit_routes > 0;
        for (int r = 0; r < count; ++r) {
            if (!hit_arm) {
                // Match the old window's clipped-ID ownership check, never indexing a table
                // with an unchecked ID. Ownership is per expert, hence whole groups agree.
                mask[r] = routes[r] >= 0 && routes[r] < g.n_expert && peer && peer->valid() &&
                          peer->has(layer, routes[r]);
            } else {
                // A published hit computes inline on the producer stream: the peer never
                // takes it (the refresh's tables exclude owned/pending experts anyway - this
                // keeps the transport disjoint even if ownership moved inside the frame).
                mask[r] = !hits[r] && routes[r] >= 0 && routes[r] < g.n_expert && peer && peer->valid() &&
                          peer->has(layer, routes[r]);
            }
            peer_kind[r] = mask[r] ? 2 : -1;
            remote += mask[r];
        }
        std::string error;
        // R3 (STRATA_MIMO_PEER_DEVICE_RESULTS): pass the shared device-output frame with the
        // WINDOW launch - the peer stream then enqueues the compact F16 transfer + ready
        // signal (destinations are this window's route indices, < the frame's capacity).
        if (remote && !peer->launch(layer, activation, routes, tokens, topk, peer_kind.data(), error,
                                    peer_device_results ? peer_output.get() : nullptr))
            throw std::runtime_error("mimo window peer launch: " + error);
        if (peer_device_results && remote == 0)
            ++peer_dev_zero_windows;   // test seam: the no-remote path with the flag on
        if (win_test_peer_fault == 1 && peer_device_results && remote)
            // TEST ONLY: the frame is in flight (see win_test_peer_fault) - unwind through
            // the worker's abort/release path right here, before any copy is enqueued.
            throw std::runtime_error("test peer frame fault: window launch queued the device "
                                     "frame (remote=" + std::to_string(remote) + " of " +
                                     std::to_string(count) + " routes, layer " +
                                     std::to_string(layer) + ")");
        strata::mimo::WindowCpuPlan plan;
        struct JoinCpu {
            K::cpu::Exl3CpuMoe* moe;
            bool armed = false;
            // Destructor path runs during unwinding: best-effort drain that must never
            // terminate (an implicit-noexcept throwing member would) and never mask the
            // original exception.  The join itself is non-throwing today; the guard is the
            // robustness contract, like the joint bench's drain.  The explicit join() below
            // DOES propagate a join error on the normal path.
            ~JoinCpu() {
                if (!armed) return;
                try { moe->run_rows_join(); } catch (...) {}
            }
            void join() { if (armed) { moe->run_rows_join(); armed = false; } }
        } cpu_join{cpu_moe.get()};
        bool bg = false;
        if (window_cpu) {
            struct Query { Engine* engine; int layer; } query{this, layer};
            strata::mimo::WindowCpuResidency oracle;
            oracle.user = &query;
            oracle.primary_hold = [](void* u, int32_t e) {
                auto& q = *static_cast<Query*>(u);
                return q.engine->experts->holds(q.layer, e);
            };
            oracle.peer_owned = [](void* u, int32_t e) {
                auto& q = *static_cast<Query*>(u);
                return (q.engine->peer && q.engine->peer->valid() && q.engine->peer->has(q.layer, e)) ||
                       q.engine->experts->excluded_elsewhere(q.layer, e);
            };
            plan = strata::mimo::plan_window_cpu(routes, tokens, topk, (int) g.n_expert,
                                                oracle, window_cpu_max_groups, window_cpu_min_groups);
            if (win_cpu_test_pending != nullptr)
                for (int i = 0; i < plan.groups; ++i)   // pending-incoming stays GPU-scratch
                    if (win_cpu_test_pending[(size_t) layer * (size_t) g.n_expert +
                                             (size_t) plan.group_id[i]] != 0)
                        throw std::runtime_error("window CPU test selected a pending-incoming "
                                                 "expert (the exclusion union failed)");
            if (plan.ok && hit_arm)
                for (int i = 0; i < plan.groups; ++i)   // a hit is a RESIDENT group: the
                    for (int r = 0; r < count; ++r)     // planner's holds() oracle must have
                        if (routes[r] == plan.group_id[i] && hits[r])   // kept it off the CPU
                            throw std::runtime_error("window CPU plan selected a resident-hit "
                                                     "expert (the holds() exclusion failed)");
            if (plan.ok && plan.rows) {
                for (int i = 0; i < plan.rows; ++i) {
                    win_cpu_ids[i] = plan.group_id[plan.row_group[i]];
                    win_cpu_tok[i] = plan.row_token[i];
                    win_cpu_row[i] = i;
                }
                const auto start = window_cpu_stats_on ? Clock::now() : Clock::time_point{};
                try {
                    if (win_cpu_test_fault == 1) throw std::runtime_error("test CPU submit before publish");
                    cpu_moe->run_rows_bg(layer, activation, g.n_embd, win_cpu_ids, win_cpu_tok,
                                         win_cpu_row, plan.rows, win_cpu_out.data(), g.n_embd);
                    cpu_join.armed = bg = true; // publish the skip ONLY after successful submission
                } catch (const std::exception& ex) {
                    if (!win_cpu_submit_logged[(size_t) layer]) {
                        win_cpu_submit_logged[(size_t) layer] = 1;
                        std::fprintf(stderr, "mimo window CPU layer %d: %s; retaining GPU rows\n", layer, ex.what());
                    }
                }
                if (window_cpu_stats_on) win_cpu_stats.submit_ms += ms_since(start);
            }
        }
        for (int r = 0; r < count; ++r)
            win_cpu_skip[r] = uint8_t(mask[r] | (bg ? plan.cpu_skip[r] : 0));
        if (bg && win_cpu_test_fault == 2) throw std::runtime_error("test GPU failure after CPU publish");
        if (bg && win_cpu_test_fault == 3 && hit_arm && hit_routes > 0)
            // TEST ONLY (STRATA_MIMO_TEST_WINDOW_CPU_FAULT=gpu-hits): the same
            // after-publish fault, but it fires ONLY on a window that actually carries
            // inline hits beside the live CPU job - a run whose windows never combine
            // the two completes normally, so a test expecting this failure cannot pass
            // vacuously. The counts ride the message for the log.
            throw std::runtime_error("test GPU failure after CPU publish (window hits active: "
                                     + std::to_string(hit_routes) + " hit / "
                                     + std::to_string(count - hit_routes) + " host routes)");
        // Rows the HOST slab still owes their bytes: non-hit (the producer wrote those) and
        // non-peer (the peer's masked copy lands those) - the complement's rows AND the
        // CPU tier's merged rows alike. Computed only for the armed-with-hits branch;
        // every other window keeps the ORIGINAL whole-slab staging below.
        // STRATA_MIMO_WINDOW_DEVICE_MISSES reshapes the split, never the row ownership: the
        // DEVICE arm serves EVERY window service call - the complement AND the plain
        // no-peer/no-CPU window the host arm ran through run_multi (there the peer-owned /
        // cpu_skip / graph-hit drops are provably no-ops, so run_multi_complement_device
        // computes the identical plan) - and scatters its rows straight into parts, so the
        // host slab owes only the CPU overlay's rows. `ng` counts the rows the device arm
        // scatters (exactly the surviving plan rows: non-hit, non-peer, non-CPU;
        // pending-incoming included - they stay GPU scratch). The test-rows oracle
        // (STRATA_MIMO_TEST_WINDOW_CPU_ROWS) validates parts_host rows and therefore
        // forces the host arm.
        const bool miss_device = window_device_misses && !win_cpu_test_rows;
        int serve_row[64];
        int ns = 0;
        int ng = 0;
        if (miss_device) {
            for (int r = 0; r < count; ++r) {
                if (mask[r]) continue;                       // the peer's rows
                if (hit_arm && hits[r]) continue;            // the producer's inline rows
                if (bg ? plan.cpu_skip[r] != 0 : false) {
                    serve_row[ns++] = r;                     // the CPU overlay's rows
                    continue;
                }
                ++ng;                                        // a GPU row the device arm scatters
            }
        } else if (hit_arm)
            for (int r = 0; r < count; ++r)
                if (!hits[r] && !mask[r]) serve_row[ns++] = r;
        // run_multi_complement drains its D2H before returning. Pending peer claims are
        // excluded only from CPU planning, NOT added to this hard skip (GPU scratch owes them).
        // The ORIGINAL gate runs for every window without hits; only an ARMED window with
        // ACTUAL hits may skip the call, and only when every row is a hit or a peer row.
        // The device arm's gate is the same rule on its own row set: skip only when no GPU
        // row remains (every row a hit, a peer row or a CPU row).
        const auto serve_primary = [&]() {
            if (miss_device)
                return experts->run_multi_complement_device(layer, h, routes, tokens, topk, parts,
                                                           service, win_cpu_skip);
            if (remote || bg)
                return experts->run_multi_complement(layer, activation, routes, tokens, topk,
                                                    parts_host, service, win_cpu_skip);
            return experts->run_multi(layer, activation, routes, tokens, topk, parts_host, service);
        };
        const bool skip_service = miss_device ? (ng == 0) : (hit_arm && ns == 0);
        const bool ok = skip_service ? true : serve_primary();
        if (miss_device && !skip_service) {
            ++wdm_calls;
            wdm_routes += ng;
            wdm_cpu_rows += ns;
            // NET bus bytes removed for THIS call, from the source-proven transport shapes
            // (never bandwidth-derived): the host arm's mapped F32 activation H2D (whole
            // window) + its whole-tile result D2H (the surviving plan's rows = ng) + its
            // result H2D (compact served rows on the hit arm, the whole slab otherwise)
            // MINUS the bytes the device arm still crosses (the CPU overlay's compact
            // result H2D, ns rows - everything else stays on the device).
            wdm_bytes_avoided += (double) tokens * g.n_embd * 4 +
                                 (double) ng * g.n_embd * 4 +
                                 (double) (hit_arm ? (ng + ns) : count) * g.n_embd * 4 -
                                 (double) ns * g.n_embd * 4;
        }
        const auto join_start = window_cpu_stats_on ? Clock::now() : Clock::time_point{};
        cpu_join.join(); // false AND throw paths drain before worker abort releases the frame
        if (window_cpu_stats_on && bg) win_cpu_stats.join_ms += ms_since(join_start);
        if (!ok) throw std::runtime_error("mimo window primary service failed");
        if (bg) {
            for (int i = 0; i < plan.rows; ++i)
                std::memcpy(parts_host + (size_t) plan.row_route[i] * g.n_embd,
                            win_cpu_out.data() + (size_t) i * g.n_embd, (size_t) g.n_embd * sizeof(float));
            if (window_cpu_stats_on) {
                ++win_cpu_stats.jobs;
                win_cpu_stats.groups += plan.groups;
                win_cpu_stats.rows += plan.rows;
            }
        }
        if (win_cpu_test_rows) {
            // TEST ONLY (STRATA_MIMO_TEST_WINDOW_CPU_ROWS=1, set by run_verify_logits; the
            // production CLI never sets it).  Three operator proofs, ordered cheapest-first:
            //   1. WIRING: the plan's rows recomputed STANDALONE on the same tier (blocking
            //      run_rows, pool idle after the join, into a separate buffer - win_cpu_out's
            //      saved context is never recomputed over) must be BITWISE identical to what
            //      the background job produced.  Any difference is an (ids, token, dense-row,
            //      overlay) wiring bug - the native-vs-band8 logits check cannot see one,
            //      because both arms would compute the same wrong rows.
            //   2. GPU retention: the complement's non-CPU rows must equal a fresh run_multi's
            //      bitwise (this oracle may admit/evict as a diagnostic - test flag only).
            //   3. CPU-vs-GPU arithmetic gate (3% RMS / 12% max, the existing real-model
            //      oracle class): printed with full context at the failing row before throwing.
            if (bg) {
                cpu_moe->run_rows(layer, activation, g.n_embd, win_cpu_ids, win_cpu_tok,
                                  win_cpu_row, plan.rows, win_cpu_test_solo.data(), g.n_embd);
                for (int i = 0; i < plan.rows; ++i)
                    if (std::memcmp(win_cpu_out.data() + (size_t) i * g.n_embd,
                                    win_cpu_test_solo.data() + (size_t) i * g.n_embd,
                                    (size_t) g.n_embd * sizeof(float)) != 0)
                        throw std::runtime_error("window CPU test standalone-CPU row differs "
                                                 "(wiring: ids/token/dense-row mapping)");
            }
            // With the hit arm active, the GPU oracle must compute EVERY row: end the host
            // bracket first so the oracle's run_multi does not drop the hit groups against
            // this record (their producer-computed bytes are validated by the stream's own
            // byte-exact oracle, exl3_verify_graph_test's memcmp class). The oracle may
            // re-observe/re-admit as a diagnostic - test flag only.
            if (hit_arm) experts->verify_graph_end_host(layer);
            if (!experts->run_multi(layer, activation, routes, tokens, topk, win_cpu_test_gpu.data(), service))
                throw std::runtime_error("window CPU test GPU oracle failed");
            int worst = -1;
            double worst_rms = 0, worst_max = 0, worst_maxref = 0, worst_ref2 = 0;
            for (int r = 0; r < count; ++r) {
                if (mask[r]) continue;
                if (hit_arm && hits[r]) continue;   // producer-computed row, not in parts_host
                const float* a = parts_host + (size_t) r * g.n_embd;
                const float* b = win_cpu_test_gpu.data() + (size_t) r * g.n_embd;
                if (!bg || !plan.cpu_skip[r]) {
                    if (std::memcmp(a, b, (size_t) g.n_embd * sizeof(float)))
                        throw std::runtime_error("window CPU test retained GPU row differs");
                } else if (win_cpu_test_gate) {   // the arithmetic gate: real-geometry
                    // fixtures only (see win_cpu_test_gate above for why the random H128
                    // fixture defers it - measured numbers in mimo_band8_parity.cpp)
                    double err2 = 0, ref2 = 0, maxerr = 0, maxref = 0;
                    for (int j = 0; j < g.n_embd; ++j) {
                        const double d = (double) a[j] - b[j]; err2 += d*d; ref2 += (double)b[j]*b[j];
                        maxerr = std::max(maxerr, std::abs(d)); maxref = std::max(maxref, std::abs((double)b[j]));
                    }
                    const double rms = std::sqrt(err2 / std::max(ref2, 1e-30));
                    const double relmax = maxerr / std::max(maxref, 1e-30);
                    const bool bad = !std::isfinite(err2) || rms > .03 || relmax > .12;
                    if (bad && worst < 0) {   // print the FIRST failing row with full context
                        int plan_row = -1;
                        for (int i = 0; i < plan.rows; ++i)
                            if (plan.row_route[i] == r) { plan_row = i; break; }
                        // Activation-quantization evidence: the GPU rounds the F32 input to
                        // F16 before its GEMV; the CPU reads raw F32.  This is the input-side
                        // share of the divergence, printed, not assumed.
                        const float* x = activation + (size_t) (r / topk) * g.n_embd;
                        double xr2 = 0, xd2 = 0;
                        for (int j = 0; j < g.n_embd; ++j) {
                            const double g16 = K::f32_from_f16(K::f16_from_f32(x[j]));
                            const double d = (double) x[j] - g16;
                            xd2 += d * d; xr2 += (double) x[j] * x[j];
                        }
                        std::fprintf(stderr,
                            "window CPU diag: layer=%d T=%d k=%d route=%d token=%d expert=%d "
                            "plan_row=%d H=%lld rel_rms=%.6f rel_max=%.6f (gate 0.03/0.12) "
                            "abs_maxerr=%.6g abs_maxref=%.6g ref_l2=%.6g out_l2=%.6g "
                            "act_f16_relrms=%.6g\n",
                            layer, tokens, topk, r, r / topk,
                            plan_row >= 0 ? (int) plan.group_id[plan.row_group[plan_row]] : -1,
                            plan_row, (long long) g.n_embd, rms, relmax, maxerr, maxref,
                            std::sqrt(ref2), std::sqrt(err2 + ref2),
                            std::sqrt(xd2 / std::max(xr2, 1e-30)));
                    }
                    if (rms > worst_rms) {
                        worst_rms = rms; worst_max = relmax; worst_maxref = maxref;
                        worst_ref2 = ref2; worst = r;
                    }
                    if (bad) {
                        std::fprintf(stderr,
                            "window CPU diag: failing window summary layer=%d T=%d k=%d "
                            "worst_route=%d worst_rel_rms=%.6f worst_rel_max=%.6f "
                            "worst_abs_maxref=%.6g worst_ref_l2=%.6g\n",
                            layer, tokens, topk, worst, worst_rms, worst_max, worst_maxref,
                            std::sqrt(worst_ref2));
                        throw std::runtime_error("window CPU test row exceeds 3% RMS / 12% max oracle");
                    }
                }
            }
        }
        if (!hit_arm && !miss_device) {
            // The original transport (feature off, fallback and zero-resident windows
            // alike): the whole slab - every row is the host's or the CPU tier's to
            // produce - then the peer's masked copy over its own rows. With the device
            // miss arm on this NEVER runs: every window's GPU rows were scattered on the
            // device, and uploading the slab would clobber them with stale entries.
            CK(cudaMemcpyAsync(parts, parts_host, (size_t) count * g.n_embd * 4,
                               cudaMemcpyHostToDevice, service));
        } else if (ns > 0) {
            // Producer-hit rows and these rows are written by DIFFERENT streams, and the
            // hit kernels may well still be running while this service stages: what makes
            // the transport safe is DISJOINT ROW WRITES (the producer writes only hit
            // rows, this path stages only non-hit/non-peer rows, and the producer's
            // decode_wait - ordered before its combine - observes the worker's completion
            // after these uploads). A whole-slab upload would break that disjointness and
            // clobber producer rows with the host slab's untouched entries: stage ONLY
            // the served rows - compact them to the front of the pinned slab, upload
            // once into the 64-row window staging buffer and scatter to their route slots
            // (the T1 service's proven transport; ascending rows never overwrite a source).
            for (int i = 0; i < ns; ++i)
                if (serve_row[i] != i)
                    std::memmove(parts_host + (size_t) i * g.n_embd,
                                 parts_host + (size_t) serve_row[i] * g.n_embd,
                                 (size_t) g.n_embd * sizeof(float));
            CK(cudaMemcpyAsync(peer_rows_dev, parts_host, (size_t) ns * g.n_embd * 4,
                               cudaMemcpyHostToDevice, service));
            CK(cudaMemcpyAsync(ids, serve_row, (size_t) ns * 4, cudaMemcpyHostToDevice, service));
            k_scatter_primary_rows<<<(unsigned) (((size_t) ns * g.n_embd + kThreads - 1) / kThreads),
                                     kThreads, 0, service>>>(parts, peer_rows_dev, ids, ns, g.n_embd);
            CK(cudaGetLastError());
        }
        // ns == 0 with the arm on: every row is a producer hit or a peer row - nothing to
        // stage. Participation counters are opt-in-only (feature armed; the test EXPECT
        // seam reads them); the print is further gated on STRATA_MIMO_WINDOW_GPU_HITS_STATS.
        if (win_cpu_test_pending != nullptr) {
            bool pending_served = false;
            for (int r = 0; r < count; ++r)
                if (win_cpu_test_pending[(size_t) layer * (size_t) g.n_expert +
                                         (size_t) routes[r]] != 0 &&
                    !(hit_arm && hits[r] != 0) && !mask[r] && !(bg ? plan.cpu_skip[r] : 0))
                    pending_served = true;   // not a hit, not CPU, not peer: the GPU
            if (hit_routes > 0 && bg && pending_served)   // complement's scratch row
                ++win_gpu_hits_stats.simultaneous;
        }
        if (window_gpu_hits) {
            auto& s = win_gpu_hits_stats;
            ++s.windows;
            s.hit_windows += hit_routes > 0;
            s.all_hit_windows += hit_routes == count;
            s.fallback_windows += experts->window_hit_tokens() < tokens;
            s.hit_routes += hit_routes;
            // HOST-served rows this window actually uploaded: the compact set when the
            // hit arm is live (complement rows + CPU rows), otherwise the whole-slab
            // upload's rows (count minus the peer's). Counts include the CPU tier's
            // merged rows on purpose - they are host-produced either way.
            s.host_served += miss_device ? ns : (hit_arm ? ns : (count - remote));
            s.peer_routes += remote;
        }
        if (remote) {
            if (peer_device_results) {
                // R3 device transport: the peer's F16 rows are already in the shared mapped
                // frame (the launch enqueued the compact transfer + ready signal on the peer
                // stream). finish_deferred kicks the peer driver once - never spins - then
                // the frame's wait+copy pair lands on the SERVICE stream: the wait releases
                // when the peer signals ready, the copy converts F16->F32 on the primary GPU
                // and writes ONLY the peer's rows into parts.
                //
                // Same-frame rank[] reuse across layers - the proof the next launch may
                // overwrite rank[] on the host: the copy kernel reads frame->rank[]; the
                // doorbell (complete_decode_handoff) is enqueued AFTER it on this same
                // stream, so it executes only once the copy finished reading; the
                // producer's decode_wait for this layer observes that doorbell before it
                // publishes layer L+1; the worker's collect for L+1 spins on THAT
                // publication, and only then calls the next launch. So the host-side
                // rank[] rewrite at layer L+1 is causally ordered behind the copy kernel's
                // completed reads at layer L - no overwrite of a live frame, and the copy
                // writes only the peer's destination rows, never the producer's inline
                // hits, the device complement's scatter or the CPU overlay's compact rows
                // (disjoint destinations by construction of mask[]/win_cpu_skip/hits).
                // The one precondition: the doorbell must ride THIS service stream
                // whenever a peer row exists - it does, because every peer row has
                // hits[r]==0, so the `misses` scan above is true and the release is
                // stream-ordered, never the immediate host write.
                if (peer_output == nullptr)
                    throw std::runtime_error("mimo window peer device results: transport missing "
                                             "(setup_peer must open it or fail the engine)");
                if (!peer->finish_deferred(error))
                    throw std::runtime_error("mimo window peer deferred finish: " + error);
                // TEST ONLY (fault=desync): advance expected HERE - after the launch queued
                // the ready signal carrying the OLD value, BEFORE the wait kernel below is
                // enqueued - so the queued wait's target provably has no signal (the host
                // write precedes the enqueue; no race with the kernel's initial read, which
                // an after-copy bump WOULD have). WORKER-ERROR RELEASE TEST: the throw below
                // unwinds through abort+release and the teardown drains - this does NOT
                // prove a mid-forward async hardware fault recovers by itself (the
                // last-layer hang documented at win_test_peer_fault stays fatal).
                if (win_test_peer_fault == 3) peer_output->test_desynchronize();
                peer_output->copy(parts, count, nullptr, 0, service);
                if (win_test_peer_fault >= 2)
                    // TEST ONLY (see win_test_peer_fault): the wait+copy pair is QUEUED on
                    // this service stream right now (2 with a possibly-signaled frame, 3
                    // with a provably-unsignaled one) - without the abort/release unwind
                    // the teardown would hang on that kernel, with it the kernel exits.
                    throw std::runtime_error("test peer frame fault: service wait+copy "
                                             "queued (fault=" +
                                             std::to_string(win_test_peer_fault) + ", remote=" +
                                             std::to_string(remote) + " of " +
                                             std::to_string(count) + " routes, layer " +
                                             std::to_string(layer) + ")");
                ++peer_dev_frames;
                peer_dev_rows += remote;
                // The old chain's host work this window skipped, from the source-proven
                // shapes: the WHOLE-slab F32 H2D (count rows, garbage included) and the CPU
                // conversion of every returned row. The compact F16 transfer still crosses
                // the peer's link (same bytes as before), and the new path uploads nothing,
                // so gross == net here.
                peer_dev_upload_avoided += (double) count * g.n_embd * 4;
                peer_dev_convert_avoided += remote;
            } else {
                if (!peer->finish(peer_rows_host, error))
                    throw std::runtime_error("mimo window peer finish: " + error);
                // Pinned mask lifetime extends to the next publication, unlike the stack mask.
                auto* pinned_mask = reinterpret_cast<uint8_t*>(peer_w_host);
                std::copy_n(mask, count, pinned_mask);
                CK(cudaMemcpyAsync(peer_mask_dev, pinned_mask, count, cudaMemcpyHostToDevice, service));
                CK(cudaMemcpyAsync(peer_rows_dev, peer_rows_host, (size_t) count * g.n_embd * 4,
                                   cudaMemcpyHostToDevice, service));
                k_copy_masked_rows<<<(unsigned) (((size_t) count * g.n_embd + kThreads - 1) / kThreads),
                                     kThreads, 0, service>>>(parts, peer_rows_dev, peer_mask_dev, count, g.n_embd);
                CK(cudaGetLastError());
            }
        }
        // TEST ONLY (STRATA_MIMO_TEST_DUMP_WINDOW): the env is read ONCE (the worker is a
        // single thread; -2 = unread, -1 = unset/off, -3 = every window, else that index)
        // so the default-off hot path costs one integer compare, not getenv+strcmp+atoi.
        if (win_dump_at == -2) {
            const char* dw = std::getenv("STRATA_MIMO_TEST_DUMP_WINDOW");
            if (dw == nullptr || dw[0] == '\0') win_dump_at = -1;
            else if (std::strcmp(dw, "-1") == 0) win_dump_at = -3;
            else win_dump_at = std::atoi(dw);
        }
        if (win_dump_at != -1 && (win_dump_at == -3 || win_dump_index == win_dump_at)) {
            // TEST ONLY: a deterministic checksum of this window's parts rows, read back
            // AFTER every service-stream write (complement scatter, CPU overlay upload and
            // the peer collection alike) - pinpoints the first diverging layer between two
            // arms of a transport A/B. The syncs are test-only, after publication ordering.
            CK(cudaStreamSynchronize(service));
            win_dump_buf.assign((size_t) count * g.n_embd, 0.f);
            CK(cudaMemcpyAsync(win_dump_buf.data(), parts, win_dump_buf.size() * 4,
                               cudaMemcpyDeviceToHost, service));
            CK(cudaStreamSynchronize(service));
            unsigned long long h1 = 1469598103934665603ull, h2 = 0;
            for (float v : win_dump_buf) {
                unsigned bits; std::memcpy(&bits, &v, 4);
                h1 = (h1 ^ bits) * 1099511628211ull;
                h2 += (unsigned long long) (bits & 0xffu);
            }
            std::fprintf(stderr, "mimo window dump: idx=%d layer=%d T=%d hit_arm=%d remote=%d "
                                 "bg=%d miss_device=%d ns=%d ng=%d fnv=%llx sum8=%llu\n",
                         win_dump_index, layer, tokens, (int) hit_arm, remote, (int) bg,
                         (int) miss_device, ns, ng, h1, h2);
        }
        if (win_dump_at != -1) ++win_dump_index;
    }

    void serve_decode_routes(int layer, const int32_t* routes, const float* activation,
                             const uint8_t* hits, cudaStream_t st) {
        count_routes(layer, routes, 1);
        int32_t comp[64];
        int c = 0, n_peer = 0;
        for (int j = 0; j < topk; ++j) {
            const bool remote = !hits[j] && peer && peer->valid() && peer->has(layer, routes[j]);
            peer_kind[j] = remote ? 2 : -1;
            n_peer += remote;
        }
        const bool has_peer = peer && peer->valid();
        if (has_peer) {
            ++peer_calls;
            if (!n_peer) ++peer_win_zero;
            else {
                ++peer_windows;
                if (n_peer == topk) ++peer_win_all;
                else ++peer_win_mixed;
            }
        }
        std::string error;
        const auto launch0 = Clock::now();
        // R3 (STRATA_MIMO_PEER_DEVICE_RESULTS): pass the shared device-output frame so the
        // launch enqueues the compact F16 transfer + ready signal on the peer stream instead
        // of the whole-rows D2H; the collection happens below on the service stream.
        if (n_peer && !peer->launch(layer, activation, routes, 1, topk, peer_kind.data(), error,
                                    peer_device_results ? peer_output.get() : nullptr))
            throw std::runtime_error("mimo handoff peer launch: " + error);
        if (win_test_peer_fault == 1 && peer_device_results && n_peer)
            // TEST ONLY: same in-flight fault as the window arm, on the T1 decode arm.
            throw std::runtime_error("test peer frame fault: decode launch queued the device "
                                     "frame (n_peer=" + std::to_string(n_peer) + " of " +
                                     std::to_string(topk) + " routes, layer " +
                                     std::to_string(layer) + ")");
        if (n_peer) {
            peer_launch_ms += ms_since(launch0);
            peer_bytes_up += (double) (g.n_embd * 4 + topk * 4);
        }
        const auto svc_cls0 = svc_time ? Clock::now() : Clock::time_point{};
        auto* map = diag_ids_host;
        for (int j = 0; j < topk; ++j)
            if (!hits[j] && peer_kind[j] != 2) { comp[c] = routes[j]; map[c++] = j; }
        if (svc_time) {
            svc_class_ms += ms_since(svc_cls0);
            ++svc_calls;
        }
        if (c) {
            const auto primary0 = Clock::now();
            int misses = 0;
            for (int j = 0; j < c; ++j) misses += !experts->holds(layer, comp[j]);
            int nc = 0;
            bool bg = false;
            uint8_t refused[64] = {};
            const auto svc_cpu0 = svc_time ? Clock::now() : Clock::time_point{};
            if (cpu_moe && handoff_cpu && misses >= handoff_cpu_min) {
                experts->classify(layer, comp, 1, c, cpu_flag.data());
                for (int j = 0; j < c; ++j) {
                    cpu_route_row[j] = -1;
                    if (!cpu_flag[j]) continue;
                    cpu_route_row[j] = nc;
                    cpu_ids[nc] = comp[j]; cpu_tok[nc] = 0; cpu_row[nc] = nc;
                    ++nc;
                }
                if (nc) {
                    try {
                        cpu_moe->run_rows_bg(layer, activation, (int64_t) g.n_embd, cpu_ids.data(),
                            cpu_tok.data(), cpu_row.data(), nc, cpu_out.data(), (int64_t) g.n_embd);
                        bg = true;
                    } catch (const std::exception& e) {
                        std::fprintf(stderr, "mimo handoff CPU: %s; retaining GPU rows\n", e.what());
                    }
                }
            }
            const auto svc_cpu1 = svc_time ? Clock::now() : Clock::time_point{};
            const auto svc_stg0 = svc_time ? Clock::now() : Clock::time_point{};
            const bool ok = experts->run_decode_device(layer, h, comp, c, peer_rows_dev, st, bg ? refused : nullptr);
            const auto svc_stg1 = svc_time ? Clock::now() : Clock::time_point{};
            const auto svc_join0 = svc_time ? Clock::now() : Clock::time_point{};
            if (bg) cpu_moe->run_rows_join(); // drain CPU writers even when GPU service refuses
            const auto svc_join1 = svc_time ? Clock::now() : Clock::time_point{};
            if (!ok) throw std::runtime_error("mimo handoff primary miss service failed");
            if (bg) {
                for (int j = 0; j < c; ++j) if (refused[j]) {
                    if (cpu_route_row[j] < 0) throw std::runtime_error("mimo handoff CPU refusal mismatch");
                    CK(cudaMemcpyAsync(peer_rows_dev + (size_t) j * g.n_embd,
                        cpu_out.data() + (size_t) cpu_route_row[j] * g.n_embd,
                        (size_t) g.n_embd * sizeof(float), cudaMemcpyHostToDevice, st));
                    ++cpu_rows_served;
                }
            }
            CK(cudaMemcpyAsync(ids, map, (size_t) c * 4, cudaMemcpyHostToDevice, st));
            k_scatter_primary_rows<<<(unsigned) (((size_t) c * g.n_embd + kThreads - 1) / kThreads),
                                     kThreads, 0, st>>>(parts, peer_rows_dev, ids, c, g.n_embd);
            CK(cudaGetLastError());
            if (svc_time) {
                svc_cpu_bg_ms += std::chrono::duration<double, std::milli>(svc_cpu1 - svc_cpu0).count();
                svc_stage_ms += std::chrono::duration<double, std::milli>(svc_stg1 - svc_stg0).count();
                svc_cpu_join_ms += std::chrono::duration<double, std::milli>(svc_join1 - svc_join0).count();
                svc_tail_ms += ms_since(svc_join1);                          // refused+ids H2D, scatter
            }
            if (has_peer) {
                peer_primary_ms += ms_since(primary0);
                peer_primary_rows += c;
                peer_primary_miss_rows += misses;
            }
        }
        if (n_peer) {
            const auto finish0 = Clock::now();
            const auto svc_fw0 = svc_time ? Clock::now() : Clock::time_point{};
            if (peer_device_results) {
                // R3: same deferred collection as the window arm - kick the peer driver,
                // then the frame's wait+copy on THIS service stream converts and scatters
                // only the peer's topk rows into parts. The complement's scatter above and
                // the CPU-refused uploads wrote different rows of parts on the same stream;
                // the doorbell release enqueued after this keeps the producer's combine
                // ordered behind all of them.
                if (peer_output == nullptr)
                    throw std::runtime_error("mimo handoff peer device results: transport missing "
                                             "(setup_peer must open it or fail the engine)");
                if (!peer->finish_deferred(error))
                    throw std::runtime_error("mimo handoff peer deferred finish: " + error);
                // TEST ONLY (fault=desync): same pre-copy bump as the window arm - the
                // queued wait's target provably has no signal (see there for the ordering
                // argument and the worker-error-release scope of the proof).
                if (win_test_peer_fault == 3) peer_output->test_desynchronize();
                peer_output->copy(parts, topk, nullptr, 0, st);
                if (win_test_peer_fault >= 2)
                    throw std::runtime_error("test peer frame fault: service wait+copy "
                                             "queued (fault=" +
                                             std::to_string(win_test_peer_fault) + ", n_peer=" +
                                             std::to_string(n_peer) + " of " +
                                             std::to_string(topk) + " routes, layer " +
                                             std::to_string(layer) + ")");
                ++peer_dev_frames;
                peer_dev_rows += n_peer;
                peer_dev_upload_avoided += (double) topk * g.n_embd * 4;
                peer_dev_convert_avoided += n_peer;
            } else if (!peer->finish(parts_host, error)) {
                throw std::runtime_error("mimo handoff peer finish: " + error);
            }
            const auto svc_fw1 = svc_time ? Clock::now() : Clock::time_point{};
            peer_finish_ms += ms_since(finish0);
            peer_rows += n_peer;
            peer_bytes_down += (double) ((int64_t) n_peer * g.n_embd * 2);
            if (!peer_device_results) {
                auto* mask = reinterpret_cast<uint8_t*>(peer_w_host);
                for (int j = 0; j < topk; ++j) mask[j] = peer_kind[j] == 2;
                CK(cudaMemcpyAsync(peer_mask_dev, mask, (size_t) topk, cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(peer_rows_dev, parts_host, (size_t) topk * g.n_embd * 4,
                                   cudaMemcpyHostToDevice, st));
                k_copy_masked_rows<<<(unsigned) (((size_t) topk * g.n_embd + kThreads - 1) / kThreads),
                                     kThreads, 0, st>>>(parts, peer_rows_dev, peer_mask_dev, topk, g.n_embd);
                CK(cudaGetLastError());
            }
            if (svc_time) {
                svc_finwait_ms += std::chrono::duration<double, std::milli>(svc_fw1 - svc_fw0).count();
                svc_fintail_ms += ms_since(svc_fw1);                         // mask/parts H2D + copy
            }
        }
    }

    /// Full forward at [pos, pos + T). Ordinary decode returns the final row's argmax;
    /// verify keeps every row's logits/states and leaves its final readback to the caller.
    int32_t forward(const int32_t* toks, int T, bool all_rows = false) {
        const int64_t pos0 = pos;
        const bool tm_dec = (T == 1 && pos0 > 0);   // the decode arm's own condition below
        const int tb = tm_dec ? 0 : 1;
        const auto tm_t0 = Clock::now();
        route_decode = tm_dec;
        if (!tm_dec) cache_trace_start = true;
        if (cache_trace && tm_dec && cache_trace_start) {
            std::fprintf(cache_trace, "B %llu %d\n", (unsigned long long) cache_trace_segment++, experts->slots_per_layer());
            for (int l = 0; l < g.n_layers; ++l) if (g.is_moe(l)) {
                std::fprintf(cache_trace, "S %d", l);
                for (int e : experts->cache_lru_order(l)) std::fprintf(cache_trace, " %d", e);
                std::fprintf(cache_trace, "\nP %d", l);
                if (peer && peer->valid()) for (int e = 0; e < g.n_expert; ++e)
                    if (peer->has(l, e)) std::fprintf(cache_trace, " %d", e);
                std::fputc('\n', cache_trace);
            }
            cache_trace_start = false;
        }
        if (peer_adapt) {
            // Both streams and worker bookkeeping are drained at the preceding forward boundary.
            CK(cudaStreamSynchronize(st)); // verify/all_rows callers may otherwise leave work queued
            std::string error;
            if (!peer_adapt->before_window(error)) throw std::runtime_error(error);
        }
        const bool handoff = decode_handoff && tm_dec && !diag && !all_rows && (cpu_moe == nullptr || handoff_cpu);
        const bool mapped_window = window_handoff && !diag && !tm_on && !handoff_eager &&
            strata::mimo::window_handoff_eligible(T, topk, (int) g.n_embd, chunk, all_rows);
        const bool async_handoff = (handoff && !handoff_eager && !tm_on) || mapped_window;
        if (handoff) {
            if (window_handoff && !handoff_t1_prepared) {
                // Startup reserved an 8-row frame WITHOUT changing window-only SMALL_M.
                // Preserve the old lazy enable(1) transition at the first real T1 handoff,
                // exactly once, while the worker is idle and before publishing its wake.
                std::unique_lock<std::mutex> lock(handoff_mu);
                handoff_cv.wait(lock, [&] { return !handoff_busy; });
                if (handoff_error) std::rethrow_exception(handoff_error);
                CK(cudaStreamSynchronize(st));
                experts->prepare_decode_handoff_t1();
                handoff_t1_prepared = true;
            }
            if (!handoff_ready) {
                // Previous forward has drained: no allocator calls against in-flight hits.
                experts->enable_decode_handoff(topk);
                peer_kind.resize(topk);
                if (!peer_rows_dev) CK(cudaMalloc(&peer_rows_dev, (size_t) topk * g.n_embd * 4));
                if (!handoff_eager && !tm_on) start_decode_worker();
                handoff_ready = true;
                std::fprintf(stderr, "strata-mimo: resident routing handoff: %s\n",
                             async_handoff ? "persistent worker + independent service stream" :
                                             "eager publication polling (diagnostic/timing mode)");
            }
            experts->refresh_verify_graph(st);
        }
        if (mapped_window && window_gpu_hits) {
            // The window-hit arm's per-forward table declaration: T is THIS forward's token
            // count, so the active >= T*k gate inside is the narrow per-window eligibility
            // (not a blanket max_tokens*k requirement). Quiescent like the T1 refresh: the
            // previous forward drained at its final decode_wait, before any readback.
            experts->refresh_verify_graph(st, T);
        }
        if (async_handoff) {
                std::lock_guard<std::mutex> lock(handoff_mu);
                if (++handoff_ring == 0) throw std::runtime_error("mimo handoff sequence exhausted");
                handoff_busy = handoff_pending = true;
                handoff_cv.notify_all();
        }
        CK(cudaMemcpyAsync(tok, toks, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        for (int r = 0; r < T; ++r) pos_stage[(size_t) r] = pos0 + r;
        CK(cudaMemcpyAsync(dpos, pos_stage.data(), (size_t) T * 8, cudaMemcpyHostToDevice, st));
        if (pos0 + T > cap) {
            std::fprintf(stderr, "strata-mimo: the context is full (%lld + %d > %lld)\n", (long long) pos0, T,
                         (long long) cap);
            std::exit(1);
        }
        tm_span(0, tb, [&] {
            k_embed<<<T, kThreads, 0, st>>>(reinterpret_cast<const uint16_t*>(pack->embed), tok, x, g.n_embd);
            CK(cudaGetLastError());
        });
        if (decode_device && T == 1 && pos0 > 0) {
            const int seq = (int) pos0 + 1;
            CK(cudaMemcpyAsync(dseq, &seq, sizeof(seq), cudaMemcpyHostToDevice, st));
        }
        if (diag) {
            if (diag_layers.size() != (size_t) g.n_layers) diag_layers.resize((size_t) g.n_layers);
            if (diag_logits == nullptr) CK(cudaMalloc(&diag_logits, (size_t) T * g.n_vocab * 4));
        }

        const dim3 scatter((unsigned) ((max_kv * g.head_dim_qk + kThreads - 1) / kThreads), (unsigned) T);
        bool window_split_used = false;   // any MoE layer routed through the window split this forward
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const MimoPack::LayerDense& d = pack->layers[(size_t) l];
            const bool full = g.is_full(l);
            const int kv = (int) g.kv_heads(l);
            const int window = full ? 0 : (int) g.sliding_window;
            const int ring = full ? 0 : swa_ring;

            // --- attention: rms -> q/k/v -> caches (+rope) -> attention -> o_proj -> residual
            const auto tm_half0 = Clock::now();
            tm_span(1, tb, [&] {
                k_rmsnorm<<<T, kThreads, 0, st>>>(x, reinterpret_cast<const float*>(d.norm_attn), h, g.n_embd,
                                                (float) g.rms_eps);
                CK(cudaGetLastError());
                // linear() consumes h directly or makes its own F16 image. This attention
                // copy is unused; the MLP/router's distinct h16 carry is still produced below.
                if (!decode_device) K::f32_to_f16_bulk(h, h16, (int64_t) T * g.n_embd, st);
            });
            tm_span(2, tb, [&] {
                const bool trio_q8 = d.q8q.valid() && d.q8k.valid() && d.q8v.valid();
                if (!(qkv_share || q8_wide_cols > 8) || !trio_q8) {
                    linear(d.q, d.q8q.w, d.tq, h, qf, T, g.qk_dim(), g.n_embd);
                    linear(d.k, d.q8k.w, d.tk, h, kf, T, (int64_t) kv * g.head_dim_qk, g.n_embd);
                    linear(d.v, d.q8v.w, d.tv, h, vf, T, (int64_t) kv * g.head_dim_qk, g.n_embd);
                } else {
                    const int nq = (int) g.qk_dim(), nkv = (int) (kv * g.head_dim_qk), n_in = (int) g.n_embd;
                    int r = 0;
                    if (q8_wide_active && T > 8) {
                        // P2's wide tiles (prefill only): one shared q8_1 image per tile; with P1
                        // along, the fused trio launch (bitwise the three wide singles, which are
                        // bitwise the <= 8 tiles).  A refusal is process-stable, never data-dependent;
                        // the tail loop below recomputes the refused range byte-identically.
                        for (; r + q8_wide_cols <= T;) {
                            const int nc = q8_wide_cols;
                            K::native_quantize_q8_1_wide(h + (int64_t) r * g.n_embd, q8_wide_scratch,
                                                         n_in, nc, st);
                            bool done = false;
                            if (qkv_share &&
                                K::native_mmvq_bunch_wide(8, d.q8q.w, d.q8k.w, d.q8v.w, q8_wide_scratch,
                                                          qf + (int64_t) r * nq, kf + (int64_t) r * nkv,
                                                          vf + (int64_t) r * nkv, n_in, nq, nkv, nkv, nc, st))
                                done = true;
                            else if (K::native_q8_0_mmvq_wide(d.q8q.w, q8_wide_scratch, qf + (int64_t) r * nq,
                                                              n_in, nq, nc, st) &&
                                     K::native_q8_0_mmvq_wide(d.q8k.w, q8_wide_scratch, kf + (int64_t) r * nkv,
                                                              n_in, nkv, nc, st) &&
                                     K::native_q8_0_mmvq_wide(d.q8v.w, q8_wide_scratch, vf + (int64_t) r * nkv,
                                                              n_in, nkv, nc, st))
                                done = true;
                            if (!done) break;
                            r += nc;
                        }
                    }
                    for (; r < T;) {
                        const int nc = std::min(8, T - r);
                        if (qkv_share) {
                            // P1: ONE q8_1 image of the shared input, then the fused trio launch or -
                            // where the library declines (gfx906 wave build, upstream layout, ...) -
                            // the three sequential MMVQs on the same bytes: bitwise today's three
                            // native_q8_0_f32 calls either way (the quantize kernel is the same one).
                            K::native_quantize_q8_1(h + (int64_t) r * g.n_embd, q8_scratch, n_in, nc, st);
                            if (nc >= 2 && K::native_mmvq_bunch_supported(8, n_in, nc) &&
                                K::native_mmvq_bunch(8, d.q8q.w, d.q8k.w, d.q8v.w, q8_scratch,
                                                     qf + (int64_t) r * nq, kf + (int64_t) r * nkv,
                                                     vf + (int64_t) r * nkv, n_in, nq, nkv, nkv, nc, st)) {
                                r += nc;
                                continue;
                            }
                            K::native_q8_0_mmvq(d.q8q.w, q8_scratch, qf + (int64_t) r * nq, n_in, nq, nc, st);
                            K::native_q8_0_mmvq(d.q8k.w, q8_scratch, kf + (int64_t) r * nkv, n_in, nkv, nc, st);
                            K::native_q8_0_mmvq(d.q8v.w, q8_scratch, vf + (int64_t) r * nkv, n_in, nkv, nc, st);
                        } else {
                            // P2 only: the <= 8 tiles keep today's per-matrix calls
                            linear(d.q, d.q8q.w, d.tq, h + (int64_t) r * g.n_embd, qf + (int64_t) r * nq,
                                   nc, g.qk_dim(), g.n_embd);
                            linear(d.k, d.q8k.w, d.tk, h + (int64_t) r * g.n_embd, kf + (int64_t) r * nkv,
                                   nc, (int64_t) kv * g.head_dim_qk, g.n_embd);
                            linear(d.v, d.q8v.w, d.tv, h + (int64_t) r * g.n_embd, vf + (int64_t) r * nkv,
                                   nc, (int64_t) kv * g.head_dim_qk, g.n_embd);
                        }
                        r += nc;
                    }
                }
            });
            tm_span(3, tb, [&] {
                K::f32_to_f16_bulk(qf, q16, (int64_t) T * g.qk_dim(), st);
                if (ring) {
                    K::f32_to_f16_bulk(kf, ring_k16, (int64_t) T * kv * g.head_dim_qk, st);
                } else {
                    k_scatter_k<<<scatter, kThreads, 0, st>>>(kf, kc[(size_t) l], dpos, kv,
                                                               (int) g.head_dim_qk,
                                                               (int64_t) kv * g.head_dim_qk);
                    k_scatter_v<<<dim3((unsigned) ((kv * g.head_dim_v + kThreads - 1) / kThreads), (unsigned) T),
                                  kThreads, 0, st>>>(vf, vc[(size_t) l], dpos, kv, (int) g.head_dim_qk,
                                                     (int) g.head_dim_v, (int64_t) kv * g.head_dim_v);
                    CK(cudaGetLastError());
                }
                // rope on q (in the window buffer) and on k (in the cache rows), per the A/B knobs
                const double th = theta_swap ? (g.is_full(l) ? g.rope_theta_swa : g.rope_theta_full)
                                             : g.rope_theta(l);
                auto rope = [&](void* rows, int heads) {
                    if (rope_gptj) {
                        const long long total = (long long) T * heads * (g.n_rot / 2);
                        k_rope_gptj<<<(unsigned) ((total + kThreads - 1) / kThreads), kThreads, 0, st>>>(
                            (uint16_t*) rows, dpos, T, heads, (int) g.head_dim_qk, (int) g.n_rot, th);
                        CK(cudaGetLastError());
                    } else {
                        K::mimo_rope_f16(rows, dpos, T, heads, (int) g.head_dim_qk, (int) g.n_rot, th, st);
                    }
                };
                rope(q16, (int) g.n_q_head);
                if (ring) {
                    rope(ring_k16, kv);
                    k_scatter_ring_k<<<scatter, kThreads, 0, st>>>(ring_k16, kc[(size_t) l], dpos,
                                                                   (int) (kv * g.head_dim_qk), ring);
                    k_scatter_ring_v<<<dim3((unsigned) ((kv * g.head_dim_v + kThreads - 1) / kThreads), (unsigned) T),
                                       kThreads, 0, st>>>(vf, vc[(size_t) l], dpos, kv,
                                                          (int) g.head_dim_qk, (int) g.head_dim_v, ring);
                    CK(cudaGetLastError());
                } else {
                    rope(kc[(size_t) l] + (size_t) pos0 * kv * g.head_dim_qk, kv);
                }
            });
            const void* sinks = d.sink;
            if (d.sink != nullptr && sink_scaled) {   // the A/B arm: the sink logit sm_scale'd
                if (sink_scaled_buf == nullptr) CK(cudaMalloc(&sink_scaled_buf, (size_t) g.n_q_head * 4));
                k_scale<<<(unsigned) ((g.n_q_head + kThreads - 1) / kThreads), kThreads, 0, st>>>(
                    reinterpret_cast<const float*>(d.sink), sink_scaled_buf, g.n_q_head, (float) sm_scale);
                CK(cudaGetLastError());
                sinks = sink_scaled_buf;
            }
            // the GQA-mapping A/B: permute q's heads (and the sinks) so the kernel's BLOCK mapping
            // (q heads [kvh*G, (kvh+1)*G) -> kvh) realizes the INTERLEAVED convention (h % n_kv), then
            // unpermute the attention output back to the model's head order for o_proj
            uint16_t const* q_attn = q16;
            float* attn_dst = attn;
            const void* sinks_attn = sinks;
            tm_span(4, tb, [&] {
                if (gqa_interleave) {
                    if (gqa_scratch_sinks == nullptr) CK(cudaMalloc(&gqa_scratch_sinks, (size_t) g.n_q_head * 4));
                    k_permute_heads<<<(unsigned) (((size_t) T * g.n_q_head * g.head_dim_qk + kThreads - 1) /
                                                  kThreads), kThreads, 0, st>>>(
                        q16, q_perm_buf, gqa_perm.data(), T, (int) g.n_q_head, (int) g.head_dim_qk);
                    if (sinks != nullptr)
                        k_permute_heads_f32<<<1, 64, 0, st>>>(reinterpret_cast<const float*>(sinks),
                                                              gqa_scratch_sinks, gqa_perm.data(), 1,
                                                              (int) g.n_q_head, 1);
                    CK(cudaGetLastError());
                    q_attn = q_perm_buf;
                    if (sinks != nullptr) sinks_attn = gqa_scratch_sinks;
                    attn_dst = attn_perm_buf;
                }
                // attention dispatch: the opt-in split path serves decode (T=1) and the
                // verify-window rows (T<=8) only; prompt chunks (T=512) keep the original
                // kernels - their workspace reservation covers 8 rows by design.  keys_hint
                // is host-side knowledge (no D2H): the chunk's exact key bound.  SWA layers
                // fall back inside the entry (window 128 -> plan 1 -> the plain kernels,
                // byte-identical), so effectively the 9 FULL layers change.
                const bool use_split = attn_split && T <= (int) attn_ws.max_rows;
                const int keys_hint = (int) std::min<int64_t>(cap, pos0 + T);
                if (T == 1 && pos0 > 0) {
                    const int seq[1] = { (int) pos0 + 1 };
                    if (!decode_device) CK(cudaMemcpyAsync(dseq, seq, 4, cudaMemcpyHostToDevice, st));
                    if (use_split)
                        K::mimo_attn_decode_split(q_attn, kc[(size_t) l], vc[(size_t) l], attn_dst,
                                                  dpos, dseq, T, (int) g.n_q_head, kv,
                                                  (int) g.head_dim_qk, (int) g.head_dim_v, (int) cap,
                                                  window, sinks_attn, sm_scale, keys_hint, 0,
                                                  &attn_ws, st, ring);
                    else
                        K::mimo_attn_decode(q_attn, kc[(size_t) l], vc[(size_t) l], attn_dst, dpos,
                                            dseq, T, (int) g.n_q_head, kv, (int) g.head_dim_qk,
                                            (int) g.head_dim_v, (int) cap, window, sinks_attn,
                                            sm_scale, st, ring);
                } else {
                    if (use_split)
                        K::mimo_attn_prefill_split(q_attn, kc[(size_t) l], vc[(size_t) l], attn_dst,
                                                   pos0, T, (int) g.n_q_head, kv,
                                                   (int) g.head_dim_qk, (int) g.head_dim_v,
                                                   (int) cap, window, sinks_attn, sm_scale,
                                                   keys_hint, 0, &attn_ws, st, ring);
                    else
                        K::mimo_attn_prefill(q_attn, kc[(size_t) l], vc[(size_t) l], attn_dst, pos0, T,
                                             (int) g.n_q_head, kv, (int) g.head_dim_qk,
                                             (int) g.head_dim_v, (int) cap, window, sinks_attn,
                                             sm_scale, st, ring);
                }
                if (gqa_interleave)
                    k_permute_heads_f32<<<(unsigned) (((size_t) T * g.n_q_head * g.head_dim_v + kThreads - 1) /
                                                      kThreads), kThreads, 0, st>>>(
                        attn_perm_buf, attn, gqa_perm.data(), T, (int) g.n_q_head, (int) g.head_dim_v);
            });
            if (diag) diag_stats(attn, (size_t) T * g.v_dim(), diag_layers[(size_t) l].attn_mean,
                                 diag_layers[(size_t) l].attn_std);
            tm_span(5, tb, [&] {
                if (!decode_device) {
                    K::f32_to_f16_bulk(attn, a16, (int64_t) T * g.v_dim(), st);
                    if (d.o != nullptr) K::f32_to_f16_bulk(attn, a16, (int64_t) T * g.v_dim(), st);
                }
                linear(d.o, d.q8o.w, d.to, attn, proj, T, g.n_embd, g.v_dim());
                k_add<<<(unsigned) (((size_t) T * g.n_embd + kThreads - 1) / kThreads), kThreads, 0, st>>>(
                    x, proj, (int64_t) T * g.n_embd);
                CK(cudaGetLastError());
            });

            if (diag) {   // x snapshot before the FFN: the MoE/dense-out stats are the delta
                diag_snap.resize((size_t) T * g.n_embd);
                CK(cudaMemcpyAsync(diag_snap.data(), x, diag_snap.size() * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
            }
            // --- the FFN half
            tm_span(6, tb, [&] {
                k_rmsnorm<<<T, kThreads, 0, st>>>(x, reinterpret_cast<const float*>(d.norm_mlp), h, g.n_embd,
                                                (float) g.rms_eps);
                CK(cudaGetLastError());
                K::f32_to_f16_bulk(h, h16, (int64_t) T * g.n_embd, st);
            });
            if (!g.is_moe(l)) {
                tm_span(8, tb, [&] {
                    linear(d.ffn_gate, nullptr, d.tffn_gate, h, gg, T, g.dense_intermediate, g.n_embd);
                    linear(d.ffn_up, nullptr, d.tffn_up, h, uu, T, g.dense_intermediate, g.n_embd);
                    k_silu_mul_f32<<<(unsigned) (((size_t) T * g.dense_intermediate + kThreads - 1) / kThreads),
                                     kThreads, 0, st>>>(gg, uu, silu_f32, (int64_t) T * g.dense_intermediate);
                    CK(cudaGetLastError());
                    linear(d.ffn_down, nullptr, d.tffn_down, silu_f32, proj, T, g.n_embd, g.dense_intermediate);
                });
            } else {
                tm_span(7, tb, [&] {
                    if (T == 1 && pos0 > 0 && !route_mono)   // the split decode arm (bit-identical
                        // picks; see mimo_router_route_decode) - one launch of the one-block
                        // kernel leaves the card idle at one token
                        K::mimo_router_route_decode(d.router, h16, d.e_score, route_scores, ids, rw,
                                                    (int) g.n_embd, (int) g.n_expert, topk, st);
                    else if (window_router_split && all_rows && T >= 2 && T <= 8) {
                        // the window split arm: the verify windows (all_rows, T=2..8) through
                        // the decode arm's kernels batched over the rows - bit-identical picks
                        // to the one-block launch below, whose T blocks re-read the whole
                        // router plane per row (the 2026-11-09 profile's 47 x 138 us/round)
                        K::mimo_router_route_window(d.router, h16, d.e_score, window_route_scores, ids,
                                                    rw, T, (int) g.n_embd, (int) g.n_expert, topk, st);
                        ++window_split_layers;
                        if (!window_split_used) window_split_used = true;
                    } else
                        K::mimo_router_route(d.router, h16, d.e_score, ids, rw, T, (int) g.n_embd,
                                             (int) g.n_expert, topk, st);
                });
                if (diag) {   // the LAST row's picks + weights, and the combine's effect on x
                    CK(cudaMemcpyAsync(diag_ids_host, ids + (size_t)(T - 1) * topk, (size_t) topk * 4,
                                       cudaMemcpyDeviceToHost, st));
                    CK(cudaMemcpyAsync(diag_w_host, rw + (size_t)(T - 1) * topk, (size_t) topk * 4,
                                       cudaMemcpyDeviceToHost, st));
                    CK(cudaStreamSynchronize(st));
                    diag_layers[(size_t) l].ids.assign(diag_ids_host, diag_ids_host + topk);
                    diag_layers[(size_t) l].w.assign(diag_w_host, diag_w_host + topk);
                }
                if (handoff) {
                    if (tm_on) tm_ms[tb][10] += ms_since(tm_half0);
                    const auto started = Clock::now();
                    if (async_handoff) {
                        experts->enqueue_decode_handoff((int) l, h, ids, topk, parts, handoff_ring, st);
                    } else {
                        const int32_t* routes = nullptr;
                        const float* activation = nullptr;
                        const uint8_t* hits = nullptr;
                        experts->decode_handoff((int) l, h, ids, topk, parts, st, routes, activation, hits);
                        serve_decode_routes((int) l, routes, activation, hits, st);
                        experts->verify_graph_end_host((int) l);
                    }
                    k_moe_combine<<<dim3((unsigned) ((g.n_embd + kThreads - 1) / kThreads), 1),
                                   kThreads, 0, st>>>(x, parts, rw, 1, topk, g.n_embd);
                    CK(cudaGetLastError());
                    if (tm_on) tm_ms[tb][11] += ms_since(started);
                } else if (T == 1 && pos0 > 0) {
                    // decode. The peer tier's rows LAUNCH first (enqueue-only: x H2D, convert,
                    // indexed FFN, rows D2H on the peer card's own stream), the primary stream
                    // computes the compact complement with its blocking host-IO run() while the
                    // peer's GPU works, and finish() lands the peer's unweighted rows at their
                    // route destinations - drive_exl3_pool's exact single-token shape; the
                    // unchanged k_moe_combine below applies the router weights to every row.
                    if (tm_on) tm_ms[tb][10] += ms_since(tm_half0);
                    const auto tm_moe0 = Clock::now();
                    const auto tm_d0 = Clock::now();
                    CK(cudaMemcpyAsync(h_host, h, (size_t) g.n_embd * 4, cudaMemcpyDeviceToHost, st));
                    CK(cudaMemcpyAsync(ids_host, ids, (size_t) topk * 4, cudaMemcpyDeviceToHost, st));
                    CK(cudaStreamSynchronize(st));
                    if (tm_on) tm_d2h_ms[tb] += ms_since(tm_d0);
                    count_routes(l, ids_host, 1);
                    bool split_done = false;
                    bool primary_on_device = false;
                    const bool use_device = decode_device && cpu_moe == nullptr;
                    bool deferred_layer = false;
                    if (peer != nullptr && peer->valid()) {
                        ++peer_calls;
                        int n_peer = 0;
                        for (int j = 0; j < topk; ++j) {
                            peer_kind[(size_t) j] = peer->has((int64_t) l, ids_host[j]) ? 2 : -1;
                            n_peer += peer_kind[(size_t) j] == 2;
                        }
                        if (n_peer == 0) ++peer_win_zero;
                        else {
                            ++peer_windows;
                            if (n_peer == topk) ++peer_win_all;
                            else ++peer_win_mixed;
                            std::string perr;
                            const auto t_launch = Clock::now();
                            bool ok = peer->launch((int64_t) l, h_host, ids_host, 1, topk,
                                                   peer_kind.data(), perr, use_device ? peer_output.get() : nullptr);
                            peer_launch_ms += ms_since(t_launch);
                            if (tm_on) tm_launch_ms[tb] += ms_since(t_launch);
                            if (ok) peer_bytes_up += (double) (g.n_embd * 4 + topk * 4);
                            if (ok && n_peer < topk) {
                                // the compact complement (PEER_SPARSE: the peer's rows never
                                // touch the primary plan): run() takes the non-peer ids and its
                                // rows scatter back to their pick-order destinations.  (Tried
                                // 2026-10-28: run_device + a device-side masked gather removes
                                // the parts D2H/H2D round trip but LOSES ~0.5 ms/layer - the
                                // device arm uploads its plan from PAGEABLE memory in five small
                                // pieces while run()'s host arm rides the default-on packed
                                // PINNED plan as one DMA, and it takes the deterministic Partials
                                // reduction (a different F32 sum order than run()'s atomicAdd
                                // arm, which flipped a near-tie token 46 tokens in).  Reverted;
                                // the notes keep the numbers.)
                                int32_t comp[64];
                                int row[64];
                                int c = 0;
                                for (int j = 0; j < topk; ++j)
                                    if (peer_kind[(size_t) j] != 2) { comp[c] = ids_host[j]; row[c] = j; ++c; }
                                const auto t_prim = Clock::now();
                                int misses = 0;
                                for (int j = 0; j < c; ++j)
                                    if (!experts->holds((int64_t) l, comp[j])) ++misses;
                                if (use_device) {
                                    ok = experts->run_decode_device((int) l, h, comp, c, peer_rows_dev, st);
                                    if (ok) {
                                        // The activation and compact FFN rows stay on this card.
                                        // Reuse decode-id scratch after the routing readback; its
                                        // pinned host source survives through the next router sync.
                                        auto* map = reinterpret_cast<int32_t*>(peer_comp_host);
                                        for (int j = 0; j < c; ++j) map[j] = row[j];
                                        CK(cudaMemcpyAsync(ids, map, (size_t) c * 4, cudaMemcpyHostToDevice, st));
                                        k_scatter_primary_rows<<<(unsigned) (((size_t) c * g.n_embd + kThreads - 1) / kThreads),
                                                                 kThreads, 0, st>>>(parts, peer_rows_dev, ids, c, g.n_embd);
                                        CK(cudaGetLastError());
                                        primary_on_device = true;
                                    }
                                } else {
                                    ok = serve_primary(l, h_host, comp, c, peer_comp_host);
                                }
                                peer_primary_ms += ms_since(t_prim);
                                if (tm_on) tm_prim_ms[tb] += ms_since(t_prim);
                                if (ok) {
                                    peer_primary_rows += c;
                                    peer_primary_miss_rows += misses;   // BEFORE admission, not after run()
                                    for (int j = 0; !primary_on_device && j < c; ++j)
                                        std::memcpy(parts_host + (size_t) row[j] * g.n_embd,
                                                    peer_comp_host + (size_t) j * g.n_embd,
                                                    (size_t) g.n_embd * 4);
                                }
                            }
                            if (ok) {
                                const auto t_fin = Clock::now();
                                if (use_device && peer_output) {
                                    ok = peer->finish_deferred(perr);
                                    if (ok) {
                                        peer_output->copy(parts, topk, nullptr, 0, st);
                                        primary_on_device = true;
                                        deferred_layer = true;
                                    }
                                } else {
                                    ok = peer->finish(parts_host, perr);
                                }
                                peer_finish_ms += ms_since(t_fin);
                                if (tm_on) tm_fin_ms[tb] += ms_since(t_fin);
                                peer_bytes_down += (double) ((int64_t) n_peer * g.n_embd * 2);
                            }
                            if (ok) {
                                peer_rows += n_peer;
                                split_done = true;
                            } else {
                                peer_fallback(l, "decode launch/finish", perr.c_str());
                            }
                        }
                    }
                    if (!split_done) {   // the plain path: the peer off, zero rows, or the fallback
                        const auto t_prim = Clock::now();
                        const bool ok = use_device
                            ? experts->run_decode_device((int) l, h, ids_host, topk, parts, st)
                            : serve_primary(l, h_host, ids_host, topk, parts_host);
                        primary_on_device = use_device && ok;
                        if (tm_on) tm_prim_ms[tb] += ms_since(t_prim);
                        if (!ok) {
                            std::fprintf(stderr, "strata-mimo: expert stream failed at layer %lld\n",
                                         (long long) l);
                            std::exit(1);
                        }
                    }
                    tm_span(8, tb, [&] {
                        if (primary_on_device && split_done && !deferred_layer) {
                            // Only peer-owned rows are valid in parts_host; local rows already
                            // reside in parts. The existing short-window mask/copy preserves them.
                            auto* mask = reinterpret_cast<uint8_t*>(peer_w_host);
                            for (int j = 0; j < topk; ++j) mask[j] = peer_kind[(size_t) j] == 2;
                            CK(cudaMemcpyAsync(peer_mask_dev, mask, (size_t) topk, cudaMemcpyHostToDevice, st));
                            CK(cudaMemcpyAsync(peer_rows_dev, parts_host, (size_t) topk * g.n_embd * 4,
                                               cudaMemcpyHostToDevice, st));
                            k_copy_masked_rows<<<(unsigned) (((size_t) topk * g.n_embd + kThreads - 1) / kThreads),
                                                 kThreads, 0, st>>>(parts, peer_rows_dev, peer_mask_dev,
                                                                     topk, g.n_embd);
                            CK(cudaGetLastError());
                        } else if (!primary_on_device) {
                            CK(cudaMemcpyAsync(parts, parts_host, (size_t) topk * g.n_embd * 4,
                                               cudaMemcpyHostToDevice, st));
                        }
                        k_moe_combine<<<dim3((unsigned) ((g.n_embd + kThreads - 1) / kThreads), (unsigned) T),
                                       kThreads, 0, st>>>(x, parts, rw, T, topk, g.n_embd);
                        CK(cudaGetLastError());
                    });
                    if (tm_on) tm_ms[tb][11] += ms_since(tm_moe0);
                } else if (mapped_window) {
                    experts->enqueue_decode_handoff((int) l, h, ids, topk, parts, handoff_ring, st, T, rw);
                    // The old short-window branch combines unweighted route rows in this exact
                    // order. Keep rw on device; its mapped image is published for inspection.
                    k_moe_combine<<<dim3((unsigned) ((g.n_embd + kThreads - 1) / kThreads), (unsigned) T),
                                   kThreads, 0, st>>>(x, parts, rw, T, topk, g.n_embd);
                    CK(cudaGetLastError());
                } else {
                    // a prompt chunk (or an MTP verify window): grouped by expert on the device,
                    // with the peer's owned experts split off. Two arms:
                    //  * SHORT windows (T <= 15, where NO group can reach the 16 rows that move
                    //    the primary's device arm to the grouped GEMM) ride the DECODE machinery:
                    //    peer->launch (ONE indexed FFN for all its rows), run_device(skip) for the
                    //    local share, finish() into the host slab, one staging H2D + one masked
                    //    gather into parts - the ordinary k_moe_combine then runs exactly the
                    //    single-GPU arithmetic (same kernel family, same order), so short windows
                    //    stay bit-identical to a primary-only run.
                    //  * Larger chunks take the Qwen prefill split (launch_prefill): the F16
                    //    activation image and the router weights come down on a SIDE stream of the
                    //    primary (overlapping the routing readback), the peer enqueues its grouped
                    //    FFN FIRST, run_device(skip) computes the local share without ever reading
                    //    the peer's weights, finish_prefill waits, and the peer's ONE WEIGHTED SUM
                    //    PER TOKEN (mapped host memory) joins through k_moe_combine_peer - the
                    //    GGUF peer-prefill exactness class, not bit-identical.
                    if (tm_on) tm_ms[tb][10] += ms_since(tm_half0);
                    const auto tm_moe0 = Clock::now();
                    const auto tm_d0 = Clock::now();
                    const bool pf_peer = peer != nullptr && peer->valid();
                    const bool pf_small = pf_peer && T <= 15;
                    const bool pf_group = pf_peer && !pf_small && peer->prefill_enabled() && T <= chunk;
                    if (pf_group) {
                        CK(cudaEventRecord(peer_ev, st));
                        CK(cudaStreamWaitEvent(peer_xs, peer_ev, 0));
                        CK(cudaMemcpyAsync(peer_x16_host, h16, (size_t) T * g.n_embd * 2,
                                           cudaMemcpyDeviceToHost, peer_xs));
                        CK(cudaMemcpyAsync(peer_w_host, rw, (size_t) T * topk * 4, cudaMemcpyDeviceToHost,
                                           peer_xs));
                    } else if (pf_small) {
                        CK(cudaEventRecord(peer_ev, st));
                        CK(cudaStreamWaitEvent(peer_xs, peer_ev, 0));
                        CK(cudaMemcpyAsync(peer_h_host, h, (size_t) T * g.n_embd * 4,
                                           cudaMemcpyDeviceToHost, peer_xs));
                    }
                    CK(cudaMemcpyAsync(ids_host, ids, (size_t) T * topk * 4, cudaMemcpyDeviceToHost, st));
                    CK(cudaStreamSynchronize(st));
                    if (tm_on) tm_d2h_ms[tb] += ms_since(tm_d0);
                    count_routes(l, ids_host, T);
                    bool peer_layer = false;
                    if (pf_peer && (pf_small || pf_group)) {
                        CK(cudaStreamSynchronize(peer_xs));
                        const uint8_t* lay_owned = peer_owned.data() + (size_t) l * (size_t) g.n_expert;
                        peer_mask.assign((size_t) T * topk, 0);
                        int64_t rows_peer = 0;
                        for (int r = 0; r < T * topk; ++r) {
                            const int32_t e = ids_host[(size_t) r];
                            if (e >= 0 && e < (int32_t) g.n_expert && lay_owned[e]) {
                                peer_mask[(size_t) r] = 1;
                                ++rows_peer;
                            }
                            if (pf_small) peer_kind[(size_t) r] = peer_mask[(size_t) r] ? 2 : -1;
                        }
                        if (rows_peer > 0) {
                            CK(cudaMemcpyAsync(peer_mask_dev, peer_mask.data(), (size_t) T * topk,
                                               cudaMemcpyHostToDevice, st));
                            std::string perr;
                            const auto t_launch = Clock::now();
                            bool ok = true;
                            if (pf_small) {   // the decode machinery: enqueue-only, ONE indexed FFN
                                ok = peer->launch((int64_t) l, peer_h_host, ids_host, T, topk,
                                                  peer_kind.data(), perr);
                                if (ok) peer_bytes_up += (double) (T * g.n_embd * 4 + T * topk * 4);
                            } else {
                                ok = peer->launch_prefill((int64_t) l, peer_x16_host, T, ids_host,
                                                           peer_mask.data(), peer_w_host, perr);
                                if (ok) peer_bytes_up += (double) (T * g.n_embd * 2 + T * topk * 4);
                            }
                            peer_pf_launch_ms += ms_since(t_launch);
                            if (tm_on) tm_launch_ms[tb] += ms_since(t_launch);
                            if (ok) {
                                const auto t_prim = Clock::now();
                                tm_span(8, tb, [&] {
                                    if (all_rows && verify_host && pf_small) {
                                        ok = experts->run_multi_complement((int) l, peer_h_host, ids_host,
                                            T, topk, parts_host, st, peer_mask.data());
                                        if (ok) CK(cudaMemcpyAsync(parts, parts_host, (size_t) T * topk * g.n_embd * 4,
                                                                   cudaMemcpyHostToDevice, st));
                                    } else ok = experts->run_device((int) l, h, ids_host, T, topk, parts, st,
                                                                     peer_mask.data());
                                });
                                peer_pf_primary_ms += ms_since(t_prim);
                                if (tm_on) tm_prim_ms[tb] += ms_since(t_prim);
                            }
                            if (ok && pf_small) {
                                const auto t_fin = Clock::now();
                                ok = peer->finish(peer_rows_host, perr);   // host rows, route order
                                peer_pf_finish_ms += ms_since(t_fin);
                                if (tm_on) tm_fin_ms[tb] += ms_since(t_fin);
                                if (ok) {
                                    peer_bytes_down += (double) (rows_peer * g.n_embd * 2 +
                                                                 (int64_t) T * topk * g.n_embd * 4);
                                    // stage the whole slab and gather the masked rows into parts,
                                    // then the ORDINARY combine - the single-GPU arithmetic
                                    tm_span(8, tb, [&] {
                                        CK(cudaMemcpyAsync(peer_rows_dev, peer_rows_host,
                                                           (size_t) T * topk * g.n_embd * 4,
                                                           cudaMemcpyHostToDevice, st));
                                        k_copy_masked_rows<<<
                                            (unsigned) (((size_t) T * topk * g.n_embd + kThreads - 1) /
                                                        kThreads), kThreads, 0, st>>>(
                                                parts, peer_rows_dev, peer_mask_dev, (int64_t) T * topk,
                                                g.n_embd);
                                        CK(cudaGetLastError());
                                        k_moe_combine<<<
                                            dim3((unsigned) ((g.n_embd + kThreads - 1) / kThreads),
                                                 (unsigned) T), kThreads, 0, st>>>(
                                                x, parts, rw, T, topk, g.n_embd);
                                        CK(cudaGetLastError());
                                    });
                                }
                            } else if (ok) {
                                const auto t_fin = Clock::now();
                                ok = peer->finish_prefill(perr);
                                peer_pf_finish_ms += ms_since(t_fin);
                                if (tm_on) tm_fin_ms[tb] += ms_since(t_fin);
                                if (ok) {
                                    // the sums cross the peer's link once (mapped host) and join
                                    // on the primary's fast one; the combine skips the masked
                                    // routes (their parts rows are stale)
                                    tm_span(8, tb, [&] {
                                        CK(cudaMemcpyAsync(peer_sums_dev, peer->prefill_sums_host(),
                                                           (size_t) T * g.n_embd * 4, cudaMemcpyHostToDevice,
                                                           st));
                                        peer_bytes_down += (double) (T * g.n_embd * 4);
                                        k_moe_combine_peer<<<
                                            dim3((unsigned) ((g.n_embd + kThreads - 1) / kThreads),
                                                 (unsigned) T), kThreads, 0, st>>>(
                                                x, parts, peer_sums_dev, rw, peer_mask_dev, T, topk,
                                                g.n_embd);
                                        CK(cudaGetLastError());
                                    });
                                }
                            }
                            if (ok) {
                                peer_pf_layers += 1;
                                peer_pf_tokens += T;
                                peer_pf_rows += rows_peer;
                                peer_pf_local_rows += (int64_t) T * topk - rows_peer;
                                peer_layer = true;
                            } else {
                                // the tier is sticky-failed: close it, drop the exclusion and
                                // cover the WHOLE layer on the primary - nothing partial is
                                // published (the peer's rows/sums never joined)
                                peer_fallback(l, pf_small ? "prompt launch/finish" : "prompt launch_prefill",
                                              perr.c_str());
                            }
                        }
                    }
                    if (!peer_layer) {
                        const auto t_prim = Clock::now();
                        tm_span(8, tb, [&] {
                            bool ok;
                            if (all_rows && verify_host && T <= 8) {
                                CK(cudaMemcpyAsync(h_host, h, (size_t) T * g.n_embd * 4, cudaMemcpyDeviceToHost, st));
                                CK(cudaStreamSynchronize(st));
                                ok = experts->run_multi((int) l, h_host, ids_host, T, topk, parts_host, st);
                                if (ok) CK(cudaMemcpyAsync(parts, parts_host, (size_t) T * topk * g.n_embd * 4,
                                                           cudaMemcpyHostToDevice, st));
                            } else ok = experts->run_device((int) l, h, ids_host, T, topk, parts, st);
                            if (!ok) {
                                std::fprintf(stderr, "strata-mimo: expert prefill failed at layer %lld\n",
                                             (long long) l);
                                std::exit(1);
                            }
                            k_moe_combine<<<
                                dim3((unsigned) ((g.n_embd + kThreads - 1) / kThreads), (unsigned) T), kThreads,
                                0, st>>>(x, parts, rw, T, topk, g.n_embd);
                            CK(cudaGetLastError());
                        });
                        if (tm_on) tm_prim_ms[tb] += ms_since(t_prim);
                    }
                    if (tm_on) tm_ms[tb][11] += ms_since(tm_moe0);
                }
                if (diag) diag_delta(x, diag_snap, (size_t) T * g.n_embd, diag_layers[(size_t) l].moe_mean,
                                    diag_layers[(size_t) l].moe_std);
                capture_dflash_tap((int) l, T);
                continue;   // the MoE residual add happened inside the combine
            }
            k_add<<<(unsigned) (((size_t) T * g.n_embd + kThreads - 1) / kThreads), kThreads, 0, st>>>(
                x, proj, (int64_t) T * g.n_embd);
            CK(cudaGetLastError());
            if (tm_on) tm_ms[tb][10] += ms_since(tm_half0);   // the dense layer: the whole layer is
                                                              // the "attention half" wall
            if (diag) diag_delta(x, diag_snap, (size_t) T * g.n_embd, diag_layers[(size_t) l].moe_mean,
                                diag_layers[(size_t) l].moe_std);
            capture_dflash_tap((int) l, T);
        }
        if (window_split_used) ++window_split_windows;

        // --- final norm + lm_head + argmax on the LAST row only
        const auto tm_tail0 = Clock::now();
        tm_span(9, tb, [&] {
            k_rmsnorm<<<T, kThreads, 0, st>>>(x, reinterpret_cast<const float*>(pack->norm_final), h, g.n_embd,
                                            (float) g.rms_eps);
            CK(cudaGetLastError());
            if (all_rows) {   // a verify window: every row's pick, and every row's state stays in `h`
                if (diag_logits == nullptr)
                    CK(cudaMalloc(&diag_logits, (size_t) 8 * g.n_vocab * 4));   // the widest window
                if (arg_all == nullptr) CK(cudaMalloc(&arg_all, 8 * 4));
                linear(pack->lm_head, pack->lm_q8.w, pack->tlm, h, diag_logits, T, g.n_vocab, g.n_embd);
                k_argmax_rows<<<dim3(1, (unsigned) T), 1024, 0, st>>>(diag_logits, g.n_vocab, arg_all);
            }
            if (!all_rows) {
                if (diag) {   // --diag-tf wants every row's logits through the same path
                    if (diag_logits == nullptr) CK(cudaMalloc(&diag_logits, (size_t) T * g.n_vocab * 4));
                    linear(pack->lm_head, pack->lm_q8.w, pack->tlm, h, diag_logits, T, g.n_vocab, g.n_embd);
                } else {
                    linear(pack->lm_head, pack->lm_q8.w, pack->tlm, h + (size_t) (T - 1) * g.n_embd, logits,
                           1, g.n_vocab, g.n_embd);
                }
                k_argmax<<<1, 1024, 0, st>>>(diag ? diag_logits : logits, g.n_vocab, arg);
            }
            CK(cudaGetLastError());
        });
        pos = pos0 + T;
        last_T = T;
        auto tm_finish = [&] {
            if (tm_on) {
                tm_tail_ms[tb] += ms_since(tm_tail0);
                tm_wall_ms[tb] += ms_since(tm_t0);
                tm_calls[tb] += 1;
            }
        };
        if (all_rows) {
            if (async_handoff) {
                std::unique_lock<std::mutex> lock(handoff_mu);
                handoff_cv.wait(lock, [&] { return !handoff_busy; });
                if (handoff_error) std::rethrow_exception(handoff_error);
            }
            report_window_cpu();
            report_window_gpu_hits();
            report_window_router_split();
            tm_finish(); return -1; // caller owns the verify readback/sync
        }
        int32_t out = -1;
        CK(cudaMemcpyAsync(&out, arg, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        if (async_handoff) {
            std::unique_lock<std::mutex> lock(handoff_mu);
            handoff_cv.wait(lock, [&] { return !handoff_busy; });
            if (handoff_error) std::rethrow_exception(handoff_error);
        }
        report_window_cpu();
        report_window_gpu_hits();
        report_window_router_split();
        tm_finish();   // include the token readback and final GPU wait in decode wall/tail
        if (tm_on) {
            tm_harvest();
            if (tb == 0 && ++tm_decodes % 32 == 0) tm_print();
        }
        return out;
    }

    struct DflashResult { std::vector<int32_t> tokens; int32_t next = -1; int committed = 0; };
    DflashResult dflash_step(int32_t carry, int remaining, const Sampling& sampling,
                             const std::vector<int32_t>& history, std::mt19937_64& rng) {
        DflashResult result;
        result.tokens.push_back(carry);
        if (remaining <= 1 || is_eos(carry) || cap - pos <= 1) return result;
        const int k_max = std::min({dflash_tokens, dflash->cfg.block - 1, remaining - 1, (int) (cap - pos - 1)});
        const int64_t start_pos = pos;
        const bool adaptive = strata::core::exl3_flag("STRATA_MIMO_DFLASH_ADAPT", nullptr, false);
        if (adaptive && (dflash_calibration < 4 || dflash_backoff > 0)) {
            DfRange df_fb("df.fallback", df_trace);
            const auto start = Clock::now();
            result.next = forward(&carry, 1);
            commit_dflash(start_pos, 1);
            const double elapsed = ms_since(start);
            dflash_serial_ms = dflash_calibration ? .75 * dflash_serial_ms + .25 * elapsed : elapsed;
            ++dflash_calibration;
            if (dflash_backoff) --dflash_backoff;
            ++dflash_fallback_tokens;
            result.committed = 1;
            df_fb.stop();
            return result;
        }
        DfRange df_round("df.round", df_trace);
        const auto draft0 = Clock::now();
        const bool short_block = strata::core::exl3_flag("STRATA_MIMO_DFLASH_SHORT_BLOCK", nullptr, false);
        DfRange df_draft("df.draft", df_trace);
        // the adaptive window (awin above) picks how many proposals THIS round verifies; the
        // drafter always produces its full block (the reference's contract - draft cost stays
        // fixed and the drafts themselves never depend on the chosen width)
        int k = k_max;
        if (awin_on && k_max >= 1) k = awin.choose(k_max);
        if (!dflash->draft(carry, pos, dflash_ids, awin_on || !short_block ? 0 : k + 1))
            throw std::runtime_error("dflash: draft failed");
        if (awin_on) ++awin.hist[(size_t) k];
        int32_t window[8] = {carry};
        CK(cudaMemcpy(window + 1, dflash_ids, (size_t) k * 4, cudaMemcpyDeviceToHost));
        dflash_draft_ms += ms_since(draft0);
        df_draft.stop();
        DfRange df_verify("df.verify", df_trace);
        const auto verify0 = Clock::now();
        forward(window, k + 1, true);
        int32_t picks[8];
        std::vector<float> probabilities;
        if (sampling.needs_logits()) {
            probabilities.resize((size_t) (k + 1) * g.n_vocab);
            CK(cudaMemcpyAsync(probabilities.data(), diag_logits, probabilities.size() * 4, cudaMemcpyDeviceToHost, st));
        } else CK(cudaMemcpyAsync(picks, arg_all, (size_t) (k + 1) * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        dflash_verify_ms += ms_since(verify0);
        df_verify.stop();
        std::vector<int32_t> accepted_history = history;
        accepted_history.push_back(carry);
        std::vector<float> row;
        int a = 0;
        for (;;) {
            int32_t predicted;
            if (sampling.needs_logits()) {
                const float* source = probabilities.data() + (size_t) a * g.n_vocab;
                row.assign(source, source + g.n_vocab);
                predicted = sample(row, sampling, accepted_history, rng);
            } else predicted = picks[a];
            result.next = predicted;
            if (a == k || predicted != window[a + 1]) break;
            result.tokens.push_back(predicted);
            accepted_history.push_back(predicted);
            ++a;
            if (is_eos(predicted)) break;
        }
        result.committed = 1 + a;
        pos = start_pos + result.committed;
        // Commit only target-computed carry/accepted rows. Rejected speculative KV
        // is outside `pos`; the SWA ring's verify slack retains all rollback history.
        DfRange df_commit("df.commit", df_trace);
        commit_dflash(start_pos, result.committed);
        ++dflash_rounds; dflash_accepted += a; dflash_offered += k;
        ++dflash_hist[a];
        if (awin_on && k >= 1) awin.update(k, a);   // labels: accepted prefix + first mismatch
        if (adaptive) {
            const double per_token = ms_since(draft0) / result.committed;
            if (per_token > 1.05 * dflash_serial_ms) ++dflash_slow;
            else dflash_slow = 0;
            if (dflash_slow >= 2) { dflash_backoff = 8; dflash_slow = 0; }
        }
        df_commit.stop();
        df_round.stop();
        return result;
    }

    /// One speculation round's verify: the window [anchor, drafts[0..k-1]] (k+1 rows <= 8, k >= 2)
    /// through the ordinary forward; `picks[r]` = the token predicted after window row r.  Returns the
    /// accepted draft count a under greedy-while-match, CAPPED at k-1: the next anchor's cell pairs
    /// picks[a] with the state of window row a+1, which exists only for a <= k-1 (a full accept's
    /// bonus would need a state one row past the window - an extra non-spec forward for the round's
    /// rare best case, not taken).  KV rows past the committed prefix are stale by construction: the
    /// next window rewrites the cache from its own anchor position on.
    int verify_window(int32_t anchor, const int32_t* drafts, int k, int32_t* picks) {
        int32_t win[8];
        win[0] = anchor;
        for (int i = 0; i < k; ++i) win[i + 1] = drafts[i];
        forward(win, k + 1, /*all_rows=*/true);
        CK(cudaMemcpyAsync(picks, arg_all, (size_t) (k + 1) * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        int a = 0;
        while (a < k && picks[a] == drafts[a]) ++a;
        return a > k - 1 ? k - 1 : a;
    }

    int last_T = 0;   // the last forward's row count (the final states in `h` are [last_T][n_embd])

    /// --mtp N: the drafter's caches (capacity = the context bound) and its scratch.  KNOWN
    /// DEFECT (2026-11-09 review): the head count N is NOT passed to MimoMtp::init - the
    /// module always drafts ALL g.n_mtp_layers (3) heads (their full chain + lm_head picks
    /// + KV upkeep), while the engine verifies only the FIRST N drafts.  The VERIFY window
    /// is [anchor, N drafts] and acceptance is capped at N-1 (verify_window's comment).
    /// N = 1 drafts 2 (a 1-wide window cannot accept and still keep a bonus state), so the
    /// effective window is never degenerate.
    bool init_mtp(int heads, std::string& err) {
        if (mtp != nullptr) return true;
        if (g.n_mtp_layers == 0 || pack->mtp.empty()) {
            err = "mimo: --mtp needs checkpoint MTP weights; they are absent or disabled by STRATA_MIMO_LOAD_MTP=0";
            return false;
        }
        mtp_heads = std::max(2, heads);
        mtp = new strata::mimo::MimoMtp();
        if (!mtp->init(g, pack.get(), 0, cap, std::max<int64_t>(chunk, 8), st, err)) return false;
        CK(cudaMalloc(&d_anchor_tok, 4));
        accept_hist.assign((size_t) mtp_heads, 0);
        if (mtp_time) raw_match_hist.assign((size_t) mtp_heads, 0);
        std::fprintf(stderr, "strata-mimo: MTP: %d head(s), %d-draft rounds (acceptance capped at %d), "
                             "%.2f GiB drafter VRAM\n",
                     heads, mtp_heads, mtp_heads - 1, mtp->vram_bytes() / 1073741824.0);
        return true;
    }

    /// One MTP round from `anchor_tok` (the last committed token, at position anchor_pos == pos - 1)
    /// whose target state is `anchor_state` (device F32 [n_embd]).  Emits up to k tokens into `out`;
    /// returns the count (0 = the context is full).  Follows the module contract exactly:
    /// draft -> verify -> rewind(anchor_pos+1) -> catch_up(the accepted drafts + the bonus, states
    /// shifted one row, same absolute-index pairing).
    int mtp_round(int32_t anchor_tok, const float* anchor_state, std::vector<int32_t>& out, int n_gen) {
        const int k = mtp_heads;
        // the anchor is the module's LAST COMMITTED CELL: after the prompt that is the prompt's last
        // token (KV written); after a round it is the bonus (its cell was committed by catch_up, its
        // target KV arrives with THIS window's first row - so the window starts at anchor_pos, which
        // for the bonus is one row EARLIER than the engine's stale `pos`)
        const int64_t anchor_pos = mtp->committed() - 1;
        if (anchor_pos < 0 || anchor_pos + k + 1 > cap) return 0;
        pos = anchor_pos;   // the verify window covers [anchor, drafts]; the anchor row's rewrite is
                            // idempotent (the same token through the same layer)
        CK(cudaMemcpyAsync(d_anchor_tok, &anchor_tok, 4, cudaMemcpyHostToDevice, st));
        int32_t ids[3] = {0, 0, 0};
        std::string err;
        const auto mtp_t0 = mtp_time ? Clock::now() : Clock::time_point{};
        if (!mtp->draft(d_anchor_tok, anchor_state, anchor_pos, ids, err)) {
            std::fprintf(stderr, "strata-mimo: mtp draft: %s\n", err.c_str());
            std::exit(1);
        }
        if (mtp_time) mtp_draft_ms += ms_since(mtp_t0);
        int32_t picks[8];
        const auto mtp_t1 = mtp_time ? Clock::now() : Clock::time_point{};
        const int a = verify_window(anchor_tok, ids, k, picks);
        if (mtp_time) mtp_verify_ms += ms_since(mtp_t1);
        if (mtp_time) {   // the uncapped per-position view of the same round (diagnostic only)
            int raw = 0;
            while (raw < k && picks[raw] == ids[raw]) ++raw;
            for (int j = 0; j < raw; ++j) ++raw_match_hist[(size_t) j];
            if (raw >= k) { ++cap_bound_rounds; cap_lost_bonus += raw - (k - 1); }
        }
        // the committed cells: the accepted drafts picks[0..a-1] and the bonus picks[a], at absolute
        // positions anchor_pos+1 .. anchor_pos+1+a, each paired with the state of window row 1+i
        int32_t cells[8];
        for (int i = 0; i <= a; ++i) cells[i] = picks[i];
        const auto mtp_t2 = mtp_time ? Clock::now() : Clock::time_point{};
        CK(cudaMemcpyAsync(tok, cells, (size_t) (a + 1) * 4, cudaMemcpyHostToDevice, st));
        CK(cudaStreamSynchronize(st));
        mtp->rewind(anchor_pos + 1);
        if (!mtp->catch_up(tok, h + (size_t) g.n_embd, a + 1, anchor_pos + 1, err)) {
            std::fprintf(stderr, "strata-mimo: mtp catch_up: %s\n", err.c_str());
            std::exit(1);
        }
        if (mtp_time) mtp_commit_ms += ms_since(mtp_t2);
        pos = anchor_pos + a + 1;   // COMMIT: only [anchor, accepted drafts] stay valid; the stale
                                    // window tail is overwritten by the next round's first row
        ++rounds;
        ++offered_total;
        accepted_total += a;
        ++accept_hist[(size_t) a];
        last_accept = a;
        int emitted = 0;
        for (int i = 0; i <= a && (int) out.size() < n_gen; ++i) {
            out.push_back(picks[i]);
            ++emitted;
            if (is_eos(picks[i])) break;
        }
        return emitted;
    }

    /// A prompt of P tokens in chunks; the argmax comes from the LAST chunk only.
    int32_t process(const int32_t* toks, int64_t P, int chunk, void (*progress)(int64_t, int64_t, double)) {
        int32_t nx = -1;
        const auto t0 = Clock::now();
        for (int64_t done = 0; done < P; done += chunk) {
            const int T = (int) std::min<int64_t>(chunk, P - done);
            nx = forward(toks + done, T);
            if (dflash) commit_dflash(pos - T, T);
            if (mtp != nullptr) {   // committed cells for the whole chunk, same-index states
                std::string err;
                if (!mtp->catch_up(tok, h, T, mtp->committed(), err)) {
                    std::fprintf(stderr, "strata-mimo: mtp catch_up(prompt): %s\n", err.c_str());
                    std::exit(1);
                }
            }
            if (progress) progress(done + T, P, ms_since(t0));
        }
        return nx;
    }

    /// The last window's logits, widened to the host (sampling in serve mode).
    void host_logits(std::vector<float>& out) {
        out.resize((size_t) g.n_vocab);
        CK(cudaMemcpyAsync(out.data(), logits, (size_t) g.n_vocab * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }
};

// ------------------------------------------------------------------ the first-light diagnostics
/// The embed self-logit check: logits(embed(t)) = lm_head @ embed(t) for a sample of tokens.  NOTE THE
/// MEASURED BASELINE: this model's UNTIED head has NO diagonal - 0 of 2048 sampled tokens self-hit with
/// VERIFIED-correct weights (numpy on the cache, both layout interpretations), so a low rate here does
/// NOT imply a scramble.  It caught nothing on first light; the layout bug was caught by the staged-vs-
/// folded comparison instead (rel 2.1e-4 once right).  Kept as a cheap future-facing probe.
int diag_embed(Engine& E) {
    const int64_t V = E.g.n_vocab, D = E.g.n_embd;
    const int batch = 256, samples = 2048;
    uint16_t* rows = nullptr;
    float* lg = nullptr;
    int32_t* toks = nullptr;
    int32_t* arg = nullptr;
    CK(cudaMalloc(&rows, (size_t) batch * D * 2));
    CK(cudaMalloc(&lg, (size_t) batch * V * 4));
    CK(cudaMalloc(&toks, batch * 4));
    CK(cudaMalloc(&arg, (size_t) batch * 4));
    int hits = 0;
    std::vector<int32_t> arg_host((size_t) batch);
    std::vector<std::pair<int32_t, float>> examples;
    for (int b = 0; b < samples; b += batch) {
        std::vector<int32_t> ids((size_t) batch);
        for (int i = 0; i < batch; ++i) ids[(size_t) i] = (int32_t)(((int64_t)(b + i) * V) / samples);
        CK(cudaMemcpy(toks, ids.data(), batch * 4, cudaMemcpyHostToDevice));
        k_embed_rows16<<<batch, kThreads, 0, E.st>>>(reinterpret_cast<const uint16_t*>(E.pack->embed), toks,
                                                    rows, D);
        CK(cudaGetLastError());
        E.gemm.f16(rows, reinterpret_cast<const uint16_t*>(E.pack->lm_head), lg, batch, V, D, 0, 0.0f);
        k_argmax_rows<<<dim3(1, (unsigned) batch), 1024, 0, E.st>>>(lg, V, arg);
        CK(cudaGetLastError());
        CK(cudaMemcpy(arg_host.data(), arg, (size_t) batch * 4, cudaMemcpyDeviceToHost));
        CK(cudaStreamSynchronize(E.st));
        for (int i = 0; i < batch; ++i)
            if (arg_host[(size_t) i] == ids[(size_t) i]) {
                ++hits;
                if ((int)examples.size() < 5) examples.emplace_back(ids[(size_t) i], 0.f);
            }
    }
    std::printf("diag-embed: argmax(lm_head @ embed(t)) == t for %d of %d sampled tokens (%.1f%%)\n",
                hits, samples, 100.0 * hits / samples);
    (void)examples;
    (void)cudaFree(rows);
    (void)cudaFree(lg);
    (void)cudaFree(toks);
    (void)cudaFree(arg);
    return 0;
}

/// Teacher-forced margins + per-layer stats: the prompt once, every position's top-5 logits, and every
/// layer's attention-out / FFN-out mean+std with the router's picks for the last position.
int diag_tf(Engine& E, const std::vector<int32_t>& prompt) {
    E.diag = true;
    const int64_t P = (int64_t) prompt.size();
    int32_t nx = E.process(prompt.data(), P, (int) std::min<int64_t>(P, 64), [](int64_t, int64_t, double) {});
    std::vector<float> lg((size_t) P * E.g.n_vocab);
    CK(cudaMemcpy(lg.data(), E.diag_logits, lg.size() * 4, cudaMemcpyDeviceToHost));
    for (int64_t t = 0; t < P; ++t) {
        const float* row = lg.data() + (size_t) t * E.g.n_vocab;
        std::vector<int> idx((size_t) E.g.n_vocab);
        for (int i = 0; i < (int) E.g.n_vocab; ++i) idx[(size_t) i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                          [&](int a, int b) { return row[a] > row[b]; });
        std::printf("pos %lld tok %d: top5", (long long) t, prompt[(size_t) t]);
        for (int i = 0; i < 5; ++i)
            std::printf(" %d/%.3f%s", idx[(size_t) i], row[idx[(size_t) i]],
                        i == 0 ? "" : std::string(" (m" + std::to_string(i) + " " +
                                                  std::to_string(row[idx[(size_t) i - 1]] -
                                                                 row[idx[(size_t) i]]) + ")").c_str());
        std::printf("\n");
    }
    for (size_t l = 0; l < E.diag_layers.size(); ++l) {
        const auto& d = E.diag_layers[l];
        std::printf("layer %2zu: attn %.4f+-%.4f  ffn %.4f+-%.4f", l, d.attn_mean, d.attn_std, d.moe_mean,
                    d.moe_std);
        if (!d.ids.empty()) {
            double h = 0;
            for (float w : d.w) h -= (double)w * std::log((double)w + 1e-30);
            std::printf("  router");
            for (size_t i = 0; i < d.ids.size(); ++i) std::printf(" %d/%.3f", d.ids[i], d.w[i]);
            std::printf(" (entropy %.3f)", h);
        }
        std::printf("\n");
    }
    (void)nx;
    return 0;
}

// ------------------------------------------------------------------ serve (serve/server.py's protocol)
int serve(Engine& E) {
    // The freeze experiment's opt-in switches, validated after the full setup (the adapter
    // exists only once setup_peer ran). No effect and no output when all are unset.
    if (const char* mk = std::getenv("STRATA_MIMO_PEER_ADAPT_MARK"); mk && mk[0] != '\0' && mk[0] != '0')
        E.adapt_boundary = true;
    if (const char* v = std::getenv("STRATA_MIMO_PEER_ADAPT_FREEZE_AFTER_REQUEST")) {
        char* end = nullptr;
        const long long n = std::strtoll(v, &end, 10);
        if (end == v || (end && *end != '\0') || n < 0)
            throw std::invalid_argument("mimo: invalid STRATA_MIMO_PEER_ADAPT_FREEZE_AFTER_REQUEST");
        if (!E.peer_adapt)
            throw std::invalid_argument("mimo: STRATA_MIMO_PEER_ADAPT_FREEZE_AFTER_REQUEST requires "
                                        "STRATA_MIMO_PEER_ADAPT=1 (refusing to measure a wrong arm)");
        E.adapt_freeze_after = n;
        E.adapt_boundary = true;   // the freeze point always reports its boundary line
        if (const char* p = std::getenv("STRATA_MIMO_PEER_ADAPT_DUMP")) E.adapt_dump_path = p;
    }
    // STRATA_MIMO_BENCH_FP16_OUT_PREFILL / _DECODE (BENCHMARK-ONLY, opt-in; both absent =
    // the byte-identical default path - the only cost is two getenv reads at serve start):
    // per-request phase control of
    // STRATA_EXL3_GEMM_FP16_OUT for the FP16-out decode-regression experiment. When either
    // is set, the serve loop forces the epilogue flag to `prefill` before every request's
    // prompt and to `decode` at the UNTIMED prompt->decode boundary, and at that SAME
    // boundary in EVERY arm (so arms are compared at a matched, quiesced state): drains all
    // in-flight peer refills (the only cross-request async work - see the boundary block
    // below), prints the residency digest (boundary_report) and the FP16-OUT entry-counter
    // delta over the prefill. A post-decode digest + counter delta prints before DONE.
    // Values are strictly "0" or "1"; anything else refuses to run (a mistyped arm label
    // must not measure silently as one side of the A/B).
    bool bench_fp16_prefill_set = false, bench_fp16_decode_set = false;
    bool bench_fp16_prefill = false, bench_fp16_decode = false;
    for (const char* which : { "STRATA_MIMO_BENCH_FP16_OUT_PREFILL", "STRATA_MIMO_BENCH_FP16_OUT_DECODE" }) {
        const char* v = std::getenv(which);
        if (!v) continue;
        const bool is_prefill = std::strstr(which, "_PREFILL") != nullptr;
        bool parsed;
        if (std::strcmp(v, "0") == 0) parsed = false;
        else if (std::strcmp(v, "1") == 0) parsed = true;
        else throw std::invalid_argument(std::string("mimo: ") + which + " must be 0 or 1, got '" + v + "'");
        if (is_prefill) { bench_fp16_prefill_set = true; bench_fp16_prefill = parsed; }
        else { bench_fp16_decode_set = true; bench_fp16_decode = parsed; }
    }
    const bool bench_fp16_on = bench_fp16_prefill_set || bench_fp16_decode_set;
    if (bench_fp16_on) {
        // the hook prints the entry counters, so it enables their (default-off) collection
        K::exl3_gemm_fp16_out_counters_set_enabled(true);
        std::fprintf(stderr, "strata-mimo: bench FP16_OUT phase switch: prefill=%d%s decode=%d%s "
                             "(env-forced per request; boundary drain + digest ON)\n",
                     (int) bench_fp16_prefill, bench_fp16_prefill_set ? "" : "(unset)",
                     (int) bench_fp16_decode, bench_fp16_decode_set ? "" : "(unset)");
    }
    std::printf("READY %lld stop\n", (long long) E.cap);
    std::fflush(stdout);
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::thread reader([&] {
        std::string l;
        while (std::getline(std::cin, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") { stop.store(true); continue; }
            std::lock_guard<std::mutex> lk(mu);
            lines.push_back(l);
            cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    });
    reader.detach();
    std::vector<int32_t> live;   // the tokens whose KV the engine holds (cell i = live[i])
    std::vector<float> lg;
    long long completed = 0;     // completed GEN requests (the boundary/freeze index base)
    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return eof || !lines.empty(); });
            if (lines.empty()) return 0;
            line = lines.front();
            lines.pop_front();
        }
        if (line.empty()) continue;
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        char* p = nullptr;
        const long long max_new = std::strtoll(line.c_str() + 4, &p, 10);
        Sampling smp;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(st0); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
        }
        const std::vector<int32_t> ids = parse_ids(p);
        if (max_new < 1 || ids.empty()) {
            std::printf("ERR bad request: max_new or the ids\n");
            std::fflush(stdout);
            continue;
        }
        if ((int64_t) ids.size() + max_new > E.cap) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%lld)\n", ids.size(),
                        max_new, (long long) E.cap);
            std::fflush(stdout);
            continue;
        }
        bool bad = false;
        for (int32_t t : ids) bad |= t < 0 || t >= (int32_t) E.g.n_vocab;
        if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); std::fflush(stdout); continue; }
        stop.store(false);
        // a request that EXTENDS the held state processes only its fresh tail (the KV cache is append-only).
        // MTP serve NEVER reuses: after a round the module's committed cells include the bonus token
        // while the engine's target-KV watermark does not (mtp_round's contract - the bonus's target
        // row arrives with the NEXT window), so live.size() and E.pos diverge by that row and a prefix
        // extension would replay from a stale watermark. Every request with the module loaded starts
        // cold; DONE's reuse field reports 0 for them. (DFlash keeps the reuse path: its commits are
        // target-verified rows only.)
        size_t reuse = 0;
        if (E.mtp != nullptr) {
            E.reset();
            live.clear();
        } else if (live.size() < ids.size() && std::equal(live.begin(), live.end(), ids.begin())) reuse = live.size();
        else {
            E.reset();
            live.clear();
        }
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        // The bench hook's prefill arm applies BEFORE the first timed line (t0 above); a
        // forced setter here is authoritative over the env's apply-once state.
        K::Exl3Fp16OutCounters bench_c0{};
        if (bench_fp16_on) {
            if (bench_fp16_prefill_set) K::exl3_gemm_fp16_out_set(bench_fp16_prefill);
            bench_c0 = K::exl3_gemm_fp16_out_counters();
        }
        auto before_routes = E.cache_route_snapshot();
        auto before_staging = E.experts->cache_stats();
        int32_t nx = E.process(ids.data() + reuse, fresh, E.chunk, [](int64_t done, int64_t total, double ms) {
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) total, ms,
                        (double) done / std::max(1e-9, ms / 1000.0));
            std::fflush(stdout);
        });
        live.insert(live.end(), ids.begin() + (long) reuse, ids.end());
        const double prompt_ms = ms_since(t0);
        E.cache_report("prompt", before_routes, before_staging);
        // The bench hook's prompt->decode boundary: UNTIMED (prompt_ms captured above, the
        // decode clock d0 starts below) and run IDENTICALLY in every arm - drain the peer's
        // in-flight refills (same call the post-request boundary uses; the last forward()
        // already drained the service worker and both streams), print the post-prefill
        // residency digest plus the FP16-OUT entry counters' prefill-phase delta, then force
        // the decode arm's epilogue flag. The prefill delta is the measured per-phase
        // eligibility answer; the decode delta below should read zero (decode windows
        // dispatch no >=16-row group arms).
        if (bench_fp16_on) {
            char tag[48];
            std::snprintf(tag, sizeof tag, "postprefill-req%lld", (long long) completed);
            if (E.peer_adapt) {
                std::string aerr;
                const auto bdr0 = Clock::now();
                if (!E.peer_adapt->drain(aerr)) throw std::runtime_error("mimo bench boundary drain: " + aerr);
                std::fprintf(stderr, "strata-mimo: bench boundary drain: %.2f ms\n", ms_since(bdr0));
            }
            const auto c1 = K::exl3_gemm_fp16_out_counters();
            std::fprintf(stderr, "strata-mimo: bench fp16out prefill req%lld: group_gemm %lld calls / "
                                 "%lld rows, recon %lld calls / %lld experts / %lld rows, fp16 epilogue %lld\n",
                         (long long) completed,
                         c1.group_gemm_calls - bench_c0.group_gemm_calls,
                         c1.group_gemm_rows - bench_c0.group_gemm_rows,
                         c1.recon_calls - bench_c0.recon_calls,
                         c1.recon_experts - bench_c0.recon_experts,
                         c1.recon_rows - bench_c0.recon_rows,
                         c1.fp16_epilogue_calls - bench_c0.fp16_epilogue_calls);
            E.boundary_report(tag);
            if (bench_fp16_decode_set) K::exl3_gemm_fp16_out_set(bench_fp16_decode);
            bench_c0 = K::exl3_gemm_fp16_out_counters();   // the decode-phase delta's baseline
        }
        before_routes = E.cache_route_snapshot();
        before_staging = E.experts->cache_stats();
        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist = ids;
        const auto d0 = Clock::now();
        // optional per-request decode ROCTX range + profiler resume/pause window:
        // STRATA_TRACE_DECODE_REQUESTS selects the request (see decode_trace.hpp); with
        // rocprofv3 --selected-regions the trace then covers ONLY this decode, excluding
        // startup and prefill.  Inert without the env and the preloaded ROCTX library.
        strata::platform::DecodeTrace decode_trace;
        // per-request baselines for BOTH drafter counter sets: the DONE fields and the stderr
        // summaries below must be per-request deltas of the drafter that actually ran (the MTP
        // module counts rounds/accepted in E.rounds/E.accepted_total, DFlash in E.dflash_*)
        const auto draft_accepted0 = E.dflash_accepted, draft_offered0 = E.dflash_offered;
        const auto draft_rounds0 = E.dflash_rounds;
        const auto draft_fallback0 = E.dflash_fallback_tokens;
        const auto mtp_accepted0 = E.accepted_total, mtp_rounds0 = E.rounds;
        // gated: with STRATA_MIMO_MTP_TIME unset nothing below copies state (default path
        // unchanged, not just unchanged-output); the timed branch pays the small copies.
        const auto mtp_hist0 = E.mtp_time ? E.accept_hist : std::vector<int64_t>{};
        const auto mtp_raw0 = E.mtp_time ? E.raw_match_hist : std::vector<int64_t>{};
        const auto mtp_cap0 = E.mtp_time ? E.cap_bound_rounds : 0;
        const auto mtp_bonus0 = E.mtp_time ? E.cap_lost_bonus : 0;
        const double mtp_draft0 = E.mtp_draft_ms, mtp_verify0 = E.mtp_verify_ms, mtp_commit0 = E.mtp_commit_ms;
        const double draft_ms0 = E.dflash_draft_ms, verify_ms0 = E.dflash_verify_ms, commit_ms0 = E.dflash_commit_ms;
        long long produced = 0;
        const char* finish = "length";
        int32_t next = smp.needs_logits() ? -1 : nx;
        if (E.dflash) {
            if (smp.needs_logits()) { E.host_logits(lg); next = sample(lg, smp, hist, rng); }
            while (produced < max_new) {
                auto round = E.dflash_step(next, (int) (max_new - produced), smp, hist, rng);
                int emitted = 0;
                for (int32_t token : round.tokens) {
                    std::printf("T %d\n", token); std::fflush(stdout);
                    ++produced; ++emitted; hist.push_back(token);
                    if (is_eos(token)) { finish = "stop"; break; }
                    if (stop.load()) { finish = "cancel"; break; }
                }
                if (emitted < round.committed) {
                    // Cancellation may interrupt a verified block. Preserve only emitted KV;
                    // discard the reusable prefix because the draft ring was already committed.
                    E.reset();
                    live.clear();
                    break;
                }
                live.insert(live.end(), round.tokens.begin(), round.tokens.begin() + round.committed);
                if (finish[0] != 'l' || round.committed == 0) break;
                next = round.next;
            }
            std::fprintf(stderr, "strata-mimo: DFlash request: rounds=%lld accepted=%lld offered=%lld "
                                "draft_ms=%.2f verify_ms=%.2f commit_ms=%.2f serial_fallback=%lld\n",
                (long long) (E.dflash_rounds - draft_rounds0), (long long) (E.dflash_accepted - draft_accepted0),
                (long long) (E.dflash_offered - draft_offered0), E.dflash_draft_ms - draft_ms0,
                E.dflash_verify_ms - verify_ms0, E.dflash_commit_ms - commit_ms0,
                (long long) (E.dflash_fallback_tokens - draft_fallback0));
            if (E.awin_on) {
                std::fprintf(stderr, "strata-mimo: DFlash awindow: rounds=%lld chosen-widths:",
                             (long long) E.awin.rounds);
                for (size_t w = 1; w < E.awin.hist.size(); ++w)
                    std::fprintf(stderr, " %zu=%lld", w, (long long) E.awin.hist[w]);
                std::fprintf(stderr, "; q:");
                for (double v : E.awin.q) std::fprintf(stderr, " %.2f", v);
                std::fprintf(stderr, "\n");
            }
        } else if (E.mtp != nullptr && !smp.needs_logits()) {
            // The MTP serve arm (greedy only; the bench protocol is temperature 0).  Mirrors
            // greedy_generate's loop: the prompt's last token is the first anchor, every round
            // drafts/verifies/commits and the bonus becomes the next anchor.  Conversation reuse
            // is disabled for EVERY request while the module is loaded (the reset above); the
            // module commits the round's cells into its own KV as it goes.
            std::vector<int32_t> out;
            int32_t anchor = ids.back();
            const float* anchor_state = E.h + (size_t) (E.last_T - 1) * E.g.n_embd;
            while (produced < max_new) {
                const size_t emitted0 = out.size();
                // n_gen is the request TOTAL: mtp_round bounds every emission by out.size() < n_gen
                (void) E.mtp_round(anchor, anchor_state, out, max_new);
                int emitted = 0;
                bool broke = false;
                for (size_t i = emitted0; i < out.size(); ++i) {
                    const int32_t t = out[i];
                    std::printf("T %d\n", t); std::fflush(stdout);
                    ++produced; ++emitted; hist.push_back(t);
                    live.push_back(t);
                    if (is_eos(t)) { finish = "stop"; broke = true; break; }
                    if (stop.load()) { finish = "cancel"; broke = true; break; }
                }
                if (emitted == 0 || broke) { if (finish[0] == 'c') { E.reset(); live.clear(); } break; }
                anchor = out.back();
                anchor_state = E.h + (size_t) (E.last_accept + 1) * E.g.n_embd;
            }
            // per-request MTP deltas (mtp_rounds0/mtp_accepted0 snapshots).  KNOWN METRIC
            // CAVEAT (2026-11-09 review): every counted round DRAFTS all n_mtp_layers (3)
            // proposals; `offered` below reports rounds x mtp_heads - the VERIFIED count -
            // kept for continuity with the historical logs, not the drafted count (the
            // early context-full return happens before the round is counted).
            const long long mtp_req_rounds = (long long) (E.rounds - mtp_rounds0);
            if (E.mtp_time) {
                // HOST-ELAPSED phases (STRATA_MIMO_MTP_TIME=1): draft includes the drain of the
                // PREVIOUS round's async catch_up plus this round's drafter GPU work (draft()'s
                // own readback sync is the attribution point); verify is the k+1-row target
                // window through the mapped handoff plus its picks readback sync; commit is
                // host-side cells/rewind/catch_up ENQUEUE ONLY - its GPU cost lands in the next
                // round's draft.  The rawmatch histogram is the UNCAPPED per-position match
                // count; accepted-histogram is what the k-1 cap actually emitted.
                const auto& rm = E.raw_match_hist;
                std::fprintf(stderr, "strata-mimo: MTP request: rounds=%lld accepted=%lld offered=%lld "
                                    "tokens/round=%.2f draft_ms=%.2f verify_ms=%.2f commit_ms=%.2f "
                                    "rawmatch(pos1..%d):",
                             mtp_req_rounds, (long long) (E.accepted_total - mtp_accepted0),
                             mtp_req_rounds * E.mtp_heads,
                             produced / std::max<double>(1, (double) mtp_req_rounds),
                             E.mtp_draft_ms - mtp_draft0, E.mtp_verify_ms - mtp_verify0,
                             E.mtp_commit_ms - mtp_commit0, E.mtp_heads);
                for (size_t j = 0; j < rm.size(); ++j)
                    std::fprintf(stderr, " %lld", (long long) (rm[j] - mtp_raw0[j]));
                std::fprintf(stderr, "; cap_bound_rounds=%lld cap_lost_bonus=%lld "
                                    "accepted-histogram(0..%d):",
                             (long long) (E.cap_bound_rounds - mtp_cap0),
                             (long long) (E.cap_lost_bonus - mtp_bonus0), E.mtp_heads - 1);
                for (size_t h = 0; h < E.accept_hist.size(); ++h)
                    std::fprintf(stderr, " %lld", (long long) (E.accept_hist[h] - mtp_hist0[h]));
                std::fprintf(stderr, "\n");
            } else {
                std::fprintf(stderr, "strata-mimo: MTP request: rounds=%lld accepted=%lld offered=%lld "
                                    "tokens/round=%.2f\n",
                             mtp_req_rounds, (long long) (E.accepted_total - mtp_accepted0),
                             mtp_req_rounds * E.mtp_heads,
                             produced / std::max<double>(1, (double) mtp_req_rounds));
            }
        } else {
            if (E.mtp != nullptr)
                std::fprintf(stderr, "strata-mimo: MTP serve arm is greedy-only; decoding serially "
                                     "(MTP requests never reuse conversation state)\n");
            while (produced < max_new) {
            if (smp.needs_logits()) {
                E.host_logits(lg);
                next = sample(lg, smp, hist, rng);
            }
            std::printf("T %d\n", next);
            std::fflush(stdout);
            ++produced;
            hist.push_back(next);
            if (is_eos(next)) { finish = "stop"; break; }
            if (stop.load()) { finish = "cancel"; break; }
            if ((int64_t) live.size() + 1 > E.cap) break;
            const int32_t one = next;
            nx = E.forward(&one, 1);
            live.push_back(next);
            if (!smp.needs_logits()) next = nx;
        }
        }
        decode_trace.stop();
        const double decode_ms = ms_since(d0);
        E.cache_report("decode", before_routes, before_staging);
        if (bench_fp16_on) {   // post-decode state: digest + the decode-phase eligibility delta
            char tag[48];
            std::snprintf(tag, sizeof tag, "postdecode-req%lld", (long long) completed);
            const auto c2 = K::exl3_gemm_fp16_out_counters();
            std::fprintf(stderr, "strata-mimo: bench fp16out decode req%lld: group_gemm %lld calls / "
                                 "%lld rows, recon %lld calls / %lld experts / %lld rows, fp16 epilogue %lld\n",
                         (long long) completed,
                         c2.group_gemm_calls - bench_c0.group_gemm_calls,
                         c2.group_gemm_rows - bench_c0.group_gemm_rows,
                         c2.recon_calls - bench_c0.recon_calls,
                         c2.recon_experts - bench_c0.recon_experts,
                         c2.recon_rows - bench_c0.recon_rows,
                         c2.fp16_epilogue_calls - bench_c0.fp16_epilogue_calls);
            E.boundary_report(tag);
        }
        // The experiment boundary (opt-in): BOTH arms drain all in-flight refills here, so
        // the landing is never inside a measured region and never asymmetric between arms.
        // Refills are the only cross-request async work: the forward boundary this request's
        // last forward() returned through already drained the service worker and both
        // streams (the before_window contract), so a completed drain quiesces the tier.
        // This block sits BEFORE the DONE line on purpose: a DONE implies the boundary is
        // finished, and the strictly serial dequeue below cannot hand the next GEN to the
        // loop before it returns - the boundary is therefore never timed as next prefill.
        if (E.adapt_boundary) {
            char tag[48];
            std::snprintf(tag, sizeof tag, "req%lld", (long long) completed);
            if (E.peer_adapt) {
                std::string aerr;
                const auto bdr0 = Clock::now();
                const bool freezing = (completed == E.adapt_freeze_after) && !E.peer_adapt->frozen();
                const bool ok = freezing ? E.peer_adapt->freeze(aerr) : E.peer_adapt->drain(aerr);
                if (!ok) throw std::runtime_error("mimo adapt boundary: " + aerr);
                const double drain_ms = ms_since(bdr0);
                if (freezing) {
                    std::snprintf(tag, sizeof tag, "freeze-req%lld", (long long) completed);
                    E.write_frozen_dump();
                    std::fprintf(stderr, "strata-mimo: peer adapt frozen after request %lld "
                                       "(boundary drain %.2f ms; planning disabled, poll/observe continue)\n",
                                 (long long) completed, drain_ms);
                } else {
                    std::fprintf(stderr, "strata-mimo: peer adapt boundary drain: %.2f ms\n", drain_ms);
                }
            }
            E.boundary_report(tag);
        }
        // fields 6/7 = accepted/offered DRAFT tokens of this request (per-request deltas of the
        // drafter that ran; MTP's offered = rounds x heads, exact)
        const bool mtp_req = E.dflash == nullptr && E.mtp != nullptr && !smp.needs_logits();
        const long long acc_f = mtp_req ? (long long) (E.accepted_total - mtp_accepted0)
                                        : (long long) (E.dflash_accepted - draft_accepted0);
        const long long off_f = mtp_req ? (long long) ((E.rounds - mtp_rounds0) * E.mtp_heads)
                                        : (long long) (E.dflash_offered - draft_offered0);
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                    decode_ms, finish, acc_f, off_f, reuse, (long long) fresh);
        std::fflush(stdout);
        ++completed;
    }
}

}  // namespace

// ---- the library core: config + pack + engine, then the greedy loop.  The CLI (run) and the parity test
// (src/mimo/mimo_engine_parity.cpp) drive the SAME code; only the argument parsing and the printing differ.
namespace strata::mimo {
namespace {

/// A serve config copied from a Qwen EXL3 setup may export the RDNA-family switches.  On experts
/// they cannot decode the first expert FFN call would ABORT on the runtime throw in
/// moe_run_experts (exl3.cu's half-integer rejection) instead of falling back.  The decode twins
/// cover: the RDNA run funnels and the fused-multirow grouped path (PRIMARY/PEER_MULTIROW) ride
/// them - take integer K and half-integer K up to 4.5 (a K 5.5+ frac run can exceed the single
/// funnel's 32 bits), and the batched WMMA twins (EXPERT_BATCH / PEER_BATCH) take K 2..6.5 in
/// the fragments.  With the pack's expert inventory in hand, say one clear line per switch that
/// the mix leaves uncovered and turn it OFF ("0" is the explicit off to every reader); the run
/// continues on the established scalar / reconstruct + grouped-GEMM path.  Loud, never fatal -
/// and it must run BEFORE the expert stream exists, because the stream's constructor reads
/// PRIMARY_MULTIROW and reaches the RDNA level's one-time static, after which the dead arm would
/// stay cached even with the env scrubbed.
void guard_rdna_family_switches(const std::vector<int>& expert_k) {
    if (expert_k.empty()) return;
    bool frac_wide = false, batch = false;
    std::string ks;   // the expert K mix, e.g. "2, 2.5"
    for (size_t k2 = 2; k2 < expert_k.size(); ++k2) {
        if (expert_k[k2] <= 0) continue;
        if (!ks.empty()) ks += ", ";
        ks += (k2 % 2 ? std::to_string(k2 / 2) + ".5" : std::to_string(k2 / 2));
        frac_wide |= (k2 % 2) != 0 && k2 > 9;   // the RDNA funnels: half-integer K up to 4.5 only
        batch |= k2 < 4 || k2 > 13;             // the WMMA fragment decode: K 2..6.5
    }
    if (!frac_wide && !batch) return;
    const auto truthy = [](const char* name) {
        const char* v = std::getenv(name);
        return v != nullptr && v[0] != '\0' && v[0] != '0';   // the off spelling every reader shares
    };
    struct Switch { const char* name; bool blocked; const char* why; };
    const Switch table[] = {
        {"STRATA_EXL3_RDNA", frac_wide, "the RDNA run-funnel kernels take integer K and half-integer K up to 4.5"},
        {"STRATA_EXL3_PRIMARY_MULTIROW", frac_wide, "the fused multirow path rides those RDNA kernels"},
        {"STRATA_EXL3_PEER_MULTIROW", frac_wide, "the peer's fused multirow path rides those RDNA kernels"},
        {"STRATA_EXL3_EXPERT_BATCH", batch, "the batched group kernels need K 2..6.5 experts"},
        {"STRATA_EXL3_PEER_BATCH", batch, "the peer's batched group kernels need K 2..6.5 experts"},
    };
    for (const Switch& s : table)
        if (s.blocked && truthy(s.name)) {
            std::fprintf(stderr, "strata-mimo: %s does not apply to this model's experts (K %s: %s) - "
                                 "ignored, continuing on the established path\n",
                         s.name, ks.c_str(), s.why);
            setenv(s.name, "0", 1);
        }
}

/// Everything from config.json to a ready engine (the pack build's progress goes to stderr).
bool load_engine(const std::string& dir, int64_t ctx, int chunk, double expert_vram, int64_t expert_ram,
                 bool warm, bool dense_f16, bool dense_q8, Engine& E, std::string& error,
                  const std::string& dflash_dir = {}) {
    if (!E.preflight_window_cpu(error)) return false;
    if (!E.preflight_window_gpu_hits(error)) return false;
    if (!E.preflight_window_router_split(error)) return false;
    if (!E.preflight_peer_device_results(error)) return false;
    if (!E.preflight_window_device_misses(error)) return false;
    const char* resident_env = std::getenv("STRATA_EXL3_RESIDENT");
    const std::string resident_mode = resident_env ? resident_env : "0";
    const bool resident = resident_mode != "0" && !resident_mode.empty();
    if (resident && resident_mode != "all" && resident_mode != "peer") {
        error = "STRATA_EXL3_RESIDENT must be 0, all (full RAM arena), or peer (peer complement)";
        return false;
    }
    if (resident && expert_ram != 0) {
        error = "managed resident EXL3 is the single host backing; --expert-ram-cache must be 0";
        return false;
    }
    if (resident_mode == "peer" && E.peer_device < 0) {
        error = "STRATA_EXL3_RESIDENT=peer requires --peer-device";
        return false;
    }
    const std::string config_path = dir + "/config.json";
    if (!strata::core::mimo_detect_config(config_path, &error)) {
        error = dir + " is not a MiMo v2 model dir (" +
                (error.empty() ? "config.json does not declare model_type mimo_v2" : error) + ")";
        return false;
    }
    try {
        E.g = strata::core::mimo_geometry_from_config(read_text(config_path, error));
    } catch (const std::exception& e) {
        error = config_path + ": " + e.what();
        return false;
    }
    // Some DFlash-only exports retain num_nextn_predict_layers in config.json but
    // omit every MTP tensor. Explicitly skip those optional heads, without changing
    // the source config or the target model's attention/router/expert geometry.
    // Unset/1 retains the established validation and loading path.
    if (const char* load_mtp = std::getenv("STRATA_MIMO_LOAD_MTP")) {
        if (std::strcmp(load_mtp, "0") == 0) {
            E.g.n_mtp_layers = 0;
            std::fprintf(stderr, "strata-mimo: checkpoint MTP weights disabled (STRATA_MIMO_LOAD_MTP=0); target and external DFlash unchanged\n");
        } else if (std::strcmp(load_mtp, "1") != 0) {
            error = "STRATA_MIMO_LOAD_MTP must be 0 or 1";
            return false;
        }
    }
    if (!strata::kernels::mimo_router_supported((int) E.g.n_expert, (int) E.g.n_expert_used)) {
        error = "this geometry's " + std::to_string(E.g.n_expert) + " experts / top-" +
                std::to_string(E.g.n_expert_used) + " is not a shape the router kernel was compiled for";
        return false;
    }
    // ---- STRATA_MIMO_BAND8 (opt-in, this phase): the packed routed-expert backing ----------
    //
    // Both knobs are read HERE, before the pack build's large GPU allocations and before any
    // cache byte is written, so a misconfigured opt-in fails fast and cheap:
    //   STRATA_MIMO_BAND8=1        - enable via the persistent packed cache (default off:
    //                               the native source path, unchanged)
    //   STRATA_MIMO_BAND8=memory   - RUNTIME repack: permute the routed experts into the
    //                               canonical Band8 order straight into the STRATA_EXL3_RESIDENT=all
    //                               RAM arena at startup. NO disk cache, descriptor or payload
    //                               file is created or read; the offline repack is untouched.
    //   STRATA_EXL3_BAND8_CACHE=<dir> - REQUIRED with =1: an explicit cache root (the packed
    //                                payload is ~the routed experts' size, ~79.5 GiB on the
    //                                full model; it must live on a drive that fits it and is
    //                                the operator's choice, never a default path). With
    //                                =memory it must be UNSET - a mixed intent is refused.
    // The build itself (below) happens AFTER the config/GPU/ISA checks and BEFORE E.init
    // builds the expert stream and registers the CPU tier - both consume the backing.
    bool band8 = false;
    bool band8_memory = false;
    std::string band8_cache_root;
    if (const char* band8_env = std::getenv("STRATA_MIMO_BAND8");
            band8_env != nullptr && band8_env[0] != '\0' && band8_env[0] != '0') {
        band8_memory = std::strcmp(band8_env, "memory") == 0;
        if (!band8_memory && std::strcmp(band8_env, "1") != 0) {
            error = "STRATA_MIMO_BAND8 must be 1 (the packed cache under "
                    "STRATA_EXL3_BAND8_CACHE) or memory (a runtime repack into the "
                    "STRATA_EXL3_RESIDENT=all RAM arena, no disk cache) - got '";
            error += band8_env;
            error += "' (refusing instead of guessing)";
            return false;
        }
        const char* root = std::getenv("STRATA_EXL3_BAND8_CACHE");
        if (band8_memory) {
            if (root != nullptr && root[0] != '\0') {
                error = "STRATA_MIMO_BAND8=memory never builds or reads a packed disk cache - "
                        "unset STRATA_EXL3_BAND8_CACHE (or use =1 for the cache mode); "
                        "refusing the mixed intent";
                return false;
            }
            // The runtime repack's ONLY destination is the full pinned RAM arena, and it
            // must be converted BEFORE E.init's first GPU fill (see the resident block
            // below) - so the resident mode is fixed here, at the cheapest possible point.
            if (!resident || resident_mode != "all") {
                error = "STRATA_MIMO_BAND8=memory repacks the routed experts into the "
                        "STRATA_EXL3_RESIDENT=all RAM arena at startup - set it (resident '";
                error += resident_mode;
                error += "' cannot host the runtime repack: the peer-complement arrangement "
                         "stages unconverted native bytes before activation and is refused "
                         "this phase, not silently mixed)";
                return false;
            }
        } else if (root == nullptr || root[0] == '\0') {
            error = "STRATA_MIMO_BAND8=1 also needs STRATA_EXL3_BAND8_CACHE=<explicit cache dir> "
                    "(the packed expert payload is about the routed experts' size and must live on a "
                    "drive that fits it; nothing is written or allocated without it)";
            return false;
        }
        band8_cache_root = root != nullptr ? root : "";
        if (expert_ram != 0) {
            // The stream refuses a managed RAM cache beside a backing (one host copy is the
            // point); saying it HERE keeps the failure a clear CLI error, not a ctor throw.
            // EXACTLY zero, not "> 0": the parser maps "auto" to 0, so anything else -
            // including a negative value that would later widen to a huge size_t - is a
            // misconfiguration this phase.
            error = "STRATA_MIMO_BAND8 keeps the packed backing as the single host copy - "
                    "--expert-ram-cache must be exactly 0 (or auto) with it, got " +
                    std::to_string(expert_ram);
            return false;
        }
#if defined(STRATA_USE_HIP)
        {   // gfx1100/gfx11 is the tested primary+peer pair this phase; the band-8 tile
            // addressing itself is GPU-independent, but no other ISA has been validated end to
            // end - refuse instead of silently accepting an untested feature.  The PEER card is
            // validated too when the tier is armed (E.peer_device is set by the caller before
            // load_engine; the run_greedy/verify helpers leave it at -1): cudaGetDeviceProperties
            // needs NO device switch, so this stays cheap and side-effect-free.
            int dev = 0;
            cudaDeviceProp prop{};
            if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess ||
                std::strncmp(prop.gcnArchName, "gfx11", 5) != 0) {
                error = "STRATA_MIMO_BAND8 is validated on gfx11 (RDNA3, gfx1100/gfx110x) HIP builds only "
                        "this phase - this GPU/build is not one (no silent untested feature)";
                return false;
            }
            if (E.peer_device >= 0) {
                int n_dev = 0;
                cudaDeviceProp pprop{};
                const char* why = nullptr;
                if (cudaGetDeviceCount(&n_dev) != cudaSuccess || E.peer_device >= n_dev)
                    why = "is out of range";
                else if (E.peer_device == dev)
                    why = "is the primary GPU";
                else if (cudaGetDeviceProperties(&pprop, E.peer_device) != cudaSuccess ||
                         std::strncmp(pprop.gcnArchName, "gfx11", 5) != 0)
                    why = "is not a gfx11 part (the band-8 pair is validated on gfx1100/gfx110x only)";
                if (why != nullptr) {
                    error = "STRATA_MIMO_BAND8 with --peer-device " + std::to_string(E.peer_device) + ": the "
                            "peer " + why + " - refusing before any cache write";
                    return false;
                }
            }
        }
#else
        error = "STRATA_MIMO_BAND8 is validated on HIP (gfx11) builds only this phase - this is a "
                "CUDA build (no silent untested feature)";
        return false;
#endif
        // Presence is not intent: '0' is an explicit OFF and must keep a GPU-only packed run
        // alive on any CPU - the SAME exl3_flag evaluation the expert stream's own tier switch
        // uses.  The tier would register Band8 matrices, so the AVX-512 band kernels (isa >= 2,
        // BW-class) are probed HERE, before the ~79 GiB payload is written - not at the first
        // refused row.  Constructing the probe spawns no pool (the pool starts on first use);
        // its ISA detection is process-static, so tests exercise this through fresh children.
        if (strata::core::exl3_flag("STRATA_EXL3_CPU_TIER", nullptr, false)) {
            if (!K::cpu::Exl3CpuMoe::supported()) {
                error = "STRATA_MIMO_BAND8 with the CPU tier needs EXL3-capable CPU kernels";
                return false;
            }
            const K::cpu::Exl3CpuMoe::Stats probe = K::cpu::Exl3CpuMoe().stats();
            if (probe.isa < 2) {
                error = "STRATA_MIMO_BAND8 with the CPU tier needs an AVX-512BW-class CPU for the "
                        "band-8 band kernels (detected ISA level " + std::to_string(probe.isa) + ", need >= 2; "
                        "STRATA_EXL3_CPU_MAX_ISA caps detection for tests)";
                return false;
            }
        }
        band8 = true;
    }
    int device = 0;
    CK(cudaGetDevice(&device));
    {
        size_t free_vram = 0, total_vram = 0;
        CK(cudaMemGetInfo(&free_vram, &total_vram));
        std::fprintf(stderr, "strata-mimo: device %d, %.2f GiB free of %.2f GiB\n", device,
                     free_vram / 1073741824.0, total_vram / 1073741824.0);
    }
    // The dense-serving mode, three arms (the pack reads these at build; the arena sizes itself
    // from what the pack leaves free, so the heavier arms auto-shrink the expert slots):
    //   trellis (default) - the raw EXL3 planes on device, exl3_gemv_f32 / exl3_dense_gemv_multi
    //   f16 (--dense-f16) - the reconstructed F16 set through cublas, the parity oracle
    //   q8 (--dense-q8)   - the reconstruction quantized at load to GGUF q8_0 blocks, MMVQ-served
    if (dense_f16 && dense_q8) {
        error = "--dense-f16 and --dense-q8 are alternative dense modes - pick one";
        return false;
    }
    setenv("STRATA_MIMO_DENSE_F16", dense_f16 ? "1" : "0", 1);
    setenv("STRATA_MIMO_DENSE_Q8", dense_q8 ? "1" : "0", 1);
    const char* dense_mode = dense_f16 ? "f16" : (dense_q8 ? "q8_0" : "trellis");
    const Clock::time_point t_pack = Clock::now();
    MimoInventory inv;
    std::unique_ptr<MimoPack> pack =
        strata::mimo_pack_build(dir, E.g, device, error, [](const char* line) { std::fprintf(stderr, "%s\n", line); },
                                &inv);
    if (!pack) return false;
    // a copied Qwen serve config's RDNA-family switches, judged against the pack's expert
    // inventory and scrubbed here (BEFORE E.init builds the expert stream - see the guard)
    guard_rdna_family_switches(inv.expert_k_counts);
    std::fprintf(stderr, "strata-mimo: pack: %.2f GiB dense in %.1f s\n", pack->vram_bytes / 1073741824.0,
                 ms_since(t_pack) / 1000.0);
    if (std::getenv("STRATA_MIMO_WARMUP") != nullptr && std::getenv("STRATA_MIMO_WARMUP")[0] == '1')
        warm = true;
    const bool host_warm = strata::core::exl3_flag("STRATA_MIMO_HOST_WARMUP", nullptr, false);
    if (warm && !band8 && !host_warm && !resident) {   // one plain sequential sweep of the shards: the page cache IS the
                            // expert arena (measured; the first prompt is ~cold otherwise),
                            // ~30 s for the 85 GiB model.  The band-8 arm does NOT warm the
                            // native routed planes (its working weights are the PACKED payload,
                            // warmed below after the cache opens) - sweeping the source here
                            // would fault 80 GB of a format the run never reads.
        const auto t_w = Clock::now();
        double swept = 0;
        for (const auto& entry : std::filesystem::directory_iterator(dir))
            if (entry.path().extension() == ".safetensors") {
                FILE* f = std::fopen(entry.path().c_str(), "rb");
                if (!f) continue;
                char buf[1 << 20];
                size_t n;
                while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) swept += (double)n;
                std::fclose(f);
            }
        std::fprintf(stderr, "strata-mimo: warmup: %.1f GiB swept in %.1f s\n", swept / 1073741824.0,
                     ms_since(t_w) / 1000.0);
    }
    if (!dflash_dir.empty() && !E.prepare_dflash(dflash_dir, std::max(chunk, 8), error)) return false;
    if (band8 && !band8_memory) {
        // The packed source is built/opened BEFORE the expert stream and the CPU tier exist
        // (E.init below), so every consumer opens against the finished generation.  The
        // request list is explicit: every MoE layer's every expert, all three projections
        // (the main routed stack; dense layer 0 and the MTP heads stay native), named by the
        // same strata::mimo_expert_tensor the stream and the pack use.  The store validates
        // each linear (mul1, K2/K2.5/K3/K4, H128 dims) before writing payload bytes, never
        // overwrites the native weights, and reuses a matching generation without a payload
        // scan.  This phase ships TINY fixtures only - no full-model cache run is claimed.
        const Clock::time_point t_b8 = Clock::now();
        std::vector<std::string> prefixes;
        {
            int64_t moe = 0;
            for (int64_t l = 0; l < E.g.n_layers; ++l)
                if (E.g.is_moe(l)) ++moe;
            prefixes.reserve((size_t) moe * (size_t) E.g.n_expert * 3);
            for (int64_t l = 0; l < E.g.n_layers; ++l) {
                if (!E.g.is_moe(l)) continue;
                for (int64_t e = 0; e < E.g.n_expert; ++e)
                    for (const char* proj : {"gate", "up", "down"})
                        prefixes.push_back(strata::mimo_expert_tensor((int) l, (int) e, proj));
            }
        }
        strata::Exl3Band8BuildOptions opts;
        opts.progress_every_triples = 256;
        try {
            E.expert_backing = strata::Exl3Band8Store::build_or_open(dir, band8_cache_root, prefixes, opts);
        } catch (const std::exception& e) {
            error = std::string("band8 expert cache: ") + e.what();
            return false;
        }
        // Startup summary: mode + the resource's own identity (owner/path/records/bytes).  No
        // prompt or request data touches this line.
        const auto* store = static_cast<const strata::Exl3Band8Store*>(E.expert_backing.get());
        std::error_code ec;
        const uintmax_t payload_bytes = std::filesystem::file_size(store->payload_path(), ec);
        std::fprintf(stderr, "strata-mimo: band8 experts: %s backing, %zu plane record(s), %.2f GiB "
                             "payload at %s (fingerprint %016llx) in %.1f s; native weights untouched, "
                             "dense/MTP projections stay native\n",
                     "shared packed", store->record_count(),
                     (double) payload_bytes / 1073741824.0, store->payload_path().c_str(),
                     (unsigned long long) store->fingerprint(), ms_since(t_b8) / 1000.0);
        if (warm && !host_warm && !resident && payload_bytes > 0) {
            // The packed arm's warm sweep: ONE bounded sequential read of the payload - the
            // physical bytes this run's staging preads - so a benchmarked --warm selects the
            // WORKING weights in both arms instead of faulting 80 GB of a format never read.
            // The scales are payload bytes too (copied at pack time); the native source's OS
            // page cache is simply left alone (no claim it is empty, no flush/fadvise of
            // anything unrelated).
            const auto t_w = Clock::now();
            FILE* f = std::fopen(store->payload_path().c_str(), "rb");
            if (f == nullptr) {
                error = "band8 warmup: cannot open the payload " + store->payload_path();
                return false;
            }
            double swept = 0;
            char buf[4 << 20];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) swept += (double)n;
            const bool bad = std::ferror(f) != 0;
            std::fclose(f);
            if (bad || (uintmax_t) swept != payload_bytes) {
                error = "band8 warmup: short/error read of " + store->payload_path() + " (" +
                        std::to_string((unsigned long long) swept) + " of " +
                        std::to_string((unsigned long long) payload_bytes) + " bytes)";
                return false;
            }
            std::fprintf(stderr, "strata-mimo: warmup: %.1f GiB packed payload swept in %.1f s\n",
                         swept / 1073741824.0, ms_since(t_w) / 1000.0);
        }
    }
    if (resident) {
        std::vector<strata::Exl3ResidentBacking::Group> groups;
        for (int l = 0; l < E.g.n_layers; ++l) if (E.g.is_moe(l))
            for (int e = 0; e < E.g.n_expert; ++e)
                groups.push_back({l,e,{strata::mimo_expert_tensor(l,e,"gate"),
                                      strata::mimo_expert_tensor(l,e,"up"),
                                      strata::mimo_expert_tensor(l,e,"down")}});
        try {
            // STRATA_MIMO_BAND8=memory: the arena IS the packer. There is no store to
            // source from - create() validates every projection against the NATIVE shards
            // (mul1, verified rate, bounded plane) and tags the arena Band8, and the
            // EARLY activate() below permutes the routed experts straight into the pinned
            // RAM BEFORE E.init builds the expert stream: every GPU fill (primary arena,
            // setup_peer's peer open()) and the CPU tier then stage the CONVERTED bytes
            // from one canonical copy, and locate() refusing the file tier keeps any
            // stager off the unconverted native planes.
            const auto repack = band8_memory ? strata::Exl3ResidentRepack::ToBand8
                                             : strata::Exl3ResidentRepack::None;
            E.resident_backing = strata::Exl3ResidentBacking::create(dir,groups,E.expert_backing,repack);
            E.expert_backing = E.resident_backing;
            E.resident_peer = resident_mode == "peer";
            if (band8_memory && !E.activate_resident(error)) return false;
        } catch (const std::exception& ex) { error = ex.what(); return false; }
    }
    if (!E.init(dir, std::move(pack), ctx, chunk, expert_vram, expert_ram, error)) return false;
    std::fprintf(stderr,
                 "strata-mimo: ready: ctx %lld, chunk %d, dense %s %.2f GiB, KV %.2f GiB, buffers %.0f MiB, "
                 "experts %.2f GiB\n",
                 (long long) E.cap, E.chunk, dense_mode, E.pack->vram_bytes / 1073741824.0,
                 E.cache_bytes / 1073741824.0, E.buffer_bytes / 1048576.0, E.experts->bytes() / 1073741824.0);
    return true;
}

/// The prompt in chunks, then greedy tokens until n_gen / eos / a full context.  `out` gets the generated
/// ids (the argmax after the prompt FIRST, matching the CLI's `gen:` line).
int greedy_generate(Engine& E, const int32_t* prompt, int64_t n_prompt, int n_gen, std::vector<int32_t>& out,
                    double& prompt_ms, double& decode_ms) {
    if (n_prompt + n_gen > E.cap) {
        std::fprintf(stderr, "strata-mimo: prompt (%lld) + n (%d) exceeds the context (%lld)\n",
                     (long long) n_prompt, n_gen, (long long) E.cap);
        return 2;
    }
    for (int32_t t : std::vector<int32_t>(prompt, prompt + n_prompt))
        if (t < 0 || t >= (int32_t) E.g.n_vocab) {
            std::fprintf(stderr, "strata-mimo: a token id is outside the vocabulary\n");
            return 2;
        }
    const auto t0 = Clock::now();
    int32_t nx = E.process(prompt, n_prompt, E.chunk, [](int64_t, int64_t, double) {});
    prompt_ms = ms_since(t0);
    const auto t1 = Clock::now();
    if (E.dflash) {
        const Sampling sampling;
        std::mt19937_64 rng(1234);
        std::vector<int32_t> history(prompt, prompt + n_prompt);
        while ((int) out.size() < n_gen) {
            auto round = E.dflash_step(nx, n_gen - (int) out.size(), sampling, history, rng);
            out.insert(out.end(), round.tokens.begin(), round.tokens.end());
            history.insert(history.end(), round.tokens.begin(), round.tokens.end());
            if (is_eos(out.back()) || !round.committed) break;
            nx = round.next;
        }
    } else if (E.mtp != nullptr) {
        // the speculation loop: the prompt's last token IS the first anchor (its state is the last
        // chunk's final row); every round drafts, verifies [anchor, k drafts], commits the accepted
        // prefix + the bonus, and the bonus becomes the next anchor
        int32_t anchor = prompt[n_prompt - 1];   // the prompt's last token
        const float* anchor_state = E.h + (size_t) (E.last_T - 1) * E.g.n_embd;
        while ((int) out.size() < n_gen) {
            const size_t before = out.size();
            const int emitted = E.mtp_round(anchor, anchor_state, out, n_gen);
            if (emitted == 0) break;   // the context is full
            // the bonus (the round's last emitted token) is the next anchor; its state is the window
            // row after the accepted prefix - mtp_round left `h` as the verify window's states
            const int a = E.last_accept;   // the verify window's row 1+a state is the bonus's
            anchor = out.back();
            anchor_state = E.h + (size_t) (a + 1) * E.g.n_embd;
            if (is_eos(out.back())) break;
        }
    } else {
        while ((int) out.size() < n_gen) {
            out.push_back(nx);
            if ((int) out.size() >= n_gen) break;
            if (is_eos(nx)) break;
            if (E.pos + 1 > E.cap) break;
            const int32_t one = nx;
            nx = E.forward(&one, 1);
        }
    }
    decode_ms = ms_since(t1);
    return 0;
}

}  // namespace

int run_verify_logits(const char* dir, const std::vector<int>& widths,
                      std::vector<float>& output, std::string& error,
                      bool calibrate_t1, bool* small_m_enabled) {
    // The pending-incoming seam's ownership tables are owned HERE, BEFORE the Engine local:
    // declaration order makes them outlive the expert stream they are handed to (the
    // set_owned_elsewhere/set_pending_elsewhere contract).
    std::vector<uint8_t> pending_zeros, pending_mask;
    const char* pending_mode = std::getenv("STRATA_MIMO_TEST_PENDING_INCOMING");
    Engine e;
    const auto env_is = [](const char* key, const char* value) {
        const char* v = std::getenv(key); return v && std::strcmp(v, value) == 0;
    };
    e.win_cpu_test_rows = env_is("STRATA_MIMO_TEST_WINDOW_CPU_ROWS", "1") ||
                          env_is("STRATA_MIMO_TEST_WINDOW_CPU_ROWS", "2");
    e.win_cpu_test_gate = env_is("STRATA_MIMO_TEST_WINDOW_CPU_ROWS", "1");
    if (env_is("STRATA_MIMO_TEST_WINDOW_CPU_FAULT", "submit")) e.win_cpu_test_fault = 1;
    if (env_is("STRATA_MIMO_TEST_WINDOW_CPU_FAULT", "gpu")) e.win_cpu_test_fault = 2;
    if (env_is("STRATA_MIMO_TEST_WINDOW_CPU_FAULT", "gpu-hits")) e.win_cpu_test_fault = 3;
    e.win_test_peer_fault = 0;
    if (env_is("STRATA_MIMO_TEST_PEER_FRAME_FAULT", "1")) e.win_test_peer_fault = 1;
    if (env_is("STRATA_MIMO_TEST_PEER_FRAME_FAULT", "after-copy")) e.win_test_peer_fault = 2;
    if (env_is("STRATA_MIMO_TEST_PEER_FRAME_FAULT", "desync")) e.win_test_peer_fault = 3;
    const char* q8 = std::getenv("STRATA_MIMO_DENSE_Q8");
    const double test_vram = env_is("STRATA_MIMO_TEST_EXPERT_CACHE_ZERO", "1") ? 0. : -1.;
    // TEST-ONLY peer tier (deterministic round-robin subset, no profile): lets the fixture
    // harness exercise the handoff worker's peer arms (R3's device-row collector among
    // them) at real geometry on a second device, exactly like --peer-device does in
    // production. setup_peer runs AFTER the resident activation, mirroring run()'s order.
    const char* test_peer = std::getenv("STRATA_MIMO_TEST_PEER_DEVICE");
    if (test_peer != nullptr && test_peer[0] != '\0' && std::strcmp(test_peer, "0") != 0) {
        const int dev = std::atoi(test_peer);
        int ndev = 0;
        if (dev <= 0 || cudaGetDeviceCount(&ndev) != cudaSuccess || dev >= ndev) {
            error = "mimo: STRATA_MIMO_TEST_PEER_DEVICE must be a visible second device "
                    "(0 disables; the fixture tier needs device 1+)";
            return 1;
        }
        e.peer_device = dev;
    }
    if (!load_engine(dir, 128, 8, test_vram, 0, false, false, q8 && q8[0] == '1', e, error)) return 1;
    // Harness hygiene, CONFINED (2026-11-12 audit-fix): fixture engines share this process,
    // and a previous engine's teardown can leave sticky CUDA error state that this engine's
    // first unchecked-launch check would misreport as its own. But a blind clear would also
    // swallow REAL errors this run introduces - so drain ONLY the one documented stale code
    // (cudaErrorInvalidValue, observed 2026-11-12 with every opt-in off) and FAIL the
    // harness on anything else, instead of silently ignoring any CUDA error. Production
    // paths have no equivalent: their CK() gates stay untouched and fatal.
    {
        const cudaError_t stale = cudaGetLastError();
        if (stale == cudaErrorInvalidValue)
            std::fprintf(stderr, "strata-mimo parity: load_engine entry carried the documented "
                                 "stale cudaErrorInvalidValue/'invalid argument' (previous engine teardown; cleared)\n");
        else if (stale != cudaSuccess) {
            error = std::string("mimo parity harness: load_engine entry carried an UNEXPECTED "
                                "sticky cuda error: ") + cudaGetErrorString(stale);
            return 1;
        }
    }
    output.clear();
    try {
        if (!e.activate_resident(error)) return 1;
        // STRATA_MIMO_TEST_PEER_SPLIT=1 (TEST-ONLY, with STRATA_MIMO_TEST_PEER_DEVICE):
        // hand setup_peer the full deterministic pair list instead of the empty one. The
        // primary's warm-fill takes the first slots' worth; the peer's walk takes the rest -
        // a deterministic MIX of primary-miss and peer-owned rows for the transport tests.
        std::vector<std::pair<int32_t, int32_t>> test_ranked;
        if (e.peer_device >= 0 && env_is("STRATA_MIMO_TEST_PEER_SPLIT", "1"))
            for (int32_t ex = 0; ex < (int32_t) e.g.n_expert; ++ex)
                for (int32_t l = 0; l < (int32_t) e.g.n_layers; ++l)
                    if (e.g.is_moe(l)) test_ranked.push_back({l, ex});
        if (e.peer_device >= 0 && !e.setup_peer(dir, std::move(test_ranked), error)) return 1;
        // Harness hygiene, CONFINED (same rule as the load_engine entry drain above): the
        // peer fixture's setup leaves a sticky cudaGetLastError("invalid argument") on this
        // thread even with every opt-in off (enable_prefill clears and LOGS an earlier
        // instance of the same; observed 2026-11-12, results stay byte-exact and the
        // doorbell protocol completes; one engine per process in production never sees it).
        // Only that exact code is drained - with a loud line - and any OTHER sticky error
        // fails the harness instead of being silently ignored.
        {
            const cudaError_t stale = cudaGetLastError();
            if (stale == cudaErrorInvalidValue)
                std::fprintf(stderr, "strata-mimo parity: peer fixture setup left the documented "
                                     "stale cudaErrorInvalidValue (cleared)\n");
            else if (stale != cudaSuccess) {
                error = std::string("mimo parity harness: peer fixture setup left an UNEXPECTED "
                                    "sticky cuda error: ") + cudaGetErrorString(stale);
                return 1;
            }
        }
        if (strata::core::exl3_flag("STRATA_EXL3_CPU_TIER", nullptr, false) && !e.init_cpu_tier(dir, error)) return 1;
        if (!e.arm_window_cpu(error)) return 1;
        if (pending_mode != nullptr && pending_mode[0] != '\0' && std::strcmp(pending_mode, "0") != 0) {
            // TEST seam: simulate an in-flight peer refill's reservation.  The dimensions
            // must exist first (set_owned_elsewhere with an all-zero STATIC table - nothing
            // is statically owned, so the complement drops nothing), then the pending
            // table excludes its experts from admission exactly as a real refill would:
            // the getter union keeps them off the CPU planner, and run_multi_complement
            // (static table only) still computes them on the GPU scratch.
            pending_zeros.assign((size_t) e.g.n_layers * e.g.n_expert, 0);
            pending_mask.assign((size_t) e.g.n_layers * e.g.n_expert, 0);
            const bool half = std::strcmp(pending_mode, "half") == 0;
            const bool one = std::strcmp(pending_mode, "one") == 0;
            if (!half && !one && std::strcmp(pending_mode, "all") != 0) {
                error = "mimo: STRATA_MIMO_TEST_PENDING_INCOMING must be all, half or one";
                return 1;
            }
            for (int64_t l = 0; l < e.g.n_layers; ++l)
                if (e.g.is_moe(l))
                    for (int64_t ex = 0; ex < e.g.n_expert; ++ex)
                        if ((half && ex < e.g.n_expert / 2) || (one && ex == 0) ||
                            (!half && !one))
                            pending_mask[(size_t) l * e.g.n_expert + ex] = 1;
            e.experts->set_owned_elsewhere(pending_zeros.data(), e.g.n_layers, e.g.n_expert);
            e.experts->set_pending_elsewhere(pending_mask.data());
            e.win_cpu_test_pending = pending_mask.data();
            std::fprintf(stderr, "strata-mimo: test pending-incoming: %s of %lld expert(s)/layer "
                                 "reserved (CPU-excluded, GPU scratch serves them)\n",
                         half ? "first half" : one ? "first expert" : "all",
                         (long long) e.g.n_expert);
        }
        if (calibrate_t1) {
            int32_t prompt[] = {3, 20};
            const int32_t next = e.forward(prompt, 2);
            e.forward(&next, 1); // real decode, not all_rows T1: triggers old lazy handoff
        }
        for (int width : widths) {
            if (width < 1 || width > 8 || e.pos + width > 128) return 2;
            int32_t tokens[8];
            for (int i = 0; i < width; ++i) tokens[i] = (17 * (e.pos + i) + 3) % e.g.n_vocab;
            // TEST ONLY (STRATA_MIMO_TEST_FORWARD_MS=1, the replay microbench): the wall of
            // ONE teacher-forced forward (the all-rows pass + its verify readback), on this
            // thread. Load/init/preflight precede the loop, so the printed number excludes
            // them by construction; the engine's own SERVICE_TIME/PROF counters attribute
            // the layers' host spans in the same run's stderr.
            const bool fwd_ms = env_is("STRATA_MIMO_TEST_FORWARD_MS", "1");
            const auto f0 = fwd_ms ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            e.forward(tokens, width, true);
            const size_t offset = output.size(), count = (size_t) width * e.g.n_vocab;
            output.resize(offset + count);
            CK(cudaMemcpyAsync(output.data() + offset, e.diag_logits, count * 4, cudaMemcpyDeviceToHost, e.st));
            CK(cudaStreamSynchronize(e.st));
            if (fwd_ms)
                std::fprintf(stderr, "strata-mimo parity: forward width=%d pos=%lld wall=%.3f ms\n",
                             width, (long long) e.pos,
                             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - f0).count());
        }
        e.experts->small_m_debug(nullptr, nullptr, nullptr, nullptr, small_m_enabled);
        if (e.window_gpu_hits && env_is("STRATA_MIMO_TEST_WINDOW_GPU_HITS_EXPECT", "hits") &&
            e.win_gpu_hits_stats.hit_routes <= 0)
            throw std::runtime_error("window GPU hits test expected actual hit routes (the "
                                     "opt-in never engaged - check slots and the refresh gate)");
        if (e.window_gpu_hits && env_is("STRATA_MIMO_TEST_WINDOW_GPU_HITS_EXPECT", "zero") &&
            e.win_gpu_hits_stats.hit_routes != 0)
            throw std::runtime_error("window GPU hits test expected zero hit routes");
        if (e.window_gpu_hits && env_is("STRATA_MIMO_TEST_WINDOW_GPU_HITS_EXPECT", "simul") &&
            e.win_gpu_hits_stats.simultaneous < 1)
            throw std::runtime_error("window GPU hits test expected at least one window with "
                                     "inline hits AND a live CPU job AND a pending expert the "
                                     "GPU complement served (the combination never occurred)");
        if (e.window_router_split && env_is("STRATA_MIMO_TEST_WINDOW_ROUTER_SPLIT_EXPECT", "used") &&
            e.window_split_layers <= 0)
            throw std::runtime_error("window router split test expected actual split window "
                                     "routes (the opt-in never engaged - no T2..8 all-rows "
                                     "window ran)");
        if (e.peer_device_results && env_is("STRATA_MIMO_TEST_PEER_DEVICE_RESULTS_EXPECT", "used") &&
            e.peer_dev_frames <= 0)
            throw std::runtime_error("peer device results test expected actual collected frames "
                                     "(the opt-in never engaged - no peer window/decode row "
                                     "was served through the mapped frame)");
        if (e.peer_device_results && env_is("STRATA_MIMO_TEST_PEER_DEVICE_RESULTS_EXPECT", "mixed") &&
            (e.peer_dev_frames <= 0 || e.peer_dev_zero_windows <= 0))
            throw std::runtime_error("peer device results test expected BOTH engaged frames and "
                                     "zero-peer window publications with the flag on (frames=" +
                                     std::to_string((long long) e.peer_dev_frames) + ", zero-peer "
                                     "windows=" + std::to_string((long long) e.peer_dev_zero_windows) +
                                     ") - the no-remote path never ran");
        if (e.window_device_misses && env_is("STRATA_MIMO_TEST_WINDOW_DEVICE_MISSES_EXPECT", "used") &&
            e.wdm_calls <= 0)
            throw std::runtime_error("window device misses test expected actual device complements "
                                     "(the opt-in never engaged - no window carried a GPU row "
                                     "for the device arm)");
        if (env_is("STRATA_MIMO_TEST_WINDOW_CPU_EXPECT", "jobs")) {
            if (!e.win_cpu_stats.jobs || !e.win_cpu_stats.rows)
                throw std::runtime_error("window CPU test expected actual CPU jobs/rows");
            const auto s = e.cpu_moe->stats();
            if (e.expert_backing && e.expert_backing->layout() == strata::Exl3TrellisLayout::Band8 &&
                    (s.repack_fills || s.cache_bytes || s.native_reads))
                throw std::runtime_error("window CPU canonical test expected fills/cache/native_reads = 0");
        }
        if (env_is("STRATA_MIMO_TEST_WINDOW_CPU_EXPECT", "zero") && e.win_cpu_stats.jobs)
            throw std::runtime_error("window CPU test expected zero jobs");
    } catch (const std::exception& ex) { error = ex.what(); return 1; }
    return 0;
}

int run_greedy(const char* exl3_dir, const int32_t* prompt, int64_t n_prompt, int n_gen, int64_t ctx, int chunk,
               std::vector<int32_t>& out, std::string& err) {
    // The parity/driver wrappers keep their dense-mode parameters false, but load_engine would then
    // force-write STRATA_MIMO_DENSE_Q8=0 over the caller's environment - and the engine-parity flag
    // combos (STRATA_MIMO_QKV_SHARE / STRATA_MIMO_Q8_PREFILL_COLS need the q8 pack) must be able to
    // pick the mode.  Honor an explicit env here; the CLI's argv path is unchanged.
    const bool q8_env = [] {
        const char* v = std::getenv("STRATA_MIMO_DENSE_Q8");
        return v != nullptr && v[0] == '1';
    }();
    Engine E;
    if (!load_engine(exl3_dir, ctx, chunk, -1.0, 0, false, false, q8_env, E, err)) return 1;
    if (!E.activate_resident(err)) return 1;
    double prompt_ms = 0, decode_ms = 0;
    const int rc = greedy_generate(E, prompt, n_prompt, n_gen, out, prompt_ms, decode_ms);
    if (rc == 0)
        std::fprintf(stderr, "strata-mimo: parity run: %lld prompt in %.1f ms, %zu generated in %.1f ms\n",
                     (long long) n_prompt, prompt_ms, out.size(), decode_ms);
    return rc;
}

int run_greedy_dflash(const char* exl3_dir, const char* draft_dir, int proposals, const int32_t* prompt,
                      int64_t n_prompt, int n_gen, int64_t ctx, int chunk,
                      std::vector<int32_t>& out, std::string& error) {
    // run_greedy's rule: honor an explicit STRATA_MIMO_DENSE_Q8=1 instead of letting load_engine
    // force-write "0" over it (the flag-combo parity needs the q8 pack here too)
    const bool q8_env = [] {
        const char* v = std::getenv("STRATA_MIMO_DENSE_Q8");
        return v != nullptr && v[0] == '1';
    }();
    Engine e;
    e.dflash_tokens = proposals;
    if (proposals < 1 || proposals > 7) { error = "dflash proposals must be 1..7"; return 2; }
    if (!load_engine(exl3_dir, ctx, std::max(chunk, 8), -1., 0, false, false, q8_env, e, error, draft_dir)) return 1;
    if (!e.activate_resident(error)) return 1;
    double prompt_ms = 0, decode_ms = 0;
    return greedy_generate(e, prompt, n_prompt, n_gen, out, prompt_ms, decode_ms);
}

}  // namespace strata::mimo

int strata::mimo::run(int argc, char** argv) {
    // The EXL3 scoped-alias defaults (strata/core/exl3_flags.hpp: STRATA_EXL3_PF_GEMM ->
    // STRATA_PF_GEMM and friends, the shared name Gemm::f16 and the other readers take) never
    // resolved in a MiMo process: generate.cpp calls apply_exl3_defaults() only AFTER the family
    // switches return, so this path ran with the shared names unset and the scoped ones unheard
    // of.  Resolve them HERE, once, before any reader caches its value; the Qwen path keeps its
    // own single call below the switch and never comes through here, so neither family runs it
    // twice (a second call is a no-op anyway - it only ever sets what is not already in place).
    (void) strata::core::apply_exl3_defaults();
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-mimo: %s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        // strata's names where they mean the same thing (the family switch passes strata's command line on)
        if (a == "--exl3" || a == "--model" || a == "--model-dir" || a == "--native") o.model_dir = next();
        else if (a == "--max-new" || a == "--n") o.n_gen = std::atoi(next().c_str());
        else if (a == "--max-context" || a == "--ctx") o.ctx = std::atoll(next().c_str());
        else if (a == "--chunk" || a == "--prefill") { const std::string v = next(); o.chunk = v == "auto" ? 0 : std::atoi(v.c_str()); }
        else if (a == "--tokens") o.tokens = next();
        else if (a == "--serve") o.serve = true;
    else if (a == "--diag-embed") o.diag_embed = true;
    else if (a == "--diag-tf") o.diag_tf = true;
        else if (a == "--warmup") o.warmup = true;
        else if (a == "--dense-f16") o.dense_f16 = true;
        else if (a == "--dense-q8") o.dense_q8 = true;
        else if (a == "--cpu-tier") o.cpu_tier = true;
        else if (a == "--peer-device") o.peer_device = std::atoi(next().c_str());
        else if (a == "--seed") o.seed = std::strtoull(next().c_str(), nullptr, 10);   // pinned 1234
        else if (a == "--temperature") (void) next();   // greedy here (temperature 0); serve parses its own
        else if (a == "--expert-profile") o.expert_profile = next();
        else if (a == "--expert-profile-save") o.expert_profile_save = next();
        else if (a == "--expert-vram") o.expert_vram = std::atof(next().c_str());   // GiB of expert slots
        else if (a == "--expert-ram-cache") {
            // 0 (and "auto") is the RECOMMENDED host arrangement: the expert reader's buffered IO
            // already rides the page cache, and a big explicit tier double-caches and HURTS
            // (measured on the real model: decode 2.6 tok/s with the page cache vs 1.4 with a
            // 64 GiB explicit tier).  MiB only, for experiments.
            const std::string v = next();
            o.expert_ram = v == "auto" ? 0 : std::atoll(v.c_str());
        }
        else if (a == "--gpu") (void) next();   // applied by strata before the switch (CUDA/HIP_VISIBLE_DEVICES)
        else if (a == "--greedy") {}            // --tokens decodes greedily here anyway
        else if (a == "--pool-workers" || a == "--cpu-threads") (void) next();   // the CPU pool is phase C
        else if (a == "--no-graph" || a == "--no-token-graph" || a == "--no-warmup") {}   // no graphs in v1
        else if (a == "--mtp") {
            const std::string v = next();
            o.mtp = std::atoi(v.c_str());
            if (o.mtp < 0 || o.mtp > 3) {
                std::fprintf(stderr, "strata-mimo: --mtp takes 0 .. 3 heads\n");
                return 2;
            }
        }
        else if (a == "--dflash" || a == "--dflash-dir") {
            o.dflash_dir = next();
        }
        else if (a == "--dflash-tokens") o.dflash_tokens = std::atoi(next().c_str());
        else if (o.serve && a.rfind("--", 0) == 0) {
            // serve passes the config's args plus a few of its own: one this engine does not have is said and
            // skipped (with its value), not fatal
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "strata-mimo: ignoring %s%s%s (not an option of this engine)\n", a.c_str(),
                         val ? " " : "", val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            std::fprintf(stderr,
                         "usage: strata-mimo --exl3 DIR [--max-context 16384] [--chunk 512] [--expert-vram GIB]\n"
                         "       (--serve | --tokens IDS [--n 64])\n");
            return 2;
        }
    }
    if (o.model_dir.empty()) { std::fprintf(stderr, "strata-mimo: --exl3 (the model dir) is required\n"); return 2; }
    if (o.dflash_tokens < 1 || o.dflash_tokens > 7 || (o.mtp && !o.dflash_dir.empty())) {
        std::fprintf(stderr, "strata-mimo: DFlash takes 1..7 proposals and cannot combine with MTP\n"); return 2;
    }
    if (!o.dflash_dir.empty()) o.chunk = std::max(o.chunk, 8);
    if (o.chunk == 0) o.chunk = 512;
    if (o.chunk < 1 || o.chunk > 4096 || o.ctx < 256) {
        std::fprintf(stderr, "strata-mimo: bad --chunk / --max-context\n");
        return 2;
    }
#if !defined(_WIN32)
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);   // every kernel loaded with the context, not in the first prompt
#endif

    if (o.mtp > 0) {
        o.chunk = std::max(o.chunk, 8);            // the drafter's cell staging uses the window buffers
        setenv("STRATA_MIMO_LM_F16", "1", 1);      // MimoMtp argmaxes through the pack's F16 lm_head
    }
    Engine E;
    std::string error;
    E.peer_device = o.peer_device;
    E.dflash_tokens = o.dflash_tokens;
    // the CPU tier's refusal machinery is read by the expert stream at OPEN time (inside
    // load_engine), so the env travels before it; the tier itself registers after the load
    if (o.cpu_tier) setenv("STRATA_EXL3_CPU_TIER", "1", 1);
    if (o.diag_tf && strata::core::exl3_flag("STRATA_MIMO_WINDOW_CPU", nullptr, false)) {
        if (!E.preflight_window_cpu(error) || E.window_cpu) {
            std::fprintf(stderr, "strata-mimo: window CPU requires mapped worker service; --diag-tf unsupported: %s\n", error.c_str());
            return 1;
        }
    }
    if (!load_engine(o.model_dir, o.ctx, o.chunk, o.expert_vram, o.expert_ram, o.warmup, o.dense_f16,
                     o.dense_q8, E, error, o.dflash_dir)) {
        std::fprintf(stderr, "strata-mimo: %s\n", error.c_str());
        return 1;
    }
    if (!E.resident_backing && o.cpu_tier && !E.init_cpu_tier(o.model_dir, error))
        std::fprintf(stderr, "strata-mimo: cpu tier: %s (continuing on the GPU alone)\n", error.c_str());
    if (!E.resident_backing && !E.arm_window_cpu(error)) {
        std::fprintf(stderr, "strata-mimo: %s\n", error.c_str());
        return 1;
    }
    // the route profile (optional) then the peer tier: with the peer on, setup_peer does the
    // hot-share interleave, the primary's warm-fill and the tier itself; without it this is the
    // established warm-fill of the primary arena alone
    std::vector<std::pair<int32_t, int32_t>> ranked;
    if (!o.expert_profile.empty() && !E.parse_profile(o.expert_profile, ranked, error)) {
        std::fprintf(stderr, "strata-mimo: expert profile: %s (continuing cold)\n", error.c_str());
        error.clear();
    }
    if (o.peer_device >= 0) {
        if (!E.setup_peer(o.model_dir, std::move(ranked), error)) {
            std::fprintf(stderr, "strata-mimo: peer tier: %s\n", error.c_str());
            return 1;
        }
    } else if (!ranked.empty() && !E.fill_primary(ranked, error)) {
        std::fprintf(stderr, "strata-mimo: expert profile: %s (continuing cold)\n", error.c_str());
    }
    if (o.mtp > 0 && !E.init_mtp(o.mtp, error)) {
        std::fprintf(stderr, "strata-mimo: mtp init: %s\n", error.c_str());
        return 1;
    }
    if (E.resident_backing) {
        if (!E.activate_resident(error) || (o.cpu_tier && !E.init_cpu_tier(o.model_dir,error)) ||
                !E.arm_window_cpu(error)) {
            std::fprintf(stderr,"strata-mimo: %s\n",error.c_str()); return 1;
        }
    }
    if (!E.resident_backing && strata::core::exl3_flag("STRATA_MIMO_HOST_WARMUP", nullptr, false)) {
        // All load-time allocations and peer placement are complete. Warm the host
        // working set LAST, excluding peer-only weights and dense/drafter bytes.
        // No routes are observed and no cache state is changed by this opt-in.
        const auto start = Clock::now();
        try {
            strata::Exl3Model model(o.model_dir, E.expert_backing);
            std::vector<std::string> prefixes;
            size_t skipped = 0;
            for (int l = 0; l < E.g.n_layers; ++l) if (E.g.is_moe(l))
                for (int e = 0; e < E.g.n_expert; ++e) {
                    if (E.peer && E.peer->valid() && E.peer->has(l, e)) { ++skipped; continue; }
                    for (const char* proj : {"gate", "up", "down"})
                        prefixes.push_back(strata::mimo_expert_tensor(l, e, proj));
                }
            const auto ranges = strata::exl3_host_warmup_plan(model, prefixes);
            const auto bytes = strata::exl3_host_warmup_read(ranges);
            std::fprintf(stderr, "strata-mimo: host warmup: %.2f GiB in %.1f s, %zu experts, "
                                 "%zu peer residents skipped, %zu ranges; no placement changes\n",
                         bytes / 1073741824.0, ms_since(start) / 1000.0,
                         prefixes.size() / 3, skipped, ranges.size());
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "strata-mimo: %s\n", ex.what());
            return 1;
        }
    }
    if (o.serve) return serve(E);
    if (o.diag_embed) return diag_embed(E);
    if (o.diag_tf) {
        const std::vector<int32_t> prompt = parse_ids(o.tokens.c_str());
        if (prompt.empty()) { std::fprintf(stderr, "strata-mimo: --diag-tf needs --tokens IDS\n"); return 2; }
        return diag_tf(E, prompt);
    }

    // ---- the CLI: --tokens IDS [--n N], greedy
    const std::vector<int32_t> prompt = parse_ids(o.tokens.c_str());
    if (prompt.empty()) { std::fprintf(stderr, "strata-mimo: --tokens, --serve is required\n"); return 2; }
    if ((int64_t) prompt.size() + o.n_gen > E.cap) {
        std::fprintf(stderr, "strata-mimo: prompt (%zu) + --n (%d) exceeds the context (%lld)\n", prompt.size(),
                     o.n_gen, (long long) E.cap);
        return 2;
    }
    bool bad = false;
    for (int32_t t : prompt) bad |= t < 0 || t >= (int32_t) E.g.n_vocab;
    if (bad) { std::fprintf(stderr, "strata-mimo: a token id is outside the vocabulary\n"); return 2; }
    double prompt_ms = 0, decode_ms = 0;
    std::vector<int32_t> gen;
    const int rc = greedy_generate(E, prompt.data(), (int64_t) prompt.size(), o.n_gen, gen, prompt_ms, decode_ms);
    if (rc != 0) return rc;
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    std::fflush(stdout);
    std::fprintf(stderr, "strata-mimo: prompt %zu tokens in %.1f ms (%.1f tok/s), %zu generated in %.1f ms "
                         "(%.1f tok/s), greedy, seed %llu\n",
                 prompt.size(), prompt_ms, prompt.size() / std::max(1e-9, prompt_ms / 1000.0), gen.size(),
                 decode_ms, gen.size() / std::max(1e-9, decode_ms / 1000.0), o.seed);
    if (!o.expert_profile_save.empty() && !E.save_profile(o.expert_profile_save))
        std::fprintf(stderr, "strata-mimo: cannot write the expert profile\n");
    if (E.mtp != nullptr) {
        std::fprintf(stderr, "strata-mimo: MTP: %lld round(s), %lld of %lld drafts accepted (%.0f%%), "
                             "%.2f tokens/round; accepted-draft histogram (0..%d):",
                     (long long) E.rounds, (long long) E.accepted_total, (long long) E.offered_total * E.mtp_heads,
                     100.0 * E.accepted_total / std::max<int64_t>(1, E.offered_total * E.mtp_heads),
                     (double) gen.size() / std::max<int64_t>(1, E.rounds), E.mtp_heads - 1);
        for (int64_t h : E.accept_hist) std::fprintf(stderr, " %lld", (long long) h);
        std::fprintf(stderr, "\n");
    }
    return 0;
}
