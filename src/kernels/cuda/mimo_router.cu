// src/kernels/cuda/mimo_router.cu - MiMo v2's router: the F16 gemv, the sigmoid scoring, the noaux_tc biased
// top-k and the unbiased renormalised combine, fused (notes/MIMO-V2-PORT.md §1; the host contract is
// mimo_router.hpp).
//
// The warp-level selection is native_router.cu's proven shape (one warp holds the experts as up-to-8 values
// per lane, k argmax rounds with the XOR butterfly, equal scores resolved to the lower expert id); what is
// new is everything around it: the dot products come from F16 weights against an F16 row with F32
// accumulation (native_router receives ready logits), the score is a sigmoid (not a softmax), the SELECTION
// score carries the e_score_correction_bias while the WEIGHT uses the unbiased sigmoid value, and the
// renormalisation is s_i / (sum over the selected + 1e-20) - llama.cpp's 2^-14 softmax clamp does not apply
// here and would be wrong.
//
// One block of 256 threads (8 warps) per token row.  The gemv phase spreads the experts over the 8 warps
// (uint4 loads of 8 halves, butterfly reduction) and parks the sigmoid scores in shared memory; warp 0 then
// runs the k selection rounds and writes the k ids and weights.  n_expert and k are RUNTIME arguments
// (the target model's 256/8 and the engine-parity fixture's 4/2 are the same code path; lanes past
// n_expert carry -FLT_MAX, which never wins a round), so no shape is compiled in.
#include "strata/kernels/mimo_router.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int kBlockThreads = 256;   // 8 warps
constexpr int kWarps = kBlockThreads / 32;
constexpr int kMaxExperts = 256;     // the host rejects more; values[8] covers 8 x 32 lanes
constexpr int kMaxK = 32;            // the lane-per-rank capture below

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
    return value;
}

/// The MiMo router over one token row: n_expert experts, top-k of the biased score, weights the unbiased
/// sigmoid scores renormalised over the selected k.  n_expert <= 256 (whole shared arrays), k <= 32.
__launch_bounds__(kBlockThreads, 1)
__global__ void mimo_route(const uint16_t* __restrict__ router, const uint16_t* __restrict__ x,
                           const float* __restrict__ e_score, int32_t* __restrict__ ids,
                           float* __restrict__ weights, int n_embd, int n_expert, int k) {
    __shared__ float s_shared[kMaxExperts];   // the UNBIASED sigmoid scores (what the weights use)
    __shared__ float b_shared[kMaxExperts];   // e_score_correction_bias (what the selection uses)
    const size_t row = blockIdx.x;
    x += row * (size_t) n_embd;
    ids += row * (size_t) k;
    weights += row * (size_t) k;
    const int warp = (int) threadIdx.x >> 5, lane = (int) threadIdx.x & 31;
    for (int e = (int) threadIdx.x; e < n_expert; e += kBlockThreads) b_shared[e] = e_score[e];
    // gemv phase: expert e lives on warp e % 8; each lane owns the uint4 chunks c = lane, lane + 32, ...
    const int n4 = n_embd >> 3;   // uint4 chunks per row (n_embd % 8 == 0 is checked on the host)
    const uint4* xq = reinterpret_cast<const uint4*>(x);
    for (int e = warp; e < n_expert; e += kWarps) {
        const uint4* wq = reinterpret_cast<const uint4*>(router + (size_t) e * n_embd);
        float acc = 0.0f;
        for (int c = lane; c < n4; c += 32) {
            const uint4 wv = wq[c], xv = xq[c];
            const __half2* w2 = reinterpret_cast<const __half2*>(&wv);   // 4 half2 per uint4
            const __half2* x2 = reinterpret_cast<const __half2*>(&xv);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 wf = __half22float2(w2[j]);
                const float2 xf = __half22float2(x2[j]);
                acc = fmaf(wf.x, xf.x, acc);
                acc = fmaf(wf.y, xf.y, acc);
            }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, mask, 32);
        if (lane == 0) {
            const float s = 1.0f / (1.0f + expf(-acc));
            s_shared[e] = __isnanf(s) ? 0.0f : s;   // a NaN dot reads as score 0, never selected on its own
        }
    }
    __syncthreads();
    if (warp != 0) return;
    // selection phase (warp 0): lane L holds the experts L, L + 32, ... as their biased scores
    const int v = (n_expert + 31) >> 5;
    float values[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int e = lane + i * 32;
        float c = -FLT_MAX;   // a lane past n_expert (or a NaN biased score) never wins a round
        if (i < v && e < n_expert) {
            c = s_shared[e] + b_shared[e];
            if (__isnanf(c)) c = -FLT_MAX;
        }
        values[i] = c;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < k; ++rank) {
        float best = values[0];
        int expert = lane < n_expert ? lane : 0x7fffffff;   // a lane past n_expert never wins a tie
#pragma unroll
        for (int i = 1; i < 8; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }   // equal: keep the lower id
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        best = s_shared[expert];   // the winner was found by c_j; its WEIGHT is the unbiased s_j
        if (expert < n_expert && (expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            // Deliberately accumulate by WINNING EXPERT lane, not output rank (selection order per lane):
            // the same layout native_router's combine downstream adds in.
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = warp_sum(selected_sum);
    const float inverse = 1.0f / (selected_sum + 1e-20f);
    if (lane < k) weights[lane] = selected * inverse;
}

/// 4-byte span with a no-overflow address range (native_router's `valid`).
bool span_ok(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}

bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}

/// The split arm's scores kernel: block (e, r) is ONE warp computing expert e's dot with row r
/// using EXACTLY the one-block kernel's per-warp arithmetic (lane L owns the uint4 chunks
/// L, L+32, ..., the same fmaf order, the same butterfly), so every sigmoid score is
/// bit-identical and the selection below can never disagree with a `mimo_router_route` run on
/// the same inputs.  The single-row decode launch passes a 1-D grid, where blockIdx.y == 0
/// reduces this to the original one-row kernel; `mimo_router_route_window` launches the
/// (n_expert, rows) grid so a T-row window spreads over n_expert * rows warps instead of the
/// one-block kernel's T blocks (T = 2..8 re-reads the whole router plane per row).
__global__ void mimo_route_scores(const uint16_t* __restrict__ router, const uint16_t* __restrict__ x,
                                  float* __restrict__ scores, int n_embd, int n_expert) {
    const int e = (int) blockIdx.x, lane = (int) threadIdx.x & 31;
    x += (size_t) blockIdx.y * (size_t) n_embd;
    scores += (size_t) blockIdx.y * (size_t) n_expert;
    const int n4 = n_embd >> 3;
    const uint4* xq = reinterpret_cast<const uint4*>(x);
    const uint4* wq = reinterpret_cast<const uint4*>(router + (size_t) e * n_embd);
    float acc = 0.0f;
    for (int c = lane; c < n4; c += 32) {
        const uint4 wv = wq[c], xv = xq[c];
        const __half2* w2 = reinterpret_cast<const __half2*>(&wv);
        const __half2* x2 = reinterpret_cast<const __half2*>(&xv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 wf = __half22float2(w2[j]);
            const float2 xf = __half22float2(x2[j]);
            acc = fmaf(wf.x, xf.x, acc);
            acc = fmaf(wf.y, xf.y, acc);
        }
    }
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, mask, 32);
    if (lane == 0) {
        const float s = 1.0f / (1.0f + expf(-acc));
        scores[e] = __isnanf(s) ? 0.0f : s;   // the same NaN rule as the one-block kernel
    }
}

/// The split arm's selection kernel: the one-block kernel's warp-0 phase verbatim, on the
/// scores the first kernel parked in `scores`.  Block r selects row r's top-k: the decode
/// launch's single block is the rows == 1 case.
__global__ void mimo_route_pick(const float* __restrict__ scores, const float* __restrict__ e_score,
                                int32_t* __restrict__ ids, float* __restrict__ weights, int n_expert, int k) {
    scores += (size_t) blockIdx.x * (size_t) n_expert;
    ids += (size_t) blockIdx.x * (size_t) k;
    weights += (size_t) blockIdx.x * (size_t) k;
    const int lane = (int) threadIdx.x & 31;
    const int v = (n_expert + 31) >> 5;
    float values[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int e = lane + i * 32;
        float c = -FLT_MAX;
        if (i < v && e < n_expert) {
            c = scores[e] + e_score[e];
            if (__isnanf(c)) c = -FLT_MAX;
        }
        values[i] = c;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < k; ++rank) {
        float best = values[0];
        int expert = lane < n_expert ? lane : 0x7fffffff;
#pragma unroll
        for (int i = 1; i < 8; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        best = scores[expert];
        if (expert < n_expert && (expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = warp_sum(selected_sum);
    const float inverse = 1.0f / (selected_sum + 1e-20f);
    if (lane < k) weights[lane] = selected * inverse;
}

}  // namespace {

bool mimo_router_supported(int n_expert, int k) {
    return n_expert >= 1 && n_expert <= kMaxExperts && k >= 1 && k <= kMaxK && k <= n_expert;
}

void mimo_router_route(const void* router, const void* x, const void* e_score, int32_t* ids, float* weights,
                       int rows, int n_embd, int n_expert, int k, void* stream) {
    if (!mimo_router_supported(n_expert, k))
        throw std::invalid_argument("mimo_router: 1..256 experts and a top-k up to 32 and to n_expert");
    if (rows < 1 || n_embd <= 0 || n_embd % 8 != 0)
        throw std::invalid_argument("mimo_router: rows >= 1 and n_embd a positive multiple of 8");
    const size_t router_bytes = (size_t) n_expert * (size_t) n_embd * 2;
    const size_t x_bytes = (size_t) rows * (size_t) n_embd * 2;
    const size_t score_bytes = (size_t) n_expert * 4;
    const size_t out_bytes = (size_t) rows * (size_t) k * 4;
    if (!span_ok(router, router_bytes) || !span_ok(x, x_bytes) || !span_ok(e_score, score_bytes) ||
        !span_ok(ids, out_bytes) || !span_ok(weights, out_bytes))
        throw std::invalid_argument("mimo_router: aligned [n_expert,n_embd]/[rows,n_embd] F16 and [rows,k] out buffers");
    const auto r16 = reinterpret_cast<uintptr_t>(router), x16 = reinterpret_cast<uintptr_t>(x);
    if (r16 % 16 != 0 || x16 % 16 != 0)
        throw std::invalid_argument("mimo_router: the F16 buffers must be 16-byte aligned (uint4 loads)");
    if (overlap(router, router_bytes, ids, out_bytes) || overlap(router, router_bytes, weights, out_bytes) ||
        overlap(x, x_bytes, ids, out_bytes) || overlap(x, x_bytes, weights, out_bytes) ||
        overlap(e_score, score_bytes, ids, out_bytes) || overlap(e_score, score_bytes, weights, out_bytes) ||
        overlap(ids, out_bytes, weights, out_bytes))
        throw std::invalid_argument("mimo_router: outputs must be disjoint from the inputs and each other");
    mimo_route<<<(unsigned) rows, kBlockThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint16_t*>(router), static_cast<const uint16_t*>(x),
        static_cast<const float*>(e_score), ids, weights, n_embd, n_expert, k);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void mimo_router_route_decode(const void* router, const void* x, const void* e_score, float* scores,
                              int32_t* ids, float* weights, int n_embd, int n_expert, int k, void* stream) {
    if (!mimo_router_supported(n_expert, k))
        throw std::invalid_argument("mimo_router: 1..256 experts and a top-k up to 32 and to n_expert");
    if (n_embd <= 0 || n_embd % 8 != 0)
        throw std::invalid_argument("mimo_router: n_embd a positive multiple of 8");
    const size_t router_bytes = (size_t) n_expert * (size_t) n_embd * 2;
    const size_t x_bytes = (size_t) n_embd * 2;
    const size_t score_bytes = (size_t) n_expert * 4;
    const size_t out_bytes = (size_t) k * 4;
    if (!span_ok(router, router_bytes) || !span_ok(x, x_bytes) || !span_ok(e_score, score_bytes) ||
        !span_ok(scores, score_bytes) || !span_ok(ids, out_bytes) || !span_ok(weights, out_bytes))
        throw std::invalid_argument("mimo_router: aligned [n_expert,n_embd]/[n_embd] F16 and [k] out buffers");
    const auto r16 = reinterpret_cast<uintptr_t>(router), x16 = reinterpret_cast<uintptr_t>(x);
    if (r16 % 16 != 0 || x16 % 16 != 0)
        throw std::invalid_argument("mimo_router: the F16 buffers must be 16-byte aligned (uint4 loads)");
    if (overlap(router, router_bytes, ids, out_bytes) || overlap(router, router_bytes, weights, out_bytes) ||
        overlap(x, x_bytes, ids, out_bytes) || overlap(x, x_bytes, weights, out_bytes) ||
        overlap(e_score, score_bytes, ids, out_bytes) || overlap(e_score, score_bytes, weights, out_bytes) ||
        overlap(scores, score_bytes, ids, out_bytes) || overlap(scores, score_bytes, weights, out_bytes) ||
        overlap(ids, out_bytes, weights, out_bytes))
        throw std::invalid_argument("mimo_router: outputs must be disjoint from the inputs and each other");
    const auto st = static_cast<cudaStream_t>(stream);
    mimo_route_scores<<<(unsigned) n_expert, 32, 0, st>>>(
        static_cast<const uint16_t*>(router), static_cast<const uint16_t*>(x), scores, n_embd, n_expert);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    mimo_route_pick<<<1, 32, 0, st>>>(scores, static_cast<const float*>(e_score), ids, weights, n_expert, k);
    error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void mimo_router_route_window(const void* router, const void* x, const void* e_score, float* scores,
                              int32_t* ids, float* weights, int rows, int n_embd, int n_expert, int k,
                              void* stream) {
    if (!mimo_router_supported(n_expert, k))
        throw std::invalid_argument("mimo_router: 1..256 experts and a top-k up to 32 and to n_expert");
    if (rows < 1 || rows > 8)
        throw std::invalid_argument("mimo_router: the window arm routes 1..8 rows (the verify-window envelope)");
    if (n_embd <= 0 || n_embd % 8 != 0)
        throw std::invalid_argument("mimo_router: n_embd a positive multiple of 8");
    const size_t router_bytes = (size_t) n_expert * (size_t) n_embd * 2;
    const size_t x_bytes = (size_t) rows * (size_t) n_embd * 2;
    const size_t bias_bytes = (size_t) n_expert * 4;
    const size_t score_bytes = (size_t) rows * (size_t) n_expert * 4;
    const size_t out_bytes = (size_t) rows * (size_t) k * 4;
    if (!span_ok(router, router_bytes) || !span_ok(x, x_bytes) || !span_ok(e_score, bias_bytes) ||
        !span_ok(scores, score_bytes) || !span_ok(ids, out_bytes) || !span_ok(weights, out_bytes))
        throw std::invalid_argument("mimo_router: aligned [n_expert,n_embd]/[rows,n_embd] F16 and "
                                    "[rows,n_expert]/[rows,k] out buffers");
    const auto r16 = reinterpret_cast<uintptr_t>(router), x16 = reinterpret_cast<uintptr_t>(x);
    if (r16 % 16 != 0 || x16 % 16 != 0)
        throw std::invalid_argument("mimo_router: the F16 buffers must be 16-byte aligned (uint4 loads)");
    // STRICTER than the decode arm (whose scores scratch cannot alias its [k] outputs in the
    // engine, but whose validation does not say so): scores is a kernel-1 OUTPUT racing the
    // router/x reads, so it counts as an output against the inputs as well.
    if (overlap(router, router_bytes, ids, out_bytes) || overlap(router, router_bytes, weights, out_bytes) ||
        overlap(x, x_bytes, ids, out_bytes) || overlap(x, x_bytes, weights, out_bytes) ||
        overlap(e_score, bias_bytes, ids, out_bytes) || overlap(e_score, bias_bytes, weights, out_bytes) ||
        overlap(router, router_bytes, scores, score_bytes) || overlap(x, x_bytes, scores, score_bytes) ||
        overlap(e_score, bias_bytes, scores, score_bytes) ||
        overlap(scores, score_bytes, ids, out_bytes) || overlap(scores, score_bytes, weights, out_bytes) ||
        overlap(ids, out_bytes, weights, out_bytes))
        throw std::invalid_argument("mimo_router: outputs must be disjoint from the inputs and each other");
    const auto st = static_cast<cudaStream_t>(stream);
    mimo_route_scores<<<dim3((unsigned) n_expert, (unsigned) rows), 32, 0, st>>>(
        static_cast<const uint16_t*>(router), static_cast<const uint16_t*>(x), scores, n_embd, n_expert);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    mimo_route_pick<<<(unsigned) rows, 32, 0, st>>>(scores, static_cast<const float*>(e_score), ids,
                                                    weights, n_expert, k);
    error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

}  // namespace strata::kernels
