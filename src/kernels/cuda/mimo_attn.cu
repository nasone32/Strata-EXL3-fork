// src/kernels/cuda/mimo_attn.cu - see include/strata/kernels/mimo_attn.hpp.
//
// MiMo v2 attention (notes/MIMO-V2-PORT.md): partial NEOX rope over 192-wide qk rows and GQA
// attention with a sliding window and per-q-head sink logits, K 192 / V 128, flat (unpaged)
// caches.  Correctness first: plain SIMT kernels, no tensor cores, fp32 softmax/accumulation.
//
//   rope    one thread per (row, head, rotated pair).  The n_rot/2 inverse frequencies are
//           computed once per block in fp64 (::pow) and kept in shared; the angle is
//           pos * inv_freq in fp64 (an fp32 angle is already wrong by ~0.1 rad at pos 1e6),
//           reduced mod 2*pi in fp64, then fp32 sincos and an fp32 rotation, F16 store.
//   attn    one 128-thread block per (query row, kv head), templated on the GQA group G:
//           the block stages its G query heads in shared fp32, streams key positions in
//           64-wide tiles, scores with one warp per cell (lane owns 8 of the first 256 qk
//           dims), and merges each tile into running (max, denominator, accumulator) - the
//           online softmax of qsa_decode_attn with the same tiling.  Thread t owns V lane t
//           of all G heads' accumulators.  The sink joins at writeout: m = max(m_run, sink),
//           denominator += e^(sink - m) (sink RAW, never sm_scale'd), zero-value virtual key.
//   prefill is the same kernel with positions/seq_lens derived from pos0 (PREFILL=true):
//           row r sits at pos0 + r and sees pos0 + rows keys - causal inside the chunk plus
//           the whole history, the window clamped at 0 exactly as in decode.
//
// Optimization notes for later (measured before adoption, per AGENTS.md): tile several query
// ROWS per block so prefill re-reads each K row once per tile instead of once per row; stage
// K tiles through LDS with half2 dots; SWA ring cache so decode loops 128 keys, not [0, p].
// positions/seq_lens are DEVICE pointers, like qsa's steps and native_rope's positions.
#include "strata/kernels/mimo_attn.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int kMaxDimQk = 256;   // widest qk row supported (MiMo 192, dflash 128); lane dots 8
constexpr int kThreads = 128;    // also the widest V row: one thread per V lane
constexpr int kWarps = kThreads / 32;
constexpr int kChunk = 64;       // key positions per tile

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__global__ void __launch_bounds__(kThreads) mimo_rope_kernel(uint16_t* __restrict__ qk,
                                                             const int64_t* __restrict__ positions,
                                                             int rows, int n_heads, int head_dim_qk,
                                                             int n_rot, double theta) {
    __shared__ double freqs[32];   // n_rot/2 <= 32 (the header rejects wider)
    const int half = n_rot / 2;
    if (threadIdx.x < half) freqs[threadIdx.x] = ::pow(theta, -(2.0 * threadIdx.x) / (double) n_rot);
    __syncthreads();
    const long long g = (long long) blockIdx.x * kThreads + threadIdx.x;
    const int i = (int) (g % half);            // rotated pair index within the head
    const long long rh = g / half;             // row * n_heads + head
    if (rh >= (long long) rows * n_heads) return;
    const int64_t pos = positions[rh / n_heads];
    // fp64 angle; pos is exact in double for every position a context can hold
    const double angle = (double) pos * freqs[i];
    const double reduced = ::fmod(angle, 6.283185307179586476925287);
    float s, c;
    sincosf((float) reduced, &s, &c);
    uint16_t* row = qk + (size_t) rh * head_dim_qk;
    const float a = f32_from_f16(row[i]), b = f32_from_f16(row[i + half]);
    row[i] = f16_from_f32(a * c - b * s);
    row[i + half] = f16_from_f32(a * s + b * c);
}

// One block per (query row, kv head).  PREFILL derives the position (pos0 + row) and the key
// count (pos0 + rows) instead of reading positions/seq_lens.
template <int G, bool PREFILL, bool RING = false>
__global__ void __launch_bounds__(kThreads) mimo_attn_kernel(const uint16_t* __restrict__ q,
                                                             const uint16_t* __restrict__ k_cache,
                                                             const uint16_t* __restrict__ v_cache,
                                                             float* __restrict__ out,
                                                             const int64_t* __restrict__ positions,
                                                             const int* __restrict__ seq_lens,
                                                             int64_t pos0, int n_q_head, int n_kv_head,
                                                             int hd_qk, int hd_v, int window,
                                                             const float* __restrict__ sinks,
                                                             float sm_scale, int cache_ring) {
    const int row = blockIdx.y, kvh = blockIdx.x;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int64_t p = PREFILL ? pos0 + row : positions[row];
    const int n = PREFILL ? (int) (pos0 + gridDim.y) : seq_lens[row];
    // valid keys: kpos in [lo, hi]; hi caps the window at the row's own history length
    // (verify rows pass per-row lengths; keys at or past seq_len are never read, so garbage
    // cache rows beyond are invisible), lo clamps the sliding window at 0
    const int hi = (int) (p < (int64_t) n - 1 ? p : (int64_t) n - 1);
    const int lo = window > 0 ? (int) (p - (window - 1) > 0 ? p - (window - 1) : 0) : 0;

    __shared__ __align__(16) float sq[G][kMaxDimQk];
    __shared__ float sp[G][kChunk];
    __shared__ float sm[G], sl[G], sr[G];   // running max, denominator, accumulator rescale
    __shared__ int skpos[kChunk];           // key position of each cell, -1 past the range

    const uint16_t* qrow = q + ((size_t) row * n_q_head + (size_t) kvh * G) * hd_qk;
    for (int i = t; i < G * kMaxDimQk; i += kThreads) {
        const int d = i & (kMaxDimQk - 1), h = i >> 8;
        sq[h][d] = d < hd_qk ? f32_from_f16(qrow[(size_t) h * hd_qk + d]) : 0.0f;   // zero-pad
    }
    if (t < G) { sm[t] = -FLT_MAX; sl[t] = 0.0f; }
    __syncthreads();

    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;

    for (int c0 = lo; c0 <= hi; c0 += kChunk) {
        const int n_here = min(kChunk, hi - c0 + 1);
        if (t < kChunk) {
            int at = c0 + t;
            if constexpr (RING) at %= cache_ring;
            skpos[t] = t < n_here ? at : -1;
        }
        __syncthreads();
        // scores: each warp takes cells warp, warp+4, ...; lane holds 8 of the first 256 dims
        for (int c = warp; c < kChunk; c += kWarps) {
            if (c >= n_here) {
                if (lane < G) sp[lane][c] = -FLT_MAX;
                continue;
            }
            float k8[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            if (lane * 8 < hd_qk) {
                uint16_t raw[8];
                *reinterpret_cast<uint4*>(raw) = *reinterpret_cast<const uint4*>(
                    k_cache + ((size_t) skpos[c] * n_kv_head + kvh) * hd_qk + lane * 8);
#pragma unroll
                for (int j = 0; j < 8; ++j) k8[j] = f32_from_f16(raw[j]);
            }
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
                float s = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                          k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
                s = warp_sum(s);
                if (lane == 0) sp[h][c] = s * sm_scale;
            }
        }
        __syncthreads();
        // tile max, then merge into the running (m, l); sr = this tile's accumulator rescale
        for (int h = warp; h < G; h += kWarps) {
            const float a = sp[h][lane], b = sp[h][lane + 32];
            const float mc = warp_max(fmaxf(a, b));
            const float m_old = sm[h], m_new = fmaxf(m_old, mc);
            const float ea = lane < n_here ? __expf(a - m_new) : 0.0f;
            const float eb = lane + 32 < n_here ? __expf(b - m_new) : 0.0f;
            sp[h][lane] = ea;
            sp[h][lane + 32] = eb;
            const float l_add = warp_sum(ea + eb);
            if (lane == 0) {
                const float rescale = m_old == -FLT_MAX ? 0.0f : __expf(m_old - m_new);
                sl[h] = sl[h] * rescale + l_add;
                sr[h] = rescale;
                sm[h] = m_new;
            }
        }
        __syncthreads();
        // values: thread t owns V lane t of every head's accumulator
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] *= sr[h];
        // V rows are hd_v wide: lanes t >= hd_v would read past the row's tail - out of the
        // buffer entirely at the last (kv head, row).  Their accumulator lanes are discarded
        // at writeout, so those threads skip the loads wholesale; no barrier sits inside, so
        // block cooperation is unchanged (the guard is thread-invariant, warp-uniform where
        // hd_v is a warp multiple, and never diverges a __syncthreads).  Shared correctness
        // fix of the single-scan, split and block kernels; numerics for valid outputs are
        // unchanged (the skipped lanes never reached the output).  MiMo's hd_v=128 never
        // trips this - the branch is for the API-admitted hd_v < 128 geometries.
        if (t < hd_v)
            for (int c = 0; c < n_here; ++c) {
                const float v = f32_from_f16(v_cache[((size_t) skpos[c] * n_kv_head + kvh) * hd_v + t]);
#pragma unroll
                for (int h = 0; h < G; ++h) acc[h] = fmaf(sp[h][c], v, acc[h]);
            }
        __syncthreads();
    }

    // the sink logit joins the denominator RAW (never sm_scale'd) as a zero-value virtual key
    for (int h = warp; h < G; h += kWarps) {
        const float m_run = sm[h];
        float m_fin = m_run, sink_e = 0.0f;
        if (sinks != nullptr) {
            const float sink = sinks[(size_t) kvh * G + h];
            m_fin = fmaxf(m_run, sink);
            sink_e = __expf(sink - m_fin);
        }
        if (lane == 0) {
            const float rescale = m_run == -FLT_MAX ? 0.0f : __expf(m_run - m_fin);
            sr[h] = rescale;
            sl[h] = sl[h] * rescale + sink_e;
        }
    }
    __syncthreads();
    if (t < hd_v) {
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float l = sl[h];
            out[((size_t) row * n_q_head + (size_t) kvh * G + h) * hd_v + t] =
                l > 0.0f ? acc[h] * sr[h] / l : 0.0f;
        }
    }
}

bool mimo_attn_args_ok(const char* fn, const void* q, const void* k_cache, const void* v_cache,
                       const void* out, int rows, int n_q_head, int n_kv_head, int head_dim_qk,
                       int head_dim_v, int capacity, int window) {
    if (!q || !k_cache || !v_cache || !out || rows < 1 || rows > 65535 || capacity < 1 ||
        n_q_head < 1 || n_kv_head < 1 || n_q_head % n_kv_head != 0 || window < 0 ||
        head_dim_qk < 8 || head_dim_qk % 8 != 0 || head_dim_qk > kMaxDimQk ||
        head_dim_v < 1 || head_dim_v > kThreads) {
        std::fprintf(stderr,
                     "%s: unsupported geometry (rows=%d nq=%d nkv=%d hdq=%d hdv=%d cap=%d win=%d)\n",
                     fn, rows, n_q_head, n_kv_head, head_dim_qk, head_dim_v, capacity, window);
        return false;
    }
    switch (n_q_head / n_kv_head) {
    case 1: case 2: case 4: case 8: case 16: return true;
    default:
        std::fprintf(stderr, "%s: GQA group %d not compiled (supported: 1, 2, 4, 8, 16)\n", fn,
                     n_q_head / n_kv_head);
        return false;
    }
}

template <int G, bool PREFILL, bool RING>
void mimo_attn_launch(const void* q, const void* k_cache, const void* v_cache, void* out,
                      const int64_t* positions, const int* seq_lens, int64_t pos0, int rows,
                      int n_q_head, int n_kv_head, int head_dim_qk, int head_dim_v, int window,
                      const void* sinks, double sm_scale, void* stream, int cache_ring) {
    const dim3 grid((unsigned) n_kv_head, (unsigned) rows);
    mimo_attn_kernel<G, PREFILL, RING><<<grid, kThreads, 0, (cudaStream_t) stream>>>(
        (const uint16_t*) q, (const uint16_t*) k_cache, (const uint16_t*) v_cache, (float*) out,
        positions, seq_lens, pos0, n_q_head, n_kv_head, head_dim_qk, head_dim_v, window,
        (const float*) sinks, (float) sm_scale, cache_ring);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "mimo_attn: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

template <bool PREFILL, bool RING = false>
void mimo_attn_dispatch(const void* q, const void* k_cache, const void* v_cache, void* out,
                        const int64_t* positions, const int* seq_lens, int64_t pos0, int rows,
                        int n_q_head, int n_kv_head, int head_dim_qk, int head_dim_v,
                        int capacity, int window, const void* sinks, double sm_scale,
                        void* stream, int cache_ring = 0) {
    const int g = n_q_head / n_kv_head;
    switch (g) {
    case 16: mimo_attn_launch<16, PREFILL, RING>(q, k_cache, v_cache, out, positions, seq_lens, pos0, rows, n_q_head, n_kv_head, head_dim_qk, head_dim_v, window, sinks, sm_scale, stream, cache_ring); break;
    case 8:  mimo_attn_launch<8, PREFILL, RING>(q, k_cache, v_cache, out, positions, seq_lens, pos0, rows, n_q_head, n_kv_head, head_dim_qk, head_dim_v, window, sinks, sm_scale, stream, cache_ring); break;
    case 4:  mimo_attn_launch<4, PREFILL, RING>(q, k_cache, v_cache, out, positions, seq_lens, pos0, rows, n_q_head, n_kv_head, head_dim_qk, head_dim_v, window, sinks, sm_scale, stream, cache_ring); break;
    case 2:  mimo_attn_launch<2, PREFILL, RING>(q, k_cache, v_cache, out, positions, seq_lens, pos0, rows, n_q_head, n_kv_head, head_dim_qk, head_dim_v, window, sinks, sm_scale, stream, cache_ring); break;
    default: mimo_attn_launch<1, PREFILL, RING>(q, k_cache, v_cache, out, positions, seq_lens, pos0, rows, n_q_head, n_kv_head, head_dim_qk, head_dim_v, window, sinks, sm_scale, stream, cache_ring); break;
    }
}

}  // namespace

void mimo_rope_f16(void* qk, const int64_t* positions, int rows, int n_heads, int head_dim_qk,
                   int n_rot, double theta, void* stream) {
    if (!qk || !positions || rows < 1 || rows > 65535 || n_heads < 1 || head_dim_qk < n_rot ||
        n_rot < 2 || (n_rot & 1) != 0 || n_rot > 64 || !(theta > 0.0) || !std::isfinite(theta)) {
        std::fprintf(stderr, "mimo_rope_f16: invalid arguments (rows=%d n_heads=%d hd=%d n_rot=%d)\n",
                     rows, n_heads, head_dim_qk, n_rot);
        std::exit(1);
    }
    const long long total = (long long) rows * n_heads * (n_rot / 2);
    const unsigned blocks = (unsigned) ((total + kThreads - 1) / kThreads);
    mimo_rope_kernel<<<blocks, kThreads, 0, (cudaStream_t) stream>>>((uint16_t*) qk, positions,
                                                                     rows, n_heads, head_dim_qk,
                                                                     n_rot, theta);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "mimo_rope_f16: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

void mimo_attn_decode(const void* q, const void* k_cache, const void* v_cache, void* out,
                      const int64_t* positions, const int* seq_lens, int rows, int n_q_head,
                      int n_kv_head, int head_dim_qk, int head_dim_v, int capacity, int window,
                      const void* sinks, double sm_scale, void* stream, int cache_ring) {
    if (!positions || !seq_lens || cache_ring < 0 ||
        (cache_ring > 0 && (window <= 0 || cache_ring < std::min(capacity, window))) ||
        !mimo_attn_args_ok("mimo_attn_decode", q, k_cache, v_cache, out, rows, n_q_head, n_kv_head,
                           head_dim_qk, head_dim_v, capacity, window))
        std::exit(1);
    if (cache_ring > 0) {
        mimo_attn_dispatch<false, true>(q, k_cache, v_cache, out, positions, seq_lens, 0, rows, n_q_head,
                                       n_kv_head, head_dim_qk, head_dim_v, capacity, window, sinks,
                                       sm_scale, stream, cache_ring);
        return;
    }
    mimo_attn_dispatch<false>(q, k_cache, v_cache, out, positions, seq_lens, 0, rows, n_q_head,
                              n_kv_head, head_dim_qk, head_dim_v, capacity, window, sinks,
                              sm_scale, stream);
}

void mimo_attn_prefill(const void* q, const void* k_cache, const void* v_cache, void* out,
                       int64_t pos0, int rows, int n_q_head, int n_kv_head, int head_dim_qk,
                       int head_dim_v, int capacity, int window, const void* sinks,
                       double sm_scale, void* stream, int cache_ring) {
    if (pos0 < 0 || pos0 + rows > (int64_t) capacity || cache_ring < 0 ||
        (cache_ring > 0 && (window <= 0 || cache_ring < std::min<int64_t>(capacity, (int64_t) window + rows - 1))) ||
        !mimo_attn_args_ok("mimo_attn_prefill", q, k_cache, v_cache, out, rows, n_q_head, n_kv_head,
                           head_dim_qk, head_dim_v, capacity, window))
        std::exit(1);
    if (cache_ring > 0) {
        mimo_attn_dispatch<true, true>(q, k_cache, v_cache, out, nullptr, nullptr, pos0, rows, n_q_head,
                                      n_kv_head, head_dim_qk, head_dim_v, capacity, window, sinks,
                                      sm_scale, stream, cache_ring);
        return;
    }
    mimo_attn_dispatch<true>(q, k_cache, v_cache, out, nullptr, nullptr, pos0, rows, n_q_head,
                             n_kv_head, head_dim_qk, head_dim_v, capacity, window, sinks,
                             sm_scale, stream);
}

// ------------------------------------------------------------------ split-sequence (opt-in)
//
// STRATA_MIMO_ATTN_SPLIT gates the *_split entries (header above): the [lo, hi] scan is cut
// into `n_splits` key ranges per (query row, kv head); each partial block runs the SAME tiling
// and online softmax as the single-scan kernel over its sub-range and writes (m_s, l_s, acc_s)
// fp32 partials to the caller's pre-reserved workspace; a merge kernel launched on the SAME
// stream afterwards folds the partials in split order (deterministic - no atomics, no arrival
// counters, no spinlocks; stream order is the only synchronization) and joins the sink logit
// exactly once there, RAW and zero-value, so the softmax conventions of the single-scan path
// hold bit-for-bit in structure: all-masked rows output zeros with or without sinks.  The
// K/V caches are only read.  Gate off, or a plan of < 2 splits, dispatches the single-scan
// kernels through the plain entries' exact code path (byte-identical outputs).

// Workspace slot layout, pinned here so reservation and kernels cannot drift: slot
// (row * n_kv_head + kvh) * n_splits + split holds G heads of (hd_v + 2) fp32 each - acc lanes
// [0, hd_v), running max at [hd_v], denominator at [hd_v + 1] - and the whole buffer is the
// header's align256(rows * n_kv * splits * G * (hd_v + 2) * 4) bytes.
namespace {

constexpr int kSplitKeys = 128;    // target keys per partial block (measured 128 vs 256 in the bench)
constexpr int kSplitTargetBlocks = 512;

int mimo_attn_split_plan(int64_t keys_bound, int rows, int n_kv_head, int max_splits) {
    int64_t n = (keys_bound + kSplitKeys - 1) / kSplitKeys;
    const int64_t occ = std::max<int64_t>(1, kSplitTargetBlocks / ((int64_t) rows * n_kv_head));
    if (n > occ) n = occ;
    if (n > (int64_t) max_splits) n = max_splits;
    return n < 1 ? 1 : (int) n;
}

template <int G, bool PREFILL, bool RING>
__global__ void __launch_bounds__(kThreads) mimo_attn_partial_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache, const int64_t* __restrict__ positions,
    const int* __restrict__ seq_lens, int64_t pos0, int n_q_head, int n_kv_head, int hd_qk,
    int hd_v, int window, float sm_scale, int cache_ring, int keys_per_split, int n_splits,
    float* __restrict__ part) {
    const int split = blockIdx.x, row = blockIdx.z, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int64_t p = PREFILL ? pos0 + row : positions[row];
    const int n = PREFILL ? (int) (pos0 + gridDim.z) : seq_lens[row];   // rows is gridDim.z here
    const int hi = (int) (p < (int64_t) n - 1 ? p : (int64_t) n - 1);
    const int lo = window > 0 ? (int) (p - (window - 1) > 0 ? p - (window - 1) : 0) : 0;
    float* const slot = part + (((size_t) row * n_kv_head + (size_t) kvh) * (size_t) n_splits +
                                (size_t) split) * (size_t) G * (hd_v + 2);
    const int64_t beg = (int64_t) lo + (int64_t) split * keys_per_split;
    if (lo > hi || beg > hi) {   // this split holds no key: an empty partial the merge skips
        if (t < G) {
            float* ph = slot + (size_t) t * (hd_v + 2);
            ph[hd_v] = -FLT_MAX;
            ph[hd_v + 1] = 0.0f;
        }
        return;
    }
    // the last split absorbs the tail, so full coverage of [lo, hi] never depends on the
    // host-side key bound being an overestimate
    int64_t end = split + 1 < n_splits ? beg + keys_per_split : (int64_t) hi + 1;
    if (end > (int64_t) hi + 1) end = hi + 1;

    __shared__ __align__(16) float sq[G][kMaxDimQk];
    __shared__ float sp[G][kChunk];
    __shared__ float sm[G], sl[G], sr[G];
    __shared__ int skpos[kChunk];

    const uint16_t* qrow = q + ((size_t) row * n_q_head + (size_t) kvh * G) * hd_qk;
    for (int i = t; i < G * kMaxDimQk; i += kThreads) {
        const int d = i & (kMaxDimQk - 1), h = i >> 8;
        sq[h][d] = d < hd_qk ? f32_from_f16(qrow[(size_t) h * hd_qk + d]) : 0.0f;
    }
    if (t < G) { sm[t] = -FLT_MAX; sl[t] = 0.0f; }
    __syncthreads();

    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;

    for (int c0 = (int) beg; c0 < (int) end; c0 += kChunk) {
        const int n_here = min(kChunk, (int) end - c0);
        if (t < kChunk) {
            int at = c0 + t;
            if constexpr (RING) at %= cache_ring;
            skpos[t] = t < n_here ? at : -1;
        }
        __syncthreads();
        for (int c = warp; c < kChunk; c += kWarps) {
            if (c >= n_here) {
                if (lane < G) sp[lane][c] = -FLT_MAX;
                continue;
            }
            float k8[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            if (lane * 8 < hd_qk) {
                uint16_t raw[8];
                *reinterpret_cast<uint4*>(raw) = *reinterpret_cast<const uint4*>(
                    k_cache + ((size_t) skpos[c] * n_kv_head + kvh) * hd_qk + lane * 8);
#pragma unroll
                for (int j = 0; j < 8; ++j) k8[j] = f32_from_f16(raw[j]);
            }
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
                float s = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                          k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
                s = warp_sum(s);
                if (lane == 0) sp[h][c] = s * sm_scale;
            }
        }
        __syncthreads();
        for (int h = warp; h < G; h += kWarps) {
            const float a = sp[h][lane], b = sp[h][lane + 32];
            const float mc = warp_max(fmaxf(a, b));
            const float m_old = sm[h], m_new = fmaxf(m_old, mc);
            const float ea = lane < n_here ? __expf(a - m_new) : 0.0f;
            const float eb = lane + 32 < n_here ? __expf(b - m_new) : 0.0f;
            sp[h][lane] = ea;
            sp[h][lane + 32] = eb;
            const float l_add = warp_sum(ea + eb);
            if (lane == 0) {
                const float rescale = m_old == -FLT_MAX ? 0.0f : __expf(m_old - m_new);
                sl[h] = sl[h] * rescale + l_add;
                sr[h] = rescale;
                sm[h] = m_new;
            }
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] *= sr[h];
        // V rows are hd_v wide: lanes t >= hd_v would read past the row's tail - out of the
        // buffer entirely at the last (kv head, row).  Their accumulator lanes are discarded
        // at writeout, so those threads skip the loads wholesale; no barrier sits inside, so
        // block cooperation is unchanged (the guard is thread-invariant, warp-uniform where
        // hd_v is a warp multiple, and never diverges a __syncthreads).  Shared correctness
        // fix of the single-scan, split and block kernels; numerics for valid outputs are
        // unchanged (the skipped lanes never reached the output).  MiMo's hd_v=128 never
        // trips this - the branch is for the API-admitted hd_v < 128 geometries.
        if (t < hd_v)
            for (int c = 0; c < n_here; ++c) {
                const float v = f32_from_f16(v_cache[((size_t) skpos[c] * n_kv_head + kvh) * hd_v + t]);
#pragma unroll
                for (int h = 0; h < G; ++h) acc[h] = fmaf(sp[h][c], v, acc[h]);
            }
        __syncthreads();
    }

    // partial writeout - NO sink here; the merge joins it exactly once, RAW and zero-value.
    // Lane 0 stores m/l at [hd_v, hd_v + 1], threads < hd_v store their accumulator lane:
    // disjoint addresses, no extra barrier needed past the loop's final __syncthreads.
    for (int h = warp; h < G; h += kWarps) {
        if (lane == 0) {
            float* ph = slot + (size_t) h * (hd_v + 2);
            ph[hd_v] = sm[h];
            ph[hd_v + 1] = sl[h];
        }
    }
    if (t < hd_v) {
        for (int h = 0; h < G; ++h) {
            float* ph = slot + (size_t) h * (hd_v + 2);
            ph[t] = acc[h];
        }
    }
}

// One 128-thread block per (query head, row): fold the split partials in split order, then
// join the sink once.  The sink joins here exactly ONCE, RAW (never sm_scale'd), zero-value
// virtual key.  Reads of m/l are same-address broadcasts; acc lanes are consecutive across t.
// Deterministic: fixed order, fp32 fmaf chains, no atomics.
template <int G>
__global__ void __launch_bounds__(kThreads) mimo_attn_merge_kernel(
    const float* __restrict__ part, float* __restrict__ out, const float* __restrict__ sinks,
    int n_q_head, int n_kv_head, int hd_v, int n_splits) {
    const int h = blockIdx.x, row = blockIdx.y, t = threadIdx.x;
    const int kvh = h / G, hl = h % G;
    const size_t G_hdv2 = (size_t) G * (hd_v + 2);
    const float* const row_base =
        part + (((size_t) row * n_kv_head + (size_t) kvh) * (size_t) n_splits) * G_hdv2 +
        (size_t) hl * (hd_v + 2);
    float M = -FLT_MAX;
    for (int s = 0; s < n_splits; ++s) M = fmaxf(M, row_base[(size_t) s * G_hdv2 + hd_v]);
    float L = 0.0f, acc = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const float* ph = row_base + (size_t) s * G_hdv2;
        const float m = ph[hd_v];
        // -FLT_MAX is the EMPTY-split sentinel written by the partial kernel's early exit.
        // A real score equals it only if q.k saturates fp32 at -3.4e38, which finite F16
        // inputs cannot produce (|q.k| <= 256*65504*65504*scale is bounded far below); the
        // parity suite's non-finite policy reports such inputs rather than silently
        // skipping them.  This is the documented sentinel contract, not a universal mask.
        if (m == -FLT_MAX) continue;   // empty split contributed nothing
        const float w = __expf(m - M);
        L = fmaf(ph[hd_v + 1], w, L);
        if (t < hd_v) acc = fmaf(ph[t], w, acc);
    }
    // the sink: zero-value virtual key, RAW logit, exactly one merge - all-masked rows keep
    // the single-scan conventions (L == 0 without sinks -> zeros; with sinks L == 1, acc == 0
    // -> zeros)
    if (sinks != nullptr) {
        const float sink = sinks[h];
        const float m_fin = fmaxf(M, sink);
        const float rescale = M == -FLT_MAX ? 0.0f : __expf(M - m_fin);
        L = L * rescale + __expf(sink - m_fin);
        acc *= rescale;
    }
    if (t < hd_v)
        out[((size_t) row * n_q_head + h) * hd_v + t] = L > 0.0f ? acc / L : 0.0f;
}

template <int G, bool PREFILL, bool RING>
void mimo_attn_split_launch(const void* q, const void* k_cache, const void* v_cache, void* out,
                            const int64_t* positions, const int* seq_lens, int64_t pos0, int rows,
                            int n_q_head, int n_kv_head, int head_dim_qk, int head_dim_v,
                            int window, const void* sinks, double sm_scale, void* stream,
                            int cache_ring, int keys_per_split, int n_splits, float* part) {
    const dim3 pgrid((unsigned) n_splits, (unsigned) n_kv_head, (unsigned) rows);
    mimo_attn_partial_kernel<G, PREFILL, RING><<<pgrid, kThreads, 0, (cudaStream_t) stream>>>(
        (const uint16_t*) q, (const uint16_t*) k_cache, (const uint16_t*) v_cache, positions,
        seq_lens, pos0, n_q_head, n_kv_head, head_dim_qk, head_dim_v, window, (float) sm_scale,
        cache_ring, keys_per_split, n_splits, part);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "mimo_attn_split partial: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    const dim3 mgrid((unsigned) n_q_head, (unsigned) rows);
    mimo_attn_merge_kernel<G><<<mgrid, kThreads, 0, (cudaStream_t) stream>>>(
        part, (float*) out, (const float*) sinks, n_q_head, n_kv_head, head_dim_v, n_splits);
    const cudaError_t e2 = cudaGetLastError();
    if (e2 != cudaSuccess) {
        std::fprintf(stderr, "mimo_attn_split merge: %s\n", cudaGetErrorString(e2));
        std::exit(1);
    }
}

template <bool PREFILL>
void mimo_attn_split_dispatch(const void* q, const void* k_cache, const void* v_cache, void* out,
                              const int64_t* positions, const int* seq_lens, int64_t pos0, int rows,
                              int n_q_head, int n_kv_head, int head_dim_qk, int head_dim_v,
                              int capacity, int window, const void* sinks, double sm_scale,
                              void* stream, int cache_ring, int keys_hint, int splits_hint,
                              const MimoAttnSplitWs* ws) {
    // the plan: bound the keys any row can attend, then pick split widths (~128 keys each)
    // with the partial grid near kSplitTargetBlocks blocks, clamped to the reservation
    int64_t bound = (int64_t) capacity;
    if (window > 0 && (int64_t) window < bound) bound = window;
    if (keys_hint > 0 && (int64_t) keys_hint < bound) bound = keys_hint;
    const int g = n_q_head / n_kv_head;
    int n_splits = 0;
    // everything about the workspace - alignment, bounds, int-range casts, the byte
    // requirement for THIS call's clamped plan - is validated by the pure-host helper; it
    // has already reported the reason on failure
    int n_checked = 1;
    if (!mimo_attn_split_ws_check(ws, (uint32_t) rows, (uint32_t) n_kv_head, (uint32_t) n_q_head,
                                  (uint32_t) head_dim_v, bound, splits_hint, &n_checked))
        std::exit(1);
    if (n_checked < 2) {   // reservation clamps the plan below 2: same byte-identical fallback
        if (cache_ring > 0)
            mimo_attn_dispatch<PREFILL, true>(q, k_cache, v_cache, out, positions, seq_lens, pos0,
                                              rows, n_q_head, n_kv_head, head_dim_qk, head_dim_v,
                                              capacity, window, sinks, sm_scale, stream, cache_ring);
        else
            mimo_attn_dispatch<PREFILL>(q, k_cache, v_cache, out, positions, seq_lens, pos0, rows,
                                        n_q_head, n_kv_head, head_dim_qk, head_dim_v, capacity,
                                        window, sinks, sm_scale, stream);
        return;
    }
    n_splits = n_checked;
    const int keys_per_split = (int) std::min<int64_t>(INT32_MAX, (bound + n_splits - 1) / n_splits);
    float* part = (float*) ws->mem;   // 256-byte aligned - the helper rejected anything else
    const bool ring = cache_ring > 0;   // the RING template maps kpos -> kpos % cache_ring
    if (!ring) {
        switch (g) {
        case 16: mimo_attn_split_launch<16, PREFILL, false>(q, k_cache, v_cache, out, positions, seq_lens, pos0, rows, n_q_head, n_kv_head, head_dim_qk, head_dim_v, window, sinks, sm_scale, stream, cache_ring, keys_per_split, n_splits, part); break;
        case 8:  mimo_attn_split_launch<8, PREFILL, false>(q, k_cache, v_cache, out, positions, seq_lens, pos0, rows, n_q_head, n_kv_head, head_dim_qk, head_dim_v, window, sinks, sm_scale, stream, cache_ring, keys_per_split, n_splits, part); break;
        case 4:  mimo_attn_split_launch<4, PREFILL, false>(q, k_cache, v_cache, out, positions, seq_lens, pos0, rows, n_q_head, n_kv_head, head_dim_qk, head_dim_v, window, sinks, sm_scale, stream, cache_ring, keys_per_split, n_splits, part); break;
        case 2:  mimo_attn_split_launch<2, PREFILL, false>(q, k_cache, v_cache, out, positions, seq_lens, pos0, rows, n_q_head, n_kv_head, head_dim_qk, head_dim_v, window, sinks, sm_scale, stream, cache_ring, keys_per_split, n_splits, part); break;
        default: mimo_attn_split_launch<1, PREFILL, false>(q, k_cache, v_cache, out, positions, seq_lens, pos0, rows, n_q_head, n_kv_head, head_dim_qk, head_dim_v, window, sinks, sm_scale, stream, cache_ring, keys_per_split, n_splits, part); break;
        }
    } else {
        switch (g) {
        case 16: mimo_attn_split_launch<16, PREFILL, true>(q, k_cache, v_cache, out, positions, seq_lens, pos0, rows, n_q_head, n_kv_head, head_dim_qk, head_dim_v, window, sinks, sm_scale, stream, cache_ring, keys_per_split, n_splits, part); break;
        case 8:  mimo_attn_split_launch<8, PREFILL, true>(q, k_cache, v_cache, out, positions, seq_lens, pos0, rows, n_q_head, n_kv_head, head_dim_qk, head_dim_v, window, sinks, sm_scale, stream, cache_ring, keys_per_split, n_splits, part); break;
        case 4:  mimo_attn_split_launch<4, PREFILL, true>(q, k_cache, v_cache, out, positions, seq_lens, pos0, rows, n_q_head, n_kv_head, head_dim_qk, head_dim_v, window, sinks, sm_scale, stream, cache_ring, keys_per_split, n_splits, part); break;
        case 2:  mimo_attn_split_launch<2, PREFILL, true>(q, k_cache, v_cache, out, positions, seq_lens, pos0, rows, n_q_head, n_kv_head, head_dim_qk, head_dim_v, window, sinks, sm_scale, stream, cache_ring, keys_per_split, n_splits, part); break;
        default: mimo_attn_split_launch<1, PREFILL, true>(q, k_cache, v_cache, out, positions, seq_lens, pos0, rows, n_q_head, n_kv_head, head_dim_qk, head_dim_v, window, sinks, sm_scale, stream, cache_ring, keys_per_split, n_splits, part); break;
        }
    }
}

}  // namespace

bool mimo_attn_split_enabled() {
    const char* e = std::getenv("STRATA_MIMO_ATTN_SPLIT");
    return e != nullptr && e[0] != '\0' && std::strcmp(e, "0") != 0;
}

uint64_t mimo_attn_split_ws_bytes(uint32_t max_rows, uint32_t n_kv_head, uint32_t n_q_head,
                                  uint32_t head_dim_v, uint32_t max_splits) {
    const char* fn = "mimo_attn_split_ws_bytes";
    const uint32_t g = n_kv_head != 0 ? n_q_head / n_kv_head : 0;
    if (max_rows < 1 || n_kv_head < 1 || n_q_head < 1 || n_q_head % n_kv_head != 0 ||
        head_dim_v < 1 || head_dim_v > (uint32_t) kThreads || max_splits < 1 ||
        (g != 1 && g != 2 && g != 4 && g != 8 && g != 16)) {   // exactly the compiled G set
        std::fprintf(stderr,
                     "%s: unsupported reservation (rows=%u kv=%u nq=%u hdv=%u splits=%u)\n", fn,
                     max_rows, n_kv_head, n_q_head, head_dim_v, max_splits);
        return 0;
    }
    const uint64_t gq = (uint64_t) g;
    // rows * kv * splits * G * (hd_v + 2) * 4, every product checked for 64-bit overflow
    const uint64_t lim = UINT64_MAX;
    uint64_t n = (uint64_t) max_rows;
    for (const uint64_t f : { (uint64_t) n_kv_head, (uint64_t) max_splits, gq,
                              (uint64_t) (head_dim_v + 2), (uint64_t) 4 }) {
        if (f != 0 && n > lim / f) {
            std::fprintf(stderr, "%s: reservation overflows 64 bits\n", fn);
            return 0;
        }
        n *= f;
    }
    const uint64_t bytes = n;   // n == rows*kv*splits*G*(hd_v+2)*4, already <= UINT64_MAX
    if (bytes > lim - 255) {
        std::fprintf(stderr, "%s: reservation overflows 64 bits\n", fn);
        return 0;
    }
    return (bytes + 255) & ~(uint64_t) 255;   // 256-byte alignment of the first slot
}

bool mimo_attn_split_ws_check(const MimoAttnSplitWs* ws, uint32_t rows, uint32_t n_kv_head,
                              uint32_t n_q_head, uint32_t head_dim_v, int64_t keys_bound,
                              int splits_hint, int* n_splits_out) {
    const char* fn = "mimo_attn_split_ws_check";
    if (n_splits_out != nullptr) *n_splits_out = 1;
    if (ws == nullptr || ws->mem == nullptr) {
        std::fprintf(stderr, "%s: no workspace (ws=%p mem=%p)\n", fn, (const void*) ws,
                     ws ? ws->mem : nullptr);
        return false;
    }
    if (((uintptr_t) ws->mem & (uintptr_t) 255) != 0) {
        std::fprintf(stderr, "%s: workspace base %p is not 256-byte aligned\n", fn, ws->mem);
        return false;
    }
    // max_rows/max_splits flow into int arithmetic downstream (the plan, grid dims): reject
    // values whose uint32->int cast would wrap negative instead of silently mis-clamping
    if (ws->max_rows < 1 || ws->max_rows > (uint32_t) INT32_MAX ||
        ws->max_splits < 1 || ws->max_splits > (uint32_t) INT32_MAX) {
        std::fprintf(stderr, "%s: reservation bounds out of range (max_rows=%u max_splits=%u)\n",
                     fn, ws->max_rows, ws->max_splits);
        return false;
    }
    if (n_kv_head < 1 || n_kv_head > (uint32_t) INT32_MAX || n_q_head < 1 ||
        n_q_head > (uint32_t) INT32_MAX) {
        std::fprintf(stderr, "%s: geometry out of range (kv=%u nq=%u)\n", fn, n_kv_head,
                     n_q_head);
        return false;
    }
    if (rows < 1 || rows > ws->max_rows) {
        std::fprintf(stderr, "%s: rows %u exceed the reservation %u\n", fn, rows, ws->max_rows);
        return false;
    }
    if (keys_bound < 1) keys_bound = 1;   // a nonsense hint must not zero the plan
    const int n = splits_hint > 0
                      ? std::min(splits_hint, (int) ws->max_splits)
                      : mimo_attn_split_plan(keys_bound, (int) rows, (int) n_kv_head,
                                             (int) ws->max_splits);
    const uint64_t need = mimo_attn_split_ws_bytes(rows, n_kv_head, n_q_head, head_dim_v,
                                                   (uint32_t) (n < 1 ? 1 : n));
    if (need == 0) return false;   // geometry/overflow - already reported
    if (ws->bytes < need) {
        std::fprintf(stderr,
                     "%s: workspace too small (%llu < %llu bytes for rows=%u kv=%u splits=%d)\n",
                     fn, (unsigned long long) ws->bytes, (unsigned long long) need, rows,
                     n_kv_head, n);
        return false;
    }
    if (n_splits_out != nullptr) *n_splits_out = n < 1 ? 1 : n;
    return true;
}

void mimo_attn_decode_split(const void* q, const void* k_cache, const void* v_cache, void* out,
                            const int64_t* positions, const int* seq_lens, int rows, int n_q_head,
                            int n_kv_head, int head_dim_qk, int head_dim_v, int capacity, int window,
                            const void* sinks, double sm_scale, int keys_hint, int splits_hint,
                            const MimoAttnSplitWs* ws, void* stream, int cache_ring) {
    if (!positions || !seq_lens || cache_ring < 0 ||
        (cache_ring > 0 && (window <= 0 || cache_ring < std::min(capacity, window))) ||
        !mimo_attn_args_ok("mimo_attn_decode_split", q, k_cache, v_cache, out, rows, n_q_head,
                           n_kv_head, head_dim_qk, head_dim_v, capacity, window))
        std::exit(1);
    if (!mimo_attn_split_enabled()) {
        // gate OFF: ws is intentionally ignored - callers that skip reserving when the
        // feature is off (the engine does) still get the byte-identical single-scan path
        if (cache_ring > 0)
            mimo_attn_dispatch<false, true>(q, k_cache, v_cache, out, positions, seq_lens, 0, rows,
                                            n_q_head, n_kv_head, head_dim_qk, head_dim_v, capacity,
                                            window, sinks, sm_scale, stream, cache_ring);
        else
            mimo_attn_dispatch<false>(q, k_cache, v_cache, out, positions, seq_lens, 0, rows,
                                      n_q_head, n_kv_head, head_dim_qk, head_dim_v, capacity,
                                      window, sinks, sm_scale, stream);
        return;
    }
    if (ws == nullptr || (ws->mem == nullptr && ws->max_splits >= 2)) {
        // gate ON: a caller that enabled splitting must hand a usable descriptor - falling
        // back silently would hide the mistake, and launching against a null buffer would
        // hand the kernels pointer 0.  (ws->mem null with max_splits < 2 is the documented
        // degenerate reservation - an explicit opt-out that falls back below.)
        std::fprintf(stderr,
                     "mimo_attn_decode_split: gate on but no usable workspace "
                     "(ws=%p mem=%p max_splits=%u)\n",
                     (const void*) ws, ws ? ws->mem : nullptr, ws ? ws->max_splits : 0u);
        std::exit(1);
    }
    if (ws->mem == nullptr) {
        if (cache_ring > 0)
            mimo_attn_dispatch<false, true>(q, k_cache, v_cache, out, positions, seq_lens, 0, rows,
                                            n_q_head, n_kv_head, head_dim_qk, head_dim_v, capacity,
                                            window, sinks, sm_scale, stream, cache_ring);
        else
            mimo_attn_dispatch<false>(q, k_cache, v_cache, out, positions, seq_lens, 0, rows,
                                      n_q_head, n_kv_head, head_dim_qk, head_dim_v, capacity,
                                      window, sinks, sm_scale, stream);
        return;
    }
    mimo_attn_split_dispatch<false>(q, k_cache, v_cache, out, positions, seq_lens, 0, rows,
                                    n_q_head, n_kv_head, head_dim_qk, head_dim_v, capacity, window,
                                    sinks, sm_scale, stream, cache_ring, keys_hint, splits_hint, ws);
}

void mimo_attn_prefill_split(const void* q, const void* k_cache, const void* v_cache, void* out,
                             int64_t pos0, int rows, int n_q_head, int n_kv_head, int head_dim_qk,
                             int head_dim_v, int capacity, int window, const void* sinks,
                             double sm_scale, int keys_hint, int splits_hint,
                             const MimoAttnSplitWs* ws, void* stream, int cache_ring) {
    if (pos0 < 0 || pos0 + rows > (int64_t) capacity || cache_ring < 0 ||
        (cache_ring > 0 && (window <= 0 || cache_ring < std::min<int64_t>(capacity, (int64_t) window + rows - 1))) ||
        !mimo_attn_args_ok("mimo_attn_prefill_split", q, k_cache, v_cache, out, rows, n_q_head,
                           n_kv_head, head_dim_qk, head_dim_v, capacity, window))
        std::exit(1);
    if (!mimo_attn_split_enabled()) {
        if (cache_ring > 0)
            mimo_attn_dispatch<true, true>(q, k_cache, v_cache, out, nullptr, nullptr, pos0, rows,
                                           n_q_head, n_kv_head, head_dim_qk, head_dim_v, capacity,
                                           window, sinks, sm_scale, stream, cache_ring);
        else
            mimo_attn_dispatch<true>(q, k_cache, v_cache, out, nullptr, nullptr, pos0, rows,
                                     n_q_head, n_kv_head, head_dim_qk, head_dim_v, capacity,
                                     window, sinks, sm_scale, stream);
        return;
    }
    if (ws == nullptr || (ws->mem == nullptr && ws->max_splits >= 2)) {
        // gate ON: no usable descriptor is an error, not a silent fallback (decode's rule)
        std::fprintf(stderr,
                     "mimo_attn_prefill_split: gate on but no usable workspace "
                     "(ws=%p mem=%p max_splits=%u)\n",
                     (const void*) ws, ws ? ws->mem : nullptr, ws ? ws->max_splits : 0u);
        std::exit(1);
    }
    if (ws->mem == nullptr) {
        if (cache_ring > 0)
            mimo_attn_dispatch<true, true>(q, k_cache, v_cache, out, nullptr, nullptr, pos0, rows,
                                           n_q_head, n_kv_head, head_dim_qk, head_dim_v, capacity,
                                           window, sinks, sm_scale, stream, cache_ring);
        else
            mimo_attn_dispatch<true>(q, k_cache, v_cache, out, nullptr, nullptr, pos0, rows,
                                     n_q_head, n_kv_head, head_dim_qk, head_dim_v, capacity,
                                     window, sinks, sm_scale, stream);
        return;
    }
    if (keys_hint <= 0 && pos0 + rows > 0) keys_hint = (int) std::min<int64_t>(INT32_MAX, pos0 + rows);
    mimo_attn_split_dispatch<true>(q, k_cache, v_cache, out, nullptr, nullptr, pos0, rows,
                                   n_q_head, n_kv_head, head_dim_qk, head_dim_v, capacity, window,
                                   sinks, sm_scale, stream, cache_ring, keys_hint, splits_hint, ws);
}

// --------------------------------------------------------------------- block-diffusion (dflash)
//
// Same tiles as the decode kernel above - one 128-thread block per (query row, kv head), 64-wide
// key tiles merged through the running (max, denominator, accumulator), sink RAW at writeout.
// The mask is the only change: no causal cap at p, and hi extends to min(p + window_right,
// seq_len - 1) so a row sees keys AHEAD of it (the draft block's own rows bidirectionally,
// bounded by the row's valid-key count).  A separate kernel rather than a flag on the decode
// one: those are done and green, and the two mask regimes are one line apart but measurably
// different code paths (hi depends on window_right only here).
template <int G>
__global__ void __launch_bounds__(kThreads) mimo_attn_block_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache, float* __restrict__ out,
    const int64_t* __restrict__ positions, const int* __restrict__ seq_lens, int n_q_head,
    int n_kv_head, int hd_qk, int hd_v, int window, int window_right,
    const float* __restrict__ sinks, float sm_scale) {
    const int row = blockIdx.y, kvh = blockIdx.x;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int64_t p = positions[row];
    const int n = seq_lens[row];
    const int hi = (int) (p + window_right < (int64_t) n - 1 ? p + window_right : (int64_t) n - 1);
    const int lo = window > 0 ? (int) (p - (window - 1) > 0 ? p - (window - 1) : 0) : 0;

    __shared__ __align__(16) float sq[G][kMaxDimQk];
    __shared__ float sp[G][kChunk];
    __shared__ float sm[G], sl[G], sr[G];
    __shared__ int skpos[kChunk];

    const uint16_t* qrow = q + ((size_t) row * n_q_head + (size_t) kvh * G) * hd_qk;
    for (int i = t; i < G * kMaxDimQk; i += kThreads) {
        const int d = i & (kMaxDimQk - 1), h = i >> 8;
        sq[h][d] = d < hd_qk ? f32_from_f16(qrow[(size_t) h * hd_qk + d]) : 0.0f;
    }
    if (t < G) { sm[t] = -FLT_MAX; sl[t] = 0.0f; }
    __syncthreads();

    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;

    for (int c0 = lo; c0 <= hi; c0 += kChunk) {
        const int n_here = min(kChunk, hi - c0 + 1);
        if (t < kChunk) skpos[t] = t < n_here ? c0 + t : -1;
        __syncthreads();
        for (int c = warp; c < kChunk; c += kWarps) {
            if (c >= n_here) {
                if (lane < G) sp[lane][c] = -FLT_MAX;
                continue;
            }
            float k8[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            if (lane * 8 < hd_qk) {
                uint16_t raw[8];
                *reinterpret_cast<uint4*>(raw) = *reinterpret_cast<const uint4*>(
                    k_cache + ((size_t) skpos[c] * n_kv_head + kvh) * hd_qk + lane * 8);
#pragma unroll
                for (int j = 0; j < 8; ++j) k8[j] = f32_from_f16(raw[j]);
            }
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
                float s = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                          k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
                s = warp_sum(s);
                if (lane == 0) sp[h][c] = s * sm_scale;
            }
        }
        __syncthreads();
        for (int h = warp; h < G; h += kWarps) {
            const float a = sp[h][lane], b = sp[h][lane + 32];
            const float mc = warp_max(fmaxf(a, b));
            const float m_old = sm[h], m_new = fmaxf(m_old, mc);
            const float ea = lane < n_here ? __expf(a - m_new) : 0.0f;
            const float eb = lane + 32 < n_here ? __expf(b - m_new) : 0.0f;
            sp[h][lane] = ea;
            sp[h][lane + 32] = eb;
            const float l_add = warp_sum(ea + eb);
            if (lane == 0) {
                const float rescale = m_old == -FLT_MAX ? 0.0f : __expf(m_old - m_new);
                sl[h] = sl[h] * rescale + l_add;
                sr[h] = rescale;
                sm[h] = m_new;
            }
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] *= sr[h];
        // V rows are hd_v wide: lanes t >= hd_v would read past the row's tail - out of the
        // buffer entirely at the last (kv head, row).  Their accumulator lanes are discarded
        // at writeout, so those threads skip the loads wholesale; no barrier sits inside, so
        // block cooperation is unchanged (the guard is thread-invariant, warp-uniform where
        // hd_v is a warp multiple, and never diverges a __syncthreads).  Shared correctness
        // fix of the single-scan, split and block kernels; numerics for valid outputs are
        // unchanged (the skipped lanes never reached the output).  MiMo's hd_v=128 never
        // trips this - the branch is for the API-admitted hd_v < 128 geometries.
        if (t < hd_v)
            for (int c = 0; c < n_here; ++c) {
                const float v = f32_from_f16(v_cache[((size_t) skpos[c] * n_kv_head + kvh) * hd_v + t]);
#pragma unroll
                for (int h = 0; h < G; ++h) acc[h] = fmaf(sp[h][c], v, acc[h]);
            }
        __syncthreads();
    }

    // the sink joins the denominator RAW, as a zero-value virtual key (decode's writeout)
    for (int h = warp; h < G; h += kWarps) {
        const float m_run = sm[h];
        float m_fin = m_run, sink_e = 0.0f;
        if (sinks != nullptr) {
            const float sink = sinks[(size_t) kvh * G + h];
            m_fin = fmaxf(m_run, sink);
            sink_e = __expf(sink - m_fin);
        }
        if (lane == 0) {
            const float rescale = m_run == -FLT_MAX ? 0.0f : __expf(m_run - m_fin);
            sr[h] = rescale;
            sl[h] = sl[h] * rescale + sink_e;
        }
    }
    __syncthreads();
    if (t < hd_v) {
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float l = sl[h];
            out[((size_t) row * n_q_head + (size_t) kvh * G + h) * hd_v + t] =
                l > 0.0f ? acc[h] * sr[h] / l : 0.0f;
        }
    }
}

void mimo_attn_block(const void* q, const void* k_cache, const void* v_cache, void* out,
                     const int64_t* positions, const int* seq_lens, int rows, int n_q_head,
                     int n_kv_head, int head_dim_qk, int head_dim_v, int capacity, int window,
                     int window_right, const void* sinks, double sm_scale, void* stream) {
    if (!positions || !seq_lens || window_right < 0 ||
        !mimo_attn_args_ok("mimo_attn_block", q, k_cache, v_cache, out, rows, n_q_head, n_kv_head,
                           head_dim_qk, head_dim_v, capacity, window)) {
        std::fprintf(stderr,
                     "mimo_attn_block: invalid arguments (rows=%d window=%d window_right=%d)\n",
                     rows, window, window_right);
        std::exit(1);
    }
    const int g = n_q_head / n_kv_head;
    if (g != 1 && g != 8) {   // args_ok admits 2/4/16 too; only the drafter's 8 and the
        std::fprintf(stderr,   // parity fixture's 1 are compiled for the non-causal mask
                     "mimo_attn_block: GQA group %d not compiled (supported: 1, 8)\n", g);
        std::exit(1);
    }
    const dim3 grid((unsigned) n_kv_head, (unsigned) rows);
    if (g == 8)
        mimo_attn_block_kernel<8><<<grid, kThreads, 0, (cudaStream_t) stream>>>(
            (const uint16_t*) q, (const uint16_t*) k_cache, (const uint16_t*) v_cache,
            (float*) out, positions, seq_lens, n_q_head, n_kv_head, head_dim_qk, head_dim_v,
            window, window_right, (const float*) sinks, (float) sm_scale);
    else
        mimo_attn_block_kernel<1><<<grid, kThreads, 0, (cudaStream_t) stream>>>(
            (const uint16_t*) q, (const uint16_t*) k_cache, (const uint16_t*) v_cache,
            (float*) out, positions, seq_lens, n_q_head, n_kv_head, head_dim_qk, head_dim_v,
            window, window_right, (const float*) sinks, (float) sm_scale);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "mimo_attn_block: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
