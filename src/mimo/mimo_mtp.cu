// src/mimo/mimo_mtp.cu - see include/strata/mimo/mtp.hpp (the contract) and notes/MIMO-V2-PORT.md §2
// ("MTP drafter").
//
// Correctness first, like src/kernels/cuda/mimo_attn.cu: plain SIMT kernels for the pieces the tree
// does not already have, the shared kernels everywhere else -
//
//   norms        kernels::rms_norm_weighted (square, in place) and prefill::rms_rows (the eh halves,
//                ld = 2*n_embd), both F32, mean-form, eps from the geometry;
//   projections  prefill::Gemm::f16 (F16 weights x F16 activation image, F32 out) for rows > 1 and a
//                one-block-per-output-row F16 gemv at rows == 1 (the q36 driver's decode shape);
//   attention    kernels::mimo_rope_f16 on q and k (theta = rope_theta_swa, absolute positions from a
//                device buffer) and mimo_attn_decode / mimo_attn_prefill over the head's flat cache
//                (window INCLUDES the query, sink RAW, sm_scale = head_dim_qk^-0.5);
//   swiglu       a private kernel (prefill::swiglu_pair hardcodes its 640-wide shared-expert shape);
//   argmax       kernels::sample_tokens with greedy parameters over the shared lm_head's logits -
//                the tree's cross-backend pick (argmax_rows is a CUDA-only shape), ties to the
//                lowest id.
//
// The eh input kernel packs [f32(embed(tok)); hidden] rows so both RMSNorm halves sit in one buffer;
// the K/V append kernel writes k rows whole (192 lanes) and v rows trimmed to 128, at the rows'
// absolute positions.  Every launch is on the drafter's stream; the only sync is draft()'s readback.
#include "strata/mimo/mtp.hpp"

#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/mimo_attn.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::mimo {
namespace {

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

constexpr int kThreads = 128;

/// dst[r] = [f32(embed(tokens[r])); hidden[r]] - the eh_proj input with both halves in place for the
/// two rms_rows calls (the halves are strided by 2*n_embd, which rms_rows' ld exists for).
__global__ void mtp_eh_input_k(const uint16_t* __restrict__ embed, const int32_t* __restrict__ tokens,
                               const float* __restrict__ hidden, float* __restrict__ dst, int64_t n) {
    const int r = blockIdx.x;
    const uint16_t* e = embed + (int64_t) tokens[r] * n;
    float* d = dst + (int64_t) r * 2 * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        d[i] = f32_from_f16(e[i]);
        d[n + i] = hidden[(int64_t) r * n + i];
    }
}

/// pos[r] = pos0 + r, len[r] = pos0 + r + 1: the absolute positions of this window's rows and each
/// row's own history length (mimo_attn_decode's device arrays; the rope kernel reads pos too).
__global__ void mtp_pos_len_k(int64_t* __restrict__ pos, int* __restrict__ len, int64_t pos0, int rows) {
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        pos[r] = pos0 + r;
        len[r] = (int) (pos0 + r + 1);
    }
}

/// Appends one window's k/v rows to a head's flat cache at the rows' absolute positions: k whole
/// ([n_kv][hd_qk], rope already applied), v trimmed to its first hd_v lanes per head (the checkpoint
/// pads v to 192; the pack keeps the zeros and the cache stores the real 128).
__global__ void mtp_kv_append_k(const uint16_t* __restrict__ k16, const uint16_t* __restrict__ v16,
                                uint16_t* __restrict__ k_cache, uint16_t* __restrict__ v_cache,
                                const int64_t* __restrict__ pos, int rows, int n_kv, int hd_qk, int hd_v) {
    const int64_t per = (int64_t) n_kv * hd_qk;   // lanes per row (the v rows are 192-wide too)
    const int64_t g = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (int64_t) rows * per) return;
    const int r = (int) (g / per), lane = (int) (g % per);
    const int kvh = lane / hd_qk, d = lane % hd_qk;
    const int64_t p = pos[r];
    k_cache[p * per + lane] = k16[g];
    if (d < hd_v) v_cache[p * n_kv * hd_v + (int64_t) kvh * hd_v + d] = v16[g];
}

/// y[o] = sum_i f32(w[o][i]) * f32(x[i]) - the 1-row projection (F16 weight, F16 activation image,
/// F32 accumulate, one block per output row, the block's lanes striding the row).  The four warp
/// partials combine through shared memory, not a partial-mask shuffle: a 4-lane `__shfl_down_sync`
/// reads inactive lanes (undefined, and a hardware exception on gfx1100).
__global__ void mtp_gemv_k(const uint16_t* __restrict__ w_in, const uint16_t* __restrict__ x, float* __restrict__ y,
                           int n_in) {
    __shared__ float sh[kThreads / 32];
    const uint16_t* w = w_in + (int64_t) blockIdx.x * n_in;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n_in; i += kThreads) acc += f32_from_f16(w[i]) * f32_from_f16(x[i]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = sh[0];
#pragma unroll
        for (int w4 = 1; w4 < kThreads / 32; ++w4) v += sh[w4];
        y[blockIdx.x] = v;
    }
}

/// y[i] = f16(silu(g[i]) * u[i]), saturated at the F16 max (what a value past 65504 must become to
/// stay a finite activation; prefill::swiglu_pair's hf_sat arithmetic at a runtime width).
__global__ void mtp_swiglu_k(const float* __restrict__ g, const float* __restrict__ u, uint16_t* __restrict__ y,
                             int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = g[i];
    float s = a / (1.0f + __expf(-a)) * u[i];
    if (s > 65504.0f) s = 65504.0f;
    else if (s < -65504.0f) s = -65504.0f;
    y[i] = f16_from_f32(s);
}

bool launch_ok(const char* what, std::string& err) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("mimo mtp: ") + what + ": " + cudaGetErrorString(e);
        return false;
    }
    return true;
}

}  // namespace

// --------------------------------------------------------------------------- lifecycle

MimoMtp::~MimoMtp() {
    if (device_ >= 0) {
        int cur = -1;
        (void) cudaGetDevice(&cur);
        if (cur != device_) (void) cudaSetDevice(device_);
    }
    if (gemm_) delete gemm_;   // its destructor frees its own buffers on the drafter's device
    for (void* p : owned_) (void) cudaFree(p);
    owned_.clear();
}

bool MimoMtp::init(const core::MimoGeometry& g, const MimoPack* pack, int device, int64_t capacity, int max_rows,
                   void* stream, std::string& err) {
    if (!pack || (int) pack->mtp.size() != (int) g.n_mtp_layers || !pack->embed || !pack->lm_head) {
        err = "mimo mtp: the pack needs " + std::to_string(g.n_mtp_layers) + " MTP heads, embed and lm_head";
        return false;
    }
    for (const auto& h : pack->mtp)
        if (!h.eh || !h.q || !h.k || !h.v || !h.o || !h.mlp_gate || !h.mlp_up || !h.mlp_down || !h.hnorm ||
            !h.enorm || !h.norm_attn || !h.norm_mlp || !h.norm_final) {
            err = "mimo mtp: an MTP head is missing tensors (sink may be null, nothing else)";
            return false;
        }
    const int group = (int) (g.n_q_head / g.n_head_kv_swa);
    if (g.n_q_head % g.n_head_kv_swa != 0 || (group != 1 && group != 2 && group != 4 && group != 8 && group != 16) ||
        g.head_dim_qk % 8 != 0 || g.head_dim_qk > 256 || g.head_dim_v > 128 || g.n_rot > 64 ||
        g.n_rot > g.head_dim_qk || g.n_rot % 2 != 0 || g.sliding_window < 1 || g.n_embd < 1 ||
        g.dense_intermediate < 1 || g.n_vocab < 1 || capacity < 4 || max_rows < 1 || max_rows > 65535) {
        err = "mimo mtp: geometry outside the kernels' limits (group, head dims, n_rot, window, sizes)";
        return false;
    }
    const cudaError_t e = cudaSetDevice(device);
    if (e != cudaSuccess) {
        err = std::string("mimo mtp: cudaSetDevice: ") + cudaGetErrorString(e);
        return false;
    }

    n_embd_ = g.n_embd;
    n_vocab_ = g.n_vocab;
    inter_ = g.dense_intermediate;
    capacity_ = capacity;
    max_rows_ = max_rows;
    committed_ = 0;
    n_heads_ = (int) g.n_mtp_layers;
    n_q_ = (int) g.n_q_head;
    n_kv_ = (int) g.n_head_kv_swa;
    n_rot_ = (int) g.n_rot;
    window_ = (int) g.sliding_window;
    hd_qk_ = (int) g.head_dim_qk;
    hd_v_ = (int) g.head_dim_v;
    theta_ = g.rope_theta_swa;
    eps_ = g.rms_eps;
    sm_scale_ = 1.0 / std::sqrt((double) g.head_dim_qk);
    device_ = device;
    stream_ = stream;
    pack_ = pack;

    head_kv_stride_ = capacity_ * n_kv_ * hd_qk_;   // the V plane's rows are shorter; sized by K
    plane_kv_stride_ = head_kv_stride_ * n_heads_;

    auto alloc = [&](void** p, size_t bytes) -> bool {
        const cudaError_t ae = cudaMalloc(p, bytes);
        if (ae != cudaSuccess) {
            err = std::string("mimo mtp: alloc of ") + std::to_string(bytes) + " bytes: " + cudaGetErrorString(ae);
            return false;
        }
        owned_.push_back(*p);
        vram_ += bytes;
        return true;
    };

    const int64_t n = n_embd_, R = max_rows_;
    if (!alloc((void**) &kv_, (size_t) 2 * plane_kv_stride_ * 2) ||                                             // F16
        !alloc((void**) &eh32_, (size_t) R * 2 * n * 4) || !alloc((void**) &eh16_, (size_t) R * 2 * n * 2) ||
        !alloc((void**) &x32_, (size_t) R * n * 4) || !alloc((void**) &t32_, (size_t) R * n * 4) ||
        !alloc((void**) &t16_, (size_t) R * n * 2) ||
        !alloc((void**) &q32_, (size_t) R * n_q_ * hd_qk_ * 4) || !alloc((void**) &q16_, (size_t) R * n_q_ * hd_qk_ * 2) ||
        !alloc((void**) &k32_, (size_t) R * n_kv_ * hd_qk_ * 4) || !alloc((void**) &k16_, (size_t) R * n_kv_ * hd_qk_ * 2) ||
        !alloc((void**) &v32_, (size_t) R * n_kv_ * hd_qk_ * 4) || !alloc((void**) &v16_, (size_t) R * n_kv_ * hd_qk_ * 2) ||
        !alloc((void**) &attn32_, (size_t) R * n_q_ * hd_v_ * 4) || !alloc((void**) &attn16_, (size_t) R * n_q_ * hd_v_ * 2) ||
        !alloc((void**) &g32_, (size_t) R * inter_ * 4) || !alloc((void**) &u32_, (size_t) R * inter_ * 4) ||
        !alloc((void**) &act16_, (size_t) R * inter_ * 2) || !alloc((void**) &head_out_, (size_t) n_heads_ * R * n * 4) ||
        !alloc((void**) &logits_, (size_t) n_vocab_ * 4) || !alloc((void**) &d_pos_, (size_t) R * 8) ||
        !alloc((void**) &d_len_, (size_t) R * 4) || !alloc((void**) &d_pick_, 3 * 4))
        return false;

    gemm_ = new prefill::Gemm();
    if (!gemm_->init(stream_, 0, err)) {
        err = "mimo mtp: " + err;
        return false;
    }
    return true;
}

void MimoMtp::rewind(int64_t len) {
    if (len < 0) len = 0;
    if (len > capacity_) len = capacity_;
    committed_ = len;
}

// --------------------------------------------------------------------------- forward

bool MimoMtp::proj(const void* w, const uint16_t* x16, float* y32, int rows, int n_out, int n_in,
                   std::string& err) {
    if (rows == 1) {
        mtp_gemv_k<<<(unsigned) n_out, kThreads, 0, (cudaStream_t) stream_>>>((const uint16_t*) w, x16, y32, n_in);
        return launch_ok("gemv", err);
    }
    gemm_->f16(x16, (const uint16_t*) w, y32, rows, n_out, n_in);
    return true;
}

bool MimoMtp::head_forward(int head, const int32_t* tokens, const float* hidden_rows, int rows, int64_t pos0,
                           bool kv_only, std::string& err) {
    if (head < 0 || head >= n_heads_ || !tokens || !hidden_rows || rows < 1 || rows > (int) max_rows_) {
        err = "mimo mtp: head_forward: bad head / pointers / row count";
        return false;
    }
    if (pos0 < 0 || pos0 + rows > capacity_) {
        err = "mimo mtp: head_forward: positions [" + std::to_string(pos0) + ", " + std::to_string(pos0 + rows) +
              ") leave the KV capacity " + std::to_string(capacity_);
        return false;
    }
    const MimoPack::MtpHead& H = pack_->mtp[head];
    const cudaStream_t st = (cudaStream_t) stream_;
    const int64_t n = n_embd_;
    const int64_t elems = (int64_t) rows * n;

    uint16_t* k_cache = kv_ + (size_t) head * head_kv_stride_;
    uint16_t* v_cache = kv_ + plane_kv_stride_ + (size_t) head * head_kv_stride_;   // V plane, K-sized stride

    // positions first (rope, the append and the decode attention all read them from the device)
    mtp_pos_len_k<<<1, kThreads, 0, st>>>(d_pos_, d_len_, pos0, rows);
    if (!launch_ok("pos/len", err)) return false;

    // x = eh_proj([enorm(embed(tok)); hnorm(hidden)]) - embedding FIRST, each half RMSNorm'd with the
    // geometry eps, concat width 2*n_embd
    mtp_eh_input_k<<<(unsigned) rows, kThreads, 0, st>>>((const uint16_t*) pack_->embed, tokens, hidden_rows, eh32_,
                                                         (int) n);
    prefill::rms_rows(eh32_, (const float*) H.enorm, rows, n, 2 * n, (float) eps_, stream_);
    prefill::rms_rows(eh32_ + n, (const float*) H.hnorm, rows, n, 2 * n, (float) eps_, stream_);
    kernels::f32_to_f16_bulk(eh32_, eh16_, 2 * elems, stream_);
    if (!proj(H.eh, eh16_, x32_, rows, (int) n, (int) (2 * n), err)) return false;

    // h = norm_attn(x); k/v from h, rope k at the absolute positions, append to the head's cache
    if (cudaMemcpyAsync(t32_, x32_, (size_t) elems * 4, cudaMemcpyDeviceToDevice, st) != cudaSuccess ||
        !launch_ok("residual copy", err))
        return false;
    kernels::rms_norm_weighted(t32_, (const float*) H.norm_attn, rows, n, (float) eps_, stream_);
    kernels::f32_to_f16_bulk(t32_, t16_, elems, stream_);

    if (!kv_only) {
        if (!proj(H.q, t16_, q32_, rows, n_q_ * hd_qk_, (int) n, err)) return false;
        kernels::f32_to_f16_bulk(q32_, q16_, (int64_t) rows * n_q_ * hd_qk_, stream_);
        kernels::mimo_rope_f16(q16_, d_pos_, rows, n_q_, hd_qk_, n_rot_, theta_, stream_);
    }
    if (!proj(H.k, t16_, k32_, rows, n_kv_ * hd_qk_, (int) n, err)) return false;
    kernels::f32_to_f16_bulk(k32_, k16_, (int64_t) rows * n_kv_ * hd_qk_, stream_);
    kernels::mimo_rope_f16(k16_, d_pos_, rows, n_kv_, hd_qk_, n_rot_, theta_, stream_);
    if (!proj(H.v, t16_, v32_, rows, n_kv_ * hd_qk_, (int) n, err)) return false;
    kernels::f32_to_f16_bulk(v32_, v16_, (int64_t) rows * n_kv_ * hd_qk_, stream_);

    {
        const int64_t per = (int64_t) n_kv_ * hd_qk_;
        const int64_t total = (int64_t) rows * per;
        const unsigned blocks = (unsigned) ((total + kThreads - 1) / kThreads);
        mtp_kv_append_k<<<blocks, kThreads, 0, st>>>(k16_, v16_, k_cache, v_cache, d_pos_, rows, n_kv_, hd_qk_,
                                                     hd_v_);
        if (!launch_ok("kv append", err)) return false;
    }
    if (kv_only) return true;

    // attention over the head's own cache: decode (1 row, its own position/length) or prefill
    // (consecutive rows, causal inside the chunk); sink raw when the head has one
    if (rows == 1)
        kernels::mimo_attn_decode(q16_, k_cache, v_cache, attn32_, d_pos_, d_len_, 1, n_q_, n_kv_, hd_qk_, hd_v_,
                                  (int) capacity_, window_, H.sink, sm_scale_, stream_);
    else
        kernels::mimo_attn_prefill(q16_, k_cache, v_cache, attn32_, pos0, rows, n_q_, n_kv_, hd_qk_, hd_v_,
                                   (int) capacity_, window_, H.sink, sm_scale_, stream_);
    if (!launch_ok("attention", err)) return false;

    // x += o_proj(attn) and the dense SwiGLU MLP, F32 residuals throughout
    kernels::f32_to_f16_bulk(attn32_, attn16_, (int64_t) rows * n_q_ * hd_v_, stream_);
    if (!proj(H.o, attn16_, t32_, rows, (int) n, n_q_ * hd_v_, err)) return false;
    kernels::add_inplace(x32_, t32_, elems, stream_);

    if (cudaMemcpyAsync(t32_, x32_, (size_t) elems * 4, cudaMemcpyDeviceToDevice, st) != cudaSuccess ||
        !launch_ok("mlp norm copy", err))
        return false;
    kernels::rms_norm_weighted(t32_, (const float*) H.norm_mlp, rows, n, (float) eps_, stream_);
    kernels::f32_to_f16_bulk(t32_, t16_, elems, stream_);
    if (!proj(H.mlp_gate, t16_, g32_, rows, (int) inter_, (int) n, err)) return false;
    if (!proj(H.mlp_up, t16_, u32_, rows, (int) inter_, (int) n, err)) return false;
    {
        const int64_t total = (int64_t) rows * inter_;
        const unsigned blocks = (unsigned) ((total + 255) / 256);
        mtp_swiglu_k<<<blocks, 256, 0, st>>>(g32_, u32_, act16_, total);
        if (!launch_ok("swiglu", err)) return false;
    }
    if (!proj(H.mlp_down, act16_, t32_, rows, (int) n, (int) inter_, err)) return false;
    kernels::add_inplace(x32_, t32_, elems, stream_);

    // the head's output state: final_layernorm(x), kept per head for the lm_head
    float* out = head_out_ + (size_t) head * max_rows_ * n;
    if (cudaMemcpyAsync(out, x32_, (size_t) elems * 4, cudaMemcpyDeviceToDevice, st) != cudaSuccess ||
        !launch_ok("state copy", err))
        return false;
    kernels::rms_norm_weighted(out, (const float*) H.norm_final, rows, n, (float) eps_, stream_);
    return true;
}

// --------------------------------------------------------------------------- the protocol

bool MimoMtp::catch_up(const int32_t* tokens, const float* hidden_rows, int n, int64_t pos0, std::string& err) {
    if (n == 0) return true;
    if (!tokens || !hidden_rows || n < 0) {
        err = "mimo mtp: catch_up: bad pointers / count";
        return false;
    }
    if (pos0 != committed_) {
        err = "mimo mtp: catch_up: pos0 " + std::to_string(pos0) + " != committed " + std::to_string(committed_) +
              " (cells append in order; rewind() moves the cursor)";
        return false;
    }
    for (int h = 0; h < n_heads_; ++h)
        for (int64_t off = 0; off < n; off += max_rows_) {
            const int rows = (int) std::min<int64_t>(max_rows_, n - off);
            if (!head_forward(h, tokens + off, hidden_rows + (size_t) off * n_embd_, rows, pos0 + off,
                              /*kv_only =*/true, err))
                return false;
        }
    committed_ = pos0 + n;
    return true;
}

bool MimoMtp::draft(const int32_t* anchor_token, const float* anchor_hidden, int64_t anchor_pos,
                    int32_t out_ids[3], std::string& err) {
    if (!anchor_token || !anchor_hidden || !out_ids) {
        err = "mimo mtp: draft: null arguments";
        return false;
    }
    if (anchor_pos != committed_ - 1) {
        err = "mimo mtp: draft: anchor position " + std::to_string(anchor_pos) + " is not the last committed cell (" +
              std::to_string(committed_ - 1) + ")";
        return false;
    }
    if (anchor_pos + n_heads_ >= capacity_) {
        err = "mimo mtp: draft: the chain would leave the KV capacity";
        return false;
    }
    const int32_t* tok = anchor_token;
    for (int k = 0; k < n_heads_; ++k) {
        // step k: heads k..2 run the token; head k fully (its pick feeds the next step), the deeper
        // ones K/V-only so every head's cache keeps the whole chain
        for (int h = k; h < n_heads_; ++h)
            if (!head_forward(h, tok, anchor_hidden, 1, anchor_pos + k, h != k, err)) return false;
        // the shared lm_head over head k's state, argmax (ties to the lowest id): sample_tokens with
        // greedy parameters is the tree's cross-backend pick (argmax_rows is a CUDA-only shape)
        kernels::f32_to_f16_bulk(head_state(k), t16_, n_embd_, stream_);
        if (!proj(pack_->lm_head, t16_, logits_, 1, (int) n_vocab_, (int) n_embd_, err)) return false;
        kernels::SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        kernels::sample_tokens(logits_, 1, (int) n_vocab_, nullptr, 0, sp, (int*) (d_pick_ + k), stream_);
        if (!launch_ok("greedy pick", err)) return false;
        tok = d_pick_ + k;
    }
    if (cudaStreamSynchronize((cudaStream_t) stream_) != cudaSuccess ||
        cudaMemcpy(out_ids, d_pick_, 3 * sizeof(int32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = std::string("mimo mtp: draft: readback: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

}  // namespace strata::mimo
