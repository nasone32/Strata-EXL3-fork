// src/mimo/mimo_dflash.cu - see include/strata/mimo/dflash.hpp.
//
// The DFlash drafter in three layers:
//
//   from_dir/dflash_load   the config.json reader (mimo_json, the same tiny parser MiMo's own
//                          geometry uses) and the BF16 -> F16 loader (safetensors.hpp, mmap);
//                          the o_proj x attention_value_scale fold happens here, exactly once.
//   the kernels            correctness-first SIMT, no tensor cores: an m-row F16 GEMV/GEMM
//                          (F32 accumulate, one output row per block so a weight row is read
//                          from VRAM once and the m tiles hit L2), a width-generic RMSNorm, a
//                          per-head RMSNorm (q_norm/k_norm - BEFORE rope, which is rope.cu's
//                          order), silu*up, conversions and the residual add.
//   the drafter            the ring bookkeeping (exllamav3's SWAState semantics: page-granular
//                          front drop, draft appends speculatively, commit overwrites from
//                          taps) plus the draft round, which calls kernels::mimo_attn_block
//                          for the bidirectional window attention with sinks.
//
// Everything runs on the default stream; positions/lengths handed to the attention kernel are
// DEVICE buffers built here (small H2D copies per round, measured away later if they matter).
#include "strata/mimo/dflash.hpp"

#include "strata/artifact/safetensors.hpp"
#include "strata/core/mimo_geometry.hpp"   // mimo_json, the tiny config reader
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/cpu/exl3.hpp"     // exl3_reconstruct_weight (the shared decode)
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/mimo_attn.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::mimo {
namespace {

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_bf16;
using strata::kernels::f32_from_f16;

constexpr int kThreads = 128;   // gemm/reduction width
constexpr int kNormThreads = 256;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// --------------------------------------------------------------------------- gemm / elementwise

// y[m][n_out] F32 = x[m][n_in] F16 @ w[n_out][n_in]^T.  Grid (ceil(m/8), n_out): the m tiles of
// one output row are consecutive in the schedule, so a weight row streams from VRAM once and the
// remaining tiles read it through L2.  Eight m rows per block; rows past m are skipped.
__global__ void df_gemm_kernel(const uint16_t* __restrict__ x, int64_t n_in,
                               const uint16_t* __restrict__ w, float* __restrict__ y,
                               int64_t n_out, int m) {
    const int mt0 = blockIdx.x * 8, o = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const uint16_t* wr = w + (int64_t) o * n_in;
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.0f;
    for (int64_t i = t; i < n_in; i += kThreads) {
        const float wv = f32_from_f16(wr[i]);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if (mt0 + j < m) acc[j] = fmaf(f32_from_f16(x[(int64_t) (mt0 + j) * n_in + i]), wv, acc[j]);
    }
    __shared__ float red[8][kThreads / 32];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float s = warp_sum(acc[j]);   // every lane shuffles; lane 0 keeps the result
        if (lane == 0) red[j][warp] = s;
    }
    __syncthreads();
    if (t == 0) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (mt0 + j >= m) continue;
            float s = 0.0f;
            for (int wq = 0; wq < kThreads / 32; ++wq) s += red[j][wq];
            y[(int64_t) (mt0 + j) * n_out + o] = s;
        }
    }
}

// The fc projection over the TAPS ARRAY: y[m][n_out] = sum_t taps[t][m][th] . w[o][t*th + i].
// taps is a DEVICE array of n_taps DEVICE pointers, each [m][th] F16 - the engine exports each
// tapped target layer's states separately, so nothing is concatenated on the way in.
__global__ void df_gemm_taps_kernel(const uint16_t* const* __restrict__ taps, int64_t th,
                                    int n_taps, const uint16_t* __restrict__ w,
                                    float* __restrict__ y, int64_t n_out, int m, int64_t skip_rows) {
    const int mt0 = blockIdx.x * 8, o = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const uint16_t* wr = w + (int64_t) o * (n_taps * th);
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.0f;
    for (int tap = 0; tap < n_taps; ++tap) {
        const uint16_t* xr = taps[tap] + skip_rows * th;
        const uint16_t* wtr = wr + (int64_t) tap * th;
        for (int64_t i = t; i < th; i += kThreads) {
            const float wv = f32_from_f16(wtr[i]);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                if (mt0 + j < m)
                    acc[j] = fmaf(f32_from_f16(xr[(int64_t) (mt0 + j) * th + i]), wv, acc[j]);
        }
    }
    __shared__ float red[8][kThreads / 32];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float s = warp_sum(acc[j]);   // every lane shuffles; lane 0 keeps the result
        if (lane == 0) red[j][warp] = s;
    }
    __syncthreads();
    if (t == 0) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (mt0 + j >= m) continue;
            float s = 0.0f;
            for (int wq = 0; wq < kThreads / 32; ++wq) s += red[j][wq];
            y[(int64_t) (mt0 + j) * n_out + o] = s;
        }
    }
}

void launch_gemm(const uint16_t* x, int64_t n_in, const uint16_t* w, float* y, int64_t n_out,
                 int m, void* stream) {
    const dim3 grid((unsigned) ((m + 7) / 8), (unsigned) n_out);
    df_gemm_kernel<<<grid, kThreads, 0, (cudaStream_t) stream>>>(x, n_in, w, y, n_out, m);
}

// y = w * x / rms(x): F16 in/out, F32 math, one block per row.  IN-PLACE SAFE (x == y): the
// sum pass reads the whole row before the write pass touches it.
__global__ void df_rmsnorm_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                                  uint16_t* __restrict__ y, int64_t n, float eps) {
    const int64_t row = blockIdx.x;
    const uint16_t* xr = x + row * n;
    uint16_t* yr = y + row * n;
    float ss = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = f32_from_f16(xr[i]);
        ss = fmaf(v, v, ss);
    }
    __shared__ float red[kNormThreads / 32];
    float s = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    __shared__ float inv;
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int i = 0; i < (int) (blockDim.x / 32); ++i) tot += red[i];
        inv = rsqrtf(tot / (float) n + eps);
    }
    __syncthreads();
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
        yr[i] = f16_from_f32(f32_from_f16(xr[i]) * inv * f32_from_f16(w[i]));
}

// Per-head RMSNorm IN PLACE over q/k rows [m][n_heads][head_dim] with weight [head_dim].  This
// is the drafter's q_norm/k_norm, and it runs BEFORE rope (rope.cu: load_head; apply_norm();
// apply_rope();) - the parity test pins that order against a swapped reference.
__global__ void df_headnorm_kernel(uint16_t* __restrict__ qk, int n_heads, int head_dim,
                                   const uint16_t* __restrict__ w, float eps) {
    (void) n_heads;   // the grid already encodes it; kept for the call-site symmetry
    const int64_t rh = blockIdx.x;   // row * n_heads + head
    uint16_t* h = qk + rh * head_dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = f32_from_f16(h[i]);
        ss = fmaf(v, v, ss);
    }
    __shared__ float red[32];
    float s = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    __shared__ float inv;
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int i = 0; i < (int) (blockDim.x / 32); ++i) tot += red[i];
        inv = rsqrtf(tot / (float) head_dim + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        h[i] = f16_from_f32(f32_from_f16(h[i]) * inv * f32_from_f16(w[i]));
}

__global__ void df_f16_to_f32_kernel(const uint16_t* __restrict__ x, float* __restrict__ y,
                                     int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f32_from_f16(x[i]);
}

__global__ void df_f32_to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y,
                                     int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f16_from_f32(x[i]);
}

__global__ void df_residual_add_kernel(float* __restrict__ x, const float* __restrict__ y,
                                       int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// act = silu(gate) * up, F32 math, F16 store (the reference's GatedMLP with interm_dtype half)
__global__ void df_silu_mul_kernel(const float* __restrict__ g, const float* __restrict__ u,
                                   uint16_t* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = g[i];
    y[i] = f16_from_f32(gv / (1.0f + __expf(-gv)) * u[i]);
}

// The draft block's rows 1..block-1 all carry the SAME mask_embedding vector (the reference
// embeds the whole block with the target's table, then overwrites the trailing block_size - 1
// rows with mask_embedding - only row 0's embed survives, which the hook wrote already).
__global__ void df_mask_rows_kernel(const uint16_t* __restrict__ mask_row,
                                    uint16_t* __restrict__ blk, int mask_rows, int64_t hidden) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = (int64_t) mask_rows * hidden;
    if (i >= n) return;
    blk[hidden + i] = mask_row[i % hidden];
}

}  // namespace

// ------------------------------------------------------------------------------- config from dir

DflashConfig from_dir(const std::string& dir, std::string* error) {
    using namespace strata::core::mimo_json;
    DflashConfig c;
    const auto fail = [&](const std::string& what) {
        if (error) *error = "dflash " + dir + ": " + what;
        c.taps.clear();   // a failed parse must not look usable
        return c;
    };
    FILE* f = std::fopen((dir + "/config.json").c_str(), "rb");
    if (!f) return fail("cannot open config.json");
    std::string text;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
    std::fclose(f);
    if (text.size() > 8u * 1024 * 1024) return fail("config.json implausibly large");

    std::shared_ptr<Value> doc;
    try {
        Parser p(text);
        doc = p.document();
    } catch (const std::exception& e) {
        return fail(std::string("config.json: ") + e.what());
    }
    if (!doc || doc->type != Value::Type::Object) return fail("config.json is not an object");

    // It must say it is a DFlash drafter (architectures carries DFlashDraftModel; model_type is
    // the target's, qwen3, so that key says nothing here).
    const Value* archs = doc->find("architectures");
    bool is_dflash = false;
    if (archs && archs->type == Value::Type::Array) {
        for (const auto& a : archs->arr)
            if (a->type == Value::Type::String && a->str == "DFlashDraftModel") is_dflash = true;
    }
    if (!is_dflash) return fail("architectures does not list DFlashDraftModel");

    // dflash_config sub-object plus the top-level fallbacks later releases use
    const Value* dc = doc->find("dflash_config");
    auto cfg_num = [&](const char* key, double& out) -> bool {
        const Value* v = dc ? dc->find(key) : nullptr;
        if (!v) v = doc->find(key);
        if (!v || v->type != Value::Type::Number) return false;
        out = v->num;
        return true;
    };
    auto cfg_int = [&](const char* key, int64_t& out) -> bool {
        double d;
        if (!cfg_num(key, d)) return false;
        out = (int64_t) std::llround(d);
        return true;
    };

    int64_t v64 = 0;
    double dv = 0.0;
    if (get_int(doc.get(), "hidden_size", v64)) c.hidden = (int) v64;
    if (get_int(doc.get(), "intermediate_size", v64)) c.inter = (int) v64;
    if (get_int(doc.get(), "num_hidden_layers", v64)) c.n_layers = (int) v64;
    if (get_int(doc.get(), "num_attention_heads", v64)) c.n_q_head = (int) v64;
    if (get_int(doc.get(), "num_key_value_heads", v64)) c.n_kv_head = (int) v64;
    if (get_int(doc.get(), "head_dim", v64)) c.head_dim_qk = (int) v64;
    if (get_int(doc.get(), "v_head_dim", v64)) c.head_dim_v = (int) v64;
    double prf = 0.5;   // the drafter's own default; the real config pins 0.5 explicitly too
    if (get_number(doc.get(), "partial_rotary_factor", prf)) c.n_rot = (int) (c.head_dim_qk * prf);
    if (get_number(doc.get(), "rope_theta", dv)) c.rope_theta = dv;
    if (get_number(doc.get(), "rms_norm_eps", dv)) c.rms_eps = dv;
    if (!cfg_int("block_size", v64)) v64 = 8;
    c.block = (int) v64;
    if (get_int(doc.get(), "sliding_window", v64)) c.window = (int) v64;
    if (!get_bool(doc.get(), "is_causal", c.causal) && !(dc && get_bool(dc, "is_causal", c.causal)))
        return fail("is_causal must be pinned (false for the bidirectional drafter)");
    if (c.causal) return fail("is_causal true is not implemented (the MiMo drafter is bidirectional)");
    if (cfg_int("mask_token_id", v64)) c.mask_token = (int) v64;
    if (cfg_num("attention_value_scale", dv)) c.value_scale = dv;
    // sinks: a bool under dflash_config / top level, else add_swa_attention_sink_bias (the real
    // config's spelling), else present (the dflash family ships sink biases)
    bool sinks = true;
    const Value* sv = dc ? dc->find("attention_sink_bias") : nullptr;
    if (!sv) sv = doc->find("attention_sink_bias");
    if (!sv) sv = doc->find("add_swa_attention_sink_bias");
    if (sv && sv->type == Value::Type::Bool) sinks = sv->b;
    c.has_sinks = sinks;
    if (get_int(doc.get(), "target_hidden_size", v64)) c.target_hidden = (int) v64;

    // taps: the checkpoint's target_layer_ids (+ tap_shift, exllamav3's default 1 - the real
    // config pins 0).  The list order is the concat order the fc weight expects.
    const Value* ids = dc ? dc->find("target_layer_ids") : nullptr;
    if (!ids) ids = doc->find("target_layer_ids");
    if (!ids || ids->type != Value::Type::Array) return fail("target_layer_ids missing");
    int64_t shift = 1;
    cfg_int("tap_shift", shift);
    c.taps.clear();
    for (const auto& e : ids->arr) {
        if (e->type != Value::Type::Number) return fail("target_layer_ids is not a list of ints");
        c.taps.push_back((int) e->num + (int) shift);
    }
    if (c.taps.empty()) return fail("target_layer_ids is empty");

    // every layer must be sliding_attention: the ring, the window and the block kernel are
    // built for windowed layers only
    const Value* lts = doc->find("layer_types");
    if (lts && lts->type == Value::Type::Array) {
        if ((int) lts->arr.size() != c.n_layers) return fail("layer_types length != num_hidden_layers");
        for (const auto& e : lts->arr)
            if (e->type != Value::Type::String || e->str != "sliding_attention")
                return fail("only all-sliding_attention drafter layers are supported");
    }

    // strata-only knobs for the scaled parity fixture (the real config carries neither)
    if (cfg_int("dflash_ring_page", v64)) c.page = (int) v64;
    if (cfg_int("dflash_ring", v64)) c.ring = (int) v64;
    c.window_right = c.causal ? 0 : c.window - 1;

    const int group = c.n_q_head / c.n_kv_head;
    if (c.n_layers < 1 || c.hidden < 1 || c.inter < 1 || c.block < 2 || c.window < 2 ||
        c.head_dim_qk < 8 || c.head_dim_qk % 8 != 0 || c.head_dim_qk > 256 ||
        c.head_dim_v != c.head_dim_qk ||   // o_proj width; the drafter has no asymmetric v head
        c.n_rot % 2 != 0 || c.n_rot > c.head_dim_qk || c.n_q_head % c.n_kv_head != 0 ||
        (group != 1 && group != 8) || c.page < 1 || !(c.rms_eps > 0.0) || !(c.rope_theta > 0.0) ||
        !(c.value_scale > 0.0) || c.target_hidden < 1)
        return fail("config geometry is outside what the drafter implements (see DflashConfig)");
    return c;
}

// ------------------------------------------------------------------------------- weight loading

namespace {

uint16_t* reconstruct_gpu(const SafetensorsFile& file, const std::string& base,
                           int64_t expected_out, int64_t expected_in);

__global__ void df_quantize_q8(const uint16_t* weights, uint8_t* blocks, int64_t count) {
    const int64_t b = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= count) return;
    float maximum = 0;
    for (int i = 0; i < 32; ++i) maximum = fmaxf(maximum, fabsf(strata::kernels::f32_from_f16(weights[b * 32 + i])));
    const float d = maximum / 127.f, inv = d ? 1.f / d : 0;
    auto* output = blocks + b * 34;
    *(uint16_t*) output = strata::kernels::f16_from_f32(d);
    for (int i = 0; i < 32; ++i)
        ((int8_t*) (output + 2))[i] = (int8_t) roundf(strata::kernels::f32_from_f16(weights[b * 32 + i]) * inv);
}

// Reads one BF16 (or F16) tensor, checks it is [rows, cols] (or [cols] when rows == 1),
// converts to F16 (x scale), uploads.  Throws std::runtime_error naming the tensor on failure.
uint16_t* upload_bf16(const SafetensorsFile& f, const std::string& name, int64_t rows, int64_t cols,
                      double scale) {
    const StTensor* t = f.find(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    if (t->shape.size() == 2) {
        if (t->shape[0] != rows || t->shape[1] != cols)
            throw std::runtime_error("tensor " + name + " is [" + std::to_string(t->shape[0]) + "," +
                                     std::to_string(t->shape[1]) + "], expected [" +
                                     std::to_string(rows) + "," + std::to_string(cols) + "]");
    } else if (t->shape.size() == 1) {
        if (rows != 1 || t->shape[0] != cols)
            throw std::runtime_error("tensor " + name + " is [" + std::to_string(t->shape[0]) +
                                     "], expected [" + std::to_string(cols) + "]");
    } else {
        throw std::runtime_error("tensor " + name + " is not 1-D or 2-D");
    }
    const size_t n = t->nelem();
    std::vector<uint16_t> host(n);
    const uint16_t* raw = reinterpret_cast<const uint16_t*>(f.data(*t));
    if (t->dtype == "BF16") {
        for (size_t i = 0; i < n; ++i) {
            const float v = f32_from_bf16(raw[i]);
            host[i] = f16_from_f32(scale == 1.0 ? v : (float) (v * scale));
        }
    } else if (t->dtype == "F16") {
        if (scale == 1.0) std::memcpy(host.data(), raw, n * 2);
        else for (size_t i = 0; i < n; ++i) host[i] = f16_from_f32((float) (f32_from_f16(raw[i]) * scale));
    } else {
        throw std::runtime_error("tensor " + name + " is " + t->dtype + ", expected BF16");
    }
    void* d = nullptr;
    const cudaError_t e = cudaMalloc(&d, n * 2);
    if (e != cudaSuccess) throw std::runtime_error("cudaMalloc for " + name + ": " + cudaGetErrorString(e));
    const cudaError_t e2 = cudaMemcpy(d, host.data(), n * 2, cudaMemcpyHostToDevice);
    if (e2 != cudaSuccess) {
        cudaFree(d);
        throw std::runtime_error("upload of " + name + ": " + cudaGetErrorString(e2));
    }
    return (uint16_t*) d;
}

float* upload_sinks(const SafetensorsFile& f, const std::string& name, int64_t n_expect) {
    const StTensor* t = f.find(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    if (t->shape.size() != 1 || t->shape[0] != n_expect)
        throw std::runtime_error("tensor " + name + " is not [" + std::to_string(n_expect) + "]");
    std::vector<float> host(t->nelem());
    const uint16_t* raw = reinterpret_cast<const uint16_t*>(f.data(*t));
    if (t->dtype == "BF16")
        for (size_t i = 0; i < host.size(); ++i) host[i] = f32_from_bf16(raw[i]);
    else if (t->dtype == "F16")
        for (size_t i = 0; i < host.size(); ++i) host[i] = f32_from_f16(raw[i]);
    else
        throw std::runtime_error("tensor " + name + " is " + t->dtype + ", expected BF16");
    void* d = nullptr;
    if (cudaMalloc(&d, host.size() * 4) != cudaSuccess ||
        cudaMemcpy(d, host.data(), host.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("upload of " + name + " failed");
    return (float*) d;
}

// ------------------------------------------------------------- EXL3 (the quantized dflash/ dir)
//
// The dflash dir is a SINGLE safetensors file (no index.json), so Exl3Model's multi-shard
// accessor does not apply; this resolves one linear out of the file with the SAME validation
// Exl3Model::linear does, and the decode itself is the shared kernels::cpu path mimo_pack
// calls - no trellis math lives here.

struct DfExl3 {
    const uint16_t* trellis = nullptr;
    const uint16_t* suh = nullptr;
    const uint16_t* svh = nullptr;
    int ki = 0, nj = 0, bits = 0, frac = 0, cb = 2;   // cb 2 = mul1
    int in() const { return ki * 16; }
    int out() const { return nj * 16; }
    int words() const { return 16 * bits + 8 * frac; }
};

DfExl3 df_exl3_linear(const SafetensorsFile& f, const std::string& base) {
    const StTensor* tr = f.find(base + ".trellis");
    if (!tr) throw std::runtime_error("missing tensor " + base + ".trellis");
    const StTensor* suh = f.find(base + ".suh");
    const StTensor* svh = f.find(base + ".svh");
    if (!suh || !svh) throw std::runtime_error("missing " + base + ".suh/.svh");
    if (tr->dtype != "I16" || tr->shape.size() != 3 || tr->shape[0] <= 0 || tr->shape[1] <= 0 ||
        tr->shape[0] % 8 != 0 || tr->shape[1] % 8 != 0 ||
        tr->shape[2] < 16 || tr->shape[2] > 128 || tr->shape[2] % 8 != 0)
        throw std::runtime_error(base + ".trellis is not I16 [in/16,out/16,16*K], K in 0.5 steps");
    if (suh->dtype != "F16" || svh->dtype != "F16" ||
        suh->shape.size() != 1 || svh->shape.size() != 1 ||
        suh->shape[0] != tr->shape[0] * 16 || svh->shape[0] != tr->shape[1] * 16)
        throw std::runtime_error(base + " needs F16 suh[in] and svh[out]");
    DfExl3 L;
    L.trellis = reinterpret_cast<const uint16_t*>(f.data(*tr));
    L.suh = reinterpret_cast<const uint16_t*>(f.data(*suh));
    L.svh = reinterpret_cast<const uint16_t*>(f.data(*svh));
    L.ki = (int) tr->shape[0];
    L.nj = (int) tr->shape[1];
    L.bits = (int) (tr->shape[2] / 16);
    L.frac = (int) ((tr->shape[2] % 16) / 8);
    L.cb = f.find(base + ".mul1") ? 2 : f.find(base + ".mcg") ? 1 : 0;
    if (L.frac != 0 && L.cb != 2)
        throw std::runtime_error(base + ": half-integer K requires the mul1 codebook");
    return L;
}

/// Runs `fn(begin, end)` over [0, n) in equal slices on the mimo_pack band-pool convention
/// (STRATA_MIMO_THREADS, default min(hardware_concurrency, 8)).  First worker error wins.
void parallel_slices(int n, const std::function<void(int, int)>& fn) {
    const char* env = std::getenv("STRATA_MIMO_THREADS");
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int workers = env != nullptr ? std::max(1, std::atoi(env)) : (int) std::min<unsigned>(hw, 8u);
    workers = std::min(workers, std::max(1, n));
    std::atomic<int> next_err{0};
    std::string err;
    std::mutex mu;
    auto slice = [&](int w) {
        const int b = (int) ((int64_t) n * w / workers), e = (int) ((int64_t) n * (w + 1) / workers);
        try {
            fn(b, e);
        } catch (const std::exception& e2) {
            std::lock_guard<std::mutex> lk(mu);
            if (next_err.fetch_add(1) == 0) err = e2.what();
        }
    };
    std::vector<std::thread> pool;
    for (int w = 1; w < workers; ++w) pool.emplace_back(slice, w);
    slice(0);
    for (std::thread& t : pool) t.join();
    if (!err.empty()) throw std::runtime_error(err);
}

/// Reconstructs one EXL3 linear to the drafter's F16 [out, in] layout (rows contiguous), with
/// `scale` folded in (o_proj's attention_value_scale, exactly once).  exl3_reconstruct_weight
/// emits the INPUT-major [in, out] image, so this reconstructs in whole-128-row bands - one
/// Hadamard block per band, bands never share a 128-row group, exactly mimo_pack's job rule -
/// and transposes banded afterwards.  `expect_out`/`expect_in` pin the config's dimensions.
std::vector<uint16_t> reconstruct_linear(const SafetensorsFile& f, const std::string& base,
                                         int64_t expect_out, int64_t expect_in, double scale) {
    const DfExl3 L = df_exl3_linear(f, base);
    if (L.in() != expect_in || L.out() != expect_out)
        throw std::runtime_error(base + " is " + std::to_string(L.out()) + "x" + std::to_string(L.in()) +
                                 ", expected " + std::to_string(expect_out) + "x" +
                                 std::to_string(expect_in));
    // 1. decode to [in][out], bands of 8 i-tiles (128 rows) across the pool
    const int nj = L.nj, words = L.words(), n_out = L.out();
    std::vector<uint16_t> rec((size_t) L.in() * L.out());
    const int bands = (L.ki + 7) / 8;
    parallel_slices(bands, [&](int b0, int b1) {
        for (int b = b0; b < b1; ++b) {
            const int i0 = b * 8, i1 = std::min(i0 + 8, L.ki);
            strata::kernels::cpu::exl3_reconstruct_weight(
                L.trellis + (size_t) i0 * nj * words, i1 - i0, nj, L.bits,
                (strata::kernels::cpu::Exl3Codebook) L.cb, L.suh + (size_t) i0 * 16, L.svh,
                rec.data() + (size_t) i0 * 16 * n_out, L.frac);
        }
    });
    // 2. the fold (o_proj only), then the transpose to [out][in], both sliced across the pool
    if (scale != 1.0)
        parallel_slices((int) rec.size(), [&](int b, int e) {
            for (int i = b; i < e; ++i)
                rec[(size_t) i] = f16_from_f32(f32_from_f16(rec[(size_t) i]) * (float) scale);
        });
    std::vector<uint16_t> w((size_t) expect_out * expect_in);
    const int64_t in_n = expect_in, out_n = expect_out;
    // 32x32 tiles keep both sides of the transpose inside cache lines
    const int64_t tile = 32;
    const int64_t tiles = ((in_n + tile - 1) / tile) * ((out_n + tile - 1) / tile);
    parallel_slices((int) tiles, [&](int b, int e) {
        for (int t = b; t < e; ++t) {
            const int64_t tx = (t % ((in_n + tile - 1) / tile)) * tile;
            const int64_t ty = (t / ((in_n + tile - 1) / tile)) * tile;
            const int64_t xe = std::min(tx + tile, in_n), ye = std::min(ty + tile, out_n);
            for (int64_t i = tx; i < xe; ++i)
                for (int64_t o = ty; o < ye; ++o)
                    w[(size_t) o * in_n + i] = rec[(size_t) i * out_n + o];
        }
    });
    return w;
}

uint16_t* reconstruct_gpu(const SafetensorsFile& file, const std::string& base,
                           int64_t expected_out, int64_t expected_in) {
    const auto m = df_exl3_linear(file, base);
    if (m.in() != expected_in || m.out() != expected_out)
        throw std::runtime_error("dflash: GPU reconstruction geometry mismatch: " + base);
    struct Buffers {
        void* p[5] = {};
        ~Buffers() { for (auto pointer : p) if (pointer) (void) cudaFree(pointer); }
    } buffers;
    auto allocate = [&](int index, size_t bytes, const void* source = nullptr) {
        if (cudaMalloc(&buffers.p[index], bytes) != cudaSuccess ||
            (source && cudaMemcpy(buffers.p[index], source, bytes, cudaMemcpyHostToDevice) != cudaSuccess))
            throw std::runtime_error("dflash: GPU reconstruction allocation/upload failed: " + base);
        return static_cast<uint16_t*>(buffers.p[index]);
    };
    auto* tr = allocate(0, (size_t) m.ki * m.nj * m.words() * 2, m.trellis);
    auto* su = allocate(1, (size_t) m.in() * 2, m.suh);
    auto* sv = allocate(2, (size_t) m.out() * 2, m.svh);
    auto* rec = allocate(3, (size_t) m.in() * m.out() * 2);
    auto* out = allocate(4, (size_t) m.in() * m.out() * 2);
    strata::kernels::exl3_reconstruct_weight(tr, m.ki, m.nj, m.bits, m.cb, su, sv, rec, nullptr, m.frac);
    strata::kernels::exl3_transpose_f16(rec, out, m.in(), m.out(), nullptr);
    if (cudaGetLastError() != cudaSuccess || cudaStreamSynchronize(nullptr) != cudaSuccess)
        throw std::runtime_error("dflash: GPU reconstruction failed: " + base);
    buffers.p[4] = nullptr;
    return out;
}

uint16_t* upload_f16(std::vector<uint16_t>& host, const char* what) {
    void* d = nullptr;
    const cudaError_t e = cudaMalloc(&d, host.size() * 2);
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("cudaMalloc for ") + what + ": " + cudaGetErrorString(e));
    const cudaError_t e2 = cudaMemcpy(d, host.data(), host.size() * 2, cudaMemcpyHostToDevice);
    if (e2 != cudaSuccess) {
        cudaFree(d);
        throw std::runtime_error(std::string("upload of ") + what + ": " + cudaGetErrorString(e2));
    }
    return (uint16_t*) d;
}

}  // namespace

void dflash_unload(Dflash* d) {
    if (!d) return;
    if (d->layers) {
        for (int l = 0; l < d->cfg.n_layers; ++l) {
            DflashLayer& ly = d->layers[l];
            (void) cudaFree(ly.ln1); (void) cudaFree(ly.ln2); (void) cudaFree(ly.q_w);
            (void) cudaFree(ly.k_w); (void) cudaFree(ly.v_w); (void) cudaFree(ly.o_w);
            (void) cudaFree(ly.q_norm); (void) cudaFree(ly.k_norm); (void) cudaFree(ly.sink);
            (void) cudaFree(ly.gate_w); (void) cudaFree(ly.up_w); (void) cudaFree(ly.down_w);
        }
        delete[] d->layers;
    }
    (void) cudaFree(d->fc_w);
    (void) cudaFree(d->hidden_norm);
    (void) cudaFree(d->mask_embed);
    (void) cudaFree(d->final_norm);
    *d = Dflash{};
}

bool dflash_load(Dflash* d, const std::string& dir, std::string* error) {
    if (!d) return false;
    dflash_unload(d);
    try {
        std::string cfg_err;
        d->cfg = from_dir(dir, &cfg_err);
        if (!cfg_err.empty()) {
            if (error) *error = cfg_err;
            return false;
        }
        const DflashConfig& c = d->cfg;
        if (c.taps.empty()) {
            if (error) *error = dir + ": no taps";
            return false;
        }

        // dir-shape pick: the quantized dir keeps everything in ONE model.safetensors whose
        // linears are trellis; the bf16 dir ships plain tensors (and a separate mask file)
        std::string wpath = dir + "/model.safetensors";
        bool have_model = false;
        {
            FILE* probe = std::fopen(wpath.c_str(), "rb");
            if (probe) { std::fclose(probe); have_model = true; }
            else wpath = dir + "/dflash_draft_model.safetensors";
        }
        SafetensorsFile wf(wpath);
        const bool quantized = have_model && wf.find("fc.trellis") != nullptr;
        const char* gpu = std::getenv("STRATA_MIMO_DFLASH_GPU_LOAD");
        const bool gpu_load = quantized && gpu && gpu[0] == '1';
        const char* q8 = std::getenv("STRATA_MIMO_DFLASH_Q8");
        d->q8_linears = q8 && q8[0] == '1';
        if (gpu_load) std::fprintf(stderr, "dflash: EXL3 GPU reconstruction + F16 transpose\n");
        d->quantized = quantized;
        // the bf16 dir's separate mask file opens lazily (the quantized dir has none)
        std::unique_ptr<SafetensorsFile> mf;
        const auto mask_file = [&]() -> const SafetensorsFile& {
            if (!mf) mf = std::make_unique<SafetensorsFile>(dir + "/mask_embedding.safetensors");
            return *mf;
        };

        // Every linear either reconstructs from the trellis (quantized) or converts from BF16.
        // THE o_proj FOLD IS VARIANT-SPECIFIC, and the real dirs pin it: the bf16 dir's tensors
        // are raw (fold x attention_value_scale at load, like exllamav3's FP16 load path),
        // while the EXL3 dir was quantized from ALREADY-FOLDED weights (exllamav3's converter
        // quantizes its loaded model, and load_exl3 never applies weight_scale), so the
        // reconstructed o_proj must NOT be folded again - measured on the real pair: the
        // reconstruction lands at 0.610x of (bf16 x 0.612), i.e. at 0.612x of the raw bf16 row.
        const auto linear = [&](const std::string& base, int64_t out, int64_t in,
                                 double scale_bf16) -> uint16_t* {
            uint16_t* result;
            if (quantized) {
                if (gpu_load) result = reconstruct_gpu(wf, base, out, in);
                else {
                    std::vector<uint16_t> host = reconstruct_linear(wf, base, out, in, 1.0);
                    result = upload_f16(host, base.c_str());
                }
            } else result = upload_bf16(wf, base + ".weight", out, in, scale_bf16);
            if (!d->q8_linears || base == "fc") return result;
            uint8_t* blocks = nullptr;
            const int64_t n = out * in / 32;
            if (in % 32 || cudaMalloc(&blocks, (size_t) n * 34) != cudaSuccess) {
                (void) cudaFree(result); throw std::runtime_error("dflash: Q8 allocation/geometry failed");
            }
            df_quantize_q8<<<(unsigned) ((n + 255) / 256), 256>>>(result, blocks, n);
            const auto status = cudaStreamSynchronize(nullptr);
            (void) cudaFree(result);
            if (status != cudaSuccess) { (void) cudaFree(blocks); throw std::runtime_error("dflash: Q8 quantization failed"); }
            return (uint16_t*) blocks; // opaque layer weight, freed by the existing owner
        };
        // the raw tensors: BF16 (the quantized file keeps them BF16 inside model.safetensors,
        // including mask_embedding; the bf16 dir keeps mask_embedding in its own file)
        const auto raw16 = [&](const SafetensorsFile& f, const std::string& name, int64_t n) {
            return upload_bf16(f, name, 1, n, 1.0);
        };

        d->fc_w = linear("fc", c.hidden, c.fc_in(), 1.0);
        d->hidden_norm = raw16(wf, "hidden_norm.weight", c.hidden);
        d->mask_embed = quantized ? raw16(wf, "mask_embedding", c.hidden)
                                  : raw16(mask_file(), "mask_embedding", c.hidden);
        d->final_norm = raw16(wf, "norm.weight", c.hidden);
        d->layers = new DflashLayer[c.n_layers];
        for (int l = 0; l < c.n_layers; ++l) {
            const std::string p = "layers." + std::to_string(l);
            DflashLayer& ly = d->layers[l];
            ly.ln1 = raw16(wf, p + ".input_layernorm.weight", c.hidden);
            ly.ln2 = raw16(wf, p + ".post_attention_layernorm.weight", c.hidden);
            ly.q_w = linear(p + ".self_attn.q_proj", c.q_dim(), c.hidden, 1.0);
            ly.k_w = linear(p + ".self_attn.k_proj", c.kv_dim(), c.hidden, 1.0);
            ly.v_w = linear(p + ".self_attn.v_proj", c.kv_dim(), c.hidden, 1.0);
            ly.o_w = linear(p + ".self_attn.o_proj", c.hidden, c.q_dim(), c.value_scale);
            ly.q_norm = raw16(wf, p + ".self_attn.q_norm.weight", c.head_dim_qk);
            ly.k_norm = raw16(wf, p + ".self_attn.k_norm.weight", c.head_dim_qk);
            ly.sink = c.has_sinks
                ? upload_sinks(wf, p + ".self_attn.attention_sink_bias", c.n_q_head)
                : nullptr;
            ly.gate_w = linear(p + ".mlp.gate_proj", c.inter, c.hidden, 1.0);
            ly.up_w = linear(p + ".mlp.up_proj", c.inter, c.hidden, 1.0);
            ly.down_w = linear(p + ".mlp.down_proj", c.hidden, c.inter, 1.0);
        }
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) throw std::runtime_error(std::string("upload: ") + cudaGetErrorString(e));
        d->loaded = true;
        return true;
    } catch (const std::exception& e) {
        if (error) *error = std::string("dflash ") + dir + ": " + e.what();
        dflash_unload(d);
        return false;
    }
}

// ------------------------------------------------------------------------------- the drafter

namespace {

void gemm(const uint16_t* x, int64_t n_in, const uint16_t* w, float* y, int64_t n_out, int m,
           DflashDrafter* drafter) {
    if (drafter->weights_view.q8_linears) {
        const auto stream = (cudaStream_t) drafter->q8_stream;
        (void) cudaEventRecord((cudaEvent_t) drafter->q8_ready, nullptr);
        (void) cudaStreamWaitEvent(stream, (cudaEvent_t) drafter->q8_ready, 0);
        df_f16_to_f32_kernel<<<(unsigned) (((int64_t) m * n_in + 511) / 512), 512, 0, stream>>>(
            x, drafter->sc_linear_input, (int64_t) m * n_in);
        for (int r = 0; r < m; r += 8)
            strata::kernels::native_q8_0_f32(w, drafter->sc_linear_input + (int64_t) r * n_in,
                drafter->q8_scratch, y + (int64_t) r * n_out, n_in, n_out, std::min(8, m - r), stream);
        (void) cudaEventRecord((cudaEvent_t) drafter->q8_done, stream);
        (void) cudaStreamWaitEvent(nullptr, (cudaEvent_t) drafter->q8_done, 0);
        return;
    }
    launch_gemm(x, n_in, w, y, n_out, m, nullptr);
}

void gemm_taps(const uint16_t* const* taps, int64_t th, int n_taps, const uint16_t* w, float* y,
               int64_t n_out, int m, int64_t skip_rows) {
    const dim3 grid((unsigned) ((m + 7) / 8), (unsigned) n_out);
    df_gemm_taps_kernel<<<grid, kThreads>>>(taps, th, n_taps, w, y, n_out, m, skip_rows);
}

void rmsnorm(const uint16_t* x, const uint16_t* w, uint16_t* y, int m, int64_t n, float eps) {
    df_rmsnorm_kernel<<<(unsigned) m, kNormThreads>>>(x, w, y, n, eps);
}

void headnorm(uint16_t* qk, int m, int n_heads, int head_dim, const uint16_t* w, float eps) {
    df_headnorm_kernel<<<(unsigned) (m * n_heads), kThreads>>>(qk, n_heads, head_dim, w, eps);
}

void f16_to_f32(const uint16_t* x, float* y, int64_t n) {
    df_f16_to_f32_kernel<<<(unsigned) ((n + 511) / 512), 512>>>(x, y, n);
}

void f32_to_f16(const float* x, uint16_t* y, int64_t n) {
    df_f32_to_f16_kernel<<<(unsigned) ((n + 511) / 512), 512>>>(x, y, n);
}

void residual_add(float* x, const float* y, int64_t n) {
    df_residual_add_kernel<<<(unsigned) ((n + 511) / 512), 512>>>(x, y, n);
}

void silu_mul(const float* g, const float* u, uint16_t* y, int64_t n) {
    df_silu_mul_kernel<<<(unsigned) ((n + 511) / 512), 512>>>(g, u, y, n);
}

void mask_rows(const uint16_t* mask_row, uint16_t* blk, int mask_rows_n, int64_t hidden) {
    const int64_t n = (int64_t) mask_rows_n * hidden;
    df_mask_rows_kernel<<<(unsigned) ((n + 511) / 512), 512>>>(mask_row, blk, mask_rows_n, hidden);
}

}  // namespace

bool DflashDrafter::init(const Dflash& w, DflashEmbedFn embed, DflashArgmaxFn argmax,
                         void* user, int max_rows) {
    free();
    if (!w.loaded || !embed || !argmax) {
        std::fprintf(stderr, "dflash: init needs a loaded Dflash and both hooks\n");
        return false;
    }
    cfg = w.cfg;
    weights_view = w;
    embed_fn = embed;
    argmax_fn = argmax;
    hooks_user = user;
    max_commit_rows = max_rows < cfg.block ? cfg.block : max_rows;

    struct Al {
        void** p;
        size_t bytes;
    };
    const int S = cfg.ring_size();
    const int64_t wide = std::max<int64_t>({cfg.q_dim(), cfg.kv_dim(), cfg.hidden});
    const int64_t vrow = (int64_t) cfg.n_kv_head * cfg.head_dim_v;
    const std::vector<Al> allocs = {
        {(void**) &ring_k, (size_t) cfg.n_layers * S * cfg.kv_dim() * 2},
        {(void**) &ring_v, (size_t) cfg.n_layers * S * vrow * 2},
        {(void**) &sc_blk, (size_t) cfg.block * cfg.hidden * 2},
        {(void**) &sc_x, (size_t) cfg.block * cfg.hidden * 4},
        {(void**) &sc_nx, (size_t) cfg.block * cfg.hidden * 2},
        {(void**) &sc_g1, (size_t) cfg.block * wide * 4},
        {(void**) &sc_q, (size_t) cfg.block * cfg.q_dim() * 2},
        {(void**) &sc_k, (size_t) cfg.block * cfg.kv_dim() * 2},
        {(void**) &sc_v, (size_t) cfg.block * cfg.kv_dim() * 2},
        {(void**) &sc_attn, (size_t) cfg.block * (int64_t) cfg.n_q_head * cfg.head_dim_v * 4},
        {(void**) &sc_o_in, (size_t) cfg.block * (int64_t) cfg.n_q_head * cfg.head_dim_v * 2},
        {(void**) &sc_gate, (size_t) cfg.block * cfg.inter * 4},
        {(void**) &sc_up, (size_t) cfg.block * cfg.inter * 4},
        {(void**) &sc_act, (size_t) cfg.block * cfg.inter * 2},
        {(void**) &sc_pos, (size_t) std::max(cfg.block, max_commit_rows) * 8},
        {(void**) &sc_rpos, (size_t) cfg.block * 8},
        {(void**) &sc_len, (size_t) cfg.block * 4},
        {(void**) &sc_state, (size_t) cfg.block * cfg.hidden * 2},
        {(void**) &sc_cf, (size_t) max_commit_rows * std::max<int64_t>(cfg.hidden, cfg.kv_dim()) * 4},
        {(void**) &sc_ctx, (size_t) max_commit_rows * cfg.hidden * 2},
        {(void**) &sc_ckv, (size_t) max_commit_rows * cfg.kv_dim() * 2},
        {(void**) &sc_taps, (size_t) cfg.taps.size() * 8},
        {(void**) &sc_shift, (size_t) S * std::max<int64_t>(cfg.kv_dim(), vrow) * 2},
    };
    for (const Al& a : allocs) {
        const cudaError_t e = cudaMalloc(a.p, a.bytes);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "dflash: init cudaMalloc failed (%s)\n", cudaGetErrorString(e));
            free();
            return false;
        }
    }
    if (w.q8_linears) {
        if (cudaStreamCreateWithFlags((cudaStream_t*) &q8_stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags((cudaEvent_t*) &q8_ready, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags((cudaEvent_t*) &q8_done, cudaEventDisableTiming) != cudaSuccess) {
            free(); return false;
        }
        const int64_t input_width = std::max<int64_t>({cfg.hidden, cfg.inter, (int64_t) cfg.n_q_head * cfg.head_dim_v});
        if (cudaMalloc(&q8_scratch, strata::kernels::native_q8_1_bytes(input_width, 8)) != cudaSuccess ||
            cudaMalloc(&sc_linear_input, (size_t) std::max(cfg.block, max_commit_rows) * input_width * 4) != cudaSuccess) {
            free(); return false;
        }
    }
    (void) cudaMemset(ring_k, 0, (size_t) cfg.n_layers * S * cfg.kv_dim() * 2);
    (void) cudaMemset(ring_v, 0, (size_t) cfg.n_layers * S * vrow * 2);
    window_beg = 0;
    committed = 0;
    ready = true;
    return true;
}

void DflashDrafter::free() {
    if (q8_stream) (void) cudaStreamSynchronize((cudaStream_t) q8_stream);
#define DF_FREE(p) (void) cudaFree(p)
    DF_FREE(ring_k); DF_FREE(ring_v);
    DF_FREE(sc_blk); DF_FREE(sc_x); DF_FREE(sc_nx); DF_FREE(sc_g1);
    DF_FREE(sc_q); DF_FREE(sc_k); DF_FREE(sc_v); DF_FREE(sc_attn); DF_FREE(sc_o_in);
    DF_FREE(sc_gate); DF_FREE(sc_up); DF_FREE(sc_act);
    DF_FREE(sc_pos); DF_FREE(sc_rpos); DF_FREE(sc_len); DF_FREE(sc_state);
    DF_FREE(sc_cf); DF_FREE(sc_ctx); DF_FREE(sc_ckv); DF_FREE(sc_taps); DF_FREE(sc_shift);
    DF_FREE(q8_scratch); DF_FREE(sc_linear_input);
#undef DF_FREE
    ring_k = ring_v = nullptr;
    sc_blk = sc_nx = sc_q = sc_k = sc_v = sc_o_in = sc_act = sc_state = sc_ctx = sc_ckv = sc_shift = nullptr;
    sc_x = sc_g1 = sc_attn = sc_gate = sc_up = sc_cf = nullptr;
    sc_pos = sc_rpos = nullptr;
    sc_len = nullptr;
    sc_taps = nullptr;
    window_beg = committed = 0;
    max_commit_rows = 0;
    ready = false;
    q8_scratch = nullptr; sc_linear_input = nullptr;
    if (q8_stream) (void) cudaStreamDestroy((cudaStream_t) q8_stream);
    if (q8_ready) (void) cudaEventDestroy((cudaEvent_t) q8_ready);
    if (q8_done) (void) cudaEventDestroy((cudaEvent_t) q8_done);
    q8_stream = q8_ready = q8_done = nullptr;
    weights_view = Dflash{};
    embed_fn = nullptr;
    argmax_fn = nullptr;
    hooks_user = nullptr;
}

void DflashDrafter::reset() {
    window_beg = 0;
    committed = 0;
}

// Moves ring rows [shift, cache_pos) down to [0, ...) in every layer, via the staging buffer
// (the two memcpys per tensor never alias).  Caller bumps window_beg by shift.
void DflashDrafter::ring_shift_(int shift, int64_t cache_pos, std::string* err) {
    const int64_t keep = cache_pos - shift;
    const int64_t vrow = (int64_t) cfg.n_kv_head * cfg.head_dim_v;
    if (keep < 0) {
        if (err) *err = "ring shift would keep a negative row count";
        return;
    }
    if (keep == 0) return;
    for (int l = 0; l < cfg.n_layers; ++l) {
        const uint16_t* k = ring_k_layer(l);
        const uint16_t* v = ring_v_layer(l);
        const size_t kb = (size_t) keep * cfg.kv_dim() * 2, vb = (size_t) keep * vrow * 2;
        if (cudaMemcpyAsync(sc_shift, k + (size_t) shift * cfg.kv_dim(), kb, cudaMemcpyDeviceToDevice) != cudaSuccess ||
            cudaMemcpyAsync((void*) k, sc_shift, kb, cudaMemcpyDeviceToDevice) != cudaSuccess ||
            cudaMemcpyAsync(sc_shift, v + (size_t) shift * vrow, vb, cudaMemcpyDeviceToDevice) != cudaSuccess ||
            cudaMemcpyAsync((void*) v, sc_shift, vb, cudaMemcpyDeviceToDevice) != cudaSuccess) {
            if (err) *err = "ring shift memcpy failed";
            return;
        }
    }
}

bool DflashDrafter::commit_from_taps(const uint16_t* const* taps, int64_t position, int rows) {
    if (!ready) {
        std::fprintf(stderr, "dflash: commit_from_taps before init\n");
        return false;
    }
    if (!taps || rows < 1 || rows > max_commit_rows) {
        std::fprintf(stderr, "dflash: commit_from_taps bad args (rows=%d max=%d)\n", rows,
                     max_commit_rows);
        return false;
    }
    if (position < window_beg || position > committed + cfg.block) {
        std::fprintf(stderr, "dflash: commit position %lld outside the ring's [%lld, %lld]\n",
                     (long long) position, (long long) window_beg, (long long) (committed + cfg.block));
        return false;
    }
    const int S = cfg.ring_size();
    int64_t cache_pos = position - window_beg;
    // plan_state_rows: page-granular front drop when the write would pass the ring's end.
    // keep > 0 keeps what fits of the old run; keep <= 0 (a jump longer than the ring) keeps
    // nothing and drops the batch's first -keep rows entirely
    int64_t skip = 0;
    if (cache_pos + rows > S) {
        const int64_t shift = (cache_pos + rows + cfg.page - 1) / cfg.page * cfg.page - S;
        const int64_t keep = cache_pos - shift;
        if (keep > 0) {
            std::string err;
            ring_shift_((int) shift, cache_pos, &err);
            if (!err.empty()) {
                std::fprintf(stderr, "dflash: %s\n", err.c_str());
                return false;
            }
            window_beg += shift;
            cache_pos -= shift;
        } else {
            skip = -keep;
            window_beg += shift;
            cache_pos = 0;
        }
    }
    const int n = (int) (rows - skip);
    if (n <= 0) {   // nothing survives the drop; the window has moved past the whole batch
        committed = position + rows;
        return true;
    }

    if (cudaMemcpyAsync(sc_taps, taps, cfg.taps.size() * sizeof(void*), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "dflash: taps pointer upload failed\n");
        return false;
    }
    // fc over the taps, then hidden_norm, once for every layer
    gemm_taps(sc_taps, cfg.target_hidden, (int) cfg.taps.size(), weights_view.fc_w, sc_cf, cfg.hidden, n, skip);
    f32_to_f16(sc_cf, sc_ctx, (int64_t) n * cfg.hidden);
    rmsnorm(sc_ctx, weights_view.hidden_norm, sc_ctx, n, cfg.hidden, (float) cfg.rms_eps);

    std::vector<int64_t> pos_host(n);
    for (int r = 0; r < n; ++r) pos_host[r] = position + skip + r;
    if (cudaMemcpyAsync(sc_pos, pos_host.data(), (size_t) n * 8, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "dflash: position upload failed\n");
        return false;
    }
    const int64_t vrow = (int64_t) cfg.n_kv_head * cfg.head_dim_v;
    for (int l = 0; l < cfg.n_layers; ++l) {
        const DflashLayer& ly = weights_view.layers[l];
        uint16_t* kdst = ring_k_layer(l) + (size_t) cache_pos * cfg.kv_dim();
        uint16_t* vdst = ring_v_layer(l) + (size_t) cache_pos * vrow;
        gemm(sc_ctx, cfg.hidden, ly.k_w, sc_cf, cfg.kv_dim(), n, this);
        f32_to_f16(sc_cf, sc_ckv, (int64_t) n * cfg.kv_dim());
        headnorm(sc_ckv, n, cfg.n_kv_head, cfg.head_dim_qk, ly.k_norm, (float) cfg.rms_eps);
        strata::kernels::mimo_rope_f16(sc_ckv, sc_pos, n, cfg.n_kv_head, cfg.head_dim_qk, cfg.n_rot,
                                       cfg.rope_theta, nullptr);
        if (cudaMemcpyAsync(kdst, sc_ckv, (size_t) n * cfg.kv_dim() * 2, cudaMemcpyDeviceToDevice) != cudaSuccess) {
            std::fprintf(stderr, "dflash: ring K write failed\n");
            return false;
        }
        gemm(sc_ctx, cfg.hidden, ly.v_w, sc_cf, cfg.kv_dim(), n, this);
        f32_to_f16(sc_cf, sc_ckv, (int64_t) n * cfg.kv_dim());
        if (cudaMemcpyAsync(vdst, sc_ckv, (size_t) n * vrow * 2, cudaMemcpyDeviceToDevice) != cudaSuccess) {
            std::fprintf(stderr, "dflash: ring V write failed\n");
            return false;
        }
    }
    committed = position + rows;   // erases any speculative block the last draft left
    return true;
}

bool DflashDrafter::draft(int32_t carry_token_id, int64_t P, int32_t* out_ids, int block_rows) {
    if (!ready) {
        std::fprintf(stderr, "dflash: draft before init\n");
        return false;
    }
    if (P != committed) {
        std::fprintf(stderr, "dflash: draft at P=%lld but the committed position is %lld\n",
                     (long long) P, (long long) committed);
        return false;
    }
    const int b = block_rows ? block_rows : cfg.block;
    if (b < 2 || b > cfg.block || !out_ids) return false;
    const int S = cfg.ring_size();
    int64_t cache_pos = P - window_beg;
    // _decode_state_prep's shift: whole pages, enough for the block's append
    if (cache_pos + b > S) {
        const int shift = (b + cfg.page - 1) / cfg.page * cfg.page;
        std::string err;
        ring_shift_(shift, cache_pos, &err);
        if (!err.empty()) {
            std::fprintf(stderr, "dflash: %s\n", err.c_str());
            return false;
        }
        window_beg += shift;
        cache_pos -= shift;
    }
    if (cache_pos < 0) cache_pos = 0;   // cannot happen while page <= ring, kept honest

    // the block: row 0 = the target's embed of the carry (hook writes it), rows 1.. = mask
    embed_fn(carry_token_id, sc_blk, hooks_user);
    mask_rows(weights_view.mask_embed, sc_blk, b - 1, cfg.hidden);
    f16_to_f32(sc_blk, sc_x, (int64_t) b * cfg.hidden);

    std::vector<int64_t> pos_host(b), rpos_host(b);
    std::vector<int> len_host(b);
    for (int r = 0; r < b; ++r) {
        pos_host[r] = P + r;
        rpos_host[r] = cache_pos + r;
        len_host[r] = (int) (cache_pos + b);
    }
    if (cudaMemcpyAsync(sc_pos, pos_host.data(), (size_t) b * 8, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpyAsync(sc_rpos, rpos_host.data(), (size_t) b * 8, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpyAsync(sc_len, len_host.data(), (size_t) b * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "dflash: position/length upload failed\n");
        return false;
    }

    const int64_t vrow = (int64_t) cfg.n_kv_head * cfg.head_dim_v;
    const int64_t att_w = (int64_t) cfg.n_q_head * cfg.head_dim_v;
    const double sm_scale = 1.0 / std::sqrt((double) cfg.head_dim_qk);
    for (int l = 0; l < cfg.n_layers; ++l) {
        const DflashLayer& ly = weights_view.layers[l];
        // attention half
        f32_to_f16(sc_x, sc_nx, (int64_t) b * cfg.hidden);
        rmsnorm(sc_nx, ly.ln1, sc_nx, b, cfg.hidden, (float) cfg.rms_eps);
        gemm(sc_nx, cfg.hidden, ly.q_w, sc_g1, cfg.q_dim(), b, this);
        f32_to_f16(sc_g1, sc_q, (int64_t) b * cfg.q_dim());
        headnorm(sc_q, b, cfg.n_q_head, cfg.head_dim_qk, ly.q_norm, (float) cfg.rms_eps);
        strata::kernels::mimo_rope_f16(sc_q, sc_pos, b, cfg.n_q_head, cfg.head_dim_qk, cfg.n_rot,
                                       cfg.rope_theta, nullptr);
        gemm(sc_nx, cfg.hidden, ly.k_w, sc_g1, cfg.kv_dim(), b, this);
        f32_to_f16(sc_g1, sc_k, (int64_t) b * cfg.kv_dim());
        headnorm(sc_k, b, cfg.n_kv_head, cfg.head_dim_qk, ly.k_norm, (float) cfg.rms_eps);
        strata::kernels::mimo_rope_f16(sc_k, sc_pos, b, cfg.n_kv_head, cfg.head_dim_qk, cfg.n_rot,
                                       cfg.rope_theta, nullptr);
        gemm(sc_nx, cfg.hidden, ly.v_w, sc_g1, cfg.kv_dim(), b, this);
        f32_to_f16(sc_g1, sc_v, (int64_t) b * cfg.kv_dim());
        // append the block SPECULATIVELY: ring slots [P, P + block)
        if (cudaMemcpyAsync((void*) (ring_k_layer(l) + (size_t) cache_pos * cfg.kv_dim()), sc_k,
                            (size_t) b * cfg.kv_dim() * 2, cudaMemcpyDeviceToDevice) != cudaSuccess ||
            cudaMemcpyAsync((void*) (ring_v_layer(l) + (size_t) cache_pos * vrow), sc_v,
                            (size_t) b * vrow * 2, cudaMemcpyDeviceToDevice) != cudaSuccess) {
            std::fprintf(stderr, "dflash: block append failed\n");
            return false;
        }
        // bidirectional window attention over [committed ring rows + this block]
        strata::kernels::mimo_attn_block(sc_q, ring_k_layer(l), ring_v_layer(l), sc_attn, sc_rpos,
                                         sc_len, b, cfg.n_q_head, cfg.n_kv_head, cfg.head_dim_qk,
                                         cfg.head_dim_v, S, cfg.window, cfg.window_right, ly.sink,
                                         sm_scale, nullptr);
        f32_to_f16(sc_attn, sc_o_in, b * att_w);
        gemm(sc_o_in, att_w, ly.o_w, sc_g1, cfg.hidden, b, this);
        residual_add(sc_x, sc_g1, (int64_t) b * cfg.hidden);
        // MLP half
        f32_to_f16(sc_x, sc_nx, (int64_t) b * cfg.hidden);
        rmsnorm(sc_nx, ly.ln2, sc_nx, b, cfg.hidden, (float) cfg.rms_eps);
        gemm(sc_nx, cfg.hidden, ly.gate_w, sc_gate, cfg.inter, b, this);
        gemm(sc_nx, cfg.hidden, ly.up_w, sc_up, cfg.inter, b, this);
        silu_mul(sc_gate, sc_up, sc_act, (int64_t) b * cfg.inter);
        gemm(sc_act, cfg.inter, ly.down_w, sc_g1, cfg.hidden, b, this);
        residual_add(sc_x, sc_g1, (int64_t) b * cfg.hidden);
    }
    f32_to_f16(sc_x, sc_nx, (int64_t) b * cfg.hidden);
    rmsnorm(sc_nx, weights_view.final_norm, sc_state, b, cfg.hidden, (float) cfg.rms_eps);
    // proposals: argmax over the TARGET lm_head of state rows 1..block-1
    argmax_fn(sc_state + cfg.hidden, b - 1, out_ids, hooks_user);
    return true;
}

}  // namespace strata::mimo
