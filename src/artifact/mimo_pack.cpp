// src/artifact/mimo_pack.cpp - the MiMo v2 EXL3 pack (notes/MIMO-V2-PORT.md, include/strata/artifact/
// mimo_pack.hpp is the contract).
//
// MiMo's dense tensors are RECONSTRUCTED to F16 on the host at load and uploaded once; the experts
// stay trellis on disk for the existing EXL3 expert machinery (Exl3ExpertStore/stream opened with
// the `model.layers.` prefix - mimo_expert_tensor below is the one spelling of that name).  Two
// load-time conversions live here and nowhere else: the BF16 embed to F16.  The o_proj
// attention_value_scale fold is NOT applied: the checkpoint's trellis is already pre-folded (the
// converter quantizes exllamav3's loaded - folded - weights; the dflash dir measured 0.997 vs its bf16
// twin with NO fold and 0.610 with one, and this model went through the same pipeline - measured below,
// the fold-on arm of the A/B degraded "The capital of France is" from " Paris, ..." to mush).
// STRATA_MIMO_O_FOLD=1 re-applies it on the upload copy for an A/B; the dense cache always stores
// UNFOLDED bytes, so one cache serves both arms.
#include "strata/artifact/mimo_pack.hpp"

#include "strata/artifact/exl3_model.hpp"
// STRATA_MIMO_DENSE_F16=1 builds the reconstructed-F16 pack (the parity oracle, ~13.2 GiB dense);
// the DEFAULT keeps the dense linears as TRELLIS on the device (~4 GiB: the attention projections,
// layer-0 MLP and lm_head through the on-device exl3_gemv_f32 / exl3_dense_gemv_multi kernels; the
// embed and the MTP heads stay F16 - no trellis for the embed, and the MTP module reads F16).
#include "strata/kernels/cpu/exl3.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <functional>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

namespace strata {
namespace {

using core::MimoGeometry;
using kernels::cpu::Exl3Codebook;

// ---- fp16/bf16 <-> fp32 (the same round-to-nearest-even as the codecs) ----
float h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31, exp = (h >> 10) & 0x1F, man = h & 0x3FF, f;
    if (exp == 0) {
        if (!man) f = sign;
        else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; --exp; } man &= 0x3FF; f = sign | (exp << 23) | (man << 13); }
    } else if (exp == 0x1F) f = sign | 0x7F800000u | (man << 13);
    else f = sign | ((exp + 127 - 15) << 23) | (man << 13);
    float o; std::memcpy(&o, &f, 4); return o;
}
uint16_t f2h(float x) {
    uint32_t u; std::memcpy(&u, &x, 4);
    uint32_t sign = (u >> 16) & 0x8000u, absu = u & 0x7FFFFFFFu;
    if (absu >= 0x7F800000u) return (uint16_t)(sign | (absu > 0x7F800000u ? 0x7E00u : 0x7C00u));
    int e = (int)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t m = u & 0x7FFFFFu;
    if (e >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        m |= 0x800000u;
        uint32_t shift = 14 - (uint32_t)e, h = m >> shift, rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) ++h;
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t)e << 10) | (m >> 13), rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
    return (uint16_t)(sign | h);
}
float bf2f(uint16_t h) { uint32_t f = (uint32_t)h << 16; float o; std::memcpy(&o, &f, 4); return o; }

/// A tensor read through the model's index: dtype/shape from Exl3Model::meta, bytes from this
/// pack's own mmap of the shard (Exl3Model mmaps shards lazily for linear(); the raw handles here
/// keep exl3_model.hpp's surface at `meta`).
class TensorReader {
public:
    explicit TensorReader(const Exl3Model& model) : model_(model) {}
    const StTensor* meta(const std::string& name) const { return model_.meta(name); }
    /// nullptr when absent; the pointer stays valid for this reader's lifetime (the shard stays mapped).
    const uint8_t* bytes(const std::string& name, const StTensor** t_out = nullptr) {
        const StTensor* t = model_.meta(name);
        if (t == nullptr) { if (t_out) *t_out = nullptr; return nullptr; }
        std::string path; size_t off = 0, n = 0;
        if (!model_.locate(name, path, off, n)) return nullptr;
        const SafetensorsFile& sf = shard(path);
        if (t_out) *t_out = t;
        return sf.data(*t);
    }

private:
    const SafetensorsFile& shard(const std::string& path) {
        auto it = shards_.find(path);
        if (it != shards_.end()) return *it->second;
        auto f = std::make_unique<SafetensorsFile>(path);
        auto* raw = f.get();
        shards_.emplace(path, std::move(f));
        return *raw;
    }
    const Exl3Model& model_;
    std::map<std::string, std::unique_ptr<SafetensorsFile>> shards_;
};

/// One reconstructed EXL3 linear: banded jobs of whole 128-row blocks (8 i-tiles) so the Hadamard's
/// 128-point column blocks never straddle a band - a band is exactly exl3_reconstruct_weight on a
/// trellis slice, and two bands of the same tensor never share a 128-row group.
struct ReconstructJob {
    const Exl3Linear* L;
    uint16_t* out;            ///< the tensor's host F16 buffer (UNFOLDED - the fold is an upload-time act)
    int i0, i1;               ///< i-tile range [i0, i1)
    std::string name;         ///< diagnostics
};

void run_reconstruct_job(const ReconstructJob& j) {
    const int nj = j.L->nj, words = j.L->words(), n = nj * 16, ki = j.i1 - j.i0;
    // ONE call per band: the row Hadamard mixes 128-row blocks, and a band boundary is always a
    // 128-row boundary, so the band sees exactly its own complete blocks
    kernels::cpu::exl3_reconstruct_weight(j.L->trellis + (size_t)j.i0 * nj * words, ki, nj, j.L->bits,
                                          (Exl3Codebook)j.L->cb, j.L->suh + (size_t)j.i0 * 16,
                                          j.L->svh, j.out + (size_t)j.i0 * 16 * n, j.L->frac);
}

/// The host-side band pool: every tensor's bands are independent jobs; STRATA_MIMO_THREADS (default
/// min(hardware_concurrency, 8), the exl3_pack requantizer's convention) threads drain them.  The
/// first worker error stops new claims and is rethrown after the join.
void run_reconstruct_jobs(std::vector<ReconstructJob>& jobs) {
    std::atomic<int> next{0};
    std::atomic<int> completed{0};
    const char* progress_env = std::getenv("STRATA_MIMO_LOAD_PROGRESS");
    const bool progress = progress_env && std::strcmp(progress_env, "1") == 0;
    const auto started = progress ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::mutex err_mu;
    std::string err;
    auto worker = [&] {
        for (;;) {
            {
                std::lock_guard<std::mutex> lk(err_mu);
                if (!err.empty()) return;
            }
            const int j = next.fetch_add(1);
            if (j >= (int)jobs.size()) return;
            try {
                run_reconstruct_job(jobs[(size_t)j]);
                if (progress) {
                    const int done = completed.fetch_add(1) + 1;
                    if (done % 1024 == 0 || done == (int)jobs.size())
                        std::fprintf(stderr, "mimo: dense reconstruction %d/%zu bands (%.1f s)\n",
                                     done, jobs.size(), std::chrono::duration<double>(
                                         std::chrono::steady_clock::now() - started).count());
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(err_mu);
                if (err.empty()) err = jobs[(size_t)j].name + ": " + e.what();
                return;
            }
        }
    };
    const char* env = std::getenv("STRATA_MIMO_THREADS");
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int n_workers = env != nullptr ? std::max(1, std::atoi(env)) : (int)std::min<unsigned>(hw, 8u);
    n_workers = std::min(n_workers, std::max(1, (int)jobs.size()));
    std::vector<std::thread> pool;
    pool.reserve((size_t)n_workers - 1);
    for (int i = 0; i < n_workers - 1; ++i) pool.emplace_back(worker);
    worker();
    for (std::thread& t : pool) t.join();
    if (!err.empty()) throw std::runtime_error(err);
}

/// exl3_reconstruct_weight emits [in x out] row-major (its `out[i*n + j]` has i over the suh/in dim,
/// j over the svh/out dim - what exl3_transpose_f16 exists to convert on the device); the pack's contract
/// and Gemm::f16 want [out x in].  This is the host twin: dst[o*in + i] = src[i*out + o], threaded over
/// out-row blocks with the same STRATA_MIMO_THREADS count as the band pool.
void transpose_linear_f16(const uint16_t* src, uint16_t* dst, int64_t rows_in, int64_t cols_out) {
    const char* env = std::getenv("STRATA_MIMO_THREADS");
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int n_workers = env != nullptr ? std::max(1, std::atoi(env)) : (int)std::min<unsigned>(hw, 8u);
    n_workers = (int)std::min<int64_t>(n_workers, std::max<int64_t>(1, cols_out / 64));
    auto block = [&](int64_t o0, int64_t o1) {
        for (int64_t o = o0; o < o1; ++o) {
            uint16_t* drow = dst + (size_t)o * rows_in;
            for (int64_t i = 0; i < rows_in; ++i) drow[i] = src[(size_t)i * cols_out + o];
        }
    };
    std::vector<std::thread> pool;
    const int64_t per = (cols_out + n_workers - 1) / n_workers;
    for (int w = 1; w < n_workers; ++w)
        pool.emplace_back(block, w * per, std::min((w + 1) * per, cols_out));
    block(0, std::min(per, cols_out));
    for (std::thread& t : pool) t.join();
}

/// The DENSE-Q8 mode's load-time quantizer: a [out x in] F16 host tensor -> GGUF Q8_0 blocks, the
/// byte layout `kernels::native_q8_0_f32` decodes (`{ fp16 d ; int8 qs[32] }`, 34 bytes per 32
/// weights, blocks packed in row-major order over the out rows - the same blocks a GGUF Q8_0 tensor
/// carries; the canonical S8 form of it is codes + a fp32 scale per 32 with code bias -128, per
/// tools/canonical_xcheck.py's Q8_0 mapping).  ggml's `quantize_row_q8_0` arithmetic: per block
/// d = amax / 127, q = round(w / d), d stored fp16 - here the quantizer is keyed to the STORED fp16
/// d (id = 1 / f32(fp16(d))) so what the kernel decodes is the exact reference the rounding aimed
/// at, and a final clamp to [-127, 127] covers the one-ulp fp16 rounding cases.  Threaded over
/// out-row blocks with the same STRATA_MIMO_THREADS count as the band pool.
void quantize_linear_q8_0(const uint16_t* w, uint8_t* out_blocks, int64_t n_in, int64_t n_out) {
    if (n_in <= 0 || n_in % 32 != 0)
        throw std::runtime_error("mimo: q8_0 needs n_in a positive multiple of 32 (got " +
                                 std::to_string(n_in) + ")");
    const char* env = std::getenv("STRATA_MIMO_THREADS");
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int n_workers = env != nullptr ? std::max(1, std::atoi(env)) : (int)std::min<unsigned>(hw, 8u);
    n_workers = (int)std::min<int64_t>(n_workers, std::max<int64_t>(1, n_out / 64));
    const int64_t blocks_per_row = n_in / 32;
    auto rows = [&](int64_t o0, int64_t o1) {
        for (int64_t o = o0; o < o1; ++o) {
            uint8_t* row = out_blocks + (size_t)o * blocks_per_row * 34;
            const uint16_t* src = w + (size_t)o * n_in;
            for (int64_t b = 0; b < blocks_per_row; ++b) {
                const uint16_t* blk = src + b * 32;
                float amax = 0.0f;
                for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(h2f(blk[j])));
                const float d = amax / 127.0f;
                const uint16_t dh = f2h(d);
                const float id = d != 0.0f ? 1.0f / h2f(dh) : 0.0f;
                uint8_t* dst = row + b * 34;
                std::memcpy(dst, &dh, 2);
                int8_t* qs = (int8_t*)(dst + 2);
                for (int j = 0; j < 32; ++j) {
                    int q = (int) std::lround((double) h2f(blk[j]) * (double) id);
                    if (q > 127) q = 127;
                    if (q < -127) q = -127;
                    qs[j] = (int8_t) q;
                }
            }
        }
    };
    std::vector<std::thread> pool;
    const int64_t per = (n_out + n_workers - 1) / n_workers;
    for (int w = 1; w < n_workers; ++w) pool.emplace_back(rows, w * per, std::min((w + 1) * per, n_out));
    rows(0, std::min(per, n_out));
    for (std::thread& t : pool) t.join();
}

// ---- v_proj's padded lanes: the checkpoint zero-pads V from head_dim_v to head_dim_qk columns
// per head BEFORE quantizing, so the dequantized padding carries the trellis's noise of a zero
// target, not zeros.  Measured mean |w| ratios (padded / live): 0.10 for the real K=2/2.5 encoder
// (Viterbi, random targets), 0.21 for a 1-step greedy zero encode, ~1.0 when the lanes hold real
// weights (a geometry misunderstanding).  0.4 separates the three with margin.
bool v_padded_lanes_look_zero(const uint16_t* w, int64_t n_embd, int64_t kv, int64_t head_dim_qk,
                              int64_t head_dim_v) {
    const int64_t per_head = head_dim_qk, live = head_dim_v;
    double pad_abs = 0, live_abs = 0;
    uint64_t pad_n = 0, live_n = 0;
    for (int64_t r = 0; r < n_embd; ++r) {
        const uint16_t* row = w + (size_t)r * (size_t)(kv * per_head);
        for (int64_t h = 0; h < kv; ++h)
            for (int64_t c = 0; c < per_head; ++c) {
                const double a = std::fabs(h2f(row[h * per_head + c]));
                if (c < live) { live_abs += a; ++live_n; }
                else { pad_abs += a; ++pad_n; }
            }
    }
    if (live_n == 0 || pad_n == 0) return true;
    const double ratio = (pad_abs / pad_n) / (live_abs / live_n + 1e-30);
    return ratio <= 0.4;
}

/// Read one plain (non-EXL3) tensor as F32; F32/F16/BF16 in, exactly `expect` elements.
void read_f32(TensorReader& rd, const std::string& name, size_t expect, float* out) {
    const StTensor* t = nullptr;
    const uint8_t* p = rd.bytes(name, &t);
    if (p == nullptr) throw std::runtime_error("mimo: no tensor " + name);
    if (t->nelem() != expect) throw std::runtime_error("mimo: " + name + " has " + std::to_string(t->nelem()) +
                                                       " elements, expected " + std::to_string(expect));
    if (t->dtype == "F32") std::memcpy(out, p, expect * 4);
    else if (t->dtype == "F16" || t->dtype == "BF16") {
        const bool bf = t->dtype == "BF16";
        const uint16_t* h = (const uint16_t*)p;
        for (size_t i = 0; i < expect; ++i) out[i] = bf ? bf2f(h[i]) : h2f(h[i]);
    } else throw std::runtime_error("mimo: " + name + ": unsupported dtype " + t->dtype);
}

/// The contract's MimoPack is a table of raw device pointers with no destructor, so the
/// allocations mimo_pack_build made for the CURRENT pack live here: the next build retires them
/// (an engine process builds one pack and holds it for its lifetime; the tests rebuild small ones),
/// and an atexit handler releases whatever the process still holds at the end.
struct DeviceAllocs {
    int device = -1;
    std::vector<void*> p;
};
std::mutex g_allocs_mu;
std::vector<DeviceAllocs> g_pack_allocs;
bool g_allocs_atexit = false;

void retire_previous_allocs() {
    std::vector<DeviceAllocs> old;
    {
        std::lock_guard<std::mutex> lk(g_allocs_mu);
        old.swap(g_pack_allocs);
    }
    for (DeviceAllocs& a : old) {
        (void)cudaSetDevice(a.device);
        for (void* q : a.p) (void)cudaFree(q);
    }
}

struct Histogram {
    /// index = K*2 (2 -> K1, 5 -> K2.5); 17 entries cover K=0.5..8
    std::vector<int> counts = std::vector<int>(17, 0);
    void add(const Exl3Linear& L) {
        const int k2 = L.bits * 2 + L.frac;
        if (k2 >= 2 && k2 < (int)counts.size()) counts[(size_t)k2] += 1;
    }
};

}  // namespace

std::string mimo_expert_tensor(int layer, int expert, const char* proj) {
    return "model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(expert) + "." + proj +
           "_proj";
}

std::string MimoInventory::summary() const {
    std::string s = std::to_string(layers) + " layers (" + std::to_string(moe_layers) + " MoE x " +
                    std::to_string(experts_per_moe_layer) + " experts), " + std::to_string(mtp_layers) +
                    " MTP heads, trellis K:";
    bool first = true;
    for (size_t k2 = 2; k2 < trellis_k_counts.size(); ++k2)
        if (trellis_k_counts[k2] > 0) {
            s += first ? " " : ", ";
            first = false;
            s += (k2 % 2 ? std::to_string(k2 / 2) + ".5" : std::to_string(k2 / 2)) + " x " +
                 std::to_string(trellis_k_counts[k2]);
        }
    return s;
}

bool mimo_validate_checkpoint(const std::string& exl3_dir, const core::MimoGeometry& g,
                              MimoInventory* inventory, std::string& error) {
    try {
        Exl3Model model(exl3_dir);
        TensorReader rd(model);
        Histogram hist;
        Histogram expert_hist;   // the routed experts apart from the dense/MTP linears

        // ---- the per-layer attention contract.  k_proj's row count IS the layer kind: the config's
        // hybrid_layer_pattern has lied before (the first group carries four SWA layers, not five),
        // so the tensors are the authority the parsed list is checked against.
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const std::string p = "model.layers." + std::to_string(l) + ".";
            const int64_t kv = g.kv_heads(l);
            auto rows = [&](const char* proj) {
                return model.linear(p + "self_attn." + proj).out();
            };
            const int64_t k_rows = rows("k_proj");
            const int64_t want = kv * g.head_dim_qk;
            if (k_rows != want) {
                error = "mimo: layer " + std::to_string(l) + " k_proj has " + std::to_string(k_rows) +
                        " rows but the config's layer kind (" + (g.is_full(l) ? "full" : "swa") +
                        ", kv=" + std::to_string(kv) + ") needs " + std::to_string(want) +
                        " - hybrid_layer_pattern disagrees with the tensors";
                return false;
            }
            if (rows("v_proj") != want) {
                error = "mimo: layer " + std::to_string(l) + " v_proj rows != k_proj rows (the "
                        "zero-padded v layout)";
                return false;
            }
            if (rows("q_proj") != g.qk_dim()) {
                error = "mimo: layer " + std::to_string(l) + " q_proj rows != qk_dim";
                return false;
            }
            {
                const Exl3Linear o = model.linear(p + "self_attn.o_proj");
                if (o.in() != g.v_dim() || o.out() != g.n_embd) {
                    error = "mimo: layer " + std::to_string(l) + " o_proj is " + std::to_string(o.in()) +
                            "x" + std::to_string(o.out()) + ", expected " + std::to_string(g.v_dim()) + "x" +
                            std::to_string(g.n_embd);
                    return false;
                }
            }
            for (const char* proj : {"q_proj", "k_proj", "v_proj"}) {   // o reads the trimmed v_dim
                const Exl3Linear L = model.linear(p + "self_attn." + proj);
                if (L.in() != g.n_embd) {
                    error = "mimo: layer " + std::to_string(l) + " " + proj + " input dim " +
                            std::to_string(L.in()) + " != hidden " + std::to_string(g.n_embd);
                    return false;
                }
                hist.add(L);
            }
            hist.add(model.linear(p + "self_attn.o_proj"));
            // sinks: present on exactly the SWA layers is the config's contract; a full layer
            // carrying one is a warning (the header's rule), a missing SWA sink is a hard error
            {
                const StTensor* sink = rd.meta(p + "self_attn.attention_sink_bias");
                if (g.is_full(l)) {
                    if (sink != nullptr)
                        std::fprintf(stderr, "mimo: warning: full-attention layer %lld carries an "
                                             "attention_sink_bias\n", (long long)l);
                } else if (sink == nullptr) {
                    error = "mimo: layer " + std::to_string(l) + " (sliding-window) has no "
                            "attention_sink_bias";
                    return false;
                } else if (sink->nelem() != (size_t)g.n_q_head) {
                    error = "mimo: layer " + std::to_string(l) + " attention_sink_bias is not [n_q_head]";
                    return false;
                }
            }
            for (const char* n : {"input_layernorm.weight", "post_attention_layernorm.weight"}) {
                const StTensor* t = rd.meta(p + n);
                if (t == nullptr || t->nelem() != (size_t)g.n_embd) {
                    error = "mimo: layer " + std::to_string(l) + " " + n + " missing or wrong width";
                    return false;
                }
            }
            if (g.is_moe(l)) {
                const StTensor* gate = rd.meta(p + "mlp.gate.weight");
                if (gate == nullptr || gate->dtype != "F16" ||
                    gate->nelem() != (size_t)(g.n_expert * g.n_embd)) {
                    error = "mimo: layer " + std::to_string(l) + " router mlp.gate.weight missing, not "
                            "F16 or not [n_expert, n_embd]";
                    return false;
                }
                const StTensor* bias = rd.meta(p + "mlp.gate.e_score_correction_bias");
                if (bias == nullptr || bias->dtype != "F32" || bias->nelem() != (size_t)g.n_expert) {
                    error = "mimo: layer " + std::to_string(l) +
                            " mlp.gate.e_score_correction_bias missing, not F32 or not [n_expert]";
                    return false;
                }
                // every expert tensor of every projection must exist; trellis/suh/svh each
                for (int64_t e = 0; e < g.n_expert; ++e)
                    for (const char* proj : {"gate", "up", "down"}) {
                        const std::string b = mimo_expert_tensor((int)l, (int)e, proj);
                        if (!model.has(b)) {
                            error = "mimo: missing expert tensor " + b;
                            return false;
                        }
                        for (const char* plane : {".suh", ".svh"}) {
                            if (rd.meta(b + plane) == nullptr) {
                                error = "mimo: missing expert tensor " + b + plane;
                                return false;
                            }
                        }
                        hist.add(model.linear(b));
                        expert_hist.add(model.linear(b));
                    }
            } else {
                for (const char* proj : {"gate_proj", "up_proj", "down_proj"}) {
                    if (!model.has(p + "mlp." + proj)) {
                        error = "mimo: dense layer " + std::to_string(l) + " has no mlp." + proj +
                                " (moe_layer_freq says dense)";
                        return false;
                    }
                    hist.add(model.linear(p + "mlp." + proj));
                }
            }
        }

        // ---- the MTP heads: one full SWA-geometry block each, eh_proj and five norms ----
        for (int64_t k = 0; k < g.n_mtp_layers; ++k) {
            const std::string p = "model.mtp.layers." + std::to_string(k) + ".";
            const Exl3Linear eh = model.linear(p + "eh_proj");
            if (eh.in() != 2 * g.n_embd || eh.out() != g.n_embd) {
                error = "mimo: MTP head " + std::to_string(k) + " eh_proj is " + std::to_string(eh.in()) +
                        "x" + std::to_string(eh.out()) + ", expected " + std::to_string(2 * g.n_embd) +
                        "x" + std::to_string(g.n_embd);
                return false;
            }
            hist.add(eh);
            const int64_t kv = g.n_head_kv_swa;
            if (model.linear(p + "self_attn.k_proj").out() != kv * g.head_dim_qk ||
                model.linear(p + "self_attn.v_proj").out() != kv * g.head_dim_qk ||
                model.linear(p + "self_attn.q_proj").out() != g.qk_dim() ||
                model.linear(p + "self_attn.o_proj").in() != g.v_dim()) {
                error = "mimo: MTP head " + std::to_string(k) + " attention projections do not match the "
                        "SWA geometry";
                return false;
            }
            for (const char* proj : {"q_proj", "k_proj", "v_proj", "o_proj"})
                hist.add(model.linear(p + "self_attn." + proj));
            // attention_sink_bias presence is PER TENSOR in the checkpoint (the port note): checked,
            // not required
            (void)rd.meta(p + "self_attn.attention_sink_bias");
            for (const char* n : {"hnorm.weight", "enorm.weight", "input_layernorm.weight",
                                  "pre_mlp_layernorm.weight", "final_layernorm.weight"}) {
                const StTensor* t = rd.meta(p + n);
                if (t == nullptr || t->nelem() != (size_t)g.n_embd) {
                    error = "mimo: MTP head " + std::to_string(k) + " " + n + " missing or wrong width";
                    return false;
                }
            }
            for (const char* proj : {"gate_proj", "up_proj", "down_proj"}) {
                if (!model.has(p + "mlp." + proj)) {
                    error = "mimo: MTP head " + std::to_string(k) + " has no mlp." + proj;
                    return false;
                }
                hist.add(model.linear(p + "mlp." + proj));
            }
        }

        // ---- global tensors ----
        {
            const StTensor* emb = rd.meta("model.embed_tokens.weight");
            if (emb == nullptr || emb->dtype != "BF16" ||
                emb->nelem() != (size_t)(g.n_vocab * g.n_embd)) {
                error = "mimo: model.embed_tokens.weight missing, not BF16 or not [vocab, hidden]";
                return false;
            }
            if (!model.has("lm_head")) { error = "mimo: no EXL3 lm_head"; return false; }
            const Exl3Linear lm = model.linear("lm_head");
            if (lm.in() != g.n_embd || lm.out() != g.n_vocab) {
                error = "mimo: lm_head is " + std::to_string(lm.in()) + "x" + std::to_string(lm.out()) +
                        ", expected " + std::to_string(g.n_embd) + "x" + std::to_string(g.n_vocab);
                return false;
            }
            hist.add(lm);
            const StTensor* norm = rd.meta("model.norm.weight");
            if (norm == nullptr || norm->nelem() != (size_t)g.n_embd) {
                error = "mimo: model.norm.weight missing or wrong width";
                return false;
            }
        }

        if (inventory != nullptr) {
            inventory->layers = (int)g.n_layers;
            inventory->moe_layers = 0;
            for (int64_t l = 0; l < g.n_layers; ++l) inventory->moe_layers += g.is_moe(l) ? 1 : 0;
            inventory->experts_per_moe_layer = (int)g.n_expert;
            inventory->mtp_layers = (int)g.n_mtp_layers;
            inventory->trellis_k_counts = hist.counts;
            inventory->expert_k_counts = expert_hist.counts;
        }
        return true;
    } catch (const std::exception& e) {
        error = std::string("mimo: ") + e.what();
        return false;
    }
}

std::unique_ptr<MimoPack> mimo_pack_build(const std::string& exl3_dir, const core::MimoGeometry& g,
                                          int device, std::string& error,
                                          void (*progress)(const char*), MimoInventory* inventory) {
    std::unique_ptr<MimoPack> pack;
    DeviceAllocs allocs;   // declared outside the try: the failure path frees what it allocated
    try {
        MimoInventory inv;
        std::string verr;
        if (!mimo_validate_checkpoint(exl3_dir, g, &inv, verr)) {
            error = verr;
            return nullptr;
        }
        if (inventory != nullptr) *inventory = inv;
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess)
            throw std::runtime_error("mimo: cannot select device " + std::to_string(device));
        struct RestoreDevice {
            int d;
            ~RestoreDevice() { if (d >= 0) (void)cudaSetDevice(d); }
        } restore_device{previous};

        Exl3Model model(exl3_dir);
        TensorReader rd(model);
        pack = std::make_unique<MimoPack>();
        pack->layers.resize((size_t)g.n_layers);
        pack->mtp.resize((size_t)g.n_mtp_layers);
        uint64_t vram = 0;

        // a failed build retires its own allocations on the way out (the registry only adopts
        // them once the pack is being returned); a successful build leaves them for its pack
        allocs.device = device;
        allocs.p.clear();
        auto upload = [&](const void* src, size_t bytes) -> void* {
            void* d = nullptr;
            if (cudaMalloc(&d, bytes ? bytes : 1) != cudaSuccess) throw std::runtime_error("mimo: cudaMalloc failed");
            allocs.p.push_back(d);
            if (bytes > 0 && cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
                throw std::runtime_error(std::string("mimo: upload failed: ") + cudaGetErrorString(cudaGetLastError()) +
                                         " (bytes " + std::to_string(bytes) + ")");
            vram += bytes;
            return d;
        };
        auto say = [&](const char* line) { if (progress != nullptr) progress(line); };

        // ---------------------------------------------------------------- the staging plan
        // Every dense tensor the pack uploads, in ONE canonical order - the same walk builds the
        // fresh host set and reads/writes the dense cache below.  `host` holds the tensor's
        // UNFOLDED bytes (the o_proj fold is applied on the upload copy, so one cache serves both
        // arms of the fold A/B); linears keep their Exl3Linear in `keep` for the band pool.
        struct Staged {
            std::string name;
            bool is_f32 = false;      // false: F16 bytes
            bool foldable = false;    // o_proj (the attention_value_scale candidate)
            bool check_v_pad = false;
            int linear = -1;          // index into `keep`, when a reconstructed linear
            int trellis_linear = -1;  // index into `keep`, when a TRELLIS-planes entry (three uploads)
            // what the engine SERVES the tensor as (the cache always stores the F16 bytes):
            //   F16 - the reconstructed weight uploaded as F16 (the --dense-f16 arm)
            //   Q8  - the F16 quantized at upload to GGUF q8_0 blocks (--dense-q8 arm)
            //   TRELLIS - the raw planes; with `linear >= 0` the F16 bytes are still reconstructed
            //             and cached (q8 mode's layer-0 MLP: the trellis is smaller AND at M = 1 the
            //             decode reads 3 x 25 MB of planes against 3 x 71 MB of q8 blocks)
            enum Serve { F16, Q8, TRELLIS } serve = F16;
            size_t bytes = 0;         // the tensor's size in bytes (the planes' total, trellis kind)
            std::vector<uint8_t> host;
            std::function<void(void*)> place;
            std::function<void(void*)> place_q8;     // the Q8 arm's placement (blocks pointer)
            std::function<void(MimoTrellis&)> place_t;   // the trellis kind's placement
        };
        std::vector<Staged> staged;
        std::vector<Exl3Linear> keep;   // the mmaps outlive this function; the structs must too
        auto add_linear = [&](const std::string& name, Exl3Linear L, bool fold_scale, bool check_v_pad,
                              std::function<void(void*)> place) {
            keep.push_back(std::move(L));
            Staged t;
            t.name = name;
            t.foldable = fold_scale;
            t.check_v_pad = check_v_pad;
            t.linear = (int)keep.size() - 1;
            t.bytes = (size_t)keep.back().in() * keep.back().out() * 2;
            t.place = std::move(place);
            staged.push_back(std::move(t));
        };
        // TRELLIS-DENSE (the default mode): the raw planes of the non-MTP dense linears, uploaded as
        // stored - no reconstruction, no cache (the F16 path's cache stores F16 bytes), no fold (the
        // trellis is pre-folded).  STRATA_MIMO_DENSE_F16=1 takes the reconstructed-F16 route instead,
        // with the trellis views left null.  STRATA_MIMO_DENSE_Q8=1 reconstructs the SAME F16 bytes
        // (one dense cache serves F16 and q8 - the plan is identical) and quantizes them to GGUF q8_0
        // blocks at upload for `q8_serve` tensors; the rest keep their trellis planes.
        const bool dense_f16 = [] {
            const char* e = std::getenv("STRATA_MIMO_DENSE_F16");
            return e != nullptr && e[0] == '1';
        }();
        const bool dense_q8 = [] {
            const char* e = std::getenv("STRATA_MIMO_DENSE_Q8");
            return e != nullptr && e[0] == '1';
        }();
        // WHICH dense linears the q8 arm quantizes (STRATA_MIMO_Q8_SET, a comma list of
        // q,k,v,o,lm; default all): the trade-curve knob - every subset keeps the rest on their
        // trellis planes, so the arena gives back the difference.  Measured per-tensor value at
        // M = 1 (gfx1100, the A/C arms of the 2026-10-28 experiment): the lm_head is the best
        // byte ever spent (2.07 -> 0.79 ms/token for +0.19 GiB net), o_proj next (6.2 -> 2.7 for
        // +0.89), the qkv set last (12.0 -> 5.9 for +1.6).
        const auto q8_want = [dense_q8](char which) {
            if (!dense_q8) return false;
            const char* e = std::getenv("STRATA_MIMO_Q8_SET");
            if (e == nullptr || e[0] == '\0') return true;
            for (const char* p = e; *p; ++p) {
                if (*p == which && (p[1] == ',' || p[1] == '\0' || p[1] == ' ')) return true;
                while (*p && *p != ',') ++p;
            }
            return false;
        };
        auto add_trellis = [&](const std::string& name, const Exl3Linear& L, bool fold_scale,
                               const std::function<void(MimoTrellis&)>& place_t,
                               const std::function<void(void*)>& place16,
                               const std::function<void(void*)>& place_q8 = nullptr, bool q8_serve = false) {
            if (dense_f16 || dense_q8) {
                add_linear(name, L, fold_scale, /*check_v_pad=*/false, place16);   // views stay null
                if (dense_q8 && q8_serve) {
                    staged.back().serve = Staged::Q8;
                    staged.back().place_q8 = place_q8;
                } else if (dense_q8) {   // keeps the trellis planes; the F16 bytes feed the cache only
                    staged.back().serve = Staged::TRELLIS;
                    staged.back().trellis_linear = staged.back().linear;
                    staged.back().place_t = place_t;
                }
                return;
            }
            keep.push_back(L);
            Staged t;
            t.name = name;
            t.trellis_linear = (int)keep.size() - 1;
            t.serve = Staged::TRELLIS;
            const Exl3Linear& K = keep.back();
            t.bytes = (size_t)K.ki * K.nj * K.words() * 2 + ((size_t)K.ki * 16 + (size_t)K.nj * 16) * 2;
            t.place_t = std::move(place_t);
            staged.push_back(std::move(t));
        };
        auto add_bytes = [&](const std::string& name, bool is_f32, std::vector<uint8_t> bytes,
                             std::function<void(void*)> place) {
            Staged t;
            t.name = name;
            t.is_f32 = is_f32;
            t.bytes = bytes.size();
            t.host = std::move(bytes);
            t.place = std::move(place);
            staged.push_back(std::move(t));
        };
        auto f32_bytes = [&](const float* v, size_t n) {
            std::vector<uint8_t> b(n * 4);
            std::memcpy(b.data(), v, b.size());
            return b;
        };

        for (int64_t l = 0; l < g.n_layers; ++l) {
            const std::string p = "model.layers." + std::to_string(l) + ".";
            const bool moe = g.is_moe(l);
            add_trellis(p + "self_attn.q_proj", model.linear(p + "self_attn.q_proj"), /*fold=*/false,
                        [pck = pack.get(), idx = (size_t)l](MimoTrellis& t) { pck->layers[idx].tq = t; },
                        [pck = pack.get(), idx = (size_t)l](void* q) { pck->layers[idx].q = q; },
                        [pck = pack.get(), idx = (size_t)l](void* q) { pck->layers[idx].q8q.w = q; }, q8_want('q'));
            add_trellis(p + "self_attn.k_proj", model.linear(p + "self_attn.k_proj"), /*fold=*/false,
                        [pck = pack.get(), idx = (size_t)l](MimoTrellis& t) { pck->layers[idx].tk = t; },
                        [pck = pack.get(), idx = (size_t)l](void* q) { pck->layers[idx].k = q; },
                        [pck = pack.get(), idx = (size_t)l](void* q) { pck->layers[idx].q8k.w = q; }, q8_want('k'));
            add_trellis(p + "self_attn.v_proj", model.linear(p + "self_attn.v_proj"), /*fold=*/false,
                        [pck = pack.get(), idx = (size_t)l](MimoTrellis& t) { pck->layers[idx].tv = t; },
                        [pck = pack.get(), idx = (size_t)l](void* q) { pck->layers[idx].v = q; },
                        [pck = pack.get(), idx = (size_t)l](void* q) { pck->layers[idx].q8v.w = q; }, q8_want('v'));
            add_trellis(p + "self_attn.o_proj", model.linear(p + "self_attn.o_proj"), /*fold=*/true,
                        [pck = pack.get(), idx = (size_t)l](MimoTrellis& t) { pck->layers[idx].to = t; },
                        [pck = pack.get(), idx = (size_t)l](void* q) { pck->layers[idx].o = q; },
                        [pck = pack.get(), idx = (size_t)l](void* q) { pck->layers[idx].q8o.w = q; }, q8_want('o'));
            if (moe) {
                const StTensor* t = rd.meta(p + "mlp.gate.weight");
                const uint8_t* raw = rd.bytes(p + "mlp.gate.weight");
                std::vector<uint8_t> gate((size_t)t->nelem() * 2);
                std::memcpy(gate.data(), raw, gate.size());   // F16, as stored
                add_bytes(p + "mlp.gate.weight", false, std::move(gate),
                          [pck = pack.get(), idx = (size_t)l](void* q) { pck->layers[idx].router = q; });
                std::vector<float> bias((size_t)g.n_expert);
                read_f32(rd, p + "mlp.gate.e_score_correction_bias", bias.size(), bias.data());
                add_bytes(p + "mlp.gate.e_score_correction_bias", true, f32_bytes(bias.data(), bias.size()),
                          [pck = pack.get(), idx = (size_t)l](void* q) { pck->layers[idx].e_score = q; });
            } else {
                add_trellis(p + "mlp.gate_proj", model.linear(p + "mlp.gate_proj"), /*fold=*/false,
                            [pck = pack.get(), idx = (size_t)l](MimoTrellis& t) {
                                pck->layers[idx].tffn_gate = t;
                            },
                            [pck = pack.get(), idx = (size_t)l](void* q) { pck->layers[idx].ffn_gate = q; });
                add_trellis(p + "mlp.up_proj", model.linear(p + "mlp.up_proj"), /*fold=*/false,
                            [pck = pack.get(), idx = (size_t)l](MimoTrellis& t) {
                                pck->layers[idx].tffn_up = t;
                            },
                            [pck = pack.get(), idx = (size_t)l](void* q) { pck->layers[idx].ffn_up = q; });
                add_trellis(p + "mlp.down_proj", model.linear(p + "mlp.down_proj"), /*fold=*/false,
                            [pck = pack.get(), idx = (size_t)l](MimoTrellis& t) {
                                pck->layers[idx].tffn_down = t;
                            },
                            [pck = pack.get(), idx = (size_t)l](void* q) { pck->layers[idx].ffn_down = q; });
            }
            std::vector<float> nrm((size_t)g.n_embd);
            read_f32(rd, p + "input_layernorm.weight", nrm.size(), nrm.data());
            add_bytes(p + "input_layernorm.weight", true, f32_bytes(nrm.data(), nrm.size()),
                      [pck = pack.get(), idx = (size_t)l](void* q) { pck->layers[idx].norm_attn = q; });
            read_f32(rd, p + "post_attention_layernorm.weight", nrm.size(), nrm.data());
            add_bytes(p + "post_attention_layernorm.weight", true, f32_bytes(nrm.data(), nrm.size()),
                      [pck = pack.get(), idx = (size_t)l](void* q) { pck->layers[idx].norm_mlp = q; });
            if (!g.is_full(l)) {
                std::vector<float> sink((size_t)g.n_q_head);
                read_f32(rd, p + "self_attn.attention_sink_bias", sink.size(), sink.data());
                add_bytes(p + "self_attn.attention_sink_bias", true, f32_bytes(sink.data(), sink.size()),
                          [pck = pack.get(), idx = (size_t)l](void* q) { pck->layers[idx].sink = q; });
            }
        }

        for (int64_t k = 0; k < g.n_mtp_layers; ++k) {
            const std::string p = "model.mtp.layers." + std::to_string(k) + ".";
            add_linear(p + "eh_proj", model.linear(p + "eh_proj"), false, false,
                       [pck = pack.get(), idx = (size_t)k](void* q) { pck->mtp[idx].eh = q; });
            add_linear(p + "self_attn.q_proj", model.linear(p + "self_attn.q_proj"), false, false,
                       [pck = pack.get(), idx = (size_t)k](void* q) { pck->mtp[idx].q = q; });
            add_linear(p + "self_attn.k_proj", model.linear(p + "self_attn.k_proj"), false, false,
                       [pck = pack.get(), idx = (size_t)k](void* q) { pck->mtp[idx].k = q; });
            add_linear(p + "self_attn.v_proj", model.linear(p + "self_attn.v_proj"), false,
                       g.head_dim_v != g.head_dim_qk,
                       [pck = pack.get(), idx = (size_t)k](void* q) { pck->mtp[idx].v = q; });
            add_linear(p + "self_attn.o_proj", model.linear(p + "self_attn.o_proj"), true, false,
                       [pck = pack.get(), idx = (size_t)k](void* q) { pck->mtp[idx].o = q; });
            add_linear(p + "mlp.gate_proj", model.linear(p + "mlp.gate_proj"), false, false,
                       [pck = pack.get(), idx = (size_t)k](void* q) { pck->mtp[idx].mlp_gate = q; });
            add_linear(p + "mlp.up_proj", model.linear(p + "mlp.up_proj"), false, false,
                       [pck = pack.get(), idx = (size_t)k](void* q) { pck->mtp[idx].mlp_up = q; });
            add_linear(p + "mlp.down_proj", model.linear(p + "mlp.down_proj"), false, false,
                       [pck = pack.get(), idx = (size_t)k](void* q) { pck->mtp[idx].mlp_down = q; });
            std::vector<float> nrm((size_t)g.n_embd);
            auto norm_stage = [&](const char* name, void* MimoPack::MtpHead::*field) {
                read_f32(rd, p + name, nrm.size(), nrm.data());
                add_bytes(p + name, true, f32_bytes(nrm.data(), nrm.size()),
                          [pck = pack.get(), idx = (size_t)k, field](void* q) { pck->mtp[idx].*field = q; });
            };
            norm_stage("hnorm.weight", &MimoPack::MtpHead::hnorm);
            norm_stage("enorm.weight", &MimoPack::MtpHead::enorm);
            norm_stage("input_layernorm.weight", &MimoPack::MtpHead::norm_attn);
            norm_stage("pre_mlp_layernorm.weight", &MimoPack::MtpHead::norm_mlp);
            norm_stage("final_layernorm.weight", &MimoPack::MtpHead::norm_final);
            if (rd.meta(p + "self_attn.attention_sink_bias") != nullptr) {   // presence is per tensor
                std::vector<float> sink((size_t)g.n_q_head);
                read_f32(rd, p + "self_attn.attention_sink_bias", sink.size(), sink.data());
                add_bytes(p + "self_attn.attention_sink_bias", true, f32_bytes(sink.data(), sink.size()),
                          [pck = pack.get(), idx = (size_t)k](void* q) { pck->mtp[idx].sink = q; });
            }
        }

        {
            // STRATA_MIMO_LM_F16=1 keeps the lm_head RECONSTRUCTED F16 as well as (not instead of)
            // the trellis planes: the MTP module argmaxes through the pack's F16 lm_head directly
            // (the engine sets this when --mtp is on; +1.25 GiB, the arena auto-fills around it)
            const bool lm_f16 = [] {
                const char* e = std::getenv("STRATA_MIMO_LM_F16");
                return e != nullptr && e[0] == '1';
            }();
            if (lm_f16)
                add_linear("lm_head", model.linear("lm_head"), false, false,
                           [pck = pack.get()](void* q) { pck->lm_head = q; });
            add_trellis("lm_head", model.linear("lm_head"), /*fold=*/false,
                        [pck = pack.get()](MimoTrellis& t) { pck->tlm = t; },
                        [pck = pack.get()](void* q) { pck->lm_head = q; },
                        [pck = pack.get()](void* q) { pck->lm_q8.w = q; }, q8_want('l'));
            const StTensor* t = rd.meta("model.embed_tokens.weight");
            const uint8_t* raw = rd.bytes("model.embed_tokens.weight");
            std::vector<uint16_t> embed16((size_t)t->nelem());
            const uint16_t* bf = (const uint16_t*)raw;
            for (size_t i = 0; i < embed16.size(); ++i) embed16[i] = f2h(bf2f(bf[i]));
            std::vector<uint8_t> eb(embed16.size() * 2);
            std::memcpy(eb.data(), embed16.data(), eb.size());
            add_bytes("model.embed_tokens.weight", false, std::move(eb),
                      [pck = pack.get()](void* q) { pck->embed = q; });
            std::vector<float> nrm((size_t)g.n_embd);
            read_f32(rd, "model.norm.weight", nrm.size(), nrm.data());
            add_bytes("model.norm.weight", true, f32_bytes(nrm.data(), nrm.size()),
                      [pck = pack.get()](void* q) { pck->norm_final = q; });
        }

        // ---------------------------------------------------------------- the dense cache
        // <dir>/strata-mimo-dense-cache[-lmf16].bin: every staged tensor's UNFOLDED bytes, plan order,
        // contiguous; <dir>/strata-mimo-dense-cache[-lmf16].manifest: format, the fingerprint (config.json's
        // bytes + the geometry that shapes the tensors) and one "name f16|f32 elems" line per tensor.
        // Read via mmap PROT_READ MAP_PRIVATE only; written once through temp names + rename.
        // STRATA_MIMO_DENSE_CACHE=0 disables both sides.  A cache hit still runs the validator above
        // (the index cross-check) and uploads from the map; a miss rebuilds and rewrites the cache.
        // The -lmf16 suffix separates the STRATA_MIMO_LM_F16=1 plan (the engine sets it with --mtp:
        // one extra staged lm_head F16) from the default plan - two valid plans must not thrash one
        // cache file (each switch would pay the full ~8-minute rebuild).
        const bool cache_enabled = [] {
            const char* e = std::getenv("STRATA_MIMO_DENSE_CACHE");
            return e == nullptr || e[0] != '0';
        }() && (dense_f16 || dense_q8);   // the cache stores the reconstructed-F16 bytes; the trellis
                                         // mode has nothing to cache (the planes upload as stored), and
                                         // the q8 mode stages the SAME F16 plan (it quantizes at upload)
        const bool lm_f16_cache = [] {
            const char* e = std::getenv("STRATA_MIMO_LM_F16");
            return e != nullptr && e[0] == '1';
        }();
        const std::string cache_bin = exl3_dir + "/strata-mimo-dense-cache" +
                                      (lm_f16_cache ? "-lmf16" : "") + ".bin";
        const std::string cache_man = exl3_dir + "/strata-mimo-dense-cache" +
                                      (lm_f16_cache ? "-lmf16" : "") + ".manifest";

        auto fnv = [](const void* p, size_t n, uint64_t h) {
            const uint8_t* b = (const uint8_t*)p;
            for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
            return h;
        };
        auto fingerprint = [&] {
            uint64_t h = 1469598103934665603ull;
            h = fnv(&g.n_embd, 8, h);   h = fnv(&g.n_layers, 8, h);      h = fnv(&g.n_vocab, 8, h);
            h = fnv(&g.n_q_head, 8, h); h = fnv(&g.head_dim_qk, 8, h);   h = fnv(&g.head_dim_v, 8, h);
            h = fnv(&g.n_head_kv_full, 8, h); h = fnv(&g.n_head_kv_swa, 8, h);
            h = fnv(&g.n_expert, 8, h); h = fnv(&g.n_expert_used, 8, h); h = fnv(&g.moe_intermediate, 8, h);
            h = fnv(&g.dense_intermediate, 8, h); h = fnv(&g.n_mtp_layers, 8, h);
            h = fnv(g.layer_kind.data(), g.layer_kind.size(), h);
            h = fnv(g.moe_layer.data(), g.moe_layer.size(), h);
            h = fnv(&g.attention_value_scale, 8, h);
            FILE* f = std::fopen((exl3_dir + "/config.json").c_str(), "rb");
            if (f != nullptr) {
                char buf[65536];
                size_t n;
                while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) h = fnv(buf, n, h);
                std::fclose(f);
            }
            return h;
        };

        // the o_proj fold: env-gated while its convention is being settled (see the header).  The
        // cache stores unfolded bytes; the fold is applied to the UPLOAD COPY only.
        const bool fold_o = [] {
            const char* e = std::getenv("STRATA_MIMO_O_FOLD");
            return e != nullptr && e[0] == '1';
        }();
        auto upload_staged = [&](const Staged& t, const void* bytes) {
            if (t.foldable && fold_o) {
                const size_t n = t.bytes / 2;
                std::vector<uint16_t> folded(n);
                const uint16_t* src = (const uint16_t*)bytes;
                for (size_t i = 0; i < n; ++i) folded[i] = f2h(h2f(src[i]) * (float)g.attention_value_scale);
                t.place(upload(folded.data(), folded.size() * 2));
            } else {
                t.place(upload(bytes, t.bytes));
            }
        };
        /// the Q8 arm's upload: (optionally folded) [out x in] F16 bytes -> GGUF q8_0 blocks -> device
        auto upload_staged_q8 = [&](const Staged& t, const void* bytes) {
            std::vector<uint16_t> folded;
            if (t.foldable && fold_o) {
                const size_t n = t.bytes / 2;
                folded.resize(n);
                const uint16_t* src = (const uint16_t*)bytes;
                for (size_t i = 0; i < n; ++i) folded[i] = f2h(h2f(src[i]) * (float)g.attention_value_scale);
                bytes = folded.data();
            }
            const Exl3Linear& L = keep[(size_t)t.linear];
            std::vector<uint8_t> blocks((size_t)L.out() * ((size_t)L.in() / 32) * 34);
            quantize_linear_q8_0((const uint16_t*)bytes, blocks.data(), L.in(), L.out());
            t.place_q8(upload(blocks.data(), blocks.size()));
        };
        /// the raw trellis planes of one entry, as stored (the default mode's dense path and the q8
        /// mode's layer-0 MLP; no fold - the trellis is pre-folded)
        auto upload_staged_trellis = [&](const Staged& t) {
            const Exl3Linear& L = keep[(size_t)t.trellis_linear];
            MimoTrellis v;
            v.trellis = upload(L.trellis, (size_t)L.ki * L.nj * L.words() * 2);
            v.suh = upload(L.suh, (size_t)L.ki * 16 * 2);
            v.svh = upload(L.svh, (size_t)L.nj * 16 * 2);
            v.ki = L.ki;
            v.nj = L.nj;
            v.bits = L.bits;
            v.cb = L.cb;
            v.frac = L.frac;
            t.place_t(v);
        };

        bool cache_hit = false;
        void* cache_map = MAP_FAILED;
        size_t cache_bytes = 0;
        struct Unmap {
            void** m;
            size_t* n;
            ~Unmap() { if (m && *m != MAP_FAILED) { munmap(*m, *n); *m = MAP_FAILED; } }
        } unmap_guard{&cache_map, &cache_bytes};
        size_t plan_bytes = 0;
        for (const Staged& t : staged) plan_bytes += t.bytes;

        if (cache_enabled) {
            FILE* mf = std::fopen(cache_man.c_str(), "rb");
            if (mf != nullptr) {
                std::string man;
                char buf[65536];
                size_t n;
                while ((n = std::fread(buf, 1, sizeof buf, mf)) > 0) man.append(buf, n);
                std::fclose(mf);
                unsigned long long fp_file = 0;
                long count = 0;
                bool ok = std::sscanf(man.c_str(), "strata-mimo-dense-cache 2 %llx %ld", &fp_file, &count) == 2 &&
                          count == (long)staged.size() && fp_file == fingerprint();
                if (ok) {   // two header lines, then one "name f16|f32 elems" line per tensor
                    size_t pos = 0;
                    for (int i = 0; i < 2 && pos != std::string::npos; ++i) pos = man.find('\n', pos) + 1;
                    for (long i = 0; ok && i < count; ++i) {
                        char name[512], dtype[8];
                        long long elems = 0;
                        const size_t e = man.find('\n', pos);
                        if (pos == std::string::npos || e == std::string::npos ||
                            std::sscanf(man.c_str() + pos, "%511s %7s %lld", name, dtype, &elems) != 3)
                            ok = false;
                        else {
                            const Staged& t = staged[(size_t)i];
                            const size_t want = t.bytes / (t.is_f32 ? 4 : 2);
                            ok = std::string(name) == t.name &&
                                 std::string(dtype) == (t.is_f32 ? "f32" : "f16") && elems == (long long)want;
                            pos = e + 1;
                        }
                    }
                }
                if (ok) {
                    int fd = ::open(cache_bin.c_str(), O_RDONLY);
                    if (fd >= 0) {
                        cache_bytes = (size_t)lseek(fd, 0, SEEK_END);
                        if (cache_bytes == plan_bytes)
                            cache_map = mmap(nullptr, cache_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
                        ::close(fd);
                        cache_hit = cache_map != MAP_FAILED;
                    }
                }
            }
        }

        if (cache_enabled && !cache_hit)
            say("mimo: dense cache miss (staged plan or fingerprint changed; rebuilding)");
        if (cache_hit) {
            say("mimo: dense cache hit - uploading");
            size_t off = 0;
            for (const Staged& t : staged) {
                if (t.serve == Staged::Q8)
                    upload_staged_q8(t, (const uint8_t*)cache_map + off);
                else if (t.serve == Staged::TRELLIS)
                    upload_staged_trellis(t);   // q8 mode's layer-0 MLP: the planes, not the cached F16
                else
                    upload_staged(t, (const uint8_t*)cache_map + off);
                off += t.bytes;
            }
        } else {
            // ---- the fresh build: fill every staged tensor's host bytes, then upload + write the cache
            say("mimo: staging norms, routers, sinks and the embedding");
            std::vector<ReconstructJob> jobs;
            for (Staged& t : staged) {
                if (t.trellis_linear >= 0) {
                    try {
                        upload_staged_trellis(t);
                    } catch (const std::exception& e) {
                        const Exl3Linear& L = keep[(size_t)t.trellis_linear];
                        throw std::runtime_error(std::string("trellis upload ") + t.name + " ki " +
                                                 std::to_string(L.ki) + " nj " + std::to_string(L.nj) + ": " +
                                                 e.what());
                    }
                    if (t.linear < 0) continue;   // the default mode: no F16 bytes exist for this entry
                    // q8 mode's layer-0 MLP falls through: the F16 bytes below feed the cache only
                }
                if (t.linear < 0) continue;
                const Exl3Linear& L = keep[(size_t)t.linear];
                t.host.resize(t.bytes);
                uint16_t* out = (uint16_t*)t.host.data();
                const int band = 8;   // i-tiles per band = 128 rows = one Hadamard block
                for (int i0 = 0; i0 < L.ki; i0 += band)
                    jobs.push_back({&L, out, i0, std::min(i0 + band, L.ki), t.name});
            }
            say("mimo: reconstructing the dense EXL3 linears (banded, multithreaded)");
            run_reconstruct_jobs(jobs);
            say(dense_q8 ? "mimo: checking the v_proj padding, quantizing to q8_0 and uploading"
                        : "mimo: checking the v_proj padding and uploading");
            std::FILE* cache_out = nullptr;
            std::string cache_tmp;
            if (cache_enabled) {
                cache_tmp = cache_bin + ".tmp";
                cache_out = std::fopen(cache_tmp.c_str(), "wb");
            }
            for (Staged& t : staged) {
                if (t.trellis_linear >= 0 && t.linear < 0) continue;   // the planes went up above
                if (t.linear >= 0) {   // recon is [in x out]; the pack serves [out x in]
                    const Exl3Linear& L = keep[(size_t)t.linear];
                    if (t.check_v_pad &&
                        !v_padded_lanes_look_zero((const uint16_t*)t.host.data(), L.in(),
                                                  L.out() / g.head_dim_qk, g.head_dim_qk, g.head_dim_v))
                        throw std::runtime_error("layer v_proj lanes " + std::to_string(g.head_dim_v) + ".." +
                                                 std::to_string(g.head_dim_qk - 1) +
                                                 " of each head are not (near-)zero (" + t.name +
                                                 ") - the padded-V layout does not hold");
                    std::vector<uint8_t> trans(t.bytes);
                    transpose_linear_f16((const uint16_t*)t.host.data(), (uint16_t*)trans.data(), L.in(),
                                         L.out());
                    t.host.swap(trans);
                }
                if (cache_out != nullptr && std::fwrite(t.host.data(), 1, t.bytes, cache_out) != t.bytes)
                    throw std::runtime_error("mimo: cannot write the dense cache (disk full?)");
                if (t.serve == Staged::Q8)
                    upload_staged_q8(t, t.host.data());
                else if (t.serve != Staged::TRELLIS)
                    upload_staged(t, t.host.data());
                std::vector<uint8_t>().swap(t.host);   // free the host copy right after its upload
            }
            if (cache_out != nullptr) {
                std::fclose(cache_out);
                std::string man = "strata-mimo-dense-cache 2\n";
                char head[160];
                std::snprintf(head, sizeof head, "%llx %zu\n", (unsigned long long)fingerprint(), staged.size());
                man += head;
                for (const Staged& t : staged) {
                    man += t.name + (t.is_f32 ? " f32 " : " f16 ") +
                           std::to_string(t.bytes / (t.is_f32 ? 4 : 2)) + "\n";
                }
                const std::string man_tmp = cache_man + ".tmp";
                FILE* mf = std::fopen(man_tmp.c_str(), "wb");
                if (mf != nullptr) {
                    std::fwrite(man.data(), 1, man.size(), mf);
                    std::fclose(mf);
                    std::rename(man_tmp.c_str(), cache_man.c_str());
                    std::rename(cache_tmp.c_str(), cache_bin.c_str());
                    say("mimo: dense cache written");
                }
            }
        }
        pack->vram_bytes = vram;
        say("mimo: pack built");
        {   // the pack owns the registry now; the previous pack's memory (if any) retires
            std::lock_guard<std::mutex> lk(g_allocs_mu);
            if (!g_allocs_atexit) {
                g_allocs_atexit = true;
                std::atexit([] { retire_previous_allocs(); });
            }
        }
        retire_previous_allocs();
        {
            std::lock_guard<std::mutex> lk(g_allocs_mu);
            g_pack_allocs.push_back(std::move(allocs));
        }
        return pack;
    } catch (const std::exception& e) {
        {   // a failed build frees everything it allocated
            for (void* q : allocs.p) (void)cudaFree(q);
            allocs.p.clear();
        }
        error = std::string("mimo: ") + e.what();
        return nullptr;
    }
}

}  // namespace strata
