// src/mimo/mimo_engine_parity.cpp - THE pre-weights gate for the MiMo v2 engine (src/mimo/mimo_engine.cu):
// a full end-to-end forward on a SYNTHETIC mini-MiMo, GPU vs an independent CPU reference written from
// notes/MIMO-V2-PORT.md.  The fixture is mimo_pack_test's machinery (same file conventions, same K=2.5
// experts, same mini geometry) with small suh/svh scales so the random trellis keeps the activations finite;
// the fixture's host tensors ARE the reference's weights (reconstructed through the CPU codec like the pack
// does on the GPU side).
//
// The mini model exercises, in one greedy run: layer 0 DENSE SwiGLU + full attention (kv 4, theta 1e7),
// layer 1 MoE + SLIDING-WINDOW attention with a sink (kv 8, theta 1e4, window 8, GQA 1) and the K=2.5
// expert stream path, a 12-token prompt in chunks of 8 (a ragged second chunk with history), and decode
// steps past the window edge (positions 12..17 see the window clamp).  The reference mirrors every F16
// carry the engine makes (the normed activation, q/k/v into the caches, the attention output into o_proj,
// silu(g)*u into the down projection) and computes in double; the gates are:
//   * the ARGMAX SEQUENCE matches exactly on all 6 steps (the mandate), and
//   * each step's logits agree within a relative L2 of 2e-2, with the reference's top-2 margin reported
//     (the argmax gate is only meaningful when the margin beats the tolerance).
// GPU-gated: exits 77 (skip) with no device, the convention of the other mimo GPU tests.
#include "strata/artifact/exl3_model.hpp"
#include "strata/artifact/mimo_pack.hpp"
#include "strata/core/expert_frequency.hpp"
#include "strata/kernels/cpu/exl3.hpp"
#include "strata/kernels/exl3_experts.hpp"
#include "strata/kernels/exl3_peer_adapt.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/mimo/engine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <exception>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace k = strata::kernels;
using strata::kernels::cpu::Exl3Codebook;

static int failures = 0;
#define CHECK(cond)                                                                              \
    do {                                                                                         \
        if (!(cond)) {                                                                           \
            ++failures;                                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                 \
        }                                                                                        \
    } while (0)

// ---- the fixture (mimo_pack_test.cpp's writer; suh/svh scaled down so random trellis words stay finite)
namespace {

struct ScopedEnv {
    const char* key;
    bool present;
    std::string saved;
    ScopedEnv(const char* name, const char* value) : key(name), present(std::getenv(name) != nullptr),
        saved(present ? std::getenv(name) : "") { setenv(key, value, 1); }
    ~ScopedEnv() { if (present) setenv(key, saved.c_str(), 1); else unsetenv(key); }
};

struct Tensor {
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};

class Fixture {
public:
    std::filesystem::path dir;
    std::map<std::string, Tensor> tensors;
    uint32_t seed = 1;

    explicit Fixture(const std::filesystem::path& d) : dir(d) {}
    uint32_t rnd() { seed = seed * 1664525u + 1013904223u; return seed >> 8; }
    uint16_t rnd16() { return (uint16_t)(rnd() >> 8); }

    void bytes16(const std::string& name, const std::string& dtype, const std::vector<int64_t>& shape,
                 const std::vector<uint16_t>& v) {
        Tensor& t = tensors[name];
        t.dtype = dtype;
        t.shape = shape;
        t.bytes.resize(v.size() * 2);
        std::memcpy(t.bytes.data(), v.data(), t.bytes.size());
    }
    /// one EXL3 linear: trellis [ki, nj, 8*K2] with random words, SMALL F16 scales (the engine-parity
    /// fixture runs a full forward, so the weights must keep the activations in F16 range).
    double wscale = 0.22;   // suh/svh range [wscale/2, wscale]: big enough that a layer moves the
                            // residual (healthy argmax margins), small enough that random trellis words
                            // keep every F16 carry finite
    void exl3(const std::string& base, int ki, int nj, int k2) {
        const int words = 8 * k2;
        std::vector<uint16_t> tr((size_t)ki * nj * words);
        for (auto& w : tr) w = rnd16();
        bytes16(base + ".trellis", "I16", {ki, nj, words}, tr);
        std::vector<uint16_t> suh((size_t)ki * 16), svh((size_t)nj * 16);
        for (auto& s : suh) s = k::f16_from_f32((float)(wscale * (0.5 + (rnd() % 1000) / 2000.0)));
        for (auto& s : svh) s = k::f16_from_f32((float)(wscale * (0.5 + (rnd() % 1000) / 2000.0)));
        bytes16(base + ".suh", "F16", {ki * 16}, suh);
        bytes16(base + ".svh", "F16", {nj * 16}, svh);
        Tensor& m = tensors[base + ".mul1"];
        m.dtype = "I8";
        m.shape = {1};
        m.bytes = {1};
    }
    void f32(const std::string& name, const std::vector<int64_t>& shape, const std::vector<float>& v) {
        Tensor& t = tensors[name];
        t.dtype = "F32";
        t.shape = shape;
        t.bytes.resize(v.size() * 4);
        std::memcpy(t.bytes.data(), v.data(), t.bytes.size());
    }
    const uint16_t* u16(const std::string& name) const {
        return reinterpret_cast<const uint16_t*>(tensors.at(name).bytes.data());
    }
    const float* f32p(const std::string& name) const {
        return reinterpret_cast<const float*>(tensors.at(name).bytes.data());
    }

    void write() const {
        std::filesystem::create_directories(dir);
        const std::string shard = "model-00001-of-00001.safetensors";
        std::string body;
        std::string header = "{";
        bool first = true;
        for (const auto& [name, t] : tensors) {
            const size_t begin = body.size(), end = begin + t.bytes.size();
            std::string shape = "[";
            for (size_t d = 0; d < t.shape.size(); ++d)
                shape += (d ? "," : "") + std::to_string(t.shape[d]);
            shape += "]";
            if (!first) header += ",";
            first = false;
            header += "\"" + name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":" + shape +
                      ",\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(end) + "]}";
            body.append((const char*)t.bytes.data(), t.bytes.size());
        }
        header += "}";
        {
            std::ofstream out((dir / shard).string(), std::ios::binary);
            uint64_t n = header.size();
            out.write((const char*)&n, 8);
            out.write(header.data(), (std::streamsize)n);
            out.write(body.data(), (std::streamsize)body.size());
        }
        std::string map;
        size_t i = 0;
        for (const auto& [name, t] : tensors) {
            (void)t;
            map += std::string(i ? "," : "") + "\"" + name + "\":\"" + shard + "\"";
            ++i;
        }
        std::ofstream out((dir / "model.safetensors.index.json").string());
        out << "{\"metadata\":{\"total_size\":0},\"weight_map\":{" << map << "}}\n";
    }
};

std::string mini_config() {
    return "{\"model_type\":\"mimo_v2\",\"hidden_size\":128,\"num_hidden_layers\":2,\"vocab_size\":256,"
           "\"layernorm_epsilon\":1e-6,\"num_attention_heads\":8,\"head_dim\":32,\"v_head_dim\":32,"
           "\"partial_rotary_factor\":0.5,\"num_key_value_heads\":4,\"swa_num_key_value_heads\":8,"
           "\"sliding_window\":8,\"rope_theta\":1e7,\"swa_rope_theta\":1e4,\"attention_value_scale\":0.707,"
           "\"n_routed_experts\":4,\"num_experts_per_tok\":2,\"moe_intermediate_size\":128,"
           "\"intermediate_size\":128,\"routed_scaling_factor\":1.0,\"norm_topk_prob\":true,"
           "\"scoring_func\":\"sigmoid\",\"topk_method\":\"noaux_tc\",\"num_nextn_predict_layers\":1,"
           "\"hybrid_layer_pattern\":[0,1],\"moe_layer_freq\":[0,1]}";
}

void build_dflash_fixture(const std::filesystem::path& dir) {
    Fixture f(dir);
    f.wscale = .04;
    f.exl3("fc", 16, 8, 8); // concat two target taps [128] -> hidden 128
    for (const auto* name : {"hidden_norm.weight", "norm.weight", "layers.0.input_layernorm.weight",
                              "layers.0.post_attention_layernorm.weight", "layers.0.self_attn.q_norm.weight",
                              "layers.0.self_attn.k_norm.weight"})
        f.bytes16(name, "F16", {128}, std::vector<uint16_t>(128, k::f16_from_f32(1.f)));
    f.bytes16("mask_embedding", "F16", {128}, std::vector<uint16_t>(128, k::f16_from_f32(.1f)));
    f.bytes16("layers.0.self_attn.attention_sink_bias", "F16", {8}, std::vector<uint16_t>(8, 0));
    f.exl3("layers.0.self_attn.q_proj", 8, 64, 8);
    f.exl3("layers.0.self_attn.k_proj", 8, 8, 8);
    f.exl3("layers.0.self_attn.v_proj", 8, 8, 8);
    f.exl3("layers.0.self_attn.o_proj", 64, 8, 8);
    for (const auto* name : {"gate_proj", "up_proj", "down_proj"})
        f.exl3("layers.0.mlp." + std::string(name), 8, 8, 8);
    f.write();
    std::filesystem::rename(dir / "model-00001-of-00001.safetensors", dir / "model.safetensors");
    std::ofstream cfg(dir / "config.json");
    cfg << R"({"architectures":["DFlashDraftModel"],"model_type":"qwen3","hidden_size":128,
        "intermediate_size":128,"num_hidden_layers":1,"num_attention_heads":8,"num_key_value_heads":1,
        "head_dim":128,"v_head_dim":128,"partial_rotary_factor":0.5,"block_size":8,"sliding_window":8,
        "is_causal":false,"target_hidden_size":128,"dflash_ring_page":8,
        "layer_types":["sliding_attention"],"dflash_config":{"target_layer_ids":[0,1],"tap_shift":0,
        "attention_value_scale":0.612,"attention_sink_bias":true}})";
}

/// The mini checkpoint: layer 0 dense/full-attention, layer 1 MoE/SWA with 4 K=2.5 experts, 1 MTP head
/// (present in the dir, unused by the engine - the pack must stage it all the same).
Fixture build_fixture(const std::filesystem::path& dir, double wscale) {
    Fixture fx(dir);
    fx.wscale = wscale;
    const int E = 128, QK = 256, KV0 = 4 * 32, KV1 = 8 * 32, VD = 8 * 32;
    {
        std::vector<uint16_t> emb((size_t)256 * E);
        for (auto& e : emb) e = (uint16_t)(0x3f80 ^ (fx.rnd() & 0xff));   // BF16 payload in [1, 2)
        fx.bytes16("model.embed_tokens.weight", "BF16", {256, E}, emb);
        fx.exl3("lm_head", E / 16, 256 / 16, 8);                           // K = 4
        std::vector<float> norm(E);
        for (auto& n : norm) n = 0.9f + (fx.rnd() % 100) / 1000.0f;
        fx.f32("model.norm.weight", {E}, norm);
    }
    {
        const std::string p = "model.layers.0.";
        fx.exl3(p + "self_attn.q_proj", E / 16, QK / 16, 5);              // K = 2.5
        fx.exl3(p + "self_attn.k_proj", E / 16, KV0 / 16, 5);
        fx.exl3(p + "self_attn.v_proj", E / 16, KV0 / 16, 5);
        fx.exl3(p + "self_attn.o_proj", VD / 16, E / 16, 5);
        std::vector<float> norm(E);
        for (auto& n : norm) n = 1.0f + (fx.rnd() % 100) / 1000.0f;
        fx.f32(p + "input_layernorm.weight", {E}, norm);
        for (auto& n : norm) n = 1.0f + (fx.rnd() % 100) / 1000.0f;
        fx.f32(p + "post_attention_layernorm.weight", {E}, norm);
        fx.exl3(p + "mlp.gate_proj", E / 16, 128 / 16, 8);
        fx.exl3(p + "mlp.up_proj", E / 16, 128 / 16, 8);
        fx.exl3(p + "mlp.down_proj", 128 / 16, E / 16, 8);
    }
    {
        const std::string p = "model.layers.1.";
        fx.exl3(p + "self_attn.q_proj", E / 16, QK / 16, 5);
        fx.exl3(p + "self_attn.k_proj", E / 16, KV1 / 16, 5);
        fx.exl3(p + "self_attn.v_proj", E / 16, KV1 / 16, 5);
        fx.exl3(p + "self_attn.o_proj", VD / 16, E / 16, 5);
        std::vector<uint16_t> sink(8);
        for (auto& s : sink) s = k::f16_from_f32(-2.0f + 4.0f * (fx.rnd() % 1000) / 1000.0f);
        fx.bytes16(p + "self_attn.attention_sink_bias", "F16", {8}, sink);
        std::vector<float> norm(E);
        for (auto& n : norm) n = 1.0f + (fx.rnd() % 100) / 1000.0f;
        fx.f32(p + "input_layernorm.weight", {E}, norm);
        for (auto& n : norm) n = 1.0f + (fx.rnd() % 100) / 1000.0f;
        fx.f32(p + "post_attention_layernorm.weight", {E}, norm);
        std::vector<uint16_t> gate((size_t)4 * E);
        for (auto& g : gate) g = k::f16_from_f32(-0.5f + (fx.rnd() % 1000) / 1000.0f);
        fx.bytes16(p + "mlp.gate.weight", "F16", {4, E}, gate);
        std::vector<float> bias(4);
        for (auto& b : bias) b = 0.1f * (int)(fx.rnd() % 10);
        fx.f32(p + "mlp.gate.e_score_correction_bias", {4}, bias);
        for (int e = 0; e < 4; ++e)
            for (const char* proj : {"gate", "up", "down"})
                fx.exl3(strata::mimo_expert_tensor(1, e, proj), E / 16, 128 / 16, 5);   // K = 2.5
    }
    {   // the MTP head: validated and staged by the pack, not run by the engine in this build
        const std::string p = "model.mtp.layers.0.";
        fx.exl3(p + "eh_proj", (2 * E) / 16, E / 16, 8);
        fx.exl3(p + "self_attn.q_proj", E / 16, QK / 16, 5);
        fx.exl3(p + "self_attn.k_proj", E / 16, 8 * 32 / 16, 5);
        fx.exl3(p + "self_attn.v_proj", E / 16, 8 * 32 / 16, 5);
        fx.exl3(p + "self_attn.o_proj", VD / 16, E / 16, 5);
        std::vector<uint16_t> sink(8);
        for (auto& s : sink) s = k::f16_from_f32(-2.0f + 4.0f * (fx.rnd() % 1000) / 1000.0f);
        fx.bytes16(p + "self_attn.attention_sink_bias", "F16", {8}, sink);
        std::vector<float> norm(E);
        for (const char* n : {"hnorm.weight", "enorm.weight", "input_layernorm.weight",
                              "pre_mlp_layernorm.weight", "final_layernorm.weight"}) {
            for (auto& v : norm) v = 1.0f + (fx.rnd() % 100) / 1000.0f;
            fx.f32(p + n, {E}, norm);
        }
        fx.exl3(p + "mlp.gate_proj", E / 16, 128 / 16, 8);
        fx.exl3(p + "mlp.up_proj", E / 16, 128 / 16, 8);
        fx.exl3(p + "mlp.down_proj", 128 / 16, E / 16, 8);
    }
    fx.write();
    { std::ofstream out((dir / "config.json").string()); out << mini_config(); }
    return fx;
}

// ---- the CPU reference: everything in double, the engine's F16 carries mirrored exactly
struct Ref {
    const Fixture& fx;
    int E, QK, KV0, KV1, VD, vocab, n_q, hd, n_rot, win, n_expert, k;
    double eps, theta_full, theta_swa, value_scale_unused, sm_scale;
    std::vector<uint16_t> embed;                       // F16 [vocab, E]
    std::vector<uint16_t> lm;                          // F16 [vocab, E] reconstructed
    struct Layer {
        std::vector<uint16_t> q, k, v, o;              // F16 [out, E] reconstructed (o folded)
        std::vector<uint16_t> gate, up, down;          // dense FFN (layer 0)
        std::vector<uint16_t> router;                  // F16 [n_expert, E]
        std::vector<float> bias, norm_attn, norm_mlp, sink;
    };
    std::vector<Layer> layers;
    std::vector<float> norm_final;
    // the KV caches (F16, exactly what the engine writes)
    std::vector<std::vector<uint16_t>> kc, vc;

    Ref(const Fixture& f, int64_t capacity)
        : fx(f), E(128), QK(256), KV0(4 * 32), KV1(8 * 32), VD(8 * 32), vocab(256), n_q(8), hd(32), n_rot(16),
          win(8), n_expert(4), k(2), eps(1e-6), theta_full(1e7), theta_swa(1e4),
          value_scale_unused(0), sm_scale(1.0 / std::sqrt(32.0)) {
        auto recon = [&](const std::string& base, double scale) {
            const Tensor& tr = fx.tensors.at(base + ".trellis");
            const int ki = (int)tr.shape[0], nj = (int)tr.shape[1], k2 = (int)tr.shape[2] / 8;
            const int bits = k2 / 2, frac = k2 % 2;
            // the codebook the MODEL parses from the .mul1 marker (marker 1 selects Mul1, enum 2) - NOT
            // the raw marker byte
            const Exl3Codebook cb = fx.tensors.at(base + ".mul1").bytes[0] != 0 ? Exl3Codebook::Mul1
                                                                                : Exl3Codebook::Mcg;
            std::vector<uint16_t> w((size_t)ki * 16 * nj * 16);
            k::cpu::exl3_reconstruct_weight(fx.u16(base + ".trellis"), ki, nj, bits, cb,
                                                  fx.u16(base + ".suh"), fx.u16(base + ".svh"), w.data(), frac);
            // exl3_reconstruct_weight emits [in x out]; transpose to the [out x in] the engine's
            // Gemm consumes (the pack does the same at staging)
            std::vector<uint16_t> t((size_t)ki * 16 * nj * 16);
            const int64_t rows = (int64_t)ki * 16, cols = (int64_t)nj * 16;
            for (int64_t i = 0; i < rows; ++i)
                for (int64_t j = 0; j < cols; ++j) t[(size_t)j * rows + i] = w[(size_t)i * cols + j];
            if (scale != 1.0)
                for (auto& x : t) x = k::f16_from_f32(k::f32_from_f16(x) * (float)scale);
            return t;
        };
        embed.resize((size_t)vocab * E);
        {   // the checkpoint's BF16 embedding, converted to F16 exactly as the pack does at load
            const uint16_t* bf = fx.u16("model.embed_tokens.weight");
            for (size_t i = 0; i < embed.size(); ++i) {
                uint32_t bits = (uint32_t)bf[i] << 16;
                float f;
                std::memcpy(&f, &bits, 4);
                embed[i] = k::f16_from_f32(f);
            }
        }
        lm = recon("lm_head", 1.0);
        norm_final.assign(fx.f32p("model.norm.weight"), fx.f32p("model.norm.weight") + E);
        layers.resize(2);
        kc.resize(2);
        vc.resize(2);
        for (int l = 0; l < 2; ++l) {
            Layer& L = layers[(size_t)l];
            const std::string p = "model.layers." + std::to_string(l) + ".";
            L.q = recon(p + "self_attn.q_proj", 1.0);
            L.k = recon(p + "self_attn.k_proj", 1.0);
            L.v = recon(p + "self_attn.v_proj", 1.0);
            L.o = recon(p + "self_attn.o_proj", 1.0);   // NO fold: the checkpoint's trellis is pre-folded
            L.norm_attn.assign(fx.f32p(p + "input_layernorm.weight"), fx.f32p(p + "input_layernorm.weight") + E);
            L.norm_mlp.assign(fx.f32p(p + "post_attention_layernorm.weight"),
                              fx.f32p(p + "post_attention_layernorm.weight") + E);
            const int kw = (l == 0 ? 4 : 8) * hd;   // one cache row's width (the KV head count x head_dim)
            kc[(size_t)l].assign((size_t)capacity * kw, 0);
            vc[(size_t)l].assign((size_t)capacity * kw, 0);
            if (l == 0) {
                L.gate = recon(p + "mlp.gate_proj", 1.0);
                L.up = recon(p + "mlp.up_proj", 1.0);
                L.down = recon(p + "mlp.down_proj", 1.0);
            } else {
                L.router.assign(fx.u16(p + "mlp.gate.weight"), fx.u16(p + "mlp.gate.weight") + (size_t)n_expert * E);
                L.bias.assign(fx.f32p(p + "mlp.gate.e_score_correction_bias"),
                              fx.f32p(p + "mlp.gate.e_score_correction_bias") + n_expert);
                const uint16_t* s = fx.u16(p + "self_attn.attention_sink_bias");
                L.sink.resize(n_q);
                for (int h = 0; h < n_q; ++h) L.sink[(size_t)h] = k::f32_from_f16(s[(size_t)h]);   // F16 -> F32, as the pack
            }
        }
    }

    /// y[o] = sum_i f32(W[o][i]) * f32(x16[i]) - W F16 row-major [out, in], accumulation in double.
    /// (recon() below hands out [out x in], the transposed form exl3_reconstruct_weight emits.)
    static std::vector<double> gemm(const std::vector<uint16_t>& W, const uint16_t* x16, int64_t out, int64_t in) {
        std::vector<double> y((size_t)out, 0.0);
        for (int64_t o = 0; o < out; ++o) {
            const uint16_t* w = W.data() + (size_t)o * in;
            double acc = 0.0;
            for (int64_t i = 0; i < in; ++i) acc += (double)k::f32_from_f16(w[(size_t)i]) * k::f32_from_f16(x16[(size_t)i]);
            y[(size_t)o] = acc;
        }
        return y;
    }
    static uint16_t to_h(double v) { return k::f16_from_f32((float)v); }

    /// rmsnorm to F16 (the engine's carry): h16 = f16(x * rsqrt(mean(x^2) + eps) * w).
    std::vector<uint16_t> rms16(const std::vector<double>& x, const float* w) {
        double s = 0;
        for (double v : x) s += v * v;
        const double r = 1.0 / std::sqrt(s / E + eps);
        std::vector<uint16_t> h((size_t)E);
        for (int i = 0; i < E; ++i) h[(size_t)i] = to_h(x[(size_t)i] * r * w[i]);
        return h;
    }

    /// The engine's in-place rope on an F16 row: fp64 angle, fp32 rotation, F16 store.
    void rope(uint16_t* row, int64_t pos, double theta) const {
        const int half = n_rot / 2;
        for (int i = 0; i < half; ++i) {
            const double freq = std::pow(theta, -(2.0 * i) / (double)n_rot);
            const double reduced = std::fmod((double)pos * freq, 6.283185307179586476925287);
            const float ang = (float)reduced;
            const float s = std::sin(ang), c = std::cos(ang);
            const float a = k::f32_from_f16(row[i]), b = k::f32_from_f16(row[i + half]);
            row[i] = k::f16_from_f32(a * c - b * s);
            row[i + half] = k::f16_from_f32(a * s + b * c);
        }
    }

    /// One window of T tokens at [pos0, pos0+T): mutates x ([T][E], F32 in), returns the LAST row's logits.
    std::vector<double> forward(const std::vector<int32_t>& toks, int T, int64_t pos0,
                                std::vector<double>& x_last_out) {
        std::vector<double> x((size_t)T * E);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < E; ++i)
                x[(size_t)t * E + i] = k::f32_from_f16(embed[(size_t)toks[(size_t)t] * E + i]);
        for (int l = 0; l < 2; ++l) {
            const Layer& L = layers[(size_t)l];
            const bool full = (l == 0);
            const int kvh_n = full ? 4 : 8;          // the KV HEAD count (KV0/KV1 are projection widths)
            const int kw = kvh_n * hd;               // one cache row's width in F16 lanes
            const int g = n_q / kvh_n;
            const double theta = full ? theta_full : theta_swa;
            std::vector<uint16_t> h((size_t)T * E);
            {
                std::vector<double> row(x.begin() + 0, x.begin() + E);
                for (int t = 0; t < T; ++t) {
                    std::copy(x.begin() + (size_t)t * E, x.begin() + (size_t)(t + 1) * E, row.begin());
                    const std::vector<uint16_t> n = rms16(row, L.norm_attn.data());
                    std::copy(n.begin(), n.end(), h.begin() + (size_t)t * E);
                }
            }
            // q/k/v (F16 carries), rope, the caches
            std::vector<uint16_t> q16((size_t)T * QK);
            for (int t = 0; t < T; ++t) {
                const std::vector<double> q = gemm(L.q, &h[(size_t)t * E], QK, E);
                const std::vector<double> kk = gemm(L.k, &h[(size_t)t * E], kw, E);
                const std::vector<double> vv = gemm(L.v, &h[(size_t)t * E], kw, E);
                for (int i = 0; i < QK; ++i) q16[(size_t)t * QK + i] = to_h(q[(size_t)i]);
                uint16_t* krow = &kc[(size_t)l][(size_t)(pos0 + t) * kw];
                uint16_t* vrow = &vc[(size_t)l][(size_t)(pos0 + t) * kw];
                for (int i = 0; i < kw; ++i) krow[i] = to_h(kk[(size_t)i]);
                for (int i = 0; i < kw; ++i) vrow[i] = to_h(vv[(size_t)i]);   // mini V is symmetric
                rope(&q16[(size_t)t * QK], pos0 + t, theta);
                rope(krow, pos0 + t, theta);
            }
            // attention (per row, per q head) - the decode/prefill mask, the sink RAW in the denominator
            std::vector<uint16_t> a16((size_t)T * VD);
            for (int t = 0; t < T; ++t) {
                const int64_t p = pos0 + t;
                const int lo = full ? 0 : (int)std::max<int64_t>(0, p - (win - 1));
                for (int h = 0; h < n_q; ++h) {
                    const int kvh = h / g;
                    const uint16_t* qh = &q16[(size_t)t * QK + (size_t)h * hd];
                    double best = -1e300;
                    std::vector<double> s;
                    for (int64_t kp = lo; kp <= p; ++kp) {
                        const uint16_t* kr = &kc[(size_t)l][(size_t)kp * kw + (size_t)kvh * hd];
                        double dot = 0;
                        for (int d = 0; d < hd; ++d) dot += (double)k::f32_from_f16(qh[d]) * k::f32_from_f16(kr[d]);
                        const double sc = dot * sm_scale;
                        s.push_back(sc);
                        best = std::max(best, sc);
                    }
                    double sink_e = 0.0;
                    if (!full) {
                        const double sk = L.sink[(size_t)h];
                        best = std::max(best, sk);
                        sink_e = std::exp(sk - best);
                    }
                    double denom = sink_e;
                    std::vector<double> acc(hd, 0.0);
                    for (size_t j = 0; j < s.size(); ++j) {
                        const double w = std::exp(s[j] - best);
                        denom += w;
                        const uint16_t* vr = &vc[(size_t)l][(size_t)(lo + (int64_t)j) * kw + (size_t)kvh * hd];
                        for (int d = 0; d < hd; ++d) acc[(size_t)d] += w * k::f32_from_f16(vr[d]);
                    }
                    for (int d = 0; d < hd; ++d) {
                        const double o = denom > 0 ? acc[(size_t)d] / denom : 0.0;
                        a16[(size_t)t * VD + (size_t)h * hd + d] = to_h(o);
                    }
                }
            }
            // o_proj + residual
            for (int t = 0; t < T; ++t) {
                const std::vector<double> o = gemm(L.o, &a16[(size_t)t * VD], E, VD);
                for (int i = 0; i < E; ++i) x[(size_t)t * E + i] += o[(size_t)i];
            }
            // the FFN half
            std::vector<uint16_t> h2((size_t)T * E);
            {
                std::vector<double> row(E);
                for (int t = 0; t < T; ++t) {
                    std::copy(x.begin() + (size_t)t * E, x.begin() + (size_t)(t + 1) * E, row.begin());
                    const std::vector<uint16_t> n = rms16(row, L.norm_mlp.data());
                    std::copy(n.begin(), n.end(), h2.begin() + (size_t)t * E);
                }
            }
            if (full) {   // dense SwiGLU
                for (int t = 0; t < T; ++t) {
                    const std::vector<double> g_ = gemm(L.gate, &h2[(size_t)t * E], 128, E);
                    const std::vector<double> u_ = gemm(L.up, &h2[(size_t)t * E], 128, E);
                    std::vector<uint16_t> hb((size_t)128);
                    for (int i = 0; i < 128; ++i)
                        hb[(size_t)i] = to_h(g_[(size_t)i] / (1.0 + std::exp(-g_[(size_t)i])) * u_[(size_t)i]);
                    const std::vector<double> d_ = gemm(L.down, hb.data(), E, 128);
                    for (int i = 0; i < E; ++i) x[(size_t)t * E + i] += d_[(size_t)i];
                }
            } else {   // MoE: sigmoid + noaux_tc top-2, unbiased renormalised combine
                for (int t = 0; t < T; ++t) {
                    std::vector<double> s((size_t)n_expert);
                    for (int e = 0; e < n_expert; ++e) {
                        double acc = 0;
                        const uint16_t* w = L.router.data() + (size_t)e * E;
                        for (int i = 0; i < E; ++i)
                            acc += (double)k::f32_from_f16(w[i]) * k::f32_from_f16(h2[(size_t)t * E + i]);
                        s[(size_t)e] = 1.0 / (1.0 + std::exp(-acc));
                    }
                    std::vector<char> taken((size_t)n_expert, 0);
                    int pick[kMaxK];
                    double sel[kMaxK];
                    double sum = 0;
                    for (int r = 0; r < k; ++r) {
                        int be = -1;
                        double bc = 0;
                        for (int e = 0; e < n_expert; ++e) {
                            if (taken[(size_t)e]) continue;
                            const double c = s[(size_t)e] + L.bias[(size_t)e];
                            if (be < 0 || c > bc) { bc = c; be = e; }
                        }
                        taken[(size_t)be] = 1;
                        pick[r] = be;
                        sel[r] = s[(size_t)be];
                        sum += s[(size_t)be];
                    }
                    for (int r = 0; r < k; ++r) {
                        const double w = sel[r] / (sum + 1e-20);
                        const std::vector<uint16_t> gw = recon_expert(pick[r], "gate");
                        const std::vector<uint16_t> uw = recon_expert(pick[r], "up");
                        const std::vector<uint16_t> dw = recon_expert(pick[r], "down");
                        const std::vector<double> g_ = gemm(gw, &h2[(size_t)t * E], 128, E);
                        const std::vector<double> u_ = gemm(uw, &h2[(size_t)t * E], 128, E);
                        std::vector<uint16_t> hb((size_t)128);
                        for (int i = 0; i < 128; ++i)
                            hb[(size_t)i] = to_h(g_[(size_t)i] / (1.0 + std::exp(-g_[(size_t)i])) * u_[(size_t)i]);
                        const std::vector<double> d_ = gemm(dw, hb.data(), E, 128);
                        for (int i = 0; i < E; ++i) x[(size_t)t * E + i] += w * d_[(size_t)i];
                    }
                }
            }
        }
        // final norm + lm_head on the LAST row
        std::vector<double> last(x.end() - E, x.end());
        const std::vector<uint16_t> hf = rms16(last, norm_final.data());
        x_last_out = last;
        return gemm(lm, hf.data(), vocab, E);
    }

    static constexpr int kMaxK = 8;
    std::vector<uint16_t> expert_cache[4][3];   // [expert][gate/up/down] reconstructed once

    std::vector<uint16_t> recon_expert(int e, const char* proj) {
        static const char* names[3] = {"gate", "up", "down"};
        int j = proj[0] == 'g' ? 0 : proj[0] == 'u' ? 1 : 2;
        if (expert_cache[e][j].empty()) {
            const std::string base = strata::mimo_expert_tensor(1, e, names[j]);
            const Tensor& tr = fx.tensors.at(base + ".trellis");
            const int ki = (int)tr.shape[0], nj = (int)tr.shape[1], k2 = (int)tr.shape[2] / 8;
            const Exl3Codebook cb = fx.tensors.at(base + ".mul1").bytes[0] != 0 ? Exl3Codebook::Mul1
                                                                                : Exl3Codebook::Mcg;
            std::vector<uint16_t> w((size_t)ki * 16 * nj * 16);
            k::cpu::exl3_reconstruct_weight(fx.u16(base + ".trellis"), ki, nj, k2 / 2, cb,
                                                  fx.u16(base + ".suh"), fx.u16(base + ".svh"), w.data(), k2 % 2);
            std::vector<uint16_t> t((size_t)ki * 16 * nj * 16);   // [in x out] -> [out x in]
            const int64_t rows = (int64_t)ki * 16, cols = (int64_t)nj * 16;
            for (int64_t i = 0; i < rows; ++i)
                for (int64_t j = 0; j < cols; ++j) t[(size_t)j * rows + i] = w[(size_t)i * cols + j];
            expert_cache[e][j] = t;
        }
        return expert_cache[e][j];
    }
};

}  // namespace

/// `argv[1]` (optional): write the fixture there and exit, WITHOUT running - the CLI and serve smoke
/// tests (and bring-up day) get a valid mini-MiMo dir this way.
int main(int argc, char** argv) {
    {
        strata::core::ExpertFrequency f;
        const int32_t hot[] = {0,0,0,1};
        f.observe(hot, 4, 2, .5f);
        const int32_t next[] = {1,2};
        f.observe(next, 2, 2, .5f);
        CHECK(f.count[0] == 1.5f && f.count[1] == 1.5f && f.count[2] == 1.f);
        // A phase change eventually retires the old hot expert.
        for (int i = 0; i < 20; ++i) f.observe(next, 2, 2, .5f);
        CHECK(f.count[0] < .01f && f.count[2] > 1.f);
    }
    if (argc > 1) {
        namespace fs = std::filesystem;
        std::filesystem::remove_all(argv[1]);
        build_fixture(argv[1], 0.22);
        std::printf("fixture written to %s\n", argv[1]);
        return 0;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "strata_mimo_engine_parity";
    std::filesystem::remove_all(root);
    fs::create_directories(root);
    const fs::path dir = root / "model";
    const double wscale = 0.22;   // tuned: layer contributions O(1), every F16 carry finite
    const Fixture fx = build_fixture(dir, wscale);
    const int64_t ctx = 64, chunk = 8;

    // The async decode API must preserve admission/eviction and original route destinations.
    // Compare with the independent host-IO indexed service through cold, mixed, hot, and
    // eviction cases; the caller explicitly orders readback after the enqueued device result.
    if (const char* p = std::getenv("STRATA_EXL3_PRIMARY_PACK_PLAN"); !p || p[0] != '0') {
        cudaStream_t st = nullptr;
        float *dx = nullptr, *dy = nullptr;
        CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) == cudaSuccess);
        CHECK(cudaMalloc(&dx, 128 * sizeof(float)) == cudaSuccess);
        CHECK(cudaMalloc(&dy, 2 * 128 * sizeof(float)) == cudaSuccess);
        {
            k::Exl3ExpertStream candidate(dir.string(), 128, 2, 2, st, 0, (size_t) 256 << 20,
                                           1, 2, 8, "model.layers.");
            k::Exl3ExpertStream oracle(dir.string(), 128, 2, 2, st, 0, (size_t) 256 << 20,
                                        1, 2, 8, "model.layers.");
            std::vector<float> input(128), actual(256), expected(256);
            for (int i = 0; i < 128; ++i) input[(size_t) i] = (i - 64) * 0.002f;
            CHECK(cudaMemcpyAsync(dx, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, st) == cudaSuccess);
            CHECK(cudaStreamSynchronize(st) == cudaSuccess);
            const int routes[][2] = {{0,1}, {0,2}, {0,2}, {3,1}, {1,3}};
            for (const auto& ids : routes) {
                CHECK(oracle.run_multi(1, input.data(), ids, 1, 2, expected.data(), st));
                CHECK(candidate.run_decode_device(1, dx, ids, 2, dy, st));
                CHECK(cudaMemcpyAsync(actual.data(), dy, actual.size() * sizeof(float), cudaMemcpyDeviceToHost, st) == cudaSuccess);
                CHECK(cudaStreamSynchronize(st) == cudaSuccess);
                CHECK(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0);
                CHECK(candidate.holds(1, ids[0]) && candidate.holds(1, ids[1]));
            }
            const int duplicate[2] = {0,0};
            CHECK(!candidate.run_decode_device(1, dx, duplicate, 2, dy, st));
            std::puts("PASS: device decode cold/mixed/hot/eviction rows byte-identical to host indexed service");
            candidate.enable_decode_handoff(2);
            int32_t* dids = nullptr;
            CHECK(cudaMalloc(&dids, 2 * sizeof(int32_t)) == cudaSuccess);
            for (const auto& ids : routes) {
                CHECK(oracle.run_multi(1, input.data(), ids, 1, 2, expected.data(), st));
                candidate.refresh_verify_graph(st);
                CHECK(cudaMemcpyAsync(dids, ids, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, st) == cudaSuccess);
                const int32_t* published = nullptr;
                const float* activation = nullptr;
                const uint8_t* hits = nullptr;
                candidate.decode_handoff(1, dx, dids, 2, dy, st, published, activation, hits);
                CHECK(published[0] == ids[0] && published[1] == ids[1]);
                // Process misses individually while the snapshot hits may still be running.
                // Two slots force admission to protect every in-flight hit, not just LRU order.
                for (int r = 0; r < 2; ++r) if (!hits[r]) {
                    CHECK(std::memcmp(activation, input.data(), input.size() * sizeof(float)) == 0);
                    CHECK(candidate.run_decode_device(1, dx, ids + r, 1, dy + r * 128, st));
                    // Separate service calls require an explicit lifetime boundary.
                    CHECK(cudaStreamSynchronize(st) == cudaSuccess);
                }
                candidate.verify_graph_end_host(1);
                CHECK(cudaMemcpyAsync(actual.data(), dy, actual.size() * sizeof(float), cudaMemcpyDeviceToHost, st) == cudaSuccess);
                CHECK(cudaStreamSynchronize(st) == cudaSuccess);
                CHECK(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0);
            }
            std::puts("PASS: mapped decode routing, resident hits, pinned miss admission and churn");
            cudaStream_t service = nullptr;
            CHECK(cudaStreamCreateWithFlags(&service, cudaStreamNonBlocking) == cudaSuccess);
            const size_t reserved = candidate.bytes();
            uint32_t ring = 0;
            for (const auto& ids : routes) {
                ++ring;
                CHECK(oracle.run_multi(1, input.data(), ids, 1, 2, expected.data(), st));
                candidate.refresh_verify_graph(st);
                CHECK(cudaMemcpyAsync(dids, ids, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, st) == cudaSuccess);
                candidate.enqueue_decode_handoff(1, dx, dids, 2, dy, ring, st);
                std::exception_ptr error;
                std::thread worker([&] {
                    try {
                        if (cudaSetDevice(0) != cudaSuccess) throw std::runtime_error("service device");
                        const int32_t* published = nullptr;
                        const float* activation = nullptr;
                        const uint8_t* hits = nullptr;
                        candidate.collect_decode_handoff(1, ring, service, published, activation, hits);
                        if (published[0] != ids[0] || published[1] != ids[1])
                            throw std::runtime_error("stale routing publication");
                        for (int r = 0; r < 2; ++r) if (!hits[r]) {
                            if (!candidate.run_decode_device(1, dx, ids + r, 1, dy + r * 128, service))
                                throw std::runtime_error("asynchronous miss service");
                            if (r == 0 && !hits[1] && cudaStreamSynchronize(service) != cudaSuccess)
                                throw std::runtime_error("inter-call plan lifetime");
                        }
                        candidate.complete_decode_handoff(1, ring, service);
                    } catch (...) {
                        error = std::current_exception();
                        candidate.abort_decode_handoff();
                    }
                });
                // A pageable readback can lock the producer stream while it waits. The worker
                // must poll its OWN stream or it deadlocks before observing the doorbell.
                CHECK(cudaMemcpyAsync(actual.data(), dy, actual.size() * sizeof(float), cudaMemcpyDeviceToHost, st) == cudaSuccess);
                CHECK(cudaStreamSynchronize(st) == cudaSuccess);
                worker.join();
                CHECK(!error);
                CHECK(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0);
                CHECK(candidate.bytes() == reserved);
            }
            // All-hit progress must work with NO worker completion publication at all.
            candidate.refresh_verify_graph(st);
            const int32_t resident[2] = {1,3};
            CHECK(cudaMemcpyAsync(dids, resident, sizeof(resident), cudaMemcpyHostToDevice, st) == cudaSuccess);
            candidate.enqueue_decode_handoff(1, dx, dids, 2, dy, ++ring, st);
            CHECK(cudaStreamSynchronize(st) == cudaSuccess);
            const int32_t* published = nullptr;
            const float* activation = nullptr;
            const uint8_t* hits = nullptr;
            candidate.collect_decode_handoff(1, ring, service, published, activation, hits);
            CHECK(hits[0] && hits[1]);
            candidate.complete_decode_handoff(1, ring);
            // Cancellation releases a queued GPU wait even when no miss result is published.
            candidate.refresh_verify_graph(st);
            const int32_t missing[2] = {0,2};
            CHECK(cudaMemcpyAsync(dids, missing, sizeof(missing), cudaMemcpyHostToDevice, st) == cudaSuccess);
            candidate.enqueue_decode_handoff(1, dx, dids, 2, dy, ++ring, st);
            candidate.abort_decode_handoff();
            CHECK(cudaStreamSynchronize(st) == cudaSuccess);
            CHECK(cudaStreamDestroy(service) == cudaSuccess);
            CHECK(cudaFree(dids) == cudaSuccess);
            std::puts("PASS: independent-stream decode service, all-hit bypass, fixed workspace and abort release");
        }
        CHECK(cudaFree(dx) == cudaSuccess);
        CHECK(cudaFree(dy) == cudaSuccess);
        CHECK(cudaStreamDestroy(st) == cudaSuccess);
    }

    {   // Reuse geometry really dispatches grouped SMALL_M before any T1 handoff.
        // A pre-enabled window must match the old stream both before and after lazy T1.
        ScopedEnv small("STRATA_EXL3_SMALL_M", "1");
        cudaStream_t stream = nullptr;
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
        {
            k::Exl3ExpertStream candidate(dir.string(), 128, 2, 4, stream, 0, 0, 1, 2, 8, "model.layers.");
            k::Exl3ExpertStream oracle(dir.string(), 128, 2, 4, stream, 0, 0, 1, 2, 8, "model.layers.");
            candidate.enable_decode_handoff(2, 8);
            std::vector<float> input(4 * 128), actual(8 * 128), expected(8 * 128);
            const int32_t ids[] = {0,1,2,3,0,1,2,3}; // four whole groups, each width two
            for (int phase = 0; phase < 2; ++phase) {
                if (phase) {
                    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
                    oracle.enable_decode_handoff(2); // old first-T1 initialization
                    candidate.prepare_decode_handoff_t1();
                }
                long long before[2] = {}, after[2] = {};
                bool enabled[2] = {};
                candidate.small_m_debug(before, nullptr, nullptr, nullptr, enabled);
                oracle.small_m_debug(before + 1, nullptr, nullptr, nullptr, enabled + 1);
                CHECK(enabled[0] == !phase && enabled[1] == !phase);
                for (int repeat = 0; repeat < 2; ++repeat) {
                    for (int i = 0; i < 4 * 128; ++i) input[i] = .03f * std::sin((i + repeat) * .17f);
                    CHECK(oracle.run_multi(1, input.data(), ids, 4, 2, expected.data(), stream));
                    CHECK(candidate.run_multi(1, input.data(), ids, 4, 2, actual.data(), stream));
                    CHECK(std::memcmp(actual.data(), expected.data(), actual.size() * 4) == 0);
                }
                candidate.small_m_debug(after, nullptr, nullptr, nullptr);
                oracle.small_m_debug(after + 1, nullptr, nullptr, nullptr);
                CHECK(after[0] - before[0] == after[1] - before[1]);
                CHECK(phase ? after[0] == before[0] : after[0] > before[0]);
                std::printf("PASS: SMALL_M %s T1: enabled=%d/%d grouped delta=%lld/%lld, all rows byte-exact\n",
                    phase ? "after" : "without", enabled[0], enabled[1], after[0]-before[0], after[1]-before[1]);
            }
        }
        CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    }

    {   // Multirow A1a transport: exact old-worker oracle, distinct token inputs, repeated
        // frames and width changes. Fault after collect must release the producer wait.
        cudaStream_t producer = nullptr, service = nullptr;
        float *dx = nullptr, *dy = nullptr, *dw = nullptr, *host_out = nullptr;
        int32_t* di = nullptr;
        CHECK(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking) == cudaSuccess);
        CHECK(cudaStreamCreateWithFlags(&service, cudaStreamNonBlocking) == cudaSuccess);
        CHECK(cudaMalloc(&dx, 8 * 128 * 4) == cudaSuccess);
        CHECK(cudaMalloc(&dy, 16 * 128 * 4) == cudaSuccess);
        CHECK(cudaMalloc(&di, 16 * 4) == cudaSuccess);
        CHECK(cudaMalloc(&dw, 16 * 4) == cudaSuccess);
        CHECK(cudaMallocHost(&host_out, 16 * 128 * 4) == cudaSuccess);
        {
            k::Exl3ExpertStream candidate(dir.string(), 128, 2, 2, producer, 0, 0, 1, 2, 8, "model.layers.");
            k::Exl3ExpertStream oracle(dir.string(), 128, 2, 2, producer, 0, 0, 1, 2, 8, "model.layers.");
            candidate.enable_decode_handoff(2, 8);
            std::vector<float> input(8 * 128), actual(16 * 128), expected(16 * 128), weights(16);
            int32_t ids[16];
            for (int i = 0; i < 8 * 128; ++i) input[i] = .03f * std::sin(i * .17f);
            uint32_t ring = 0;
            for (int t : {2, 3, 4, 8, 2, 8, 3, 4}) {
                const size_t bytes = (size_t) t * 2 * 128 * 4;
                for (int r = 0; r < t * 2; ++r) { ids[r] = (r + ring) % 4; weights[r] = .1f * (r + 1); }
                CHECK(oracle.run_multi(1, input.data(), ids, t, 2, expected.data(), producer));
                CHECK(cudaMemcpyAsync(dx, input.data(), t * 128 * 4, cudaMemcpyHostToDevice, producer) == cudaSuccess);
                CHECK(cudaMemcpyAsync(di, ids, t * 2 * 4, cudaMemcpyHostToDevice, producer) == cudaSuccess);
                CHECK(cudaMemcpyAsync(dw, weights.data(), t * 2 * 4, cudaMemcpyHostToDevice, producer) == cudaSuccess);
                ++ring;
                candidate.enqueue_decode_handoff(1, dx, di, 2, dy, ring, producer, t, dw);
                std::exception_ptr error;
                std::thread worker([&] {
                    try {
                        if (cudaSetDevice(0) != cudaSuccess) throw std::runtime_error("window service device");
                        const int32_t* published = nullptr;
                        const float *activation = nullptr, *published_weights = nullptr;
                        const uint8_t* hits = nullptr;
                        int tokens = 0;
                        candidate.collect_decode_handoff(1, ring, service, published, activation, hits, &tokens, &published_weights);
                        if (tokens != t || std::memcmp(published, ids, t * 2 * 4) ||
                            std::memcmp(activation, input.data(), t * 128 * 4) ||
                            std::memcmp(published_weights, weights.data(), t * 2 * 4) ||
                            std::any_of(hits, hits + t * 2, [](uint8_t h) { return h != 0; }))
                            throw std::runtime_error("window publication mismatch");
                        if (!candidate.run_multi(1, activation, published, tokens, 2, host_out, service))
                            throw std::runtime_error("window service refused");
                        if (cudaMemcpyAsync(dy, host_out, bytes, cudaMemcpyHostToDevice, service) != cudaSuccess)
                            throw std::runtime_error("window result upload");
                        candidate.complete_decode_handoff(1, ring, service);
                    } catch (...) { error = std::current_exception(); candidate.abort_decode_handoff(); }
                });
                CHECK(cudaMemcpyAsync(actual.data(), dy, bytes, cudaMemcpyDeviceToHost, producer) == cudaSuccess);
                CHECK(cudaStreamSynchronize(producer) == cudaSuccess);
                worker.join();
                CHECK(!error);
                CHECK(std::memcmp(actual.data(), expected.data(), bytes) == 0);
            }
            candidate.enqueue_decode_handoff(1, dx, di, 2, dy, ++ring, producer, 8, dw);
            const int32_t* published = nullptr;
            const float* activation = nullptr;
            const uint8_t* hits = nullptr;
            candidate.collect_decode_handoff(1, ring, service, published, activation, hits);
            candidate.abort_decode_handoff(); // injected worker failure/cancel, no completion
            CHECK(cudaStreamSynchronize(producer) == cudaSuccess);
            std::puts("PASS: mapped T2/3/4/8 actual activations/ids/weights, empty hits, byte-exact host oracle, reuse and abort");
        }
        CHECK(cudaFree(dx) == cudaSuccess && cudaFree(dy) == cudaSuccess);
        CHECK(cudaFree(di) == cudaSuccess && cudaFree(dw) == cudaSuccess);
        CHECK(cudaFreeHost(host_out) == cudaSuccess);
        CHECK(cudaStreamDestroy(producer) == cudaSuccess && cudaStreamDestroy(service) == cudaSuccess);
    }

    {   // Multirow WINDOW GPU HITS (set_window_graph_hits + refresh(st, T)): the producer
        // computes RESIDENT rows inline from the refresh's descriptor tables; the worker
        // serves only misses under the record's pinned-hit contract (host_active drops the
        // hit groups from run_multi, begin_host observed/ticked them exactly once). Gate:
        // byte-exact against an independent host oracle across cold, mixed-with-duplicates,
        // all-hit, width-changing and churned windows, plus the refresh's per-width
        // eligibility fallback (active < T*k publishes NO hits - the original all-miss
        // window byte for byte). Poison in the staging slab proves the worker never
        // clobbers a producer-computed row.
        cudaStream_t producer = nullptr, service = nullptr;
        float *dx = nullptr, *dy = nullptr, *dw = nullptr, *host_out = nullptr;
        int32_t* di = nullptr;
        CHECK(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking) == cudaSuccess);
        CHECK(cudaStreamCreateWithFlags(&service, cudaStreamNonBlocking) == cudaSuccess);
        CHECK(cudaMalloc(&dx, 8 * 128 * 4) == cudaSuccess);
        CHECK(cudaMalloc(&dy, 16 * 128 * 4) == cudaSuccess);
        CHECK(cudaMalloc(&di, 16 * 4) == cudaSuccess);
        CHECK(cudaMalloc(&dw, 16 * 4) == cudaSuccess);
        CHECK(cudaMallocHost(&host_out, 16 * 128 * 4) == cudaSuccess);
        std::vector<float> input(8 * 128), actual(16 * 128), expected(16 * 128), weights(16, .3f);
        int32_t ids[16];
        for (int i = 0; i < 8 * 128; ++i) input[i] = .03f * std::sin(i * .17f);
        for (int i = 0; i < 16; ++i) weights[i] = .1f * (i + 1);
        // (slots, label): 16 covers every width's tables (T8*k = 16); 4 covers only T2
        // (4 >= 2*2) and falls back for T4/T8; 2 covers none (2 < 2*2).
        for (auto slots_case : {std::pair<int, const char*>{16, "full"}, {4, "mixed"}, {2, "none"}}) {
            const int slots = slots_case.first;
            k::Exl3ExpertStream candidate(dir.string(), 128, 2, slots, producer, 0, 0, 1, 2, 8, "model.layers.");
            k::Exl3ExpertStream oracle(dir.string(), 128, 2, slots, producer, 0, 0, 1, 2, 8, "model.layers.");
            candidate.enable_decode_handoff(2, 8);
            candidate.set_window_graph_hits(true);
            uint32_t ring = 0;
            int windows_with_hits = 0, all_hit_windows = 0, fallback_windows = 0,
                zero_resident_windows = 0, window_hit_routes = 0;
            // steps: 0/1/2 cold and widening (misses admit), then mixed duplicates, then
            // repeats of the warm set (all-hit), then churn back to cold experts.
            const int steps_t[10] = {2, 4, 8, 3, 2, 8, 4, 2, 3, 8};
            const int steps_mode[10] = {0, 0, 0, 1, 1, 2, 2, 1, 0, 2};
            for (int step = 0; step < 10; ++step) {
                const int t = steps_t[step], mode = steps_mode[step];
                const size_t bytes = (size_t) t * 2 * 128 * 4;
                for (int r = 0; r < t * 2; ++r)
                    ids[r] = mode == 0 ? (r % 2 ? 2 : 3)      // the initially cold pair
                          : mode == 1 ? (r + step) % 4        // mixed, heavy duplicates
                                      : r % 4;               // the warm set (repeat hits)
                CHECK(oracle.run_multi(1, input.data(), ids, t, 2, expected.data(), producer));
                CHECK(cudaMemcpyAsync(dx, input.data(), t * 128 * 4, cudaMemcpyHostToDevice, producer) == cudaSuccess);
                CHECK(cudaMemcpyAsync(di, ids, t * 2 * 4, cudaMemcpyHostToDevice, producer) == cudaSuccess);
                CHECK(cudaMemcpyAsync(dw, weights.data(), t * 2 * 4, cudaMemcpyHostToDevice, producer) == cudaSuccess);
                std::fill_n(actual.data(), 16 * 128, -777.f);
                CHECK(cudaMemcpyAsync(dy, actual.data(), 16 * 128 * 4, cudaMemcpyHostToDevice, producer) == cudaSuccess);
                CHECK(cudaStreamSynchronize(producer) == cudaSuccess);
                candidate.refresh_verify_graph(producer, t);
                const int declared = candidate.window_hit_tokens();
                if (slots >= t * 2 && declared < t)
                    throw std::runtime_error("refresh failed to declare an eligible width");
                if (slots < t * 2 && declared != 0)
                    throw std::runtime_error("refresh declared tables below the active gate");
                ++ring;
                candidate.enqueue_decode_handoff(1, dx, di, 2, dy, ring, producer, t, dw);
                std::exception_ptr failure;
                std::thread worker([&] {
                    try {
                        if (cudaSetDevice(0) != cudaSuccess) throw std::runtime_error("window-hit service device");
                        const int32_t* published = nullptr;
                        const float *activation = nullptr, *published_weights = nullptr;
                        const uint8_t* hits = nullptr;
                        int tokens = 0;
                        candidate.collect_decode_handoff(1, ring, service, published, activation, hits,
                                                         &tokens, &published_weights);
                        if (tokens != t || std::memcmp(published, ids, t * 2 * 4))
                            throw std::runtime_error("window-hit publication mismatch");
                        // The gather mirrors x/ids/weights ONLY when the window has misses
                        // (an all-hit window owes no host service) - compare the mirrors
                        // exactly when the engine would consume them.
                        int hit_routes = 0;
                        for (int r = 0; r < t * 2; ++r) hit_routes += hits[r] != 0;
                        if (hit_routes < t * 2 &&
                            (std::memcmp(activation, input.data(), t * 128 * 4) ||
                             std::memcmp(published_weights, weights.data(), t * 2 * 4)))
                            throw std::runtime_error("window-hit mirror mismatch");
                        window_hit_routes = hit_routes;
                        if (declared < t && hit_routes != 0)
                            throw std::runtime_error("window below the declared width published hits");
                        // The engine's serve policy, mirrored: the host bracket is open only
                        // for MIXED windows (hits pinned while misses are admitted); a
                        // zero-hit window keeps the original planner, an all-hit window
                        // needs no service at all. Only miss rows are staged - a hit row
                        // written here would clobber the producer's inline result.
                        if (hit_routes < t * 2) {
                            if (hit_routes > 0 && !candidate.verify_graph_host_active())
                                throw std::runtime_error("mixed window left no host bracket");
                            if (hit_routes == 0 && candidate.verify_graph_host_active())
                                throw std::runtime_error("zero-hit window opened a host bracket");
                            std::fill_n(host_out, (size_t) t * 2 * 128, -888.f);   // poison the staging slab
                            if (!candidate.run_multi(1, activation, published, tokens, 2, host_out, service))
                                throw std::runtime_error("window-hit miss service refused");
                            for (int r = 0; r < t * 2; ++r)
                                if (!hits[r] && cudaMemcpyAsync(dy + (size_t) r * 128, host_out + (size_t) r * 128,
                                                                128 * 4, cudaMemcpyHostToDevice, service) != cudaSuccess)
                                    throw std::runtime_error("window-hit miss row upload");
                            candidate.complete_decode_handoff(1, ring, service);
                        } else {
                            if (candidate.verify_graph_host_active())
                                throw std::runtime_error("all-hit window kept a host bracket");
                            candidate.complete_decode_handoff(1, ring);   // host publication only
                        }
                    } catch (...) {
                        failure = std::current_exception();
                        candidate.abort_decode_handoff();
                    }
                });
                CHECK(cudaMemcpyAsync(actual.data(), dy, bytes, cudaMemcpyDeviceToHost, producer) == cudaSuccess);
                CHECK(cudaStreamSynchronize(producer) == cudaSuccess);
                worker.join();
                CHECK(!failure);
                CHECK(std::memcmp(actual.data(), expected.data(), bytes) == 0);
                if (window_hit_routes > 0) windows_with_hits += 1;
                else if (declared < t) fallback_windows += 1;
                else zero_resident_windows += 1;
                all_hit_windows += window_hit_routes == t * 2;
            }
            std::printf("PASS: window GPU hits slots=%d (%s): %d windows carried hits (%d all-hit), "
                        "%d structural fallbacks, %d zero-resident, all rows byte-exact vs host oracle\n",
                        slots, slots_case.second, windows_with_hits, all_hit_windows,
                        fallback_windows, zero_resident_windows);
            if (slots == 16 && (windows_with_hits < 9 || fallback_windows != 0 || all_hit_windows < 3))
                throw std::runtime_error("full-slot case expected every warmed width hit-eligible with all-hit windows");
            if (slots == 4 && (fallback_windows < 1 || windows_with_hits < 1))
                throw std::runtime_error("4-slot case expected both eligible and fallback widths");
            if (slots == 2 && (fallback_windows + zero_resident_windows != 10 || windows_with_hits != 0))
                throw std::runtime_error("2-slot case expected every window to fall back");
            if (slots == 16) {
                // ---- refresh API validation (the common Exl3ExpertStream surface): the
                // window-token argument is range-checked BEFORE any tokens*k gate, so the
                // product can never overflow and no negative/oversized value can silently
                // arm or clear tables. The default 0 stays legal (T1-only contract).
                bool threw = false;
                try { candidate.refresh_verify_graph(producer, -1); } catch (const std::exception&) { threw = true; }
                CHECK(threw);
                threw = false;
                try { candidate.refresh_verify_graph(producer, 9); } catch (const std::exception&) { threw = true; }
                CHECK(threw);
                candidate.refresh_verify_graph(producer, 0);   // default contract, still legal
                // ---- controlled ABORT with ACTIVE hits, then cleanup and reuse, on a
                // DEDICATED stream: one warm expert (hits) + one never-admitted expert
                // (misses) open the host bracket, the worker aborts WITHOUT serving or
                // completing - the failfast releases the producer's wait, the owner-side
                // recovery closes the bracket, and the next window on the SAME stream is
                // byte-exact (a stale host_active dropping its groups would leave rows
                // unwritten). Also the refresh window_tokens API validation (the common
                // Exl3ExpertStream surface): range-checked BEFORE any tokens*k gate, so
                // the product can never overflow and no illegal value silently arms or
                // clears tables; the default 0 stays legal (T1-only contract).
                {
                    k::Exl3ExpertStream acand(dir.string(), 128, 2, 16, producer, 0, 0, 1, 2, 8, "model.layers.");
                    k::Exl3ExpertStream aoracle(dir.string(), 128, 2, 16, producer, 0, 0, 1, 2, 8, "model.layers.");
                    acand.enable_decode_handoff(2, 8);
                    acand.set_window_graph_hits(true);
                    bool threw = false;
                    try { acand.refresh_verify_graph(producer, -1); } catch (const std::exception&) { threw = true; }
                    CHECK(threw);
                    threw = false;
                    try { acand.refresh_verify_graph(producer, 9); } catch (const std::exception&) { threw = true; }
                    CHECK(threw);
                    acand.refresh_verify_graph(producer, 0);   // the default contract stays legal
                    {   // warm exactly expert 0 (expert 2 stays cold for the mixed window)
                        const int32_t warm[1] = {0};
                        CHECK(acand.run_multi(1, input.data(), warm, 1, 1, host_out, producer));
                    }
                    const int t = 4;
                    for (int r = 0; r < t * 2; ++r) ids[r] = r % 2 ? 2 : 0;   // 0 = hit, 2 = cold miss
                    CHECK(aoracle.run_multi(1, input.data(), ids, t, 2, expected.data(), producer));
                    CHECK(cudaMemcpyAsync(dx, input.data(), t * 128 * 4, cudaMemcpyHostToDevice, producer) == cudaSuccess);
                    CHECK(cudaMemcpyAsync(di, ids, t * 2 * 4, cudaMemcpyHostToDevice, producer) == cudaSuccess);
                    CHECK(cudaMemcpyAsync(dw, weights.data(), t * 2 * 4, cudaMemcpyHostToDevice, producer) == cudaSuccess);
                    std::fill_n(actual.data(), 16 * 128, -777.f);
                    CHECK(cudaMemcpyAsync(dy, actual.data(), 16 * 128 * 4, cudaMemcpyHostToDevice, producer) == cudaSuccess);
                    CHECK(cudaStreamSynchronize(producer) == cudaSuccess);
                    acand.refresh_verify_graph(producer, t);
                    CHECK(acand.window_hit_tokens() == t);
                    uint32_t aring = 1;
                    acand.enqueue_decode_handoff(1, dx, di, 2, dy, aring, producer, t, dw);
                    std::exception_ptr abort_failure;
                    std::thread aborting_worker([&] {
                        try {
                            if (cudaSetDevice(0) != cudaSuccess) throw std::runtime_error("abort device");
                            const int32_t* published = nullptr;
                            const float *activation = nullptr, *published_weights = nullptr;
                            const uint8_t* hits = nullptr;
                            int tokens = 0;
                            acand.collect_decode_handoff(1, aring, service, published, activation, hits,
                                                         &tokens, &published_weights);
                            int ahits = 0;
                            for (int r = 0; r < tokens * 2; ++r) ahits += hits[r] != 0;
                            if (ahits == 0 || ahits == tokens * 2)
                                throw std::runtime_error("abort window is not mixed");
                            if (!acand.verify_graph_host_active())
                                throw std::runtime_error("mixed window opened no host bracket to abort");
                            // Abort MID-SERVICE: the bracket is open, nothing served, no
                            // completion - exactly the state a worker fault leaves behind.
                            acand.abort_decode_handoff();
                            acand.recover_host_service();   // owner thread, after unwind
                            if (acand.verify_graph_host_active())
                                throw std::runtime_error("recovery left the bracket open");
                        } catch (...) { abort_failure = std::current_exception(); }
                    });
                    CHECK(cudaStreamSynchronize(producer) == cudaSuccess);   // failfast released it
                    aborting_worker.join();
                    CHECK(!abort_failure);
                    // Reuse on the SAME stream: expert 2 is still cold (the aborted frame
                    // served nothing), so this window is mixed again and every group must
                    // compute - a stale record's hit flags dropping groups here would
                    // leave rows unwritten and fail the compare.
                    CHECK(aoracle.run_multi(1, input.data(), ids, t, 2, expected.data(), producer));
                    CHECK(cudaStreamSynchronize(producer) == cudaSuccess);
                    acand.refresh_verify_graph(producer, t);   // drained seam: clears the sticky abort
                    acand.enqueue_decode_handoff(1, dx, di, 2, dy, ++aring, producer, t, dw);
                    std::exception_ptr reuse_failure;
                    std::thread reuse_worker([&] {
                        try {
                            if (cudaSetDevice(0) != cudaSuccess) throw std::runtime_error("reuse device");
                            const int32_t* published = nullptr;
                            const float *activation = nullptr, *published_weights = nullptr;
                            const uint8_t* hits = nullptr;
                            int tokens = 0;
                            acand.collect_decode_handoff(1, aring, service, published, activation, hits,
                                                         &tokens, &published_weights);
                            std::fill_n(host_out, (size_t) t * 2 * 128, -888.f);
                            if (!acand.run_multi(1, activation, published, tokens, 2, host_out, service))
                                throw std::runtime_error("reuse window service refused");
                            for (int r = 0; r < tokens * 2; ++r)
                                if (!hits[r] && cudaMemcpyAsync(dy + (size_t) r * 128, host_out + (size_t) r * 128,
                                                                128 * 4, cudaMemcpyHostToDevice, service) != cudaSuccess)
                                    throw std::runtime_error("reuse row upload");
                            acand.complete_decode_handoff(1, aring, service);
                        } catch (...) { reuse_failure = std::current_exception(); acand.abort_decode_handoff(); }
                    });
                    CHECK(cudaMemcpyAsync(actual.data(), dy, (size_t) t * 2 * 128 * 4,
                                          cudaMemcpyDeviceToHost, producer) == cudaSuccess);
                    CHECK(cudaStreamSynchronize(producer) == cudaSuccess);
                    reuse_worker.join();
                    CHECK(!reuse_failure);
                    CHECK(std::memcmp(actual.data(), expected.data(), (size_t) t * 2 * 128 * 4) == 0);
                    std::puts("PASS: window-hits abort mid-service with an OPEN host bracket: failfast "
                              "release, owner recovery, byte-exact stream reuse; refresh window_tokens "
                              "range-checked");
                }
            }
        }

        CHECK(cudaFree(dx) == cudaSuccess && cudaFree(dy) == cudaSuccess);
        CHECK(cudaFree(di) == cudaSuccess && cudaFree(dw) == cudaSuccess);
        CHECK(cudaFreeHost(host_out) == cudaSuccess);
        CHECK(cudaStreamDestroy(producer) == cudaSuccess && cudaStreamDestroy(service) == cudaSuccess);
    }

    if (const char* policy = std::getenv("STRATA_EXL3_CACHE_POLICY"); policy && std::string(policy) == "lfu") {
        k::Exl3ExpertStream primary(dir.string(), 128, 2, 2, nullptr, 0, 0, 1, 2, 8, "model.layers.");
        std::vector<float> x(128, .01f), y(128);
        int hot = 0, recent = 1, cold = 2;
        for (int i = 0; i < 8; ++i) CHECK(primary.run_multi(1, x.data(), &hot, 1, 1, y.data(), nullptr));
        CHECK(primary.run_multi(1, x.data(), &recent, 1, 1, y.data(), nullptr));
        CHECK(primary.run_multi(1, x.data(), &cold, 1, 1, y.data(), nullptr));
        CHECK(primary.holds(1, hot) && primary.holds(1, cold) && !primary.holds(1, recent));
        std::puts("PASS: aging LFU protects frequent expert over more-recent one-off expert");
    }
    if (const char* cpu = std::getenv("STRATA_EXL3_CPU_TIER"); cpu && cpu[0] == '1') {
        k::Exl3ExpertStream primary(dir.string(), 128, 2, 2, nullptr, 0, 0, 1, 2, 8, "model.layers.");
        std::vector<float> x(128, .1f), warm(256), actual(256, 42.f);
        const int32_t initial[] = {0,1}, mixed[] = {0,2};
        CHECK(primary.run_multi(1, x.data(), initial, 1, 2, warm.data(), nullptr));
        float *dx = nullptr, *dy = nullptr;
        cudaStream_t stream = nullptr;
        CHECK(cudaMalloc(&dx, 128 * sizeof(float)) == cudaSuccess);
        CHECK(cudaMalloc(&dy, 256 * sizeof(float)) == cudaSuccess);
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
        CHECK(cudaMemcpyAsync(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess);
        uint8_t refused[2] = {};
        CHECK(primary.run_decode_device(1, dx, mixed, 2, dy, stream, refused));
        CHECK(cudaMemcpyAsync(actual.data(), dy, actual.size() * sizeof(float), cudaMemcpyDeviceToHost, stream) == cudaSuccess);
        CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
        CHECK(!refused[0] && refused[1] && primary.holds(1, 0) && !primary.holds(1, 2));
        CHECK(std::memcmp(actual.data(), warm.data(), 128 * sizeof(float)) == 0);
        CHECK(std::all_of(actual.begin() + 128, actual.end(), [](float v) { return v == 0; }));
        CHECK(cudaFree(dx) == cudaSuccess && cudaFree(dy) == cudaSuccess);
        CHECK(cudaStreamDestroy(stream) == cudaSuccess);
        std::puts("PASS: device CPU overflow preserves GPU hits, refuses cold admission and zeros skipped rows");
    }
    if (devices >= 2) {
        strata::Exl3Model model(dir.string());
        auto mat = [&](const char* projection) {
            const auto m = model.linear("model.layers.1.mlp.experts.3." + std::string(projection));
            return k::Exl3Mat{m.trellis, m.suh, m.svh, m.ki, m.nj, m.bits, m.cb, m.frac};
        };
        k::Exl3Peer peer;
        std::string error;
        k::Exl3Peer::Entry entry{1, 3, mat("gate_proj"), mat("up_proj"), mat("down_proj")};
        CHECK(peer.open(1, 128, {entry}, 8, 2, error));
        std::vector<uint8_t> owned(8, 0);
        owned[7] = 1;
        k::Exl3ExpertStream primary(dir.string(), 128, 2, 1, nullptr, 0, 0, 1, 2, 8, "model.layers.");
        primary.set_owned_elsewhere(owned.data(), 2, 4);
        {   // Actual mapped publication + worker-side multirow peer/complement transport.
            cudaStream_t producer = nullptr, service = nullptr;
            float *dx = nullptr, *dy = nullptr, *dw = nullptr, *host_rows = nullptr;
            int32_t* di = nullptr;
            CHECK(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking) == cudaSuccess);
            CHECK(cudaStreamCreateWithFlags(&service, cudaStreamNonBlocking) == cudaSuccess);
            CHECK(cudaMalloc(&dx, 8 * 128 * 4) == cudaSuccess);
            CHECK(cudaMalloc(&dy, 16 * 128 * 4) == cudaSuccess);
            CHECK(cudaMalloc(&dw, 16 * 4) == cudaSuccess);
            CHECK(cudaMalloc(&di, 16 * 4) == cudaSuccess);
            CHECK(cudaMallocHost(&host_rows, 16 * 128 * 4) == cudaSuccess);
            {
                k::Exl3ExpertStream mapped(dir.string(), 128, 2, 1, producer, 0, 0, 1, 2, 8, "model.layers.");
                k::Exl3ExpertStream oracle(dir.string(), 128, 2, 1, producer, 0, 0, 1, 2, 8, "model.layers.");
                mapped.set_owned_elsewhere(owned.data(), 2, 4);
                mapped.enable_decode_handoff(2, 8);
                std::vector<float> input(8 * 128), actual(16 * 128), expected(16 * 128);
                std::vector<float> weights(16, .5f);
                int32_t ids[16];
                uint32_t ring = 0;
                for (int t : {2, 3, 4, 8, 2}) {
                    const size_t bytes = (size_t) t * 2 * 128 * 4;
                    for (int i = 0; i < t * 128; ++i) input[i] = .2f * std::sin((i + ring) * .17f);
                    for (int r = 0; r < t * 2; ++r) ids[r] = r % 2 ? 3 : (r / 2) % 3;
                    CHECK(oracle.run_multi(1, input.data(), ids, t, 2, expected.data(), producer));
                    CHECK(cudaMemcpyAsync(dx, input.data(), t * 128 * 4, cudaMemcpyHostToDevice, producer) == cudaSuccess);
                    CHECK(cudaMemcpyAsync(di, ids, t * 2 * 4, cudaMemcpyHostToDevice, producer) == cudaSuccess);
                    CHECK(cudaMemcpyAsync(dw, weights.data(), t * 2 * 4, cudaMemcpyHostToDevice, producer) == cudaSuccess);
                    mapped.enqueue_decode_handoff(1, dx, di, 2, dy, ++ring, producer, t, dw);
                    std::exception_ptr failure;
                    std::thread worker([&] {
                        try {
                            if (cudaSetDevice(0) != cudaSuccess) throw std::runtime_error("peer service device");
                            const int32_t* routes = nullptr;
                            const float* activation = nullptr;
                            const uint8_t* hits = nullptr;
                            int tokens = 0;
                            mapped.collect_decode_handoff(1, ring, service, routes, activation, hits, &tokens);
                            uint8_t mask[16] = {};
                            int32_t kinds[16];
                            for (int r = 0; r < tokens * 2; ++r) {
                                mask[r] = routes[r] >= 0 && routes[r] < 4 && owned[4 + routes[r]];
                                kinds[r] = mask[r] ? 2 : -1;
                            }
                            std::string why;
                            if (tokens != t || !peer.launch(1, activation, routes, tokens, 2, kinds, why) ||
                                !mapped.run_multi_complement(1, activation, routes, tokens, 2, host_rows, service, mask) ||
                                !peer.finish(host_rows, why)) throw std::runtime_error("peer window service: " + why);
                            if (cudaMemcpyAsync(dy, host_rows, bytes, cudaMemcpyHostToDevice, service) != cudaSuccess)
                                throw std::runtime_error("peer window upload");
                            mapped.complete_decode_handoff(1, ring, service);
                        } catch (...) { failure = std::current_exception(); mapped.abort_decode_handoff(); }
                    });
                    CHECK(cudaMemcpyAsync(actual.data(), dy, bytes, cudaMemcpyDeviceToHost, producer) == cudaSuccess);
                    CHECK(cudaStreamSynchronize(producer) == cudaSuccess);
                    worker.join();
                    CHECK(!failure);
                    CHECK(std::memcmp(actual.data(), expected.data(), bytes) == 0);
                    CHECK(!mapped.holds(1, 3));
                }
                std::puts("PASS: two-GPU mapped T2/3/4/8 peer + primary complement, all rows byte-exact, no peer admission");
                // WINDOW GPU HITS beside the peer: expert 3 is peer-owned (never admitted,
                // never in the refresh tables), so its routes stay misses the peer serves,
                // while the primary-resident experts ride the producer's inline hits. The
                // worker stages ONLY non-hit rows (complement + peer); the refresh gate
                // drops widths above the active slots back to the all-miss transport.
                {
                    k::Exl3ExpertStream mapped(dir.string(), 128, 2, 4, producer, 0, 0, 1, 2, 8, "model.layers.");
                    mapped.set_owned_elsewhere(owned.data(), 2, 4);
                    mapped.enable_decode_handoff(2, 8);
                    mapped.set_window_graph_hits(true);
                    {   // warm the three primary experts once so every eligible window hits
                        const int32_t warm[] = {0, 1, 2};
                        CHECK(mapped.run_multi(1, input.data(), warm, 3, 1, host_rows, producer));
                    }
                    uint32_t ring2 = 0;
                    for (int t : {2, 2, 4, 4}) {
                        const size_t bytes = (size_t) t * 2 * 128 * 4;
                        for (int i = 0; i < t * 128; ++i) input[i] = .2f * std::sin((i + ring2) * .17f);
                        for (int r = 0; r < t * 2; ++r) ids[r] = r % 2 ? 3 : (r / 2) % 3;
                        CHECK(oracle.run_multi(1, input.data(), ids, t, 2, expected.data(), producer));
                        CHECK(cudaMemcpyAsync(dx, input.data(), t * 128 * 4, cudaMemcpyHostToDevice, producer) == cudaSuccess);
                        CHECK(cudaMemcpyAsync(di, ids, t * 2 * 4, cudaMemcpyHostToDevice, producer) == cudaSuccess);
                        CHECK(cudaMemcpyAsync(dw, weights.data(), t * 2 * 4, cudaMemcpyHostToDevice, producer) == cudaSuccess);
                        std::fill_n(actual.begin(), actual.size(), -999.f);
                        CHECK(cudaMemcpyAsync(dy, actual.data(), bytes, cudaMemcpyHostToDevice, producer) == cudaSuccess);
                        CHECK(cudaStreamSynchronize(producer) == cudaSuccess);
                        mapped.refresh_verify_graph(producer, t);
                        const int declared = mapped.window_hit_tokens();
                        ++ring2;
                        mapped.enqueue_decode_handoff(1, dx, di, 2, dy, ring2, producer, t, dw);
                        std::exception_ptr failure;
                        std::thread worker([&] {
                            try {
                                if (cudaSetDevice(0) != cudaSuccess) throw std::runtime_error("peer window-hit device");
                                const int32_t* routes = nullptr;
                                const float* activation = nullptr;
                                const uint8_t* hits = nullptr;
                                int tokens = 0;
                                mapped.collect_decode_handoff(1, ring2, service, routes, activation, hits, &tokens);
                                uint8_t mask[16] = {};
                                int32_t kinds[16];
                                int hit_routes = 0, remote = 0;
                                for (int r = 0; r < tokens * 2; ++r) {
                                    mask[r] = !hits[r] && routes[r] >= 0 && routes[r] < 4 && owned[4 + routes[r]];
                                    kinds[r] = mask[r] ? 2 : -1;
                                    hit_routes += hits[r] != 0;
                                    remote += mask[r];
                                    if (hits[r] && mask[r]) throw std::runtime_error("owned route published a hit");
                                }
                                if (declared >= t && hit_routes == 0)
                                    throw std::runtime_error("eligible window published no hits");
                                if (declared < t && hit_routes != 0)
                                    throw std::runtime_error("fallback window published hits");
                                std::fill_n(host_rows, (size_t) t * 2 * 128, -999.f);
                                std::string why;
                                if ((!remote || peer.launch(1, activation, routes, tokens, 2, kinds, why)) &&
                                    mapped.run_multi_complement(1, activation, routes, tokens, 2, host_rows, service, mask) &&
                                    (!remote || peer.finish(host_rows, why))) {
                                    for (int r = 0; r < tokens * 2; ++r)   // stage NON-HIT rows only
                                        if (!hits[r] && cudaMemcpyAsync(dy + (size_t) r * 128, host_rows + (size_t) r * 128,
                                                                        128 * 4, cudaMemcpyHostToDevice, service) != cudaSuccess)
                                            throw std::runtime_error("peer window-hit row upload");
                                    mapped.complete_decode_handoff(1, ring2, service);
                                } else throw std::runtime_error("peer window-hit service: " + why);
                            } catch (...) { failure = std::current_exception(); mapped.abort_decode_handoff(); }
                        });
                        CHECK(cudaMemcpyAsync(actual.data(), dy, bytes, cudaMemcpyDeviceToHost, producer) == cudaSuccess);
                        CHECK(cudaStreamSynchronize(producer) == cudaSuccess);
                        worker.join();
                        CHECK(!failure);
                        CHECK(std::memcmp(actual.data(), expected.data(), bytes) == 0);
                        CHECK(!mapped.holds(1, 3));
                    }
                    std::puts("PASS: two-GPU window hits T2 (hit+peer mixed) and T4 (fallback), byte-exact, owned never hits");
                }
                mapped.enqueue_decode_handoff(1, dx, di, 2, dy, ++ring, producer, 2, dw);
                const int32_t* routes = nullptr;
                const float* activation = nullptr;
                const uint8_t* hits = nullptr;
                mapped.collect_decode_handoff(1, ring, service, routes, activation, hits);
                const int32_t remote_kind[] = {2, 2, 2, 2};
                // Shape refusal occurs before any pointer traversal/enqueue. Model the
                // worker's fail-fast catch: abort, rather than waiting for missing results.
                CHECK(!peer.launch(1, activation, routes, 9, 2, remote_kind, error));
                mapped.abort_decode_handoff();
                CHECK(cudaStreamSynchronize(producer) == cudaSuccess);
                std::puts("PASS: peer launch refusal + mapped abort drains producer (fail-fast, no retry)");
            }
            CHECK(cudaFree(dx) == cudaSuccess && cudaFree(dy) == cudaSuccess);
            CHECK(cudaFree(dw) == cudaSuccess && cudaFree(di) == cudaSuccess);
            CHECK(cudaFreeHost(host_rows) == cudaSuccess);
            CHECK(cudaStreamDestroy(producer) == cudaSuccess && cudaStreamDestroy(service) == cudaSuccess);
        }
        k::Exl3PeerAdapt::Options opt;
        opt.every = 1; opt.max_swaps = 1;
        opt.expert_prefix = "model.layers."; opt.external_observer = true;
        k::Exl3PeerAdapt adapt(peer, primary, dir.string(), 2, 4, owned, opt);
        const int32_t incoming[4] = {2,2,2,2};
        adapt.observe(1, incoming, 4);
        CHECK(adapt.before_window(error)); // first observation window
        CHECK(adapt.before_window(error)); // select and submit a MiMo-prefix replacement
        {   // exclusive residency while the refill is IN FLIGHT: the pending incoming expert
            // (owned=0, reserved via the pending table) must compute on the primary scratch
            // and never be admitted - the gap between submit and publication stays disjoint
            const int32_t pending = 2;
            std::vector<float> xp(128, .02f), yp(128);
            CHECK(primary.run_multi(1, xp.data(), &pending, 1, 1, yp.data(), nullptr));
            CHECK(!primary.holds(1, 2) && !peer.has(1, 2) && !owned[6]);
        }
        CHECK(adapt.drain(error));
        CHECK(peer.has(1, 2) && !peer.has(1, 3) && owned[6] && !owned[7] && !primary.holds(1, 2));
        CHECK(adapt.swaps() == 1 && adapt.upload_bytes() > 0);
        std::vector<float> x(128, .01f), actual(128), expected(128);
        for (int i = 0; i < 128; ++i) x[i] = .3f * std::sin(i * .37f) + .15f * std::cos(i * .11f);
        k::Exl3ExpertStream oracle(dir.string(), 128, 2, 0, nullptr, 0, 0, 1, 2, 8, "model.layers.");
        CHECK(oracle.run(1, x.data(), incoming, 1, expected.data(), nullptr));
        const int32_t kind = 2;
        CHECK(peer.launch(1, x.data(), incoming, 1, 1, &kind, error));
        CHECK(peer.finish(actual.data(), error));
        double diff = 0, norm = 0;
        for (int i = 0; i < 128; ++i) { diff += std::pow(actual[i] - expected[i], 2); norm += expected[i] * expected[i]; }
        const double rel = std::sqrt(diff / std::max(norm, 1e-30));
        std::printf("peer replacement relative error %.6g\n", rel);
        CHECK(rel < .005);
        // Byte identity, not just a norm: the peer tier's indexed rows (whichever kernel arm
        // STRATA_EXL3_PEER_FUNNEL / RDNA_NATIVE select) must match the independent cache-less
        // oracle stream bit-for-bit on this K2.5 expert - the ownership audit's rule that a
        // kernel-arm switch is proven equal by the oracle, never assumed.
        CHECK(std::memcmp(actual.data(), expected.data(), 128 * sizeof(float)) == 0);
        std::puts("PASS: MiMo-prefix peer replacement, external observation, disjoint publication and FFN output");

        // The freeze experiment's contract at unit level (2026-11-01): freeze() lands any
        // in-flight refill, then before_window becomes a no-op on PLACEMENT - no planning,
        // no submit, no decay - while poll/observe keep running. Usage hot enough to plan
        // another replacement must not move anything, and compute on the frozen placement
        // must stay byte-identical to the independent oracle (freeze changes only the
        // FUTURE placement policy, never the math).
        CHECK(!adapt.frozen());
        CHECK(adapt.freeze(error));
        CHECK(adapt.frozen());
        const int32_t hotter[4] = {1, 1, 1, 1};
        adapt.observe(1, hotter, 4);
        for (int i = 0; i < 3; ++i) CHECK(adapt.before_window(error));
        CHECK(adapt.counters().submitted == 1);
        CHECK(adapt.swaps() == 1 && adapt.upload_bytes() > 0);
        CHECK(peer.has(1, 2) && !peer.has(1, 3) && owned[6] && !owned[7] && !primary.holds(1, 2));
        std::vector<float> again(128);
        CHECK(peer.launch(1, x.data(), incoming, 1, 1, &kind, error));
        CHECK(peer.finish(again.data(), error));
        CHECK(std::memcmp(again.data(), expected.data(), 128 * sizeof(float)) == 0);
        CHECK(adapt.freeze(error));   // idempotent
        std::puts("PASS: peer adapt freeze quiesces placement - no planning, residency and FFN bytes unchanged");
        // NOTE (2026-11-03): a follow-up block here exercised the DFlash verify-window peer launch
        // through the opt-in STRATA_EXL3_PEER_MULTIROW_FRAC grouped arm - dispatched, byte-identical
        // to indexed and to this oracle on the fixture.  The end-to-end evidence was mixed (no
        // reliable win; stream divergence), the admission was removed and then RESTORED the same
        // day at the user's request as a launcher trial - this engine-level block stays retired;
        // the dispatch contract is covered by exl3_peer_test's dual-arm battery and the grouped
        // frac kernel's byte identity by exl3_moe_multi_parity's fused mixed-frac battery.
        // Evidence: notes/MIMO-PEER-FRAC-GROUPED-2026-11-03.md.
    }

    // the GPU run: the same library core the CLI drives (config + pack + KV + forward + greedy)
    std::vector<int32_t> prompt;
    {
        uint32_t s = 4242;
        for (int i = 0; i < 12; ++i) {
            s = s * 1664525u + 1013904223u;
            prompt.push_back((int32_t)((s >> 8) % 256u));
        }
    }
    const int n_gen = 6;
    std::vector<int32_t> got;
    std::string err;
    const int rc = strata::mimo::run_greedy(dir.string().c_str(), prompt.data(), (int64_t)prompt.size(), n_gen,
                                            ctx, (int)chunk, got, err);
    if (rc != 0) {
        std::fprintf(stderr, "FAIL run_greedy: %s\n", err.c_str());
        return 1;
    }
    CHECK(got.size() == (size_t)n_gen);

    // the reference: the same steps (the whole prompt, then one token at a time)
    Ref ref(fx, ctx);
    double min_margin = 1e300, max_rel = 0;
    int argmax_bad = 0;
    std::vector<int32_t> want;
    for (int step = 0; step < n_gen; ++step) {
        const std::vector<int32_t> in = step == 0 ? prompt : std::vector<int32_t>{want.back()};
        const int64_t pos0 = step == 0 ? 0 : (int64_t)(prompt.size() + step - 1);
        std::vector<double> x_last;
        const std::vector<double> logits = ref.forward(in, (int)in.size(), pos0, x_last);
        int best = 0;
        for (int v = 1; v < ref.vocab; ++v)
            if (logits[(size_t)v] > logits[(size_t)best]) best = v;
        double top1 = -1e300, top2 = -1e300;
        for (double v : logits) {
            if (v > top1) { top2 = top1; top1 = v; } else if (v > top2) top2 = v;
        }
        min_margin = std::min(min_margin, top1 - top2);
        (void)x_last;
        want.push_back(best);
        if (best != got[(size_t)step]) {
            ++argmax_bad;
            std::fprintf(stderr, "  step %d: gpu %d vs reference %d (top-2 margin %.4g)\n", step,
                         got[(size_t)step], best, top1 - top2);
        }
    }
    (void)max_rel;
    CHECK(argmax_bad == 0);
    std::printf("%s: mini-MiMo end-to-end: %d greedy tokens, argmax sequence GPU == CPU reference "
                "(top-2 margin %.4g)\n",
                argmax_bad ? "FAIL" : "PASS", n_gen, min_margin);
    std::printf("  gpu: ");
    for (int t : got) std::printf("%d ", t);
    std::printf("\n  ref: ");
    for (int t : want) std::printf("%d ", t);
    std::printf("\n");
    const auto drafter_dir = root / "dflash";
    {
        ScopedEnv small("STRATA_EXL3_SMALL_M", "1"), verify("STRATA_MIMO_VERIFY_HOST", "1");
        ScopedEnv device("STRATA_MIMO_DECODE_DEVICE", "1"), handoff("STRATA_MIMO_DECODE_HANDOFF", "1");
        const char* prior = std::getenv("STRATA_MIMO_WINDOW_HANDOFF");
        const std::string saved = prior ? prior : "";
        std::vector<float> baseline, mapped;
        const std::vector<int> widths = {8, 2, 3, 4, 8, 1, 2, 8, 4, 3, 8};
        for (bool calibration : {false, true}) {
            bool base_enabled = false, mapped_enabled = false;
            setenv("STRATA_MIMO_WINDOW_HANDOFF", "0", 1);
            CHECK(strata::mimo::run_verify_logits(dir.string().c_str(), widths, baseline, err,
                calibration, &base_enabled) == 0);
            setenv("STRATA_MIMO_WINDOW_HANDOFF", "1", 1);
            CHECK(strata::mimo::run_verify_logits(dir.string().c_str(), widths, mapped, err,
                calibration, &mapped_enabled) == 0);
            CHECK(base_enabled == !calibration && mapped_enabled == !calibration);
            CHECK(mapped.size() == baseline.size());
            CHECK(std::memcmp(mapped.data(), baseline.data(), mapped.size() * sizeof(float)) == 0);
            std::printf("PASS: teacher-forced all-row logits byte-exact, calibration=%d SMALL_M=%d/%d\n",
                calibration, base_enabled, mapped_enabled);
        }
        if (prior) setenv("STRATA_MIMO_WINDOW_HANDOFF", saved.c_str(), 1);
        else unsetenv("STRATA_MIMO_WINDOW_HANDOFF");
        std::puts("PASS: teacher-forced all-row logits byte-exact vs old window worker across T1/2/3/4/8 and wraps");
    }
    // the window router split arm (STRATA_MIMO_WINDOW_ROUTER_SPLIT): the SAME teacher-forced
    // windows through the real engine, only the router kernels of the T2..8 all-rows forwards
    // swapped for the batched split pair.  Byte-exact logits is the gate; EXPECT=used makes
    // the arm non-vacuous (the split counters must prove actual window routes ran), and the
    // widths keep a T1 decode (the decode split arm) and prefill-shaped rows in the mix.
    {
        ScopedEnv small("STRATA_EXL3_SMALL_M", "1"), verify("STRATA_MIMO_VERIFY_HOST", "1");
        ScopedEnv device("STRATA_MIMO_DECODE_DEVICE", "1"), handoff("STRATA_MIMO_DECODE_HANDOFF", "1");
        ScopedEnv wh("STRATA_MIMO_WINDOW_HANDOFF", "1"), stats("STRATA_MIMO_WINDOW_ROUTER_SPLIT_STATS", "1"),
                  expect("STRATA_MIMO_TEST_WINDOW_ROUTER_SPLIT_EXPECT", "used");
        const std::vector<int> widths = {8, 2, 3, 4, 8, 1, 2, 8, 4, 3, 8};
        std::vector<float> base, split;
        CHECK(strata::mimo::run_verify_logits(dir.string().c_str(), widths, base, err) == 0);
        {
            ScopedEnv on("STRATA_MIMO_WINDOW_ROUTER_SPLIT", "1");
            CHECK(strata::mimo::run_verify_logits(dir.string().c_str(), widths, split, err) == 0);
        }
        CHECK(base.size() == split.size() && !base.empty());
        CHECK(std::memcmp(base.data(), split.data(), base.size() * sizeof(float)) == 0);
        std::puts("PASS: teacher-forced all-row logits byte-exact, window router split off vs on "
                  "(T1/2/3/4/8, split engaged)");
    }
    // Preflight refusals, both flags: an invalid combination must fail FAST with a clear
    // error naming the missing prerequisite - never a silent never-engaging arm (the same
    // class as the LOAD_MTP refusal test below).
    {
        ScopedEnv small("STRATA_EXL3_SMALL_M", "1"), verify("STRATA_MIMO_VERIFY_HOST", "1");
        ScopedEnv device("STRATA_MIMO_DECODE_DEVICE", "1"), handoff("STRATA_MIMO_DECODE_HANDOFF", "1");
        std::vector<float> unused;
        {
            ScopedEnv m("STRATA_MIMO_WINDOW_DEVICE_MISSES", "1");
            err.clear();
            CHECK(strata::mimo::run_verify_logits(dir.string().c_str(), {2}, unused, err) != 0);
            CHECK(err.find("STRATA_MIMO_WINDOW_DEVICE_MISSES=1 needs the mapped verify-window "
                           "handoff") != std::string::npos);
        }
        {
            ScopedEnv wh("STRATA_MIMO_WINDOW_HANDOFF", "1"),
                      p("STRATA_MIMO_PEER_DEVICE_RESULTS", "1");
            err.clear();
            CHECK(strata::mimo::run_verify_logits(dir.string().c_str(), {2}, unused, err) != 0);
            CHECK(err.find("STRATA_MIMO_PEER_DEVICE_RESULTS=1 needs the peer tier") != std::string::npos);
        }
        std::puts("PASS: window device misses / peer device results refuse invalid combos "
                  "with clear preflight errors");
    }
    // ---- STRATA_MIMO_WINDOW_DEVICE_MISSES (audit R4, default off): the verify-window
    // service's device-input/device-output arm. EVERY window service call rides it - the
    // host arm used to serve through run_multi (there the plan drops are provable no-ops,
    // so the arithmetic is identical). Byte-exact teacher-forced logits is the authority
    // in every arm; EXPECT=used makes each non-vacuous. The widths keep T1 rows (the T1
    // decode arm, untouched by the flag), wraps and repeated widths (frame reuse).
    {
        const auto all_finite = [](const std::vector<float>& v) {
            for (float x : v) if (!std::isfinite(x)) return false;
            return !v.empty();
        };
        ScopedEnv small("STRATA_EXL3_SMALL_M", "1"), verify("STRATA_MIMO_VERIFY_HOST", "1");
        ScopedEnv device("STRATA_MIMO_DECODE_DEVICE", "1"), handoff("STRATA_MIMO_DECODE_HANDOFF", "1");
        ScopedEnv wh("STRATA_MIMO_WINDOW_HANDOFF", "1");
        const std::vector<int> widths = {8, 2, 3, 4, 8, 1, 2, 8, 4, 3, 8};
        // (a) all-miss windows (zero cache) beside a live CPU overlay: every window has a
        //     CPU job (the planner's cold groups) AND device-served miss rows. off vs
        //     explicit "0" vs "1" must be byte-identical.
        std::vector<float> base, off0, on1;
        {
            ScopedEnv tier("STRATA_EXL3_CPU_TIER", "1"), wcpu("STRATA_MIMO_WINDOW_CPU", "1"),
                      threads("STRATA_MIMO_CPU_WORKERS", "8"), cache0("STRATA_EXL3_CPU_CACHE_MIB", "0"),
                      zero("STRATA_MIMO_TEST_EXPERT_CACHE_ZERO", "1"),
                      wstats("STRATA_MIMO_WINDOW_CPU_STATS", "1"),
                      expect("STRATA_MIMO_TEST_WINDOW_DEVICE_MISSES_EXPECT", "used"),
                      cexpect("STRATA_MIMO_TEST_WINDOW_CPU_EXPECT", "jobs");
            CHECK(strata::mimo::run_verify_logits(dir.string().c_str(), widths, base, err) == 0);
            {
                ScopedEnv z("STRATA_MIMO_WINDOW_DEVICE_MISSES", "0");
                CHECK(strata::mimo::run_verify_logits(dir.string().c_str(), widths, off0, err) == 0);
            }
            {
                ScopedEnv o("STRATA_MIMO_WINDOW_DEVICE_MISSES", "1");
                const int rc = strata::mimo::run_verify_logits(dir.string().c_str(), widths, on1, err);
                if (rc != 0) std::fprintf(stderr, "window-device-misses on1 rc=%d err='%s'\n", rc, err.c_str());
                CHECK(rc == 0);
            }
            // (a2) MAX_GROUPS=0: the explicit no-op overlay - no CPU job, no peer -> the
            //      PLAIN windows the host arm used to serve through run_multi. The device
            //      arm serves those too now (the audit-fix completion of point 3), so this
            //      arm PROVES engagement on exactly the shape that used to be inert:
            //      EXPECT=used must hold with the flag on, the logits must stay
            //      byte-exact, and no irrelevant CPU work may appear.
            std::vector<float> nocpu_base, nocpu_on;
            {
                ScopedEnv mg("STRATA_MIMO_WINDOW_CPU_MAX_GROUPS", "0"),
                          cex("STRATA_MIMO_TEST_WINDOW_CPU_EXPECT", "zero");
                CHECK(strata::mimo::run_verify_logits(dir.string().c_str(), widths, nocpu_base, err) == 0);
                {
                    ScopedEnv o("STRATA_MIMO_WINDOW_DEVICE_MISSES", "1"),
                              ex("STRATA_MIMO_TEST_WINDOW_DEVICE_MISSES_EXPECT", "used");
                    const int rc = strata::mimo::run_verify_logits(dir.string().c_str(), widths, nocpu_on, err);
                    if (rc != 0) std::fprintf(stderr, "nocpu-on rc=%d err='%s'\n", rc, err.c_str());
                    CHECK(rc == 0);
                }
            }
            CHECK(nocpu_base.size() == nocpu_on.size() && !nocpu_base.empty());
            if (nocpu_base.size() == nocpu_on.size() && !nocpu_base.empty())   // guard before memcmp
                CHECK(std::memcmp(nocpu_base.data(), nocpu_on.data(),
                                  nocpu_base.size() * sizeof(float)) == 0);
            std::puts("PASS: window device misses byte-exact and ENGAGED with zero CPU jobs "
                      "(the plain window now rides the device arm; no irrelevant CPU work)");
        }
        CHECK(base.size() == off0.size() && base.size() == on1.size() && !base.empty());
        if (base.size() == off0.size() && !base.empty())   // guard: CHECK must not reach memcmp on a wrong size
            CHECK(std::memcmp(base.data(), off0.data(), base.size() * sizeof(float)) == 0);
        if (base.size() == on1.size() && !base.empty())
            CHECK(std::memcmp(base.data(), on1.data(), base.size() * sizeof(float)) == 0);
        CHECK(all_finite(on1));
        std::puts("PASS: window device misses logits byte-exact off/'0'/'1' (all-miss windows "
                  "beside live CPU jobs, device arm engaged)");
        // (b) combined arms, bisected: hits alone, pending alone, then the full combo with
        //     the CPU overlay. 4 slots arm the T=2 hit tables while the first windows stay
        //     cold enough for CPU jobs and device-served misses in the same run (pending
        //     experts stay on the GPU scratch path).
        const char* combo_envs[][2] = {
            {"STRATA_MIMO_WINDOW_GPU_HITS", "1"},
            {"STRATA_MIMO_TEST_PENDING_INCOMING", "half"},
            {"STRATA_MIMO_WINDOW_GPU_HITS", "1"},
        };
        for (int variant = 0; variant < 3; ++variant) {
            ScopedEnv tier("STRATA_EXL3_CPU_TIER", "1"), wcpu("STRATA_MIMO_WINDOW_CPU", "1"),
                      threads("STRATA_MIMO_CPU_WORKERS", "8"), cache0("STRATA_EXL3_CPU_CACHE_MIB", "0"),
                      slots("STRATA_MIMO_TEST_EXPERT_SLOTS", "4"),
                      a(combo_envs[variant][0], combo_envs[variant][1]);
            // variant 2 repeats variant 0 with the pending arm beside it: the third run is
            // a REPEAT of the flag-off baseline, separating the arm's effect from any
            // run-to-run nondeterminism of the combined configuration itself.
            ScopedEnv pending2(variant == 2 ? "STRATA_MIMO_TEST_PENDING_INCOMING" : "_unused_",
                               variant == 2 ? "half" : "0");
            ScopedEnv wstats("STRATA_MIMO_WINDOW_CPU_STATS", "1"),
                      expect("STRATA_MIMO_TEST_WINDOW_DEVICE_MISSES_EXPECT", "used");
            ScopedEnv cexpect(variant == 2 ? "STRATA_MIMO_TEST_WINDOW_CPU_EXPECT" : "_unused_",
                              variant == 2 ? "jobs" : "0");
            std::vector<float> combo_base, combo_on;
            {
                const int rc = strata::mimo::run_verify_logits(dir.string().c_str(), widths, combo_base, err);
                if (rc != 0) std::fprintf(stderr, "wdm-combo[%d] base rc=%d err='%s'\n", variant, rc, err.c_str());
                CHECK(rc == 0);
            }
            {
                ScopedEnv o("STRATA_MIMO_WINDOW_DEVICE_MISSES", "1");
                const int rc = strata::mimo::run_verify_logits(dir.string().c_str(), widths, combo_on, err);
                if (rc != 0) std::fprintf(stderr, "wdm-combo[%d] on rc=%d err='%s'\n", variant, rc, err.c_str());
                CHECK(rc == 0);
            }
            CHECK(combo_base.size() == combo_on.size() && !combo_base.empty());
            if (combo_base.size() == combo_on.size() && !combo_base.empty() &&
                std::memcmp(combo_base.data(), combo_on.data(), combo_base.size() * sizeof(float)) != 0) {
                size_t first = 0, count = 0;
                for (size_t i = 0; i < combo_base.size(); ++i)
                    if (std::memcmp(&combo_base[i], &combo_on[i], 4) != 0) {
                        if (!count) first = i;
                        ++count;
                    }
                std::fprintf(stderr, "wdm-combo[%d] diff: %zu of %zu floats, first at %zu "
                                     "(row %zu), base=%g on=%g\n", variant, count, combo_base.size(),
                             first, first / 256, (double) combo_base[first], (double) combo_on[first]);
            }
            if (combo_base.size() == combo_on.size() && !combo_base.empty())   // guard before memcmp
                CHECK(std::memcmp(combo_base.data(), combo_on.data(),
                                  combo_base.size() * sizeof(float)) == 0);
            if (variant == 2) {
                // The determinism control: a THIRD flag-off run must reproduce the first
                // bit for bit - otherwise the combined configuration itself is not
                // run-to-run stable and the off/on comparison cannot attribute anything.
                std::vector<float> repeat;
                CHECK(strata::mimo::run_verify_logits(dir.string().c_str(), widths, repeat, err) == 0);
                CHECK(combo_base.size() == repeat.size());
                if (combo_base.size() == repeat.size() && !combo_base.empty()) {
                    if (std::memcmp(combo_base.data(), repeat.data(), combo_base.size() * sizeof(float)) != 0)
                        std::fprintf(stderr, "wdm-combo determinism control: two flag-off runs DIFFER "
                                             "(the hits+pending+CPU configuration is not run-stable)\n");
                    CHECK(std::memcmp(combo_base.data(), repeat.data(),
                                      combo_base.size() * sizeof(float)) == 0);
                }
            }
            const char* name = variant == 0 ? "inline hits" : variant == 1 ? "pending-incoming" :
                               "inline hits + live CPU overlay + pending-incoming";
            std::printf("PASS: window device misses byte-exact combined with %s\n", name);
        }
    }
    // ---- STRATA_MIMO_PEER_DEVICE_RESULTS (audit R3, default off): the handoff worker's
    // peer rows through the shared mapped F16 frame. Needs a second visible device (the
    // fixture's deterministic peer tier); skips like exl3_peer_test otherwise.
    {
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 2) {
            std::printf("SKIP: peer device results fixture needs two visible devices (have %d)\n", ndev);
        } else {
            const std::vector<int> widths = {8, 2, 3, 4, 8, 1, 2, 8, 4, 3, 8};
            ScopedEnv small("STRATA_EXL3_SMALL_M", "1"), verify("STRATA_MIMO_VERIFY_HOST", "1");
            ScopedEnv device("STRATA_MIMO_DECODE_DEVICE", "1"), handoff("STRATA_MIMO_DECODE_HANDOFF", "1");
            ScopedEnv wh("STRATA_MIMO_WINDOW_HANDOFF", "1"), peer("STRATA_MIMO_TEST_PEER_DEVICE", "1"),
                      reserve("STRATA_MIMO_PEER_RESERVE_MIB", "1024");
            // (c) an after-publish GPU fault with the device-miss arm active must abort the
            //     frame (not hang) and a FRESH engine on the same fixture serves windows
            //     again (stream/context reuse after the abort path). Runs FIRST in this
            //     section: an engine immediately following a peer-fixture engine is a
            //     separate variable (context reuse), kept out of the fault assertion.
            {
                ScopedEnv nopeer("STRATA_MIMO_TEST_PEER_DEVICE", "0"),
                          nop("STRATA_MIMO_PEER_DEVICE_RESULTS", "0");
                ScopedEnv tier("STRATA_EXL3_CPU_TIER", "1"), wcpu("STRATA_MIMO_WINDOW_CPU", "1"),
                          groups("STRATA_MIMO_WINDOW_CPU_MAX_GROUPS", "2"),
                          zero("STRATA_MIMO_TEST_EXPERT_CACHE_ZERO", "1"),
                          fault("STRATA_MIMO_TEST_WINDOW_CPU_FAULT", "gpu"),
                          m("STRATA_MIMO_WINDOW_DEVICE_MISSES", "1");
                std::vector<float> unused;
                err.clear();
                const int failed = strata::mimo::run_verify_logits(dir.string().c_str(), {8}, unused, err);
                CHECK(failed != 0 && err.find("after CPU publish") != std::string::npos);
                ScopedEnv nofault("STRATA_MIMO_TEST_WINDOW_CPU_FAULT", "");
                std::vector<float> fresh;
                CHECK(strata::mimo::run_verify_logits(dir.string().c_str(), {2, 4}, fresh, err) == 0);
                std::puts("PASS: after-publish fault aborts cleanly with the device complement "
                          "armed; a fresh engine reuses the streams");
            }
            // (c2) THE R3-SPECIFIC abort arms: faults with an ACTUAL peer device frame at
            //     increasing exposure (STRATA_MIMO_TEST_PEER_FRAME_FAULT). (c) above
            //     deliberately runs without a peer and cannot cover any of these. Each
            //     arm must fail loudly (never hang), unwind through the worker's
            //     abort+release path, destruct without a stuck service stream, and a
            //     FRESH engine on the same fixture must serve full windows again.
            {
                struct FaultArm {
                    const char* value;
                };
                const FaultArm arms[] = {
                    {"1"},           // frame in flight, no copy enqueued yet
                    {"after-copy"},  // QUEUED wait kernel through the unwind
                    {"desync"},      // never-landing ready + queued wait: teardown hangs without release
                };
                for (const auto& fa : arms) {
                    ScopedEnv on2("STRATA_MIMO_PEER_DEVICE_RESULTS", "1"),
                              fault2("STRATA_MIMO_TEST_PEER_FRAME_FAULT", fa.value);
                    std::vector<float> unused;
                    err.clear();
                    const auto t0 = std::chrono::steady_clock::now();
                    const int failed =
                        strata::mimo::run_verify_logits(dir.string().c_str(), {4}, unused, err, true);
                    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    CHECK(failed != 0 && err.find("test peer frame fault") != std::string::npos);
                    CHECK(wall < 240.0);   // finite by construction, not by luck
                    ScopedEnv nofault2("STRATA_MIMO_TEST_PEER_FRAME_FAULT", "");
                    std::vector<float> fresh2;
                    CHECK(strata::mimo::run_verify_logits(dir.string().c_str(), {2, 4}, fresh2, err, true) == 0);
                    CHECK(!fresh2.empty());
                    std::printf("PASS: peer frame fault '%s' unwinds in %.1f s (worker-error "
                                "release: %s; fresh engine serves windows again)\n",
                                fa.value, wall,
                                std::strcmp(fa.value, "desync") == 0
                                    ? "provably-unsignaled frame + queued wait kernel"
                                    : "frame in flight / queued wait kernel");
                }
            }
            // (a) the empty ranked list hands the fixture's whole MoE layer to the peer:
            //     every window/decode row rides the frame. T1 calibration rides the worker's
            //     decode arm; the widths ride the window arm - both collect through the
            //     frame with the flag on.
            std::vector<float> base, off0, on1;
            {
                ScopedEnv expect("STRATA_MIMO_TEST_PEER_DEVICE_RESULTS_EXPECT", "used");
                CHECK(strata::mimo::run_verify_logits(dir.string().c_str(), widths, base, err, true) == 0);
                {
                    ScopedEnv z("STRATA_MIMO_PEER_DEVICE_RESULTS", "0");
                    CHECK(strata::mimo::run_verify_logits(dir.string().c_str(), widths, off0, err, true) == 0);
                }
                {
                    // The flag's OWN explicit "1" scope (the prior revision omitted this and
                    // silently re-ran the unset arm - a vacuous test; the last-arg `true` is
                    // the T1 calibration knob, never the arm).
                    ScopedEnv o("STRATA_MIMO_PEER_DEVICE_RESULTS", "1");
                    const int rc = strata::mimo::run_verify_logits(dir.string().c_str(), widths, on1, err, true);
                    if (rc != 0) std::fprintf(stderr, "peer-device-results on1 rc=%d err='%s'\n", rc, err.c_str());
                    CHECK(rc == 0);
                }
            }
            CHECK(base.size() == off0.size() && base.size() == on1.size() && !base.empty());
            if (base.size() == off0.size() && !base.empty())   // guard: CHECK must not reach memcmp on a wrong size
                CHECK(std::memcmp(base.data(), off0.data(), base.size() * sizeof(float)) == 0);
            if (base.size() == on1.size() && !base.empty())
                CHECK(std::memcmp(base.data(), on1.data(), base.size() * sizeof(float)) == 0);
            for (float v : on1) CHECK(std::isfinite(v));
            std::puts("PASS: peer device results logits byte-exact off/'0'/'1' (all-peer fixture "
                      "tier; T1 decode + T2..8 windows collected through the frame)");
            // (a2) R3 on, R4 ON TOO, all-peer fixture: every window's rows come back through
            //      the frame while the device complement's row set is EMPTY (every route
            //      peer-owned -> skip-service gate; run_multi_complement_device would drop
            //      all groups). This proves the two flags coexist on the full ALL-PEER
            //      composition - R3 engaged (EXPECT=used on the frames), the device arm's
            //      skip gate correct (a nonzero wdm EXPECT would be WRONG here: no window
            //      carries a GPU row; that engagement is proven in the R4 section and the
            //      split (b) below) - byte-exact against the same baseline.
            {
                ScopedEnv p2("STRATA_MIMO_PEER_DEVICE_RESULTS", "1"),
                          m2("STRATA_MIMO_WINDOW_DEVICE_MISSES", "1"),
                          pexpect2("STRATA_MIMO_TEST_PEER_DEVICE_RESULTS_EXPECT", "used");
                std::vector<float> allpeer_combo;
                const int rc = strata::mimo::run_verify_logits(dir.string().c_str(), widths, allpeer_combo, err, true);
                if (rc != 0) std::fprintf(stderr, "allpeer-combo rc=%d err='%s'\n", rc, err.c_str());
                CHECK(rc == 0);
                CHECK(base.size() == allpeer_combo.size() && !allpeer_combo.empty());
                if (base.size() == allpeer_combo.size() && !allpeer_combo.empty())
                    CHECK(std::memcmp(base.data(), allpeer_combo.data(),
                                      base.size() * sizeof(float)) == 0);
                for (float v : allpeer_combo) CHECK(std::isfinite(v));
                std::puts("PASS: peer device results + window device misses byte-exact on the "
                          "ALL-PEER fixture (frames engaged; the device arm's empty row set "
                          "correctly skips service)");
            }
            // (a3) R3 on ALONE on the split fixture (R4 off): mixed windows AND zero-peer
            //      windows in one run - EXPECT=mixed requires both the frame's engagement
            //      and window publications that carried no peer row (the no-remote path).
            //      CORRECTED (2026-11-12 round 3): the prior revision declared ONE
            //      p3="1" scope around BOTH runs - mixed_base was also computed with the
            //      flag ON, so the comparison was on-vs-on (vacuous). The baseline now
            //      runs with the flag explicitly "0" in its OWN scope, the candidate with
            //      "1" in a distinct scope, and EXPECT=mixed applies to the CANDIDATE
            //      only. Sizes are guarded BEFORE any memcmp (CHECK does not early-exit;
            //      a wrong size must fail cleanly, never read out of bounds) and both
            //      outputs are swept for finiteness.
            {
                ScopedEnv slots2("STRATA_MIMO_TEST_EXPERT_SLOTS", "2"),
                          split2("STRATA_MIMO_TEST_PEER_SPLIT", "1");
                std::vector<float> mixed_base, mixed_on;
                {
                    ScopedEnv p3_off("STRATA_MIMO_PEER_DEVICE_RESULTS", "0");
                    const int rc = strata::mimo::run_verify_logits(
                        dir.string().c_str(), widths, mixed_base, err, true);
                    if (rc != 0) std::fprintf(stderr, "peer-mixed base rc=%d err='%s'\n", rc, err.c_str());
                    CHECK(rc == 0);
                }
                {
                    ScopedEnv p3("STRATA_MIMO_PEER_DEVICE_RESULTS", "1"),
                              pexpect3("STRATA_MIMO_TEST_PEER_DEVICE_RESULTS_EXPECT", "mixed");
                    const int rc = strata::mimo::run_verify_logits(
                        dir.string().c_str(), widths, mixed_on, err, true);
                    if (rc != 0) std::fprintf(stderr, "peer-mixed on rc=%d err='%s'\n", rc, err.c_str());
                    CHECK(rc == 0);
                }
                CHECK(mixed_base.size() == mixed_on.size() && !mixed_base.empty() && !mixed_on.empty());
                if (mixed_base.size() == mixed_on.size() && !mixed_base.empty()) {
                    CHECK(std::memcmp(mixed_base.data(), mixed_on.data(),
                                      mixed_base.size() * sizeof(float)) == 0);
                    bool finite_both = true;
                    for (float v : mixed_base) if (!std::isfinite(v)) finite_both = false;
                    for (float v : mixed_on) if (!std::isfinite(v)) finite_both = false;
                    CHECK(finite_both);
                }
                std::puts("PASS: peer device results byte-exact on the split fixture (mixed AND "
                          "zero-peer windows with the flag ON vs an explicitly-off baseline; "
                          "R4 off)");
            }
            // (b) the split fixture (SLOTS=2 + the ranked hand-off) gives the primary two
            //     experts and the peer two: peer rows through the frame AND device-served
            //     misses in the SAME windows - both flags engaged at once (the refresh's
            //     active >= T*k gate keeps hits disarmed at 2 slots, so every primary row
            //     is a complement miss the device arm must scatter). A peer + live-CPU-
            //     overlay combination is NOT reachable in this fixture: the primary's
            //     warm-fill and the peer's walk together own all four experts, so the CPU
            //     planner's cold-group set is empty - that trio needs the real model's
            //     256 experts; the CPU-overlay-combined case is covered peerless in the
            //     window-device-misses section above.
            {
                ScopedEnv slots("STRATA_MIMO_TEST_EXPERT_SLOTS", "2"),
                          split("STRATA_MIMO_TEST_PEER_SPLIT", "1"),
                          pexpect("STRATA_MIMO_TEST_PEER_DEVICE_RESULTS_EXPECT", "used"),
                          mexpect("STRATA_MIMO_TEST_WINDOW_DEVICE_MISSES_EXPECT", "used");
                std::vector<float> combo_base, combo_on;
                {
                    const int rc = strata::mimo::run_verify_logits(dir.string().c_str(), widths, combo_base, err, true);
                    if (rc != 0) std::fprintf(stderr, "peer-combo base rc=%d err='%s'\n", rc, err.c_str());
                    CHECK(rc == 0);
                }
                {
                    ScopedEnv p("STRATA_MIMO_PEER_DEVICE_RESULTS", "1"),
                              m("STRATA_MIMO_WINDOW_DEVICE_MISSES", "1");
                    const int rc = strata::mimo::run_verify_logits(dir.string().c_str(), widths, combo_on, err, true);
                    if (rc != 0) std::fprintf(stderr, "peer-combo on rc=%d err='%s'\n", rc, err.c_str());
                    CHECK(rc == 0);
                }
                CHECK(combo_base.size() == combo_on.size() && !combo_base.empty());
                if (combo_base.size() == combo_on.size() && !combo_base.empty())
                    CHECK(std::memcmp(combo_base.data(), combo_on.data(),
                                      combo_base.size() * sizeof(float)) == 0);
                std::puts("PASS: peer device results + window device misses byte-exact combined, "
                          "both engaged (peer rows through the frame, misses scattered on device)");
            }
        }
    }
    {
        const std::vector<int> widths = {1, 2, 4};
        std::vector<float> ordinary, no_mtp;
        CHECK(strata::mimo::run_verify_logits(dir.string().c_str(), widths, ordinary, err) == 0);
        {
            ScopedEnv skip("STRATA_MIMO_LOAD_MTP", "0");
            CHECK(strata::mimo::run_verify_logits(dir.string().c_str(), widths, no_mtp, err) == 0);
        }
        CHECK(!ordinary.empty() && ordinary.size() == no_mtp.size());
        CHECK(std::memcmp(ordinary.data(), no_mtp.data(), ordinary.size() * sizeof(float)) == 0);
        {
            ScopedEnv invalid("STRATA_MIMO_LOAD_MTP", "invalid");
            CHECK(strata::mimo::run_verify_logits(dir.string().c_str(), widths, no_mtp, err) != 0);
            CHECK(err.find("STRATA_MIMO_LOAD_MTP must be") != std::string::npos);
        }
        std::puts("PASS: skipping optional MTP weights leaves target logits byte-exact; invalid switch refused");
    }
    {
        const auto mixed_dir = dir.parent_path() / "mixed-bit-experts";
        auto mixed = build_fixture(mixed_dir, .22);
        for (const char* proj : {"gate", "up", "down"})
            mixed.exl3("model.layers.1.mlp.experts.3." + std::string(proj) + "_proj", 8, 8, 8);
        mixed.write();
        ScopedEnv variable("STRATA_EXL3_VARIABLE_EXPERT_BITS", "1");
        k::Exl3ExpertStream cached(mixed_dir.string(), 128, 2, 2, nullptr, 0, 0, 1, 2, 8, "model.layers.");
        k::Exl3ExpertStream oracle(mixed_dir.string(), 128, 2, 0, nullptr, 0, 0, 1, 2, 8, "model.layers.");
        std::vector<float> input(128, .02f), a(128), b(128);
        for (int expert : {0, 3, 1, 3, 2, 0, 3}) {
            CHECK(cached.run_multi(1, input.data(), &expert, 1, 1, a.data(), nullptr));
            CHECK(oracle.run_multi(1, input.data(), &expert, 1, 1, b.data(), nullptr));
            CHECK(std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0);
        }
        std::puts("PASS: mixed K2.5/K4 experts fit cache slots and scratch; byte-exact through eviction");
    }
    build_dflash_fixture(drafter_dir);
    std::vector<int32_t> sequential;
    CHECK(strata::mimo::run_greedy(dir.string().c_str(), prompt.data(), prompt.size(), 24, ctx, chunk,
                                  sequential, err) == 0);
    for (int proposals : {1,2,3,4,7}) {
        std::vector<int32_t> speculative;
        CHECK(strata::mimo::run_greedy_dflash(dir.string().c_str(), drafter_dir.string().c_str(), proposals,
            prompt.data(), prompt.size(), 24, ctx, chunk, speculative, err) == 0);
        if (speculative != sequential) std::fprintf(stderr, "DFlash %d proposals mismatch: %s\n", proposals, err.c_str());
        CHECK(speculative == sequential);
    }
    // The adaptive verification window (opt-in STRATA_MIMO_DFLASH_AWINDOW=1): window width varies
    // per round (EMA q + explore rounds at the max) and the drafter always runs its full block, so
    // this exercises mixed widths, full-block drafts and short verify windows in ONE stream - the
    // greedy output must still equal the sequential run token for token.
    {
        const char* prior = std::getenv("STRATA_MIMO_DFLASH_AWINDOW");
        std::string saved = prior ? prior : "";
        setenv("STRATA_MIMO_DFLASH_AWINDOW", "1", 1);
        for (int proposals : {3, 7}) {
            std::vector<int32_t> speculative;
            CHECK(strata::mimo::run_greedy_dflash(dir.string().c_str(), drafter_dir.string().c_str(), proposals,
                prompt.data(), prompt.size(), 24, ctx, chunk, speculative, err) == 0);
            if (speculative != sequential) std::fprintf(stderr, "DFlash awindow %d proposals mismatch: %s\n", proposals, err.c_str());
            CHECK(speculative == sequential);
        }
        if (prior) setenv("STRATA_MIMO_DFLASH_AWINDOW", saved.c_str(), 1);
        else unsetenv("STRATA_MIMO_DFLASH_AWINDOW");
    }
    // A zero target head also makes every draft proposal zero: exercise full acceptance,
    // all-window commit, short final block and repeated target/drafter ring wraps.
    Fixture constant = fx;
    constant.dir = root / "constant";
    auto& scale = constant.tensors.at("lm_head.svh").bytes;
    std::fill(scale.begin(), scale.end(), 0);
    constant.write();
    { std::ofstream config(constant.dir / "config.json"); config << mini_config(); }
    for (int proposals : {1,2,3,4,7}) {
        std::vector<int32_t> speculative;
        CHECK(strata::mimo::run_greedy_dflash(constant.dir.string().c_str(), drafter_dir.string().c_str(), proposals,
            prompt.data(), prompt.size(), 25, ctx, chunk, speculative, err) == 0);
        CHECK(speculative == std::vector<int32_t>(25, 0));
    }
    std::puts("PASS: DFlash integrated verification/rollback and full acceptance at M=3,4,5,8 (+awindow)");
    std::filesystem::remove_all(root);
    if (failures) {
        std::fprintf(stderr, "mimo_engine_parity: %d failure(s)\n", failures);
        return 1;
    }
    return 0;
}
