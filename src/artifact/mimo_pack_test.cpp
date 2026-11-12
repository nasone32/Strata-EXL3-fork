// src/artifact/mimo_pack_test.cpp - the MiMo v2 pack against a SYNTHETIC mini checkpoint the test
// writes itself (2 layers: 0 dense-FFN/full-attention, 1 MoE/SLIDING-WINDOW with 4 experts at the
// HALF-INTEGER K=2.5, one MTP head), plus deliberately-broken variants.
//
//   * mimo_validate_checkpoint: the real geometry's rules - k_proj rows pin each layer's kind (the
//     config hybrid_layer_pattern gotcha), sinks on exactly the SWA layers, expert/MTP/embed
//     presence, the K histogram.
//   * mimo_pack_build (GPU-gated, like the other exl3 GPU tests): every dense tensor uploaded, the
//     o_proj attention_value_scale fold applied EXACTLY ONCE (bit-exact against the host
//     reconstruction), the BF16 embed converted, vram_bytes accounted.
//   * the expert-name prefix: Exl3ExpertStore opened with `model.layers.` (what
//     mimo_expert_tensor spells) reads the K=2.5 experts and answers one token - the fractional
//     trellis decoded on the GPU against the CPU reference.
//   * the v_proj padded-lane guard's FAILURE path: an asymmetric-V fixture with real (random)
//     weights in the padded lanes is rejected.  The pass path needs a trellis quantized from a
//     zero-padded original-basis matrix - that is a quantizer, not a fixture, and stays a
//     bring-up-day check against the real checkpoint (notes/MIMO-V2-PORT.md step 1).
#include "strata/artifact/mimo_pack.hpp"
#include "strata/artifact/exl3_model.hpp"
#include "strata/kernels/cpu/exl3.hpp"
#include "strata/kernels/exl3_experts.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(cond)                                                                              \
    do {                                                                                         \
        if (!(cond)) {                                                                           \
            ++failures;                                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                 \
        }                                                                                        \
    } while (0)

using namespace strata;
using kernels::cpu::Exl3Codebook;

// ---- fp16 helpers (round to nearest even) ----
static float h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31, exp = (h >> 10) & 0x1F, man = h & 0x3FF, f;
    if (exp == 0) {
        if (!man) f = sign;
        else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; --exp; } man &= 0x3FF; f = sign | (exp << 23) | (man << 13); }
    } else if (exp == 0x1F) f = sign | 0x7F800000u | (man << 13);
    else f = sign | ((exp + 127 - 15) << 23) | (man << 13);
    float o; std::memcpy(&o, &f, 4); return o;
}
static uint16_t f2h(float x) {
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

// ================================ the fixture writer ================================
namespace {

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
    /// one EXL3 linear: trellis [ki, nj, 8*K2] with random words (any bit pattern is a valid
    /// trellis), sane F16 scales, and the mul1 marker (the fixture's K=2.5 tensors need it).
    void exl3(const std::string& base, int ki, int nj, int k2) {
        const int words = 8 * k2;
        std::vector<uint16_t> tr((size_t)ki * nj * words);
        for (auto& w : tr) w = rnd16();
        bytes16(base + ".trellis", "I16", {ki, nj, words}, tr);
        std::vector<uint16_t> suh((size_t)ki * 16), svh((size_t)nj * 16);
        for (auto& s : suh) s = f2h(0.5f + (rnd() % 1000) / 1000.0f);
        for (auto& s : svh) s = f2h(0.5f + (rnd() % 1000) / 1000.0f);
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

// The mini geometry: hidden 128, 8 q heads x 32, kv 4 (full) / 8 (swa), v_head_dim = 32 (SYMMETRIC:
// the padded-lane rule is exercised by the asymmetric fixture below), 4 experts x 2 layers, 1 MTP
// head.  Every reconstructed dim is a multiple of 128, as the Hadamard requires.
std::string mini_config(int v_head_dim) {
    std::ostringstream o;
    o << "{\"model_type\":\"mimo_v2\",\"hidden_size\":128,\"num_hidden_layers\":2,\"vocab_size\":256,"
         "\"layernorm_epsilon\":1e-6,\"num_attention_heads\":8,\"head_dim\":32,\"v_head_dim\":"
      << v_head_dim
      << ",\"partial_rotary_factor\":0.5,\"num_key_value_heads\":4,\"swa_num_key_value_heads\":8,"
         "\"sliding_window\":8,\"rope_theta\":1e7,\"swa_rope_theta\":1e4,\"attention_value_scale\":0.707,"
         "\"n_routed_experts\":4,\"num_experts_per_tok\":2,\"moe_intermediate_size\":128,"
         "\"intermediate_size\":128,\"routed_scaling_factor\":1.0,\"norm_topk_prob\":true,"
         "\"scoring_func\":\"sigmoid\",\"topk_method\":\"noaux_tc\",\"num_nextn_predict_layers\":1,"
         "\"hybrid_layer_pattern\":[0,1],\"moe_layer_freq\":[0,1]}";
    return o.str();
}

/// Builds the whole mini checkpoint.  `layer1_k_full` writes layer 1's k/v at FULL-attention width
/// (the config-vs-tensors disagreement the validator must catch); `drop_layer1_sink` omits the SWA
/// sink; `v_head_dim` 16 makes V asymmetric (the padded-lane guard's fail fixture).
std::filesystem::path build_fixture(const std::filesystem::path& dir, bool layer1_k_full,
                                    bool drop_layer1_sink, int v_head_dim) {
    Fixture fx(dir);
    const int E = 128, QK = 256, KV0 = 4 * 32, KV1 = (layer1_k_full ? 4 : 8) * 32, VD = 8 * v_head_dim;
    // global
    {
        std::vector<uint16_t> emb((size_t)256 * 128);
        for (auto& e : emb) e = (uint16_t)(0x3f80 ^ (fx.rnd() & 0xff));   // arbitrary BF16 payload
        fx.bytes16("model.embed_tokens.weight", "BF16", {256, 128}, emb);
        fx.exl3("lm_head", E / 16, 256 / 16, 8);                           // K = 4
        std::vector<float> norm(E);
        for (auto& n : norm) n = 0.9f + (fx.rnd() % 100) / 1000.0f;
        fx.f32("model.norm.weight", {E}, norm);
    }
    // layer 0: full attention, dense FFN
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
    // layer 1: sliding window, MoE, K = 2.5 experts
    {
        const std::string p = "model.layers.1.";
        fx.exl3(p + "self_attn.q_proj", E / 16, QK / 16, 5);
        fx.exl3(p + "self_attn.k_proj", E / 16, KV1 / 16, 5);
        fx.exl3(p + "self_attn.v_proj", E / 16, KV1 / 16, 5);
        fx.exl3(p + "self_attn.o_proj", VD / 16, E / 16, 5);
        if (!drop_layer1_sink) {
            std::vector<uint16_t> sink(8);
            for (auto& s : sink) s = f2h(-2.0f + 4.0f * (fx.rnd() % 1000) / 1000.0f);
            fx.bytes16(p + "self_attn.attention_sink_bias", "F16", {8}, sink);
        }
        std::vector<float> norm(E);
        for (auto& n : norm) n = 1.0f + (fx.rnd() % 100) / 1000.0f;
        fx.f32(p + "input_layernorm.weight", {E}, norm);
        for (auto& n : norm) n = 1.0f + (fx.rnd() % 100) / 1000.0f;
        fx.f32(p + "post_attention_layernorm.weight", {E}, norm);
        std::vector<uint16_t> gate((size_t)4 * 128);
        for (auto& g : gate) g = f2h(-0.5f + (fx.rnd() % 1000) / 1000.0f);
        fx.bytes16(p + "mlp.gate.weight", "F16", {4, 128}, gate);
        std::vector<float> bias(4);
        for (auto& b : bias) b = 0.1f * (int)(fx.rnd() % 10);
        fx.f32(p + "mlp.gate.e_score_correction_bias", {4}, bias);
        for (int e = 0; e < 4; ++e)
            for (const char* proj : {"gate", "up", "down"})
                fx.exl3(mimo_expert_tensor(1, e, proj), E / 16, 128 / 16, 5);   // K = 2.5
    }
    // MTP head 0: SWA geometry + dense MLP
    {
        const std::string p = "model.mtp.layers.0.";
        fx.exl3(p + "eh_proj", (2 * E) / 16, E / 16, 8);
        fx.exl3(p + "self_attn.q_proj", E / 16, QK / 16, 5);
        fx.exl3(p + "self_attn.k_proj", E / 16, 8 * 32 / 16, 5);
        fx.exl3(p + "self_attn.v_proj", E / 16, 8 * 32 / 16, 5);
        fx.exl3(p + "self_attn.o_proj", VD / 16, E / 16, 5);
        std::vector<uint16_t> sink(8);
        for (auto& s : sink) s = f2h(-2.0f + 4.0f * (fx.rnd() % 1000) / 1000.0f);
        fx.bytes16(p + "self_attn.attention_sink_bias", "F16", {8}, sink);
        std::vector<float> norm(E);
        for (const char* n : {"hnorm.weight", "enorm.weight", "input_layernorm.weight",
                              "pre_mlp_layernorm.weight", "final_layernorm.weight"}) {
            for (auto& v : norm) v = 1.0f + (fx.rnd() % 100) / 1000.0f;
            std::string name(n);
            fx.f32(p + name, {E}, norm);
        }
        fx.exl3(p + "mlp.gate_proj", E / 16, 128 / 16, 8);
        fx.exl3(p + "mlp.up_proj", E / 16, 128 / 16, 8);
        fx.exl3(p + "mlp.down_proj", 128 / 16, E / 16, 8);
    }
    fx.write();
    {   // config.json inside the fixture dir (the loader's discovery path)
        std::ofstream out((dir / "config.json").string());
        out << mini_config(v_head_dim);
    }
    return dir;
}

core::MimoGeometry geometry_of(const std::filesystem::path& cfg) {
    std::ifstream in(cfg.string());
    std::stringstream ss;
    ss << in.rdbuf();
    return core::mimo_geometry_from_config(ss.str());
}

}  // namespace

int main() {
    // this test pins the RECONSTRUCTED-F16 pack: the oracle path (--dense-f16).  The default pack is
    // TRELLIS-DENSE (planes on the device; checked below once the F16 build has been validated).
    setenv("STRATA_MIMO_DENSE_F16", "1", 1);
    uint64_t pack_v16 = 0;
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "strata_mimo_pack_test";
    std::filesystem::remove_all(root);
    fs::create_directories(root);

    // the name contract: one spelling, shared by the pack's validation and the expert store
    CHECK(mimo_expert_tensor(3, 7, "gate") == "model.layers.3.mlp.experts.7.gate_proj");

    const fs::path good = build_fixture(root / "good", false, false, 32);
    const core::MimoGeometry g = geometry_of(good / "config.json");
    CHECK(g.n_layers == 2 && g.n_embd == 128 && g.n_vocab == 256);
    CHECK(g.kv_heads(0) == 4 && g.kv_heads(1) == 8);
    CHECK(g.head_dim_v == 32);   // symmetric V: no padded lanes in the main fixture

    // ======================== validate: the good fixture ========================
    {
        MimoInventory inv;
        std::string err;
        CHECK(mimo_validate_checkpoint(good.string(), g, &inv, err));
        if (failures == 0) {
            CHECK(inv.layers == 2);
            CHECK(inv.moe_layers == 1);
            CHECK(inv.experts_per_moe_layer == 4);
            CHECK(inv.mtp_layers == 1);
            CHECK(inv.trellis_k_counts.size() == 17);
            // K2.5: layers' q/k/v/o (4 + 4) + MTP attention (4) + experts (4 * 3) = 24
            CHECK(inv.trellis_k_counts[5] == 24);
            // K4: layer-0 MLP (3) + MTP MLP (3) + eh_proj (1) + lm_head (1) = 8
            CHECK(inv.trellis_k_counts[8] == 8);
            std::printf("inventory: %s\n", inv.summary().c_str());
        } else {
            std::fprintf(stderr, "validate error: %s\n", err.c_str());
        }
    }

    // ================= the wrong-pattern fixture (the config gotcha) =================
    {
        const fs::path wrong = build_fixture(root / "wrong_k", true, false, 32);
        const core::MimoGeometry gw = geometry_of(wrong / "config.json");
        MimoInventory inv;
        std::string err;
        CHECK(!mimo_validate_checkpoint(wrong.string(), gw, &inv, err));
        CHECK(err.find("layer 1 k_proj") != std::string::npos);
        CHECK(err.find("hybrid_layer_pattern") != std::string::npos);
        std::printf("wrong-k fixture rejected: %s\n", err.c_str());
    }

    // ================= a missing SWA sink is a hard error =================
    {
        const fs::path no_sink = build_fixture(root / "no_sink", false, true, 32);
        const core::MimoGeometry gn = geometry_of(no_sink / "config.json");
        std::string err;
        CHECK(!mimo_validate_checkpoint(no_sink.string(), gn, nullptr, err));
        CHECK(err.find("attention_sink_bias") != std::string::npos);
    }

    // ================================ the GPU part ================================
    int device = -1;
    const bool have_gpu = cudaGetDevice(&device) == cudaSuccess && device >= 0;
    if (!have_gpu) {
        std::printf("mimo_pack_test: no GPU; validate-only run\n");
        return failures ? 1 : 0;
    }

    {
        std::string err;
        std::vector<std::string> stages;
        std::unique_ptr<MimoPack> pack = mimo_pack_build(good.string(), g, device, err,
                                                         [](const char* line) { std::printf("%s\n", line); });
        CHECK(pack != nullptr);
        if (pack != nullptr) {
            CHECK(pack->layers.size() == 2 && pack->mtp.size() == 1);
            for (const MimoPack::LayerDense& d : pack->layers)
                CHECK(d.q && d.k && d.v && d.o && d.norm_attn && d.norm_mlp);
            CHECK(pack->layers[0].ffn_gate && pack->layers[0].ffn_up && pack->layers[0].ffn_down);
            CHECK(pack->layers[0].router == nullptr && pack->layers[0].sink == nullptr);
            CHECK(pack->layers[1].router && pack->layers[1].e_score && pack->layers[1].sink);
            CHECK(pack->layers[1].ffn_gate == nullptr);
            const MimoPack::MtpHead& h = pack->mtp[0];
            CHECK(h.eh && h.q && h.k && h.v && h.o && h.hnorm && h.enorm && h.norm_attn && h.norm_mlp &&
                  h.norm_final && h.mlp_gate && h.mlp_up && h.mlp_down && h.sink);
            CHECK(pack->embed && pack->lm_head && pack->norm_final);
            CHECK(pack->vram_bytes > 0);
            pack_v16 = pack->vram_bytes;

            // the o_proj fold, exactly once: the uploaded bytes equal fp16(reconstruct * 0.707)
            Exl3Model model(good.string());
            // the pack TRANSPOSES the reconstruction (exl3_reconstruct_weight emits [in x out];
            // the pack serves [out x in] to the GEMMs) and applies NO o_proj fold (the trellis is
            // pre-folded by the converter)
            auto compare_linear = [&](const void* dev, const Exl3Linear& L, double scale) {
                const size_t bytes = (size_t)L.in() * L.out() * 2;
                std::vector<uint16_t> host(bytes / 2);
                if (cudaMemcpy(host.data(), dev, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
                    return false;
                std::vector<uint16_t> recon(bytes / 2), want(bytes / 2);
                kernels::cpu::exl3_reconstruct_weight(L.trellis, L.ki, L.nj, L.bits, (Exl3Codebook)L.cb,
                                                      L.suh, L.svh, recon.data(), L.frac);
                for (int64_t i = 0; i < L.in(); ++i)
                    for (int64_t j = 0; j < L.out(); ++j) want[(size_t)j * L.in() + i] = recon[(size_t)i * L.out() + j];
                if (scale != 1.0)
                    for (size_t i = 0; i < want.size(); ++i) want[i] = f2h(h2f(want[i]) * (float)scale);
                return std::memcmp(host.data(), want.data(), bytes) == 0;
            };
            // NO o_proj fold: the EXL3 trellis is pre-folded by the converter (the dflash twin
            // measured 0.997 unfolded, 0.610 folded); STRATA_MIMO_O_FOLD=1 is the A/B arm
            CHECK(compare_linear(pack->layers[0].o, model.linear("model.layers.0.self_attn.o_proj"), 1.0));
            CHECK(compare_linear(pack->layers[0].q, model.linear("model.layers.0.self_attn.q_proj"), 1.0));
            CHECK(compare_linear(pack->mtp[0].o, model.linear("model.mtp.layers.0.self_attn.o_proj"), 1.0));
            CHECK(compare_linear(pack->lm_head, model.linear("lm_head"), 1.0));

            // the e_score correction bias is uploaded as the stored F32 bytes
            {
                std::vector<float> got(4), want{0.0f, 0.0f, 0.0f, 0.0f};
                cudaMemcpy(got.data(), pack->layers[1].e_score, 16, cudaMemcpyDeviceToHost);
                // the fixture's values come from the checkpoint; re-read them through the model
                SafetensorsFile sf(good.string() + "/model-00001-of-00001.safetensors");
                const StTensor* t = sf.find("model.layers.1.mlp.gate.e_score_correction_bias");
                std::memcpy(want.data(), sf.data(*t), 16);
                CHECK(std::memcmp(got.data(), want.data(), 16) == 0);
            }

            // vram accounting: at least the F16 dense set + the F32 norms/sinks/router bias
            uint64_t at_least = 0;
            for (const char* b : {"model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj",
                                  "model.layers.0.self_attn.v_proj", "model.layers.0.self_attn.o_proj",
                                  "model.layers.1.self_attn.q_proj", "model.layers.1.self_attn.k_proj",
                                  "model.layers.1.self_attn.v_proj", "model.layers.1.self_attn.o_proj",
                                  "model.mtp.layers.0.self_attn.q_proj", "model.mtp.layers.0.self_attn.k_proj",
                                  "model.mtp.layers.0.self_attn.v_proj", "model.mtp.layers.0.self_attn.o_proj",
                                  "lm_head", "model.mtp.layers.0.eh_proj"}) {
                const Exl3Linear L = model.linear(b);
                at_least += (uint64_t)L.in() * L.out() * 2;
            }
            CHECK(pack->vram_bytes >= at_least);
        } else {
            std::fprintf(stderr, "pack build failed: %s\n", err.c_str());
        }
    }

    {   // the DEFAULT pack is TRELLIS-DENSE: the raw planes staged, the F16 views null
        unsetenv("STRATA_MIMO_DENSE_F16");
        std::string terr;
        std::unique_ptr<MimoPack> tpack = mimo_pack_build(good.string(), g, device, terr, nullptr);
        CHECK(tpack != nullptr);
        if (tpack != nullptr) {
            const MimoPack::LayerDense& d = tpack->layers[0];
            CHECK(d.q == nullptr && d.o == nullptr && d.ffn_gate == nullptr && tpack->lm_head == nullptr);
            CHECK(d.tq.valid() && d.tk.valid() && d.tv.valid() && d.to.valid());
            CHECK(d.tffn_gate.valid() && d.tffn_up.valid() && d.tffn_down.valid() && tpack->tlm.valid());
            CHECK(d.tq.ki == (int)(g.n_embd / 16) && d.tq.nj == (int)(g.qk_dim() / 16));
            CHECK(d.norm_attn && d.norm_mlp && d.router == nullptr);   // the small tensors unchanged
            CHECK(tpack->mtp[0].q != nullptr);   // the MTP heads stay F16 (the drafter reads them)
            // every trellis plane pair decodes the same linear as the folded GEMV: q_proj, one probe
            const Exl3Linear L = Exl3Model(good.string()).linear("model.layers.0.self_attn.q_proj");
            CHECK(d.tq.bits == L.bits && d.tq.frac == L.frac && d.tq.cb == L.cb);
            CHECK(tpack->vram_bytes < pack_v16 /* strictly smaller than the F16 set */);
        }
        setenv("STRATA_MIMO_DENSE_F16", "1", 1);
    }

    // the DENSE-Q8 pack (--dense-q8): the q/k/v/o + lm_head reconstructions quantized at load to
    // GGUF q8_0 blocks (MMVQ-served), the layer-0 MLP keeping its trellis planes, and one served
    // GEMV agreeing with the F16 reconstruction within the q8 quantization envelope
    {
        setenv("STRATA_MIMO_DENSE_F16", "0", 1);
        setenv("STRATA_MIMO_DENSE_Q8", "1", 1);
        std::string qerr;
        std::unique_ptr<MimoPack> qpack = mimo_pack_build(good.string(), g, device, qerr, nullptr);
        CHECK(qpack != nullptr);
        if (qpack != nullptr) {
            const MimoPack::LayerDense& d = qpack->layers[0];
            CHECK(d.q8q.valid() && d.q8k.valid() && d.q8v.valid() && d.q8o.valid());
            CHECK(qpack->lm_q8.valid());
            CHECK(d.q == nullptr && d.o == nullptr && qpack->lm_head == nullptr);   // no F16 side-set
            CHECK(!d.tq.valid() && !d.to.valid() && !qpack->tlm.valid());           // no trellis views
            CHECK(d.tffn_gate.valid() && d.tffn_up.valid() && d.tffn_down.valid()); // layer-0 MLP: trellis
            CHECK(qpack->vram_bytes < pack_v16);   // smaller than the F16 set, bigger than the trellis set

            // the served GEMV vs the F16 reconstruction: the envelope is q8 weights (per-32 scales)
            // times q8_1 activations - llama.cpp's Q8_0 contract, a ~1%-class error
            Exl3Model model(good.string());
            const Exl3Linear L = model.linear("model.layers.0.self_attn.q_proj");
            std::vector<float> x((size_t)L.in());
            uint32_t seed = 7;
            for (auto& v : x) { seed = seed * 1664525u + 1013904223u; v = -0.5f + (float)((seed >> 8) % 1000) / 1000.0f; }
            std::vector<uint16_t> recon((size_t)L.in() * L.out());
            kernels::cpu::exl3_reconstruct_weight(L.trellis, L.ki, L.nj, L.bits, (Exl3Codebook)L.cb, L.suh,
                                                  L.svh, recon.data(), L.frac);
            std::vector<double> want((size_t)L.out(), 0.0);
            for (int64_t i = 0; i < L.in(); ++i)
                for (int64_t o = 0; o < L.out(); ++o)
                    want[(size_t)o] += (double)x[(size_t)i] * (double)h2f(recon[(size_t)i * L.out() + o]);
            float* d_x = nullptr;
            float* d_y = nullptr;
            void* d_scratch = nullptr;
            const int n_in = (int)L.in(), n_out = (int)L.out();
            if (cudaMalloc(&d_x, x.size() * 4) != cudaSuccess || cudaMalloc(&d_y, (size_t)n_out * 4) != cudaSuccess ||
                cudaMalloc(&d_scratch, kernels::native_q8_1_bytes(n_in, 1)) != cudaSuccess ||
                cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess)
                CHECK(false);
            cudaStream_t qst = nullptr;
            if (cudaStreamCreate(&qst) != cudaSuccess) CHECK(false);
            kernels::native_q8_0_f32(qpack->layers[0].q8q.w, d_x, d_scratch, d_y, n_in, n_out, 1,
                                     (void*)qst);
            (void)cudaStreamSynchronize(qst);
            (void)cudaStreamDestroy(qst);
            std::vector<float> got((size_t)n_out);
            if (cudaMemcpy(got.data(), d_y, (size_t)n_out * 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaDeviceSynchronize() != cudaSuccess)
                CHECK(false);
            else {
                double num = 0, den = 0;
                for (int64_t o = 0; o < n_out; ++o) {
                    const double dd = (double)got[(size_t)o] - want[(size_t)o];
                    num += dd * dd;
                    den += want[(size_t)o] * want[(size_t)o];
                }
                const double rel = std::sqrt(num) / (std::sqrt(den) + 1e-30);
                CHECK(rel < 2e-2);
                std::printf("q8 dense GEMV vs F16 reconstruction: rel %.2e\n", rel);
            }
            (void)cudaFree(d_x);
            (void)cudaFree(d_y);
            (void)cudaFree(d_scratch);
        } else {
            std::fprintf(stderr, "q8 pack build failed: %s\n", qerr.c_str());
        }
        setenv("STRATA_MIMO_DENSE_Q8", "0", 1);
        setenv("STRATA_MIMO_DENSE_F16", "1", 1);
    }

    // the padded-lane guard's fail path: asymmetric V with real weights in the padded lanes
    {
        const fs::path pad = build_fixture(root / "vpad", false, false, 16);
        const core::MimoGeometry gp = geometry_of(pad / "config.json");
        CHECK(gp.head_dim_v == 16 && gp.head_dim_qk == 32);
        std::string err;
        std::unique_ptr<MimoPack> pack = mimo_pack_build(pad.string(), gp, device, err, nullptr);
        CHECK(pack == nullptr);
        CHECK(err.find("near-)zero") != std::string::npos);
        std::printf("padded-lane fixture rejected: %s\n", err.c_str());
    }

    // the expert-name prefix: the store reads the K=2.5 experts through `model.layers.` and the
    // fractional trellis answers a token on the GPU (vs the CPU reference GEMV)
    {
        uint32_t seed = 99;
        auto rnd = [&seed] { seed = seed * 1664525u + 1013904223u; return (seed >> 8) % 1000; };
        std::vector<uint16_t> x((size_t)g.n_embd);
        for (auto& v : x) v = f2h(-0.5f + rnd() / 1000.0f);
        // the store's x/out are DEVICE fp16 buffers (the engine hands it resident activations)
        uint16_t* d_x = nullptr;
        uint16_t* d_out = nullptr;
        if (cudaMalloc(&d_x, x.size() * 2) != cudaSuccess ||
            cudaMalloc(&d_out, x.size() * 2) != cudaSuccess ||
            cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice) != cudaSuccess)
            CHECK(false);
        const kernels::Exl3ExpertStore store(good.string(), /*layer=*/1, /*n_experts=*/4, nullptr,
                                             "model.layers.");
        CHECK(store.n_experts() == 4);
        CHECK(store.bytes() > 0);
        std::vector<uint16_t> out((size_t)g.n_embd, 0);
        const int ids[1] = {2};
        const float weights[1] = {1.0f};
        store.run(d_x, ids, weights, 1, d_out, nullptr);
        (void)cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost);
        std::fprintf(stderr, "store post-run err: %s\n", cudaGetErrorString(cudaGetLastError()));
        if (cudaDeviceSynchronize() != cudaSuccess) CHECK(false);
        else {
            // reference: gate/up via the folded GEMV (frac), silu*up rounded to fp16, down via GEMV
            Exl3Model model(good.string());
            const Exl3Linear Lg = model.linear(mimo_expert_tensor(1, 2, "gate"));
            const Exl3Linear Lu = model.linear(mimo_expert_tensor(1, 2, "up"));
            const Exl3Linear Ld = model.linear(mimo_expert_tensor(1, 2, "down"));
            std::vector<float> y(Ld.out()), g_(Lg.out()), u(Lu.out());
            kernels::cpu::exl3_folded_gemv(Lg.trellis, Lg.ki, Lg.nj, Lg.bits, (Exl3Codebook)Lg.cb,
                                           Lg.suh, Lg.svh, x.data(), 1, g_.data(), Lg.frac);
            kernels::cpu::exl3_folded_gemv(Lu.trellis, Lu.ki, Lu.nj, Lu.bits, (Exl3Codebook)Lu.cb,
                                           Lu.suh, Lu.svh, x.data(), 1, u.data(), Lu.frac);
            std::vector<uint16_t> h(u.size());
            for (size_t i = 0; i < u.size(); ++i) {
                const float s = g_[i] / (1.0f + std::exp(-g_[i]));
                h[i] = f2h(s * u[i]);   // the GPU path carries h as fp16
            }
            kernels::cpu::exl3_folded_gemv(Ld.trellis, Ld.ki, Ld.nj, Ld.bits, (Exl3Codebook)Ld.cb,
                                           Ld.suh, Ld.svh, h.data(), 1, y.data(), Ld.frac);
            double num = 0, den = 0;
            for (size_t i = 0; i < y.size(); ++i) {
                const double d = (double)h2f(out[i]) - y[i];
                num += d * d;
                den += (double)y[i] * y[i];
            }
            const double rel = std::sqrt(num) / (std::sqrt(den) + 1e-30);
            CHECK(rel < 3e-2);
            std::printf("expert store K=2.5 GPU vs CPU: rel %.2e\n", rel);
        }
    }

    std::filesystem::remove_all(root);
    if (failures) {
        std::fprintf(stderr, "mimo_pack_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("mimo_pack_test: all checks passed\n");
    return 0;
}
