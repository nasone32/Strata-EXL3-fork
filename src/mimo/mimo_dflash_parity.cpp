// src/mimo/mimo_dflash_parity.cpp - GPU DFlash drafter vs the fp64 CPU reference
// (notes/MIMO-V2-PORT.md §1, include/strata/mimo/dflash.hpp).
//
// No model: the test writes a SYNTHETIC dflash dir on the fly (scaled: hidden 512, 5 layers,
// 8 q / 8 kv heads x 128 - head_dim stays 128 because q_norm/k_norm are that wide - inter 1024,
// window 256, drop page 64 -> ring 384, block 8, taps 3 x 512) as real safetensors files and
// loads it through the real loader (from_dir + dflash_load), so the BF16 -> F16 conversion, the
// o_proj x 0.612 fold and the config reader are under test too.
//
// The CPU reference is deliberately independent of the drafter but MIRRORS ITS STORAGE: every
// weight is held as the F16 image of the same BF16 bits the file carries (built here through the
// same bf16 -> f32 -> f16 widening, not read back from the loader), every value the GPU stores
// as F16 (normed inputs, projected q/k/v, attention output before o_proj, silu*up, ring rows)
// is snapped to F16 in the reference too, and the math between those points runs in fp64.  The
// comparison then measures the GPU's fp32 arithmetic alone, and the gates sit at 2e-2 like the
// MTP drafter's test.  A pick whose fp64 logit margin is within 1e-3 relative of the GPU's is a
// near-tie, not a failure.
//
// Cases:
//   kernel    mimo_attn_block alone: parity (GQA 1 and 8), the RIGHT extent honored and capped,
//             the LEFT window clamp, sinks dominating the softmax max, and BIDIRECTIONALITY
//             (row 0's output moves when a FUTURE row's V is perturbed - a causal-only kernel
//             cannot pass this).
//   drafter   one full round trip: commit(300 rows) -> draft -> commit(3 rows, overwriting the
//             speculative block) -> draft -> commit(303 rows, RING SHIFT) -> draft at P=606 >
//             ring 384.  Ring K/V and state rows are compared against the reference after every
//             step; proposals must match (near-ties excepted).  The erasure case also proves the
//             committed rows equal the taps-derived values, not the stale speculative ones.
//   order     q/k_norm BEFORE rope: a reference that swaps them (rope first) must sit far from
//             the GPU while the correct order sits close.
//   mask      the mask rows use mask_embedding, not the target's embedding of mask_token: a
//             reference that embeds the mask token must sit far from the GPU.
//   exl3      the whole drafter flow runs a SECOND time on a synthetic QUANTIZED dir (every
//             linear a random K=4 mul1 trellis + suh/svh, the raw tensors BF16/F16 in-file, no
//             separate mask file) through the real reconstruction path, against a reference
//             whose linears are the same trellises decoded by the shared CPU function.
//   real dir  STRATA_DFLASH_REAL_DIR (or the on-disk model) loads the REAL dflash/ (EXL3) and
//             dflash-bf16/ dirs: mask_embedding and the norms BIT-IDENTICAL between the two
//             variants, sinks F16-close, and a reconstructed o_proj row vs the bf16 dir's same
//             row x 0.612 (cosine + norm ratio: the fold applied exactly once).  Skipped with a
//             line of output when the dir is absent.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "strata/artifact/safetensors.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/cpu/exl3.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/mimo_attn.hpp"
#include "strata/mimo/dflash.hpp"

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_bf16;
using strata::kernels::f32_from_f16;

namespace {

// The GPU-call guards: CHK returns 1 from an int context (main, the int-returning lambdas);
// CU exits outright where no value can be returned (the void hooks, void capture lambdas).
#define CHK(e, what)                                                                              \
    do {                                                                                          \
        const cudaError_t _e = (e);                                                               \
        if (_e != cudaSuccess) {                                                                  \
            std::printf("CUDA %s: %s\n", what, cudaGetErrorString(_e));                          \
            return 1;                                                                             \
        }                                                                                         \
    } while (0)
#define CU(e, what)                                                                               \
    do {                                                                                          \
        const cudaError_t _e = (e);                                                               \
        if (_e != cudaSuccess) {                                                                  \
            std::printf("CUDA %s: %s\n", what, cudaGetErrorString(_e));                          \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

// ----------------------------------------------------------------------------------- test geometry

constexpr int L = 5, H = 512, INTER = 1024;
constexpr int NQ = 8, NKV = 8, HD = 128, ROT = 64, TH = 512, TAPS = 3;
constexpr int BLOCK = 8, WINDOW = 256, RIGHT = 255, PAGE = 64, RING = 384;
constexpr double EPS = 1e-6, THETA = 1e4, VSCALE = 0.612;
constexpr int QD = NQ * HD, KVD = NKV * HD, FCIN = TAPS * TH;
constexpr int VOCAB = 1024, VOCAB_LM = 256;
constexpr double SM = 0.08838834764831845;   // 128^-0.5
constexpr double TOL = 2e-2;         // one commit pass (fc + norms + rope)
constexpr double STACK_TOL = 5e-2;   // the 5-layer draft stack (see Compare::ok)

// ------------------------------------------------------------ BF16 + safetensors fixture writer

uint16_t bf16_bits(double v) {
    // round-to-nearest-even f32 -> bf16, enough for a synthetic fixture
    uint32_t x;
    const float f = (float) v;
    std::memcpy(&x, &f, 4);
    const uint32_t rounded = x + 0x7FFFu + ((x >> 16) & 1u);
    return (uint16_t) (rounded >> 16);
}
double bf16_value(double v) { return (double) f32_from_bf16(bf16_bits(v)); }

struct Fixture {
    struct Layer {
        std::vector<double> ln1, ln2, q, k, v, o, qn, kn, sink, gate, up, down;
    };
    std::vector<Layer> layers;
    std::vector<double> fc, hidden_norm, mask_embed, final_norm;
    std::vector<double> embed, lm;   // the "target's" tables (hook-side, never in the dir)

    // QUANTIZED variant: when the qimg_* images are non-empty, the linears ship as trellis
    // (written by write_dir_quantized below) and the reference uses these F16 images - which
    // the test derives from the SAME trellises through the shared CPU decode, so the loader's
    // band/transpose/fold path is checked against an independently computed oracle.  o's image
    // is the FOLDED one, exactly as uploaded.  sinks_f16: the quantized dir stores sinks F16.
    bool quant = false;
    bool sinks_f16 = false;
    std::vector<uint16_t> qimg_fc;
    struct LayerQ {
        std::vector<uint16_t> q, k, v, o, gate, up, down;
    };
    std::vector<LayerQ> qlayers;
};

std::vector<double> rand_norm(std::mt19937_64& rng, size_t n, double sigma) {
    std::normal_distribution<double> nd(0.0, sigma);
    std::vector<double> v(n);
    for (double& x : v) x = nd(rng);
    return v;
}

std::vector<double> ones_noisy(std::mt19937_64& rng, size_t n, double sigma) {
    std::vector<double> v = rand_norm(rng, n, sigma);
    for (double& x : v) x += 1.0;
    return v;
}

Fixture make_fixture(uint64_t seed) {
    std::mt19937_64 rng(seed);
    Fixture fx;
    fx.fc = rand_norm(rng, (size_t) H * FCIN, 0.05);
    fx.hidden_norm = ones_noisy(rng, H, 0.05);
    fx.mask_embed = rand_norm(rng, H, 0.5);
    fx.final_norm = ones_noisy(rng, H, 0.05);
    for (int l = 0; l < L; ++l) {
        Fixture::Layer ly;
        ly.ln1 = ones_noisy(rng, H, 0.05);
        ly.ln2 = ones_noisy(rng, H, 0.05);
        ly.q = rand_norm(rng, (size_t) QD * H, 0.05);
        ly.k = rand_norm(rng, (size_t) KVD * H, 0.05);
        ly.v = rand_norm(rng, (size_t) KVD * H, 0.05);
        ly.o = rand_norm(rng, (size_t) H * QD, 0.05);
        ly.qn = ones_noisy(rng, HD, 0.05);
        ly.kn = ones_noisy(rng, HD, 0.05);
        ly.sink.resize(NQ);
        for (int h = 0; h < NQ; ++h)
            // half the heads' sinks dominate the softmax max: a kernel that scales the sink by
            // sm_scale, or skips it, cannot match the reference on those heads
            ly.sink[h] = (h % 2) == 0 ? -1.0 + 2.0 * std::generate_canonical<double, 10>(rng)
                                      : 6.0 + 6.0 * std::generate_canonical<double, 10>(rng);
        ly.gate = rand_norm(rng, (size_t) INTER * H, 0.05);
        ly.up = rand_norm(rng, (size_t) INTER * H, 0.05);
        ly.down = rand_norm(rng, (size_t) H * INTER, 0.05);
        fx.layers.push_back(std::move(ly));
    }
    // the embed rows are snapped to F16 values so the hook's H2D copy is exact on both sides
    fx.embed = rand_norm(rng, (size_t) VOCAB * H, 0.5);
    for (double& x : fx.embed) x = (double) f32_from_f16(f16_from_f32((float) x));
    fx.lm = rand_norm(rng, (size_t) VOCAB_LM * H, 0.1);
    return fx;
}

struct StTensorOut {
    std::string name;
    std::vector<uint16_t> data;
    int64_t rows = 1;   // 2-D [rows, cols] when cols > 0, else 1-D [rows]
    int64_t cols = 0;
};

void st_write(const std::string& path, const std::vector<StTensorOut>& tensors) {
    std::string json = "{";
    size_t off = 0;
    bool first = true;
    std::vector<uint8_t> data;
    for (const StTensorOut& t : tensors) {
        const size_t bytes = t.data.size() * 2;
        char buf[512];   // tensor names here are short and fixed
        if (t.cols > 0)
            std::snprintf(buf, sizeof buf,
                          "%s\"%s\":{\"dtype\":\"BF16\",\"shape\":[%lld,%lld],\"data_offsets\":[%zu,%zu]}",
                          first ? "" : ",", t.name.c_str(), (long long) t.rows,
                          (long long) t.cols, off, off + bytes);
        else
            std::snprintf(buf, sizeof buf,
                          "%s\"%s\":{\"dtype\":\"BF16\",\"shape\":[%zu],\"data_offsets\":[%zu,%zu]}",
                          first ? "" : ",", t.name.c_str(), t.data.size(), off, off + bytes);
        json += buf;
        first = false;
        off += bytes;
        for (uint16_t b : t.data) {
            data.push_back((uint8_t) (b & 0xFF));
            data.push_back((uint8_t) (b >> 8));
        }
    }
    json += "}";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        std::exit(1);
    }
    const uint64_t n = json.size();
    std::fwrite(&n, 8, 1, f);
    std::fwrite(json.data(), 1, json.size(), f);
    std::fwrite(data.data(), 1, data.size(), f);
    std::fclose(f);
}

/// The fixture's config.json text, shared by the bf16 and quantized dir writers.
std::string write_dir_config_text() {
    return std::string(R"({
  "architectures": ["DFlashDraftModel"],
  "model_type": "qwen3",
  "hidden_size": 512,
  "intermediate_size": 1024,
  "num_hidden_layers": 5,
  "num_attention_heads": 8,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "v_head_dim": 128,
  "partial_rotary_factor": 0.5,
  "block_size": 8,
  "dflash_config": {
    "target_layer_ids": [2, 7, 11],
    "mask_token_id": 42,
    "attention_value_scale": 0.612,
    "tap_shift": 0
  },
  "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
  "sliding_window": 256,
  "is_causal": false,
  "num_target_layers": 48,
  "target_hidden_size": 512,
  "rope_theta": 10000.0,
  "rms_norm_eps": 1e-06,
  "add_swa_attention_sink_bias": true,
  "dflash_ring_page": 64
}
)");
}

void write_dir(const std::string& dir, const Fixture& fx) {
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    const std::string cfg = write_dir_config_text();
    FILE* f = std::fopen((dir + "/config.json").c_str(), "wb");
    std::fwrite(cfg.data(), 1, cfg.size(), f);
    std::fclose(f);

    const auto vec1d = [](const std::vector<double>& v, const char* name) {
        return StTensorOut{name, [&] {
                               std::vector<uint16_t> out;
                               out.reserve(v.size());
                               for (double x : v) out.push_back(bf16_bits(x));
                               return out;
                           }()};
    };
    const auto mat2d = [](const std::vector<double>& v, int64_t rows, int64_t cols, const char* name) {
        return StTensorOut{name, [&] {
                               std::vector<uint16_t> out;
                               out.reserve(v.size());
                               for (double x : v) out.push_back(bf16_bits(x));
                               return out;
                           }(), rows, cols};
    };
    std::vector<StTensorOut> main;
    main.push_back(mat2d(fx.fc, H, FCIN, "fc.weight"));
    main.push_back(vec1d(fx.hidden_norm, "hidden_norm.weight"));
    main.push_back(vec1d(fx.final_norm, "norm.weight"));
    for (int l = 0; l < L; ++l) {
        const std::string p = "layers." + std::to_string(l);
        const Fixture::Layer& ly = fx.layers[l];
        main.push_back(vec1d(ly.ln1, (p + ".input_layernorm.weight").c_str()));
        main.push_back(vec1d(ly.ln2, (p + ".post_attention_layernorm.weight").c_str()));
        main.push_back(mat2d(ly.q, QD, H, (p + ".self_attn.q_proj.weight").c_str()));
        main.push_back(mat2d(ly.k, KVD, H, (p + ".self_attn.k_proj.weight").c_str()));
        main.push_back(mat2d(ly.v, KVD, H, (p + ".self_attn.v_proj.weight").c_str()));
        main.push_back(mat2d(ly.o, H, QD, (p + ".self_attn.o_proj.weight").c_str()));
        main.push_back(vec1d(ly.qn, (p + ".self_attn.q_norm.weight").c_str()));
        main.push_back(vec1d(ly.kn, (p + ".self_attn.k_norm.weight").c_str()));
        main.push_back(vec1d(ly.sink, (p + ".self_attn.attention_sink_bias").c_str()));
        main.push_back(mat2d(ly.gate, INTER, H, (p + ".mlp.gate_proj.weight").c_str()));
        main.push_back(mat2d(ly.up, INTER, H, (p + ".mlp.up_proj.weight").c_str()));
        main.push_back(mat2d(ly.down, H, INTER, (p + ".mlp.down_proj.weight").c_str()));
    }
    st_write(dir + "/dflash_draft_model.safetensors", main);
    st_write(dir + "/mask_embedding.safetensors", {vec1d(fx.mask_embed, "mask_embedding")});
}

// ----------------------------------------------------- the QUANTIZED (EXL3) dflash dir writer
//
// Every linear becomes a RANDOM K=4 mul1 trellis + suh/svh (the exl3_frac_k_test convention:
// random 16-bit codes, scales in a sane F16 range) and the expected F16 image is derived here
// through the SAME shared CPU decode the loader calls - the oracle is independent of the
// loader's banding/transpose/upload, the trellis math itself is exl3_parity's scope.

struct StRaw {
    std::string name, dtype;         // "BF16" / "F16" / "I16" / "I32"
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;      // little-endian, packed
};

void st_write_raw(const std::string& path, const std::vector<StRaw>& ts) {
    std::string json = "{";
    size_t off = 0;
    bool first = true;
    std::vector<uint8_t> data;
    for (const StRaw& t : ts) {
        char shape[128], buf[512];
        shape[0] = '[';
        size_t sp = 1;
        for (size_t i = 0; i < t.shape.size(); ++i)
            sp += (size_t) std::snprintf(shape + sp, sizeof shape - sp, "%s%lld", i ? "," : "",
                                         (long long) t.shape[i]);
        sp += (size_t) std::snprintf(shape + sp, sizeof shape - sp, "]");
        std::snprintf(buf, sizeof buf, "%s\"%s\":{\"dtype\":\"%s\",\"shape\":%s,\"data_offsets\":[%zu,%zu]}",
                      first ? "" : ",", t.name.c_str(), t.dtype.c_str(), shape, off,
                      off + t.bytes.size());
        json += buf;
        first = false;
        off += t.bytes.size();
        data.insert(data.end(), t.bytes.begin(), t.bytes.end());
    }
    json += "}";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); std::exit(1); }
    const uint64_t n = json.size();
    std::fwrite(&n, 8, 1, f);
    std::fwrite(json.data(), 1, json.size(), f);
    std::fwrite(data.data(), 1, data.size(), f);
    std::fclose(f);
}

const uint32_t kMul1Marker = 2212286765u;   // the real quantization_config's mul1_multiplier

StRaw u16_tensor(const std::string& name, const std::string& dtype,
                 const std::vector<int64_t>& shape, const std::vector<uint16_t>& vals) {
    StRaw t{name, dtype, shape, {}};
    t.bytes.reserve(vals.size() * 2);
    for (uint16_t v : vals) {
        t.bytes.push_back((uint8_t) (v & 0xFF));
        t.bytes.push_back((uint8_t) (v >> 8));
    }
    return t;
}

struct TrellisSynth {
    std::vector<uint16_t> trellis, suh, svh;
    int64_t in = 0, out = 0;
};

TrellisSynth synth_trellis(std::mt19937_64& rng, int64_t in, int64_t out) {
    constexpr int64_t kWords = 64;   // 16 * K for the integer K = 4 the real dflash dir uses
    TrellisSynth t;
    t.in = in;
    t.out = out;
    t.trellis.resize((size_t) (in / 16) * (out / 16) * kWords);
    for (auto& w : t.trellis) w = (uint16_t) (rng() & 0xFFFF);
    const auto scale16 = [&]() {
        return f16_from_f32((float) (0.5 + 1.5 * std::generate_canonical<double, 10>(rng)));
    };
    t.suh.resize((size_t) in);
    t.svh.resize((size_t) out);
    for (auto& s : t.svh) s = scale16();
    for (auto& s : t.suh) s = scale16();
    // SIZING PASS: random trellis codes decode to wild magnitudes (the mul1 codebook is the
    // fp16-mantissa LUT), which blows the 5-layer residual stack up to inf.  The image is
    // linear in suh, so one decode with the base scales measures sigma and a single factor on
    // suh lands the weight sigma at the bf16 fixture's 0.05.
    {
        std::vector<uint16_t> probe((size_t) in * out);
        strata::kernels::cpu::exl3_reconstruct_weight(
            t.trellis.data(), (int) (in / 16), (int) (out / 16), 4,
            strata::kernels::cpu::Exl3Codebook::Mul1, t.suh.data(), t.svh.data(), probe.data(), 0);
        double ss = 0.0;
        for (uint16_t x : probe) ss += std::pow((double) f32_from_f16(x), 2.0);
        const double alpha = 0.05 / (std::sqrt(ss / (double) probe.size()) + 1e-30);
        for (auto& s : t.suh) s = f16_from_f32((float) ((double) f32_from_f16(s) * alpha));
    }
    return t;
}

/// The oracle's F16 image [out, in] of one trellis: the shared CPU decode (which emits the
/// input-major [in, out] image), transposed here, with the o_proj fold rounded like the loader.
std::vector<uint16_t> image_of(const TrellisSynth& t, double fold) {
    std::vector<uint16_t> rec((size_t) t.in * t.out);
    strata::kernels::cpu::exl3_reconstruct_weight(
        t.trellis.data(), (int) (t.in / 16), (int) (t.out / 16), /*bits=*/4,
        strata::kernels::cpu::Exl3Codebook::Mul1, t.suh.data(), t.svh.data(), rec.data(), 0);
    std::vector<uint16_t> w((size_t) t.out * t.in);
    for (int64_t i = 0; i < t.in; ++i)
        for (int64_t o = 0; o < t.out; ++o)
            w[(size_t) o * t.in + i] = rec[(size_t) i * t.out + o];
    if (fold != 1.0)
        for (uint16_t& x : w) x = f16_from_f32(f32_from_f16(x) * (float) fold);
    return w;
}

/// Writes the quantized dir shape (ONE model.safetensors, raw tensors in-file, no separate
/// mask file) and fills `fx`'s quant images so RefModel builds from the same trellises.
void write_dir_quantized(const std::string& dir, Fixture& fx, uint64_t seed) {
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    {   // same config as the bf16 fixture; the trellis tensors carry the quantization
        FILE* f = std::fopen((dir + "/config.json").c_str(), "wb");
        if (!f) { std::fprintf(stderr, "cannot write config\n"); std::exit(1); }
        const std::string cfg = write_dir_config_text();
        std::fwrite(cfg.data(), 1, cfg.size(), f);
        std::fclose(f);
    }
    std::mt19937_64 rng(seed);
    std::vector<StRaw> ts;
    const auto lin = [&](const std::string& base, int64_t in, int64_t out,
                         double fold) -> std::vector<uint16_t> {
        const TrellisSynth t = synth_trellis(rng, in, out);
        ts.push_back(u16_tensor(base + ".trellis", "I16", {in / 16, out / 16, 64}, t.trellis));
        ts.push_back(u16_tensor(base + ".suh", "F16", {in}, t.suh));
        ts.push_back(u16_tensor(base + ".svh", "F16", {out}, t.svh));
        StRaw mul1;
        mul1.name = base + ".mul1";
        mul1.dtype = "I32";
        mul1.shape = {};
        for (int b = 0; b < 4; ++b) mul1.bytes.push_back((uint8_t) (kMul1Marker >> (8 * b)));
        ts.push_back(std::move(mul1));
        return image_of(t, fold);
    };
    const auto bf16raw = [&](const std::string& name, const std::vector<double>& v) {
        std::vector<uint16_t> bits;
        bits.reserve(v.size());
        for (double x : v) bits.push_back(bf16_bits(x));
        ts.push_back(u16_tensor(name, "BF16", {(int64_t) v.size()}, bits));
    };
    fx.quant = true;
    fx.sinks_f16 = true;   // the real quantized dir stores the sinks F16
    fx.qimg_fc = lin("fc", FCIN, H, 1.0);
    bf16raw("hidden_norm.weight", fx.hidden_norm);
    bf16raw("norm.weight", fx.final_norm);
    bf16raw("mask_embedding", fx.mask_embed);   // INSIDE the file in this variant
    fx.qlayers.resize(L);
    for (int l = 0; l < L; ++l) {
        const std::string p = "layers." + std::to_string(l);
        bf16raw(p + ".input_layernorm.weight", fx.layers[l].ln1);
        bf16raw(p + ".post_attention_layernorm.weight", fx.layers[l].ln2);
        bf16raw(p + ".self_attn.q_norm.weight", fx.layers[l].qn);
        bf16raw(p + ".self_attn.k_norm.weight", fx.layers[l].kn);
        std::vector<uint16_t> sink16(NQ);
        for (int h = 0; h < NQ; ++h) sink16[h] = f16_from_f32((float) fx.layers[l].sink[h]);
        ts.push_back(u16_tensor(p + ".self_attn.attention_sink_bias", "F16", {NQ}, sink16));
        fx.qlayers[l].q = lin(p + ".self_attn.q_proj", H, QD, 1.0);
        fx.qlayers[l].k = lin(p + ".self_attn.k_proj", H, KVD, 1.0);
        fx.qlayers[l].v = lin(p + ".self_attn.v_proj", H, KVD, 1.0);
        fx.qlayers[l].o = lin(p + ".self_attn.o_proj", QD, H, 1.0);   // the trellis is pre-folded; the loader folds nothing here
        fx.qlayers[l].gate = lin(p + ".mlp.gate_proj", H, INTER, 1.0);
        fx.qlayers[l].up = lin(p + ".mlp.up_proj", H, INTER, 1.0);
        fx.qlayers[l].down = lin(p + ".mlp.down_proj", INTER, H, 1.0);
    }
    st_write_raw(dir + "/model.safetensors", ts);
}

// ----------------------------------------------------------------------------------- fp64 reference

struct Compare {
    double max_abs = 0.0, max_rel = 0.0, norm_rel = 0.0;
    size_t bad = 0;
    // The tolerance ladder: one commit pass through fc/norm/rope lands at ~4e-3 max_rel; a
    // 5-layer draft stack at ~2.5e-2 (a single F16 ulp flip in a rope'd key seeds a smooth
    // perturbation the residual stack carries forward - the device sincosf in mimo_rope_f16
    // cannot be bit-mirrored on the host).  Both sit far below a structural failure, which the
    // norm_rel gate (1e-2) catches separately: the WRONG-order and WRONG-input references below
    // land at norm_rel O(0.1..1).
    bool ok(double tol) const { return bad == 0 && max_rel < tol && norm_rel < 1e-2; }
    bool ok() const { return ok(TOL); }
};

Compare compare(const std::vector<double>& gpu, const std::vector<double>& ref) {
    Compare c;
    double num = 0.0, den = 0.0, ref_max = 0.0;
    for (double v : ref) ref_max = std::max(ref_max, std::fabs(v));
    const double floor_den = 0.05 * ref_max + 1e-12;
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

// Weights as the F16 images the LOADER uploads (bf16 file bits widened to f16, built here from
// the fixture doubles through the same bf16 rounding the file takes; o_proj folded first).
// Math runs in fp64 over these images, snapped back to F16 at every GPU staging point.
struct RefModel {
    const Fixture& fx;
    int64_t window_beg = 0, committed = 0;
    std::vector<double> rk, rv;   // [L][RING * KVD], values F16-snapped

    struct Layer {
        std::vector<uint16_t> ln1, ln2, q, k, v, o_folded, qn, kn, gate, up, down;
        std::vector<float> sink;
    };
    std::vector<Layer> layers;
    std::vector<uint16_t> fc, hidden_norm, mask_embed, final_norm;

    static double d16(uint16_t h) { return (double) f32_from_f16(h); }
    static uint16_t f16(double x) { return f16_from_f32((float) x); }
    static double snap(double x) { return (double) f32_from_f16(f16_from_f32((float) x)); }
    static std::vector<uint16_t> image(const std::vector<double>& v, double scale = 1.0) {
        std::vector<uint16_t> out;
        out.reserve(v.size());
        for (double x : v) {
            const double b = bf16_value(x);   // what the file carries
            out.push_back(scale == 1.0 ? f16(b) : f16(b * scale));
        }
        return out;
    }

    explicit RefModel(const Fixture& f) : fx(f), rk((size_t) L * RING * KVD),
                                          rv((size_t) L * RING * KVD) {
        fc = f.quant ? f.qimg_fc : image(f.fc);
        hidden_norm = image(f.hidden_norm);
        mask_embed = image(f.mask_embed);
        final_norm = image(f.final_norm);
        for (int l = 0; l < L; ++l) {
            Layer ly;
            ly.ln1 = image(f.layers[l].ln1);
            ly.ln2 = image(f.layers[l].ln2);
            if (f.quant) {   // the images the shared CPU decode produced from the same trellises
                ly.q = f.qlayers[l].q;
                ly.k = f.qlayers[l].k;
                ly.v = f.qlayers[l].v;
                ly.o_folded = f.qlayers[l].o;   // already folded, like the loader's upload
                ly.gate = f.qlayers[l].gate;
                ly.up = f.qlayers[l].up;
                ly.down = f.qlayers[l].down;
            } else {
                ly.q = image(f.layers[l].q);
                ly.k = image(f.layers[l].k);
                ly.v = image(f.layers[l].v);
                ly.o_folded = image(f.layers[l].o, VSCALE);
                ly.gate = image(f.layers[l].gate);
                ly.up = image(f.layers[l].up);
                ly.down = image(f.layers[l].down);
            }
            ly.qn = image(f.layers[l].qn);
            ly.kn = image(f.layers[l].kn);
            ly.sink.reserve(NQ);
            for (double s : f.layers[l].sink)
                ly.sink.push_back(f.sinks_f16
                                      ? f32_from_f16(f16_from_f32((float) s))
                                      : (float) bf16_value(s));
            layers.push_back(std::move(ly));
        }
    }

    static void rmsnorm(double* x, const uint16_t* w, int n) {
        double ss = 0.0;
        for (int i = 0; i < n; ++i) ss += x[i] * x[i];
        const double inv = 1.0 / std::sqrt(ss / n + EPS);
        for (int i = 0; i < n; ++i) x[i] = snap(x[i] * inv * d16(w[i]));
    }
    static void gemv(const double* x, const uint16_t* w, double* y, int n_in, int n_out) {
        for (int o = 0; o < n_out; ++o) {
            double s = 0.0;
            const uint16_t* wr = w + (size_t) o * n_in;
            for (int i = 0; i < n_in; ++i) s += x[i] * d16(wr[i]);
            y[o] = s;
        }
    }
    static void headnorm(double* h, const uint16_t* w) {   // one HD-wide head, in place
        double ss = 0.0;
        for (int i = 0; i < HD; ++i) ss += h[i] * h[i];
        const double inv = 1.0 / std::sqrt(ss / HD + EPS);
        for (int i = 0; i < HD; ++i) h[i] = snap(h[i] * inv * d16(w[i]));
    }
    static void rope(double* h, int64_t pos) {   // NEOX partial: pairs (i, i + ROT/2), i < ROT/2
        const int half = ROT / 2;
        for (int i = 0; i < half; ++i) {
            const double freq = std::pow(THETA, -2.0 * i / ROT);
            const double ang = std::fmod((double) pos * freq, 2.0 * 3.14159265358979323846);
            const double c = std::cos(ang), s = std::sin(ang);
            const double a = h[i], b = h[i + half];
            h[i] = snap(a * c - b * s);
            h[i + half] = snap(a * s + b * c);
        }
    }

    void shift_ring(int shift, int64_t cache_pos) {
        const int64_t keep = cache_pos - shift;
        if (keep <= 0) return;
        for (int l = 0; l < L; ++l) {
            std::memmove(&rk[(size_t) l * RING * KVD], &rk[((size_t) l * RING + shift) * KVD],
                         (size_t) keep * KVD * sizeof(double));
            std::memmove(&rv[(size_t) l * RING * KVD], &rv[((size_t) l * RING + shift) * KVD],
                         (size_t) keep * KVD * sizeof(double));
        }
    }

    // commit: fc -> hidden_norm -> per layer k (k_norm, rope at ABSOLUTE positions) and v into
    // the ring; mirrors DflashDrafter::commit_from_taps's bookkeeping exactly.  When
    // `speculative` is non-null it receives the k/v rows sitting at [position, position + n)
    // BEFORE the overwrite (the stale draft block, for the erasure check).
    void commit(const std::vector<double>& taps /* [TAPS][rows][TH] flattened, F16-snapped */,
                int64_t position, int rows, std::vector<double>* speculative = nullptr) {
        int64_t cache_pos = position - window_beg;
        if (cache_pos + rows > RING) {
            const int64_t shift = (cache_pos + rows + PAGE - 1) / PAGE * PAGE - RING;
            const int64_t keep = cache_pos - shift;
            if (keep > 0) {
                shift_ring((int) shift, cache_pos);
                window_beg += shift;
                cache_pos -= shift;
            } else {
                window_beg += shift;
                cache_pos = 0;
            }
        }
        const int64_t skip = cache_pos < 0 ? -cache_pos : 0;
        if (skip > 0) cache_pos = 0;
        const int n = rows - (int) skip;
        if (n <= 0) {
            committed = position + rows;
            return;
        }
        if (speculative)
            for (int l = 0; l < L; ++l) {
                const double* ksrc = &rk[((size_t) l * RING + cache_pos) * KVD];
                const double* vsrc = &rv[((size_t) l * RING + cache_pos) * KVD];
                speculative->insert(speculative->end(), ksrc, ksrc + (size_t) n * KVD);
                speculative->insert(speculative->end(), vsrc, vsrc + (size_t) n * KVD);
            }

        std::vector<double> ctx((size_t) n * H);
        for (int r = 0; r < n; ++r)
            for (int o = 0; o < H; ++o) {
                double s = 0.0;
                const uint16_t* wr = &fc[(size_t) o * FCIN];
                for (int t = 0; t < TAPS; ++t) {
                    const double* tr = &taps[((size_t) t * rows + r) * TH];
                    const uint16_t* wtr = wr + (size_t) t * TH;
                    for (int i = 0; i < TH; ++i) s += tr[i] * d16(wtr[i]);
                }
                ctx[(size_t) r * H + o] = snap(s);   // the F16 round before hidden_norm
            }
        for (int r = 0; r < n; ++r) rmsnorm(&ctx[(size_t) r * H], hidden_norm.data(), H);

        std::vector<double> kv((size_t) n * KVD);
        for (int l = 0; l < L; ++l) {
            const Layer& ly = layers[l];
            for (int r = 0; r < n; ++r) {
                gemv(&ctx[(size_t) r * H], ly.k.data(), &kv[(size_t) r * KVD], H, KVD);
                for (int i = 0; i < KVD; ++i) kv[(size_t) r * KVD + i] = snap(kv[(size_t) r * KVD + i]);
                for (int h = 0; h < NKV; ++h) {
                    headnorm(&kv[(size_t) r * KVD + (size_t) h * HD], ly.kn.data());
                    rope(&kv[(size_t) r * KVD + (size_t) h * HD], position + skip + r);
                }
            }
            std::memcpy(&rk[((size_t) l * RING + cache_pos) * KVD], kv.data(),
                        (size_t) n * KVD * sizeof(double));
            for (int r = 0; r < n; ++r) {
                gemv(&ctx[(size_t) r * H], ly.v.data(), &kv[(size_t) r * KVD], H, KVD);
                for (int i = 0; i < KVD; ++i) kv[(size_t) r * KVD + i] = snap(kv[(size_t) r * KVD + i]);
            }
            std::memcpy(&rv[((size_t) l * RING + cache_pos) * KVD], kv.data(),
                        (size_t) n * KVD * sizeof(double));
        }
        committed = position + rows;
    }

    // One draft round; returns the final-normed state [BLOCK][H] (F16-snapped).  When `ids` is
    // given it also fills it and the per-row LM logits.  swap_norm_rope runs rope BEFORE the
    // per-head norms (the wrong order, for the order check); embed_mask_with_target fills the
    // mask rows with the target's embed of the mask token instead of mask_embedding (the wrong
    // input, for the mask check).  Both variants still append their own k/v rows to the ring,
    // like the real one - the callers arrange for everything after them to overwrite.
    std::vector<double> draft(int32_t carry, int64_t P, std::vector<int>* ids,
                              std::vector<double>* logits = nullptr,
                              bool swap_norm_rope = false, bool embed_mask_with_target = false) {
        int64_t cache_pos = P - window_beg;
        if (cache_pos + BLOCK > RING) {
            const int shift = (BLOCK + PAGE - 1) / PAGE * PAGE;
            shift_ring(shift, cache_pos);
            window_beg += shift;
            cache_pos -= shift;
        }
        std::vector<double> x((size_t) BLOCK * H);
        std::memcpy(&x[0], &fx.embed[(size_t) carry * H], sizeof(double) * H);
        const double* mrow = nullptr;
        std::vector<double> mask_row_img(H);
        if (embed_mask_with_target) {
            std::memcpy(mask_row_img.data(), &fx.embed[(size_t) 42 * H], sizeof(double) * H);
            mrow = mask_row_img.data();
        } else {
            for (int i = 0; i < H; ++i) mask_row_img[i] = d16(mask_embed[i]);
            mrow = mask_row_img.data();
        }
        for (int r = 1; r < BLOCK; ++r) std::memcpy(&x[(size_t) r * H], mrow, sizeof(double) * H);

        std::vector<double> nx(H), q((size_t) BLOCK * QD), k((size_t) BLOCK * KVD),
            v((size_t) BLOCK * KVD), o((size_t) BLOCK * QD), y(H);
        std::vector<double> gate((size_t) BLOCK * INTER), up((size_t) BLOCK * INTER),
            act((size_t) BLOCK * INTER);
        for (int l = 0; l < L; ++l) {
            const Layer& ly = layers[l];
            double* rk_l = &rk[(size_t) l * RING * KVD];
            double* rv_l = &rv[(size_t) l * RING * KVD];
            for (int r = 0; r < BLOCK; ++r) {
                for (int i = 0; i < H; ++i) nx[i] = snap(x[(size_t) r * H + i]);
                rmsnorm(nx.data(), ly.ln1.data(), H);
                gemv(nx.data(), ly.q.data(), &q[(size_t) r * QD], H, QD);
                gemv(nx.data(), ly.k.data(), &k[(size_t) r * KVD], H, KVD);
                gemv(nx.data(), ly.v.data(), &v[(size_t) r * KVD], H, KVD);
                for (int e = 0; e < QD; ++e) q[(size_t) r * QD + e] = snap(q[(size_t) r * QD + e]);
                for (int e = 0; e < KVD; ++e) {
                    k[(size_t) r * KVD + e] = snap(k[(size_t) r * KVD + e]);
                    v[(size_t) r * KVD + e] = snap(v[(size_t) r * KVD + e]);
                }
                for (int h = 0; h < NQ; ++h) {
                    double* qh = &q[(size_t) r * QD + (size_t) h * HD];
                    if (!swap_norm_rope) { headnorm(qh, ly.qn.data()); rope(qh, P + r); }
                    else { rope(qh, P + r); headnorm(qh, ly.qn.data()); }
                }
                for (int h = 0; h < NKV; ++h) {
                    double* kh = &k[(size_t) r * KVD + (size_t) h * HD];
                    if (!swap_norm_rope) { headnorm(kh, ly.kn.data()); rope(kh, P + r); }
                    else { rope(kh, P + r); headnorm(kh, ly.kn.data()); }
                }
            }
            std::memcpy(&rk_l[(size_t) cache_pos * KVD], k.data(), sizeof(double) * BLOCK * KVD);
            std::memcpy(&rv_l[(size_t) cache_pos * KVD], v.data(), sizeof(double) * BLOCK * KVD);
            for (int r = 0; r < BLOCK; ++r) {
                const int64_t pr = cache_pos + r;
                const int lo = (int) std::max<int64_t>(0, pr - (WINDOW - 1));
                const int hi = (int) std::min<int64_t>(pr + RIGHT, cache_pos + BLOCK - 1);
                for (int h = 0; h < NQ; ++h) {
                    const int kvh = h / (NQ / NKV);
                    const double* qh = &q[(size_t) r * QD + (size_t) h * HD];
                    std::vector<double> s;
                    double m = -1e300;
                    for (int j = lo; j <= hi; ++j) {
                        const double* kh = &rk_l[(size_t) j * KVD + (size_t) kvh * HD];
                        double dot = 0.0;
                        for (int d = 0; d < HD; ++d) dot += qh[d] * kh[d];
                        dot *= SM;
                        s.push_back(dot);
                        m = std::max(m, dot);
                    }
                    const double sink = ly.sink[h];
                    m = std::max(m, sink);
                    double den = std::exp(sink - m);
                    std::vector<double> acc(HD, 0.0);
                    for (int j = lo; j <= hi; ++j) {
                        const double w = std::exp(s[(size_t) (j - lo)] - m);
                        den += w;
                        const double* vh = &rv_l[(size_t) j * KVD + (size_t) kvh * HD];
                        for (int d = 0; d < HD; ++d) acc[d] += w * vh[d];
                    }
                    for (int d = 0; d < HD; ++d)
                        o[(size_t) r * QD + (size_t) h * HD + d] = snap(acc[d] / den);
                }
            }
            for (int r = 0; r < BLOCK; ++r) {
                gemv(&o[(size_t) r * QD], ly.o_folded.data(), y.data(), QD, H);
                for (int i = 0; i < H; ++i) x[(size_t) r * H + i] += y[i];
            }
            for (int r = 0; r < BLOCK; ++r) {
                for (int i = 0; i < H; ++i) nx[i] = snap(x[(size_t) r * H + i]);
                rmsnorm(nx.data(), ly.ln2.data(), H);
                gemv(nx.data(), ly.gate.data(), &gate[(size_t) r * INTER], H, INTER);
                gemv(nx.data(), ly.up.data(), &up[(size_t) r * INTER], H, INTER);
                for (int i = 0; i < INTER; ++i) {   // gate/up stay wide; the PRODUCT rounds
                    const double g = gate[(size_t) r * INTER + i];
                    act[(size_t) r * INTER + i] = snap(g / (1.0 + std::exp(-g)) * up[(size_t) r * INTER + i]);
                }
                gemv(&act[(size_t) r * INTER], ly.down.data(), y.data(), INTER, H);
                for (int i = 0; i < H; ++i) x[(size_t) r * H + i] += y[i];
            }
        }
        std::vector<double> state((size_t) BLOCK * H);
        for (int r = 0; r < BLOCK; ++r) {
            for (int i = 0; i < H; ++i) nx[i] = snap(x[(size_t) r * H + i]);
            rmsnorm(nx.data(), final_norm.data(), H);
            std::memcpy(&state[(size_t) r * H], nx.data(), sizeof(double) * H);
        }
        if (ids) {
            ids->assign(BLOCK - 1, 0);
            logits->assign((size_t) (BLOCK - 1) * VOCAB_LM, 0.0);
            for (int r = 1; r < BLOCK; ++r) {
                int best = 0;
                double bestv = -1e300;
                for (int t = 0; t < VOCAB_LM; ++t) {
                    double dot = 0.0;
                    const double* wr = &fx.lm[(size_t) t * H];
                    const double* sr = &state[(size_t) r * H];
                    for (int i = 0; i < H; ++i) dot += sr[i] * wr[i];
                    (*logits)[(size_t) (r - 1) * VOCAB_LM + t] = dot;
                    if (dot > bestv) { bestv = dot; best = t; }
                }
                (*ids)[r - 1] = best;
            }
        }
        return state;
    }
};

// ------------------------------------------------------------------------------- hook side (target)

struct Hooks {
    const Fixture* fx;
    std::vector<uint16_t> state_stash;   // [n][H] F16, refreshed by every argmax call

    static void embed(int32_t token_id, uint16_t* out_row, void* user) {
        Hooks* h = (Hooks*) user;
        std::vector<uint16_t> row(H);
        for (int i = 0; i < H; ++i) row[i] = f16_from_f32((float) h->fx->embed[(size_t) token_id * H + i]);
        CU(cudaMemcpy(out_row, row.data(), H * 2, cudaMemcpyHostToDevice), "embed row");
    }

    static void argmax(const uint16_t* state_rows, int n, int32_t* out_ids, void* user) {
        Hooks* h = (Hooks*) user;
        h->state_stash.resize((size_t) n * H);
        CU(cudaMemcpy(h->state_stash.data(), state_rows, (size_t) n * H * 2, cudaMemcpyDeviceToHost),
           "state stash");
        std::vector<double> s((size_t) n * H);
        for (size_t i = 0; i < s.size(); ++i) s[i] = f32_from_f16(h->state_stash[i]);
        std::vector<int32_t> ids(n);
        for (int r = 0; r < n; ++r) {
            int best = 0;
            double bestv = -1e300;
            for (int t = 0; t < VOCAB_LM; ++t) {
                double dot = 0.0;
                const double* wr = &h->fx->lm[(size_t) t * H];
                const double* sr = &s[(size_t) r * H];
                for (int i = 0; i < H; ++i) dot += sr[i] * wr[i];
                if (dot > bestv) { bestv = dot; best = t; }
            }
            ids[r] = best;
        }
        CU(cudaMemcpy(out_ids, ids.data(), (size_t) n * 4, cudaMemcpyHostToDevice), "ids");
    }
};

// ------------------------------------------------------------------------- mimo_attn_block alone

uint16_t rand16(std::mt19937_64& rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    return f16_from_f32(u(rng));
}

// One mimo_attn_block case: fp64 parity of the base run, then (perturb_row >= 0) a second run
// with V[perturb_row] zeroed; the two outputs are handed back for the mask-semantics checks.
int run_block_kernel(const char* name, int rows, int64_t p0, int total, int capacity, int window,
                     int right, int n_q, int n_kv, uint64_t seed, int perturb_row,
                     std::vector<float>* out_a, std::vector<float>* out_b) {
    std::mt19937_64 rng(seed);
    const int g = n_q / n_kv;
    std::vector<uint16_t> q((size_t) rows * n_q * HD), k((size_t) capacity * n_kv * HD),
        v((size_t) capacity * n_kv * HD);
    for (auto& x : q) x = rand16(rng);
    for (auto& x : k) x = rand16(rng);
    for (auto& x : v) x = rand16(rng);
    std::vector<float> sinks(n_q);
    for (int h = 0; h < n_q; ++h)
        sinks[h] = (h % 2) == 0 ? -1.0f + 2.0f * (float) std::generate_canonical<double, 10>(rng)
                                : 6.0f + 6.0f * (float) std::generate_canonical<double, 10>(rng);
    std::vector<int64_t> pos(rows);
    std::vector<int> len(rows, total);
    for (int r = 0; r < rows; ++r) pos[r] = p0 + r;

    std::vector<double> ref((size_t) rows * n_q * HD, 0.0);
    for (int r = 0; r < rows; ++r) {
        const int lo = (int) std::max<int64_t>(0, pos[r] - (window - 1));
        const int hi = (int) std::min<int64_t>(pos[r] + right, total - 1);
        for (int h = 0; h < n_q; ++h) {
            const int kvh = h / g;
            std::vector<double> s;
            double m = -1e300;
            for (int j = lo; j <= hi; ++j) {
                double dot = 0.0;
                for (int d = 0; d < HD; ++d)
                    dot += (double) f32_from_f16(q[((size_t) r * n_q + h) * HD + d]) *
                           (double) f32_from_f16(k[((size_t) j * n_kv + kvh) * HD + d]);
                dot *= SM;
                s.push_back(dot);
                m = std::max(m, dot);
            }
            m = std::max(m, (double) sinks[h]);
            double den = std::exp((double) sinks[h] - m);
            for (int j = lo; j <= hi; ++j) {
                const double w = std::exp(s[(size_t) (j - lo)] - m);
                den += w;
                for (int d = 0; d < HD; ++d)
                    ref[((size_t) r * n_q + h) * HD + d] +=
                        w * (double) f32_from_f16(v[((size_t) j * n_kv + kvh) * HD + d]);
            }
            for (int d = 0; d < HD; ++d) ref[((size_t) r * n_q + h) * HD + d] /= den;
        }
    }

    uint16_t *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    float* d_out = nullptr;
    int64_t* d_pos = nullptr;
    int* d_len = nullptr;
    float* d_sinks = nullptr;
    CHK(cudaMalloc(&d_q, q.size() * 2), "q");
    CHK(cudaMalloc(&d_k, k.size() * 2), "k");
    CHK(cudaMalloc(&d_v, v.size() * 2), "v");
    CHK(cudaMalloc(&d_out, (size_t) rows * n_q * HD * 4), "out");
    CHK(cudaMalloc(&d_pos, pos.size() * 8), "pos");
    CHK(cudaMalloc(&d_len, len.size() * 4), "len");
    CHK(cudaMalloc(&d_sinks, sinks.size() * 4), "sinks");
    CHK(cudaMemcpy(d_q, q.data(), q.size() * 2, cudaMemcpyHostToDevice), "cq");
    CHK(cudaMemcpy(d_k, k.data(), k.size() * 2, cudaMemcpyHostToDevice), "ck");
    CHK(cudaMemcpy(d_v, v.data(), v.size() * 2, cudaMemcpyHostToDevice), "cv");
    CHK(cudaMemcpy(d_pos, pos.data(), pos.size() * 8, cudaMemcpyHostToDevice), "cpos");
    CHK(cudaMemcpy(d_len, len.data(), len.size() * 4, cudaMemcpyHostToDevice), "clen");
    CHK(cudaMemcpy(d_sinks, sinks.data(), sinks.size() * 4, cudaMemcpyHostToDevice), "csinks");

    const auto run = [&](std::vector<float>& out) {
        strata::kernels::mimo_attn_block(d_q, d_k, d_v, d_out, d_pos, d_len, rows, n_q, n_kv, HD,
                                         HD, capacity, window, right, d_sinks, SM, nullptr);
        out.resize((size_t) rows * n_q * HD);
        CU(cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost), "attn out");
    };
    std::vector<float> a, b;
    run(a);
    if (out_a) *out_a = a;
    if (perturb_row >= 0) {
        const std::vector<uint16_t> zero((size_t) n_kv * HD, 0);
        CHK(cudaMemcpy(d_v + (size_t) perturb_row * n_kv * HD, zero.data(), (size_t) n_kv * HD * 2,
                       cudaMemcpyHostToDevice), "perturb");
        run(b);
        if (out_b) *out_b = b;
    }
    const std::vector<double> ad(a.begin(), a.end());
    const Compare c = compare(ad, ref);
    const bool ok = c.ok();
    std::printf("%-46s max_rel=%.3e norm_rel=%.3e nonfinite=%zu%s\n", name, c.max_rel, c.norm_rel,
                c.bad, ok ? "" : "  (FAIL)");
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    cudaFree(d_pos); cudaFree(d_len); cudaFree(d_sinks);
    return ok ? 0 : 1;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b, int row, int width) {
    double m = 0.0;
    for (int i = row * width; i < (row + 1) * width; ++i)
        m = std::max(m, std::fabs((double) a[(size_t) i] - (double) b[(size_t) i]));
    return m;
}


// ---- one full drafter round trip against the fp64 reference; BOTH dir shapes run this ----
//
// The dir must already be written (bf16 or quantized); the Fixture carries the reference's
// weights for whichever shape it is.  Returns the fail count, or 100 on a catastrophic failure
// (a load that did not even complete - the caller stops the whole test there).
int drafter_flow(const Fixture& fx, const std::string& dir, const char* tag) {
    int fails = 0;
    const std::string pfx = std::string("[") + tag + "] ";
    std::string err;
    const strata::mimo::DflashConfig cfg = strata::mimo::from_dir(dir, &err);
    if (!err.empty()) {
        std::printf("%sfrom_dir: %s (FAIL)\n", pfx.c_str(), err.c_str());
        return 100;
    }
    if (cfg.n_layers != L || cfg.hidden != H || cfg.inter != INTER || cfg.n_q_head != NQ ||
        cfg.n_kv_head != NKV || cfg.head_dim_qk != HD || cfg.head_dim_v != HD ||
        cfg.n_rot != ROT || cfg.block != BLOCK || cfg.window != WINDOW ||
        cfg.window_right != RIGHT || cfg.causal || cfg.mask_token != 42 ||
        cfg.value_scale != VSCALE || !cfg.has_sinks || cfg.target_hidden != TH ||
        cfg.taps.size() != TAPS || cfg.taps[0] != 2 || cfg.taps[1] != 7 || cfg.taps[2] != 11 ||
        cfg.page != PAGE || cfg.ring_size() != RING) {
        std::printf("%sfrom_dir: parsed geometry does not match the fixture (FAIL)\n", pfx.c_str());
        return 100;
    }
    std::printf("%sfrom_dir: ring %d (page %d), taps {%d,%d,%d}, scale %g\n", pfx.c_str(),
                cfg.ring_size(), cfg.page, cfg.taps[0], cfg.taps[1], cfg.taps[2], cfg.value_scale);

    strata::mimo::Dflash dflash;
    if (!strata::mimo::dflash_load(&dflash, dir, &err)) {
        std::printf("%sdflash_load: %s (FAIL)\n", pfx.c_str(), err.c_str());
        return 100;
    }
    if (dflash.quantized != fx.quant) {
        std::printf("%sdflash_load: quantized=%d but the fixture says %d (FAIL)\n", pfx.c_str(),
                    (int) dflash.quantized, (int) fx.quant);
        strata::mimo::dflash_unload(&dflash);
        return 100;
    }
    Hooks hooks{&fx, {}};
    strata::mimo::DflashDrafter dr;
    if (!dr.init(dflash, Hooks::embed, Hooks::argmax, &hooks, 512)) {
        std::printf("%sdrafter init failed (FAIL)\n", pfx.c_str());
        return 100;
    }
    RefModel ref(fx);
    int32_t* d_ids = nullptr;
    CHK(cudaMalloc(&d_ids, (BLOCK - 1) * 4), "ids alloc");

    // taps: snapped to F16 values so the GPU's F16 input is bit-identical to the reference's
    std::mt19937_64 rng(7777);
    const auto make_taps = [&](int rows, std::vector<double>& flat, std::vector<uint16_t*>& ptrs) {
        flat.assign((size_t) TAPS * rows * TH, 0.0);
        ptrs.assign(TAPS, nullptr);
        std::normal_distribution<double> nd(0.0, 1.0);
        std::vector<uint16_t> f16buf((size_t) rows * TH);
        for (int t = 0; t < TAPS; ++t) {
            for (int i = 0; i < rows * TH; ++i) {
                const double v = (double) f32_from_f16(f16_from_f32((float) nd(rng)));
                flat[((size_t) t * rows) * TH + i] = v;
                f16buf[i] = f16_from_f32((float) v);
            }
            CU(cudaMalloc(&ptrs[t], (size_t) rows * TH * 2), "taps alloc");
            CU(cudaMemcpy(ptrs[t], f16buf.data(), (size_t) rows * TH * 2, cudaMemcpyHostToDevice),
               "taps upload");
        }
    };

    const auto check_ring = [&](const char* name, int64_t first_row, int rows, double tol = TOL) {
        Compare worst;
        int bad = 0;
        for (int l = 0; l < L; ++l) {
            std::vector<uint16_t> gk((size_t) rows * KVD), gv((size_t) rows * KVD);
            CU(cudaMemcpy(gk.data(), dr.ring_k_layer(l) + (size_t) first_row * KVD,
                          (size_t) rows * KVD * 2, cudaMemcpyDeviceToHost), "ring k read");
            CU(cudaMemcpy(gv.data(), dr.ring_v_layer(l) + (size_t) first_row * KVD,
                          (size_t) rows * KVD * 2, cudaMemcpyDeviceToHost), "ring v read");
            std::vector<double> gpu, want;
            gpu.reserve((size_t) 2 * rows * KVD);
            want.reserve((size_t) 2 * rows * KVD);
            for (uint16_t x : gk) gpu.push_back(f32_from_f16(x));
            for (uint16_t x : gv) gpu.push_back(f32_from_f16(x));
            want.insert(want.end(), &ref.rk[((size_t) l * RING + first_row) * KVD],
                        &ref.rk[((size_t) l * RING + first_row + rows) * KVD]);
            want.insert(want.end(), &ref.rv[((size_t) l * RING + first_row) * KVD],
                        &ref.rv[((size_t) l * RING + first_row + rows) * KVD]);
            const Compare c = compare(gpu, want);
            if (c.max_rel > worst.max_rel) {
                worst = c;
                if (!c.ok(tol)) {   // worst-element diagnostics, same intent as check_state's
                    size_t wi = 0;
                    double w = -1.0, ref_max = 0.0;
                    for (double v : want) ref_max = std::max(ref_max, std::fabs(v));
                    for (size_t i = 0; i < want.size(); ++i) {
                        const double rel = std::fabs(gpu[i] - want[i]) /
                                           std::max(std::fabs(want[i]), 0.05 * ref_max + 1e-12);
                        if (rel > w) { w = rel; wi = i; }
                    }
                    const char* kv = wi < (size_t) rows * KVD ? "k" : "v";
                    const size_t off = wi < (size_t) rows * KVD ? wi : wi - (size_t) rows * KVD;
                    std::printf("  (L%d %s row %zu lane %zu ref=%.6f gpu=%.6f)\n", l, kv,
                                off / KVD, off % KVD, want[wi], gpu[wi]);
                }
            }
            if (!c.ok(tol)) ++bad;
        }
        std::printf("%-46s max_rel=%.3e norm_rel=%.3e%s\n", name, worst.max_rel, worst.norm_rel,
                    bad ? "  (FAIL)" : "");
        return bad ? 1 : 0;
    };

    const auto gpu_state_rows = [&]() {
        std::vector<double> gpu((size_t) (BLOCK - 1) * H);
        for (size_t i = 0; i < gpu.size(); ++i) gpu[i] = f32_from_f16(hooks.state_stash[i]);
        return gpu;
    };
    // ids must match; a pick whose fp64 logit sits within 1e-3 relative of the top is a near-tie
    const auto check_ids = [&](const char* what, const std::vector<int>& ref_ids,
                               const std::vector<double>& ref_logits) {
        int32_t ids[BLOCK - 1];
        CU(cudaMemcpy(ids, d_ids, (BLOCK - 1) * 4, cudaMemcpyDeviceToHost), "ids read");
        bool ok = true;
        for (int r = 0; r < BLOCK - 1; ++r) {
            if (ids[r] == ref_ids[r]) continue;
            const double top = ref_logits[(size_t) r * VOCAB_LM + ref_ids[r]];
            const double got = ref_logits[(size_t) r * VOCAB_LM + ids[r]];
            if (got < top - 1e-3 * (std::fabs(top) + 1.0)) ok = false;
            std::printf("  (%s row %d: gpu pick %d vs ref %d, logits %.4f vs %.4f%s)\n", what, r,
                        ids[r], ref_ids[r], got, top,
                        got >= top - 1e-3 * (std::fabs(top) + 1.0) ? " near-tie" : "");
        }
        return ok;
    };
    const auto check_state = [&](const char* name, const std::vector<double>& ref_state,
                                 bool ids_ok, double tol = TOL) {
        const std::vector<double> gpu = gpu_state_rows();
        const std::vector<double> want(ref_state.begin() + H, ref_state.end());
        const Compare c = compare(gpu, want);
        if (!c.ok(tol)) {   // worst element diagnostics: a real divergence is actionable, a
            size_t wi = 0;   // near-zero ref element is gate noise
            double worst = -1.0;
            double ref_max = 0.0;
            for (double v : want) ref_max = std::max(ref_max, std::fabs(v));
            for (size_t i = 0; i < want.size(); ++i) {
                const double rel = std::fabs(gpu[i] - want[i]) /
                                   std::max(std::fabs(want[i]), 0.05 * ref_max + 1e-12);
                if (rel > worst) { worst = rel; wi = i; }
            }
            std::printf("  (worst: row %zu lane %zu ref=%.6f gpu=%.6f abs=%.2e)\n",
                        wi / H, wi % H, want[wi], gpu[wi], std::fabs(gpu[wi] - want[wi]));
        }
        const bool ok = c.ok(tol) && ids_ok;
        std::printf("%-52s max_rel=%.3e norm_rel=%.3e ids=%s%s\n", name, c.max_rel, c.norm_rel,
                    ids_ok ? "match" : "MISMATCH", ok ? "" : "  (FAIL)");
        return ok ? 0 : 1;
    };

    // 1. commit 300 rows at 0 (a prefill-shaped commit)
    {
        std::vector<double> flat;
        std::vector<uint16_t*> ptrs;
        make_taps(300, flat, ptrs);
        if (!dr.commit_from_taps(ptrs.data(), 0, 300)) { std::printf("%scommit 1 failed\n", pfx.c_str()); return 100; }
        ref.commit(flat, 0, 300);
        fails += check_ring((pfx + "commit 300 rows @0 ring k/v").c_str(), 0, 300);
        for (auto p : ptrs) CU(cudaFree(p), "taps free");
    }

    // 2. draft at P=300 (ring rows 300..307 become speculative).  The same round is replayed in
    //    the reference with the WRONG norm/rope order and the WRONG mask input for the order and
    //    mask checks - both variants rewrite the same speculative rows, which everything after
    //    this point overwrites, so the ring state stays canonical for the later rounds
    std::vector<double> gpu_round2, state_correct, state_swapped, state_masked;
    std::vector<int> ref_ids;
    {
        state_swapped = ref.draft(123, 300, nullptr, nullptr, /*swap_norm_rope=*/true);
        state_masked = ref.draft(123, 300, nullptr, nullptr, /*swap_norm_rope=*/false,
                                 /*embed_mask_with_target=*/true);
        std::vector<double> ref_logits;
        state_correct = ref.draft(123, 300, &ref_ids, &ref_logits);
        if (!dr.draft(123, 300, d_ids)) { std::printf("%sdraft 1 failed\n", pfx.c_str()); return 100; }
        gpu_round2 = gpu_state_rows();
        const bool ids_ok = check_ids("draft @300", ref_ids, ref_logits);
        fails += check_state((pfx + "draft @300 state rows 1..7").c_str(), state_correct, ids_ok,
                             STACK_TOL);
        fails += check_ring((pfx + "draft @300 speculative ring rows").c_str(), 300, BLOCK,
                            STACK_TOL);

        const Compare csw = compare(gpu_round2,
                                    std::vector<double>(state_swapped.begin() + H,
                                                        state_swapped.end()));
        const Compare cmk = compare(gpu_round2,
                                    std::vector<double>(state_masked.begin() + H,
                                                        state_masked.end()));
        const Compare cok = compare(gpu_round2,
                                    std::vector<double>(state_correct.begin() + H,
                                                        state_correct.end()));
        const bool ok_order = cok.ok(STACK_TOL) && csw.max_rel > 5 * cok.max_rel;
        const bool ok_mask = cok.ok(STACK_TOL) && cmk.max_rel > 5 * cok.max_rel;
        std::printf("%-52s correct=%.3e swapped=%.3e%s\n", (pfx + "q/k_norm BEFORE rope pinned").c_str(),
                    cok.max_rel, csw.max_rel, ok_order ? "" : "  (FAIL)");
        std::printf("%-52s correct=%.3e wrong-input=%.3e%s\n", (pfx + "mask_embedding overwrite used").c_str(),
                    cok.max_rel, cmk.max_rel, ok_mask ? "" : "  (FAIL)");
        fails += ok_order ? 0 : 1;
        fails += ok_mask ? 0 : 1;
    }

    // 3. commit 3 rows at 300, overwriting the speculative block: the committed rows must equal
    //    the taps-derived values and sit far from the stale speculative ones
    {
        std::vector<double> flat, stale;
        std::vector<uint16_t*> ptrs;
        make_taps(3, flat, ptrs);
        ref.commit(flat, 300, 3, &stale);
        if (!dr.commit_from_taps(ptrs.data(), 300, 3)) { std::printf("%scommit 2 failed\n", pfx.c_str()); return 100; }
        fails += check_ring((pfx + "commit 3 rows @300 (spec. erased)").c_str(), 300, 3);
        std::vector<uint16_t> gk((size_t) 3 * KVD), gv((size_t) 3 * KVD);
        CU(cudaMemcpy(gk.data(), dr.ring_k_layer(0) + (size_t) 300 * KVD, (size_t) 3 * KVD * 2,
                      cudaMemcpyDeviceToHost), "stale k read");
        CU(cudaMemcpy(gv.data(), dr.ring_v_layer(0) + (size_t) 300 * KVD, (size_t) 3 * KVD * 2,
                      cudaMemcpyDeviceToHost), "stale v read");
        std::vector<double> gpu;
        for (uint16_t x : gk) gpu.push_back(f32_from_f16(x));
        for (uint16_t x : gv) gpu.push_back(f32_from_f16(x));
        double stale_gap = 0.0;
        for (size_t i = 0; i < gpu.size(); ++i)
            stale_gap = std::max(stale_gap, std::fabs(gpu[i] - stale[i]));
        const bool teeth = stale_gap > 1e-2;
        std::printf("%-52s stale-vs-committed gap=%.3e%s\n", (pfx + "speculative rows overwritten").c_str(),
                    stale_gap, teeth ? "" : "  (FAIL)");
        fails += teeth ? 0 : 1;
        for (auto p : ptrs) CU(cudaFree(p), "taps free");
    }

    // 4. draft at P=303 (its attention must read the taps-derived rows 300..302)
    {
        std::vector<double> ref_logits;
        const std::vector<double> st = ref.draft(77, 303, &ref_ids, &ref_logits);
        if (!dr.draft(77, 303, d_ids)) { std::printf("%sdraft 2 failed\n", pfx.c_str()); return 100; }
        const bool ids_ok = check_ids("draft @303", ref_ids, ref_logits);
        fails += check_state((pfx + "draft @303 state rows 1..7").c_str(), st, ids_ok, STACK_TOL);
    }

    // 5. commit 303 rows at 303 -> position 606: cache_pos 303 + 303 > 384, so the ring drops
    //    256 rows from the front (page 64)
    {
        std::vector<double> flat;
        std::vector<uint16_t*> ptrs;
        make_taps(303, flat, ptrs);
        if (!dr.commit_from_taps(ptrs.data(), 303, 303)) { std::printf("%scommit 3 failed\n", pfx.c_str()); return 100; }
        ref.commit(flat, 303, 303);
        if (dr.committed_pos() != 606 || ref.committed != 606 || dr.window_beg != ref.window_beg ||
            dr.window_beg != 256) {
            std::printf("%scommit 3 bookkeeping: gpu %lld/beg %lld ref %lld/beg %lld (FAIL)\n",
                        pfx.c_str(), (long long) dr.committed_pos(), (long long) dr.window_beg,
                        (long long) ref.committed, (long long) ref.window_beg);
            fails += 1;
        }
        fails += check_ring((pfx + "commit 303 rows @303 ring k/v (shift 256)").c_str(), 0, 350);
        for (auto p : ptrs) CU(cudaFree(p), "taps free");
    }

    // 6. draft at P=606 > ring 384 (wraparound: the window lives at ring rows 350..357)
    {
        std::vector<double> ref_logits;
        const std::vector<double> st = ref.draft(55, 606, &ref_ids, &ref_logits);
        if (!dr.draft(55, 606, d_ids)) { std::printf("%sdraft 3 failed\n", pfx.c_str()); return 100; }
        const bool ids_ok = check_ids("draft @606", ref_ids, ref_logits);
        fails += check_state((pfx + "draft @606 (P > ring) state 1..7").c_str(), st, ids_ok,
                             STACK_TOL);
    }

    CU(cudaFree(d_ids), "ids free");
    dr.free();
    strata::mimo::dflash_unload(&dflash);
    return fails;
}

// ---- the REAL dirs side by side: quantized dflash/ (EXL3) against dflash-bf16/
//
// The two variants carry the SAME raw tensors (norms, mask_embedding) and the same linears, one
// quantized.  So: mask_embedding and the norms must upload BIT-IDENTICAL, the sinks F16-close
// (the quantized dir stores them F16, the bf16 dir BF16), and a reconstructed o_proj row must
// sit near the bf16 dir's same row scaled by 0.612 - the cosine checks the reconstruction, the
// NORM RATIO checks the fold was applied exactly once (twice: ~0.61x off; never: ~1.63x off).
int real_dir_smoke(const std::string& root) {
    namespace fs = std::filesystem;
    int fails = 0;
    const std::string pfx = "[real] ";
    if (!fs::is_directory(root + "/dflash") || !fs::is_directory(root + "/dflash-bf16")) {
        std::printf("%sSKIPPED: %s has no dflash/ + dflash-bf16/ pair\n", pfx.c_str(), root.c_str());
        return 0;
    }
    std::string err;
    strata::mimo::Dflash dq, db;
    const auto t0 = std::chrono::steady_clock::now();
    if (!strata::mimo::dflash_load(&dq, root + "/dflash", &err)) {
        std::printf("%sdflash/ (EXL3): %s (FAIL)\n", pfx.c_str(), err.c_str());
        return 1;
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (!strata::mimo::dflash_load(&db, root + "/dflash-bf16", &err)) {
        std::printf("%sdflash-bf16/: %s (FAIL)\n", pfx.c_str(), err.c_str());
        return 1;
    }
    const auto t2 = std::chrono::steady_clock::now();
    std::printf("%sEXL3 load+reconstruct %.1fs, bf16 load %.1fs (quantized=%d/%d)\n", pfx.c_str(),
                std::chrono::duration<double>(t1 - t0).count(),
                std::chrono::duration<double>(t2 - t1).count(), (int) dq.quantized,
                (int) db.quantized);
    if (!dq.quantized || db.quantized) {
        std::printf("%sdir-shape detection wrong (FAIL)\n", pfx.c_str());
        return 1;
    }
    const int H2 = dq.cfg.hidden;   // the real geometry

    const auto eq_bits = [&](const uint16_t* a, const uint16_t* b, int64_t n) {
        std::vector<uint16_t> va((size_t) n), vb((size_t) n);
        CU(cudaMemcpy(va.data(), a, (size_t) n * 2, cudaMemcpyDeviceToHost), "a read");
        CU(cudaMemcpy(vb.data(), b, (size_t) n * 2, cudaMemcpyDeviceToHost), "b read");
        return va == vb;
    };
    bool ok = eq_bits(dq.mask_embed, db.mask_embed, H2) &&
              eq_bits(dq.hidden_norm, db.hidden_norm, H2) &&
              eq_bits(dq.final_norm, db.final_norm, H2) &&
              eq_bits(dq.layers[0].ln1, db.layers[0].ln1, H2) &&
              eq_bits(dq.layers[0].q_norm, db.layers[0].q_norm, dq.cfg.head_dim_qk);
    std::printf("%-52s %s%s\n", (pfx + "raw tensors (mask/norms/q_norm) identical").c_str(),
                ok ? "yes" : "NO", ok ? "" : "  (FAIL)");
    fails += ok ? 0 : 1;

    {   // sinks: F16 vs BF16 storage of the same values
        std::vector<float> sa(dq.cfg.n_q_head), sb(dq.cfg.n_q_head);
        CU(cudaMemcpy(sa.data(), dq.layers[0].sink, sa.size() * 4, cudaMemcpyDeviceToHost), "sink q");
        CU(cudaMemcpy(sb.data(), db.layers[0].sink, sb.size() * 4, cudaMemcpyDeviceToHost), "sink b");
        double worst = 0.0;
        for (size_t i = 0; i < sa.size(); ++i) worst = std::max(worst, (double) std::fabs(sa[i] - sb[i]));
        const bool sok = worst < 1e-2;
        std::printf("%-52s max_abs=%.3e%s\n", (pfx + "sinks F16 vs BF16 (L0)").c_str(), worst,
                    sok ? "" : "  (FAIL)");
        fails += sok ? 0 : 1;
    }

    // o_proj row 0 layer 0: the reconstruction (which the loader does NOT fold - the trellis
    // is pre-folded by exllamav3's converter) vs the bf16 dir's raw row x 0.612: the cosine
    // checks the reconstruction, the norm ratio checks the fold sits on exactly one side
    // (0.612x off would mean the loader folded what was already folded)
    {
        const strata::SafetensorsFile bf(root + "/dflash-bf16/dflash_draft_model.safetensors");
        const strata::StTensor* t = bf.find("layers.0.self_attn.o_proj.weight");
        if (!t || t->dtype != "BF16" || t->shape.size() != 2) {
            std::printf("%sbf16 o_proj tensor missing (FAIL)\n", pfx.c_str());
            return ++fails;
        }
        const int64_t row = 0, n = t->shape[1];
        const uint16_t* raw = reinterpret_cast<const uint16_t*>(bf.data(*t)) + (size_t) row * n;
        std::vector<uint16_t> gpu((size_t) n);
        CU(cudaMemcpy(gpu.data(), dq.layers[0].o_w, (size_t) n * 2, cudaMemcpyDeviceToHost),
           "o row read");
        double dot = 0, nq = 0, nb = 0;
        for (int64_t i = 0; i < n; ++i) {
            const double g = f32_from_f16(gpu[(size_t) i]);
            const double b = (double) f32_from_bf16(raw[(size_t) i]) * dq.cfg.value_scale;
            dot += g * b;
            nq += g * g;
            nb += b * b;
        }
        const double cos = dot / (std::sqrt(nq) * std::sqrt(nb) + 1e-30);
        const double ratio = std::sqrt(nq) / (std::sqrt(nb) + 1e-30);
        const bool ook = cos > 0.99 && ratio > 0.95 && ratio < 1.05;
        std::printf("%-52s cos=%.4f norm-ratio=%.3f%s\n",
                    (pfx + "o_proj L0 row0: EXL3 vs bf16 x0.612").c_str(), cos, ratio,
                    ook ? "" : "  (FAIL)");
        fails += ook ? 0 : 1;
        // a fold-free sanity on a linear that is NOT folded
        std::vector<uint16_t> gq((size_t) n);
        const strata::StTensor* tq = bf.find("layers.0.self_attn.q_proj.weight");
        const int64_t nq_in = tq->shape[1];
        CU(cudaMemcpy(gq.data(), dq.layers[0].q_w, (size_t) nq_in * 2, cudaMemcpyDeviceToHost),
           "q row read");
        const uint16_t* rawq =
            reinterpret_cast<const uint16_t*>(bf.data(*tq)) + (size_t) 0 * nq_in;
        double dot2 = 0, na = 0, nc = 0;
        for (int64_t i = 0; i < nq_in; ++i) {
            const double g = f32_from_f16(gq[(size_t) i]);
            const double b = (double) f32_from_bf16(rawq[(size_t) i]);
            dot2 += g * b;
            na += g * g;
            nc += b * b;
        }
        const double cosq = dot2 / (std::sqrt(na) * std::sqrt(nc) + 1e-30);
        const double ratioq = std::sqrt(na) / (std::sqrt(nc) + 1e-30);
        const bool qok = cosq > 0.99 && ratioq > 0.95 && ratioq < 1.05;
        std::printf("%-52s cos=%.4f norm-ratio=%.3f%s\n",
                    (pfx + "q_proj L0 row0: EXL3 vs bf16 (no fold)").c_str(), cosq, ratioq,
                    qok ? "" : "  (FAIL)");
        fails += qok ? 0 : 1;
    }
    strata::mimo::dflash_unload(&dq);
    strata::mimo::dflash_unload(&db);
    return fails;
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA/HIP device");
        return 77;
    }
    int fails = 0;

    // ---- mimo_attn_block on its own: parity (GQA 1 and 8) with sinks half-dominating the max,
    //       bidirectionality, the right extent honored, the left window clamp
    {
        fails += run_block_kernel("block g1 w256 r255 sinks parity", 8, 100, 108, 200, 256, 255,
                                  NQ, NKV, 101, -1, nullptr, nullptr);
        std::vector<float> a, b;
        run_block_kernel("block g1 (bidir setup)", 8, 100, 108, 200, 256, 255, NQ, NKV, 101, 107,
                         &a, &b);
        const double d0 = max_abs_diff(a, b, 0, NQ * HD);
        const bool ok = d0 > 1e-4 && std::isfinite(d0);   // noise ~4e-6; causal gives exactly 0
        std::printf("%-46s row0 delta=%.3e%s\n", "block g1 bidirectional (v[107] moves row 0)", d0,
                    ok ? "" : "  (FAIL)");
        fails += ok ? 0 : 1;

        run_block_kernel("block g1 right=3 (setup)", 8, 100, 108, 200, 256, 3, NQ, NKV, 103, 104,
                         &a, &b);
        const double dn = max_abs_diff(a, b, 0, NQ * HD), d4 = max_abs_diff(a, b, 4, NQ * HD);
        const bool okr = dn < 1e-5 && d4 > 1e-2;
        std::printf("%-46s row0 delta=%.3e row4 delta=%.3e%s\n", "block g1 right extent 3 (v[104])",
                    dn, d4, okr ? "" : "  (FAIL)");
        fails += okr ? 0 : 1;

        fails += run_block_kernel("block g8 w1024 r1023 sinks parity", 8, 2000, 2008, 2100, 1024,
                                  1023, 64, 8, 104, -1, nullptr, nullptr);
        std::vector<float> c, d;
        run_block_kernel("block g8 (left clamp setup)", 8, 2000, 2008, 2100, 1024, 1023, 64, 8,
                         104, 900, &c, &d);
        double dmax = 0.0;
        for (int r = 0; r < 8; ++r) dmax = std::max(dmax, max_abs_diff(c, d, r, 64 * HD));
        const bool okl = dmax < 1e-5;
        std::printf("%-46s max row delta=%.3e%s\n", "block g8 left clamp (v[900] invisible)", dmax,
                    okl ? "" : "  (FAIL)");
        fails += okl ? 0 : 1;
        run_block_kernel("block g8 (bidir setup)", 8, 2000, 2008, 2100, 1024, 1023, 64, 8, 104,
                         2007, &c, &d);
        const double dg8 = max_abs_diff(c, d, 0, 64 * HD);
        const bool okg = dg8 > 1e-4;   // noise ~9e-6; causal gives exactly 0
        std::printf("%-46s row0 delta=%.3e%s\n", "block g8 bidirectional (v[2007] moves row 0)",
                    dg8, okg ? "" : "  (FAIL)");
        fails += okg ? 0 : 1;
    }

    // ---- the drafter flow, once per dir shape, against the fp64 reference
    {
        const Fixture fx = make_fixture(20261027u);
        const std::string dir = "dflash_parity_fixture";
        write_dir(dir, fx);
        const int f = drafter_flow(fx, dir, "bf16");
        std::filesystem::remove_all(dir);
        if (f >= 100) { std::printf("FAIL\n"); return 1; }
        fails += f;
    }
    {
        Fixture fxq = make_fixture(314159u);
        const std::string dir = "dflash_parity_fixture_q";
        write_dir_quantized(dir, fxq, 271828u);
        const int f = drafter_flow(fxq, dir, "exl3");
        std::filesystem::remove_all(dir);
        if (f >= 100) { std::printf("FAIL\n"); return 1; }
        fails += f;
    }

    // ---- optional real-model check, only when explicitly selected
    {
        const char* env = std::getenv("STRATA_DFLASH_REAL_DIR");
        if (env && *env) fails += real_dir_smoke(env);
        else std::puts("SKIP: real-model DFlash check (set STRATA_DFLASH_REAL_DIR)");
    }

    std::printf("%s\n", fails ? "FAIL" : "OK");
    return fails ? 1 : 0;
}
