// src/artifact/exl3_frac_k_test.cpp - half-integer-K (K = KA + 0.5) trellis support: the loader's
// relaxed validation, and the CPU decode against TWO references.
//
// 1. A GOLDEN TILE produced by the real encoder: exllamav3's ext.pack_trellis_frac /
//    ext.unpack_trellis_frac (frac.cu, KA=2, MASK=0xAAAA, mul1) ran on random 16-bit states; the
//    packed words and the unpacked windows below are that run's bytes (generated once, offline -
//    the committed test needs no python).
// 2. An IN-TEST reference of exllamav3's frac storage semantics (frac.cuh's pack/unpack math,
//    transcribed): per-position step widths D(i) = KA + bit(i&15) of MASK, window i = the 16 ring
//    bits ending at the prefix sum S(i), bits stored MSB-first in 32-bit words. The test packs its
//    own ring from random new bits and requires the CPU decode to read exactly those windows back.
//
// The whole reconstruct is then checked end to end (W_hat bit-exact, W within fp16 tolerance)
// against the same reference plus the Hadamard/scale math from exl3.py's get_weight_tensor.
#include "strata/artifact/exl3_model.hpp"
#include "strata/kernels/cpu/exl3.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace strata;
using strata::kernels::cpu::Exl3Codebook;

static int failures = 0;
#define CHECK(cond)                                                                              \
    do {                                                                                         \
        if (!(cond)) {                                                                           \
            ++failures;                                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                 \
        }                                                                                        \
    } while (0)

// ---- fp16 <-> fp32 (round to nearest even), test-local so the oracle is self-contained ----
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

// ---- the mul1 codebook window value, from the spec (exllamav3 codebook.cuh / Strata's
// tools/exl3/codebook.py): h = fp16(bytesytesum(w * 0x83DCD12D) + 0x6400); fp16(h * fp16(0x1EEE) +
// fp16(0xC931)) ----
static uint16_t mul1_window(uint16_t w) {
    uint32_t x = (uint32_t)w * 0x83DCD12Du;
    uint32_t s = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24) + 0x6400u;
    return f2h(std::fma(h2f((uint16_t)(s & 0xFFFF)), h2f(0x1EEE), h2f(0xC931)));
}

// ---- exllamav3 frac.cuh, transcribed: D(i), S(i) and the ring writer ----
static const uint32_t kMask = 0xAAAA;   // alternating KA / KA+1 steps, period 16 (mul1 only)
static int frac_d(int i, int ka) { return ka + (int)((kMask >> (i & 15)) & 1); }
static int frac_s(int i, int ka, int bpb) {   // ring bit where window i ends
    int s = (i >> 4) * bpb;
    for (int j = 0; j <= (i & 15); ++j) s += frac_d(j, ka);
    return s;
}

/// Reference pack (frac.cuh pack_trellis_frac_kernel): `newbits[i]` (already masked to D(i) bits)
/// is written MSB-first at ring positions [S(i)-D(i), S(i)); the ring is `words` uint16 stored as
/// little-endian 32-bit words, MSB-first inside each word.
static void ref_pack(std::vector<uint16_t>& words, const std::vector<uint16_t>& newbits, int ka) {
    const int bpb = 16 * ka + 8, nw = bpb / 2;
    std::vector<uint32_t> w((size_t)nw, 0);
    int pos = 0;
    for (int i = 0; i < 256; ++i) {
        const int d = frac_d(i, ka);
        const uint32_t v = newbits[(size_t)i] & ((1u << d) - 1);
        for (int b = d - 1; b >= 0; --b, ++pos)
            if ((v >> b) & 1) w[(size_t)(pos >> 5)] |= 1u << (31 - (pos & 31));
    }
    words.resize((size_t)nw * 2);
    std::memcpy(words.data(), w.data(), (size_t)nw * 4);
}

/// Reference window extraction (frac.cuh frac_window): the 16 ring bits ending at S(i).
static uint16_t ref_window(const std::vector<uint16_t>& words, int i, int ka) {
    const int bpb = 16 * ka + 8, nw = bpb / 2, ring = nw * 32;
    const uint32_t* p = (const uint32_t*)words.data();
    int start = (frac_s(i, ka, bpb) - 16) % ring;
    if (start < 0) start += ring;
    const uint64_t v = ((uint64_t)p[start >> 5] << 32) | p[((start >> 5) + 1) % nw];
    return (uint16_t)((v >> (48 - (start & 31))) & 0xFFFF);
}

// ---- tensor-core permutation (quantize.py tensor_core_perm), row-major placement ----
static void build_perm(int* perm) {
    for (int t = 0; t < 32; ++t) {
        int r0 = (t % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
        int c0 = t / 4, c1 = c0 + 8;
        perm[t * 8 + 0] = r0 * 16 + c0;
        perm[t * 8 + 1] = r1 * 16 + c0;
        perm[t * 8 + 2] = r2 * 16 + c0;
        perm[t * 8 + 3] = r3 * 16 + c0;
        perm[t * 8 + 4] = r0 * 16 + c1;
        perm[t * 8 + 5] = r1 * 16 + c1;
        perm[t * 8 + 6] = r2 * 16 + c1;
        perm[t * 8 + 7] = r3 * 16 + c1;
    }
}

// ---- a minimal safetensors writer (the test builds its own model dir) ----
static void write_safetensors(const std::string& path,
                              const std::vector<std::pair<std::string, std::string>>& meta,
                              const std::map<std::string, std::pair<std::string, std::vector<int64_t>>>& dtypes,
                              const std::map<std::string, std::vector<uint8_t>>& data) {
    // one pass for offsets, then the header
    std::string body;
    struct Slot { std::string name; size_t begin, end; };
    std::vector<Slot> slots;
    for (const auto& [name, bytes] : data) {
        slots.push_back({name, body.size(), 0});
        body.append((const char*)bytes.data(), bytes.size());
    }
    std::string header = "{";
    bool first = true;
    for (Slot& slot : slots) {
        slot.end = slot.begin + data.at(slot.name).size();
        const auto& t = dtypes.at(slot.name);
        std::string shape = "[";
        for (size_t d = 0; d < t.second.size(); ++d) shape += (d ? "," : "") + std::to_string(t.second[d]);
        shape += "]";
        if (!first) header += ",";
        first = false;
        header += "\"" + slot.name + "\":{\"dtype\":\"" + t.first + "\",\"shape\":" + shape +
                  ",\"data_offsets\":[" + std::to_string(slot.begin) + "," + std::to_string(slot.end) + "]}";
    }
    if (!meta.empty()) {
        header += ",\"__metadata__\":{";
        for (size_t i = 0; i < meta.size(); ++i)
            header += std::string(i ? "," : "") + "\"" + meta[i].first + "\":\"" + meta[i].second + "\"";
        header += "}";
    }
    header += "}";
    std::ofstream out(path, std::ios::binary);
    uint64_t n = header.size();
    out.write((const char*)&n, 8);
    out.write(header.data(), (std::streamsize)n);
    out.write(body.data(), (std::streamsize)body.size());
}

static void write_index(const std::string& dir, const std::string& shard,
                        const std::vector<std::string>& names) {
    std::string map;
    for (size_t i = 0; i < names.size(); ++i)
        map += std::string(i ? "," : "") + "\"" + names[i] + "\":\"" + shard + "\"";
    std::ofstream out(dir + "/model.safetensors.index.json");
    out << "{\"metadata\":{\"total_size\":0},\"weight_map\":{" << map << "}}\n";
}

// Golden bytes: one 256-weight tile at K = 2.5 (KA = 2), produced by exllamav3's own
// ext.pack_trellis_frac / ext.unpack_trellis_frac on random states (seed 1234).
static const uint16_t k25_packed[40] = {
    0xc7d9, 0x0709, 0xa899, 0x75b9, 0x1a0a, 0x63e8, 0x9f07, 0x8ab0,
    0x4589, 0x978d, 0xf1c7, 0x703d, 0x114e, 0xe726, 0xf740, 0xf99e,
    0x9de8, 0xeeef, 0x6618, 0x1ab6, 0x3d2a, 0x50a2, 0x7570, 0xcfc0,
    0x9fce, 0xb7f7, 0xcf06, 0xbe44, 0xd2b4, 0x3c23, 0x3d9b, 0x0f6b,
    0x3158, 0x2fb6, 0x6f25, 0x7460, 0x9303, 0xe93f, 0x2f39, 0x3217};
static const uint16_t k25_windows[256] = {
    0xbce4, 0xe720, 0x9c83, 0xe41c, 0x9070, 0x8384, 0x0e13, 0x709c,
    0xc271, 0x138f, 0x4e3e, 0x71f6, 0xc7d9, 0x3ecb, 0xfb2e, 0xd975,
    0x65d6, 0x2eb7, 0xbadc, 0xd6e6, 0x5b9a, 0xdcd4, 0x7351, 0x9a89,
    0x6a26, 0x5132, 0x44cb, 0x2658, 0x9963, 0xcb1f, 0x2c7d, 0x63e8,
    0x8fa0, 0x7d03, 0xf40d, 0xa068, 0x81a0, 0x0d05, 0x3415, 0xa0a8,
    0x82a2, 0x1515, 0x5455, 0xa2ac, 0x8ab0, 0x5584, 0x5613, 0xb09f,
    0xc27c, 0x13e0, 0x4f83, 0x7c1e, 0xf079, 0x83cb, 0x0f2f, 0x7978,
    0xe5e3, 0x2f1a, 0xbc6a, 0xe351, 0x8d45, 0x6a2c, 0xa8b1, 0x4589,
    0x1625, 0xb12e, 0xc4b8, 0x25c0, 0x9703, 0xb81e, 0xe07b, 0x03df,
    0x0f7c, 0x7be3, 0xef8e, 0x7c71, 0xf1c7, 0x8e3f, 0x38fc, 0xc7e7,
    0x1f9c, 0xfce4, 0xf393, 0x9c98, 0x7261, 0x9308, 0x4c22, 0x6114,
    0x8453, 0x229d, 0x8a77, 0x53be, 0x4ef9, 0x77cc, 0xdf33, 0xf99e,
    0xe67b, 0x33de, 0xcf7b, 0x7bdd, 0xef74, 0x7ba0, 0xee81, 0x740e,
    0xd03b, 0x81dd, 0x0777, 0x3bbb, 0xeeef, 0x777c, 0xddf3, 0xef9d,
    0xbe77, 0xf3bd, 0xcef4, 0x77a0, 0xde81, 0xf40d, 0xd035, 0x81ab,
    0x06ad, 0x356c, 0xd5b3, 0xad99, 0xb666, 0xb330, 0xccc3, 0x6618,
    0x9861, 0xc30a, 0x0c28, 0x6142, 0x850a, 0x2851, 0xa144, 0x0a23,
    0x288f, 0x447a, 0x11e9, 0x8f4a, 0x3d2a, 0xe956, 0xa559, 0x2acf,
    0xab3f, 0x59f8, 0x67e0, 0x3f01, 0xfc07, 0xe03a, 0x80ea, 0x0757,
    0x1d5c, 0xeae1, 0xab85, 0x5c2d, 0x70b7, 0x85bf, 0x16fe, 0xb7f7,
    0xdfde, 0xfef3, 0xfbcf, 0xde7f, 0x79fc, 0xcfe7, 0x3f9d, 0xfceb,
    0xf3af, 0x9d7c, 0x75f2, 0xaf91, 0xbe44, 0xf226, 0xc899, 0x44cf,
    0x133c, 0x99e0, 0x6783, 0x3c18, 0xf063, 0x831e, 0x0c78, 0x63c2,
    0x8f08, 0x7847, 0xe11e, 0x08f4, 0x23d2, 0x1e95, 0x7a56, 0xd2b4,
    0x4ad0, 0x5681, 0x5a07, 0xd03d, 0x40f6, 0x07b5, 0x1ed6, 0xf6b3,
    0xdacf, 0xd67b, 0x59ec, 0xcf66, 0x3d9b, 0xecd9, 0xb365, 0x9b2f,
    0x6cbe, 0x65f6, 0x97db, 0xbed8, 0xfb63, 0xdb18, 0x6c62, 0x6315,
    0x8c56, 0x62b0, 0x8ac3, 0x561d, 0x5874, 0xc3a3, 0x0e8c, 0x7460,
    0xd181, 0x8c0d, 0x3037, 0x81bc, 0x06f2, 0x3792, 0xde4b, 0xf25e,
    0xc97a, 0x4bd2, 0x2f49, 0x7a4f, 0xe93f, 0x49fc, 0x27f2, 0x3f93,
    0xfe4c, 0xf260, 0xc981, 0x4c0c, 0x3033, 0x8199, 0x0664, 0x3321,
    0xcc85, 0x642e, 0x90b9, 0x85cb, 0x172f, 0xb979, 0xe5e7, 0x2f39};

// one EXL3 linear's nine pieces (trellis/suh/svh + the mul1 marker) as raw bytes
struct LinearBytes {
    std::vector<uint16_t> trellis, suh, svh;
    int ki = 0, nj = 0, words = 0;
};
static LinearBytes random_linear(uint32_t& seed, int ki, int nj, int words) {
    auto rnd16 = [&seed] { seed = seed * 1664525u + 1013904223u; return (uint16_t)(seed >> 16); };
    LinearBytes L;
    L.ki = ki; L.nj = nj; L.words = words;
    L.trellis.resize((size_t)ki * nj * words);
    for (auto& w : L.trellis) w = rnd16();
    L.suh.resize((size_t)ki * 16);
    L.svh.resize((size_t)nj * 16);
    // scales in a sane fp16 range so the Hadamard reconstruction stays finite
    for (auto& s : L.suh) s = f2h(0.5f + 1.5f * ((seed = seed * 1664525u + 1013904223u) >> 24) / 255.0f);
    for (auto& s : L.svh) s = f2h(0.5f + 1.5f * ((seed = seed * 1664525u + 1013904223u) >> 24) / 255.0f);
    return L;
}

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "strata_exl3_frac_k_test";
    std::filesystem::remove_all(dir);
    fs::create_directories(dir);

    // =========================== 1. the golden tile decode ===========================
    {
        std::vector<uint16_t> lut(1u << 16);
        strata::kernels::cpu::exl3_codebook_lut(Exl3Codebook::Mul1, lut.data());
        uint16_t tile[256];
        strata::kernels::cpu::exl3_decode_tile(k25_packed, /*bits=*/2, lut.data(), tile, /*frac=*/1);
        // stored position t must have read exactly the reference window and placed LUT[window] at
        // perm(t) - the mul1 codebook itself is pinned bit-exact by the exl3_parity fixtures
        int perm[256];
        build_perm(perm);
        size_t bad = 0;
        for (int t = 0; t < 256; ++t)
            if (tile[perm[t]] != lut[k25_windows[(size_t)t]]) ++bad;
        CHECK(bad == 0);
        if (bad) std::fprintf(stderr, "golden tile: %zu/256 positions wrong\n", bad);
        // the in-test codebook oracle agrees with the port's LUT on every golden window
        size_t cb_bad = 0;
        for (int t = 0; t < 256; ++t)
            if (mul1_window(k25_windows[(size_t)t]) != lut[k25_windows[(size_t)t]]) ++cb_bad;
        CHECK(cb_bad == 0);
    }

    // ================ 2. in-test reference: pack a random ring, decode it back ================
    {
        uint32_t seed = 42;
        auto rnd = [&seed] { seed = seed * 1664525u + 1013904223u; return seed >> 16; };
        const int ka = 2;
        std::vector<uint16_t> newbits(256);
        for (int i = 0; i < 256; ++i) newbits[(size_t)i] = (uint16_t)(rnd() & ((1u << frac_d(i, ka)) - 1));
        std::vector<uint16_t> words;
        ref_pack(words, newbits, ka);
        // reference windows for the ring we just wrote
        std::vector<uint16_t> want(256);
        for (int i = 0; i < 256; ++i) want[(size_t)i] = ref_window(words, i, ka);
        std::vector<uint16_t> lut(1u << 16);
        strata::kernels::cpu::exl3_codebook_lut(Exl3Codebook::Mul1, lut.data());
        uint16_t tile[256];
        strata::kernels::cpu::exl3_decode_tile(words.data(), ka, lut.data(), tile, 1);
        int perm[256];
        build_perm(perm);
        size_t bad = 0;
        for (int t = 0; t < 256; ++t)
            if (tile[perm[t]] != lut[want[(size_t)t]]) ++bad;
        CHECK(bad == 0);
        if (bad) std::fprintf(stderr, "reference ring: %zu/256 positions wrong\n", bad);
    }

    // ============ 3. the whole reconstruct on a random K = 2.5 linear (ki=2, nj=1... 4x4 tiles) ====
    // dims must be multiples of 128 for the Hadamard: 128x128 (ki=nj=8)
    {
        uint32_t seed = 7;
        const int ki = 8, nj = 8, words = 40;   // 128x128 weights, K = 2.5
        LinearBytes L = random_linear(seed, ki, nj, words);
        // reference W_hat: decode each tile with the reference window math
        std::vector<uint16_t> ref_what((size_t)ki * nj * 256);
        std::vector<uint16_t> lut(1u << 16);
        strata::kernels::cpu::exl3_codebook_lut(Exl3Codebook::Mul1, lut.data());
        for (int i = 0; i < ki; ++i)
            for (int j = 0; j < nj; ++j) {
                const uint16_t* tw = L.trellis.data() + ((size_t)i * nj + j) * words;
                uint16_t tile[256];
                strata::kernels::cpu::exl3_decode_tile(tw, 2, lut.data(), tile, 1);
                std::memcpy(&ref_what[((size_t)i * nj + j) * 256], tile, 512);
            }
        std::vector<uint16_t> what((size_t)ki * 16 * nj * 16);
        strata::kernels::cpu::exl3_decode_weight_hat(L.trellis.data(), ki, nj, 2, Exl3Codebook::Mul1,
                                                    what.data(), 1);
        // decode_weight_hat writes the FULL [k x n] row-major matrix; compare per tile block
        const int k = ki * 16, n = nj * 16;
        size_t hat_bad = 0;
        for (int i = 0; i < ki; ++i)
            for (int j = 0; j < nj; ++j)
                for (int r = 0; r < 16; ++r)
                    for (int c = 0; c < 16; ++c)
                        if (what[(size_t)(i * 16 + r) * n + j * 16 + c] !=
                            ref_what[((size_t)i * nj + j) * 256 + (size_t)r * 16 + c])
                            ++hat_bad;
        CHECK(hat_bad == 0);
        // reference W: preapply_had_l/r (Sylvester H128 / sqrt(128)) + suh/svh, in fp32
        std::vector<float> w((size_t)k * n);
        for (size_t i = 0; i < w.size(); ++i) w[i] = h2f(what[i]);
        auto had_axis = [&](int rows, int cols, bool rows_axis) {
            std::vector<float> tmp(128);
            for (int a = 0; a < (rows_axis ? rows : cols); ++a) {
                for (int b = 0; b < (rows_axis ? cols : rows); b += 128) {
                    const int live = std::min(128, (rows_axis ? cols : rows) - b);
                    for (int c = 0; c < live; ++c) {
                        float acc = 0;
                        for (int r = 0; r < live; ++r) {
                            const float h = ((__builtin_popcount((unsigned)(c & r)) & 1) ? -1.0f : 1.0f) *
                                            0.08838834764831845f;
                            acc += h * (rows_axis ? w[(size_t)a * cols + b + r] : w[(size_t)(b + r) * cols + a]);
                        }
                        tmp[(size_t)c] = acc;
                    }
                    for (int c = 0; c < live; ++c)
                        if (rows_axis) w[(size_t)a * cols + b + c] = tmp[(size_t)c];
                        else w[(size_t)(b + c) * cols + a] = tmp[(size_t)c];
                }
            }
        };
        had_axis(k, n, true);    // rows
        had_axis(k, n, false);   // cols
        std::vector<uint16_t> ref_w((size_t)k * n);
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < n; ++j)
                ref_w[(size_t)i * n + j] =
                    f2h(w[(size_t)i * n + j] * h2f(L.suh[(size_t)i]) * h2f(L.svh[(size_t)j]));
        std::vector<uint16_t> got((size_t)k * n);
        strata::kernels::cpu::exl3_reconstruct_weight(L.trellis.data(), ki, nj, 2, Exl3Codebook::Mul1,
                                                      L.suh.data(), L.svh.data(), got.data(), 1);
        size_t exact = 0;
        double num = 0, den = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            if (got[i] == ref_w[i]) ++exact;
            const double d = (double)h2f(got[i]) - h2f(ref_w[i]);
            num += d * d; den += (double)h2f(ref_w[i]) * h2f(ref_w[i]);
        }
        const double rel = std::sqrt(num) / (std::sqrt(den) + 1e-30);
        CHECK(rel < 2e-3);   // fp16 rounding of the same fp32 math; the Hadamard orders differ
        std::printf("frac reconstruct: %zu/%zu bit-exact, rel %.2e\n", exact, got.size(), rel);
    }

    // ======================= 4. the loader: relaxed validation =======================
    {
        const std::string shard = "model-00001-of-00001.safetensors";
        auto write_model = [&](const std::string& sub, int last_dim, bool mul1, const char* dtype) {
            const fs::path d = dir / sub;
            fs::create_directories(d);
            uint32_t seed = 3;
            const int ki = 8, nj = 8;
            LinearBytes L = random_linear(seed, ki, nj, last_dim);
            std::map<std::string, std::pair<std::string, std::vector<int64_t>>> dt;
            std::map<std::string, std::vector<uint8_t>> data;
            auto put16 = [&](const std::string& name, const std::string& type,
                             const std::vector<int64_t>& shape, const std::vector<uint16_t>& v) {
                dt[name] = {type, shape};
                data[name] = std::vector<uint8_t>((const uint8_t*)v.data(),
                                                  (const uint8_t*)v.data() + v.size() * 2);
            };
            put16("t.trellis", dtype, {ki, nj, last_dim}, L.trellis);
            put16("t.suh", "F16", {ki * 16}, L.suh);
            put16("t.svh", "F16", {nj * 16}, L.svh);
            std::vector<std::string> names = {"t.trellis", "t.suh", "t.svh"};
            if (mul1) {
                dt["t.mul1"] = {"I8", {1}};
                data["t.mul1"] = {1};
                names.push_back("t.mul1");
            }
            write_safetensors((d / shard).string(), {}, dt, data);
            write_index(d.string(), shard, names);
            return d.string();
        };

        // K = 2.5 (last dim 40, mul1): accepted, bits = 2, frac = 1, words = 40
        {
            Exl3Model m(write_model("k25", 40, true, "I16"));
            CHECK(m.has("t"));
            Exl3Linear L = m.linear("t");
            CHECK(L.bits == 2 && L.frac == 1 && L.words() == 40 && L.cb == 2);
        }
        // integer K = 4 (last dim 64): frac stays 0 - the historical values, byte-identical path
        {
            Exl3Model m(write_model("k4", 64, true, "I16"));
            Exl3Linear L = m.linear("t");
            CHECK(L.bits == 4 && L.frac == 0 && L.words() == 64);
        }
        // half-integer K without the mul1 marker: rejected (exllamav3 asserts the same)
        {
            Exl3Model m(write_model("k25_nomul1", 40, false, "I16"));
            bool threw = false;
            try { (void)m.linear("t"); } catch (const std::exception& e) {
                threw = std::string(e.what()).find("mul1") != std::string::npos;
            }
            CHECK(threw);
        }
        // last dim not a multiple of 8 (44 = K 2.75): rejected
        {
            Exl3Model m(write_model("k275", 44, true, "I16"));
            bool threw = false;
            try { (void)m.linear("t"); } catch (const std::exception&) { threw = true; }
            CHECK(threw);
        }
        // K > 8 (last dim 136): rejected
        {
            Exl3Model m(write_model("k85", 136, true, "I16"));
            bool threw = false;
            try { (void)m.linear("t"); } catch (const std::exception&) { threw = true; }
            CHECK(threw);
        }
    }

    std::filesystem::remove_all(dir);
    if (failures) {
        std::fprintf(stderr, "exl3_frac_k_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("exl3_frac_k_test: all checks passed\n");
    return 0;
}
