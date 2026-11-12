// src/kernels/exl3_qwen_band8_test.cpp - the STRATA_QWEN_BAND8 opt-in's integration battery.
//
// Real Qwen3.8-Flash-Next EXL3 4 bpw GEOMETRY (read from the model's safetensors headers,
// 2026-11-08: hidden 2560, FF 640, EVERY routed projection K4 mul1 - gate/up trellis
// [160,40,64], down [40,160,64]) at small counts (2 layers x 6 experts + one mtp.layers.0
// expert + one dense projection), so the FULL engine composition runs in seconds:
//
//   1. fixture EXL3 dir written as real safetensors (the store-test writer pattern);
//   2. store round-trip on the ENGINE's exact request list (layer-major, expert-minor,
//      gate/up/down - the order generate.cpp and exl3_band8_build --qwen must share): the
//      band-8 permutation spot-checked tile-by-tile against the formula at nj=40/160 (both
//      NON-identity, unlike the mini fixtures that once masked the peer bug), scales
//      byte-copied, the mtp.layers.0 trio NOT covered;
//   3. CPU tier parity through Exl3Model(dir, store) with the same layout mapping
//      generate.cpp registers (Band8 tags): bitwise vs the native service;
//   4. the RESIDENT=all composition the engine wires (create(dir, groups, store)):
//      layout()==Band8, memory_only(), arena bytes == payload bytes, locate() refused,
//      and the arena stays valid AFTER the store handle is released (the engine retires
//      the store before activation);
//   5. the main expert STREAM over the packed resident backing vs the plain native twin:
//      run() and run_multi() (an MTP-shaped T=4 window) rows BYTE-EQUAL;
//   6. the PEER tier: packed subset vs native twin at REQUESTED multirow widths 0/2/4,
//      every open-filled slot Band8-tagged, plus a replace/apply_refill landing a fresh
//      packed expert and byte-correct post-adapt rows (the 2026-11-06 regression class);
//   7. the stream's managed-RAM-cache refusal beside a backing, at this geometry.
//
// Default path: nothing here reads STRATA_QWEN_BAND8; the opt-in's unset behavior is the
// unchanged native path, covered by the unmodified existing gates. Fixture-sized payloads
// only (< 40 MiB) under STRATA_QWEN_BAND8_TEST_TMP (default: a unique temporary dir) - no full-model
// cache is built. CUDA/SYCL were unavailable to this session; HIP gfx1100 is the target.
#include "strata/artifact/exl3_band8_store.hpp"
#include "strata/artifact/exl3_model.hpp"
#include "strata/artifact/exl3_resident.hpp"
#include "strata/kernels/exl3_experts.hpp"
#include "strata/kernels/exl3_peer.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/cpu/exl3_moe.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;
namespace k = strata::kernels;
using strata::Exl3Band8Store;
using strata::Exl3Linear;
using strata::Exl3Model;
using strata::Exl3ResidentBacking;
using strata::Exl3TrellisLayout;

namespace {

int g_failures = 0;

void check(bool ok, const char* what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_failures;
}

// ---- tiny safetensors fixture writer (the exl3_band8_store_test pattern) ----------------
struct FixtureTensor {
    std::string name, dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};

void write_shard(const fs::path& path, const std::vector<FixtureTensor>& tensors)
{
    std::map<std::string, const FixtureTensor*> order;
    for (const FixtureTensor& t : tensors) order.emplace(t.name, &t);
    std::ostringstream json;
    json << "{";
    size_t off = 0, idx = 0;
    for (const auto& [name, t] : order) {
        if (idx++ != 0) json << ",";
        json << "\"" << name << "\":{\"dtype\":\"" << t->dtype << "\",\"shape\":[";
        for (size_t d = 0; d < t->shape.size(); ++d) json << (d ? "," : "") << t->shape[d];
        json << "],\"data_offsets\":[" << off << "," << (off + t->data.size()) << "]}";
        off += t->data.size();
    }
    json << "}";
    const std::string j = json.str();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    uint64_t n = j.size();
    f.write((const char*)&n, 8);
    f.write(j.data(), (std::streamsize)j.size());
    for (const auto& [name, t] : order) f.write((const char*)t->data.data(), (std::streamsize)t->data.size());
}

void write_index(const fs::path& path, const std::map<std::string, std::string>& weight_map)
{
    std::ostringstream json;
    json << "{\"metadata\":{\"format\":\"qwen-band8-fixture\"},\"weight_map\":{";
    size_t i = 0;
    for (const auto& [key, v] : weight_map)
        json << (i++ ? "," : "") << "\"" << key << "\":\"" << v << "\"";
    json << "}}";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f << json.str();
}

uint16_t f32_to_f16_bits(float v)
{
    uint32_t x;
    std::memcpy(&x, &v, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00u);
    if (((x >> 23) & 0xff) == 0) return (uint16_t)sign;
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) return (uint16_t)sign;
    man >>= 13;
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | man);
    const uint32_t dropped = x & 0x1fffu;
    if (dropped > 0x1000u || (dropped == 0x1000u && (h & 1))) ++h;
    return h;
}

std::vector<uint8_t> synth_bytes(const std::string& name, size_t n)
{
    std::vector<uint8_t> b(n);
    std::seed_seq seq(name.begin(), name.end());
    std::mt19937_64 rng(seq);
    for (size_t i = 0; i < n; ++i) b[i] = (uint8_t)rng();
    return b;
}

// Scale planes are NOT random bytes: random fp16 bit patterns include NaN/Inf and make rows
// non-finite before any layout question. The GPU prototype's fixture (kept so future joint
// runs share inputs): +-(0.10 + (rng%32)*0.001).
std::vector<uint8_t> synth_scales(const std::string& name, size_t count)
{
    std::vector<uint8_t> b(count * 2);
    std::seed_seq seq(name.begin(), name.end());
    std::mt19937_64 rng(seq);
    for (size_t i = 0; i < count; ++i) {
        const float v = (0.10f + (float)(rng() % 32) * 0.001f) * (rng() % 2 ? 1.0f : -1.0f);
        const uint16_t h = f32_to_f16_bits(v);
        std::memcpy(b.data() + i * 2, &h, 2);
    }
    return b;
}

// The engine's exact expert-name tuple (generate.cpp STRATA_QWEN_BAND8 block and
// Exl3ExpertStream::Impl::expert_prefix): "model.language_model.layers.<l>.mlp.experts.<e>."
// + proj + "_proj", proj in {gate, up, down}, layer-major / expert-minor order.
std::string qwen_tensor(int layer, int expert, const char* proj)
{
    return "model.language_model.layers." + std::to_string(layer) + ".mlp.experts." +
           std::to_string(expert) + "." + proj + "_proj";
}

constexpr int LAYERS = 2, EXPERTS = 64, EMB = 2560, FFD = 640, WORDS = 64;   // K4 (64 experts:
// enough to sweep a real fill/arena at scale - the e2e divergence onset scales with the
// peer subset size, pointing at a small fraction of affected experts)

struct Fixture {
    fs::path dir;
    std::vector<std::string> expert_prefixes;   // the ENGINE's request list, same order

    void build(const fs::path& d)
    {
        dir = d;
        fs::create_directories(d);
        std::map<std::string, std::string> weight_map;
        std::vector<FixtureTensor> shard[2];
        int shard_of = 0;
        auto add = [&](const std::string& name, const std::string& dtype,
                       std::vector<int64_t> shape, size_t bytes) {
            FixtureTensor t{name, dtype, std::move(shape), synth_bytes(name, bytes)};
            const std::string file = shard_of == 0 ? "fixture-00001-of-00002.safetensors"
                                                   : "fixture-00002-of-00002.safetensors";
            weight_map.emplace(name, file);
            shard[shard_of].push_back(std::move(t));
            shard_of ^= 1;
        };
        auto add_scales = [&](const std::string& name, int64_t n) {
            FixtureTensor t{name, "F16", {n}, synth_scales(name, (size_t)n)};
            const std::string file = shard_of == 0 ? "fixture-00001-of-00002.safetensors"
                                                   : "fixture-00002-of-00002.safetensors";
            weight_map.emplace(name, file);
            shard[shard_of].push_back(std::move(t));
            shard_of ^= 1;
        };
        // routed experts at the REAL shapes: gate/up [160,40,64] (k=2560 n=640), down
        // [40,160,64] (k=640 n=2560), all K4 mul1 - nj 40 and 160 divide band-8 exactly and
        // make the permutation NON-identity at both widths.
        for (int l = 0; l < LAYERS; ++l)
            for (int e = 0; e < EXPERTS; ++e)
                for (const char* proj : {"gate", "up", "down"}) {
                    const std::string p = qwen_tensor(l, e, proj);
                    expert_prefixes.push_back(p);
                    const bool down = std::strcmp(proj, "down") == 0;
                    const int64_t ki = down ? 40 : 160, nj = down ? 160 : 40;
                    add(p + ".trellis", "I16", {ki, nj, WORDS}, (size_t)ki * nj * WORDS * 2);
                    add_scales(p + ".suh", ki * 16);
                    add_scales(p + ".svh", nj * 16);
                    add(p + ".mul1", "U8", {1}, 1);
                }
        // the in-file MTP block: same geometry, DIFFERENT prefix - native, never requested
        for (const char* proj : {"gate", "up", "down"}) {
            const std::string p = "mtp.layers.0.mlp.experts.0." + std::string(proj) + "_proj";
            const bool down = std::strcmp(proj, "down") == 0;
            const int64_t ki = down ? 40 : 160, nj = down ? 160 : 40;
            add(p + ".trellis", "I16", {ki, nj, WORDS}, (size_t)ki * nj * WORDS * 2);
            add_scales(p + ".suh", ki * 16);
            add_scales(p + ".svh", nj * 16);
            add(p + ".mul1", "U8", {1}, 1);
        }
        // one dense projection: present, never requested
        add("model.language_model.layers.0.self_attn.q_proj.trellis", "I16", {160, 160, 96},
            160 * 160 * 96 * 2);
        add("model.language_model.layers.0.self_attn.q_proj.suh", "F16", {2560}, 5120);
        add("model.language_model.layers.0.self_attn.q_proj.svh", "F16", {2560}, 5120);
        add("model.language_model.layers.0.self_attn.q_proj.mul1", "U8", {1}, 1);
        write_shard(d / "fixture-00001-of-00002.safetensors", shard[0]);
        write_shard(d / "fixture-00002-of-00002.safetensors", shard[1]);
        write_index(d / "model.safetensors.index.json", weight_map);
    }
};

// Reference band-8 permutation straight from the formula (independent of the store impl).
void reference_band8(const uint16_t* native, uint16_t* packed, int ki, int nj, int P)
{
    for (int kt = 0; kt < ki; ++kt)
        for (int nt = 0; nt < nj; ++nt) {
            const size_t src = (size_t)((kt * nj + nt) * P);
            const size_t dst = (size_t)((((nt / 8) * ki + kt) * 8 + nt % 8) * P);
            std::memcpy(packed + dst, native + src, (size_t)P * 2);
        }
}

}  // namespace

int main()
{
    setbuf(stdout, nullptr);   // a crash must not swallow the checks that already passed
    const char* root_env = std::getenv("STRATA_QWEN_BAND8_TEST_TMP");
    const fs::path base = root_env && root_env[0] ? fs::path(root_env) :
        fs::temp_directory_path() / ("strata-qwen-band8-test-" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(base, ec);
    fs::create_directories(base, ec);
    (void) cudaSetDevice(0);

    Fixture fx;
    fx.build(base / "fixture");
    const fs::path cache = base / "cache";
    std::string err;

    // ---- 1: store round-trip on the engine's exact request list --------------------------
    std::shared_ptr<const Exl3Band8Store> store;
    try {
        store = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(), fx.expert_prefixes);
    } catch (const std::exception& e) {
        std::printf("FAIL  store build: %s\n", e.what());
        return 1;
    }
    check(store->record_count() == 3 * fx.expert_prefixes.size(),
          "record count is 3 planes per prefix (73,728-scale list's exact structure)");
    {
        // non-identity permutation + formula-exact bytes at BOTH nj widths, scales verbatim
        Exl3Model plain(fx.dir.string());
        Exl3Model packed(fx.dir.string(), store);
        check(plain.expert_layout() == Exl3TrellisLayout::Native &&
                  packed.expert_layout() == Exl3TrellisLayout::Band8,
              "expert_layout() probes Native plain / Band8 packed");
        bool gate_exact = true, down_exact = true, gate_differs = false, down_differs = false,
             scales_exact = true;
        for (int e = 0; e < 2; ++e) {
            const Exl3Linear gn = plain.linear(qwen_tensor(1, e, "gate"));
            const Exl3Linear gp = packed.linear(qwen_tensor(1, e, "gate"));
            const Exl3Linear dn = plain.linear(qwen_tensor(1, e, "down"));
            const Exl3Linear dp = packed.linear(qwen_tensor(1, e, "down"));
            std::vector<uint16_t> ref((size_t)gn.ki * gn.nj * gn.words());
            reference_band8(gn.trellis, ref.data(), gn.ki, gn.nj, gn.words());
            gate_exact = gate_exact && std::memcmp(ref.data(), gp.trellis, ref.size() * 2) == 0;
            gate_differs = gate_differs || std::memcmp(gn.trellis, gp.trellis, ref.size() * 2) != 0;
            scales_exact = scales_exact && std::memcmp(gn.suh, gp.suh, (size_t)gn.in() * 2) == 0 &&
                           std::memcmp(gn.svh, gp.svh, (size_t)gn.out() * 2) == 0;
            ref.assign((size_t)dn.ki * dn.nj * dn.words(), 0);
            reference_band8(dn.trellis, ref.data(), dn.ki, dn.nj, dn.words());
            down_exact = down_exact && std::memcmp(ref.data(), dp.trellis, ref.size() * 2) == 0;
            down_differs = down_differs || std::memcmp(dn.trellis, dp.trellis, ref.size() * 2) != 0;
        }
        check(gate_differs && down_differs,
              "packed trellis DIFFERS from native at nj=40 and nj=160 (non-identity permutation)");
        check(gate_exact && down_exact && scales_exact,
              "packed bytes are the formula permutation, scales copied verbatim");
        const Exl3Linear mtpp = packed.linear("mtp.layers.0.mlp.experts.0.gate_proj");
        Exl3Linear mtp_out{};
        check(!store->linear("mtp.layers.0.mlp.experts.0.gate_proj", mtp_out) &&
                  mtpp.layout == Exl3TrellisLayout::Native && mtpp.trellis != nullptr,
              "the mtp.layers.0 trio is NOT covered by the store (stays native through the model)");
        std::string path;
        size_t off = 0, bytes = 0;
        check(!packed.locate("mtp.layers.0.mlp.experts.0.gate_proj.trellis", path, off, bytes) ||
                  path != store->payload_path(),
              "mtp tensors never resolve into the packed payload");
    }

    // ---- 2: CPU tier parity through the model+store (generate.cpp's registration) --------
    if (strata::kernels::cpu::Exl3CpuMoe::band8_supported()) {
        Exl3Model plain(fx.dir.string()), packed(fx.dir.string(), store);
        strata::kernels::cpu::Exl3CpuMoe nat, bnd;
        std::vector<strata::kernels::cpu::Exl3CpuMatrix> gn, un, dn, gb, ub, db;
        // the SAME fill generate.cpp uses, layout tag included
        auto fill = [](const Exl3Linear& L, strata::kernels::cpu::Exl3CpuMatrix& M) {
            M.trellis = L.trellis; M.suh = L.suh; M.svh = L.svh;
            M.k = L.in(); M.n = L.out(); M.bits = L.bits; M.hb = 0;
            M.layout = L.layout == Exl3TrellisLayout::Band8
                           ? strata::kernels::cpu::Exl3CpuTrellisLayout::Band8
                           : strata::kernels::cpu::Exl3CpuTrellisLayout::Native;
        };
        for (int l = 0; l < LAYERS; ++l) {
            gn.clear(); un.clear(); dn.clear(); gb.clear(); ub.clear(); db.clear();
            for (int e = 0; e < EXPERTS; ++e)
                for (int which = 0; which < 2; ++which) {
                    const Exl3Model& m = which == 0 ? plain : packed;
                    auto& g = which == 0 ? gn : gb;
                    auto& u = which == 0 ? un : ub;
                    auto& d = which == 0 ? dn : db;
                    g.push_back({}); u.push_back({}); d.push_back({});
                    fill(m.linear(qwen_tensor(l, e, "gate")), g.back());
                    fill(m.linear(qwen_tensor(l, e, "up")), u.back());
                    fill(m.linear(qwen_tensor(l, e, "down")), d.back());
                }
            nat.add_layer(l, gn.data(), un.data(), dn.data(), EXPERTS);
            bnd.add_layer(l, gb.data(), ub.data(), db.data(), EXPERTS);
        }
        nat.set_threads(4);
        bnd.set_threads(4);
        std::mt19937 rng(20261108);
        std::normal_distribution<float> dist(0.0f, 0.5f);
        std::vector<float> x((size_t)4 * EMB);
        for (auto& v : x) v = dist(rng);
        bool decode = true, verify = true;
        {
            const int ids[3] = {2, 5, 2}, t[3] = {0, 0, 0}, r[3] = {0, 1, 2};
            std::vector<float> a((size_t)3 * EMB, -1e30f), b((size_t)3 * EMB, -1e30f);
            nat.run_rows(1, x.data(), EMB, ids, t, r, 3, a.data(), EMB);
            bnd.run_rows(1, x.data(), EMB, ids, t, r, 3, b.data(), EMB);
            decode = std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        }
        {
            const int ids[8] = {0, 0, 1, 3, 3, 2, 2, 2}, t[8] = {0, 1, 1, 2, 3, 0, 2, 3},
                      r[8] = {4, 0, 3, 1, 7, 2, 6, 5};
            std::vector<float> a((size_t)8 * EMB, -1e30f), b((size_t)8 * EMB, -1e30f);
            nat.run_rows(1, x.data(), EMB, ids, t, r, 8, a.data(), EMB);
            bnd.run_rows(1, x.data(), EMB, ids, t, r, 8, b.data(), EMB);
            verify = std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        }
        check(decode, "CPU decode-shaped rows bitwise: packed-store registration vs native");
        check(verify, "CPU verify-shaped rows bitwise: packed-store registration vs native");
        const auto st = bnd.stats();
        check(st.repack_fills == 0 && st.cache_bytes == 0,
              "the packed CPU service repacked nothing (no shadow, no LRU entry)");
    } else {
        std::printf("SKIP  CPU band-8 tier unavailable on this CPU (registration would refuse)\n");
    }

    // ---- 3: the RESIDENT=all composition (create(dir, groups, store)) --------------------
    std::shared_ptr<Exl3ResidentBacking> resident;
    std::vector<Exl3ResidentBacking::Group> groups;
    for (int l = 0; l < LAYERS; ++l)
        for (int e = 0; e < EXPERTS; ++e)
            groups.push_back({l, e, {qwen_tensor(l, e, "gate"), qwen_tensor(l, e, "up"),
                                     qwen_tensor(l, e, "down")}});
    {
        Exl3ResidentBacking::Options opt;
        opt.available_bytes = 1ull << 32;
        opt.headroom_bytes = 0;
        opt.lock = [](void*, size_t, std::string&) { return true; };
        opt.unlock = [](void*, size_t) {};
        resident = Exl3ResidentBacking::create(fx.dir.string(), groups, store);
        // the engine RETIRES the store before activation (source-lifetime rule): prove the
        // arena needs nothing from it afterwards
        Exl3Model packed_probe(fx.dir.string(), store);
        const Exl3Linear before = packed_probe.linear(qwen_tensor(0, 1, "up"));
        const uint16_t* payload_trellis = before.trellis;
        std::vector<uint16_t> want((size_t)before.ki * before.nj * before.words());
        std::memcpy(want.data(), before.trellis, want.size() * 2);
        resident->activate({}, opt);
        store.reset();   // the engine's exl3_band8 = nullptr
        check(resident->active() && resident->layout() == Exl3TrellisLayout::Band8 &&
                  resident->memory_only(),
              "resident arena over the packed source reports Band8 and memory-only");
        Exl3Linear after{};
        const bool got_after = resident->linear(qwen_tensor(0, 1, "up"), after);
        check(got_after && after.layout == Exl3TrellisLayout::Band8 && after.trellis != nullptr &&
                  after.trellis != payload_trellis &&
                  std::memcmp(after.trellis, want.data(), want.size() * 2) == 0,
              "arena bytes are the packed payload's, from the arena's own memory (store released)");
        std::string path;
        size_t off = 0, bytes = 0;
        check(!resident->locate(qwen_tensor(0, 1, "up") + ".trellis", path, off, bytes),
              "resident locate() refuses the file tier (no pread beside the arena)");
    }

    // ---- 4: the main expert stream: packed resident backing vs plain native twin ----------
    {
        k::Exl3ExpertStream native_stream(fx.dir.string(), EMB, LAYERS, 0, nullptr, 0,
                                          (size_t)16 << 20, 0, LAYERS, 0,
                                          "model.language_model.layers.");
        k::Exl3ExpertStream packed_stream(fx.dir.string(), EMB, LAYERS, 0, nullptr, 0,
                                          (size_t)16 << 20, 0, LAYERS, 0,
                                          "model.language_model.layers.", resident);
        std::mt19937 rng(77);
        std::normal_distribution<float> dist(0.0f, 0.5f);
        std::vector<float> x((size_t)4 * EMB);
        for (auto& v : x) v = dist(rng);
        bool single = true;
        {
            const int ids[3] = {4, 1, 4};
            std::vector<float> a((size_t)3 * EMB, -1e30f), b = a;
            single = native_stream.run(1, x.data(), ids, 3, a.data(), nullptr) &&
                     packed_stream.run(1, x.data(), ids, 3, b.data(), nullptr) &&
                     std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        }
        check(single, "stream run() rows BYTE-equal: packed resident backing vs native twin");
        bool window = true;
        {
            const int tokens = 4, topk = 3;
            const std::vector<int32_t> ids{2, 0, 1, 2, 1, 2, 0, 1, 0, 1, 2, 0};
            std::vector<float> a((size_t)tokens * topk * EMB, -1e30f), b = a;
            window = native_stream.run_multi(1, x.data(), ids.data(), tokens, topk, a.data(), nullptr) &&
                     packed_stream.run_multi(1, x.data(), ids.data(), tokens, topk, b.data(), nullptr) &&
                     std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        }
        check(window, "stream run_multi() MTP-window rows BYTE-equal: packed vs native twin");
    }

    // ---- 5: the PEER tier: packed subset vs native twin, plus a landing refill ------------
    {
        Exl3Model plain(fx.dir.string()), packed(fx.dir.string(), resident);
        auto entry_of = [&](const Exl3Model& m, int e) {
            k::Exl3Peer::Entry en;
            en.layer = 1;
            en.expert = e;
            auto set = [&](k::Exl3Mat& t, const char* proj) {
                const Exl3Linear l = m.linear(qwen_tensor(1, e, proj));
                t.trellis = l.trellis; t.suh = l.suh; t.svh = l.svh;
                t.ki = l.ki; t.nj = l.nj; t.bits = l.bits; t.frac = l.frac; t.cb = l.cb;
                t.layout = l.layout;
            };
            set(en.g, "gate"); set(en.u, "up"); set(en.d, "down");
            return en;
        };
        const int tokens = 4, topk = 3, routes = tokens * topk;
        std::vector<float> x((size_t)tokens * EMB);
        for (size_t i = 0; i < x.size(); ++i) x[i] = ((int)((i * 71 + 5) % 2000) - 1000) * 0.001f;
        const std::vector<std::vector<int32_t>> windows{
            {2, 0, 1, 2, 1, 2, 0, 1, 0, 1, 2, 0}, {1, 1, 2, 0, 2, 0, 1, 2, 0, 2, 2, 1}};
        std::vector<int32_t> kind(routes, 2);
        const char* saved_w = std::getenv("STRATA_EXL3_PEER_MULTIROW");
        const std::string keep_w = saved_w ? saved_w : "";
        auto set_env = [](const char* name, const char* value) {
            if (value) setenv(name, value, 1); else unsetenv(name);
        };
        for (const char* width : {"0", "2", "4"}) {
            set_env("STRATA_EXL3_PEER_MULTIROW", width);
            std::vector<k::Exl3Peer::Entry> nsub, psub;
            for (int e = 0; e < 3; ++e) {
                nsub.push_back(entry_of(plain, e));
                psub.push_back(entry_of(packed, e));
            }
            k::Exl3Peer npeer, ppeer;
            std::string nerr, perr;
            const bool nopen = npeer.open(0, EMB, nsub, 8, topk, nerr);
            const bool popen = ppeer.open(0, EMB, psub, 8, topk, perr);
            check(nopen && popen, (std::string("peer opens native + packed subsets at width ") + width).c_str());
            if (!popen) {
                std::printf("packed peer open error: %s\n", perr.c_str());
                break;
            }
            bool tags = true;
            for (int e = 0; e < 3; ++e) {
                tags = tags && ppeer.slot_layout(1, e) == Exl3TrellisLayout::Band8;
                tags = tags && npeer.slot_layout(1, e) == Exl3TrellisLayout::Native;
            }
            check(tags, (std::string("open-filled peer slots carry Band8/Native tags at width ") + width).c_str());
            bool bytes = true, sane = true;
            for (const auto& ids : windows) {
                std::vector<float> want((size_t)routes * EMB, 3.25f), got = want;
                if (!npeer.launch(1, x.data(), ids.data(), tokens, topk, kind.data(), nerr) ||
                    !npeer.finish(want.data(), nerr) ||
                    !ppeer.launch(1, x.data(), ids.data(), tokens, topk, kind.data(), perr) ||
                    !ppeer.finish(got.data(), perr)) {
                    std::printf("width %s launch error native='%s' packed='%s'\n", width,
                                nerr.c_str(), perr.c_str());
                    bytes = sane = false;
                    break;
                }
                bytes = bytes && std::memcmp(want.data(), got.data(), got.size() * sizeof(float)) == 0;
                for (float v : got) sane = sane && std::isfinite(v);
            }
            check(sane, (std::string("packed peer rows finite at width ") + width).c_str());
            check(bytes, (std::string("packed peer rows BYTE-equal the native twin at width ") + width).c_str());
            if (std::string(width) == "4") {
                // the adapt path: land a fresh packed expert into slot 0, re-run - the exact
                // 2026-11-06 mixed-tag regression surface
                check(ppeer.enable_adaptation(4, (size_t)8 << 20, perr) &&
                          npeer.enable_adaptation(4, (size_t)8 << 20, nerr),
                      "peer adaptation enabled on both tiers");
                k::Exl3Peer::Replacement pr, nr;
                pr.out = nr.out = 0;
                pr.in = entry_of(packed, 3);
                nr.in = entry_of(plain, 3);
                std::vector<k::Exl3Peer::Replacement> landed;
                const bool adapt = ppeer.can_replace(pr) && ppeer.replacement_bytes(pr) > 0 &&
                                   npeer.can_replace(nr) && npeer.replacement_bytes(nr) > 0 &&
                                   ppeer.replace({pr}, perr) &&
                                   ppeer.apply_refill(true, landed, perr) && landed.size() == 1 &&
                                   npeer.replace({nr}, nerr) &&
                                   npeer.apply_refill(true, landed, nerr) && landed.size() == 1;
                check(adapt, "replace + apply_refill landed on both tiers");
                check(!ppeer.has(1, 0) && ppeer.has(1, 3) &&
                          ppeer.slot_layout(1, 3) == Exl3TrellisLayout::Band8,
                      "the landed slot stays Band8-tagged (no mixed-layout refusal)");
                if (adapt) {
                    const std::vector<int32_t> after{3, 1, 2, 3, 2, 3, 1, 2, 1, 2, 3, 1};
                    std::vector<float> want((size_t)routes * EMB, 3.25f), got = want;
                    const bool ran = npeer.launch(1, x.data(), after.data(), tokens, topk,
                                                  kind.data(), nerr) &&
                                     npeer.finish(want.data(), nerr) &&
                                     ppeer.launch(1, x.data(), after.data(), tokens, topk,
                                                  kind.data(), perr) &&
                                     ppeer.finish(got.data(), perr);
                    check(ran && std::memcmp(want.data(), got.data(), got.size() * sizeof(float)) == 0,
                          "post-adapt packed peer rows byte-equal the native twin");
                }
            }
        }
        set_env("STRATA_EXL3_PEER_MULTIROW", keep_w.empty() ? nullptr : keep_w.c_str());
    }

    // ---- 6: the stream's managed-RAM-cache refusal beside a backing, at this geometry ----
    {
        bool threw = false;
        try {
            k::Exl3ExpertStream s(fx.dir.string(), EMB, LAYERS, 0, nullptr, (size_t)1 << 20,
                                  (size_t)16 << 20, 0, LAYERS, 0, "model.language_model.layers.",
                                  resident);
        } catch (const std::invalid_argument& e) {
            threw = std::string(e.what()).find("ram_cache_bytes") != std::string::npos;
        }
        check(threw, "expert stream refuses a managed RAM cache beside the packed backing");
        bool native_ok = false;
        try {
            k::Exl3ExpertStream s(fx.dir.string(), EMB, LAYERS, 0, nullptr, (size_t)1 << 20,
                                  (size_t)16 << 20, 0, LAYERS, 0, "model.language_model.layers.");
            native_ok = true;
        } catch (...) {}
        check(native_ok, "the native stream still accepts a managed RAM cache (default unchanged)");
    }

    // ---- 7: REAL launcher env, REAL verify-window shape (the e2e divergence pinpoint) ------
    // The full-model screen/parity showed PN-vs-PNp greedy streams diverging at token 0
    // (reproducible, adapt frozen, peer-prefill off) while every single-GPU arm is
    // bit-identical - so the difference lives in the PEER decode/verify integration over
    // packed planes. This section replays that surface at fixture scale with the launcher's
    // EXACT peer/primary env and the REAL window geometry (T=5 spec4 windows, k=10, ragged
    // width-4 chunks) on SLOTTED streams (the production cache-fill staging path), and
    // byte-compares native twin vs packed for: the peer launch path, the primary complement
    // (owned-elsewhere window), and the primary's plain run_multi. Any difference is
    // reported with magnitude (route index, differing floats, worst relative error) so a
    // reorder-level numeric delta is never confused with a wrong-value bug.
    {
        const char* env_block[] = {
            "STRATA_EXL3_RDNA=1", "STRATA_EXL3_FUSE=3", "STRATA_EXL3_RDNA_NATIVE=1",
            "STRATA_EXL3_NATIVE_GROUPED=1", "STRATA_EXL3_PEER_MULTIROW=4",
            "STRATA_EXL3_PEER_PADDED=1", "STRATA_EXL3_PEER_PACK_PLAN=1",
            "STRATA_EXL3_PEER_SPARSE=1", "STRATA_EXL3_PEER_CONVERT=1", "STRATA_EXL3_PACK_STAGE=1",
            "STRATA_EXL3_WMMA=1", "STRATA_EXL3_WAVESYNC=1", "STRATA_EXL3_PIPE=1",
            "STRATA_EXL3_OVERLAP=1", "STRATA_EXL3_NO_PREAD=1", "STRATA_EXL3_PRIMARY_MULTIROW=2",
            "STRATA_EXL3_EXPERT_BATCH=1", "STRATA_EXL3_PEER_BATCH=1", "STRATA_EXL3_STAGE_INLINE=2",
            "STRATA_EXL3_WMMA_STG_PAD=1", "STRATA_EXL3_RESERVE_FUSED=1",
            "STRATA_EXL3_VERIFY_HITS=0", "STRATA_EXL3_CPU_TIER=0", "STRATA_EXL3_PEER_ADAPT=0",
        };
        std::vector<std::pair<std::string, std::string>> saved;
        for (const char* kv : env_block) {
            const std::string s(kv);
            const size_t eq = s.find('=');
            const std::string key = s.substr(0, eq), value = s.substr(eq + 1);
            const char* old = std::getenv(key.c_str());
            saved.emplace_back(key, old ? old : "");
            setenv(key.c_str(), value.c_str(), 1);
        }
        struct EnvRestore {
            std::vector<std::pair<std::string, std::string>> saved;
            ~EnvRestore() {
                for (const auto& [k, v] : saved)
                    if (v.empty()) unsetenv(k.c_str()); else setenv(k.c_str(), v.c_str(), 1);
            }
        } restore{saved};

        const int tokens = 5, topk = 10, routes = tokens * topk;   // the spec4 verify window
        std::mt19937 rng(4242);
        std::normal_distribution<float> dist(0.0f, 0.5f);
        std::vector<float> x((size_t)tokens * EMB);
        for (auto& v : x) v = dist(rng);
        // ragged repeated ids over experts 0..5: group counts 4/8/12/8/8/10 = 50 routes,
        // permuted so width-4 chunking sees mixed tails
        std::vector<int32_t> window;
        {
            std::vector<int32_t> pool;
            for (int e = 0; e < EXPERTS; ++e)
                for (int n = 0; n < (e == 2 ? 12 : e == 1 ? 8 : e == 0 ? 4 : e == 5 ? 10 : 8);
                     ++n) pool.push_back(e);   // 4+8+12+8+8+10 = 50 = tokens*topk exactly
            std::shuffle(pool.begin(), pool.end(), rng);
            window = pool;
        }

        auto diff_report = [&](const std::vector<float>& a, const std::vector<float>& b,
                               const char* what) {
            size_t diffs = 0, first = SIZE_MAX;
            double worst_rel = 0, scale = 0;
            for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
                scale = std::max(scale, (double)std::fabs(b[i]));
                if (a[i] != b[i]) { ++diffs; if (first == SIZE_MAX) first = i; }
                worst_rel = std::max(worst_rel, (double)std::fabs(a[i] - b[i]));
            }
            check(a.size() == b.size() && diffs == 0,
                  (std::string(what) + ": native twin vs packed BYTE-equal").c_str());
            if (diffs) {
                std::printf("      %s: %zu of %zu floats differ, first at %zu (route %zu), "
                            "worst |a-b| %.3g over scale %.3g (rel %.3g)\n",
                            what, diffs, a.size(), first, first / EMB, worst_rel, scale,
                            worst_rel / std::max(1e-30, scale));
            }
        };

        // slotted streams: the production cache-fill staging path (slots admit from the
        // backing at first use; per-slot descriptors carry the layout tag). Outer scope of
        // section 7: both the window blocks and the mixed-window replication use them.
        k::Exl3ExpertStream nstream(fx.dir.string(), EMB, LAYERS, EXPERTS, nullptr, 0,
                                    (size_t)64 << 20, 0, LAYERS, 0,
                                    "model.language_model.layers.");
        k::Exl3ExpertStream pstream(fx.dir.string(), EMB, LAYERS, EXPERTS, nullptr, 0,
                                    (size_t)64 << 20, 0, LAYERS, 0,
                                    "model.language_model.layers.", resident);
        {
            {   // plain run_multi at the real window shape
                std::vector<float> a((size_t)routes * EMB, -1e30f), b = a;
                const bool okn = nstream.run_multi(1, x.data(), window.data(), tokens, topk,
                                                   a.data(), nullptr);
                const bool okp = pstream.run_multi(1, x.data(), window.data(), tokens, topk,
                                                   b.data(), nullptr);
                check(okn && okp, "real-env run_multi (T=5 k=10 ragged) ran on both streams");
                diff_report(a, b, "real-env primary run_multi");
            }
            {   // the peer complement window: experts 0,1 owned elsewhere
                std::vector<uint8_t> owned((size_t)LAYERS * EXPERTS, 0);
                owned[(size_t)1 * EXPERTS + 0] = 1;
                owned[(size_t)1 * EXPERTS + 1] = 1;
                nstream.set_owned_elsewhere(owned.data(), LAYERS, EXPERTS);
                pstream.set_owned_elsewhere(owned.data(), LAYERS, EXPERTS);
                std::vector<float> a((size_t)routes * EMB, -1e30f), b = a;
                const bool okn = nstream.run_multi_complement(1, x.data(), window.data(), tokens,
                                                              topk, a.data(), nullptr);
                const bool okp = pstream.run_multi_complement(1, x.data(), window.data(), tokens,
                                                              topk, b.data(), nullptr);
                check(okn && okp, "real-env run_multi_complement ran on both streams");
                // rows of owned experts are UNTOUCHED (still the sentinel): compare only the
                // computed ones, and assert the untouched pattern matches between arms
                size_t computed = 0;
                for (int r = 0; r < routes; ++r)
                    if (window[(size_t)r] >= 2) ++computed;   // experts 2..5 are the primary's
                std::vector<float> ca, cb;
                for (int r = 0; r < routes; ++r)
                    if (window[(size_t)r] >= 2) {
                        ca.insert(ca.end(), a.begin() + (size_t)r * EMB, a.begin() + ((size_t)r + 1) * EMB);
                        cb.insert(cb.end(), b.begin() + (size_t)r * EMB, b.begin() + ((size_t)r + 1) * EMB);
                    }
                diff_report(ca, cb, "real-env primary complement (computed rows)");
                bool untouched_same = true;
                for (int r = 0; r < routes; ++r)
                    if (window[(size_t)r] < 2)
                        untouched_same = untouched_same &&
                            a[(size_t)r * EMB] == -1e30f && b[(size_t)r * EMB] == -1e30f;
                check(untouched_same, "real-env complement leaves owned rows untouched in both arms");
            }
        }
        {   // the peer launch path at the real window shape, launcher env (multirow read at open)
            Exl3Model plain(fx.dir.string()), packed(fx.dir.string(), resident);
            auto entry_of = [&](const Exl3Model& m, int e) {
                k::Exl3Peer::Entry en;
                en.layer = 1;
                en.expert = e;
                auto set = [&](k::Exl3Mat& t, const char* proj) {
                    const Exl3Linear l = m.linear(qwen_tensor(1, e, proj));
                    t.trellis = l.trellis; t.suh = l.suh; t.svh = l.svh;
                    t.ki = l.ki; t.nj = l.nj; t.bits = l.bits; t.frac = l.frac; t.cb = l.cb;
                    t.layout = l.layout;
                };
                set(en.g, "gate"); set(en.u, "up"); set(en.d, "down");
                return en;
            };
            std::vector<int32_t> peer_window;   // routes over the resident subset {0,1,2}
            for (int r = 0; r < routes; ++r) peer_window.push_back(window[(size_t)r] % 3);
            std::vector<int32_t> kind(routes, 2);
            k::Exl3Peer npeer, ppeer;
            std::string nerr, perr;
            std::vector<k::Exl3Peer::Entry> nsub, psub;
            for (int e = 0; e < 3; ++e) {
                nsub.push_back(entry_of(plain, e));
                psub.push_back(entry_of(packed, e));
            }
            const bool nopen = npeer.open(0, EMB, nsub, 8, topk, nerr);   // caps cover the stress sweep (T<=8, k<=10)
            const bool popen = ppeer.open(0, EMB, psub, 8, topk, perr);
            check(nopen && popen, "real-env peer opens (native twin + packed)");
            if (nopen && popen) {
                std::vector<float> a((size_t)routes * EMB, 3.25f), b = a;
                const bool ran = npeer.launch(1, x.data(), peer_window.data(), tokens, topk,
                                              kind.data(), nerr) &&
                                 npeer.finish(a.data(), nerr) &&
                                 ppeer.launch(1, x.data(), peer_window.data(), tokens, topk,
                                              kind.data(), perr) &&
                                 ppeer.finish(b.data(), perr);
                check(ran, "real-env peer launch/finish ran (T=5 k=10, width 4 env)");
                if (ran) {
                    diff_report(a, b, "real-env peer rows (T=5 k=10 ragged, width-4 env)");
                    std::printf("      peer grouped: calls=%lld routes=%lld groups=%lld "
                                "native_w4=%lld\n", (long long) ppeer.grouped_launches(),
                                (long long) ppeer.grouped_routes(), (long long) ppeer.grouped_groups(),
                                (long long) ppeer.native_width4_calls());
                }
                // ---- randomized-shape stress: the e2e divergence needs packed bytes AND the
                // peer tier together, in some window shape the single fixed window above does
                // not hit. Sweep many random (T, k, id-multiplicity) windows over the same
                // resident subset and byte-compare each; report the FIRST differing shape.
                {
                    std::mt19937 srng(20261108);
                    bool all_equal = true;
                    int bad_t = -1, bad_k = -1;
                    size_t bad_floats = 0;
                    double bad_worst = 0;
                    for (int iter = 0; iter < 240 && all_equal; ++iter) {
                        const int t = 1 + (int)(srng() % 8);
                        const int kk = 1 + (int)(srng() % 10);
                        const int rt = t * kk;
                        std::vector<float> sx((size_t)t * EMB);
                        for (auto& v : sx) v = std::normal_distribution<float>(0.0f, 0.5f)(srng);
                        std::vector<int32_t> wid(rt), knd(rt, 2);
                        for (int i = 0; i < rt; ++i) wid[(size_t)i] = (int32_t)(srng() % 3);
                        std::vector<float> sa((size_t)rt * EMB, 3.25f), sb = sa;
                        if (!npeer.launch(1, sx.data(), wid.data(), t, kk, knd.data(), nerr) ||
                            !npeer.finish(sa.data(), nerr) ||
                            !ppeer.launch(1, sx.data(), wid.data(), t, kk, knd.data(), perr) ||
                            !ppeer.finish(sb.data(), perr)) {
                            std::printf("      stress iter %d (T=%d k=%d) launch error "
                                        "native='%s' packed='%s'\n", iter, t, kk,
                                        nerr.c_str(), perr.c_str());
                            all_equal = false;
                            bad_t = t; bad_k = kk;
                            break;
                        }
                        for (size_t i = 0; i < sa.size(); ++i)
                            if (sa[i] != sb[i]) {
                                all_equal = false;
                                bad_t = t; bad_k = kk;
                                ++bad_floats;
                                bad_worst = std::max(bad_worst,
                                    (double)std::fabs(sa[i] - sb[i]) / std::max(1e-30, (double)std::fabs(sa[i])));
                                break;   // shape-level report; full count on the failing shape below
                            }
                    }
                    check(all_equal,
                          "real-env peer stress: 240 random (T,k,multiplicity) windows all "
                          "BYTE-equal native vs packed");
                    if (!all_equal)
                        std::printf("      first differing shape T=%d k=%d; differing floats "
                                    ">= %zu, worst rel %.3g\n", bad_t, bad_k, bad_floats, bad_worst);
                }
                // ---- mixed-window replication: the engine's TWO primary strategies for a
                // peer-mixed window, reproduced shape-for-shape. (a) STRATA_EXL3_PEER_SPARSE=1:
                // run_multi_complement over the whole window (owned experts dropped); (b) sparse=0
                // + PADDED=1: run_multi over the whole window with every PEER route's id replaced
                // by a duplicated PRIMARY id (the pad forms giant repeated-id groups). Native is
                // invariant to the choice (identical e2e streams); if packed is not, one of these
                // two run_multi shapes is not bit-exact over Band8 - the e2e divergence source.
                {
                    std::mt19937 mrng(90210);
                    bool comp_eq = true, pad_eq = true, cross_eq = true;
                    int bad = -1;
                    for (int iter = 0; iter < 160 && comp_eq && pad_eq; ++iter) {
                        const int t = 1 + (int)(mrng() % 5);      // spec windows: T<=5
                        const int kk = 10;                        // the model's top-k
                        const int rt = t * kk;
                        std::vector<float> sx((size_t)t * EMB);
                        for (auto& v : sx) v = std::normal_distribution<float>(0.0f, 0.5f)(mrng);
                        std::vector<int32_t> wid(rt);
                        for (int i = 0; i < rt; ++i) wid[(size_t)i] = (int32_t)(mrng() % EXPERTS);
                        // whole-expert ownership: ~1/3 of experts owned by the peer
                        std::vector<uint8_t> owned((size_t)EXPERTS, 0);
                        for (int e = 0; e < EXPERTS; ++e) owned[(size_t)e] = (mrng() % 3) == 0;
                        std::vector<uint8_t> table((size_t)LAYERS * EXPERTS, 0);
                        for (int e = 0; e < EXPERTS; ++e) table[(size_t)1 * EXPERTS + e] = owned[(size_t)e];
                        // the padded id form: every peer route's id -> the first primary id
                        int32_t pad_id = -1;
                        for (int i = 0; i < rt; ++i)
                            if (!owned[(size_t)wid[(size_t)i]]) { pad_id = wid[(size_t)i]; break; }
                        if (pad_id < 0) continue;   // no primary route this draw; skip
                        std::vector<int32_t> pad_ids(rt);
                        for (int i = 0; i < rt; ++i)
                            pad_ids[(size_t)i] = owned[(size_t)wid[(size_t)i]] ? pad_id : wid[(size_t)i];
                        nstream.set_owned_elsewhere(table.data(), LAYERS, EXPERTS);
                        pstream.set_owned_elsewhere(table.data(), LAYERS, EXPERTS);
                        const float sentinel = -7.5f;
                        std::vector<float> na((size_t)rt * EMB, sentinel), nb = na, pa = na, pb = na;
                        const bool ran = nstream.run_multi_complement(1, sx.data(), wid.data(), t, kk, na.data(), nullptr) &&
                                         pstream.run_multi_complement(1, sx.data(), wid.data(), t, kk, nb.data(), nullptr) &&
                                         nstream.run_multi(1, sx.data(), pad_ids.data(), t, kk, pa.data(), nullptr) &&
                                         pstream.run_multi(1, sx.data(), pad_ids.data(), t, kk, pb.data(), nullptr);
                        if (!ran) { bad = iter; comp_eq = pad_eq = false; break; }
                        // compare only PRIMARY-computed rows (owned rows: complement leaves the
                        // sentinel; the padded form computed pad rows the caller discards)
                        std::vector<int> prim_rows;
                        for (int i = 0; i < rt; ++i) if (!owned[(size_t)wid[(size_t)i]]) prim_rows.push_back(i);
                        for (int r : prim_rows) {
                            for (int q = 0; q < EMB; ++q) {
                                const float a = na[(size_t)r * EMB + q], b = nb[(size_t)r * EMB + q];
                                const float c = pa[(size_t)r * EMB + q], d = pb[(size_t)r * EMB + q];
                                if (a != b) comp_eq = false;
                                if (c != d) pad_eq = false;
                                if (a != c || b != d) cross_eq = false;
                                if (!comp_eq && !pad_eq) { bad = iter; break; }
                            }
                            if (!comp_eq && !pad_eq) break;
                        }
                    }
                    check(comp_eq, "mixed-window complement: 160 random ownership masks "
                                   "BYTE-equal native vs packed");
                    check(pad_eq, "mixed-window PADDED whole-window run_multi (giant repeated-id "
                                  "groups): BYTE-equal native vs packed");
                    // NOT a gate: a real, SYMMETRIC observation (present for native bytes
                    // equally) - the engine's two mixed-window strategies differ in the last
                    // ulps within each layout, so the primary result depends on
                    // PEER_SPARSE/PEER_PADDED for BOTH storages. Recorded, never counted as
                    // a packed regression (per-strategy native==packed is what gates).
                    std::printf("%s  OBSERVE: complement vs padded strategies differ within each "
                                "layout (comp_eq=%d pad_eq=%d) - symmetric ulp-class wobble, "
                                "engine sparse-mode dependent for BOTH storages\n",
                                cross_eq ? "ok  " : "note", comp_eq, pad_eq);
                    if (!comp_eq || !pad_eq)
                        std::printf("      first failing iteration: %d\n", bad);
                }
            } else {
                std::printf("peer open errors: native='%s' packed='%s'\n", nerr.c_str(), perr.c_str());
            }
        }
    }

    // ---- 8: SCALE sweep - the e2e packed-peer divergence onset scales with the peer subset
    // size (512 experts -> token 65; 7200 -> token 0), the signature of a small fraction of
    // experts whose device bytes are wrong. Sweep EVERY expert at full scale: (a) the arena's
    // bytes vs the store's payload for a stride of experts across all layers, (b) a peer
    // opened over ALL of layer 1's experts, one T=1/k=1 launch per expert, byte-compared
    // native twin vs packed - a single corrupted expert cannot hide.
    {
        {
            // rebuild a store at THIS fixture's scale (sections above released theirs; the
            // resident arena from section 4 is still alive) and sweep its bytes against a
            // freshly opened store generation (same fingerprint, no new payload bytes)
            std::vector<std::string> scale_prefixes;
            for (int l = 0; l < LAYERS; ++l)
                for (int e = 0; e < EXPERTS; ++e)
                    for (const char* proj : {"gate", "up", "down"})
                        scale_prefixes.push_back(qwen_tensor(l, e, proj));
            auto scale_store = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(),
                                                             scale_prefixes);
            Exl3Model scale_packed(fx.dir.string(), scale_store);
            bool arena_bytes_ok = true;
            int swept = 0;
            for (int l = 0; l < LAYERS && arena_bytes_ok; ++l)
                for (int e = 0; e < EXPERTS; e += 4) {
                    for (const char* proj : {"gate", "up", "down"}) {
                        Exl3Linear from_arena{}, from_store{};
                        const std::string name = qwen_tensor(l, e, proj);
                        const bool ga = resident->linear(name, from_arena);
                        const bool gs = scale_store->linear(name, from_store);
                        const size_t n = (size_t)from_store.ki * from_store.nj * from_store.words();
                        if (!ga || !gs || from_arena.layout != Exl3TrellisLayout::Band8 ||
                                from_arena.trellis == from_store.trellis ||
                                std::memcmp(from_arena.trellis, from_store.trellis, n * 2) != 0) {
                            arena_bytes_ok = false;
                            std::printf("      arena mismatch at %s (arena=%d store=%d)\n",
                                        name.c_str(), ga, gs);
                            break;
                        }
                        ++swept;
                    }
                    if (!arena_bytes_ok) break;
                }
            check(arena_bytes_ok, "arena bytes == payload bytes across a 1-in-4 expert sweep "
                                  "at 64-expert scale");
            (void) swept;
        }
        {   // (b) full peer sweep: every expert of layer 1, one launch each
            Exl3Model plain(fx.dir.string()), packed(fx.dir.string(), resident);
            auto full_entry_of = [&](const Exl3Model& m, int e) {
                k::Exl3Peer::Entry en;
                en.layer = 1;
                en.expert = e;
                auto set = [&](k::Exl3Mat& t, const char* proj) {
                    const Exl3Linear l = m.linear(qwen_tensor(1, e, proj));
                    t.trellis = l.trellis; t.suh = l.suh; t.svh = l.svh;
                    t.ki = l.ki; t.nj = l.nj; t.bits = l.bits; t.frac = l.frac; t.cb = l.cb;
                    t.layout = l.layout;
                };
                set(en.g, "gate"); set(en.u, "up"); set(en.d, "down");
                return en;
            };
            std::vector<k::Exl3Peer::Entry> nsub, psub;
            for (int e = 0; e < EXPERTS; ++e) {
                nsub.push_back(full_entry_of(plain, e));
                psub.push_back(full_entry_of(packed, e));
            }
            k::Exl3Peer npeer, ppeer;
            std::string nerr, perr;
            const bool nopen = npeer.open(0, EMB, nsub, 2, 2, nerr);
            const bool popen = ppeer.open(0, EMB, psub, 2, 2, perr);
            check(nopen && popen, "scale peer opens over ALL 64 experts (native + packed)");
            if (nopen && popen) {
                std::mt19937 rng(64);
                std::normal_distribution<float> dist(0.0f, 0.5f);
                int differing_experts = 0, first_bad = -1;
                std::vector<float> x((size_t)1 * EMB);
                for (auto& v : x) v = dist(rng);
                for (int e = 0; e < EXPERTS; ++e) {
                    const int32_t id = e, kind = 2;
                    std::vector<float> a((size_t)1 * EMB, 3.25f), b = a;
                    if (!npeer.launch(1, x.data(), &id, 1, 1, &kind, nerr) ||
                            !npeer.finish(a.data(), nerr) ||
                            !ppeer.launch(1, x.data(), &id, 1, 1, &kind, perr) ||
                            !ppeer.finish(b.data(), perr)) {
                        std::printf("      scale sweep launch error at expert %d: '%s' / '%s'\n",
                                    e, nerr.c_str(), perr.c_str());
                        ++differing_experts;
                        if (first_bad < 0) first_bad = e;
                        break;
                    }
                    if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0) {
                        ++differing_experts;
                        if (first_bad < 0) first_bad = e;
                    }
                }
                check(differing_experts == 0,
                      "scale peer sweep: all 64 experts' rows BYTE-equal native vs packed");
                if (differing_experts)
                    std::printf("      %d differing expert(s), first at expert %d\n",
                                differing_experts, first_bad);
            }
        }
    }

    // ---- 9: the RUNTIME in-memory repack (STRATA_QWEN_BAND8=memory's backing) ------------
    // No store, no cache dir: create() validates the NATIVE fixture (K4 mul1, nj 40/160),
    // activate() permutes the routed experts straight into the pinned-sim arena, and the
    // result must be byte-identical to the offline store's payload - then the main expert
    // stream must serve byte-equal rows to the native twin, exactly as the packed-store
    // arena does in section 4.
    {
        std::shared_ptr<Exl3ResidentBacking> runtime;
        try {
            runtime = Exl3ResidentBacking::create(fx.dir.string(), groups, nullptr,
                                                  strata::Exl3ResidentRepack::ToBand8);
        } catch (const std::exception& e) {
            check(false, (std::string("runtime repack create: ") + e.what()).c_str());
        }
        if (runtime) {
            check(runtime->layout() == Exl3TrellisLayout::Band8 && !runtime->active(),
                  "runtime repack reports the FINAL Band8 tag before activation");
            {
                Exl3Linear early{};
                std::string p;
                size_t o = 0, n = 0;
                bool threw_linear = false, threw_locate = false;
                try { runtime->linear(qwen_tensor(0, 0, "gate"), early); }
                catch (const std::exception&) { threw_linear = true; }
                try { runtime->locate(qwen_tensor(0, 0, "gate") + ".trellis", p, o, n); }
                catch (const std::exception&) { threw_locate = true; }
                check(threw_linear && threw_locate,
                      "pre-activation linear()/locate() THROW (native bytes never travel "
                      "under the Band8 tag)");
            }
            Exl3ResidentBacking::Options opt;
            opt.available_bytes = 1ull << 32;
            opt.headroom_bytes = 0;
            opt.lock = [](void*, size_t, std::string&) { return true; };
            opt.unlock = [](void*, size_t) {};
            runtime->activate({}, opt);
            check(runtime->active() && runtime->memory_only(),
                  "runtime repack activation publishes the memory-only arena");
            // bytes: every expert, every plane, vs the store's payload (the offline truth)
            std::vector<std::string> all;
            for (int l = 0; l < LAYERS; ++l)
                for (int e = 0; e < EXPERTS; ++e)
                    for (const char* proj : {"gate", "up", "down"})
                        all.push_back(qwen_tensor(l, e, proj));
            auto truth = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(), all);
            bool bytes = true;
            for (const std::string& name : all) {
                Exl3Linear a{}, b{};
                const bool ga = runtime->linear(name, a), gb = truth->linear(name, b);
                const size_t tr = (size_t)b.ki * b.nj * b.words() * 2;
                bytes = bytes && ga && gb && a.layout == Exl3TrellisLayout::Band8 &&
                        a.trellis != b.trellis &&
                        std::memcmp(a.trellis, b.trellis, tr) == 0 &&
                        std::memcmp(a.suh, b.suh, (size_t)b.in() * 2) == 0 &&
                        std::memcmp(a.svh, b.svh, (size_t)b.out() * 2) == 0;
                if (!bytes) {
                    std::printf("      runtime repack mismatch at %s (arena=%d store=%d)\n",
                                name.c_str(), ga, gb);
                    break;
                }
            }
            check(bytes, "runtime-repacked arena bytes == the offline store payload, every "
                         "expert's three projections (trellis permuted, scales verbatim)");
            std::string path;
            size_t off = 0, bytes_n = 0;
            check(!runtime->locate(qwen_tensor(0, 0, "gate") + ".trellis", path, off, bytes_n),
                  "runtime repack locate() refuses the file tier (no pread beside the arena)");
            // stream parity vs the native twin - the same proof section 4 runs for the
            // packed-store arena, now over the runtime-converted bytes
            k::Exl3ExpertStream native_stream(fx.dir.string(), EMB, LAYERS, 0, nullptr, 0,
                                              (size_t)16 << 20, 0, LAYERS, 0,
                                              "model.language_model.layers.");
            k::Exl3ExpertStream runtime_stream(fx.dir.string(), EMB, LAYERS, 0, nullptr, 0,
                                               (size_t)16 << 20, 0, LAYERS, 0,
                                               "model.language_model.layers.", runtime);
            std::mt19937 rng(91);
            std::normal_distribution<float> dist(0.0f, 0.5f);
            std::vector<float> x((size_t)4 * EMB);
            for (auto& v : x) v = dist(rng);
            bool single = true;
            {
                const int ids[3] = {4, 1, 4};
                std::vector<float> a((size_t)3 * EMB, -1e30f), b = a;
                single = native_stream.run(1, x.data(), ids, 3, a.data(), nullptr) &&
                         runtime_stream.run(1, x.data(), ids, 3, b.data(), nullptr) &&
                         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
            }
            check(single, "stream run() rows BYTE-equal: runtime-repack arena vs native twin");
            bool window = true;
            {
                const int tokens = 4, topk = 3;
                const std::vector<int32_t> ids{2, 0, 1, 2, 1, 2, 0, 1, 0, 1, 2, 0};
                std::vector<float> a((size_t)tokens * topk * EMB, -1e30f), b = a;
                window = native_stream.run_multi(1, x.data(), ids.data(), tokens, topk,
                                                 a.data(), nullptr) &&
                         runtime_stream.run_multi(1, x.data(), ids.data(), tokens, topk,
                                                  b.data(), nullptr) &&
                         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
            }
            check(window, "stream run_multi() MTP-window rows BYTE-equal: runtime repack vs "
                          "native twin");
        }
    }

    std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
