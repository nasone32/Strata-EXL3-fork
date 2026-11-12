// src/artifact/exl3_band8_build.cpp - CPU-only band-8 expert cache builder CLI.
//
// Builds (or reuses) the SAME cache generation the engine's STRATA_MIMO_BAND8 path opens
// (strata::Exl3Band8Store::build_or_open), WITHOUT a GPU: no CUDA/HIP context, no dense pack,
// no drafter - so an 80 GiB-class pack job can run beside idle GPUs without holding their
// memory.  The engine later opens the finished generation as-is (fingerprint match, no
// payload scan); the native source is never modified and nothing is auto-written anywhere
// until this binary is invoked with an explicit --cache root.
//
// CPU-only by construction: mimo_geometry.hpp (inline config parse) + mimo_pack.hpp's
// mimo_expert_tensor (the exact validated expert-name tuple the engine uses) + the store.
// Expected-bytes estimate: metadata-only, through Exl3Model::locate over the requested
// prefixes (the source shards are mapped, never copied - no native RAM copy is made).
//
// Usage: exl3_band8_build --model <dir> --cache <root> [--progress N]
//   --model    a MiMo v2 EXL3 model dir (config.json + shards + index.json)
//   --cache    the EXPLICIT cache root (engine: STRATA_EXL3_BAND8_CACHE) - the payload is
//              about the routed experts' size (MiMo ~79.5 GiB); pick a drive that fits it
//   --progress stderr progress line every N projection entries (default 256)
#include "strata/artifact/exl3_band8_store.hpp"
#include "strata/artifact/exl3_model.hpp"
#include "strata/core/mimo_geometry.hpp"

// mimo_expert_tensor's exact tuple, spelled here because its definition lives in
// src/artifact/mimo_pack.cpp inside the GPU strata_engine library (this binary deliberately
// links no GPU code).  Keep byte-identical with that source:
//   "model.layers." + layer + ".mlp.experts." + expert + "." + proj + "_proj"
// The engine's request list uses proj in {gate, up, down}, exactly as below.
namespace {
std::string expert_prefix_of(int layer, int expert, const char* proj) {
    return "model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(expert) + "." + proj +
           "_proj";
}
// The Qwen family's exact tuple - byte-identical with the STRATA_QWEN_BAND8 request list the
// engine builds in generate.cpp (layer-major, expert-minor, gate/up/down) and with the expert
// stream's own Exl3ExpertStream::Impl::expert_prefix.  The list ORDER is part of the cache
// fingerprint: the two loops must not drift.
std::string qwen_expert_prefix_of(int layer, int expert, const char* proj) {
    return "model.language_model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(expert) +
           "." + proj + "_proj";
}
}  // namespace

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

std::string read_text(const std::string& path, std::string& error) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        error = "cannot open " + path;
        return {};
    }
    std::string text;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
    const bool bad = std::ferror(f) != 0;
    std::fclose(f);
    if (bad) {
        error = "cannot read " + path;
        return {};
    }
    return text;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, cache;
    size_t progress = 256;
    bool qwen = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has_next = i + 1 < argc;
        if (a == "--model" && has_next) model = argv[++i];
        else if (a == "--cache" && has_next) cache = argv[++i];
        else if (a == "--qwen") qwen = true;
        else if (a == "--progress" && has_next) {
            progress = (size_t) std::strtoull(argv[++i], nullptr, 10);
            if (progress == 0) progress = 1;
        } else {
            std::fprintf(stderr,
                         "usage: %s --model <dir> --cache <root> [--qwen] [--progress N]\n"
                         "  CPU-only band-8 expert cache builder (the engine's STRATA_MIMO_BAND8 /\n"
                         "  STRATA_QWEN_BAND8 generation; no GPU, no engine context; the payload is\n"
                         "  about the routed experts' size - pick a --cache drive that fits it)\n"
                         "  --qwen  the Qwen EXL3 family: pack model.language_model.layers.<l>.mlp.\n"
                         "          experts.<e>.{gate,up,down}_proj (layer counts are derived from the\n"
                         "          safetensors index; the in-file mtp.layers.0 block stays native)\n",
                         argv[0]);
            return 2;
        }
    }
    (void) 0;
    if (model.empty() || cache.empty()) {
        std::fprintf(stderr, "band8-build: --model and --cache are both required\n");
        return 2;
    }

    // ---- the geometry, CPU-inline (mimo_geometry.hpp) -------------------------------------
    std::string error;
    int64_t moe_layers = 0, n_expert = 0;
    // layer_is_moe[l] marks the MoE layers by their REAL index (dense layers stay false), so the
    // request list below can skip them exactly like the engine's own list does.  Qwen's MoE
    // layers are the dense grid 0..max, so every derived layer is marked.
    std::vector<char> layer_is_moe;
    // `false` keeps the historical MiMo behavior byte-identical (config-driven, sparse MoE
    // layer map); `--qwen` derives the counts from the safetensors index itself.
    std::function<std::string(int, int, const char*)> expert_name = expert_prefix_of;
    if (qwen) {
        const std::string index_json = read_text(model + "/model.safetensors.index.json", error);
        if (index_json.empty()) {
            std::fprintf(stderr, "band8-build: %s: %s\n", (model + "/model.safetensors.index.json").c_str(),
                         error.empty() ? "cannot read the index" : error.c_str());
            return 1;
        }
        // Every model.language_model.layers.<l>.mlp.experts.<e>.<proj>_proj entry implies
        // the (l, e) expert exists; the locate pass below then verifies each generated
        // name individually, so a malformed index cannot pack a wrong list.  The in-file
        // mtp.layers.0 block never matches this prefix and stays native.
        static const std::string key = "\"model.language_model.layers.";
        int64_t max_l = -1, max_e = -1;
        for (size_t at = index_json.find(key); at != std::string::npos;
             at = index_json.find(key, at + 1)) {
            const char* c = index_json.c_str() + at + key.size();
            char* end = nullptr;
            const long long l = std::strtoll(c, &end, 10);
            if (end == c || *end != '.' || std::strncmp(end, ".mlp.experts.", 13) != 0) continue;
            char* end2 = nullptr;
            const long long e = std::strtoll(end + 13, &end2, 10);
            if (end2 == end + 13 || *end2 != '.') continue;
            max_l = std::max(max_l, (int64_t) l);
            max_e = std::max(max_e, (int64_t) e);
        }
        if (max_l < 0 || max_e < 0) {
            std::fprintf(stderr, "band8-build: %s declares no model.language_model.layers.*.mlp.experts.* "
                                 "tensors - is this a Qwen EXL3 export?\n",
                         (model + "/model.safetensors.index.json").c_str());
            return 1;
        }
        moe_layers = max_l + 1;
        n_expert = max_e + 1;
        layer_is_moe.assign((size_t) moe_layers, 1);
        expert_name = qwen_expert_prefix_of;
        std::fprintf(stderr, "band8-build: qwen family: %lld layer(s) x %lld experts from the index\n",
                     (long long) moe_layers, (long long) n_expert);
    } else {
        const std::string config_path = model + "/config.json";
        if (!strata::core::mimo_detect_config(config_path, &error)) {
            std::fprintf(stderr, "band8-build: %s is not a MiMo v2 model dir (%s)\n", model.c_str(),
                         error.empty() ? "config.json does not declare model_type mimo_v2" : error.c_str());
            return 1;
        }
        strata::core::MimoGeometry g;
        try {
            g = strata::core::mimo_geometry_from_config(read_text(config_path, error));
        } catch (const std::exception& e) {
            std::fprintf(stderr, "band8-build: %s: %s\n", config_path.c_str(), e.what());
            return 1;
        }
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const char moe = g.is_moe(l) ? 1 : 0;
            layer_is_moe.push_back(moe);
            moe_layers += moe;
        }
        n_expert = g.n_expert;
    }
    if (moe_layers == 0 || n_expert <= 0) {
        std::fprintf(stderr, "band8-build: config declares no MoE experts - nothing to pack\n");
        return 1;
    }

    // ---- the SAME explicit request list the engine builds ---------------------------------
    // The loop visits the ACTUAL MoE layer indices, skipping dense layers exactly like the
    // engine's own list (mimo_engine.cu STRATA_MIMO_BAND8 block: `if (!E.g.is_moe(l)) continue;`).
    // MiMo v2's layer 0 is a dense FFN and layers 1..47 are MoE, so a plain 0..count-1 loop both
    // requested layer 0 (no expert tensors there - a hard locate failure) and dropped layer 47;
    // and any list that differs from the engine's changes the cache fingerprint, which would
    // build a NEW ~79.5 GiB generation instead of reusing the one the engine opens.
    std::vector<std::string> prefixes;
    prefixes.reserve((size_t) moe_layers * (size_t) n_expert * 3);
    for (size_t l = 0; l < layer_is_moe.size(); ++l) {
        if (!layer_is_moe[l]) continue;
        for (int64_t e = 0; e < n_expert; ++e)
            for (const char* proj : {"gate", "up", "down"})
                prefixes.push_back(expert_name((int) l, (int) e, proj));
    }
    std::fprintf(stderr, "band8-build: %lld MoE layer(s) x %lld experts x 3 projections = %zu prefix "
                         "entries (%zu plane records)\n",
                 (long long) moe_layers, (long long) n_expert, prefixes.size(),
                 prefixes.size() * 3);

    // ---- the expected payload size, metadata-only (mapped shards, no native copy) ----------
    uint64_t expected = 0;
    {
        strata::Exl3Model m(model);
        std::string path;
        size_t off = 0, bytes = 0;
        bool missing = false;
        for (const std::string& p : prefixes)
            for (const char* plane : {".trellis", ".suh", ".svh"}) {
                if (!m.locate(p + plane, path, off, bytes)) {
                    std::fprintf(stderr, "band8-build: %s%s is not in the model index\n", p.c_str(), plane);
                    missing = true;
                    break;
                }
                expected += bytes;
            }
        if (missing) return 1;
        std::fprintf(stderr, "band8-build: expected payload ~%.2f GiB (%llu bytes, metadata estimate)\n",
                     (double) expected / 1073741824.0, (unsigned long long) expected);
    }

    // ---- the SAME generation the engine opens ---------------------------------------------
    const auto t0 = Clock::now();
    strata::Exl3Band8BuildOptions opts;
    opts.progress_every_triples = progress;
    std::shared_ptr<const strata::Exl3Band8Store> store;
    try {
        store = strata::Exl3Band8Store::build_or_open(model, cache, prefixes, opts);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "band8-build: %s\n", e.what());
        return 1;
    }
    std::error_code ec;
    const uintmax_t payload = std::filesystem::file_size(store->payload_path(), ec);
    std::fprintf(stderr,
                 "band8-build: OK %zu plane record(s), %.2f GiB payload (expected ~%.2f GiB) at %s "
                 "(fingerprint %016llx) in %.1f s; native source untouched; the engine reuses this "
                 "generation (%s=1 STRATA_EXL3_BAND8_CACHE=%s)\n",
                 store->record_count(), (double) payload / 1073741824.0, (double) expected / 1073741824.0,
                 store->payload_path().c_str(), (unsigned long long) store->fingerprint(),
                 ms_since(t0) / 1000.0, qwen ? "STRATA_QWEN_BAND8" : "STRATA_MIMO_BAND8", cache.c_str());
    return 0;
}
