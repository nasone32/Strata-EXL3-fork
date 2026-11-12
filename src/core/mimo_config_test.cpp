// src/core/mimo_config_test.cpp - MimoGeometry + config.json reader against the real
// benthecarman/MiMo-V2.6-Flash-RL-exl3 config.json (tests/fixtures/mimo-v2-config.json) and
// synthetic edge cases.  Run from the repo root: ./build/mimo_config_test
#include "strata/core/mimo_geometry.hpp"
#include "strata/mimo/peer_reserve.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

static int failures = 0;
#define CHECK(cond)                                                                              \
    do {                                                                                         \
        if (!(cond)) {                                                                           \
            ++failures;                                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                 \
        }                                                                                        \
    } while (0)

static std::string read_file(const std::filesystem::path& p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main() {
    namespace fs = std::filesystem;
    fs::path fixture = fs::path(STRATA_SOURCE_DIR) / "tests/fixtures/mimo-v2-config.json";
    std::string text = read_file(fixture);
    CHECK(!text.empty());

    using strata::core::MimoGeometry;
    MimoGeometry g{};
    try {
        g = strata::core::mimo_geometry_from_config(text);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL parse threw: %s\n", e.what());
        return 1;
    }

    // Scalar geometry verified against the checkpoint's safetensors headers.
    CHECK(g.n_embd == 4096);
    CHECK(g.n_layers == 48);
    CHECK(g.n_vocab == 152576);
    CHECK(g.n_q_head == 64);
    CHECK(g.head_dim_qk == 192);
    CHECK(g.head_dim_v == 128);
    CHECK(g.n_rot == 64);            // int(192 * 0.334)
    CHECK(g.n_head_kv_full == 4);
    CHECK(g.n_head_kv_swa == 8);
    CHECK(g.sliding_window == 128);
    CHECK(g.rope_theta_full == 1e7);
    CHECK(g.rope_theta_swa == 1e4);
    CHECK(g.attention_value_scale == 0.707);
    CHECK(g.n_expert == 256);
    CHECK(g.n_expert_used == 8);
    CHECK(g.moe_intermediate == 2048);
    CHECK(g.dense_intermediate == 16384);
    CHECK(g.n_mtp_layers == 3);
    CHECK(g.norm_topk_prob);
    CHECK(g.sigmoid_scoring);
    CHECK(g.topk_noaux_tc);

    // The layer lists: full attention exactly at {0,5,11,17,23,29,35,41,47} (the config's first
    // group has FOUR swa layers, not five - this is the gotcha the fixture pins down), MoE
    // everywhere except layer 0.
    const int full_layers[] = {0, 5, 11, 17, 23, 29, 35, 41, 47};
    for (int l = 0; l < 48; ++l) {
        bool expect_full = false;
        for (int fl : full_layers) expect_full |= (l == fl);
        CHECK(g.is_full(l) == expect_full);
        CHECK(g.kv_heads(l) == (expect_full ? 4 : 8));
        CHECK(g.rope_theta(l) == (expect_full ? 1e7 : 1e4));
        CHECK(g.is_moe(l) == (l != 0));
    }
    CHECK(g.n_swa_layers() == 39);
    CHECK(g.qk_dim() == 64 * 192);
    CHECK(g.v_dim() == 64 * 128);

    // mimo_detect_config: true for the fixture, false for a foreign config, error text on a
    // mimo_v2-typed but broken document.
    std::string err;
    CHECK(strata::core::mimo_detect_config(fixture.string(), &err));
    fs::path tmp = fs::temp_directory_path() / "strata_mimo_cfg_test.json";
    {
        std::ofstream o(tmp);
        o << "{\"model_type\": \"qwen4_exp_text\"}";
    }
    CHECK(!strata::core::mimo_detect_config(tmp.string(), &err));
    {
        std::ofstream o(tmp);
        o << "{\"model_type\": \"mimo_v2\", \"hybrid_layer_pattern\": [0,1]}";  // length != layers
    }
    CHECK(!strata::core::mimo_detect_config(tmp.string(), &err));
    CHECK(!err.empty());
    std::filesystem::remove(tmp);

    // Parser edges: nested objects, arrays of objects (vision_config), nulls, escapes.
    try {
        auto g2 = strata::core::mimo_geometry_from_config(
            "{\"model_type\":\"mimo_v2\",\"hidden_size\":4096,\"num_hidden_layers\":2,"
            "\"hybrid_layer_pattern\":[0,1],\"moe_layer_freq\":[0,1],"
            "\"vision_config\":{\"depth\":28,\"vit_window_attn_types\":[ -1,0,1]},"
            "\"n_shared_experts\":null,\"rope_parameters\":{\"rope_type\":\"default\"}}");
        CHECK(g2.n_layers == 2);
        CHECK(g2.is_full(0) && !g2.is_full(1));
        CHECK(!g2.is_moe(0) && g2.is_moe(1));
        CHECK(g2.routed_scaling_factor == 1.0);  // absent/null keeps the default
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL nested parse threw: %s\n", e.what());
        ++failures;
    }

    // Wrong model_type must throw (never silently reinterpret another family's config).
    bool threw = false;
    try {
        strata::core::mimo_geometry_from_config("{\"model_type\":\"qwen4_exp\"}");
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);

    // STRATA_MIMO_PEER_RESERVE_MIB parsing (include/strata/mimo/peer_reserve.hpp): a full,
    // nonempty base-10 integer in [1024, 16384].  strtol alone would accept "1024garbage".
    {
        using strata::mimo::parse_peer_reserve_mib;
        long long r = 0;
        std::string perr;
        CHECK(parse_peer_reserve_mib("1024", r, perr) && r == 1024);
        CHECK(parse_peer_reserve_mib("4096", r, perr) && r == 4096);
        CHECK(parse_peer_reserve_mib("16384", r, perr) && r == 16384);
        CHECK(parse_peer_reserve_mib(nullptr, r, perr) == false && !perr.empty());
        perr.clear();
        const char* bad[] = {"",          " ",           "1024garbage", "garbage",
                             "0x400",     "-1024",       "1023",        "16385",
                             "1024 ",     "1e4",         "999999999999999999999999"};
        for (const char* s : bad) {
            perr.clear();
            if (parse_peer_reserve_mib(s, r, perr)) {
                std::fprintf(stderr, "FAIL peer reserve: accepted %s as %lld\n", s, r);
                ++failures;
            } else if (perr.empty()) {
                std::fprintf(stderr, "FAIL peer reserve: rejected %s without an error\n", s);
                ++failures;
            }
        }
    }

    if (failures) {
        std::fprintf(stderr, "mimo_config_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("mimo_config_test: all checks passed\n");
    return 0;
}
