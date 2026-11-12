// include/strata/core/mimo_geometry.hpp - Xiaomi MiMo v2 geometry, read from an EXL3 model dir's
// config.json.
//
// MiMo v2 is a different beast from the qwen4exp ModelGeometry: per-layer attention kinds come
// as an explicit list (full / sliding-window, with different kv-head counts, rope thetas and
// sink biases per kind), layer 0 has a dense FFN while layers 1..N-1 are MoE, and the EXL3
// checkpoint's own tensors (k_proj width, attention_sink_bias presence) are the authority the
// parsed config is validated against at load.  notes/MIMO-V2-PORT.md is the full contract.
//
// Dependency-free on purpose: included by the loader (C++), the engine driver (.cu) and tests.
// The JSON reader below is deliberately small - it builds a value tree for exactly the keys
// MiMo's config.json needs; no external JSON dependency enters the tree.
#pragma once

#include "strata/artifact/model_family.hpp"

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::core {

/// One attention kind per layer, as MiMo's hybrid_layer_pattern declares it (0 = full, 1 = SWA).
/// Values beyond the config come from the checkpoint at load (sink presence is per-tensor).
enum class MiMoLayerKind { Full = 0, Swa = 1 };

struct MimoGeometry {
    int64_t n_embd = 4096;              ///< hidden_size
    int64_t n_layers = 48;              ///< num_hidden_layers
    int64_t n_vocab = 152576;           ///< vocab_size
    double rms_eps = 1e-6;              ///< layernorm_epsilon (fallback rms_norm_eps)

    // attention (shared by both kinds unless noted)
    int64_t n_q_head = 64;              ///< num_attention_heads
    int64_t head_dim_qk = 192;          ///< head_dim (q and k width; v is v_head_dim)
    int64_t head_dim_v = 128;           ///< v_head_dim (checkpoint pads v rows to head_dim_qk)
    int64_t n_rot = 64;                 ///< int(head_dim_qk * partial_rotary_factor): rotated pairs * 2
    int64_t n_head_kv_full = 4;         ///< num_key_value_heads
    int64_t n_head_kv_swa = 8;          ///< swa_num_key_value_heads
    double rope_theta_full = 1e7;       ///< rope_theta
    double rope_theta_swa = 1e4;        ///< swa_rope_theta (also every MTP head)
    int64_t sliding_window = 128;       ///< keys per query INCLUDING the query itself

    // load-time folds (applied exactly once, at load; the checkpoint is NOT pre-scaled)
    double attention_value_scale = 0.707;  ///< multiplies o_proj weights

    // router / MoE (layers with moe_layer_freq == 1)
    int64_t n_expert = 256;             ///< n_routed_experts
    int64_t n_expert_used = 8;          ///< num_experts_per_tok
    int64_t moe_intermediate = 2048;    ///< moe_intermediate_size
    int64_t dense_intermediate = 16384; ///< intermediate_size (layer 0's dense FFN)
    double routed_scaling_factor = 1.0; ///< routed_scaling_factor (null -> 1)
    bool norm_topk_prob = true;
    bool sigmoid_scoring = true;        ///< scoring_func == "sigmoid"
    bool topk_noaux_tc = true;          ///< topk_method == "noaux_tc"

    // MTP heads in the checkpoint (model.mtp.layers.{k}.*)
    int64_t n_mtp_layers = 3;           ///< num_nextn_predict_layers

    // Dflash drafter (optional sibling dir; loaded separately, kept here for one stop)
    bool has_dflash = false;

    /// Parsed hybrid_layer_pattern, one entry per layer (0 = Full, 1 = Swa).
    std::vector<uint8_t> layer_kind;
    /// Parsed moe_layer_freq, one entry per layer (0 = dense FFN, 1 = MoE).
    std::vector<uint8_t> moe_layer;

    MiMoLayerKind kind(int64_t layer) const { return layer_kind[layer] == 0 ? MiMoLayerKind::Full : MiMoLayerKind::Swa; }
    bool is_full(int64_t layer) const { return layer_kind[layer] == 0; }
    bool is_moe(int64_t layer) const { return moe_layer[layer] != 0; }
    int64_t kv_heads(int64_t layer) const { return is_full(layer) ? n_head_kv_full : n_head_kv_swa; }
    double rope_theta(int64_t layer) const { return is_full(layer) ? rope_theta_full : rope_theta_swa; }
    int64_t qk_dim() const { return n_q_head * head_dim_qk; }             ///< q_proj rows
    int64_t v_dim() const { return n_q_head * head_dim_v; }               ///< attention output width = o_proj cols
    int64_t n_swa_layers() const {
        int64_t n = 0;
        for (uint8_t k : layer_kind) n += (k != 0);
        return n;
    }
};

// --------------------------------------------------------------------------- tiny JSON reader

namespace mimo_json {

struct Value;
using Object = std::map<std::string, std::shared_ptr<Value>>;

struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object } type = Type::Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<std::shared_ptr<Value>> arr;
    Object obj;

    const Value* find(const std::string& key) const {
        if (type != Type::Object) return nullptr;
        auto it = obj.find(key);
        return it == obj.end() ? nullptr : it->second.get();
    }
};

class Parser {
public:
    explicit Parser(const std::string& text) : s_(text) {}

    /// Parses the whole document; throws std::runtime_error with position info on malformed input.
    std::shared_ptr<Value> document() {
        auto v = parse_value();
        skip_ws();
        if (pos_ != s_.size()) fail("trailing content");
        return v;
    }

private:
    const std::string& s_;
    size_t pos_ = 0;

    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error(std::string("mimo config json: ") + what + " at offset " + std::to_string(pos_));
    }
    void skip_ws() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r')) ++pos_;
    }
    char peek() {
        skip_ws();
        if (pos_ >= s_.size()) fail("unexpected end");
        return s_[pos_];
    }
    void expect(char c) {
        if (peek() != c) fail("unexpected character");
        ++pos_;
    }

    std::shared_ptr<Value> parse_value() {
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') {
            auto v = std::make_shared<Value>();
            v->type = Value::Type::String;
            v->str = parse_string();
            return v;
        }
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        return parse_number();
    }
    std::shared_ptr<Value> parse_object() {
        expect('{');
        auto v = std::make_shared<Value>();
        v->type = Value::Type::Object;
        if (peek() == '}') { ++pos_; return v; }
        for (;;) {
            if (peek() != '"') fail("expected key");
            std::string key = parse_string();
            expect(':');
            v->obj[key] = parse_value();
            char c = peek();
            if (c == ',') { ++pos_; continue; }
            if (c == '}') { ++pos_; return v; }
            fail("expected , or }");
        }
    }
    std::shared_ptr<Value> parse_array() {
        expect('[');
        auto v = std::make_shared<Value>();
        v->type = Value::Type::Array;
        if (peek() == ']') { ++pos_; return v; }
        for (;;) {
            v->arr.push_back(parse_value());
            char c = peek();
            if (c == ',') { ++pos_; continue; }
            if (c == ']') { ++pos_; return v; }
            fail("expected , or ]");
        }
    }
    std::string parse_string() {
        expect('"');
        std::string out;
        while (pos_ < s_.size() && s_[pos_] != '"') {
            char c = s_[pos_++];
            if (c != '\\') { out.push_back(c); continue; }
            if (pos_ >= s_.size()) fail("bad escape");
            char e = s_[pos_++];
            switch (e) {
            case '"': case '\\': case '/': out.push_back(e); break;
            case 'n': out.push_back('\n'); break;
            case 't': out.push_back('\t'); break;
            case 'r': out.push_back('\r'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'u': {
                if (pos_ + 4 > s_.size()) fail("bad \\u");
                unsigned cp = 0;
                for (int i = 0; i < 4; ++i) {
                    char h = s_[pos_++];
                    cp <<= 4;
                    if (h >= '0' && h <= '9') cp |= unsigned(h - '0');
                    else if (h >= 'a' && h <= 'f') cp |= unsigned(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') cp |= unsigned(h - 'A' + 10);
                    else fail("bad hex digit");
                }
                if (cp < 0x80) {
                    out.push_back(char(cp));
                } else if (cp < 0x800) {  // 2-byte UTF-8
                    out.push_back(char(0xC0 | (cp >> 6)));
                    out.push_back(char(0x80 | (cp & 0x3F)));
                } else {  // 3-byte UTF-8 (config.json keys/values stay in this range)
                    out.push_back(char(0xE0 | (cp >> 12)));
                    out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
                    out.push_back(char(0x80 | (cp & 0x3F)));
                }
                break;
            }
            default: fail("bad escape char");
            }
        }
        if (pos_ >= s_.size()) fail("unterminated string");
        ++pos_;  // closing quote
        return out;
    }
    std::shared_ptr<Value> parse_bool() {
        auto v = std::make_shared<Value>();
        v->type = Value::Type::Bool;
        if (s_.compare(pos_, 4, "true") == 0) { v->b = true; pos_ += 4; return v; }
        if (s_.compare(pos_, 5, "false") == 0) { v->b = false; pos_ += 5; return v; }
        fail("bad literal");
    }
    std::shared_ptr<Value> parse_null() {
        if (s_.compare(pos_, 4, "null") != 0) fail("bad literal");
        pos_ += 4;
        return std::make_shared<Value>();
    }
    std::shared_ptr<Value> parse_number() {
        size_t start = pos_;
        if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
        bool digits = false;
        while (pos_ < s_.size() && ((s_[pos_] >= '0' && s_[pos_] <= '9') || s_[pos_] == '.' || s_[pos_] == 'e' ||
                                     s_[pos_] == 'E' || s_[pos_] == '-' || s_[pos_] == '+')) {
            if (s_[pos_] >= '0' && s_[pos_] <= '9') digits = true;
            ++pos_;
        }
        if (!digits) fail("bad number");
        auto v = std::make_shared<Value>();
        v->type = Value::Type::Number;
        v->num = std::stod(s_.substr(start, pos_ - start));
        return v;
    }
};

// Typed getters used by the config reader: nullptr / nullopt-style semantics via found flag.
inline bool get_number(const Value* v, const char* key, double& out) {
    if (!v) return false;
    const Value* f = v->find(key);
    if (!f || f->type != Value::Type::Number) return false;
    out = f->num;
    return true;
}
inline bool get_int(const Value* v, const char* key, int64_t& out) {
    double d;
    if (!get_number(v, key, d)) return false;
    out = int64_t(llround(d));
    return true;
}
inline bool get_bool(const Value* v, const char* key, bool& out) {
    if (!v) return false;
    const Value* f = v->find(key);
    if (!f || f->type != Value::Type::Bool) return false;
    out = f->b;
    return true;
}
inline bool get_string(const Value* v, const char* key, std::string& out) {
    if (!v) return false;
    const Value* f = v->find(key);
    if (!f || f->type != Value::Type::String) return false;
    out = f->str;
    return true;
}
/// Reads a flat array of integers (hybrid_layer_pattern, moe_layer_freq).
inline bool get_int_array(const Value* v, const char* key, std::vector<uint8_t>& out) {
    if (!v) return false;
    const Value* f = v->find(key);
    if (!f || f->type != Value::Type::Array) return false;
    out.clear();
    out.reserve(f->arr.size());
    for (const auto& e : f->arr) {
        if (e->type != Value::Type::Number) return false;
        out.push_back(uint8_t(llround(e->num)));
    }
    return true;
}

}  // namespace mimo_json

// --------------------------------------------------------------------------- config.json reader

/// Parses MiMo v2's config.json text into a MimoGeometry.  Throws std::runtime_error on a
/// document that does not declare model_type "mimo_v2", or whose per-layer lists disagree with
/// num_hidden_layers.  Everything else keeps the struct defaults; the checkpoint cross-check
/// (kv widths vs pattern, sink presence) happens later, in the loader, against real tensors.
inline MimoGeometry mimo_geometry_from_config(const std::string& config_json) {
    using namespace mimo_json;
    Parser p(config_json);
    std::shared_ptr<Value> doc = p.document();
    if (!doc || doc->type != Value::Type::Object) throw std::runtime_error("mimo config json: not an object");

    std::string model_type;
    if (get_string(doc.get(), "model_type", model_type) && model_type != "mimo_v2")
        throw std::runtime_error("mimo config json: model_type '" + model_type + "' is not mimo_v2");

    MimoGeometry g;
    get_int(doc.get(), "hidden_size", g.n_embd);
    get_int(doc.get(), "num_hidden_layers", g.n_layers);
    get_int(doc.get(), "vocab_size", g.n_vocab);
    double eps;
    if (get_number(doc.get(), "layernorm_epsilon", eps) || get_number(doc.get(), "rms_norm_eps", eps)) g.rms_eps = eps;

    get_int(doc.get(), "num_attention_heads", g.n_q_head);
    get_int(doc.get(), "head_dim", g.head_dim_qk);
    get_int(doc.get(), "v_head_dim", g.head_dim_v);
    get_int(doc.get(), "num_key_value_heads", g.n_head_kv_full);
    get_int(doc.get(), "swa_num_key_value_heads", g.n_head_kv_swa);
    get_int(doc.get(), "sliding_window", g.sliding_window);
    get_number(doc.get(), "rope_theta", g.rope_theta_full);
    get_number(doc.get(), "swa_rope_theta", g.rope_theta_swa);
    get_number(doc.get(), "attention_value_scale", g.attention_value_scale);
    double prf = 0.334;
    // Truncation matches exllamav3's int(head_dim * partial_rotary_factor) exactly; the real
    // config's 0.334 gives 64.128 -> 64. A hypothetical config written as 0.33333333 would
    // yield 63 in BOTH engines - reference-faithful beats round-to-even here.
    if (get_number(doc.get(), "partial_rotary_factor", prf)) g.n_rot = int64_t(g.head_dim_qk * prf);

    get_int(doc.get(), "n_routed_experts", g.n_expert);
    get_int(doc.get(), "num_experts_per_tok", g.n_expert_used);
    get_int(doc.get(), "moe_intermediate_size", g.moe_intermediate);
    get_int(doc.get(), "intermediate_size", g.dense_intermediate);
    get_number(doc.get(), "routed_scaling_factor", g.routed_scaling_factor);
    get_bool(doc.get(), "norm_topk_prob", g.norm_topk_prob);
    std::string scoring, topk;
    if (get_string(doc.get(), "scoring_func", scoring)) g.sigmoid_scoring = (scoring == "sigmoid");
    if (get_string(doc.get(), "topk_method", topk)) g.topk_noaux_tc = (topk == "noaux_tc");
    get_int(doc.get(), "num_nextn_predict_layers", g.n_mtp_layers);

    if (!get_int_array(doc.get(), "hybrid_layer_pattern", g.layer_kind))
        g.layer_kind.assign(size_t(g.n_layers), 0);  // all full attention
    if (!get_int_array(doc.get(), "moe_layer_freq", g.moe_layer))
        g.moe_layer.assign(size_t(g.n_layers), 1);  // all MoE
    if (int64_t(g.layer_kind.size()) != g.n_layers)
        throw std::runtime_error("mimo config json: hybrid_layer_pattern length != num_hidden_layers");
    if (int64_t(g.moe_layer.size()) != g.n_layers)
        throw std::runtime_error("mimo config json: moe_layer_freq length != num_hidden_layers");

    if (g.n_embd <= 0 || g.n_layers <= 0 || g.n_vocab <= 0 || g.head_dim_qk <= 0 || g.head_dim_v <= 0 ||
        g.head_dim_qk % 16 != 0 || g.n_rot % 2 != 0 || g.n_rot > g.head_dim_qk || g.n_expert <= 0 ||
        g.n_expert_used <= 0 || g.n_q_head <= 0 || g.n_head_kv_full <= 0 || g.n_head_kv_swa <= 0 ||
        g.n_q_head % g.n_head_kv_full != 0 || g.n_q_head % g.n_head_kv_swa != 0)
        throw std::runtime_error("mimo config json: implausible geometry (see MimoGeometry fields)");
    return g;
}

/// True when an EXL3 model dir's config.json declares MiMo v2 (reads the file; false if the
/// file is missing or unparsable - other families' configs must not throw out of here).
inline bool mimo_detect_config(const std::string& config_path, std::string* error = nullptr) {
    FILE* f = fopen(config_path.c_str(), "rb");
    if (!f) {
        if (error) *error = "cannot open " + config_path;
        return false;
    }
    std::string text;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
    fclose(f);
    if (text.size() > 8u * 1024 * 1024) {  // a MiMo config is ~10 KB; anything huge is not ours
        if (error) *error = "config.json implausibly large";
        return false;
    }
    // Cheap pre-check before the parser: a non-mimo_v2 document must not throw diagnostics.
    if (text.find("\"mimo_v2\"") == std::string::npos) return false;
    try {
        mimo_geometry_from_config(text);
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

}  // namespace strata::core
