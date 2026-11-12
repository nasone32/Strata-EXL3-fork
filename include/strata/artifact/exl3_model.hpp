// include/strata/artifact/exl3_model.hpp - multi-shard EXL3 model accessor (docs/EXL3.md).
//
// An EXL3 model is a set of safetensors shards plus `model.safetensors.index.json` (tensor -> shard).
// Every quantized linear is `prefix.trellis` (I16 [ki,nj,256*K/16]), `prefix.suh`/`prefix.svh` (F16) and a
// `prefix.mul1` / `prefix.mcg` marker.  This resolves a linear across shards and hands back the raw
// pointers the decode kernels take.  Shards are mmap'd lazily and never copied.
#pragma once

#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "strata/artifact/exl3_expert_backing.hpp"
#include "strata/artifact/safetensors.hpp"

namespace strata {

struct Exl3Linear {
    const uint16_t* trellis = nullptr;
    const uint16_t* suh = nullptr;
    const uint16_t* svh = nullptr;
    int ki = 0, nj = 0, bits = 0;
    // Fractional bitrates (MiMo v2's 2.26 bpw checkpoint) are half-integer K = KA + 0.5: an
    // alternating KA / KA+1-bit trellis step per stored position (exllamav3 frac.cuh, MASK 0xAAAA).
    // `bits` carries KA and `frac` the extra half, so K = bits + frac * 0.5; integer-K files keep
    // frac == 0 and every historical consumer sees exactly the values it saw before.
    int frac = 0;
    int cb = 2;                        // 0=3inst, 1=mcg, 2=mul1
    // Physical trellis layout of the bytes `trellis` points at: Native source shards, or Band8
    // when an Exl3ExpertBacking covers this expert (kernels/prefill readers must check it).
    // Deliberately the LAST field: existing aggregate initializers keep compiling and every
    // historical consumer sees the default.
    Exl3TrellisLayout layout = Exl3TrellisLayout::Native;
    int in() const { return ki * 16; }
    int out() const { return nj * 16; }
    /// uint16 words per 256-weight tile: 16*K (40 for K=2.5 - NOT expressible as 256*bits/16).
    int words() const { return 16 * bits + 8 * frac; }
};

namespace exl3_detail {

// Parse the `weight_map` object of a safetensors index: {"tensor.name":"model-0000N-of-0000M.safetensors", ...}.
inline std::map<std::string, std::string> parse_weight_map(const std::string& json) {
    std::map<std::string, std::string> out;
    size_t p = json.find("\"weight_map\"");
    if (p == std::string::npos) throw std::runtime_error("index.json: no weight_map");
    p = json.find('{', p);
    if (p == std::string::npos) throw std::runtime_error("index.json: malformed weight_map");
    ++p;
    auto skip_ws = [&]() { while (p < json.size() && (json[p] == ' ' || json[p] == '\n' || json[p] == '\r' || json[p] == '\t')) ++p; };
    auto str = [&]() -> std::string {
        if (p >= json.size() || json[p] != '"') throw std::runtime_error("index.json: expected string");
        ++p;
        std::string s;
        while (p < json.size() && json[p] != '"') {
            if (json[p] == '\\' && p + 1 < json.size()) { ++p; }
            s.push_back(json[p++]);
        }
        ++p;
        return s;
    };
    skip_ws();
    if (json[p] == '}') return out;
    for (;;) {
        skip_ws();
        std::string key = str();
        skip_ws(); ++p;                        // ':'
        skip_ws();
        std::string val = str();
        out.emplace(std::move(key), std::move(val));
        skip_ws();
        if (json[p] == ',') { ++p; continue; }
        if (json[p] == '}') break;
        throw std::runtime_error("index.json: malformed weight_map entry");
    }
    return out;
}

}  // namespace exl3_detail

class Exl3Model {
public:
    // `expert_backing` (optional) serves packed routed experts from a shared CPU/GPU cache:
    // linear()/locate() consult it FIRST and fall back to the native source shards only for
    // names it does not cover (dense projections, unpacked tensors). The model keeps a copy, so
    // the backing's lifetime equals the model's; close_shards() never unmaps it. nullptr (the
    // default and every pre-existing call site) leaves the native behavior byte-identical.
    explicit Exl3Model(const std::string& dir,
                       std::shared_ptr<const Exl3ExpertBacking> expert_backing = {})
        : dir_(dir), backing_(std::move(expert_backing)) {
        std::string path = dir + "/model.safetensors.index.json";
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("exl3: cannot open " + path);
        std::ostringstream ss;
        ss << in.rdbuf();
        weight_map_ = exl3_detail::parse_weight_map(ss.str());
    }

    bool has(const std::string& base) const { return weight_map_.count(base + ".trellis") != 0; }
    /// The shard path and the FILE byte range of a tensor, for readers that pread instead of touching the mmap.
    /// Packed experts are intercepted first: .trellis/.suh/.svh of a covered expert resolve to
    /// the SAME packed file/range linear() reads; everything else resolves in the source shards.
    bool locate(const std::string& name, std::string& path, size_t& off, size_t& bytes) const {
        if (backing_ && backing_->locate(name, path, off, bytes)) return true;
        if (backing_ && backing_->memory_only() && backing_->covers_tensor(name)) return false;
        auto it = weight_map_.find(name);
        if (it == weight_map_.end()) return false;
        const SafetensorsFile& sf = shard(it->second);
        const StTensor* t = sf.find(name);
        if (t == nullptr) return false;
        path = dir_ + "/" + it->second;
        off = sf.data_offset() + t->begin;
        bytes = t->nbytes();
        return true;
    }
    size_t tensor_count() const { return weight_map_.size(); }

    /// The routed experts' physical trellis layout: the backing's layout when one covers them
    /// (Band8), Native otherwise.  A cheap MODE PROBE (no shard is opened, no expert is searched)
    /// so a consumer can pick its dispatch once, up front; every linear()/locate() result must
    /// still be checked against it - a name the backing does not cover resolves Native.
    Exl3TrellisLayout expert_layout() const noexcept {
        return backing_ ? backing_->layout() : Exl3TrellisLayout::Native;
    }

    /// The optional packed-expert backing this accessor was built with (nullptr without one),
    /// for capability questions the BACKING must answer about bytes it owns (e.g. the R1
    /// prototype's direct-DMA eligibility, STRATA_EXL3_RESIDENT_DIRECT_DMA).  Read-only; the
    /// model keeps the backing alive, so the pointer outlives this accessor.
    const Exl3ExpertBacking* expert_backing() const noexcept { return backing_.get(); }

    /// Unmap every shard this accessor opened.  The pack calls this once its dense tensors are on the
    /// GPU: a page that is still mapped survives POSIX_FADV_DONTNEED, so the unmap comes before the
    /// fadvise.  Later lookups lazily re-map (nothing does after a build).
    void close_shards() { shards_.clear(); }

    /// Shape/dtype of ANY indexed tensor (nullptr when absent): the non-trellis roles (norms,
    /// routers, embeddings) never go through linear(), but the MiMo pack validates them the same
    /// way.  Opens the shard lazily, like linear() does.
    const StTensor* meta(const std::string& name) const {
        auto it = weight_map_.find(name);
        if (it == weight_map_.end()) return nullptr;
        return shard(it->second).find(name);
    }

    Exl3Linear linear(const std::string& base) const {
        // Packed experts first: pointers come from the backing's immutable mmap and the raw
        // source payload below is never touched for them. Uncovered names (dense projections,
        // experts outside the request list) take the original native path unchanged.
        if (backing_) {
            Exl3Linear packed;
            if (backing_->linear(base, packed)) return packed;
        }
        auto tr = tensor(base + ".trellis");
        auto suh = tensor(base + ".suh");
        auto svh = tensor(base + ".svh");
        // Validate before narrowing dimensions or letting a decode kernel read the mapped bytes.
        // In particular, a half-integer K must not be silently rounded down to an integer K.
        // The last dim is a multiple of 8 iff K is a multiple of 0.5 (exllamav3 exl3.py:67-68 keeps
        // K = shape[-1] / 16 as a float for the 2.26 bpw MiMo checkpoints; the 16K-bit tile stays a
        // whole number of uint16 words because K is half-integer, never thirds).
        if (tr->dtype != "I16" || tr->shape.size() != 3 ||
            tr->shape[0] <= 0 || tr->shape[1] <= 0 ||
            tr->shape[0] > std::numeric_limits<int>::max() / 16 ||
            tr->shape[1] > std::numeric_limits<int>::max() / 16 ||
            tr->shape[0] % 8 != 0 || tr->shape[1] % 8 != 0 ||
            tr->shape[2] < 16 || tr->shape[2] > 128 || tr->shape[2] % 8 != 0) {
            throw std::runtime_error("exl3: " + base +
                ": need I16 trellis [in/16,out/16,16*K], H128-aligned dimensions and K=1..8 in 0.5 steps");
        }
        if (suh->dtype != "F16" || svh->dtype != "F16" ||
            suh->shape != std::vector<int64_t>{tr->shape[0] * 16} ||
            svh->shape != std::vector<int64_t>{tr->shape[1] * 16}) {
            throw std::runtime_error("exl3: " + base + ": need F16 suh[in] and svh[out]");
        }
        if (weight_map_.count(base + ".bias")) {
            throw std::runtime_error("exl3: " + base + ": biased EXL3 linears are not supported");
        }
        Exl3Linear L;
        L.trellis = (const uint16_t*)data(*tr);
        L.suh = (const uint16_t*)data(*suh);
        L.svh = (const uint16_t*)data(*svh);
        L.ki = (int)tr->shape[0];
        L.nj = (int)tr->shape[1];
        L.bits = (int)(tr->shape[2] / 16);
        L.frac = (int)((tr->shape[2] % 16) / 8);   // 0 or 1: K = bits + frac/2
        L.cb = weight_map_.count(base + ".mul1") ? 2 : weight_map_.count(base + ".mcg") ? 1 : 0;
        // Half-integer bitrates are a mul1-only extension upstream (exl3.py asserts the same); the
        // alternating-step window math is only ever produced beside the mul1 marker.
        if (L.frac != 0 && L.cb != 2)
            throw std::runtime_error("exl3: " + base + ": half-integer K " +
                std::to_string(L.bits + 0.5) + " requires the mul1 codebook");
        return L;
    }

private:
    const SafetensorsFile& shard(const std::string& name) const {
        auto it = shards_.find(name);
        if (it != shards_.end()) return *it->second;
        auto f = std::make_unique<SafetensorsFile>(dir_ + "/" + name);
        auto* raw = f.get();
        shards_.emplace(name, std::move(f));
        return *raw;
    }
    const StTensor* tensor(const std::string& name) const {
        auto it = weight_map_.find(name);
        if (it == weight_map_.end()) throw std::runtime_error("exl3: no tensor " + name);
        const StTensor* t = shard(it->second).find(name);
        if (!t) throw std::runtime_error("exl3: tensor " + name + " not in " + it->second);
        return t;
    }
    const uint8_t* data(const StTensor& t) const {
        auto it = weight_map_.find(t.name);
        return shard(it->second).data(t);
    }

    std::string dir_;
    std::shared_ptr<const Exl3ExpertBacking> backing_;   // optional packed expert cache, same lifetime
    std::map<std::string, std::string> weight_map_;
    mutable std::map<std::string, std::unique_ptr<SafetensorsFile>> shards_;
};

}  // namespace strata
