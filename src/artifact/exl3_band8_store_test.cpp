// src/artifact/exl3_band8_store_test.cpp - CPU-only fixture test for the persistent band-8
// packed-expert store (notes/EXL3-BAND8-FEASIBILITY-2026-11-06.md). No GPU, no model download:
// a tiny synthetic safetensors model exercises all four verified rates and every refusal and
// recovery path. Everything stays under the OS temporary directory and is bounded.

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/artifact/exl3_band8_store.hpp"
#include "strata/artifact/exl3_model.hpp"
#include "strata/artifact/exl3_host_warmup.hpp"
#include "strata/artifact/exl3_resident.hpp"
#include "strata/artifact/safetensors.hpp"

namespace fs = std::filesystem;

static int g_failures = 0;
static int g_checks = 0;
#define CHECK(cond, msg) \
    do { ++g_checks; if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, (msg)); ++g_failures; } } while (0)
#define CHECK_THROW(expr, needle)                                            \
    do {                                                                     \
        bool threw_ = false; std::string what_;                              \
        try { expr; } catch (const std::exception& e_) { threw_ = true; what_ = e_.what(); }  \
        if (!threw_) { std::printf("FAIL %s:%d: expected throw (%s)\n", __FILE__, __LINE__, (needle)); ++g_failures; } \
        else if (what_.find(needle) == std::string::npos) {                  \
            std::printf("FAIL %s:%d: throw without \"%s\": %s\n", __FILE__, __LINE__, (needle), what_.c_str()); ++g_failures; } \
    } while (0)

using strata::Exl3Band8BuildOptions;
using strata::Exl3Band8Store;
using strata::Exl3Model;
using strata::Exl3TrellisLayout;

// ---- tiny safetensors fixture writer -----------------------------------------
struct FixtureTensor {
    std::string name, dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};

void write_shard(const fs::path& path, std::vector<FixtureTensor> tensors) {
    std::map<std::string, const FixtureTensor*> order;   // sorted: canonical layout
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

std::string quoted(const std::string& k, const std::string& v) { return "\"" + k + "\":\"" + v + "\""; }

void write_index(const fs::path& path, const std::map<std::string, std::string>& weight_map) {
    std::ostringstream json;
    json << "{\"metadata\":{\"format\":\"band8-fixture\"},\"weight_map\":{";
    size_t i = 0;
    for (const auto& [k, v] : weight_map)
        json << (i++ ? "," : "") << quoted(k, v);
    json << "}}";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f << json.str();
}

// Deterministic per-tensor bytes: stable across runs, so every comparison is reproducible.
std::vector<uint8_t> synth_bytes(const std::string& name, size_t n) {
    std::vector<uint8_t> b(n);
    std::seed_seq seq(name.begin(), name.end());
    std::mt19937_64 rng(seq);
    for (size_t i = 0; i < n; ++i) b[i] = (uint8_t)rng();
    return b;
}

struct Rate { int last_dim; };   // shape[2]: K = last_dim/16 (+0.5 if last_dim%16==8)

// One synthetic EXL3 model: `layers` x `experts` routed triples at the four verified rates, one
// dense projection (never requested) and one non-linear tensor, split over two shards.
struct FixtureModel {
    fs::path dir;
    std::vector<std::string> expert_prefixes;   // the request list, caller-generated
    // `weight_tag` shifts every synthetic byte deterministically: two fixtures built with the
    // same shapes but different tags are byte-disjoint - same sizes, different weights.
    void build(const fs::path& d, int layers, int experts, const std::string& weight_tag = "") {
        dir = d;
        fs::create_directories(d);
        const std::string tag = weight_tag;
        std::map<std::string, std::string> weight_map;
        std::vector<FixtureTensor> shard[2];
        int shard_of = 0;
        auto add = [&](const std::string& name, const std::string& dtype,
                       std::vector<int64_t> shape, size_t bytes) {
            FixtureTensor t{name, dtype, std::move(shape), synth_bytes(name + tag, bytes)};
            const std::string file = shard_of == 0 ? "fixture-00001-of-00002.safetensors"
                                                   : "fixture-00002-of-00002.safetensors";
            weight_map.emplace(name, file);
            shard[shard_of].push_back(std::move(t));
        };
        auto rate_of = [&](int layer, int expert) -> int {
            if (layer == 0) return 32;                       // K2
            if (layer == 1) return 40;                       // K2.5
            return expert < 2 ? 48 : 64;                     // K3 / K4
        };
        for (int l = 0; l < layers; ++l) {
            for (int e = 0; e < experts; ++e) {
                for (const char* proj : {"gate_proj", "up_proj", "down_proj"}) {
                    const std::string p =
                        "model.layers." + std::to_string(l) + ".mlp.experts." + std::to_string(e) + "." + proj;
                    expert_prefixes.push_back(p);
                    const int last_dim = rate_of(l, e);
                    const int64_t ki = std::string(proj) == "down_proj" ? 8 : 16;
                    const int64_t nj = std::string(proj) == "down_proj" ? 16 : 8;
                    const size_t tre_bytes = (size_t)ki * nj * last_dim * 2;
                    add(p + ".trellis", "I16", {ki, nj, last_dim}, tre_bytes);
                    add(p + ".suh", "F16", {ki * 16}, (size_t)ki * 32);
                    add(p + ".svh", "F16", {nj * 16}, (size_t)nj * 32);
                    add(p + ".mul1", "U8", {1}, 1);
                    shard_of ^= 1;   // spread triples over both shards
                }
            }
        }
        // Dense projection: present in the source, deliberately NOT in the request list.
        add("model.layers.0.self_attn.q_proj.trellis", "I16", {8, 8, 32}, 8 * 8 * 32 * 2);
        add("model.layers.0.self_attn.q_proj.suh", "F16", {128}, 256);
        add("model.layers.0.self_attn.q_proj.svh", "F16", {128}, 256);
        add("model.layers.0.self_attn.q_proj.mul1", "U8", {1}, 1);
        add("model.layers.0.input_layernorm.weight", "F16", {128}, 256);
        write_shard(d / "fixture-00001-of-00002.safetensors", shard[0]);
        write_shard(d / "fixture-00002-of-00002.safetensors", shard[1]);
        write_index(d / "model.safetensors.index.json", weight_map);
    }
};

// Reference band-8 permutation, written tile-by-tile straight from the formula (independent of
// the store's band-grouped memcpy implementation).
std::vector<uint16_t> reference_band8(const std::vector<uint16_t>& native, int ki, int nj, int P) {
    std::vector<uint16_t> out(native.size());
    for (int kt = 0; kt < ki; ++kt)
        for (int nt = 0; nt < nj; ++nt) {
            const size_t src = ((size_t)kt * nj + nt) * P;
            const size_t dst = (((size_t)nt / 8 * ki + kt) * 8 + nt % 8) * P;
            std::memcpy(&out[dst], &native[src], (size_t)P * 2);
        }
    return out;
}

std::vector<uint8_t> source_bytes(const fs::path& dir, const std::string& tensor) {
    Exl3Model m(dir.string());   // legacy accessor, no backing
    std::string path;
    size_t off = 0, bytes = 0;
    if (!m.locate(tensor, path, off, bytes)) throw std::runtime_error("no tensor " + tensor);
    std::vector<uint8_t> b(bytes);
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0 || ::pread(fd, b.data(), bytes, (off_t)off) != (ssize_t)bytes)
        throw std::runtime_error("pread failed for " + tensor);
    ::close(fd);
    return b;
}

std::vector<uint8_t> pread_span(const std::string& path, size_t off, size_t bytes) {
    std::vector<uint8_t> b(bytes);
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("open failed: " + path);
    size_t done = 0;
    while (done < bytes) {
        const ssize_t r = ::pread(fd, b.data() + done, bytes - done, (off_t)(off + done));
        if (r <= 0) { ::close(fd); throw std::runtime_error("pread failed: " + path); }
        done += (size_t)r;
    }
    ::close(fd);
    return b;
}

ino_t inode_of(const fs::path& p) {
    struct stat sb {};
    if (::stat(p.c_str(), &sb) != 0) return 0;
    return sb.st_ino;
}

void touch_mtime(const fs::path& p) {
    struct stat sb {};
    if (::stat(p.c_str(), &sb) != 0) throw std::runtime_error("stat failed for touch");
    struct timespec ts[2] {};
    ts[0].tv_nsec = UTIME_OMIT;
    ts[1].tv_sec = sb.st_mtim.tv_sec + 5;   // strictly later than any previous mtime
    if (::utimensat(AT_FDCWD, p.c_str(), ts, 0) != 0) throw std::runtime_error("utimensat failed");
}

fs::path gen_of(const std::shared_ptr<const strata::Exl3Band8Store>& s) {
    return fs::path(s->payload_path()).parent_path();
}

struct timespec mtime_of(const fs::path& p) {
    struct stat sb {};
    if (::stat(p.c_str(), &sb) != 0) throw std::runtime_error("stat failed");
    return sb.st_mtim;
}
void touch_exact(const fs::path& p, const struct timespec& t) {
    struct timespec ts[2] {};
    ts[0].tv_nsec = UTIME_OMIT;
    ts[1] = t;   // full precision: the fingerprint hashes tv_sec AND tv_nsec
    if (::utimensat(AT_FDCWD, p.c_str(), ts, 0) != 0) throw std::runtime_error("utimensat failed");
}

size_t tree_bytes(const fs::path& p) {
    size_t total = 0;
    for (const auto& e : fs::recursive_directory_iterator(p))
        if (e.is_regular_file()) total += (size_t)e.file_size();
    return total;
}

size_t peak_rss_kb() {
    std::ifstream in("/proc/self/status");
    std::string key;
    std::string line;
    while (std::getline(in, line))
        if (line.rfind("VmHWM:", 0) == 0) {
            size_t v = 0;
            std::sscanf(line.c_str(), "VmHWM: %zu kB", &v);
            return v;
        }
    return 0;
}

// One packed expert, checked byte-for-byte against the reference permutation, via BOTH the
// linear() pointers and the pread span locate() reports (the critical CPU/GPU equivalence).
void check_expert_bytes(const std::string& tag, const Exl3Band8Store& store,
                        const Exl3Model& model, const fs::path& src_dir,
                        const std::string& prefix, int ki, int nj, int last_dim) {
    strata::Exl3Linear L{};
    CHECK(store.linear(prefix, L), (tag + ": store.linear covers").c_str());
    CHECK(model.linear(prefix).layout == Exl3TrellisLayout::Band8, (tag + ": model layout Band8").c_str());
    const strata::Exl3Linear ML = model.linear(prefix);
    const int P = last_dim / 8;   // uint16 words per tile: 16*K = 2*last_dim/... (last_dim = 16K)
    // words per tile = 16*bits + 8*frac = last_dim  (since last_dim = 16K and K = last_dim/16)
    std::vector<uint16_t> native;
    {
        const std::vector<uint8_t> raw = source_bytes(src_dir, prefix + ".trellis");
        native.resize(raw.size() / 2);
        std::memcpy(native.data(), raw.data(), raw.size());
    }
    const std::vector<uint16_t> want = reference_band8(native, ki, nj, last_dim);
    CHECK(std::memcmp(want.data(), L.trellis, want.size() * 2) == 0, (tag + ": linear trellis == band8 reference").c_str());
    CHECK(std::memcmp(ML.trellis, L.trellis, want.size() * 2) == 0, (tag + ": model.linear == store.linear").c_str());
    CHECK(L.ki == ki && L.nj == nj && L.words() == last_dim && L.cb == 2 &&
          L.layout == Exl3TrellisLayout::Band8, (tag + ": linear metadata").c_str());
    CHECK(L.bits == last_dim / 16 && L.frac == (last_dim % 16) / 8, (tag + ": rate fields").c_str());
    // pRead path: locate must describe the SAME packed bytes the pointers read.
    std::string path;
    size_t off = 0, bytes = 0;
    CHECK(store.locate(prefix + ".trellis", path, off, bytes) && bytes == want.size() * 2,
          (tag + ": locate trellis span").c_str());
    CHECK(pread_span(path, off, bytes) == std::vector<uint8_t>((const uint8_t*)L.trellis,
                                                               (const uint8_t*)L.trellis + bytes),
          (tag + ": pread(locate) bytes == linear pointer bytes (critical)").c_str());
    std::string mpath;
    size_t moff = 0, mbytes = 0;
    CHECK(model.locate(prefix + ".trellis", mpath, moff, mbytes) && mpath == path && moff == off && mbytes == bytes,
          (tag + ": model.locate points at the packed file/range").c_str());
    // Scales: verbatim copies from the source, also mapped from the CACHE file.
    for (const char* plane : {".suh", ".svh"}) {
        const std::vector<uint8_t> src = source_bytes(src_dir, prefix + plane);
        CHECK(store.locate(prefix + plane, path, off, bytes) && bytes == src.size(),
              (tag + std::string(plane) + ": locate span").c_str());
        CHECK(pread_span(path, off, bytes) == src, (tag + std::string(plane) + ": pread bytes == source").c_str());
        const uint16_t* mapped = std::string(plane) == ".suh" ? L.suh : L.svh;
        CHECK(std::memcmp(mapped, src.data(), src.size()) == 0,
              (tag + std::string(plane) + ": mapped bytes == source").c_str());
    }
    (void)P;
}

void patch_manifest(const fs::path& gen, size_t record_index, size_t field_off, uint64_t value) {
    fs::path mp = gen / "manifest.bin";
    std::vector<uint8_t> b;
    {
        std::ifstream in(mp, std::ios::binary);
        b.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }
    const size_t at = 80 + record_index * 40 + field_off;   // record layout: offset@0, bytes@8
    if (at + 8 > b.size()) throw std::runtime_error("patch out of range");
    for (int i = 0; i < 8; ++i) b[at + i] = uint8_t(value >> (8 * i));
    std::ofstream out(mp, std::ios::binary | std::ios::trunc);
    out.write((const char*)b.data(), (std::streamsize)b.size());
}

int main() {
    // Respect TMPDIR rather than depending on the developer's scratch directory.
    const fs::path root_parent = fs::temp_directory_path();
    if (!fs::is_directory(root_parent)) {
        std::printf("FAIL: parent scratch dir %s missing\n", root_parent.c_str());
        return 1;
    }
    const fs::path root = root_parent / ("strata-band8-store-test-" + std::to_string(::getpid()));
    fs::remove_all(root);
    fs::create_directories(root);
    std::printf("fixture root: %s\n", root.c_str());

    Exl3Band8BuildOptions opts;
    opts.progress_every_triples = 4;

    // -- build + byte-exact permutation over all four verified rates --------------------
    FixtureModel fx;
    fx.build(root / "model", 3, 4);
    const fs::path cache = root / "cache-root";   // generation dirs live UNDER this root
    std::shared_ptr<const Exl3Band8Store> store;
    try {
        store = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(), fx.expert_prefixes, opts);
        std::printf("PASS build_or_open built: %zu records, payload %zu bytes, generation %016" PRIx64 "\n",
                    store->record_count(), fs::file_size(store->payload_path()), store->fingerprint());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL build_or_open: %s\n", e.what());
        std::printf("FAIL build_or_open: %s\n", e.what());
        ++g_failures;
    }
    CHECK(store != nullptr && store->record_count() == fx.expert_prefixes.size() * 3,
          "record count = 3 plane records per projection entry");
    CHECK(store->layout() == Exl3TrellisLayout::Band8, "store layout is Band8");
    // ALL later paths derive from the store's own payload location, never a hardcoded root.
    const fs::path gen = gen_of(store);
    CHECK(gen.filename().string().size() == 16 && gen.parent_path() == cache,
          "generation dir is cache-root/<fingerprint-hex16>");
    CHECK(fs::exists(gen / "manifest.bin") && !fs::exists(gen / "payload.bin.partial") &&
          !fs::exists(gen / "manifest.bin.partial") && !fs::exists(gen / "build.lock"),
          "published generation has no partial/lock leftovers");

    {
        Exl3Model model(fx.dir.string(), store);
        check_expert_bytes("K2-gate", *store, model, fx.dir, fx.expert_prefixes[0], 16, 8, 32);
        check_expert_bytes("K2-down", *store, model, fx.dir, fx.expert_prefixes[2], 8, 16, 32);
        check_expert_bytes("K2.5", *store, model, fx.dir, fx.expert_prefixes[15], 16, 8, 40);
        check_expert_bytes("K3", *store, model, fx.dir, fx.expert_prefixes[27], 16, 8, 48);
        check_expert_bytes("K4", *store, model, fx.dir, fx.expert_prefixes[30], 16, 8, 64);
    }

    // Host warmup must skip the caller's peer set, dense tensors, and source shadows.
    // Exercise the exact range planner/reader against both physical backings.
    for (bool packed : {false, true}) {
        Exl3Model model(fx.dir.string(), packed ? store : nullptr);
        std::vector<std::string> selected, excluded;
        for (size_t i = 0; i < fx.expert_prefixes.size(); ++i)
            ((i / 3) % 2 ? excluded : selected).push_back(fx.expert_prefixes[i]);
        uint64_t expected = 0;
        for (const auto& base : selected) for (const char* plane : {".trellis", ".suh", ".svh"}) {
            std::string p; size_t o = 0, n = 0;
            CHECK(model.locate(base + plane, p, o, n), "warmup selected plane exists");
            expected += n;
        }
        selected.push_back(selected.front());  // duplicate must not cause another read
        const auto ranges = strata::exl3_host_warmup_plan(model, selected);
        for (const auto& r : ranges)
            CHECK((r.path == store->payload_path()) == packed, "warmup uses working backing only");
        excluded.push_back("model.layers.0.self_attn.q_proj");
        for (const auto& base : excluded) for (const char* plane : {".trellis", ".suh", ".svh"}) {
            std::string p; size_t o = 0, n = 0;
            CHECK(model.locate(base + plane, p, o, n), "warmup excluded plane exists");
            for (const auto& r : ranges)
                CHECK(r.path != p || r.offset >= o + n || r.offset + r.bytes <= o,
                      "warmup never reads excluded peer/dense plane");
        }
        CHECK(strata::exl3_host_warmup_read(ranges) == expected, "warmup reads each selected byte once");
        CHECK(strata::exl3_host_warmup_read({}) == 0, "empty warmup is a no-op");
        CHECK_THROW(strata::exl3_host_warmup_plan(model, {"missing"}), "missing/invalid");
    }
    CHECK_THROW(strata::exl3_host_warmup_read({{(root / "missing").string(), 0, 1}}), "cannot open");
    CHECK_THROW(strata::exl3_host_warmup_read({{store->payload_path(), fs::file_size(store->payload_path()), 1}}),
                "short/error read");

    // Managed RAM has stable addresses, exact native/packed bytes, no locate
    // fallthrough, bounded locking, and copy-back-before-release transitions.
    std::vector<strata::Exl3ResidentBacking::Group> resident_groups;
    for (size_t i = 0; i < fx.expert_prefixes.size(); i += 3)
        resident_groups.push_back({(int)(i/12),(int)(i/3%4),
            {fx.expert_prefixes[i],fx.expert_prefixes[i+1],fx.expert_prefixes[i+2]}});
    for (bool packed : {false,true}) for (bool complement : {false,true}) {
        uint64_t locked = 0;
        strata::Exl3ResidentBacking::Options opt;
        opt.available_bytes = 1ull<<32; opt.headroom_bytes = 0;
        opt.lock = [&](void*,size_t n,std::string&) { locked += n; return true; };
        opt.unlock = [&](void*,size_t n) { locked -= n; };
        {
            auto ram = strata::Exl3ResidentBacking::create(fx.dir.string(),resident_groups,packed ? store : nullptr);
            Exl3Model model(fx.dir.string(),ram), ref(fx.dir.string(),packed ? store : nullptr);
            CHECK(!ram->memory_only(),"resident source delegates until activation");
            ram->activate(complement ? std::vector<std::pair<int,int>>{{0,0}} :
                                      std::vector<std::pair<int,int>>{},opt);
            CHECK(ram->memory_only(),"resident activation switches off file tier");
            const auto reads = ram->stats().file_reads;
            auto check_group = [&](const auto& group,bool copy) {
                for (const auto& base : group.projections) {
                    auto a = model.linear(base), b = ref.linear(base);
                    const size_t sizes[] = {(size_t)b.ki*b.nj*b.words()*2,(size_t)b.in()*2,(size_t)b.out()*2};
                    const void* src[] = {b.trellis,b.suh,b.svh};
                    const void* dst[] = {a.trellis,a.suh,a.svh};
                    CHECK(a.layout == b.layout && a.trellis != b.trellis,"resident tagged pointer is independent");
                    for (int p = 0; p < 3; ++p) {
                        if (copy) std::memcpy(const_cast<void*>(dst[p]),src[p],sizes[p]);
                        CHECK(std::memcmp(dst[p],src[p],sizes[p]) == 0,"resident plane bytes exact");
                    }
                    std::string path; size_t off = 0, n = 0;
                    CHECK(!model.locate(base+".trellis",path,off,n),"resident tensor NEVER falls back to file");
                }
            };
            for (const auto& group : resident_groups)
                if (ram->resident(group.layer,group.expert)) check_group(group,false);
            std::string path; size_t off = 0,n = 0;
            CHECK(model.locate("model.layers.0.self_attn.q_proj.trellis",path,off,n),"dense file locate unaffected");
            if (complement) {
                const auto stable = model.linear(resident_groups[0].projections[0]).trellis;
                CHECK(!ram->resident(0,0) && ram->has_gpu_only(),"peer expert has no RAM owner");
                CHECK(ram->prepare_return(0,0),"GPU eviction locks unpublished return target");
                CHECK(!ram->resident(0,0),"return not readable until GPU copy completes");
                CHECK_THROW(ram->prepare_return(0,0),"already pending");
                check_group(resident_groups[0],true);
                ram->finish_return(0,0);
                CHECK(ram->resident(0,0),"finished return is RAM resident");
                ram->release_to_gpu(0,1);
                CHECK(!ram->resident(0,1),"incoming RAM released only explicitly after publish");
                CHECK(ram->prepare_return(0,1),"returned incoming target can be prepared");
                ram->cancel_return(0,1);
                CHECK(!ram->resident(0,1),"cancel preserves original GPU ownership");
                CHECK(model.linear(resident_groups[0].projections[0]).trellis == stable,"CPU address stable through exchange");
                // The Qwen peer-complement BORROW rhythm (STRATA_EXL3_RESIDENT=peer +
                // STRATA_EXL3_PEER_PREFILL_BORROW): a lend RETURNS the outgoing expert's
                // planes GPU->RAM (prepare/copy/finish), the prompt reads/stages them there,
                // the restore re-uploads and RELEASES the span back to GPU-only - and the
                // NEXT prompt repeats the whole cycle on the same fixed address.  The
                // returned/released counters must track the repeats, and a released expert
                // must be returnable again immediately.
                ram->release_to_gpu(0,0);
                CHECK(!ram->resident(0,0) && ram->has_gpu_only(),"post-restore release returns the span to GPU-only");
                CHECK(ram->prepare_return(0,0),"repeat cycle: a re-return opens the same target");
                CHECK_THROW(ram->prepare_return(0,0),"already pending");
                check_group(resident_groups[0],true);
                ram->finish_return(0,0);
                CHECK(ram->resident(0,0),"repeat cycle: the second lend's copy is RAM again");
                ram->release_to_gpu(0,0);
                CHECK(!ram->resident(0,0) && ram->stats().released >= 3 &&
                      ram->stats().returned >= 2,"repeat cycle: return/release counters track the rhythm");
            }
            CHECK(ram->stats().file_reads == reads,"exchange/lookup never reopens source files");
            CHECK(locked == ram->stats().locked_bytes,"all live committed RAM has lock ownership");
        }
        CHECK(locked == 0,"resident destruction releases all locks");
    }
    // ---- direct-DMA eligibility (R1 prototype, STRATA_EXL3_RESIDENT_DIRECT_DMA) ------------
    // The capability is the BACKING's own answer for bytes it committed: false for every
    // default backing and every unproven configuration, true only for the full arena whose
    // registrar declared its ranges GPU-registered, and per-SPAN so a DMA never reads across
    // two registrations.  All CPU-only (a fake registrar stands in for cudaHostRegister; the
    // gpu_registered flag is what the production registrar sets, and it stays off here unless
    // the case says otherwise - an unpinned fake must never qualify).
    {
        const strata::Exl3ExpertBacking* raw = store.get();
        CHECK(!raw->direct_dma_pinned(nullptr, 4096), "default backing: null is not a range");
        CHECK(!raw->direct_dma_pinned("stack", 1), "default backing (packed store mmap): never eligible");
        // Read-only open (allow_build=false, the R1 microbench's contract): a MISSING
        // generation is a hard refusal, and the refusal writes NOTHING - not the cache dir,
        // not a generation dir, not a partial.  The fixture cache root is its own directory
        // under the temporary fixture; the real asset cache is never touched.
        {
            const fs::path missing = root / "band8-missing-cache";
            fs::remove_all(missing);
            Exl3Band8BuildOptions ro;
            ro.allow_build = false;
            CHECK_THROW(strata::Exl3Band8Store::build_or_open(fx.dir.string(), missing.string(),
                                                              fx.expert_prefixes, ro),
                        "read-only open refused");
            CHECK(!fs::exists(missing), "read-only refusal created no cache directory");
            // A PRESENT generation opens unchanged under allow_build=false (no rebuild, no
            // lock, no partial), and a fingerprint-mismatched generation is refused rather
            // than rebuilt: point the read-only open at the existing cache with a DIFFERENT
            // prefix list (order is part of the fingerprint).
            auto reused = strata::Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(),
                                                                fx.expert_prefixes, ro);
            CHECK(reused && reused->record_count() > 0,
                  "read-only open reuses an existing generation");
            // The same read-only open succeeds with the cache made PERMISSION-read-only:
            // nothing in the reuse path needs write access (no lock, no partial, no dir
            // creation), so a 0555 root must open cleanly.  Permissions are restored
            // afterwards; the generation count is checked to have stayed put throughout.
            size_t before_ro = 0;
            for (auto& entry : fs::directory_iterator(cache)) ++before_ro;
            std::error_code perm;
            fs::permissions(cache, fs::perms::owner_read | fs::perms::owner_exec |
                                    fs::perms::group_read | fs::perms::group_exec |
                                    fs::perms::others_read | fs::perms::others_exec, perm);
            if (!perm) {
                auto locked_open = strata::Exl3Band8Store::build_or_open(fx.dir.string(),
                                                                         cache.string(),
                                                                         fx.expert_prefixes, ro);
                CHECK(locked_open && locked_open->record_count() == reused->record_count(),
                      "read-only open works under read-only cache permissions");
                fs::permissions(cache, fs::perms::owner_all | fs::perms::group_read |
                                         fs::perms::group_exec | fs::perms::others_read |
                                         fs::perms::others_exec, perm);
            }
            size_t after_ro = 0;
            for (auto& entry : fs::directory_iterator(cache)) ++after_ro;
            CHECK(before_ro == after_ro, "the permissions probe created no generation");
            std::vector<std::string> reordered = fx.expert_prefixes;
            std::reverse(reordered.begin(), reordered.end());
            size_t generations_before = 0;
            for (auto& entry : fs::directory_iterator(cache)) ++generations_before;
            CHECK_THROW(strata::Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(),
                                                              reordered, ro),
                        "read-only open refused");
            size_t generations_after = 0;
            for (auto& entry : fs::directory_iterator(cache)) ++generations_after;
            CHECK(generations_before == generations_after,
                  "a fingerprint-mismatched read-only open wrote no new generation");
        }
        uint64_t locked = 0;
        strata::Exl3ResidentBacking::Options opt;
        opt.available_bytes = 1ull<<32; opt.headroom_bytes = 0;
        opt.lock = [&](void*,size_t n,std::string&) { locked += n; return true; };
        opt.unlock = [&](void*,size_t n) { locked -= n; };
        {
            auto ram = strata::Exl3ResidentBacking::create(fx.dir.string(),resident_groups,store);
            Exl3Model model(fx.dir.string(),ram);
            const auto foreign = model.linear(resident_groups[0].projections[0]);
            CHECK(!ram->direct_dma_pinned(foreign.trellis, 1),
                  "resident before activation: never eligible");
            ram->activate({},opt);
            const auto arena = model.linear(resident_groups[0].projections[0]);
            CHECK(arena.trellis != foreign.trellis, "activation repoints at the arena");
            CHECK(!ram->direct_dma_pinned(arena.trellis, 1),
                  "fake registrar without gpu_registered: the pinned answer stays off");
            CHECK(!ram->direct_dma_pinned(foreign.trellis, 16),
                  "a pointer into another mapping (the store's mmap) is refused");
        }
        CHECK(locked == 0,"eligibility probes lock and unlock nothing");
        {
            opt.gpu_registered = true;
            auto ram = strata::Exl3ResidentBacking::create(fx.dir.string(),resident_groups,store);
            Exl3Model model(fx.dir.string(),ram);
            ram->activate({},opt);
            std::vector<const void*> starts;
            std::vector<size_t> spans;
            for (const auto& group : resident_groups) {
                size_t expert_bytes = 0; const void* start = nullptr;
                for (const auto& base : group.projections) {
                    const auto a = model.linear(base);
                    const size_t sizes[] = {(size_t)a.ki*a.nj*a.words()*2,(size_t)a.in()*2,(size_t)a.out()*2};
                    const void* planes[] = {a.trellis,a.suh,a.svh};
                    for (int p = 0; p < 3; ++p) {
                        if (start == nullptr) start = planes[p];
                        expert_bytes += sizes[p];
                        // every plane, and every prefix of one plane, is eligible on its own
                        CHECK(ram->direct_dma_pinned(planes[p], sizes[p]),
                              "whole plane eligible inside its expert's registration");
                        CHECK(ram->direct_dma_pinned(planes[p], sizes[p] > 1 ? 1 : sizes[p]),
                              "subrange (1 byte) eligible inside the same registration");
                    }
                }
                starts.push_back(start); spans.push_back(expert_bytes);
                CHECK(ram->direct_dma_pinned(start, expert_bytes),
                      "the expert's full nine-plane span is eligible (source contiguity target)");
            }
            // Bounds and cross-registration refusals.  ALL probe arithmetic goes through
            // uintptr_t: forming (uint8_t*)p + k / p - k on a FOREIGN pointer is itself
            // undefined, so the tests never do it - they pass INTEGER-derived addresses,
            // which is exactly what the implementation is required to tolerate.  An
            // expert's span is page-rounded, so a range past its ACTUAL bytes still fits
            // the page pad; only crossing into the NEXT expert's registration must fail,
            // plus everything outside the arena.
            const uintptr_t base0 = (uintptr_t) starts[0];
            const uintptr_t base1 = (uintptr_t) starts[1];
            const uintptr_t arena_end = (uintptr_t) starts.back() + spans.back();
            CHECK(ram->direct_dma_pinned(starts[0], spans[0] + 1) ||
                      spans[0] % 4096 == 0,
                  "page pad of one registration is still that registration's bytes");
            CHECK(!ram->direct_dma_pinned((const void*) base0, (size_t) (base1 - base0) + 1),
                  "a range crossing into the next expert's registration is refused");
            CHECK(!ram->direct_dma_pinned((const void*) (base0 + 1), SIZE_MAX - 1),
                  "overflow-length range refused (no wraparound)");
            CHECK(!ram->direct_dma_pinned((const void*) (base0 >= 4096 ? base0 - 4096 : 0), 8192),
                  "range below the arena refused");
            CHECK(!ram->direct_dma_pinned((const void*) (arena_end + (1u << 20)), 4096),
                  "range past the arena end refused");
            CHECK(!ram->direct_dma_pinned((const void*) ((uintptr_t) - 4096), 4096),
                  "address-space limit probe refused");
            CHECK(!ram->direct_dma_pinned(nullptr, 4096), "null pointer refused");
            CHECK(!ram->direct_dma_pinned(starts[0], 0), "empty range refused");
            {   // FOREIGN stack and heap addresses: legal API inputs, never eligible
                volatile int on_stack = 0;
                CHECK(!ram->direct_dma_pinned((const void*) &on_stack, sizeof on_stack),
                      "foreign stack address refused");
                auto on_heap = std::vector<uint64_t>(64, 1);
                CHECK(!ram->direct_dma_pinned((const void*) on_heap.data(), 64 * sizeof(uint64_t)),
                      "foreign heap address refused");
            }
            // The mixed-rate layer (K3/K4 experts in layer 2) still has per-expert contiguous
            // source planes - capability is per-registration; whether the stream can COALESCE
            // is a destination question the engine answers separately.
            size_t mixed = 0;
            for (size_t g = 8; g < resident_groups.size(); ++g)
                if (resident_groups[g].layer == 2) ++mixed;
            CHECK(mixed > 0, "fixture has mixed-rate layer-2 experts");
            for (size_t g = 0; g < resident_groups.size(); ++g)
                if (resident_groups[g].layer == 2)
                    CHECK(ram->direct_dma_pinned(starts[g], spans[g]),
                          "mixed-rate expert's own span is eligible (rates do not change commitment)");
            // A mutation (the peer exchange's release) disarms the whole answer.
            ram->release_to_gpu(resident_groups[1].layer, resident_groups[1].expert);
            CHECK(!ram->direct_dma_pinned(starts[0], spans[0]),
                  "exchange mutation disarms direct DMA for the WHOLE arena");
        }
        {
            // Peer complement: GPU-only experts exist, so the arena is not immutable - the
            // answer must stay off even with the GPU-registered registrar.
            opt.gpu_registered = true;
            auto ram = strata::Exl3ResidentBacking::create(fx.dir.string(),resident_groups,store);
            Exl3Model model(fx.dir.string(),ram);
            ram->activate({{0,0}},opt);
            const auto second = model.linear(resident_groups[3].projections[0]);
            CHECK(ram->resident(0,1), "complement keeps the other experts in RAM");
            CHECK(!ram->direct_dma_pinned(second.trellis, 16),
                  "peer-complement arena is never direct-DMA eligible");
        }
        CHECK(locked == 0,"eligibility cases leave no locks outstanding");
    }
    {
        auto source = strata::Exl3Band8Store::build_or_open(fx.dir.string(),cache.string(),fx.expert_prefixes);
        std::weak_ptr<const strata::Exl3ExpertBacking> weak_source = source;
        auto ram = strata::Exl3ResidentBacking::create(fx.dir.string(),resident_groups,source);
        source.reset();
        CHECK(!weak_source.expired(),"resident keeps source alive for initial GPU placement");
        strata::Exl3ResidentBacking::Options opt;
        opt.available_bytes = 1ull<<32; opt.headroom_bytes = 0;
        opt.lock = [&](void*,size_t,std::string&) {
            CHECK(weak_source.expired(),"source mmap retired BEFORE any arena registration");
            return true;
        };
        opt.unlock = [](void*,size_t) {};
        ram->activate({},opt);
        Exl3Model model(fx.dir.string(),ram), ref(fx.dir.string(),store);
        const auto a = model.linear(fx.expert_prefixes[0]), b = ref.linear(fx.expert_prefixes[0]);
        CHECK(std::memcmp(a.trellis,b.trellis,(size_t)b.ki*b.nj*b.words()*2) == 0,
              "direct startup load works after source mapping retirement");
    }
    {
        auto ram = strata::Exl3ResidentBacking::create(fx.dir.string(),resident_groups);
        strata::Exl3ResidentBacking::Options opt;
        opt.available_bytes = 1; opt.headroom_bytes = 0;
        opt.lock = [](void*,size_t,std::string& e) { e="injected lock refusal"; return false; };
        opt.unlock = [](void*,size_t) {};
        CHECK_THROW(ram->activate({},opt),"insufficient available RAM");
        opt.available_bytes = 1ull<<32;
        CHECK_THROW(ram->activate({},opt),"injected lock refusal");
        CHECK(!ram->active(),"failed activation is never published");
        strata::Exl3Linear matrix;
        CHECK_THROW(ram->linear(fx.expert_prefixes[0],matrix),"activation incomplete");
        std::string path; size_t offset = 0, bytes = 0;
        CHECK_THROW(ram->locate(fx.expert_prefixes[0]+".trellis",path,offset,bytes),"activation incomplete");
    }
    {
        uint64_t locked = 0; int calls = 0;
        strata::Exl3ResidentBacking::Options opt;
        opt.available_bytes = 1ull<<32; opt.headroom_bytes = 0;
        opt.lock = [&](void*,size_t n,std::string& error) {
            if (++calls == 3) { error="third lock fault"; return false; }
            locked += n; return true;
        };
        opt.unlock = [&](void*,size_t n) { locked -= n; };
        {
            auto ram = strata::Exl3ResidentBacking::create(fx.dir.string(),resident_groups);
            CHECK_THROW(ram->activate({},opt),"third lock fault");
            CHECK(!ram->active() && locked > 0,"partial startup never publishes RAM backing");
        }
        CHECK(locked == 0,"partial startup unwinds every successful registration");
        FixtureModel broken;
        broken.build(root / "resident-truncated-source",1,1);
        std::vector<strata::Exl3ResidentBacking::Group> one{{0,0,
            {broken.expert_prefixes[0],broken.expert_prefixes[1],broken.expert_prefixes[2]}}};
        auto ram = strata::Exl3ResidentBacking::create(broken.dir.string(),one);
        fs::resize_file(broken.dir / "fixture-00001-of-00002.safetensors",1);
        calls = 0;
        CHECK_THROW(ram->activate({},opt),"short/error load");
        CHECK(!ram->active(),"short direct read is fatal before READY");
    }

    // ---- runtime in-memory repack (Exl3ResidentRepack::ToBand8): RAM only, no artifact ----
    {
        const size_t root_bytes_before = tree_bytes(root);
        uint64_t locked = 0;
        strata::Exl3ResidentBacking::Options opt;
        opt.available_bytes = 1ull<<32; opt.headroom_bytes = 0;
        opt.lock = [&](void*,size_t n,std::string&) { locked += n; return true; };
        opt.unlock = [&](void*,size_t n) { locked -= n; };
        std::shared_ptr<strata::Exl3ResidentBacking> ram;
        try {
            ram = strata::Exl3ResidentBacking::create(fx.dir.string(),resident_groups,nullptr,
                                                      strata::Exl3ResidentRepack::ToBand8);
        } catch (const std::exception& e) {
            CHECK(false, (std::string("repack create failed: ") + e.what()).c_str());
        }
        CHECK(ram && ram->layout() == Exl3TrellisLayout::Band8,
              "repack backing reports the FINAL Band8 tag from create() on");
        {   // Between create() and activate() the backing must not serve anything: the
            // metadata is native while the mode probe says Band8.
            strata::Exl3Linear L{};
            CHECK_THROW(ram->linear(fx.expert_prefixes[0],L), "not converted yet");
            std::string p; size_t o = 0, n = 0;
            CHECK_THROW(ram->locate(fx.expert_prefixes[0]+".trellis",p,o,n), "no file tier");
        }
        ram->activate({},opt);
        CHECK(ram->active() && ram->memory_only(),
              "repack activation publishes the memory-only arena");
        // Every plane of every expert, bitwise, against BOTH the offline store's packed
        // bytes (built above over the same fixture) and the independent reference
        // permutation - the runtime conversion must equal the proven offline one.
        Exl3Model via_store(fx.dir.string(), store), via_ram(fx.dir.string(), ram);
        Exl3Model native_view(fx.dir.string());
        uint64_t want_reads = 0;
        for (const auto& group : resident_groups) {
            for (size_t pi = 0; pi < 3; ++pi) {
                const std::string& base = group.projections[pi];
                const strata::Exl3Linear a = via_ram.linear(base), b = via_store.linear(base);
                const size_t sizes[] = {(size_t)b.ki*b.nj*b.words()*2,(size_t)b.in()*2,(size_t)b.out()*2};
                const uint16_t* planes_a[] = {a.trellis,a.suh,a.svh};
                const uint16_t* planes_b[] = {b.trellis,b.suh,b.svh};
                CHECK(a.layout == Exl3TrellisLayout::Band8 && b.layout == Exl3TrellisLayout::Band8 &&
                          a.trellis != b.trellis && a.trellis != nullptr,
                      "repack arena hands out its own Band8 pointers");
                for (int q = 0; q < 3; ++q) {
                    CHECK(std::memcmp(planes_a[q],planes_b[q],sizes[q]) == 0,
                          "repack plane bytes == the offline store's packed bytes");
                }
                // trellis: the independent reference permutation as a second witness
                std::vector<uint8_t> raw = source_bytes(fx.dir, base + ".trellis");
                std::vector<uint16_t> native(raw.size()/2);
                std::memcpy(native.data(),raw.data(),raw.size());
                const std::vector<uint16_t> want = reference_band8(native,a.ki,a.nj,a.words());
                CHECK(std::memcmp(a.trellis,want.data(),want.size()*2) == 0,
                      "repack trellis == the independent reference permutation");
                want_reads += 3;   // exactly one bounded read per plane
            }
        }
        // Exact plane-count reads; the byte total is page-rounded per bounded pread (the
        // final read of a shard may stop at EOF inside its padded tail, so bound it).
        uint64_t plane_bytes_min = 0, plane_bytes_max = 0;
        for (const auto& group : resident_groups)
            for (const auto& base : group.projections) {
                std::string p; size_t o = 0, n = 0;
                for (const char* suffix : {".trellis", ".suh", ".svh"}) {
                    if (!native_view.locate(base+suffix,p,o,n)) throw std::runtime_error("plane");
                    plane_bytes_min += n;
                    plane_bytes_max += ((o & size_t(4095)) + n + 4095) & ~size_t(4095);
                }
            }
        const auto st = ram->stats();
        CHECK(st.file_reads == want_reads,
              "repack stats: exactly one bounded read per plane (no expert-sized I/O)");
        CHECK(st.file_bytes >= plane_bytes_min && st.file_bytes <= plane_bytes_max,
              "repack stats: native byte count within the page-rounded read bounds");
        CHECK(locked == st.locked_bytes && st.virtual_bytes == st.locked_bytes,
              "repack arena: full-arena lock ownership matches the file-backed path");
        CHECK(tree_bytes(root) == root_bytes_before,
              "the runtime repack wrote NO file (model dir, cache root and scratch all unchanged)");
        CHECK(!fs::exists(root / "cache-root-repack") , "no cache root is created by the runtime path");
        // gpu-only (peer complement) is refused explicitly, before any lock is taken.
        {
            auto fresh = strata::Exl3ResidentBacking::create(fx.dir.string(),resident_groups,nullptr,
                                                             strata::Exl3ResidentRepack::ToBand8);
            CHECK_THROW(fresh->activate({{0,0}},opt), "FULL arena");
            CHECK(!fresh->active() && locked == st.locked_bytes, "refused complement leaves RAM untouched");
        }
        // create-time refusals: a packed source backing, a non-mul1 projection, an
        // unverified rate - all BEFORE any arena byte exists.
        CHECK_THROW(strata::Exl3ResidentBacking::create(fx.dir.string(),resident_groups,store,
                        strata::Exl3ResidentRepack::ToBand8), "already canonical");
        auto one_triple_dir = [&](const fs::path& d, int last_dim, bool mul1) {
            fs::create_directories(d);
            std::map<std::string,std::string> wm;
            std::vector<FixtureTensor> t;
            const int64_t ki = 16, nj = 8;
            for (const char* suffix : {".trellis", ".suh", ".svh"}) {
                const std::string name = "model.layers.0.mlp.experts.0.gate_proj" + std::string(suffix);
                if (suffix == std::string(".trellis")) {
                    const size_t bytes = (size_t)ki*nj*last_dim*2;
                    t.push_back({name,"I16",{ki,nj,last_dim},synth_bytes(name,bytes)});
                } else if (suffix == std::string(".suh")) {
                    t.push_back({name,"F16",{ki*16},synth_bytes(name,(size_t)ki*32)});
                } else {
                    t.push_back({name,"F16",{nj*16},synth_bytes(name,(size_t)nj*32)});
                }
                wm.emplace(name,"fixture-00001-of-00001.safetensors");
            }
            if (mul1) {
                t.push_back({"model.layers.0.mlp.experts.0.gate_proj.mul1","U8",{1},
                             synth_bytes("mul1",1)});
                wm.emplace("model.layers.0.mlp.experts.0.gate_proj.mul1","fixture-00001-of-00001.safetensors");
            }
            write_shard(d / "fixture-00001-of-00001.safetensors", t);
            write_index(d / "model.safetensors.index.json", wm);
        };
        {
            const fs::path d = root / "repack-no-mul1";
            one_triple_dir(d, 32, false);
            std::vector<strata::Exl3ResidentBacking::Group> one{{0,0,
                {"model.layers.0.mlp.experts.0.gate_proj",
                 "model.layers.0.mlp.experts.0.up_proj",
                 "model.layers.0.mlp.experts.0.down_proj"}}};
            CHECK_THROW(strata::Exl3ResidentBacking::create(d.string(),one,nullptr,
                            strata::Exl3ResidentRepack::ToBand8), "mul1");
        }
        {
            const fs::path d = root / "repack-k5";
            one_triple_dir(d, 80, true);   // K5: outside the verified set
            std::vector<strata::Exl3ResidentBacking::Group> one{{0,0,
                {"model.layers.0.mlp.experts.0.gate_proj",
                 "model.layers.0.mlp.experts.0.up_proj",
                 "model.layers.0.mlp.experts.0.down_proj"}}};
            CHECK_THROW(strata::Exl3ResidentBacking::create(d.string(),one,nullptr,
                            strata::Exl3ResidentRepack::ToBand8), "verified set");
        }
        // Memory failures stay explicit: short available RAM and a failing lock unwind
        // exactly like the file-backed path, and a truncated source is fatal pre-READY.
        {
            auto fresh = strata::Exl3ResidentBacking::create(fx.dir.string(),resident_groups,nullptr,
                                                             strata::Exl3ResidentRepack::ToBand8);
            strata::Exl3ResidentBacking::Options bad = opt;
            bad.available_bytes = 1;
            CHECK_THROW(fresh->activate({},bad), "insufficient available RAM");
            CHECK(!fresh->active(), "refused RAM budget never publishes");
        }
        {
            int calls = 0; uint64_t part = 0;
            strata::Exl3ResidentBacking::Options fail3 = opt;
            fail3.lock = [&](void*,size_t n,std::string& error) {
                if (++calls == 3) { error="third lock fault"; return false; }
                part += n; return true;
            };
            fail3.unlock = [&](void*,size_t n) { part -= n; };
            {
                auto fresh = strata::Exl3ResidentBacking::create(fx.dir.string(),resident_groups,nullptr,
                                                                 strata::Exl3ResidentRepack::ToBand8);
                CHECK_THROW(fresh->activate({},fail3), "third lock fault");
                CHECK(!fresh->active() && part > 0, "partial repack never publishes RAM backing");
            }
            CHECK(part == 0, "partial repack unwinds every successful registration");
        }
        {
            FixtureModel broken;
            broken.build(root / "repack-truncated-source",1,1);
            std::vector<strata::Exl3ResidentBacking::Group> one{{0,0,
                {broken.expert_prefixes[0],broken.expert_prefixes[1],broken.expert_prefixes[2]}}};
            auto fresh = strata::Exl3ResidentBacking::create(broken.dir.string(),one,nullptr,
                                                             strata::Exl3ResidentRepack::ToBand8);
            fs::resize_file(broken.dir / "fixture-00001-of-00002.safetensors",1);
            CHECK_THROW(fresh->activate({},opt), "short/error load");
            CHECK(!fresh->active(), "short native read is fatal before READY");
        }
    }

    // -- two model instances share one owner; legacy model stays Native -----------------
    {
        Exl3Model m1(fx.dir.string(), store), m2(fx.dir.string(), store);
        const strata::Exl3Linear a = m1.linear(fx.expert_prefixes[2]);
        const strata::Exl3Linear b = m2.linear(fx.expert_prefixes[2]);
        CHECK(std::memcmp(a.trellis, b.trellis, (size_t)a.ki * a.nj * a.words() * 2) == 0,
              "two instances: identical packed trellis bytes");
        CHECK(std::memcmp(a.suh, b.suh, (size_t)a.ki * 32) == 0 &&
              std::memcmp(a.svh, b.svh, (size_t)a.nj * 32) == 0, "two instances: identical scale bytes");
        // Legacy model (no backing): unchanged native behavior and source-file locate.
        Exl3Model legacy(fx.dir.string());
        const strata::Exl3Linear n = legacy.linear(fx.expert_prefixes[2]);
        CHECK(n.layout == Exl3TrellisLayout::Native, "legacy model layout Native");
        CHECK(std::memcmp(n.trellis, a.trellis, (size_t)n.ki * n.nj * n.words() * 2) != 0,
              "legacy native bytes are the un-permuted source (differ from packed)");
        std::string p;
        size_t o = 0, nb = 0;
        CHECK(legacy.locate(fx.expert_prefixes[7] + ".trellis", p, o, nb),
              "legacy locate still resolves the source shard");
        CHECK(p.find("fixture-") != std::string::npos, "legacy locate points at the source, not the cache");
        // close_shards() must not disturb the packed mapping (source mmaps are separate).
        std::vector<uint8_t> before((const uint8_t*)a.trellis,
                                    (const uint8_t*)a.trellis + (size_t)a.ki * a.nj * a.words() * 2);
        m1.close_shards();
        const strata::Exl3Linear after = m1.linear(fx.expert_prefixes[2]);
        CHECK(after.layout == Exl3TrellisLayout::Band8 &&
              std::memcmp(after.trellis, before.data(), before.size()) == 0,
              "packed bytes valid after close_shards");
        CHECK(m1.locate(fx.expert_prefixes[2] + ".trellis", p, o, nb) &&
              p == store->payload_path(), "packed locate unchanged after close_shards");
        // Dense projection falls back to the native source path through the backed model.
        const strata::Exl3Linear dense = m1.linear("model.layers.0.self_attn.q_proj");
        CHECK(dense.layout == Exl3TrellisLayout::Native, "uncovered dense projection stays Native");
        CHECK(m1.locate("model.layers.0.input_layernorm.weight", p, o, nb) &&
              p.find("fixture-") != std::string::npos, "non-trellis tensors resolve in the source");
    }

    // -- reopen reuses exactly (no payload scan, no rebuild) ----------------------------
    {
        const ino_t ino = inode_of(gen / "payload.bin");
        struct stat sb {};
        ::stat((gen / "payload.bin").c_str(), &sb);
        const time_t mtime = sb.st_mtim.tv_sec;
        ::sleep(1);   // a rebuild would produce a fresh inode; reuse must not
        std::string p0;
        size_t o0 = 0, b0 = 0;
        store->locate(fx.expert_prefixes[5] + ".trellis", p0, o0, b0);
        auto again = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(), fx.expert_prefixes, opts);
        std::string p1;
        size_t o1 = 0, b1 = 0;
        again->locate(fx.expert_prefixes[5] + ".trellis", p1, o1, b1);
        CHECK(inode_of(gen / "payload.bin") == ino, "reopen reuses the same payload file");
        CHECK(o1 == o0 && b1 == b0 && p1 == p0, "reopen reports identical record offsets");
        ::stat((gen / "payload.bin").c_str(), &sb);
        CHECK(sb.st_mtim.tv_sec == mtime, "reopen did not rewrite the payload");
    }

    // -- refusal: requested-but-missing coverage never falls back -----------------------
    {
        const std::vector<std::string> dup{fx.expert_prefixes[0], fx.expert_prefixes[0]};
        CHECK_THROW(auto s = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(), dup, opts);
                    (void)s, "duplicate");
        // A prefix absent from the source index is a constructor-time error, not a later miss.
        const std::vector<std::string> absent{"model.layers.9.mlp.experts.0.gate_proj"};
        CHECK_THROW(auto s = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(), absent, opts);
                    (void)s, "no .trellis");
    }

    // -- invalid-source variants (each its own tiny model dir) --------------------------
    auto bad_model = [&](const std::string& name, int last_dim, bool with_mul1, bool drop_suh,
                         int nj, const char* needle) {
        const fs::path d = root / name;
        fs::create_directories(d);
        const std::string p = "model.layers.0.mlp.experts.0.gate_proj";
        std::map<std::string, std::string> wmap;
        std::vector<FixtureTensor> ts;
        const int64_t ki = 16;
        const size_t tre = (size_t)ki * nj * last_dim * 2;
        auto add = [&](const std::string& n, const std::string& dt, std::vector<int64_t> sh, size_t bytes) {
            wmap.emplace(n, "fixture-00001-of-00001.safetensors");
            ts.push_back({n, dt, std::move(sh), synth_bytes(n, bytes)});
        };
        add(p + ".trellis", "I16", {ki, nj, last_dim}, tre);
        if (!drop_suh) add(p + ".suh", "F16", {ki * 16}, (size_t)ki * 32);
        add(p + ".svh", "F16", {nj * 16}, (size_t)nj * 32);
        if (with_mul1) add(p + ".mul1", "U8", {1}, 1);
        write_shard(d / "fixture-00001-of-00001.safetensors", ts);
        write_index(d / "model.safetensors.index.json", wmap);
        CHECK_THROW(auto s = Exl3Band8Store::build_or_open(d.string(), (root / (name + "-cache")).string(),
                                                           std::vector<std::string>{p}, opts);
                    (void)s, needle);
    };
    bad_model("rate-k35", 56, true, false, 8, "verified set");     // K=3.5: unverified half rate
    bad_model("rate-k8", 128, true, false, 8, "verified set");     // K=8
    bad_model("no-mul1", 32, false, false, 8, "mul1");
    bad_model("nj-not-8", 32, true, false, 4, "H128");             // nj=4 tiles: not bandable
    bad_model("incomplete", 32, true, true, 8, "complete");        // missing .suh in the index

    // -- corrupted manifest: bounds checks fire BEFORE any pointer arithmetic -----------
    {
        const ino_t ino = inode_of(gen / "payload.bin");
        patch_manifest(gen, 3, 0 /*record offset*/, 0);          // reserved zero offset
        auto s = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(), fx.expert_prefixes, opts);
        CHECK(inode_of(gen / "payload.bin") != ino && s->record_count() == fx.expert_prefixes.size() * 3,
              "offset-0 record invalidates the cache (rebuilt, never reused)");
    }
    {
        const ino_t ino = inode_of(gen / "payload.bin");
        patch_manifest(gen, 3, 0, 0xFFFFFFFFFFFFFF00ull);        // near-wrap offset
        auto s = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(), fx.expert_prefixes, opts);
        CHECK(inode_of(gen / "payload.bin") != ino, "wrap-offset record invalidates the cache");
    }
    {
        const ino_t ino = inode_of(gen / "payload.bin");
        patch_manifest(gen, 3, 8 /*record bytes*/, 0xFFFFFFFFFFFFFFFFull);  // size overflow
        auto s = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(), fx.expert_prefixes, opts);
        CHECK(inode_of(gen / "payload.bin") != ino, "byte-count overflow invalidates the cache");
    }
    {
        // Corrupt one record NAME so the sorted-unique invariant breaks (duplicate/unsorted).
        const ino_t ino = inode_of(gen / "payload.bin");
        fs::path mp = gen / "manifest.bin";
        std::vector<uint8_t> b;
        {
            std::ifstream in(mp, std::ios::binary);
            b.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        }
        const size_t pool_beg = 80 + fx.expert_prefixes.size() * 3 * 40;
        for (size_t i = pool_beg; i + 4 < b.size(); ++i)   // inside the name pool
            if (b[i] == 'g' && b[i + 1] == 'a' && b[i + 2] == 't' && b[i + 3] == 'e') {
                b[i] = 'z'; b[i + 1] = 'z'; b[i + 2] = 'z'; b[i + 3] = 'z';
                break;
            }
        {
            std::ofstream out(mp, std::ios::binary | std::ios::trunc);
            out.write((const char*)b.data(), (std::streamsize)b.size());
        }
        auto s = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(), fx.expert_prefixes, opts);
        CHECK(inode_of(gen / "payload.bin") != ino && s->record_count() == fx.expert_prefixes.size() * 3,
              "name corruption (unsorted/duplicate) invalidates the cache");
    }
    {
        // Truncated payload with a published manifest: detected via size mismatch, rebuilt.
        const ino_t ino = inode_of(gen / "payload.bin");
        fs::resize_file(gen / "payload.bin", fs::file_size(gen / "payload.bin") - 1);
        auto s = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(), fx.expert_prefixes, opts);
        CHECK(inode_of(gen / "payload.bin") != ino && s->record_count() == fx.expert_prefixes.size() * 3,
              "truncated payload invalidates the cache");
    }
    {
        // A NEWER manifest version is refused outright - never overwritten, never reused.
        patch_manifest(gen, 0, 0, 0);   // restore clean state first (this patch is repaired below)
        fs::path mp = gen / "manifest.bin";
        std::vector<uint8_t> b;
        {
            std::ifstream in(mp, std::ios::binary);
            b.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        }
        const uint32_t v2 = 2;
        std::memcpy(b.data() + 8, &v2, 4);
        {
            std::ofstream out(mp, std::ios::binary | std::ios::trunc);
            out.write((const char*)b.data(), (std::streamsize)b.size());
        }
        const ino_t ino = inode_of(gen / "payload.bin");
        CHECK_THROW(auto s = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(),
                                                           fx.expert_prefixes, opts);
                    (void)s, "unsupported manifest version");
        CHECK(inode_of(gen / "payload.bin") == ino, "refused generation is left untouched");
        // Repair by rebuilding from the still-valid source.
        fs::remove(mp);
        auto s = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(), fx.expert_prefixes, opts);
        CHECK(s->record_count() == fx.expert_prefixes.size() * 3, "cache rebuilt after refusal");
    }

    // -- fingerprint: a changed source selects a NEW generation; the old one is never touched.
    //    Detection is via bytes+size+mtime (a change detector, not a cryptographic identity).
    const struct timespec idx_mtime0 = mtime_of(fx.dir / "model.safetensors.index.json");
    const struct timespec shard_mtime0 = mtime_of(fx.dir / "fixture-00002-of-00002.safetensors");
    fs::path gen2;
    {
        touch_mtime(fx.dir / "model.safetensors.index.json");
        auto s = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(), fx.expert_prefixes, opts);
        gen2 = gen_of(s);
        CHECK(gen2 != gen, "index mtime change builds a new generation");
        CHECK(fs::exists(gen / "payload.bin") && fs::exists(gen / "manifest.bin"),
              "old generation files are never deleted by a new build");
        CHECK(s->fingerprint() != 0, "new generation reports its fingerprint");
    }
    {
        touch_mtime(fx.dir / "fixture-00002-of-00002.safetensors");
        auto s = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(), fx.expert_prefixes, opts);
        CHECK(gen_of(s) != gen2,
              "referenced shard mtime change builds another generation");
    }
    {
        // Restoring the original index AND shard mtimes restores the original fingerprint: the
        // OLD generation is reused exactly (no rebuild), proving generations are immutable and
        // selected by fingerprint, not by "most recent".
        touch_exact(fx.dir / "model.safetensors.index.json", idx_mtime0);
        touch_exact(fx.dir / "fixture-00002-of-00002.safetensors", shard_mtime0);
        auto s = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(), fx.expert_prefixes, opts);
        CHECK(gen_of(s) == gen, "restored fingerprint reuses the old generation");
        CHECK(inode_of(gen / "payload.bin") == inode_of(gen / "payload.bin") &&
              fs::exists(gen / "manifest.bin"), "old generation still complete");
    }

    // -- builder lock: one builder key per GENERATION.  A pure reuse-open deliberately takes
    //    no lock (readers never block); force the build path inside this generation by removing
    //    its manifest (same fingerprint => deterministic in-place repair).  ------------------
    {
        fs::path lk = gen / "build.lock";
        fs::remove(gen / "manifest.bin");           // force same-generation rebuild
        {
            std::ofstream f(lk);
            f << "pid=" << ::getpid() << "\n";     // we are alive: must refuse
        }
        CHECK_THROW(auto s = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(),
                                                           fx.expert_prefixes, opts);
                    (void)s, "another builder");
        CHECK(fs::exists(lk), "live lock is not removed");
        fs::remove(lk);
    }
    {
        fs::path lk = gen / "build.lock";
        fs::remove(gen / "manifest.bin");
        {
            std::ofstream f(lk);
            f << "pid=4194303\n";                  // parseable and provably dead: reclaimed
        }
        auto s = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(), fx.expert_prefixes, opts);
        CHECK(!fs::exists(lk), "stale lock (valid dead pid) reclaimed");
        CHECK(gen_of(s) == gen, "stale-lock recovery stays in-generation");
    }
    {
        // Empty/unparseable lock = BUSY, not garbage: the creator may be mid-write of its pid
        // line, so it must not be blindly reclaimed while possibly alive.
        fs::path lk = gen / "build.lock";
        fs::remove(gen / "manifest.bin");
        { std::ofstream f(lk); f << ""; }           // empty: creator mid-write or crashed?
        CHECK_THROW(auto s = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(),
                                                           fx.expert_prefixes, opts);
                    (void)s, "refusing");
        CHECK(fs::exists(lk), "unparseable lock is left in place");
        fs::remove(lk);
    }
    {
        // Same-fingerprint repair is byte-stable: the deterministic layout reproduces the exact
        // payload bytes and offsets, so the payload rename over the old (removed) manifest is a
        // content no-op and old readers' paths stay truthful.
        fs::path lk = gen / "build.lock";
        fs::remove(gen / "manifest.bin");
        auto s = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(), fx.expert_prefixes, opts);
        CHECK(gen_of(s) == gen, "repair stays in-generation");
        CHECK(s->record_count() == fx.expert_prefixes.size() * 3, "repair republishes all records");
    }

    // -- interrupted build leaves no publishable state; the root is never clobbered ---------
    {
        fs::path c2 = root / "cache-interrupted";
        fs::create_directories(c2);
        { std::ofstream(c2 / "user-file.txt") << "not this cache's to delete"; }
        auto first = Exl3Band8Store::build_or_open(fx.dir.string(), c2.string(), fx.expert_prefixes, opts);
        const fs::path g2 = gen_of(first);
        fs::remove(g2 / "manifest.bin");            // simulate: payload committed, manifest not
        { std::ofstream(g2 / "payload.bin.partial", std::ios::binary) << "garbage"; }
        { std::ofstream(g2 / "manifest.bin.partial", std::ios::binary) << "garbage"; }
        auto s = Exl3Band8Store::build_or_open(fx.dir.string(), c2.string(), fx.expert_prefixes, opts);
        CHECK(s->record_count() == fx.expert_prefixes.size() * 3 &&
              fs::exists(gen_of(s) / "manifest.bin") &&
              !fs::exists(g2 / "payload.bin.partial"),
              "interrupted generation artifacts are cleared and replaced by a complete publish");
        CHECK(fs::exists(c2 / "user-file.txt"), "foreign files in the cache root are never deleted");
    }

    // -- low-disk preflight: clean failure reporting the byte gap ---------------------------
    {
        fs::path c3 = root / "cache-lowdisk";
        Exl3Band8BuildOptions low = opts;
        low.reported_free_bytes_for_tests = uint64_t{1};
        CHECK_THROW(auto s = Exl3Band8Store::build_or_open(fx.dir.string(), c3.string(),
                                                           fx.expert_prefixes, low);
                    (void)s, "missing");
        // The empty generation directory this failed build created is fine (its own artifact);
        // what must NOT exist anywhere under the root is a payload or partial payload.
        bool any_payload = false;
        for (const auto& e : fs::recursive_directory_iterator(c3))
            if (e.is_regular_file() && (e.path().filename().string().rfind("payload", 0) == 0))
                any_payload = true;
        CHECK(!any_payload, "low-disk failure leaves no payload/partial anywhere under the root");
    }

    // -- optional payload checksum: corruption is refused when verification is requested ----
    {
        fs::path c4 = root / "cache-checksum";
        Exl3Band8BuildOptions co = opts;
        co.record_payload_checksum = true;
        auto s = Exl3Band8Store::build_or_open(fx.dir.string(), c4.string(), fx.expert_prefixes, co);
        CHECK(s->record_count() == fx.expert_prefixes.size() * 3, "checksum build succeeds");
        // Flip one payload byte (in the middle of a trellis plane).
        fs::path pl = s->payload_path();
        std::vector<uint8_t> b;
        {
            std::ifstream in(pl, std::ios::binary);
            b.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        }
        b[b.size() / 2] ^= 0x55;
        {
            std::ofstream out(pl, std::ios::binary | std::ios::trunc);
            out.write((const char*)b.data(), (std::streamsize)b.size());
        }
        Exl3Band8BuildOptions vo = opts;
        vo.verify_payload = true;
        CHECK_THROW(auto t = Exl3Band8Store::build_or_open(fx.dir.string(), c4.string(),
                                                           fx.expert_prefixes, vo);
                    (void)t, "checksum mismatch");
        // Without verification the same damage is (documented) undetected on the fast path,
        // but bounds/metadata still validate - and a WITHOUT-checksum cache cannot verify:
        CHECK_THROW(auto t = Exl3Band8Store::build_or_open(fx.dir.string(), cache.string(),
                                                           fx.expert_prefixes, vo);
                    (void)t, "without a payload checksum");
    }


    // -- CONCURRENCY REGRESSION (the audit bug): two same-geometry models, one cache ROOT.
    //    Old flat design: builder B renamed payload.bin while A's manifest still stood; a new
    //    reader of A then validated old-manifest + new-payload (same sizes) and served WRONG
    //    weights. Generations fix it: A and B live in fingerprint-named directories and never
    //    share bytes.  -----------------------------------------------------------------------
    {
        FixtureModel fa, fb, fd;
        fa.build(root / "modelA", 3, 4, "A");
        fb.build(root / "modelB", 3, 4, "B");
        fd.build(root / "modelD", 3, 4, "D");
        const fs::path shared = root / "shared-root";
        auto storeA = Exl3Band8Store::build_or_open(fa.dir.string(), shared.string(),
                                                    fa.expert_prefixes, opts);
        const fs::path genA = gen_of(storeA);
        // Snapshot A's bytes through the public API (linear pointers + pread of locate spans).
        std::map<std::string, std::vector<uint8_t>> a_bytes;
        for (const char* plane : {".trellis", ".suh", ".svh"}) {
            std::string path;
            size_t off = 0, nb = 0;
            CHECK(storeA->locate(fa.expert_prefixes[2] + plane, path, off, nb), "A locate");
            a_bytes[plane] = pread_span(path, off, nb);
        }
        auto storeB = Exl3Band8Store::build_or_open(fb.dir.string(), shared.string(),
                                                    fb.expert_prefixes, opts);
        const fs::path genB = gen_of(storeB);
        CHECK(genA != genB && storeA->fingerprint() != storeB->fingerprint(),
              "same geometry + different weights = different generations (never shared bytes)");
        CHECK(fs::exists(genA / "manifest.bin") && fs::exists(genA / "payload.bin"),
              "A's generation files untouched while B built");
        // Prefix ORDER is part of the layout: same model B, reversed list = another generation.
        std::vector<std::string> rev(fb.expert_prefixes.rbegin(), fb.expert_prefixes.rend());
        auto storeC = Exl3Band8Store::build_or_open(fb.dir.string(), shared.string(), rev, opts);
        CHECK(gen_of(storeC) != genB && storeC->fingerprint() != storeB->fingerprint(),
              "different prefix order = different generation");
        // Reopen A after B and C exist: still exactly A's bytes.
        {
            auto ra = Exl3Band8Store::build_or_open(fa.dir.string(), shared.string(),
                                                    fa.expert_prefixes, opts);
            CHECK(gen_of(ra) == genA, "reopen A selects A's generation");
            std::string path;
            size_t off = 0, nb = 0;
            for (const char* plane : {".trellis", ".suh", ".svh"}) {
                CHECK(ra->locate(fa.expert_prefixes[2] + plane, path, off, nb) &&
                          pread_span(path, off, nb) == a_bytes[plane],
                          (std::string("A bytes unchanged after B/C builds: ") + plane).c_str());
            }
            strata::Exl3Linear la{};
            CHECK(ra->linear(fa.expert_prefixes[2], la), "reopen A linear covers the expert");
            CHECK(std::memcmp(la.trellis, a_bytes[".trellis"].data(),
                              a_bytes[".trellis"].size()) == 0,
                  "A linear pointers still serve A's weights");
        }
        // B serves B's weights (reference permutation of B's OWN source).
        {
            Exl3Model mb(fb.dir.string(), storeB);
            check_expert_bytes("B-after-A", *storeB, mb, fb.dir, fb.expert_prefixes[2], 8, 16, 32);
        }
        // NEGATIVE CONTROL (file surgery only - no production path is overwritten): reproduce
        // the OLD flat layout by pairing A's manifest with B's payload in one directory. The
        // pairing VALIDATES (same geometry/size) yet serves foreign bytes - exactly the race
        // the generation namespace removes.
        {
            const fs::path flat = root / "flat-old-design";
            fs::create_directories(flat);
            fs::copy_file(genA / "manifest.bin", flat / "manifest.bin");
            fs::copy_file(genB / "payload.bin", flat / "payload.bin");
            CHECK(fs::file_size(flat / "manifest.bin") == fs::file_size(genA / "manifest.bin") &&
                      fs::file_size(flat / "payload.bin") == fs::file_size(genA / "payload.bin"),
                  "negative control: old flat pairing is size-valid (manifest would accept it)");
            std::string path;
            size_t off = 0, nb = 0;
            CHECK(storeA->locate(fa.expert_prefixes[2] + ".trellis", path, off, nb) &&
                      pread_span((flat / "payload.bin").string(), off, nb) != a_bytes[".trellis"],
                  "negative control: A's manifest offsets into B's payload yield B's weights "
                  "(the old race); generation dirs make this pairing unreachable");
        }
        // Concurrent-reader seam: while D's payload is committed but its manifest is NOT, a
        // fresh reader of A gets A's complete generation - never D's payload or a partial.
        {
            Exl3Band8BuildOptions seam = opts;
            seam.after_payload_publish_for_tests = [&]() {
                fs::path pending;
                for (const auto& e : fs::directory_iterator(shared))
                    if (e.is_directory() && fs::exists(e.path() / "payload.bin") &&
                        !fs::exists(e.path() / "manifest.bin"))
                        pending = e.path();
                CHECK(!pending.empty(), "publish window: payload committed, manifest absent");
                auto ra = Exl3Band8Store::build_or_open(fa.dir.string(), shared.string(),
                                                        fa.expert_prefixes, opts);
                CHECK(gen_of(ra) == genA,
                      "fresh reader during foreign publish opens its own complete generation");
                std::string path;
                size_t off = 0, nb = 0;
                CHECK(ra->locate(fa.expert_prefixes[2] + ".trellis", path, off, nb) &&
                          pread_span(path, off, nb) == a_bytes[".trellis"],
                      "fresh reader bytes == A's original bytes, not the pending payload");
            };
            auto storeD = Exl3Band8Store::build_or_open(fd.dir.string(), shared.string(),
                                                        fd.expert_prefixes, seam);
            CHECK(storeD->record_count() == fd.expert_prefixes.size() * 3,
                  "seamed build still completes and publishes");
            Exl3Model md(fd.dir.string(), storeD);
            check_expert_bytes("D", *storeD, md, fd.dir, fd.expert_prefixes[2], 8, 16, 32);
        }
        // Source-stability recheck: mutate the source DURING the publish window; the build
        // must refuse to stamp the manifest (mixed-source payload never becomes a generation).
        {
            const fs::path shared2 = root / "shared-root-race";
            Exl3Band8BuildOptions race = opts;
            race.after_payload_publish_for_tests = [&]() {
                touch_mtime(fd.dir / "model.safetensors.index.json");
            };
            CHECK_THROW(auto s = Exl3Band8Store::build_or_open(fd.dir.string(), shared2.string(),
                                                               fd.expert_prefixes, race);
                        (void)s, "source changed while packing");
            fs::path pending;
            for (const auto& e : fs::directory_iterator(shared2))
                if (e.is_directory() && !fs::exists(e.path() / "manifest.bin")) pending = e.path();
            CHECK(!pending.empty() && !fs::exists(pending / "manifest.bin"),
                  "mixed-source generation is NOT published (no manifest)");
            CHECK(!fs::exists(pending / "manifest.bin.partial"),
                  "failed publish cleans its own partial manifest");
        }
    }

    // -- bounds: the whole test stays small, and peak RSS is reported -----------------------
    const size_t total = tree_bytes(root);
    std::printf("fixture+cache total: %zu bytes (%.2f MiB), peak RSS: %zu kB (VmHWM)\n",
                total, double(total) / (1024 * 1024), peak_rss_kb());
    CHECK(total <= 128u << 20, "fixture stays within the 128 MiB bound");
    CHECK(peak_rss_kb() > 0 && peak_rss_kb() < 512u << 10, "peak RSS bounded (<512 MiB)");

    if (g_failures == 0) std::printf("ALL BAND8 STORE CHECKS PASSED (%d checks)\n", g_checks);
    else std::printf("%d/%d CHECK(S) FAILED\n", g_failures, g_checks);
    fs::remove_all(root);
    return g_failures == 0 ? 0 : 1;
}
