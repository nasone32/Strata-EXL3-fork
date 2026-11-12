// src/artifact/exl3_band8_store.cpp - persistent band-8 packed-expert cache (see header for the
// contract). POSIX only, mirroring safetensors.hpp's mapping approach: the Windows branch throws
// at open time instead of silently mapping nothing (no Windows machine was available to test).

#include "strata/artifact/exl3_band8_store.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

#include "strata/artifact/exl3_model.hpp"
#include "strata/core/exl3_band8_permute.hpp"

namespace strata {
namespace {

// ---- platform shims ---------------------------------------------------------
#if defined(_WIN32)
[[noreturn]] void posix_only(const char* what) {
    throw std::runtime_error(std::string("exl3-band8: ") + what +
                             " requires POSIX mmap (Windows path not implemented yet)");
}
#endif

// FNV-1a 64, the same checksum family route_digest.hpp uses for route fingerprints. Not a
// security primitive: it detects changed sources, truncated files and (when recorded) flipped
// payload bytes at streaming speed, which is what this cache needs.
constexpr uint64_t kFnvOffset = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;
uint64_t fnv1a(uint64_t h, const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= kFnvPrime; }
    return h;
}
uint64_t fnv_u64(uint64_t h, uint64_t v) {
    uint8_t b[8];
    std::memcpy(b, &v, 8);   // little-endian hosts, same convention as the safetensors header
    return fnv1a(h, b, 8);
}

// Little-endian fixed-width encoding for the manifest (LE hosts, like safetensors.hpp's header).
void put_u16(std::vector<uint8_t>& b, uint16_t v) { b.push_back(uint8_t(v)); b.push_back(uint8_t(v >> 8)); }
void put_u32(std::vector<uint8_t>& b, uint32_t v) {
    b.push_back(uint8_t(v)); b.push_back(uint8_t(v >> 8)); b.push_back(uint8_t(v >> 16)); b.push_back(uint8_t(v >> 24));
}
void put_u64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(uint8_t(v >> (8 * i)));
}
uint16_t get_u16(const uint8_t* p) { return uint16_t(p[0]) | uint16_t(p[1]) << 8; }
uint32_t get_u32(const uint8_t* p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
uint64_t get_u64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

// Checked size arithmetic: every record bound is proven before any pointer is formed.
size_t checked_add(size_t a, size_t b, const char* what) {
    if (b > std::numeric_limits<size_t>::max() - a)
        throw std::runtime_error(std::string("exl3-band8: offset+bytes overflow in ") + what);
    return a + b;
}
size_t checked_mul(size_t a, size_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
        throw std::runtime_error(std::string("exl3-band8: size overflow in ") + what);
    return a * b;
}

// Two failure classes. CacheInvalid means "our artifact is unusable; rebuilding is safe".
// CacheRefused means "do not touch it, and never fall back to native": a newer manifest format,
// or a caller-requested expert that the cache does not cover.
struct CacheInvalid : std::runtime_error { using std::runtime_error::runtime_error; };
struct CacheRefused : std::runtime_error { using std::runtime_error::runtime_error; };

constexpr uint8_t kMagic[8] = {'S', 'T', 'R', 'B', '8', 0, 0, 0};
constexpr uint32_t kManifestVersion = 1;
constexpr size_t kHeaderBytes = 80;
constexpr size_t kRecordBytes = 40;
constexpr size_t kRecordAlign = 256;      // every record starts 256-byte aligned
constexpr size_t kMaxNameLen = 512;
constexpr size_t kMaxManifestBytes = 256u << 20;
constexpr uint8_t kKindTrellis = 1, kKindSuh = 2, kKindSvh = 3;
constexpr uint64_t kPayloadChecksumFlag = 1;

// The four feasibility-verified rates (notes/EXL3-BAND8-FEASIBILITY-2026-11-06.md). Anything
// else (K1, K8, unverified halves like 3.5) is refused, not silently packed. Shared with the
// runtime in-memory repack through exl3_band8_permute.hpp (one formula, no drift copies).
bool verified_rate(int bits, int frac) {
    return strata::exl3_band8_verified_rate(bits, frac);
}

// ---- POSIX file plumbing ----------------------------------------------------
#if !defined(_WIN32)
struct FileHandle {
    int fd = -1;
    explicit FileHandle(int f) : fd(f) {}
    ~FileHandle() { if (fd >= 0) ::close(fd); }
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
};

// Immutable read-only mapping of one whole file: the same open/fstat/mmap/MAP_PRIVATE pattern
// SafetensorsFile uses, factored out for the cache's plain byte payload.
class MappedFile {
public:
    explicit MappedFile(const std::string& path) {
        int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error("exl3-band8: cannot open " + path + ": " + std::strerror(errno));
        struct stat sb {};
        if (::fstat(fd, &sb) != 0) {
            ::close(fd);
            throw std::runtime_error("exl3-band8: fstat failed for " + path);
        }
        size_ = (size_t)sb.st_size;
        base_ = (const uint8_t*)::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (base_ == MAP_FAILED) {
            base_ = nullptr;
            throw std::runtime_error("exl3-band8: mmap failed for " + path);
        }
    }
    ~MappedFile() { if (base_) ::munmap((void*)base_, size_); }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    const uint8_t* data() const { return base_; }
    size_t size() const { return size_; }
private:
    const uint8_t* base_ = nullptr;
    size_t size_ = 0;
};

struct ShardStat { uint64_t size = 0, mtime_ns = 0; };
ShardStat stat_file(const std::string& path) {
    struct stat sb {};
    if (::stat(path.c_str(), &sb) != 0)
        throw std::runtime_error("exl3-band8: cannot stat " + path + ": " + std::strerror(errno));
    return {(uint64_t)sb.st_size,
            (uint64_t)sb.st_mtim.tv_sec * 1000000000ull + (uint64_t)sb.st_mtim.tv_nsec};
}
#endif  // !defined(_WIN32)

// ---- manifest ---------------------------------------------------------------
struct Record {
    uint64_t offset = 0, bytes = 0;
    uint32_t name_off = 0;
    uint16_t name_len = 0;
    uint8_t kind = 0, bits = 0, frac = 0;
    uint32_t ki = 0, nj = 0;
    std::string name;                      // resolved from the pool
};

struct Manifest {
    uint32_t manifest_version = 0, flags = 0;
    uint64_t fingerprint = 0, record_count = 0, payload_bytes = 0, payload_fnv = 0;
    std::vector<Record> records;           // sorted by name on disk; strictly increasing here
};

std::vector<uint8_t> encode_manifest(const Manifest& m, const std::vector<std::string>& names) {
    std::vector<uint8_t> b;
    b.reserve(kHeaderBytes + m.records.size() * kRecordBytes);
    size_t pool = 0;
    for (const std::string& n : names) pool += n.size() + 1;
    b.insert(b.end(), kMagic, kMagic + 7);
    b.push_back(kExl3Band8FormatVersion);           // magic[7] = band8 format version
    put_u32(b, m.manifest_version);
    put_u32(b, m.flags);
    put_u64(b, m.fingerprint);
    put_u64(b, m.records.size());
    put_u64(b, m.payload_bytes);
    put_u64(b, m.payload_fnv);
    put_u64(b, pool);
    put_u64(b, 0); put_u64(b, 0); put_u64(b, 0);   // reserved
    size_t off = 0;
    for (size_t i = 0; i < m.records.size(); ++i) {
        const Record& r = m.records[i];
        const std::string& n = names[i];
        put_u64(b, r.offset);
        put_u64(b, r.bytes);
        put_u32(b, (uint32_t)off);
        put_u16(b, (uint16_t)n.size());
        b.push_back(r.kind);
        b.push_back(r.bits);
        b.push_back(r.frac);
        b.push_back(0);                              // pad byte
        put_u16(b, 0);                               // pad u16
        put_u32(b, r.ki);
        put_u32(b, r.nj);
        put_u32(b, 0);                               // reserved
        off += n.size() + 1;
    }
    for (const std::string& n : names) { b.insert(b.end(), n.begin(), n.end()); b.push_back(0); }
    return b;
}

// Parse + fully validate the manifest. Every bound below is proven BEFORE the payload is mapped,
// so a corrupt record can never push pointer arithmetic out of the mapping.
Manifest decode_manifest(const std::string& path, size_t payload_file_size) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw CacheInvalid("cannot open manifest " + path);
    std::vector<uint8_t> b((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (b.size() < kHeaderBytes) throw CacheInvalid("manifest shorter than its header");
    if (b.size() > kMaxManifestBytes) throw CacheInvalid("manifest implausibly large");
    if (std::memcmp(b.data(), kMagic, 7) != 0 || b[7] != kExl3Band8FormatVersion)
        throw CacheRefused("manifest magic/format not recognized (band8 format version " +
                           std::to_string((unsigned)b[7]) + ")");
    Manifest m;
    m.manifest_version = get_u32(b.data() + 8);
    if (m.manifest_version != kManifestVersion)
        throw CacheRefused("unsupported manifest version " + std::to_string(m.manifest_version) +
                           " (refusing; this build will not overwrite a newer-format cache)");
    m.flags = get_u32(b.data() + 12);
    m.fingerprint = get_u64(b.data() + 16);
    m.record_count = get_u64(b.data() + 24);
    m.payload_bytes = get_u64(b.data() + 32);
    m.payload_fnv = get_u64(b.data() + 40);
    const uint64_t pool = get_u64(b.data() + 48);
    if (m.record_count > b.size() / kRecordBytes || pool > b.size())
        throw CacheInvalid("manifest count/pool out of range");
    const size_t need = checked_add(kHeaderBytes,
                                    checked_mul((size_t)m.record_count, kRecordBytes, "manifest"),
                                    "manifest");
    const size_t body = checked_add(need, (size_t)pool, "manifest");
    if (body != b.size()) throw CacheInvalid("manifest size does not match its counts");
    const uint8_t* pool_beg = b.data() + need;
    const uint8_t* rp = b.data() + kHeaderBytes;
    m.records.resize((size_t)m.record_count);
    for (size_t i = 0; i < m.record_count; ++i, rp += kRecordBytes) {
        Record& r = m.records[i];
        r.offset = get_u64(rp);
        r.bytes = get_u64(rp + 8);
        r.name_off = get_u32(rp + 16);
        r.name_len = get_u16(rp + 20);
        r.kind = rp[22];
        r.bits = rp[23];
        r.frac = rp[24];
        r.ki = get_u32(rp + 28);
        r.nj = get_u32(rp + 32);
        if (r.name_len == 0 || r.name_len > kMaxNameLen)
            throw CacheInvalid("record name length out of range");
        const size_t pool_end = checked_add((size_t)r.name_off, (size_t)r.name_len + 1, "manifest name");
        if (pool_end > (size_t)pool) throw CacheInvalid("record name outside the manifest name pool");
        const char* nm = reinterpret_cast<const char*>(pool_beg + r.name_off);
        if (nm[r.name_len] != 0) throw CacheInvalid("record name not NUL-terminated");
        r.name.assign(nm, r.name_len);
        // Tile-level validation: same rules the packer enforced, so a stale or hand-edited
        // manifest cannot describe a plane the kernels would misread.
        const std::string what = "record " + r.name;
        if (r.ki == 0 || r.nj == 0 || r.ki > (1u << 26) || r.nj > (1u << 26) ||
            r.ki % 8 != 0 || r.nj % 8 != 0)
            throw CacheInvalid("record tile dims not H128-aligned: " + r.name);
        if (r.kind == kKindTrellis) {
            if (!verified_rate(r.bits, r.frac))
                throw CacheRefused("trellis record " + r.name + " has unverified rate K=" +
                                   std::to_string(r.bits + 0.5 * r.frac));
            const size_t words = checked_mul(checked_mul(checked_mul((size_t)r.ki, r.nj, what.c_str()),
                                                          (size_t)(16 * r.bits + 8 * r.frac), what.c_str()),
                                             2, what.c_str());
            if (words != r.bytes) throw CacheInvalid("trellis record byte count mismatch: " + r.name);
        } else if (r.kind == kKindSuh) {
            if (checked_mul(r.ki, 32, what.c_str()) != r.bytes)
                throw CacheInvalid("suh record byte count mismatch: " + r.name);
        } else if (r.kind == kKindSvh) {
            if (checked_mul(r.nj, 32, what.c_str()) != r.bytes)
                throw CacheInvalid("svh record byte count mismatch: " + r.name);
        } else {
            throw CacheInvalid("unknown record kind");
        }
        // Offset 0 is structurally illegal: the first 256 payload bytes are reserved so an
        // unpatched zero offset is always caught here, before any pointer exists.
        if (r.offset < kRecordAlign || r.offset % kRecordAlign != 0)
            throw CacheInvalid("record offset is reserved/unaligned: " + r.name);
        // Overflow-proof bound: payload_bytes equals a real file size, so compare against it
        // instead of computing offset+bytes (a crafted wrap-around record lands here too).
        if (r.bytes > m.payload_bytes || r.offset > m.payload_bytes - r.bytes)
            throw CacheInvalid("record range outside payload: " + r.name);
    }
    if (m.payload_bytes != payload_file_size)
        throw CacheInvalid("payload file size " + std::to_string(payload_file_size) +
                           " != manifest " + std::to_string(m.payload_bytes) +
                           " (truncated or damaged payload)");
    for (size_t i = 1; i < m.records.size(); ++i)
        if (!(m.records[i - 1].name < m.records[i].name))
            throw CacheInvalid("duplicate or unsorted record names: " + m.records[i].name);
    return m;
}

const Record* find_record(const Manifest& m, const std::string& name) {
    const auto it = std::lower_bound(m.records.begin(), m.records.end(), name,
        [](const Record& r, const std::string& n) { return r.name < n; });
    return (it != m.records.end() && it->name == name) ? &*it : nullptr;
}

// Every requested prefix must be covered by a COMPLETE triple with consistent metadata. This is
// the constructor-time coverage check: a missing requested expert is an error (CacheRefused),
// never a late native fallback at linear() time.
void check_coverage(const Manifest& m, const std::vector<std::string>& prefixes) {
    for (const std::string& p : prefixes) {
        const Record* tr = find_record(m, p + ".trellis");
        const Record* suh = find_record(m, p + ".suh");
        const Record* svh = find_record(m, p + ".svh");
        if (tr == nullptr || suh == nullptr || svh == nullptr)
            throw CacheRefused("requested expert projection is not covered by the cache: " + p);
        if (suh->ki != tr->ki || svh->nj != tr->nj ||
            suh->bits != tr->bits || suh->frac != tr->frac ||
            svh->bits != tr->bits || svh->frac != tr->frac)
            throw CacheInvalid("covered triple has inconsistent metadata: " + p);
    }
}

// ---- source model access (metadata + pread offsets only) ---------------------
std::string canonical_path(const std::string& p) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path c = fs::weakly_canonical(fs::path(p), ec);
    if (ec || c.empty()) c = fs::absolute(fs::path(p), ec);
    std::string s = c.string();
    while (s.size() > 1 && s.back() == '/') s.pop_back();
    return s;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("exl3-band8: cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// The fingerprint changes on ANY input the packed bytes depend on: the schema/formula version,
// the canonical model dir, the index.json bytes AND its size/mtime, every referenced shard's
// canonical path/size/mtime, and the exact requested prefix list (order included). Changed
// inputs are therefore EXPECTED to select a different generation. This is a fast, ordered
// change detector, not a cryptographic identity: an adversarial edit that preserves every
// hashed datum (e.g. payload bytes rewritten with size+mtime kept) can still alias.
uint64_t compute_fingerprint(const std::string& model_dir, const std::vector<std::string>& prefixes,
                             const std::map<std::string, std::string>& weight_map) {
    uint64_t h = fnv1a(kFnvOffset, "strata/exl3-band8-store/1", 26);
    h = fnv_u64(h, kExl3Band8FormatVersion);
    const std::string dir = canonical_path(model_dir);
    h = fnv_u64(h, dir.size());
    h = fnv1a(h, dir.data(), dir.size());

    const std::string index_path = model_dir + "/model.safetensors.index.json";
    const std::string idx = read_file(index_path);   // missing file throws: no source, no cache
    h = fnv_u64(h, idx.size());
    h = fnv1a(h, idx.data(), idx.size());
#if defined(_WIN32)
    posix_only("fingerprint stat");
#else
    const ShardStat is = stat_file(index_path);
    h = fnv_u64(h, is.size);
    h = fnv_u64(h, is.mtime_ns);

    std::set<std::string> shards;
    for (const std::string& p : prefixes)
        for (const char* suffix : {".trellis", ".suh", ".svh"}) {
            auto it = weight_map.find(p + suffix);
            if (it != weight_map.end()) shards.insert(it->second);
        }
    h = fnv_u64(h, shards.size());
    for (const std::string& s : shards) {   // std::set iterates sorted: order-stable
        const ShardStat st = stat_file((model_dir + "/" + s));
        h = fnv_u64(h, s.size());
        h = fnv1a(h, s.data(), s.size());
        const std::string full = canonical_path(model_dir + "/" + s);
        h = fnv_u64(h, full.size());
        h = fnv1a(h, full.data(), full.size());
        h = fnv_u64(h, st.size);
        h = fnv_u64(h, st.mtime_ns);
    }
#endif
    h = fnv_u64(h, prefixes.size());
    for (const std::string& p : prefixes) {
        h = fnv_u64(h, p.size());
        h = fnv1a(h, p.data(), p.size());
    }
    return h;
}

// ---- builder ----------------------------------------------------------------
struct PlaneSpec {
    uint32_t ki = 0, nj = 0;
    uint8_t bits = 0, frac = 0;
    size_t trellis_bytes = 0, suh_bytes = 0, svh_bytes = 0;
};

// Validate one requested triple against the SOURCE metadata only (no payload read): dims, dtype,
// verified rate, mul1 marker, no bias, bounded plane size.
PlaneSpec validate_source(const Exl3Model& model,
                          const std::map<std::string, std::string>& wmap,
                          const std::string& p, const Exl3Band8BuildOptions& opts) {
    for (const char* suffix : {".trellis", ".suh", ".svh"})
        if (wmap.count(p + suffix) == 0)
            throw std::runtime_error("exl3-band8: requested projection " + p + " has no " + suffix +
                                     " in the source index (all requested triples must be complete)");
    const StTensor* tr = model.meta(p + ".trellis");
    const StTensor* suh = model.meta(p + ".suh");
    const StTensor* svh = model.meta(p + ".svh");
    if (tr == nullptr || suh == nullptr || svh == nullptr)
        throw std::runtime_error("exl3-band8: requested projection " + p +
                                 " is indexed but missing from its shard (corrupt source)");
    if (model.meta(p + ".bias") != nullptr)
        throw std::runtime_error("exl3-band8: " + p + ": biased EXL3 linears are not supported");
    if (model.meta(p + ".mul1") == nullptr)
        throw std::runtime_error("exl3-band8: " + p + " lacks the mul1 marker (store is mul1-only)");
    if (tr->dtype != "I16" || tr->shape.size() != 3 ||
        tr->shape[0] <= 0 || tr->shape[1] <= 0 ||
        tr->shape[0] % 8 != 0 || tr->shape[1] % 8 != 0 ||
        tr->shape[2] < 16 || tr->shape[2] > 128 || tr->shape[2] % 8 != 0 ||
        tr->shape[0] > (1 << 26) || tr->shape[1] > (1 << 26))
        throw std::runtime_error("exl3-band8: " + p + ": need I16 trellis [in/16,out/16,16*K], H128 dims");
    PlaneSpec s;
    s.ki = (uint32_t)tr->shape[0];
    s.nj = (uint32_t)tr->shape[1];
    s.bits = (uint8_t)(tr->shape[2] / 16);
    s.frac = (uint8_t)((tr->shape[2] % 16) / 8);
    if (!verified_rate(s.bits, s.frac))
        throw std::runtime_error("exl3-band8: " + p + ": rate K=" +
                                 std::to_string(s.bits + 0.5 * s.frac) +
                                 " is outside the verified set {2, 2.5, 3, 4}");
    if (suh->dtype != "F16" || svh->dtype != "F16" ||
        suh->shape != std::vector<int64_t>{tr->shape[0] * 16} ||
        svh->shape != std::vector<int64_t>{tr->shape[1] * 16})
        throw std::runtime_error("exl3-band8: " + p + ": need F16 suh[in] and svh[out]");
    const size_t P = (size_t)(16 * s.bits + 8 * s.frac);   // uint16 words per 256-weight tile
    const std::string what = "plane " + p;
    s.trellis_bytes = checked_mul(checked_mul(checked_mul((size_t)s.ki, s.nj, what.c_str()),
                                              P, what.c_str()), 2, what.c_str());
    s.suh_bytes = checked_mul((size_t)s.ki, 32, what.c_str());
    s.svh_bytes = checked_mul((size_t)s.nj, 32, what.c_str());
    if (s.trellis_bytes > opts.max_projection_bytes)
        throw std::runtime_error("exl3-band8: " + p + ": trellis plane " +
                                 std::to_string(s.trellis_bytes >> 20) + " MiB exceeds the bounded "
                                 "build budget " + std::to_string(opts.max_projection_bytes >> 20) + " MiB");
    return s;
}

// Native -> band8 through the ONE shared permutation (exl3_band8_permute.hpp): the
// offline pack and the runtime in-memory repack cannot drift apart.
void permute_band8(const uint16_t* src, uint16_t* dst, uint32_t ki, uint32_t nj, size_t P) {
    strata::exl3_band8_permute(src, dst, ki, nj, P);
}

#if !defined(_WIN32)
// Exclusive per-generation builder lock. A VALID owner pid that is alive (kill(pid,0) succeeds
// or EPERM) blocks; a valid pid that is provably dead (ESRCH) is OUR stale artifact and is
// removed. An EMPTY or unparseable lock is treated as BUSY and refused: the creator may be
// mid-write of its own pid line, so blindly reclaiming it would race a live builder. Only an
// explicit parseable dead pid is ever reclaimed.
class BuildLock {
public:
    explicit BuildLock(const std::string& path) : path_(path) {
        for (int attempt = 0; attempt < 2; ++attempt) {
            fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
            if (fd_ >= 0) {
                char buf[64];
                const int n = std::snprintf(buf, sizeof buf, "pid=%ld\n", (long)::getpid());
                if (::write(fd_, buf, (size_t)n) != n) {
                    ::close(fd_); fd_ = -1;
                    ::unlink(path.c_str());
                    throw std::runtime_error("exl3-band8: cannot write " + path);
                }
                held_ = true;
                return;
            }
            if (errno != EEXIST)
                throw std::runtime_error("exl3-band8: cannot create lock " + path + ": " + std::strerror(errno));
            long other = 0;
            if (!read_owner_pid(&other))
                throw std::runtime_error("exl3-band8: lock " + path +
                                         " exists but is empty or unparseable (a builder may be "
                                         "creating it); refusing - remove it manually if this "
                                         "build crashed while locking");
            bool live = false;
            if (::kill((pid_t)other, 0) == 0) live = true;
            else if (errno == EPERM) live = true;                  // exists, another user's
            if (live)
                throw std::runtime_error("exl3-band8: another builder holds " + path +
                                         " (pid " + std::to_string(other) + "); not touching it");
            std::fprintf(stderr, "exl3-band8: removing stale lock from dead pid %ld\n", other);
            if (::unlink(path.c_str()) != 0)
                throw std::runtime_error("exl3-band8: cannot remove stale lock " + path);
        }
        throw std::runtime_error("exl3-band8: lock retry loop exhausted for " + path);
    }
    ~BuildLock() {
        if (fd_ >= 0) ::close(fd_);
        if (held_) ::unlink(path_.c_str());   // our own artifact only
    }
    BuildLock(const BuildLock&) = delete;
    BuildLock& operator=(const BuildLock&) = delete;
private:
    // false = empty/truncated/unparseable content: caller must treat the lock as busy.
    bool read_owner_pid(long* out) const {
        std::ifstream in(path_);
        char buf[64] = {0};
        in.read(buf, sizeof buf - 1);
        long v = 0;
        if (std::sscanf(buf, "pid=%ld", &v) == 1 && v > 0) {
            *out = v;
            return true;
        }
        return false;
    }
    std::string path_;
    int fd_ = -1;
    bool held_ = false;
};

// Bounded sequential writer with a streaming checksum. One projection's packed twin buffer is
// written through it in io_block-sized write() calls; ENOSPC surfaces as a clean error.
class PayloadWriter {
public:
    PayloadWriter(const std::string& path, const Exl3Band8BuildOptions& opts)
        : opts_(opts), fd_(::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644)),
          path_(path) {
        if (fd_.fd < 0)
            throw std::runtime_error("exl3-band8: cannot create " + path + ": " + std::strerror(errno));
    }
    void write_all(const uint8_t* p, size_t n) {
        while (n > 0) {
            const size_t chunk = std::min(n, opts_.io_block_bytes ? opts_.io_block_bytes : (4u << 20));
            const ssize_t w = ::write(fd_.fd, p, chunk);
            if (w < 0) {
                if (errno == EINTR) continue;
                if (errno == ENOSPC)
                    throw std::runtime_error("exl3-band8: cache disk full while writing " + path_ +
                                             " (partial cache removed; free more bytes and retry)");
                throw std::runtime_error("exl3-band8: write failed on " + path_ + ": " + std::strerror(errno));
            }
            fnv_ = fnv1a(fnv_, p, (size_t)w);
            written_ += (size_t)w;
            p += w;
            n -= (size_t)w;
        }
    }
    void pad_to_align() {
        const size_t rem = written_ % kRecordAlign;
        if (rem != 0) {
            const uint8_t z[kRecordAlign] = {0};
            write_all(z, kRecordAlign - rem);
        }
    }
    size_t written() const { return written_; }
    uint64_t fnv() const { return fnv_; }
    void sync() {
        if (::fsync(fd_.fd) != 0)
            throw std::runtime_error("exl3-band8: fsync failed on " + path_ + ": " + std::strerror(errno));
    }
private:
    Exl3Band8BuildOptions opts_;
    FileHandle fd_;
    std::string path_;
    size_t written_ = 0;
    uint64_t fnv_ = kFnvOffset;
};

// pread exactly `n` bytes at `off` (EINTR-safe, full-count): a source shard's expert payload
// enters this build ONLY through this bounded buffer. (SafetensorsFile still maps each shard
// file virtually to validate headers - those mappings are lazily populated, and nothing here
// forces the OS page cache one way or the other; the claim is only that the BUILD's own
// buffers are bounded, not that source pages are absent.)
void pread_exact(int fd, uint8_t* dst, size_t n, uint64_t off, const std::string& what) {
    while (n > 0) {
        const ssize_t r = ::pread(fd, dst, n, (off_t)off);
        if (r < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("exl3-band8: pread failed on " + what + ": " + std::strerror(errno));
        }
        if (r == 0) throw std::runtime_error("exl3-band8: short pread on " + what + " (source changed?)");
        dst += r; n -= (size_t)r; off += (uint64_t)r;
    }
}

uint64_t disk_free_bytes(const std::filesystem::path& dir, const Exl3Band8BuildOptions& opts) {
    if (opts.reported_free_bytes_for_tests.has_value()) return *opts.reported_free_bytes_for_tests;
    struct statvfs sv {};
    if (::statvfs(dir.c_str(), &sv) != 0)
        throw std::runtime_error("exl3-band8: statvfs failed for " + dir.string());
    return (uint64_t)sv.f_bavail * (uint64_t)sv.f_frsize;
}

// Payload file size, as a rebuildable cache fault (a missing/damaged payload is OUR artifact).
size_t payload_size_or_invalid(const std::filesystem::path& payload) {
    std::error_code e;
    if (!std::filesystem::exists(payload, e)) throw CacheInvalid("payload.bin missing while manifest exists");
    const uintmax_t sz = std::filesystem::file_size(payload, e);
    if (e) throw CacheInvalid("cannot stat payload.bin");
    if (sz > std::numeric_limits<size_t>::max()) throw CacheInvalid("payload.bin size overflow");
    return (size_t)sz;
}
#endif  // !defined(_WIN32)

}  // namespace

// ---- store impl -------------------------------------------------------------
struct Exl3Band8Store::Impl {
    std::string payload_path;
#if !defined(_WIN32)
    std::unique_ptr<MappedFile> payload;
#endif
    uint64_t fingerprint = 0;
    std::map<std::string, Record> by_name;

    const Record* find(const std::string& name, uint8_t kind) const {
        auto it = by_name.find(name);
        if (it == by_name.end() || it->second.kind != kind) return nullptr;
        return &it->second;
    }
};

Exl3Band8Store::~Exl3Band8Store() = default;

size_t Exl3Band8Store::record_count() const { return impl_->by_name.size(); }
uint64_t Exl3Band8Store::fingerprint() const { return impl_->fingerprint; }
const std::string& Exl3Band8Store::payload_path() const { return impl_->payload_path; }

bool Exl3Band8Store::linear(const std::string& base, Exl3Linear& out) const {
    const Record* tr = impl_->find(base + ".trellis", kKindTrellis);
    const Record* suh = impl_->find(base + ".suh", kKindSuh);
    const Record* svh = impl_->find(base + ".svh", kKindSvh);
    if (tr == nullptr || suh == nullptr || svh == nullptr) return false;
#if defined(_WIN32)
    posix_only("linear");
#else
    const uint8_t* base_ptr = impl_->payload->data();
    out.trellis = reinterpret_cast<const uint16_t*>(base_ptr + (size_t)tr->offset);
    out.suh = reinterpret_cast<const uint16_t*>(base_ptr + (size_t)suh->offset);
    out.svh = reinterpret_cast<const uint16_t*>(base_ptr + (size_t)svh->offset);
#endif
    out.ki = (int)tr->ki;
    out.nj = (int)tr->nj;
    out.bits = (int)tr->bits;
    out.frac = (int)tr->frac;
    out.cb = 2;                                    // mul1-only store
    out.layout = Exl3TrellisLayout::Band8;
    return true;
}

bool Exl3Band8Store::locate(const std::string& tensor, std::string& path,
                            size_t& offset, size_t& bytes) const {
    auto it = impl_->by_name.find(tensor);
    if (it == impl_->by_name.end()) return false;
    path = impl_->payload_path;
    offset = (size_t)it->second.offset;
    bytes = (size_t)it->second.bytes;
    return true;
}

// ---- build_or_open ----------------------------------------------------------
std::shared_ptr<const Exl3Band8Store> Exl3Band8Store::build_or_open(
    const std::string& model_dir, const std::string& cache_dir,
    const std::vector<std::string>& expert_prefixes, const Exl3Band8BuildOptions& opts) {
#if defined(_WIN32)
    (void)model_dir; (void)cache_dir; (void)expert_prefixes; (void)opts;
    posix_only("build_or_open");
#else
    namespace fs = std::filesystem;
    // Duplicate or absurd prefixes are refused up front: one cache, one meaning per name.
    {
        const std::set<std::string> uniq(expert_prefixes.begin(), expert_prefixes.end());
        if (uniq.size() != expert_prefixes.size())
            throw std::runtime_error("exl3-band8: duplicate prefix in the requested expert list");
        for (const std::string& p : expert_prefixes)
            if (p.empty() || p.size() > kMaxNameLen)
                throw std::runtime_error("exl3-band8: prefix length out of range");
    }

    // The index is parsed once here (fingerprint + completeness); Exl3Model supplies shard-header
    // metadata (meta) and pread offsets (locate) during the build only.
    const std::string index_json = read_file(model_dir + "/model.safetensors.index.json");
    const std::map<std::string, std::string> wmap = exl3_detail::parse_weight_map(index_json);
    const Exl3Model model(model_dir);
    const uint64_t want = compute_fingerprint(model_dir, expert_prefixes, wmap);

    // IMMUTABLE GENERATION NAMESPACE: every artifact of one fingerprint lives under
    // cache_dir/<fingerprint-hex16>/ and no other generation ever replaces these bytes. This is
    // the fix for the flat payload.bin+manifest.bin pair: a reader between two renames could
    // previously pair the old manifest with a NEW same-sized payload and serve wrong weights.
    // Different model / prefix order => different fingerprint => different directory. The
    // generation directory is created empty by this builder (no clobber of foreign content);
    // old generations are never deleted by this code.
    char gen_name[17];
    std::snprintf(gen_name, sizeof gen_name, "%016" PRIx64, want);
    const fs::path gen = fs::path(cache_dir) / gen_name;
    // Read-only open (opts.allow_build=false): refuse BEFORE the first filesystem mutation -
    // no cache_dir/generation creation, no partial cleanup, no build lock, no payload byte.
    // Callers that must never write as a side effect (R1's microbench, inspection tools) get a
    // hard refusal naming the missing generation instead of an implicit ~80 GiB pack job.
    auto readonly_refuse = [&](const std::string& why) {
        throw std::runtime_error("exl3-band8: read-only open refused (" + why + ") - generation " +
                                 gen.string() + " is not usable; build it explicitly, this "
                                 "caller never writes");
    };
    if (!opts.allow_build) {
        std::error_code e1, e2;
        if (!fs::is_directory(fs::path(cache_dir), e1) || e1)
            readonly_refuse("cache dir " + cache_dir + " does not exist");
        if (!fs::exists(gen / "manifest.bin", e1) || e1 || !fs::exists(gen / "payload.bin", e2) || e2)
            readonly_refuse("no complete generation for this model/prefix list");
    }
    // Directory creation is a BUILD-side effect: a read-only open never creates anything,
    // so a directory removed between the precheck and here surfaces as the manifest/payload
    // open failing below - the readonly contract cannot recreate the path underneath a
    // racing deleter (no create-then-use TOCTOU).
    std::error_code ec;
    if (opts.allow_build) {
        fs::create_directories(fs::path(cache_dir), ec);
        if (!fs::is_directory(fs::path(cache_dir)))
            throw std::runtime_error("exl3-band8: cannot create cache dir " + cache_dir);
        fs::create_directories(gen, ec);
        if (!fs::is_directory(gen))
            throw std::runtime_error("exl3-band8: cannot create generation dir " + gen.string());
    }
    const fs::path payload = gen / "payload.bin";
    const fs::path manifest = gen / "manifest.bin";
    const fs::path lock_path = gen / "build.lock";
    const fs::path payload_partial = gen / "payload.bin.partial";
    const fs::path manifest_partial = gen / "manifest.bin.partial";
    // Only names this builder itself creates are ever removed (stale partials from an
    // interrupted run). payload.bin/manifest.bin are replaced by rename, never unlinked.
    auto clean_partials = [&]() {
        std::error_code e;
        fs::remove(payload_partial, e);
        fs::remove(manifest_partial, e);
    };

    // 1) Reuse path: valid manifest + matching fingerprint + full requested coverage, no payload
    //    byte scan (bounds + sizes + fingerprint only; checksum verify is explicit opt-in).
    if (fs::exists(manifest)) {
        bool rebuild = false;
        Manifest m{};
        try {
            const size_t psz = payload_size_or_invalid(payload);
            m = decode_manifest(manifest.string(), psz);
        } catch (const CacheRefused&) {
            throw;                                   // never overwrite/ignore what we do not understand
        } catch (const CacheInvalid& e) {
            if (!opts.allow_build) readonly_refuse(std::string("manifest invalid: ") + e.what());
            std::fprintf(stderr, "exl3-band8: generation %s invalid (%s); rebuilding\n",
                         gen_name, e.what());
            rebuild = true;
        }
        if (!rebuild && m.fingerprint != want) {
            // The directory name and the manifest disagree: treat as invalid, rebuild in place
            // (same fingerprint => same deterministic layout => old readers keep valid paths).
            if (!opts.allow_build) readonly_refuse("manifest fingerprint mismatch");
            std::fprintf(stderr, "exl3-band8: generation %s manifest fingerprint mismatch; "
                                 "rebuilding\n", gen_name);
            rebuild = true;
        }
        if (!rebuild) {
            try {
                check_coverage(m, expert_prefixes);
                if (opts.verify_payload) {
                    if ((m.flags & kPayloadChecksumFlag) == 0)
                        throw CacheRefused("verify_payload requested but the cache was built "
                                           "without a payload checksum");
                    MappedFile pf(payload.string());
                    std::vector<uint8_t> buf((size_t)std::min<uint64_t>(m.payload_bytes,
                                        opts.io_block_bytes ? opts.io_block_bytes : (4u << 20)));
                    FileHandle fd(::open(payload.c_str(), O_RDONLY | O_CLOEXEC));
                    if (fd.fd < 0) throw CacheInvalid("cannot reopen payload for verify");
                    uint64_t h = kFnvOffset;
                    for (uint64_t off = 0; off < m.payload_bytes;) {
                        const size_t chunk = (size_t)std::min<uint64_t>(buf.size(), m.payload_bytes - off);
                        pread_exact(fd.fd, buf.data(), chunk, off, payload.string());
                        h = fnv1a(h, buf.data(), chunk);
                        off += chunk;
                    }
                    if (h != m.payload_fnv)
                        throw CacheRefused("payload checksum mismatch (payload bytes changed on "
                                           "disk); refusing the damaged cache - remove " +
                                           payload.string() + " to force a rebuild");
                }
            } catch (const CacheRefused&) {
                throw;
            } catch (const CacheInvalid& e) {
                if (!opts.allow_build) readonly_refuse(std::string("cache unusable: ") + e.what());
                std::fprintf(stderr, "exl3-band8: cache unusable (%s); rebuilding\n", e.what());
                rebuild = true;
            }
        }
        if (!rebuild) {
            auto store = std::shared_ptr<Exl3Band8Store>(new Exl3Band8Store());
            store->impl_ = std::make_unique<Impl>();
            store->impl_->payload_path = payload.string();
            store->impl_->payload = std::make_unique<MappedFile>(payload.string());
            store->impl_->fingerprint = want;
            for (const Record& r : m.records) store->impl_->by_name.emplace(r.name, r);
            return store;
        }
    } else {
        if (!opts.allow_build) readonly_refuse("no manifest for this generation");
        clean_partials();   // interrupted earlier build: its partials are ours to clear
    }

    // 2) Build path. Validate EVERY requested triple against source metadata first; the
    //    free-space preflight and the streaming pass then run on known sizes.
    std::vector<PlaneSpec> specs;
    specs.reserve(expert_prefixes.size());
    uint64_t need = 0;
    for (const std::string& p : expert_prefixes) {
        const PlaneSpec s = validate_source(model, wmap, p, opts);
        specs.push_back(s);
        need += (uint64_t)s.trellis_bytes + s.suh_bytes + s.svh_bytes + 3 * kRecordAlign;
    }
    const uint64_t free_bytes = disk_free_bytes(fs::path(cache_dir), opts);
    if (free_bytes < need)   // report the gap in bytes; never an interactive approval flow
        throw std::runtime_error("exl3-band8: not enough free space for the packed cache: need " +
                                 std::to_string(need) + " bytes, have " + std::to_string(free_bytes) +
                                 " (missing " + std::to_string(need - free_bytes) + " bytes)");

    try {
        BuildLock lock(lock_path.string());
        clean_partials();
        const auto t0 = std::chrono::steady_clock::now();
        PayloadWriter out(payload_partial.string(), opts);
        std::vector<Record> records;
        records.reserve(expert_prefixes.size() * 3);
        {   // Reserve the first 256 payload bytes so a zero record offset is structurally
            // illegal and always caught in decode_manifest before any pointer exists.
            const uint8_t z[kRecordAlign] = {0};
            out.write_all(z, kRecordAlign);
        }
        std::map<std::string, int> shard_fds;  // one read fd per source shard, closed at the end

        auto shard_fd = [&](const std::string& path) -> int {
            const std::string shard = fs::path(path).filename().string();
            auto it = shard_fds.find(shard);
            if (it != shard_fds.end()) return it->second;
            const int fd = ::open((fs::path(model_dir) / shard).c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0)
                throw std::runtime_error("exl3-band8: cannot open source shard " + shard + ": " +
                                         std::strerror(errno));
            shard_fds.emplace(shard, fd);
            return fd;
        };
        auto append = [&](const std::string& name, uint8_t kind, const PlaneSpec& s,
                          const uint8_t* bytes, size_t n) {
            out.pad_to_align();
            Record r;
            r.kind = kind;
            r.bits = s.bits;
            r.frac = s.frac;
            r.ki = s.ki;
            r.nj = s.nj;
            r.offset = out.written();
            r.bytes = n;
            r.name = name;
            out.write_all(bytes, n);
            records.push_back(r);
        };

        std::vector<uint16_t> native, packed;  // reused per projection: ONE projection in RAM
        std::vector<uint8_t> scales;
        uint64_t done_triples = 0;
        for (size_t i = 0; i < expert_prefixes.size(); ++i) {
            const std::string& p = expert_prefixes[i];
            const PlaneSpec s = specs[i];
            // trellis: pread native -> permute -> append (byte-exact pure permutation)
            native.resize(s.trellis_bytes / 2);
            packed.resize(s.trellis_bytes / 2);
            std::string src_path;
            size_t off = 0, nb = 0;
            if (!model.locate(p + ".trellis", src_path, off, nb) || nb != s.trellis_bytes)
                throw std::runtime_error("exl3-band8: cannot locate source trellis for " + p);
            pread_exact(shard_fd(src_path), (uint8_t*)native.data(), s.trellis_bytes, off, src_path);
            permute_band8(native.data(), packed.data(), s.ki, s.nj, (size_t)(16 * s.bits + 8 * s.frac));
            append(p + ".trellis", kKindTrellis, s, (const uint8_t*)packed.data(), s.trellis_bytes);
            // scales: verbatim copies; after publish they are mapped from the CACHE, never the source
            scales.resize(std::max(s.suh_bytes, s.svh_bytes));
            for (int plane = 0; plane < 2; ++plane) {
                const std::string tensor = p + (plane == 0 ? ".suh" : ".svh");
                const size_t n = plane == 0 ? s.suh_bytes : s.svh_bytes;
                if (!model.locate(tensor, src_path, off, nb) || nb != n)
                    throw std::runtime_error("exl3-band8: cannot locate source " + tensor);
                pread_exact(shard_fd(src_path), scales.data(), n, off, src_path);
                append(tensor, uint8_t(plane == 0 ? kKindSuh : kKindSvh), s, scales.data(), n);
            }
            native.clear(); native.shrink_to_fit();
            packed.clear(); packed.shrink_to_fit();
            ++done_triples;
            if (opts.progress_every_triples && done_triples % opts.progress_every_triples == 0) {
                const double sec =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::fprintf(stderr,
                             "exl3-band8: packed %llu/%llu triples, %.2f GiB written, %.1f s elapsed\n",
                             (unsigned long long)done_triples,
                             (unsigned long long)expert_prefixes.size(),
                             (double)out.written() / (1ull << 30), sec);
            }
        }
        for (auto& kv : shard_fds) ::close(kv.second);

        // 3) Publish atomically, manifest LAST. A valid manifest is the single "complete"
        //    marker for this generation. Within one generation the payload rename can pair with
        //    an OLD manifest only during a same-fingerprint repair, where the deterministic
        //    layout and identical sources make old and new payloads byte-identical - so the
        //    pairing is always sound. Across generations nothing is shared, so a reader of any
        //    other generation is never exposed to this publish at all.
        out.sync();
        if (::rename(payload_partial.c_str(), payload.c_str()) != 0)
            throw std::runtime_error("exl3-band8: publish rename failed for payload.bin: " +
                                     std::string(std::strerror(errno)));
        if (opts.after_payload_publish_for_tests) opts.after_payload_publish_for_tests();
        // Source-stability recheck BEFORE stamping the manifest: if anything the fingerprint
        // covers changed while we were packing, the payload mixes old and new source bytes and
        // must not be published under `want`. Clean our partial and fail loudly.
        {
            const std::string idx2 = read_file(model_dir + "/model.safetensors.index.json");
            const uint64_t now = compute_fingerprint(model_dir, expert_prefixes,
                                                     exl3_detail::parse_weight_map(idx2));
            if (now != want)
                throw std::runtime_error("exl3-band8: source changed while packing (index/shard "
                                         "metadata moved); generation NOT published - rerun the "
                                         "build against the settled source");
        }
        std::sort(records.begin(), records.end(),
                  [](const Record& a, const Record& b) { return a.name < b.name; });
        for (size_t i = 1; i < records.size(); ++i)
            if (records[i - 1].name == records[i].name)
                throw std::runtime_error("exl3-band8: internal error: duplicate record built");
        Manifest m{};
        m.manifest_version = kManifestVersion;
        m.flags = opts.record_payload_checksum ? kPayloadChecksumFlag : 0;
        m.fingerprint = want;
        m.payload_bytes = out.written();
        m.payload_fnv = opts.record_payload_checksum ? out.fnv() : 0;
        m.records = records;
        std::vector<std::string> names;
        names.reserve(records.size());
        for (const Record& r : records) names.push_back(r.name);
        const std::vector<uint8_t> blob = encode_manifest(m, names);
        {
            FileHandle mf(::open(manifest_partial.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644));
            if (mf.fd < 0)
                throw std::runtime_error("exl3-band8: cannot create manifest partial: " +
                                         std::string(std::strerror(errno)));
            size_t wo = 0;
            while (wo < blob.size()) {
                const ssize_t w = ::write(mf.fd, blob.data() + wo, blob.size() - wo);
                if (w < 0 && errno == EINTR) continue;
                if (w <= 0)
                    throw std::runtime_error("exl3-band8: manifest write failed: " +
                                             std::string(std::strerror(errno)));
                wo += (size_t)w;
            }
            if (::fsync(mf.fd) != 0)
                throw std::runtime_error("exl3-band8: manifest fsync failed: " +
                                         std::string(std::strerror(errno)));
        }
        if (::rename(manifest_partial.c_str(), manifest.c_str()) != 0)
            throw std::runtime_error("exl3-band8: publish rename failed for manifest.bin: " +
                                     std::string(std::strerror(errno)));
        {
            FileHandle df(::open(gen.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
            if (df.fd >= 0) (void)::fsync(df.fd);   // best-effort durability of the renames
        }

        // Final verification through the SAME open path a later launch takes, then hand out the
        // mapped store. A failure here leaves a published-but-invalid generation; the next open
        // classifies it via decode_manifest and rebuilds in place.
        const size_t psz = payload_size_or_invalid(payload);
        const Manifest verified = decode_manifest(manifest.string(), psz);
        check_coverage(verified, expert_prefixes);
        if (opts.record_payload_checksum && verified.payload_fnv != out.fnv())
            throw CacheInvalid("published checksum does not match what was written");

        auto store = std::shared_ptr<Exl3Band8Store>(new Exl3Band8Store());
        store->impl_ = std::make_unique<Impl>();
        store->impl_->payload_path = payload.string();
        store->impl_->payload = std::make_unique<MappedFile>(payload.string());
        store->impl_->fingerprint = want;
        for (const Record& r : verified.records) store->impl_->by_name.emplace(r.name, r);
        return store;
    } catch (...) {
        clean_partials();   // only ever .partial names created by THIS builder
        throw;
    }
#endif
}

}  // namespace strata
