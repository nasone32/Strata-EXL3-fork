#include "strata/artifact/exl3_resident.hpp"
#include "strata/core/exl3_band8_permute.hpp"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata {
namespace {
size_t add(size_t a, size_t b) {
    if (a > SIZE_MAX - b) throw std::runtime_error("resident EXL3: size overflow");
    return a + b;
}
size_t round_page(size_t n) { return add(n, 4095) & ~size_t(4095); }
// The runtime repack's bounded per-projection budget, deliberately the offline store's
// default (Exl3Band8BuildOptions::max_projection_bytes): one native trellis plane plus
// its packed twin must fit beside the arena without a second expert-sized allocation.
constexpr size_t kRepackMaxProjectionBytes = 32ull << 20;
struct Plane { std::string path; size_t source = 0, bytes = 0, target = 0; };
struct Expert {
    std::array<Exl3Linear, 3> mats;
    std::array<Plane, 9> planes;
    size_t offset = 0, span = 0;
    enum State { GPU, Returning, RAM } state = GPU;
    bool locked = false;
};
}

struct Exl3ResidentBacking::Impl {
    std::unique_ptr<Exl3Model> source;
    std::map<std::pair<int,int>, Expert> experts;
    std::map<std::string, std::pair<std::pair<int,int>, int>> names;
    Exl3TrellisLayout layout = Exl3TrellisLayout::Native;
    Exl3ResidentRepack repack = Exl3ResidentRepack::None;
    uint8_t* base = nullptr;
    size_t bytes = 0;
    bool active = false;
    Options options;
    uint64_t reads = 0, read_bytes = 0, returned = 0, released = 0;
    // R1 prototype (STRATA_EXL3_RESIDENT_DIRECT_DMA) eligibility state - the backing's OWN
    // answer for bytes it committed, never a pointer-shape guess by a caller:
    //   * `spans` are the committed (registered) ranges sorted by arena offset; create()
    //     laid the experts end to end, so they tile the arena exactly;
    //   * `gpu_registered` is the registrar's own declaration (a trusted-caller assertion,
    //     see Options) - only the production CUDA registrar sets it;
    //   * `immutable_all` says the arena is the FULL, fully-committed, never-mutated set:
    //     activated with no GPU-only expert and untouched since.  The peer-complement
    //     exchange mutates (prepare_return/release_to_gpu between drained windows), so it
    //     never qualifies; any mutation clears the flag and disarms eligibility for every
    //     LATER question.
    // LIFETIME, stated plainly: clearing the flag protects FUTURE DMAs only.  A copy the
    // stream enqueued before the mutation may still be draining when drop() runs; that is
    // the pre-existing drained-window contract this class already documents ("mutations
    // are ONLY between drained verifier windows, never from service jobs") and the same
    // contract the expert stream's per-plane direct branch has always relied on for these
    // identical arena bytes.  R1 adds no new exposure: it merges nine already-direct
    // copies into one.
    bool gpu_registered = false;
    bool immutable_all = false;
    std::vector<std::pair<uint64_t,uint64_t>> spans;   // (offset, bytes) of the committed experts

    void lock(Expert& e) {
        std::string error;
        if (!options.lock(base + e.offset, e.span, error))
            throw std::runtime_error("resident EXL3: cannot lock expert RAM: " + error);
        e.locked = true;
    }
    void drop(Expert& e) {
        immutable_all = false;   // an exchange mutation: direct DMA is disarmed from here on
        if (e.locked) { options.unlock(base + e.offset, e.span); e.locked = false; }
#if !defined(_WIN32)
        if (madvise(base + e.offset, e.span, MADV_DONTNEED) != 0)
            throw std::runtime_error("resident EXL3: cannot release anonymous expert pages");
        if (mprotect(base + e.offset,e.span,PROT_NONE) != 0)
            throw std::runtime_error("resident EXL3: cannot protect GPU-only expert pages");
#endif
        e.state = Expert::GPU;
    }
};

Exl3ResidentBacking::Exl3ResidentBacking() : impl_(new Impl) {}
Exl3ResidentBacking::~Exl3ResidentBacking() {
    auto& m = *impl_;
    if (m.base) {
        for (auto& [key, e] : m.experts) if (e.locked) m.options.unlock(m.base + e.offset, e.span);
#if !defined(_WIN32)
        munmap(m.base, m.bytes);
#endif
    }
}

std::shared_ptr<Exl3ResidentBacking> Exl3ResidentBacking::create(
        const std::string& dir, const std::vector<Group>& groups,
        std::shared_ptr<const Exl3ExpertBacking> source,
        Exl3ResidentRepack repack) {
    if (groups.empty()) throw std::runtime_error("resident EXL3: empty expert list");
    if (repack == Exl3ResidentRepack::ToBand8) {
        // The runtime repack reads the NATIVE shards and produces the packed bytes in RAM;
        // a source backing would either duplicate a packed copy or mix two sources.
        if (source && source->layout() != Exl3TrellisLayout::Native)
            throw std::runtime_error("resident EXL3: runtime repack needs the native shards - "
                                     "a packed source backing is already canonical (pass no "
                                     "source, or repack None)");
    }
    auto result = std::shared_ptr<Exl3ResidentBacking>(new Exl3ResidentBacking);
    auto& m = *result->impl_;
    m.source = std::make_unique<Exl3Model>(dir, std::move(source));
    m.layout = m.source->expert_layout();
    if (repack == Exl3ResidentRepack::ToBand8 && m.layout != Exl3TrellisLayout::Native)
        throw std::runtime_error("resident EXL3: runtime repack needs a native-layout source");
    m.repack = repack;
    for (const auto& g : groups) {
        const auto key = std::make_pair(g.layer, g.expert);
        if (g.layer < 0 || g.expert < 0 || m.experts.count(key))
            throw std::runtime_error("resident EXL3: duplicate/invalid expert key");
        Expert e;
        e.offset = m.bytes;
        size_t at = 0;
        for (int p = 0; p < 3; ++p) {
            e.mats[p] = m.source->linear(g.projections[p]);
            if (e.mats[p].layout != m.layout || !m.names.emplace(g.projections[p], std::make_pair(key,p)).second)
                throw std::runtime_error("resident EXL3: duplicate projection or mixed layout");
            if (repack == Exl3ResidentRepack::ToBand8) {
                // The SAME packability rules the offline store enforces per projection,
                // checked here (before any arena byte exists): mul1 codebook, one of the
                // four feasibility-verified rates, H128-aligned dims (Exl3Model::linear
                // already proved those), and a bounded trellis plane for the scratch.
                const Exl3Linear& L = e.mats[p];
                if (L.cb != 2)
                    throw std::runtime_error("resident EXL3: runtime repack is mul1-only - " +
                                             g.projections[p] + " lacks the mul1 marker");
                if (!exl3_band8_verified_rate(L.bits, L.frac))
                    throw std::runtime_error("resident EXL3: runtime repack rate K=" +
                                             std::to_string(L.bits + 0.5 * L.frac) +
                                             " is outside the verified set {2, 2.5, 3, 4} (" +
                                             g.projections[p] + ")");
                const size_t trellis = (size_t)L.ki * (size_t)L.nj * (size_t)L.words() * 2;
                if (trellis > kRepackMaxProjectionBytes)
                    throw std::runtime_error("resident EXL3: runtime repack trellis plane " +
                                             std::to_string(trellis >> 20) + " MiB exceeds the "
                                             "bounded budget " +
                                             std::to_string(kRepackMaxProjectionBytes >> 20) +
                                             " MiB (" + g.projections[p] + ")");
                // Final tag from the start: the arena WILL hold permuted bytes. Between
                // create() and activate() linear()/locate() throw (below), so the tag can
                // never label bytes that were not converted yet.
                e.mats[p].layout = Exl3TrellisLayout::Band8;
            }
            const char* suffix[] = {".trellis", ".suh", ".svh"};
            const auto& l = e.mats[p];
            const size_t expected[] = {(size_t)l.ki * l.nj * l.words() * 2, (size_t)l.in() * 2, (size_t)l.out() * 2};
            for (int q = 0; q < 3; ++q) {
                auto& plane = e.planes[p*3+q];
                if (!m.source->locate(g.projections[p] + suffix[q], plane.path, plane.source, plane.bytes) ||
                        plane.bytes != expected[q] || plane.source > SIZE_MAX - plane.bytes)
                    throw std::runtime_error("resident EXL3: missing/invalid plane " + g.projections[p] + suffix[q]);
                plane.target = at;
                at = add(at, plane.bytes);
            }
        }
        e.span = round_page(at);
        m.bytes = add(m.bytes, e.span);
        m.experts.emplace(key, std::move(e));
    }
    if (repack == Exl3ResidentRepack::ToBand8) m.layout = Exl3TrellisLayout::Band8;
    return result;
}

Exl3TrellisLayout Exl3ResidentBacking::layout() const noexcept { return impl_->layout; }
bool Exl3ResidentBacking::active() const noexcept { return impl_->active; }
bool Exl3ResidentBacking::memory_only() const noexcept { return active(); }
bool Exl3ResidentBacking::covers_tensor(const std::string& name) const {
    for (const char* suffix : {".trellis", ".suh", ".svh"}) {
        const size_t n = std::strlen(suffix);
        if (name.size() >= n && name.compare(name.size()-n,n,suffix) == 0)
            return impl_->names.count(name.substr(0,name.size()-n)) != 0;
    }
    return false;
}
bool Exl3ResidentBacking::linear(const std::string& name, Exl3Linear& out) const {
    const auto& m = *impl_;
    auto it = m.names.find(name);
    if (it == m.names.end()) return false;
    if (!m.active) {
        if (m.repack == Exl3ResidentRepack::ToBand8)
            throw std::runtime_error("resident EXL3: runtime repack not converted yet - the "
                                     "backing serves Band8 bytes only after activate() (native "
                                     "pointers must never travel under a Band8 mode probe)");
        if (!m.source) throw std::runtime_error("resident EXL3: source retired; activation incomplete");
        out = m.source->linear(name); return true;
    }
    const auto& e = m.experts.at(it->second.first);
    const int p = it->second.second;
    out = e.mats[p];
    out.trellis = (const uint16_t*)(m.base + e.offset + e.planes[p*3].target);
    out.suh = (const uint16_t*)(m.base + e.offset + e.planes[p*3+1].target);
    out.svh = (const uint16_t*)(m.base + e.offset + e.planes[p*3+2].target);
    return true;
}
bool Exl3ResidentBacking::locate(const std::string& name, std::string& path, size_t& off, size_t& bytes) const {
    if (impl_->active || !covers_tensor(name)) return false;
    if (impl_->repack == Exl3ResidentRepack::ToBand8)
        throw std::runtime_error("resident EXL3: runtime repack has no file tier - the packed "
                                 "bytes exist only in RAM after activate() (a pre-activation "
                                 "locate would leak unconverted native ranges)");
    if (!impl_->source) throw std::runtime_error("resident EXL3: source retired; activation incomplete");
    return impl_->source->locate(name,path,off,bytes);
}
uint64_t Exl3ResidentBacking::required_bytes(const std::vector<std::pair<int,int>>& gpu) const {
    std::set<std::pair<int,int>> skip;
    for (auto key : gpu) if (!impl_->experts.count(key) || !skip.insert(key).second)
        throw std::runtime_error("resident EXL3: unknown/duplicate GPU-only expert");
    uint64_t n = 0;
    for (const auto& [key,e] : impl_->experts) if (!skip.count(key)) n += e.span;
    return n;
}

void Exl3ResidentBacking::activate(const std::vector<std::pair<int,int>>& gpu, Options options) {
#if defined(_WIN32)
    throw std::runtime_error("resident EXL3: Windows allocation/locking not validated");
#else
    auto& m = *impl_;
    if (m.base || m.active || !options.lock || !options.unlock)
        throw std::runtime_error("resident EXL3: activation needs fresh backing and lock/unlock callbacks");
    if (m.repack == Exl3ResidentRepack::ToBand8 && !gpu.empty())
        throw std::runtime_error("resident EXL3: runtime in-memory repack supports the FULL arena "
                                 "only - a peer-complement activation would stage unconverted "
                                 "native bytes before conversion (refused, not guessed)");
    const uint64_t need = required_bytes(gpu);
    // Reserve one maximum-batch worth of return targets (256 MiB) beside the live
    // complement, so a copy-back cannot overcommit before incoming RAM is freed.
    if (need > options.available_bytes || options.headroom_bytes > options.available_bytes - need ||
            (256ull << 20) > options.available_bytes - need - options.headroom_bytes)
        throw std::runtime_error("resident EXL3: insufficient available RAM for resident weights + headroom + exchange");
    m.options = std::move(options);
    void* base = mmap(nullptr,m.bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (base == MAP_FAILED) throw std::runtime_error("resident EXL3: anonymous reservation failed");
    m.base = (uint8_t*)base;
    std::set<std::pair<int,int>> skip(gpu.begin(),gpu.end());
    // Read via bounded, aligned O_DIRECT requests: no second 60–80 GiB file-cache
    // population beside pinned RAM. Never evict another process's file pages.
    size_t max_span = 0;
    for (const auto& [key,e] : m.experts) max_span = std::max(max_span,e.span);
    if (max_span > (256u << 20)) throw std::runtime_error("resident EXL3: expert exceeds 256 MiB load bound");
    void* raw = nullptr;
    if (posix_memalign(&raw,4096,add(max_span,8192)) != 0) throw std::bad_alloc();
    std::unique_ptr<void,decltype(&std::free)> scratch(raw,&std::free);
    // Initial GPU fills are drained before activation. All source ranges and
    // matrix metadata were captured in create(); direct loading needs no mapped
    // pointers. Retire the mmap BEFORE committing the new arena, not afterward:
    // a GPU runtime can retain pins on pageable source pages until unmapping.
    // Keeping those pages alongside the full RAM pool can defeat MemAvailable's
    // reclaimable-file estimate. No fallback is permitted after this point.
    m.source.reset();
    struct Files { std::map<std::string,int> fd; ~Files() { for (auto [p,f] : fd) close(f); } } files;
    auto read = [&](const std::string& path, size_t offset, size_t n) {
        auto it = files.fd.find(path);
        if (it == files.fd.end()) {
            int flags = O_RDONLY;
#if defined(O_DIRECT)
            if (m.options.direct_io) flags |= O_DIRECT;
#else
            if (m.options.direct_io) throw std::runtime_error("resident EXL3: direct loading unsupported");
#endif
            const int fd = open(path.c_str(),flags);
            if (fd < 0) throw std::runtime_error("resident EXL3: load open " + path + ": " + std::strerror(errno));
            it = files.fd.emplace(path,fd).first;
        }
        const size_t start = offset & ~size_t(4095), prefix = offset-start;
        const size_t want = round_page(add(prefix,n));
        ssize_t got;
        do { got = pread(it->second,raw,want,(off_t)start); } while (got < 0 && errno == EINTR);
        if (got < 0 || (size_t)got < prefix+n)
            throw std::runtime_error("resident EXL3: short/error load " + path + ": " + std::strerror(errno));
        ++m.reads; m.read_bytes += got;
        return (const uint8_t*)raw + prefix;
    };
    size_t done = 0;
    const bool repack = m.repack == Exl3ResidentRepack::ToBand8;
    for (auto& [key,e] : m.experts) {
        if (!skip.count(key)) {
            m.lock(e); // commit/pin a bounded expert at a time, before writing
            if (!repack) {
                size_t lo = SIZE_MAX, hi = 0;
                bool one_file = true;
                for (const auto& p : e.planes) {
                    one_file &= p.path == e.planes[0].path;
                    lo = std::min(lo,p.source); hi = std::max(hi,p.source+p.bytes);
                }
                if (one_file && hi-lo <= e.span) {
                    const auto* src = read(e.planes[0].path,lo,hi-lo);
                    for (const auto& p : e.planes) std::memcpy(m.base+e.offset+p.target,src+p.source-lo,p.bytes);
                } else for (const auto& p : e.planes)
                    std::memcpy(m.base+e.offset+p.target,read(p.path,p.source,p.bytes),p.bytes);
            } else {
                // Runtime repack: bounded native read + the ONE shared Band8 permutation
                // straight into the final registered RAM (one projection's trellis is in
                // the scratch at a time - no second expert-sized allocation); the scale
                // planes are verbatim copies. No file anywhere is written.
                for (int p = 0; p < 3; ++p)
                    for (int q = 0; q < 3; ++q) {
                        const Plane& pl = e.planes[p*3+q];
                        const uint8_t* src = read(pl.path,pl.source,pl.bytes);
                        uint8_t* dst = m.base+e.offset+pl.target;
                        if (q == 0)
                            exl3_band8_permute((const uint16_t*)src,(uint16_t*)dst,
                                (uint32_t)e.mats[p].ki,(uint32_t)e.mats[p].nj,
                                (size_t)e.mats[p].words());
                        else
                            std::memcpy(dst,src,pl.bytes);
                    }
            }
            e.state = Expert::RAM;
        } else if (mprotect(m.base+e.offset,e.span,PROT_NONE) != 0) {
            throw std::runtime_error("resident EXL3: cannot protect initial GPU-only expert pages");
        }
        if (m.options.progress && (++done % 256 == 0 || done == m.experts.size()))
            m.options.progress(done,m.experts.size());
    }
    m.active = true;
    // R1 eligibility: publish the committed spans (sorted by offset; they tile the arena)
    // and arm the answer only for the FULL, fully-committed arena whose registrar really
    // GPU-registered it. A peer-complement activation keeps gpu_only non-empty and never
    // arms it; a later exchange mutation clears `immutable_all` in drop().
    for (const auto& [key,e] : m.experts) m.spans.emplace_back((uint64_t) e.offset,(uint64_t) e.span);
    std::sort(m.spans.begin(), m.spans.end());
    m.gpu_registered = m.options.gpu_registered;
    m.immutable_all = gpu.empty();
#endif
}
bool Exl3ResidentBacking::direct_dma_pinned(const void* ptr, size_t bytes) const noexcept {
    const auto& m = *impl_;
    if (!m.active || !m.immutable_all || !m.gpu_registered || ptr == nullptr || bytes == 0)
        return false;
    // INTEGER range validation only: the API accepts FOREIGN pointers (callers probe, tests
    // pass stack/heap/other-mapping addresses), and comparing or subtracting unrelated
    // POINTERS is unspecified/undefined - so both sides go through uintptr_t first.  Every
    // arithmetic step is bounded below by the arena's own size, so nothing wraps.
    const uintptr_t base = (uintptr_t) m.base;
    const uintptr_t p = (uintptr_t) ptr;
    if (p < base) return false;                       // below the arena
    const uint64_t off = (uint64_t) (p - base);
    if (off >= m.bytes || bytes > (size_t) (m.bytes - off)) return false;   // the whole range is arena bytes
    // One DMA reads from ONE registration: the span containing the start must also contain
    // the end, so a range straddling two experts' committed spans is refused even though
    // every byte of it is pinned (a caller's nine planes never straddle - this guards the
    // contract, not an expected shape).  off + bytes <= m.bytes bounds the addition.
    const auto it = std::upper_bound(m.spans.begin(), m.spans.end(), off,
        [](uint64_t value, const std::pair<uint64_t,uint64_t>& s) { return value < s.first; });
    if (it == m.spans.begin()) return false;
    const auto& span = *(it - 1);
    return off + (uint64_t) bytes <= span.first + span.second;
}
bool Exl3ResidentBacking::resident(int l,int e) const {
    return impl_->experts.at({l,e}).state == Expert::RAM;
}
bool Exl3ResidentBacking::has_gpu_only() const {
    for (const auto& [key,e] : impl_->experts) if (e.state != Expert::RAM) return true;
    return false;
}
bool Exl3ResidentBacking::prepare_return(int l,int e) {
    auto& m = *impl_; auto& x = m.experts.at({l,e});
    if (!m.active) throw std::runtime_error("resident EXL3: return before activation");
    if (x.state == Expert::RAM) return false;
    if (x.state != Expert::GPU) throw std::runtime_error("resident EXL3: return already pending");
#if !defined(_WIN32)
    if (mprotect(m.base+x.offset,x.span,PROT_READ|PROT_WRITE) != 0)
        throw std::runtime_error("resident EXL3: cannot open expert return target");
#endif
    try { m.lock(x); }
    catch (...) {
#if !defined(_WIN32)
        (void)mprotect(m.base+x.offset,x.span,PROT_NONE);
#endif
        throw;
    }
    x.state = Expert::Returning;
    return true;
}
void Exl3ResidentBacking::finish_return(int l,int e) {
    auto& x = impl_->experts.at({l,e});
    if (x.state != Expert::Returning) throw std::runtime_error("resident EXL3: no pending return");
    x.state = Expert::RAM; ++impl_->returned;
}
void Exl3ResidentBacking::cancel_return(int l,int e) {
    auto& x = impl_->experts.at({l,e});
    if (x.state == Expert::Returning) impl_->drop(x);
}
void Exl3ResidentBacking::release_to_gpu(int l,int e) {
    auto& x = impl_->experts.at({l,e});
    if (!impl_->active || x.state != Expert::RAM) throw std::runtime_error("resident EXL3: release needs RAM owner");
    impl_->drop(x); ++impl_->released;
}
Exl3ResidentBacking::Stats Exl3ResidentBacking::stats() const {
    const auto& m = *impl_; Stats s{m.bytes,0,0,m.reads,m.read_bytes,m.returned,m.released};
    for (const auto& [key,e] : m.experts) {
        if (e.state == Expert::RAM) s.resident_bytes += e.span;
        if (e.locked) s.locked_bytes += e.span;
    }
    return s;
}
} // namespace strata
