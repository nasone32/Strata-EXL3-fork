// src/kernels/exl3_peer.cpp - see include/strata/kernels/exl3_peer.hpp for the API contract.
#include "strata/kernels/exl3_peer.hpp"

#include "strata/core/exl3_flags.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/cpu/f16_convert.hpp"
#include "strata/kernels/exl3_route_plan.hpp"

#include <cuda_runtime.h>

#include <array>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <utility>

namespace strata::kernels {

namespace {

/// Checked device switch; the caller restores its own device best-effort afterwards.
bool switch_to(int dev) { return cudaSetDevice(dev) == cudaSuccess; }

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("exl3 peer: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

bool mul_ok(size_t a, size_t b, size_t* out) {
    if (a != 0 && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

bool add_ok(size_t a, size_t b, size_t* out) {
    if (a > SIZE_MAX - b) return false;
    *out = a + b;
    return true;
}

bool layout(const Exl3Peer::Entry& e, int n, int hidden, std::array<size_t, 9>& bytes) {
    const Exl3Mat mats[] = {e.g, e.u, e.d};
    if (e.layer < 0 || e.expert < 0) return false;
    for (int p = 0; p < 3; ++p) {
        const auto& m = mats[p];
        if (!m.trellis || !m.suh || !m.svh || m.ki <= 0 || m.nj <= 0 ||
            m.ki > INT32_MAX/16 || m.nj > INT32_MAX/16 || m.ki % 8 || m.nj % 8 ||
            m.bits < 1 || m.bits > 8 || m.cb < 0 || m.cb > 2) return false;
        size_t tiles = 0;
        if (!mul_ok((size_t) m.ki, (size_t) m.nj, &tiles) ||
            !mul_ok(tiles, (size_t) m.bits * 32, &bytes[p*3])) return false;
        bytes[p*3+1] = (size_t) m.ki * 32;
        bytes[p*3+2] = (size_t) m.nj * 32;
        for (int q = 0; q < 3; ++q) if (bytes[p*3+q] > ((size_t) 1 << 32)) return false;
    }
    return e.g.ki*16 == n && e.g.nj*16 == hidden && e.u.ki == e.g.ki &&
           e.u.nj == e.g.nj && e.d.ki == e.g.nj && e.d.nj == e.g.ki;
}

struct PackedPlan {
    size_t x, indices, gate, up, down, first, rows, bytes;
    PackedPlan(int tokens, int n, int routes, int groups) {
        size_t at = 0;
        auto take = [&](size_t bytes) { const size_t offset = at; at = (at + bytes + 255) & ~(size_t)255; return offset; };
        x = take((size_t)tokens * n * sizeof(float));
        indices = take((size_t)routes * sizeof(int));
        gate = take((size_t)groups * sizeof(Exl3Mat));
        up = take((size_t)groups * sizeof(Exl3Mat));
        down = take((size_t)groups * sizeof(Exl3Mat));
        first = take((size_t)groups * sizeof(int));
        rows = take((size_t)groups * sizeof(int));
        this->bytes = at;
    }
};

}  // namespace

struct Exl3Peer::Impl {
    int device = -1;
    int n_embd = 0;
    int hidden = 0;
    int max_tokens = 0, max_k = 0;
    int64_t max_routes = 0;
    bool failed = false;
    bool pending = false;
    bool deferred = false;
    int64_t launches = 0, rows = 0, enqueues = 0;
    int pending_rows = 0;
    bool profile_finish = false;
    bool vector_convert = false;
    int multirow = 0;
    bool pack_plan = false;
    uint8_t *h_plan = nullptr, *d_plan = nullptr;
    size_t plan_capacity = 0;
    int64_t packed_calls = 0, packed_bytes = 0;
    int64_t grouped_calls = 0, grouped_rows = 0, grouped_groups = 0;
    int64_t finishes = 0;
    double wait_ms = 0, scatter_ms = 0;

    // per-instance test seam (no globals: independent instances cannot race):
    // STRATA_EXL3_PEER_FAULT=N fails the Nth injectable operation of the next open()/launch().
    int fault_target = -1;
    int64_t fault_ops = 0;
    void fault_arm(const char* env_name) {
        fault_ops = 0;
        const char* f = std::getenv(env_name);
        fault_target = (f && f[0]) ? std::atoi(f) : -1;
    }
    bool fault_trip(std::string& err, const char* what) {
        if (fault_target >= 0 && ++fault_ops == fault_target) {
            err = std::string("exl3 peer: injected fault at ") + what;
            return true;
        }
        return false;
    }

    struct DevEntry {
        int64_t layer = -1;
        int32_t expert = -1;
        // Ownership rule: a plane pointer becomes owned the moment its allocation succeeds
        // (published here immediately); the ONLY cleanup path is free_all(), exactly once.
        uint8_t* plane[9] = {};
        Exl3Mat g, u, d;            // device pointers
        size_t bytes = 0;
        std::array<size_t, 9> capacity{};
    };
    std::vector<DevEntry> dev;
    std::map<std::pair<int64_t, int32_t>, size_t> where;

    float* d_x32 = nullptr;
    uint16_t* d_x16 = nullptr;
    int* d_rows_idx = nullptr;
    uint16_t* d_rows16 = nullptr;

    float* h_x32 = nullptr;        // pinned: launch copies the request here before returning
    int32_t* h_rows_idx = nullptr;
    int *h_group_first = nullptr, *h_group_rows = nullptr;
    uint16_t* h_rows16 = nullptr;
    Exl3Mat *h_g = nullptr, *h_u = nullptr, *h_d = nullptr;   // pinned per-launch descriptors

    std::vector<int> row_of;

    cudaStream_t stream = nullptr;
    cudaStream_t refill = nullptr;
    cudaEvent_t refill_done = nullptr;
    cudaEvent_t refill_start = nullptr;
    double refill_ms = 0;
    uint8_t* h_refill = nullptr;
    size_t refill_capacity = 0;
    int refill_max = 0;
    std::vector<Replacement> replacing;
    std::vector<size_t> refill_slots;
    std::unique_ptr<Exl3MoeWorkspace> ws;   // created under the peer device guard (it binds to
                                            // the device current at construction)

    // ---- STRATA_EXL3_PEER_PREFILL: the prompt chunks' peer-owned experts (enable_prefill) ----
    // Every buffer is allocated once by enable_prefill() at setup; a layer's work is enqueued by
    // launch_prefill() as: one H2D of the mapped FP16 activations, one pinned plan slab, the
    // grouped WMMA FFN per expert group into the unweighted rows slab, one scatter-add per group
    // into the per-token sums, and one D2H of the sums into mapped host memory.
    bool pf_on = false;
    bool pf_pending = false;
    bool pf_batch = false;             // STRATA_EXL3_PEER_BATCH=1: consecutive uniform groups per
                                       // launch through exl3_moe_expert_batch (the C1 launch
                                       // compaction on the peer's prefill chain)
    int pf_k = 0;
    int64_t pf_n_expert = 0;
    int64_t pf_tokens_cap = 0, pf_rows_cap = 0;
    int64_t pf_last_rows = 0, pf_layers = 0, pf_rows_total = 0, pf_groups_total = 0, pf_batches = 0;
    uint16_t* pf_d_x16 = nullptr;       // tokens x n_embd FP16 activations
    float* pf_d_rows = nullptr;         // rows_peer x n_embd unweighted FFN rows
    float* pf_d_sum = nullptr;          // tokens x n_embd weighted per-token sums
    float* pf_d_wk = nullptr;           // tokens x k router weights
    int32_t* pf_d_ires = nullptr;       // per peer route: its token's row in x16
    int32_t* pf_d_pair = nullptr;       // per peer route: its position in the ids array (t*k + kk)
    int32_t* pf_d_dest = nullptr;       // per route: its row in the rows slab (a static iota)
    Exl3Group* pf_d_grp = nullptr;      // per group descriptors (batched path; first = within-batch)
    float* pf_host_sum = nullptr;       // mapped+portable: the sums D2H target the caller reads
    cudaEvent_t pf_done = nullptr;
    // pinned staging, uploaded as one slab per layer: [ires | pair | wk | dest | groups]
    uint8_t* pf_h_slab = nullptr;
    uint8_t* pf_d_slab = nullptr;
    size_t pf_slab_bytes = 0;
    size_t pf_off[5] = {};              // the slab layout: ires, pair, wk, dest, groups
    int pf_groups = 0;
    std::vector<int> pf_first;          // per group: its absolute row in the rows slab (host-side)
    std::vector<int> pf_rowsn;          // per group: its row count (host-side)
    std::vector<int> pf_batch_end;      // per group: one-past its batch's last group, or -1: standalone
    void pf_free() {
        // device buffers (cudaFree) / host allocations (cudaFreeHost; covers pinned and mapped)
        for (void** p : {(void**) &pf_d_x16, (void**) &pf_d_rows, (void**) &pf_d_sum, (void**) &pf_d_wk,
                         (void**) &pf_d_ires, (void**) &pf_d_pair, (void**) &pf_d_dest,
                         (void**) &pf_d_grp, (void**) &pf_d_slab})
            if (*p) { (void) cudaFree(*p); *p = nullptr; }
        for (void** p : {(void**) &pf_h_slab, (void**) &pf_host_sum})
            if (*p) { (void) cudaFreeHost(*p); *p = nullptr; }
        if (pf_done) { (void) cudaEventDestroy(pf_done); pf_done = nullptr; }
        pf_on = pf_pending = false;
        pf_slab_bytes = 0;
    }

    void free_all() {
        for (DevEntry& e : dev)
            for (uint8_t*& p : e.plane) { if (p) { (void) cudaFree(p); p = nullptr; } }
        dev.clear();
        where.clear();
        row_of.clear();
        for (void** p : {(void**) &d_x32, (void**) &d_x16, (void**) &d_rows_idx, (void**) &d_rows16, (void**) &d_plan})
            if (*p) { (void) cudaFree(*p); *p = nullptr; }
        for (void** p : {(void**) &h_x32, (void**) &h_rows_idx, (void**) &h_rows16,
                         (void**) &h_g, (void**) &h_u, (void**) &h_d,
                         (void**) &h_group_first, (void**) &h_group_rows, (void**) &h_plan})
            if (*p) { (void) cudaFreeHost(*p); *p = nullptr; }
        ws.reset();
        pack_plan = false; plan_capacity = 0;
        if (h_refill) { (void) cudaFreeHost(h_refill); h_refill = nullptr; }
        if (refill_done) { (void) cudaEventDestroy(refill_done); refill_done = nullptr; }
        if (refill_start) { (void) cudaEventDestroy(refill_start); refill_start = nullptr; }
        if (refill) { (void) cudaStreamDestroy(refill); refill = nullptr; }
        replacing.clear(); refill_slots.clear(); refill_capacity = 0; refill_max = 0; refill_ms = 0;
        if (stream) { (void) cudaStreamDestroy(stream); stream = nullptr; }
    }
    // A failed device switch at close: release nothing device-side (freeing across contexts is
    // not safe) - the process exit reclaims; tracking is dropped so nothing is double-freed.
    void drop_without_free() {
        dev.clear();
        where.clear();
        row_of.clear();
        d_x32 = nullptr;
        d_x16 = nullptr;
        d_rows_idx = nullptr;
        d_rows16 = nullptr;
        h_x32 = nullptr;
        h_rows_idx = nullptr;
        h_group_first = h_group_rows = nullptr;
        h_rows16 = nullptr;
        h_g = h_u = h_d = nullptr;
        h_plan = d_plan = nullptr; pack_plan = false; plan_capacity = 0;
        pf_d_x16 = nullptr; pf_d_rows = pf_d_sum = nullptr;
        pf_d_slab = nullptr; pf_h_slab = nullptr; pf_host_sum = nullptr;
        pf_d_ires = pf_d_pair = nullptr; pf_d_wk = nullptr;
        pf_d_dest = nullptr; pf_d_grp = nullptr;
        pf_done = nullptr; pf_on = pf_pending = false; pf_slab_bytes = 0;
        ws.release();
        stream = nullptr;
        refill = nullptr; refill_done = refill_start = nullptr; h_refill = nullptr;
        replacing.clear(); refill_slots.clear(); refill_capacity = 0; refill_max = 0;
    }
};

Exl3Peer::Exl3Peer() : impl_(new Impl()) {}
Exl3Peer::~Exl3Peer() { close(); }

bool Exl3Peer::valid() const { return impl_ && impl_->device >= 0 && !impl_->failed; }
int Exl3Peer::device() const { return impl_ ? impl_->device : -1; }
bool Exl3Peer::pending() const { return impl_ && impl_->pending; }
bool Exl3Peer::failed() const { return impl_ && impl_->failed; }
int64_t Exl3Peer::launches() const { return impl_ ? impl_->launches : 0; }
int64_t Exl3Peer::rows() const { return impl_ ? impl_->rows : 0; }
int64_t Exl3Peer::enqueues() const { return impl_ ? impl_->enqueues : 0; }

bool Exl3Peer::has(int64_t layer, int32_t expert) const {
    if (!valid()) return false;
    const auto it = impl_->where.find({layer, expert});
    return it != impl_->where.end() && it->second != SIZE_MAX;
}

int64_t Exl3Peer::resident() const {
    return valid() ? (int64_t) (impl_->where.size() - impl_->replacing.size()) : 0;
}

size_t Exl3Peer::bytes() const {
    if (!valid()) return 0;
    size_t total = 0;
    for (const Impl::DevEntry& e : impl_->dev) total += e.bytes;
    return total;
}

bool Exl3Peer::busy() const {
    if (impl_ == nullptr || impl_->device < 0) return false;
    int caller = -1;
    (void) cudaGetDevice(&caller);
    if (!switch_to(impl_->device)) return false;
    struct Restore {
        int prev;
        ~Restore() { (void) cudaSetDevice(prev); }
    } restore{caller};
    return cudaStreamQuery(impl_->stream) != cudaSuccess;
}

bool Exl3Peer::refilling() const { return impl_ && !impl_->replacing.empty(); }
double Exl3Peer::refill_ms() const { return impl_->refill_ms; }

void Exl3Peer::residency(int layers, int experts, std::vector<int32_t>& out) const {
    out.assign((size_t) layers * experts, -1);
    if (!valid()) return;
    for (const auto& item : impl_->where)
        if (item.second != SIZE_MAX && item.first.first < layers && item.first.second < experts)
            out[(size_t) item.first.first * experts + item.first.second] = (int32_t) item.second;
}

bool Exl3Peer::enable_adaptation(int max_swaps, size_t byte_cap, std::string& err) {
    if (!valid() || pending() || refilling() || max_swaps <= 0 || impl_->refill) {
        err = "exl3 peer: adaptation setup needs a valid idle tier and positive swap cap"; return false;
    }
    auto& m = *impl_;
    size_t largest = 0, capacity = 0;
    for (const auto& e : m.dev) largest = std::max(largest, e.bytes);
    max_swaps = (int) std::min<size_t>((size_t) max_swaps, m.dev.size());
    if (!mul_ok(largest, (size_t) max_swaps, &capacity)) { err = "exl3 peer: refill capacity overflow"; return false; }
    if (byte_cap) capacity = std::min(capacity, byte_cap);
    if (capacity == 0 || capacity > ((size_t) 1 << 30)) { err = "exl3 peer: refill staging must fit in 1 GiB"; return false; }
    int caller = -1; (void) cudaGetDevice(&caller);
    if (!switch_to(m.device)) { err = "exl3 peer: refill device switch failed"; return false; }
    struct Restore { int prev; ~Restore() { (void) cudaSetDevice(prev); } } restore{caller};
    try {
        m.replacing.reserve(max_swaps);
        m.refill_slots.reserve(max_swaps);
        if (!ck(cudaStreamCreateWithFlags(&m.refill, cudaStreamNonBlocking), "refill stream", err) ||
            !ck(cudaEventCreateWithFlags(&m.refill_done, 0), "refill event", err) ||
            !ck(cudaEventCreateWithFlags(&m.refill_start, 0), "refill start event", err) ||
            !ck(cudaHostAlloc((void**) &m.h_refill, capacity, cudaHostAllocPortable), "refill staging", err)) {
            close(); return false;
        }
        m.refill_capacity = capacity; m.refill_max = max_swaps;
    } catch (const std::exception& e) { err = e.what(); close(); return false; }
    return true;
}

bool Exl3Peer::can_replace(const Replacement& r) const {
    if (!valid() || r.out == r.in.expert || has(r.in.layer, r.in.expert)) return false;
    const auto it = impl_->where.find({r.in.layer, r.out});
    if (it == impl_->where.end() || it->second == SIZE_MAX) return false;
    std::array<size_t, 9> sizes{};
    if (!layout(r.in, impl_->n_embd, impl_->hidden, sizes)) return false;
    const auto& slot = impl_->dev[it->second];
    for (size_t i = 0; i < sizes.size(); ++i) if (sizes[i] > slot.capacity[i]) return false;
    return true;
}

size_t Exl3Peer::replacement_bytes(const Replacement& r) const {
    if (!can_replace(r)) return 0;
    std::array<size_t, 9> sizes{}; (void) layout(r.in, impl_->n_embd, impl_->hidden, sizes);
    size_t total = 0;
    for (size_t b : sizes) if (!add_ok(total, b, &total)) return 0;
    return total;
}

bool Exl3Peer::replace(const std::vector<Replacement>& replacements, std::string& err) {
    auto& m = *impl_;
    if (!valid() || pending() || refilling() || !m.refill || replacements.size() > (size_t) m.refill_max) {
        err = "exl3 peer: refill needs an idle configured tier and a bounded batch"; return false;
    }
    if (replacements.empty()) return true;
    size_t total = 0;
    for (size_t i = 0; i < replacements.size(); ++i) {
        const auto& r = replacements[i]; const size_t b = replacement_bytes(r);
        if (!b || !add_ok(total, b, &total) || total > m.refill_capacity) {
            err = "exl3 peer: invalid/incompatible replacement or refill byte cap exceeded"; return false;
        }
        for (size_t j = 0; j < i; ++j) if (r.in.layer == replacements[j].in.layer &&
            (r.out == replacements[j].out || r.in.expert == replacements[j].in.expert)) {
            err = "exl3 peer: duplicate refill source/destination"; return false;
        }
    }
    int caller = -1; (void) cudaGetDevice(&caller);
    if (!switch_to(m.device)) { err = "exl3 peer: refill device switch failed"; return false; }
    struct Restore { int prev; ~Restore() { (void) cudaSetDevice(prev); } } restore{caller};
    m.fault_arm("STRATA_EXL3_PEER_FAULT_REFILL");
    try {
        // Preallocate map nodes before enqueuing anything. SIZE_MAX is an unpublished entry;
        // apply_refill performs only non-allocating assignments after DMA completion.
        for (const auto& r : replacements) m.where.emplace(std::make_pair(r.in.layer, r.in.expert), SIZE_MAX);
        size_t offset = 0;
        for (const auto& r : replacements) {
            const Exl3Mat mats[] = {r.in.g, r.in.u, r.in.d};
            std::array<size_t, 9> sizes{}; (void) layout(r.in, m.n_embd, m.hidden, sizes);
            for (int p = 0; p < 3; ++p) {
                const void* src[] = {mats[p].trellis, mats[p].suh, mats[p].svh};
                for (int q = 0; q < 3; ++q) {
                    const size_t b = sizes[p*3+q];
                    std::memcpy(m.h_refill + offset, src[q], b); offset += b;
                }
            }
        }
        m.replacing = replacements;
        m.refill_slots.clear();
        if (!ck(cudaEventRecord(m.refill_start, m.refill), "refill start", err)) { m.failed = true; return false; }
        // Source pointers are no longer read after this call: all bytes are in pinned staging.
        offset = 0;
        for (const auto& r : m.replacing) {
            const size_t slot = m.where.at({r.in.layer, r.out});
            m.refill_slots.push_back(slot);
            auto& d = m.dev[slot];
            std::array<size_t, 9> sizes{}; (void) layout(r.in, m.n_embd, m.hidden, sizes);
            m.where.erase({r.in.layer, r.out});
            for (int p = 0; p < 3; ++p) {
                for (int q = 0; q < 3; ++q) {
                    const size_t b = sizes[p*3+q];
                    if (m.fault_trip(err, "refill copy") || !ck(cudaMemcpyAsync(d.plane[p*3+q],
                            m.h_refill + offset, b, cudaMemcpyHostToDevice, m.refill), "refill copy", err)) {
                        m.failed = true; return false;
                    }
                    offset += b;
                }
            }
        }
        if (m.fault_trip(err, "refill event") || !ck(cudaEventRecord(m.refill_done, m.refill), "refill event", err)) {
            m.failed = true; return false;
        }
    } catch (const std::exception& e) {
        err = std::string("exl3 peer: refill: ") + e.what();
        if (!m.replacing.empty()) m.failed = true;
        else for (const auto& r : replacements) m.where.erase({r.in.layer, r.in.expert});
        return false;
    }
    return true;
}

bool Exl3Peer::apply_refill(bool wait, std::vector<Replacement>& landed, std::string& err) {
    landed.clear();
    if (!valid() || pending()) { err = "exl3 peer: refill publication needs a valid idle compute tier"; return false; }
    auto& m = *impl_;
    if (!refilling()) return true;
    int caller = -1; (void) cudaGetDevice(&caller);
    if (!switch_to(m.device)) { err = "exl3 peer: refill publication device switch failed"; return false; }
    struct Restore { int prev; ~Restore() { (void) cudaSetDevice(prev); } } restore{caller};
    const auto result = wait ? cudaEventSynchronize(m.refill_done) : cudaEventQuery(m.refill_done);
    if (!wait && result == cudaErrorNotReady) return true;
    if (!ck(result, "refill completion", err)) { m.failed = true; return false; }
    float elapsed = 0;
    if (!ck(cudaEventElapsedTime(&elapsed, m.refill_start, m.refill_done), "refill timing", err)) {
        m.failed = true; return false;
    }
    landed = m.replacing; // allocate before publishing; the caller normally reserves this capacity
    for (size_t i = 0; i < m.replacing.size(); ++i) {
        const auto& r = m.replacing[i];
        const size_t slot = m.refill_slots[i];
        auto& d = m.dev[slot];
        d.g = r.in.g; d.u = r.in.u; d.d = r.in.d;
        Exl3Mat* mats[] = {&d.g, &d.u, &d.d};
        for (int p = 0; p < 3; ++p) {
            mats[p]->trellis = (const uint16_t*) d.plane[p*3];
            mats[p]->suh = (const uint16_t*) d.plane[p*3+1];
            mats[p]->svh = (const uint16_t*) d.plane[p*3+2];
        }
        d.expert = r.in.expert; d.layer = r.in.layer;
        std::array<size_t, 9> sizes{}; (void) layout(r.in, m.n_embd, m.hidden, sizes);
        d.bytes = 0; for (size_t b : sizes) d.bytes += b;
        m.where.at({r.in.layer, r.in.expert}) = slot;
    }
    m.replacing.clear();
    m.refill_slots.clear();
    m.refill_ms += elapsed;
    return true;
}

bool Exl3Peer::open(int device, int n_embd, const std::vector<Entry>& subset,
                    int max_tokens, int max_k, std::string& err) {
    close();
    Impl& m = *impl_;
    m.fault_arm("STRATA_EXL3_PEER_FAULT");
    // ---- validation BEFORE any allocation, any device switch and any host plane read
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || device < 0 || device >= ndev) {
        err = "exl3 peer: device out of range";
        return false;
    }
    if (n_embd <= 0 || n_embd % 16 != 0) { err = "exl3 peer: n_embd must be a positive multiple of 16"; return false; }
    if (subset.empty()) { err = "exl3 peer: empty subset"; return false; }
    if (max_tokens < 1 || max_k < 1) { err = "exl3 peer: max_tokens/max_k must be >= 1"; return false; }
    const int64_t routes_cap = (int64_t) max_tokens * (int64_t) max_k;
    if (routes_cap > 65535) { err = "exl3 peer: max_tokens x max_k exceeds 65535 (the indexed kernel's limit)"; return false; }
    const int k_max = INT32_MAX / 16;   // dims must fit *16 in int before any such multiply
    int hidden = 0;
    std::vector<std::array<size_t, 9>> plane_bytes;
    size_t plane_total = 0;
    try {
    plane_bytes.resize(subset.size());
    for (size_t i = 0; i < subset.size(); ++i) {
        const Entry& e = subset[i];
        const Exl3Mat mats[3] = {e.g, e.u, e.d};
        for (const Exl3Mat& mm : mats) {
            if (mm.ki < 1 || mm.ki > k_max || mm.nj < 1 || mm.nj > k_max) {
                err = "exl3 peer: entry " + std::to_string(i) + ": ki/nj must be in [1, INT32_MAX/16]";
                return false;
            }
            if (mm.ki % 8 || mm.nj % 8) {   // the kernels' H128 alignment requirement
                err = "exl3 peer: entry " + std::to_string(i) + ": ki/nj must be multiples of 8";
                return false;
            }
            if (mm.bits < 1 || mm.bits > 8 || mm.cb < 0 || mm.cb > 2) {
                err = "exl3 peer: entry " + std::to_string(i) + ": bits must be 1..8 and cb 0..2";
                return false;
            }
            if (!mm.trellis || !mm.suh || !mm.svh) {
                err = "exl3 peer: entry " + std::to_string(i) + ": null plane pointer";
                return false;
            }
        }
        const bool rel = e.layer >= 0 && e.expert >= 0 &&
                         e.g.ki == e.u.ki && e.g.nj == e.u.nj &&
                         e.d.ki == e.g.nj && e.d.nj == e.g.ki &&
                         e.g.ki * 16 == n_embd;   // *16 now safe (ki <= INT32_MAX/16)
        if (!rel) {
            err = "exl3 peer: entry " + std::to_string(i) + ": mat relations/n_embd mismatch";
            return false;
        }
        const int h = e.g.nj * 16;
        if (i == 0) hidden = h;
        else if (h != hidden) { err = "exl3 peer: mixed hidden widths are not supported"; return false; }
        for (size_t j = 0; j < i; ++j)
            if (subset[j].layer == e.layer && subset[j].expert == e.expert) {
                err = "exl3 peer: entry " + std::to_string(i) + ": duplicate (layer, expert)";
                return false;
            }
        // ---- checked size math for EVERY entry before the first allocation or source read
        const size_t kPlaneCap = (size_t) 1 << 32;
        for (int p = 0; p < 3; ++p) {
            size_t tiles = 0, tr = 0;
            if (!mul_ok((size_t) mats[p].ki, (size_t) mats[p].nj, &tiles) ||
                !mul_ok(tiles, (size_t) (256 * mats[p].bits / 16) * 2, &tr)) {
                err = "exl3 peer: plane size overflow";
                return false;
            }
            plane_bytes[i][(size_t) p * 3 + 0] = tr;
            plane_bytes[i][(size_t) p * 3 + 1] = (size_t) mats[p].ki * 16 * 2;
            plane_bytes[i][(size_t) p * 3 + 2] = (size_t) mats[p].nj * 16 * 2;
            for (int q = 0; q < 3; ++q) {
                const size_t b = plane_bytes[i][(size_t) p * 3 + q];
                if (b == 0 || b > kPlaneCap || !add_ok(plane_total, b, &plane_total)) {
                    err = "exl3 peer: plane size cap/overflow";
                    return false;
                }
            }
        }
    }
    // ---- worst-case workspace/scratch accounting, checked (the workspace reserves its own):
    // descriptors 3 x r x Exl3Mat, weights, g/u/h/d rows, xh/z, the INDEXED PARTIALS
    // (r x max(ceil(n_in/256) x n_ff, ceil(n_ff/256) x n_out) floats) and pinv
    {
        size_t worst = 0;
        const size_t r = (size_t) routes_cap;
        const size_t wide = (size_t) std::max(n_embd, hidden);
        const size_t gate_parts = ((size_t) n_embd + 255) / 256, down_parts = ((size_t) hidden + 255) / 256;
        const size_t partial = std::max(gate_parts * (size_t) hidden, down_parts * (size_t) n_embd);
        const size_t terms[7] = {r * sizeof(Exl3Mat) * 3, r * sizeof(float), r * (size_t) hidden * sizeof(float) * 3,
                                 r * (size_t) n_embd * sizeof(float), wide * r * sizeof(float) * 2,
                                 std::max((size_t) hidden, (size_t) n_embd) * r * sizeof(float),
                                 r * partial * sizeof(float)};
        for (size_t t : terms)
            if (!add_ok(worst, t, &worst)) { err = "exl3 peer: workspace size overflow"; return false; }
        if (!add_ok(worst, 256 * sizeof(int), &worst) || worst > ((size_t) 1 << 34)) {
            err = "exl3 peer: workspace size cap/overflow";
            return false;
        }
    }
    } catch (const std::exception& e) {
        err = std::string("exl3 peer: open (validation phase): ") + e.what();
        return false;
    }
    // ---- resource phase: one ownership path.  Any failure - returned error, kernel/API
    // exception, or std::bad_alloc - unwinds through close() (free_all, exactly once).
    m.device = device;
    m.n_embd = n_embd;
    m.hidden = hidden;
    m.max_tokens = max_tokens;
    m.max_k = max_k;
    m.max_routes = routes_cap;
    m.profile_finish = std::getenv("STRATA_EXL3_PROF") != nullptr || std::getenv("STRATA_DECODE_TIMING") != nullptr;
    m.vector_convert = strata::core::exl3_flag("STRATA_EXL3_PEER_CONVERT", nullptr, true);
    int caller = -1;
    (void) cudaGetDevice(&caller);
    if (!switch_to(device)) {
        err = "exl3 peer: cannot switch to the peer device";
        m.device = -1;
        return false;
    }
    struct Restore {
        int prev;
        ~Restore() { (void) cudaSetDevice(prev); }
    } restore{caller};
    try {
        auto alloc = [&](void** p, size_t bytes, bool host, const char* what) {
            if (m.fault_trip(err, what)) return false;
            return ck(host ? cudaHostAlloc(p, bytes, cudaHostAllocPortable) : cudaMalloc(p, bytes), what, err);
        };
        if (m.fault_trip(err, "stream create") ||
            !ck(cudaStreamCreateWithFlags(&m.stream, cudaStreamNonBlocking), "stream", err) ||
            !alloc((void**) &m.d_x32, (size_t) max_tokens * n_embd * sizeof(float), false, "x32") ||
            !alloc((void**) &m.d_x16, (size_t) max_tokens * n_embd * sizeof(uint16_t), false, "x16") ||
            !alloc((void**) &m.d_rows_idx, (size_t) routes_cap * sizeof(int), false, "rows idx") ||
            !alloc((void**) &m.d_rows16, (size_t) routes_cap * n_embd * sizeof(uint16_t), false, "rows") ||
            !alloc((void**) &m.h_x32, (size_t) max_tokens * n_embd * sizeof(float), true, "pinned x") ||
            !alloc((void**) &m.h_rows_idx, (size_t) routes_cap * sizeof(int), true, "pinned idx") ||
            !alloc((void**) &m.h_rows16, (size_t) routes_cap * n_embd * sizeof(uint16_t), true, "pinned rows") ||
            !alloc((void**) &m.h_g, (size_t) routes_cap * sizeof(Exl3Mat), true, "pinned g") ||
            !alloc((void**) &m.h_u, (size_t) routes_cap * sizeof(Exl3Mat), true, "pinned u") ||
            !alloc((void**) &m.h_d, (size_t) routes_cap * sizeof(Exl3Mat), true, "pinned d")) {
            close();
            return false;
        }
        m.row_of.assign((size_t) routes_cap, 0);   // inside the try; released by close()
        m.ws = std::make_unique<Exl3MoeWorkspace>();   // binds to the peer device (we are on it)
        m.ws->reserve((int) routes_cap, n_embd, hidden, n_embd, true);
        const auto enabled = [](const char* name) {
            const char* v = std::getenv(name);
            return v && v[0] && v[0] != '0';
        };
        if (strata::core::exl3_flag("STRATA_EXL3_PEER_SPARSE", nullptr, true) || enabled("STRATA_EXL3_RESERVE_FUSED"))
            m.ws->reserve_runtime_scratch((int) routes_cap, n_embd, hidden, n_embd);
        const char* mr = std::getenv("STRATA_EXL3_PEER_MULTIROW");
        m.multirow = mr ? std::max(0, std::min(4, std::atoi(mr))) : 0;
        if (m.multirow == 1) m.multirow = 2;
        if (m.multirow && (!exl3_fused_multirow_supported() ||
            (size_t) std::max(n_embd, hidden) * (m.multirow <= 2 ? 2 : 4) * sizeof(float) > 40 * 1024)) {
            std::fprintf(stderr, "exl3 peer multirow: unsupported mode/geometry; using indexed FFNs\n");
            m.multirow = 0;
        }
        if (m.multirow) {
            if (!alloc((void**) &m.h_group_first, (size_t) routes_cap * sizeof(int), true, "pinned group first") ||
                !alloc((void**) &m.h_group_rows, (size_t) routes_cap * sizeof(int), true, "pinned group rows")) {
                close(); return false;
            }
            m.ws->reserve_small_m((int) routes_cap, (int) routes_cap, n_embd, hidden, n_embd);
            m.ws->reserve_runtime_scratch((int) routes_cap, n_embd, hidden, n_embd);
            // default on: data transport only (one pinned plan slab per layer); the RDNA fused
            // kernels stay gated by PEER_MULTIROW above
            m.pack_plan = strata::core::exl3_flag("STRATA_EXL3_PEER_PACK_PLAN", nullptr, true);
            if (m.pack_plan) {
                m.plan_capacity = PackedPlan(max_tokens, n_embd, (int)routes_cap, (int)routes_cap).bytes;
                if (!alloc((void**)&m.h_plan, m.plan_capacity, true, "pinned packed plan") ||
                    !alloc((void**)&m.d_plan, m.plan_capacity, false, "device packed plan")) {
                    close(); return false;
                }
                std::memset(m.h_plan, 0, m.plan_capacity);
            }
        }
        for (size_t i = 0; i < subset.size(); ++i) {
            const Entry& e = subset[i];
            m.where[{e.layer, e.expert}] = m.dev.size();
            m.dev.emplace_back();   // owned from here; a failure below unwinds through close()
            Impl::DevEntry& d = m.dev.back();
            d.layer = e.layer;
            d.expert = e.expert;
            const Exl3Mat host[3] = {e.g, e.u, e.d};
            Exl3Mat* devm[3] = {&d.g, &d.u, &d.d};
            size_t total = 0;
            for (int p = 0; p < 3; ++p) {
                const void* src[3] = {host[p].trellis, host[p].suh, host[p].svh};
                for (int q = 0; q < 3; ++q) {
                    const size_t b = plane_bytes[i][(size_t) p * 3 + q];
                    d.capacity[(size_t) p * 3 + q] = b;
                    if (m.fault_trip(err, "plane alloc/copy") ||
                        !ck(cudaMalloc(&d.plane[p * 3 + q], b), "plane alloc", err) ||
                        m.fault_trip(err, "plane copy") ||
                        !ck(cudaMemcpy(d.plane[p * 3 + q], src[q], b, cudaMemcpyHostToDevice), "plane copy", err)) {
                        close();   // free_all releases every plane published so far, exactly once
                        return false;
                    }
                    total += b;
                }
                devm[p]->trellis = (const uint16_t*) d.plane[p * 3 + 0];
                devm[p]->suh = (const uint16_t*) d.plane[p * 3 + 1];
                devm[p]->svh = (const uint16_t*) d.plane[p * 3 + 2];
                devm[p]->ki = host[p].ki;
                devm[p]->nj = host[p].nj;
                devm[p]->bits = host[p].bits;
                devm[p]->cb = host[p].cb;
            }
            d.bytes = total;
        }
    } catch (const std::exception& e) {
        err = std::string("exl3 peer: open: ") + e.what();
        close();
        return false;
    }
    m.failed = false;
    m.pending = false;
    return true;
}

void Exl3Peer::close() {
    if (!impl_) return;
    Impl& m = *impl_;
    if (m.packed_calls)
        std::fprintf(stderr, "exl3 peer packed plan: %lld calls, %lld H2D bytes; one upload and no output D2D per grouped call\n",
                     (long long)m.packed_calls, (long long)m.packed_bytes);
    m.packed_calls = m.packed_bytes = 0;
    if (m.multirow && m.grouped_calls)
        std::fprintf(stderr, "exl3 peer multirow: width %d, %lld calls, %lld routes / %lld groups\n",
                     m.multirow, (long long) m.grouped_calls, (long long) m.grouped_rows, (long long) m.grouped_groups);
    m.multirow = 0;
    m.grouped_calls = m.grouped_rows = m.grouped_groups = 0;
    if (m.profile_finish && m.finishes > 0)
        std::fprintf(stderr, "exl3 peer finish split: %lld calls, wait %.3f ms/call, host fp16->fp32 scatter %.3f ms/call\n",
                     (long long) m.finishes, m.wait_ms / m.finishes, m.scatter_ms / m.finishes);
    m.finishes = 0;
    m.wait_ms = m.scatter_ms = 0;
    if (m.device >= 0) {
        int caller = -1;
        (void) cudaGetDevice(&caller);
        if (switch_to(m.device)) {
            struct Restore {
                int prev;
                ~Restore() { (void) cudaSetDevice(prev); }
            } restore{caller};
            if (m.stream) {
                cudaError_t e;
                while ((e = cudaStreamQuery(m.stream)) == cudaErrorNotReady) { }
                (void) e;
            }
            if (m.refill) (void) cudaStreamSynchronize(m.refill);
            if (m.pf_layers > 0) {
                const std::string batches = m.pf_batch
                    ? ", " + std::to_string(m.pf_batches) + " batched launch chains" : std::string();
                std::fprintf(stderr, "exl3 peer prefill: %lld layers, %lld rows / %lld groups computed on the peer%s\n",
                             (long long) m.pf_layers, (long long) m.pf_rows_total,
                             (long long) m.pf_groups_total, batches.c_str());
            }
            m.free_all();
            m.pf_free();
        } else {
            m.drop_without_free();   // freeing across contexts is unsafe; the process exit reclaims
        }
    }
    m.device = -1;
    m.failed = m.pending = false;
    m.pending_rows = 0;
    m.launches = m.rows = m.enqueues = 0;
}

bool Exl3Peer::launch(int64_t layer, const float* x, const int32_t* ids, int tokens, int k,
                      const int32_t* kind, std::string& err, Exl3PeerOutput* device_output) {
    if (!valid()) { err = failed() ? "exl3 peer: tier failed earlier" : "exl3 peer: not open"; return false; }
    Impl& m = *impl_;
    m.fault_arm("STRATA_EXL3_PEER_FAULT_LAUNCH");   // launch-only seam (open has its own)
    if (m.pending || m.pf_pending) { err = "exl3 peer: a launch is already pending"; return false; }
    // ---- full validation before ANY device switch or enqueue
    if (tokens < 0 || tokens > m.max_tokens || k < 0 || k > m.max_k) {
        err = "exl3 peer: tokens/k out of the open() caps";
        return false;
    }
    if (tokens > 0 && (x == nullptr || ids == nullptr || kind == nullptr)) {
        err = "exl3 peer: null x/ids/kind";
        return false;
    }
    const int64_t routes = (int64_t) tokens * (int64_t) k;
    if (routes > m.max_routes) { err = "exl3 peer: tokens x k overflows the open() cap"; return false; }
    int64_t m_rows = 0;
    for (int64_t i = 0; i < routes; ++i) {
        if (kind[i] != 2) continue;
        const int32_t e = ids[i];
        if (e < 0 || !has(layer, e)) {
            err = "exl3 peer: route " + std::to_string(i) + " (expert " + std::to_string(e) +
                  ") is not resident on this tier - refused, nothing enqueued";
            return false;
        }
        ++m_rows;
    }
    if (m_rows == 0) {   // empty launch: explicit no-op success
        ++m.launches;
        return true;
    }
    m.pending_rows = (int) m_rows;
    m.deferred = device_output != nullptr;
    std::memcpy(m.h_x32, x, (size_t) tokens * m.n_embd * sizeof(float));   // copied before return
    int64_t j = 0;
    for (int64_t i = 0; i < routes; ++i) {
        if (kind[i] != 2) continue;
        m.row_of[(size_t) j++] = (int) i;
    }
    bool grouped = m.multirow > 0;
    if (grouped) {
        for (int64_t q = 0; q < m_rows; ++q) {
            const auto& e = m.dev[m.where.at({layer, ids[m.row_of[(size_t) q]]})];
            if (e.g.cb != 2 || e.u.cb != 2 || e.d.cb != 2) { grouped = false; break; }
        }
    }
    if (grouped)
        std::sort(m.row_of.begin(), m.row_of.begin() + m_rows, [&](int a, int b) {
            return ids[a] != ids[b] ? ids[a] < ids[b] : a < b;
        });
    int groups = 0;
    for (int64_t q = 0; q < m_rows; ++q) {
        const int i = m.row_of[(size_t) q];
        const size_t at = m.where.at({layer, ids[i]});
        m.h_g[q] = m.dev[at].g;
        m.h_u[q] = m.dev[at].u;
        m.h_d[q] = m.dev[at].d;
        m.h_rows_idx[q] = (int32_t) (i / k);
        if (grouped) {
            if (q == 0 || ids[i] != ids[m.row_of[(size_t) q - 1]] || m.h_group_rows[groups - 1] == m.multirow) {
                m.h_group_first[groups] = (int) q;
                m.h_group_rows[groups++] = 0;
            }
            ++m.h_group_rows[groups - 1];
        }
    }
    grouped = grouped && groups < m_rows;
    if (grouped) {
        for (int g = 0; g < groups; ++g) {
            const int first = m.h_group_first[g];
            m.h_g[g] = m.h_g[first]; m.h_u[g] = m.h_u[first]; m.h_d[g] = m.h_d[first];
        }
        ++m.grouped_calls; m.grouped_rows += m_rows; m.grouped_groups += groups;
    }
    // ---- enqueue only (the seam the enqueues() counter marks): no sync, no alloc, no file IO
    int caller = -1;
    (void) cudaGetDevice(&caller);
    if (!switch_to(m.device)) {   // a failed switch must NOT enqueue on the caller's device
        err = "exl3 peer: cannot switch to the tier device";
        return false;
    }
    struct Restore {
        int prev;
        ~Restore() { (void) cudaSetDevice(prev); }
    } restore{caller};
    bool enq = false;
    try {
        ++m.enqueues;
        const uint16_t* result = m.d_rows16;
        if (m.pack_plan && grouped) {
            // A pending launch prevents host-buffer reuse; finish/close drains
            // this stream before another callback or residency replacement.
            const PackedPlan p(tokens, m.n_embd, (int)m_rows, groups);
            if (p.bytes > m.plan_capacity) throw std::runtime_error("packed plan exceeds reserved capacity");
            auto put = [&](size_t offset, const void* src, size_t bytes) { std::memcpy(m.h_plan + offset, src, bytes); };
            put(p.x, m.h_x32, (size_t)tokens * m.n_embd * sizeof(float));
            put(p.indices, m.h_rows_idx, (size_t)m_rows * sizeof(int));
            put(p.gate, m.h_g, (size_t)groups * sizeof(Exl3Mat));
            put(p.up, m.h_u, (size_t)groups * sizeof(Exl3Mat));
            put(p.down, m.h_d, (size_t)groups * sizeof(Exl3Mat));
            put(p.first, m.h_group_first, (size_t)groups * sizeof(int));
            put(p.rows, m.h_group_rows, (size_t)groups * sizeof(int));
            enq = !m.fault_trip(err, "launch packed plan H2D") &&
                ck(cudaMemcpyAsync(m.d_plan, m.h_plan, p.bytes, cudaMemcpyHostToDevice, m.stream), "packed plan H2D", err);
            if (enq) {
                f32_to_f16_bulk((const float*)(m.d_plan + p.x), m.d_x16,
                               (int64_t)tokens * m.n_embd, m.stream);
                enq = ck(cudaGetLastError(), "packed activation conversion", err);
            }
            if (enq) {
                const Exl3MoeDevicePlan plan{(const Exl3Mat*)(m.d_plan + p.gate),
                    (const Exl3Mat*)(m.d_plan + p.up), (const Exl3Mat*)(m.d_plan + p.down),
                    (const int*)(m.d_plan + p.first), (const int*)(m.d_plan + p.rows)};
                result = exl3_moe_rows_grouped_device_plan(m.h_g, m.h_u, m.h_d,
                    m.h_group_first, m.h_group_rows, groups, (int)m_rows, m.d_x16,
                    (const int*)(m.d_plan + p.indices), plan, m.stream, *m.ws, true);
                enq = ck(cudaGetLastError(), "packed grouped FFN", err);
                ++m.packed_calls; m.packed_bytes += (int64_t)p.bytes;
            }
        } else {
            enq =
            !m.fault_trip(err, "launch x H2D") &&
            ck(cudaMemcpyAsync(m.d_x32, m.h_x32, (size_t) tokens * m.n_embd * sizeof(float),
                               cudaMemcpyHostToDevice, m.stream), "x H2D", err) &&
            [&] { f32_to_f16_bulk(m.d_x32, m.d_x16, (int64_t) tokens * m.n_embd, m.stream);
                  return cudaGetLastError() == cudaSuccess; }() &&
            !m.fault_trip(err, "launch idx H2D") &&
            ck(cudaMemcpyAsync(m.d_rows_idx, m.h_rows_idx, (size_t) m_rows * sizeof(int),
                               cudaMemcpyHostToDevice, m.stream), "idx H2D", err) &&
            [&] {
                if (grouped)
                    exl3_moe_rows_grouped(m.h_g, m.h_u, m.h_d, m.h_group_first, m.h_group_rows,
                        groups, (int) m_rows, m.d_x16, m.d_rows_idx, m.d_rows16, m.stream, *m.ws, true);
                else
                    exl3_moe_rows_indexed(m.h_g, m.h_u, m.h_d, (int) m_rows,
                                          m.d_x16, m.d_rows_idx, m.d_rows16, m.stream, *m.ws);
                return cudaGetLastError() == cudaSuccess;
            }();
        }
        enq = enq && !m.fault_trip(err, "launch rows kernel") &&
            !m.fault_trip(err, "launch rows D2H") &&
            (device_output ? device_output->enqueue(result,m.row_of.data(),(int)m_rows,m.stream,err) :
                ck(cudaMemcpyAsync(m.h_rows16, result, (size_t) m_rows * m.n_embd * sizeof(uint16_t),
                                   cudaMemcpyDeviceToHost, m.stream), "rows D2H", err));
    } catch (const std::exception& e) {
        err = std::string("exl3 peer: enqueue: ") + e.what();
        enq = false;
    }
    if (!enq) {
        m.failed = true;       // sticky until close(); nothing valid is ever advertised
        m.pending_rows = 0;
        return false;
    }
    m.pending = true;
    ++m.launches;
    m.rows += m_rows;
    return true;
}

bool Exl3Peer::finish(float* out, std::string& err) {
    if (!valid()) { err = failed() ? "exl3 peer: tier failed earlier" : "exl3 peer: not open"; return false; }
    Impl& m = *impl_;
    if (out == nullptr) {   // explicit semantics: nullptr out is ALWAYS an error, even with no
        err = "exl3 peer: finish(nullptr out)";   // pending launch (it can never complete rows)
        return false;
    }
    if (!m.pending) return true;   // no-op when nothing (or an empty launch) is pending
    if (m.deferred) { err="exl3 peer: deferred output must be consumed by the primary graph"; return false; }
    int caller = -1;
    (void) cudaGetDevice(&caller);
    if (!switch_to(m.device)) {
        err = "exl3 peer: cannot switch to the tier device";
        return false;
    }
    struct Restore {
        int prev;
        ~Restore() { (void) cudaSetDevice(prev); }
    } restore{caller};
    using Clock = std::chrono::steady_clock;
    const auto t0 = m.profile_finish ? Clock::now() : Clock::time_point{};
    cudaError_t e;
    while ((e = cudaStreamQuery(m.stream)) == cudaErrorNotReady) { }
    if (!ck(e, "finish", err)) { m.failed = true; m.pending = false; return false; }
    const auto t1 = m.profile_finish ? Clock::now() : Clock::time_point{};
    for (int64_t j = 0; j < m.pending_rows; ++j) {   // host scatter; destinations unique
        float* dst = out + (size_t) m.row_of[(size_t) j] * (size_t) m.n_embd;
        const uint16_t* src = m.h_rows16 + (size_t) j * (size_t) m.n_embd;
        if (m.vector_convert) cpu::f16_to_f32_exact(src, dst, (size_t) m.n_embd);
        else for (int i = 0; i < m.n_embd; ++i) dst[i] = f32_from_f16(src[i]);
    }
    if (m.profile_finish) {
        m.wait_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        m.scatter_ms += std::chrono::duration<double, std::milli>(Clock::now() - t1).count();
        ++m.finishes;
    }
    m.pending = false;
    m.pending_rows = 0;
    return true;
}

bool Exl3Peer::finish_deferred(std::string& err) {
    if(!valid()) {err="exl3 peer: deferred completion on invalid tier";return false;}
    auto& m=*impl_;
    if(!m.pending) return true;
    if(!m.deferred) {err="exl3 peer: launch did not select device output";return false;}
    // Enter the peer driver once to submit/progress queued work. The host no
    // longer spin-queries this stream; deferring must not also defer its kick.
    int previous=0;(void)cudaGetDevice(&previous);
    if(!switch_to(m.device)) {err="exl3 peer: deferred device switch failed";return false;}
    const auto status=cudaStreamQuery(m.stream);
    (void)cudaSetDevice(previous);
    if(status!=cudaSuccess && status!=cudaErrorNotReady) {
        err=cudaGetErrorString(status);m.failed=true;return false;
    }
    m.pending=false; m.pending_rows=0; m.deferred=false;
    return true;
}

// ---- STRATA_EXL3_PEER_PREFILL: the prompt chunks' peer-owned experts (see exl3_peer.hpp) ----

bool Exl3Peer::enable_prefill(int64_t max_tokens, int k, int64_t n_expert, int64_t max_rows, std::string& err) {
    if (!valid()) { err = "exl3 peer: prefill needs an open tier"; return false; }
    Impl& m = *impl_;
    if (m.pf_on) return true;
    if (k != 10) { err = "exl3 peer: prefill transport assumes top-10 routing"; return false; }
    if (max_tokens <= 0 || max_tokens > 1 << 20 || n_expert <= 0 || max_rows <= 0 ||
        max_tokens > INT64_MAX / k || max_tokens > INT64_MAX / (int64_t) m.n_embd) {
        err = "exl3 peer: prefill caps out of range"; return false;
    }
    if (m.dev.empty()) { err = "exl3 peer: prefill needs resident entries"; return false; }
    const int64_t routes = max_tokens * k;
    const int64_t n = m.n_embd;
    // One pinned/device slab per layer: [ires | pair | wk | gate | up | down | first | rowsn].
    // The per-group tables are sized for the degenerate one-expert-per-route case (n_expert
    // groups max); the per-route arrays for the caller's row cap.
    size_t off = 0;
    const auto take = [&](size_t bytes) { const size_t at = off; off = (off + bytes + 255) & ~(size_t)255; return at; };
    const size_t o_ires = take((size_t) max_rows * sizeof(int32_t));
    const size_t o_pair = take((size_t) max_rows * sizeof(int32_t));
    const size_t o_wk   = take((size_t) routes * sizeof(float));
    const size_t o_dest = take((size_t) max_rows * sizeof(int32_t));
    const size_t o_grp  = take((size_t) n_expert * sizeof(Exl3Group));
    const size_t slab = off;
    m.pf_off[0] = o_ires; m.pf_off[1] = o_pair; m.pf_off[2] = o_wk; m.pf_off[3] = o_dest; m.pf_off[4] = o_grp;
    // The workspace's group buffers (wmma_x, group_y, the f16 slabs) for a whole-chunk group,
    // grown once here: no allocation under queued work.  `ws` binds to the device current at
    // each call, so this must run under the peer-device guard below.
    const int in = m.dev[0].g.ki * 16, ff = m.dev[0].g.nj * 16;
    const char* pb_env = std::getenv("STRATA_EXL3_PEER_BATCH");
    m.pf_batch = pb_env && pb_env[0] && pb_env[0] != '0';

    int caller = -1;
    (void) cudaGetDevice(&caller);
    if (!switch_to(m.device)) { err = "exl3 peer: cannot switch to the tier device"; return false; }
    struct Restore { int prev; ~Restore() { (void) cudaSetDevice(prev); } } restore{caller};
    m.ws->reserve_group_scratch((int) max_tokens, in, ff, in, true);
    if (m.pf_batch) m.ws->reserve_expert_batch(8, (int) std::min<int64_t>(4096, max_rows), in, ff, in);
    size_t free_b = 0, total_b = 0;
    const bool have_free = cudaMemGetInfo(&free_b, &total_b) == cudaSuccess;
    const size_t need = (size_t) max_tokens * n * 2 +          // x16
                        (size_t) max_rows * n * 4 +            // rows
                        (size_t) max_tokens * n * 4 +          // sum
                        (size_t) routes * 4 * 3 + slab * 2 + ((size_t) 1 << 20);
    if (have_free && free_b < need) {
        err = "exl3 peer: prefill buffers need " + std::to_string(need >> 20) + " MiB, the peer card has " +
              std::to_string(free_b >> 20) + " MiB free";
        return false;
    }
    auto alloc_d = [&](void** p, size_t bytes, const char* what) {
        if (*p) return true;
        if (cudaMalloc(p, bytes) != cudaSuccess) { err = std::string("exl3 peer: prefill ") + what; return false; }
        return true;
    };
    if (!alloc_d((void**) &m.pf_d_x16, (size_t) max_tokens * n * 2, "x16") ||
        !alloc_d((void**) &m.pf_d_rows, (size_t) max_rows * n * 4, "rows") ||
        !alloc_d((void**) &m.pf_d_sum, (size_t) max_tokens * n * 4, "sum") ||
        !alloc_d((void**) &m.pf_d_slab, slab, "plan slab") ||
        cudaHostAlloc((void**) &m.pf_h_slab, slab, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &m.pf_host_sum, (size_t) max_tokens * n * 4,
                      cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaEventCreateWithFlags(&m.pf_done, cudaEventDisableTiming) != cudaSuccess) {
        m.pf_free();
        if (err.empty()) err = "exl3 peer: prefill pinned/event allocation";
        return false;
    }
    std::memset(m.pf_h_slab, 0, slab);
    {   // the destination table is a static iota: route r's rows slab position is r itself
        auto* dest = (int32_t*) ((unsigned char*) m.pf_h_slab + o_dest);
        for (int64_t i = 0; i < max_rows; ++i) dest[i] = (int32_t) i;
    }
    m.pf_d_ires = (int32_t*)  (m.pf_d_slab + o_ires);
    m.pf_d_pair = (int32_t*)  (m.pf_d_slab + o_pair);
    m.pf_d_wk   = (float*)    (m.pf_d_slab + o_wk);
    m.pf_d_dest = (int32_t*)  (m.pf_d_slab + o_dest);
    m.pf_d_grp  = (Exl3Group*) (m.pf_d_slab + o_grp);
    m.pf_k = (int) k;
    m.pf_n_expert = n_expert;
    m.pf_tokens_cap = max_tokens;
    m.pf_rows_cap = max_rows;
    m.pf_slab_bytes = slab;
    m.pf_on = true;
    return true;
}

bool Exl3Peer::prefill_enabled() const { return impl_ && impl_->pf_on; }
int64_t Exl3Peer::prefill_rows_cap() const { return impl_ ? impl_->pf_rows_cap : 0; }
const float* Exl3Peer::prefill_sums_host() const { return impl_ ? impl_->pf_host_sum : nullptr; }
int64_t Exl3Peer::prefill_last_rows() const { return impl_ ? impl_->pf_last_rows : 0; }
int64_t Exl3Peer::prefill_layers() const { return impl_ ? impl_->pf_layers : 0; }
int64_t Exl3Peer::prefill_rows_total() const { return impl_ ? impl_->pf_rows_total : 0; }

bool Exl3Peer::launch_prefill(int64_t layer, const uint16_t* x16, int64_t tokens,
                              const int32_t* ids, const uint8_t* peer_mask, const float* weights,
                              std::string& err) {
    if (!valid()) { err = failed() ? "exl3 peer: tier failed earlier" : "exl3 peer: not open"; return false; }
    Impl& m = *impl_;
    if (!m.pf_on) { err = "exl3 peer: prefill not enabled"; return false; }
    if (m.pending || m.pf_pending) { err = "exl3 peer: a launch is already pending"; return false; }
    if (tokens <= 0 || tokens > m.pf_tokens_cap) { err = "exl3 peer: prefill tokens out of the enabled cap"; return false; }
    if (!x16 || !ids || !peer_mask || !weights) { err = "exl3 peer: prefill null x16/ids/mask/weights"; return false; }
    const int64_t routes = tokens * m.pf_k;

    // ---- host plan (before any enqueue): the masked routes, expert-major, per route validated
    Exl3RoutePlan plan;
    plan.build(ids, (int) tokens, m.pf_k, (int) m.pf_n_expert);
    unsigned char* h = (unsigned char*) m.pf_h_slab;
    std::memcpy((float*) (h + m.pf_off[2]), weights, (size_t) routes * sizeof(float));
    auto* out_ires = (int32_t*) (h + m.pf_off[0]);
    auto* out_pair = (int32_t*) (h + m.pf_off[1]);
    auto* out_grp  = (Exl3Group*) (h + m.pf_off[4]);
    m.pf_first.clear();
    m.pf_rowsn.clear();
    m.pf_batch_end.clear();
    int64_t total = 0;
    int groups = 0;
    for (size_t g = 0; g < plan.experts.size(); ++g) {
        const int32_t e = plan.experts[g];
        int taken = 0;
        for (int r = plan.offsets[g]; r < plan.offsets[g + 1]; ++r) {
            const int dst = plan.destinations[(size_t) r];
            if (!peer_mask[dst]) continue;
            if (!has(layer, e)) {
                err = "exl3 peer: prefill mask routes expert " + std::to_string(e) +
                      " (layer " + std::to_string(layer) + ") not resident on this tier";
                return false;
            }
            out_ires[total + taken] = dst / m.pf_k;
            out_pair[total + taken] = dst;
            ++taken;
        }
        if (taken == 0) continue;
        const auto& d = m.dev[m.where.at({layer, e})];
        out_grp[groups] = Exl3Group{d.g, d.u, d.d, 0, taken};
        m.pf_first.push_back((int) total);
        m.pf_rowsn.push_back(taken);
        ++groups;
        total += taken;
    }
    if (total == 0) return true;   // nothing peer-owned this layer: explicit no-op success
    if (total > m.pf_rows_cap) { err = "exl3 peer: prefill rows over the enabled cap"; return false; }
    // STRATA_EXL3_PEER_BATCH: consecutive groups with >= 16 rows each, one row-tile regime and
    // <= 4096 rows total run as one exl3_moe_expert_batch launch chain (the group descriptors'
    // `first` becomes the within-batch prefix, as that entry validates).  Anything else -
    // small groups, regime changes, lone tails - stays a standalone group_wmma call.
    m.pf_batch_end.assign((size_t) groups, -1);
    if (m.pf_batch) {
        int i = 0;
        while (i < groups) {
            if (m.pf_rowsn[(size_t) i] < 16) { ++i; continue; }
            int j = i, rows = 0;
            while (j < groups && j - i < 8 && m.pf_rowsn[(size_t) j] >= 16 &&
                   (m.pf_rowsn[(size_t) j] >= 24) == (m.pf_rowsn[(size_t) i] >= 24) &&
                   rows + m.pf_rowsn[(size_t) j] <= 4096) {
                rows += m.pf_rowsn[(size_t) j];
                ++j;
            }
            if (j - i >= 2) {
                int prefix = 0;
                for (int q = i; q < j; ++q) {
                    out_grp[q].first = prefix;           // the batched entry's within-batch prefix
                    prefix += m.pf_rowsn[(size_t) q];
                    m.pf_batch_end[(size_t) q] = j;      // every member points one past the batch
                }
                i = j;
            } else {
                ++i;
            }
        }
    }
    m.pf_groups = groups;
    m.pf_last_rows = total;

    // ---- enqueue only
    int caller = -1;
    (void) cudaGetDevice(&caller);
    if (!switch_to(m.device)) { err = "exl3 peer: cannot switch to the tier device"; return false; }
    struct Restore { int prev; ~Restore() { (void) cudaSetDevice(prev); } } restore{caller};
    bool enq = true;
    try {
        const size_t used = m.pf_off[4] + (size_t) groups * sizeof(Exl3Group);
        enq = ck(cudaMemcpyAsync(m.pf_d_slab, m.pf_h_slab, used, cudaMemcpyHostToDevice, m.stream),
                 "prefill plan H2D", err) &&
              ck(cudaMemcpyAsync(m.pf_d_x16, x16, (size_t) tokens * m.n_embd * sizeof(uint16_t),
                                 cudaMemcpyHostToDevice, m.stream), "prefill x16 H2D", err) &&
              ck(cudaMemsetAsync(m.pf_d_sum, 0, (size_t) tokens * m.n_embd * sizeof(float), m.stream),
                 "prefill sums clear", err);
        // A batch's rows are consecutive in the rows slab, so its scatter-add is ONE launch over
        // the whole range; the batched FFN scatters its output rows through the iota dest table.
        auto standalone = [&](int g) {
            const int rowsn = m.pf_rowsn[(size_t) g];
            const int h_first = m.pf_first[(size_t) g];
            float* dst = m.pf_d_rows + (size_t) h_first * m.n_embd;
            if (!exl3_moe_group_wmma(out_grp[g].gate, out_grp[g].up, out_grp[g].down, rowsn,
                                     m.pf_d_x16, m.pf_d_ires + h_first, dst, m.stream, *m.ws))
                exl3_moe_group_gemm(out_grp[g].gate, out_grp[g].up, out_grp[g].down, rowsn,
                                    m.pf_d_x16, m.pf_d_ires + h_first, dst, m.stream, *m.ws);
            exl3_pf_scatter_add(m.pf_d_sum, dst, m.pf_d_wk, m.pf_d_pair + h_first,
                                rowsn, m.pf_k, m.n_embd, m.stream);
        };
        int g = 0;
        while (enq && g < groups) {
            const int j = m.pf_batch_end[(size_t) g];
            if (j >= g + 2 &&
                exl3_moe_expert_batch((const Exl3Group*) ((unsigned char*) m.pf_h_slab + m.pf_off[4]) + g,
                                      j - g, m.pf_d_x16, m.pf_d_ires + m.pf_first[(size_t) g],
                                      m.pf_d_dest + m.pf_first[(size_t) g], m.pf_d_rows, m.stream, *m.ws)) {
                const int rows = (int) ((j < groups ? (int64_t) m.pf_first[(size_t) j] : total) -
                                        (int64_t) m.pf_first[(size_t) g]);
                exl3_pf_scatter_add(m.pf_d_sum, m.pf_d_rows + (size_t) m.pf_first[(size_t) g] * m.n_embd,
                                    m.pf_d_wk, m.pf_d_pair + m.pf_first[(size_t) g],
                                    rows, m.pf_k, m.n_embd, m.stream);
                ++m.pf_batches;
                g = j;   // the whole batch is dispatched; skip its members
                continue;
            }
            standalone(g);   // a batch start that the entry refused (unexpected) or a lone group
            ++g;
        }
        enq = enq && ck(cudaGetLastError(), "prefill FFN chain", err) &&
              ck(cudaMemcpyAsync(m.pf_host_sum, m.pf_d_sum, (size_t) tokens * m.n_embd * sizeof(float),
                                 cudaMemcpyDeviceToHost, m.stream), "prefill sums D2H", err) &&
              ck(cudaEventRecord(m.pf_done, m.stream), "prefill done event", err);
    } catch (const std::exception& e) {
        err = std::string("exl3 peer: prefill enqueue: ") + e.what();
        enq = false;
    }
    if (!enq) {
        m.failed = true;   // sticky until close(); same contract as the decode launch
        m.pf_last_rows = 0;
        return false;
    }
    m.pf_pending = true;
    ++m.pf_layers;
    m.pf_rows_total += total;
    m.pf_groups_total += groups;
    return true;
}

bool Exl3Peer::finish_prefill(std::string& err) {
    if (!valid()) { err = failed() ? "exl3 peer: tier failed earlier" : "exl3 peer: not open"; return false; }
    Impl& m = *impl_;
    if (!m.pf_pending) return true;
    int caller = -1;
    (void) cudaGetDevice(&caller);
    if (!switch_to(m.device)) { err = "exl3 peer: cannot switch to the tier device"; return false; }
    struct Restore { int prev; ~Restore() { (void) cudaSetDevice(prev); } } restore{caller};
    const cudaError_t e = cudaEventSynchronize(m.pf_done);
    if (e != cudaSuccess) {
        err = std::string("exl3 peer: prefill layer failed: ") + cudaGetErrorString(e);
        m.failed = true;
        return false;
    }
    m.pf_pending = false;
    return true;
}

}  // namespace strata::kernels
