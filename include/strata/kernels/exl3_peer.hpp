// include/strata/kernels/exl3_peer.hpp - the async EXL3 peer tier primitive.
//
// The native tier's shape (strata/core/peer_experts.hpp: open/has/launch/finish) for a resident
// resident subset of (layer, expert) EXL3 experts computed on a second device, unweighted row
// outputs (the engine's moe_combine applies router weights). The initial subset is static
// unless Exl3PeerAdapt enables bounded asynchronous replacement; this primitive owns no policy.
//
// API contract:
//  - open(device, n_embd, subset, max_tokens, max_k): validates every entry's EXl3 geometry
//    BEFORE any allocation, device switch or host plane read: dims in [1, INT32_MAX/16] and
//    MULTIPLES OF 8 (the kernels' H128 alignment), bits 1..8, cb 0..2, the mat relations,
//    uniform n_embd/hidden, no duplicate (layer, expert), and precomputes ALL plane sizes with
//    checked products/sums (planes <= 4 GiB, aggregate and worst-case workspace bounded,
//    max_tokens x max_k <= 65535 - the indexed kernel's routes limit).  The API carries no
//    source lengths, so the CALLER guarantees the host planes hold at least the derived bytes.
//    The resource phase is TRANSACTIONAL: every acquired resource is released exactly once on
//    any failure - returned errors, kernel/API exceptions, injected faults - and the tier is
//    left empty (valid() false, resident() 0, has() false).  Precisely: the try covers the
//    sizing pass and the allocation/upload phase (including std::bad_alloc from the vectors
//    they build); only the scalar argument checks - which allocate nothing but their error
//    strings - are outside it.
//    The device switch is checked; a failed switch fails open rather than allocating on the
//    wrong device.
//  - launch(layer, x, ids, tokens, k, kind): validates the FULL request first (nulls, dims,
//    caps, overflow, residency of every kind[i]==2 id).  A non-resident id (or any other
//    invalid input) rejects the WHOLE request with nothing enqueued and no slot fallback.
//    Then it ENQUEUES ONLY (H2D of x/ids, f32->f16, indexed expert rows, D2D/D2H of rows) and
//    returns: no stream sync or file IO. Optional RDNA/FUSE scratch must be pre-reserved
//    for allocation-free variable shapes: STRATA_EXL3_PEER_SPARSE=1 or
//    STRATA_EXL3_RESERVE_FUSED=1 does this at open(), before a verifier can spin.
//    x/ids/kind are copied into
//    preallocated pinned buffers during launch (they need to live only for the call).  ONE
//    launch may be pending: a nested launch is rejected without touching state; an empty kind
//    set is an explicit no-op success.  A mid-flight enqueue failure - including the indexed
//    entry's descriptor upload, which is checked in src/kernels/cuda/exl3.cu and throws - marks
//    the tier FAILED (sticky until close): valid()/has() then report false, finish() reports
//    the failure and nothing invalid is ever advertised.  The descriptors live in pinned
//    memory, so their upload is a genuine async copy.
//  - finish(out): spin-waits the pending work (never a sleeping sync) and writes the
//    unweighted rows into the CALLER's out: out[route * n_embd] = ffn(x[route/k], expert).
//    Destinations are unique per route; rows of non-peer routes are never touched.  out must
//    live across the finish call (a future direct-scatter variant would additionally require a
//    portable-registered out).  With nothing pending, finish(out) is a no-op.
//    STRATA_EXL3_PEER_CONVERT=1 uses the exact bulk host fp16 converter for this scatter.
//    finish(nullptr) is ALWAYS an error (even with nothing pending): it can never complete rows.
//  - Optional replace/apply_refill uses a separate refill stream and setup-allocated pinned
//    staging. Replaced entries are withdrawn immediately and published only after its event.
//    Unrelated resident entries may compute meanwhile. Replacement calls/publication are
//    between windows, with no compute launch pending; the owner reserves incoming experts
//    against primary admission until publication. Plane capacities and VRAM allocation stay fixed.
//  - close() (and the destructor) drain the compute AND refill streams before freeing any buffer or the
//    workspace, also with a launch pending; a failed tier is cleared by close().  DEGRADED
//    RUNTIME LIMITATION (not a normal path): if the device switch fails at close, nothing
//    device-side is freed (cross-context frees are unsafe) - the pointers are dropped and the
//    process exit reclaims.
//  - Device switches are CHECKED and the caller's device is restored best-effort afterwards; a
//    failed switch fails the call safely (nothing is enqueued or freed on the wrong device).
//    Nothing stronger is claimed under runtime failure.
//
// Counters: launches()/rows() count ACCEPTED launches and rows (an empty launch counts as one
// launch); enqueues() counts launches that reached the enqueue chain - the seam proof that
// rejected requests do no work.  busy() is an observational stream query.
//
// Concurrency: one instance is a single-owner object (like PeerExperts) - no internal locking;
// independent instances do not share state (the test seam is per-instance).
//
// The API surface is backend-agnostic (std types + Exl3Mat PODs); the implementation rides the
// existing CUDA/HIP kernel path (strata_kernels).  The sycl/ tree has no EXL3 sources and is
// not covered. The engine installs the tier only under explicit STRATA_EXL3_PEER opt-in.
#pragma once

#include "strata/kernels/exl3.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::kernels {

// Experimental bounded fp16 transport with device-side completion/scatter. Owns its
// mapped frame independently of peer residency, so a closed/failed peer cannot
// invalidate a captured verifier graph. The primary graph must consume a frame
// before the next layer's callback reuses it. Ready means producer completion,
// not consumer completion: this is enforced by the serial verifier's next-layer
// doorbell, not by enqueue(). One group/window only; callers must drain the
// primary graph before destroying the transport or replacing peer residency.
class Exl3PeerOutput {
public:
    Exl3PeerOutput();
    ~Exl3PeerOutput();
    bool open(int primary, int peer, int embd, int capacity, std::string& err,
              bool p2p = false, bool kernel_transfer = false);
    bool enqueue(const uint16_t* rows, const int* destinations, int count, void* stream, std::string& err);
    void empty();
    void copy(float* parts, int routes, const uint32_t* skip, uint32_t ring, void* stream);
    void release(); // watchdog: host stores only; graph refuses subsequent windows
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// STRATA_EXL3_PEER_PACK_PLAN=1 packs eligible grouped dispatch into one pinned
// upload and reads the workspace output directly; the pending-launch contract
// keeps its source slab immutable until completion. Indexed fallback is unchanged.
class Exl3Peer {
public:
    /// One resident (layer, expert): host plane descriptors, read and copied during open().
    struct Entry {
        int64_t layer = -1;
        int32_t expert = -1;
        Exl3Mat g, u, d;
    };

    Exl3Peer();
    ~Exl3Peer();
    Exl3Peer(const Exl3Peer&) = delete;
    Exl3Peer& operator=(const Exl3Peer&) = delete;

    bool open(int device, int n_embd, const std::vector<Entry>& subset,
              int max_tokens, int max_k, std::string& err);
    void close();

    /// false while closed OR failed (a failed tier advertises nothing until close()).
    bool valid() const;
    int device() const;
    bool has(int64_t layer, int32_t expert) const;
    int64_t resident() const;
    size_t bytes() const;   ///< uploaded subset bytes (nine planes per entry, geometry-derived)

    bool launch(int64_t layer, const float* x, const int32_t* ids, int tokens, int k,
                const int32_t* kind, std::string& err, Exl3PeerOutput* device_output = nullptr);
    bool finish(float* out, std::string& err);
    bool finish_deferred(std::string& err);

    // ---- STRATA_EXL3_PEER_PREFILL=1: the prompt chunks' peer-owned experts, on this tier ----
    //
    // The GGUF peer-prefill shape (src/prefill/prefill.cpp's PeerPrefill) with EXL3 kernels: the
    // caller splits each MoE layer's routes into peer-owned (this tier, `has()`) and local, sends
    // the layer's FP16 activation image and router weights, and this tier computes its experts'
    // rows with the same grouped WMMA FFN the primary's device arm uses (exl3_moe_group_wmma),
    // folding the rows into ONE WEIGHTED SUM PER TOKEN (the eddoursul/Strata transport: T x N
    // floats back instead of rows_peer x N) through mapped host memory.  The summation order
    // differs from a single-GPU prompt - this is the GGUF peer-prefill exactness class, NOT
    // bit-identity; decode windows are untouched.
    //
    // enable_prefill() allocates every device/pinned buffer ONCE at setup (before any prompt or
    // window) and reserves the FFN workspace for the full chunk: no allocation under queued work.
    // `max_rows` caps the rows this tier takes per layer; routes beyond the cap stay local (the
    // caller learns the effective cap from prefill_rows_cap()).  Refuses when the peer card lacks
    // the bytes (the desktop reserve stays intact); the prompt path then stays on the primary.
    // `k` must be 10 and `n_expert` the model's expert count (the route plan and the fixed-k
    // transport kernels assume them).
    bool enable_prefill(int64_t max_tokens, int k, int64_t n_expert, int64_t max_rows, std::string& err);
    bool prefill_enabled() const;
    int64_t prefill_rows_cap() const;
    /// `x16` is the caller's MAPPED pinned FP16 image of the layer's mixed activations (T x n_embd),
    /// complete when this is called; `ids`/`weights` are host arrays of tokens x k.  `peer_mask`
    /// (tokens x k) is the caller's split: entry 1 = this tier computes that route (it must be
    /// resident; every masked id is validated).  Enqueues only.  The mapped sums buffer
    /// (prefill_sums_host()) is valid after finish_prefill().
    bool launch_prefill(int64_t layer, const uint16_t* x16, int64_t tokens,
                        const int32_t* ids, const uint8_t* peer_mask, const float* weights,
                        std::string& err);
    bool finish_prefill(std::string& err);   ///< drains the pending prefill layer (nothing = no-op)
    const float* prefill_sums_host() const;
    int64_t prefill_last_rows() const;       ///< the pending/last layer's peer rows
    int64_t prefill_layers() const;          ///< counters (STRATA_EXL3_TIME reports them at close)
    int64_t prefill_rows_total() const;

    // Optional adaptation resources, allocated at setup only. Replacements reuse the
    // existing slot planes; larger/different-layer candidates are ineligible.
    struct Replacement { int32_t out = -1; Entry in; };
    bool enable_adaptation(int max_swaps, size_t byte_cap, std::string& err);
    bool can_replace(const Replacement& replacement) const;
    size_t replacement_bytes(const Replacement& replacement) const;
    // Between windows, with no compute launch pending. Sources are packed into pinned
    // staging before returning. Outgoing entries become non-resident immediately;
    // incoming entries become resident only at apply_refill after the event lands.
    bool replace(const std::vector<Replacement>& replacements, std::string& err);
    bool apply_refill(bool wait, std::vector<Replacement>& landed, std::string& err);
    bool refilling() const;
    double refill_ms() const; // completed batches' GPU-event copy/enqueue spans
    void residency(int layers, int experts, std::vector<int32_t>& out) const;

    bool pending() const;
    bool failed() const;
    bool busy() const;      ///< observational: the enqueued work is still executing

    int64_t launches() const;
    int64_t rows() const;
    int64_t enqueues() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::kernels
