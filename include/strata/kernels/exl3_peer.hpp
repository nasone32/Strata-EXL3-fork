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
#include <utility>
#include <vector>

namespace strata { class Exl3ExpertBacking; }
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
    /// TEST ONLY (engine fault seams): advance `expected` past every queued ready signal,
    /// making the LIVE frame permanently unsignaled. Call it BETWEEN the launch's signal
    /// enqueue and the wait/copy enqueue - the host write then precedes the wait kernel's
    /// enqueue, so the queued wait's target provably has no signal (an after-enqueue bump
    /// would race the kernel's initial read). Scope: the WORKER-ERROR release tests use
    /// this to prove abort/release drains a genuinely stuck frame; it does NOT model a
    /// mid-forward async hardware fault that no thread observes (that case stays fatal by
    /// design - see the engine's win_test_peer_fault comment). Host stores only; a
    /// released/absent frame is a no-op. Subsequent enqueues are refused, exactly like a
    /// failed tier.
    void test_desynchronize();
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

    // min_free_bytes optionally preserves a device reserve DURING allocation, not only after
    // a trial open. STRATA_EXL3_PEER_ARENA=1 packs weight planes into one aligned allocation.
    // vram_short is only set for device-memory exhaustion or a reserve refusal.
    bool open(int device, int n_embd, const std::vector<Entry>& subset,
              int max_tokens, int max_k, std::string& err, size_t min_free_bytes = 0,
              bool* vram_short = nullptr);
    void close();

    /// false while closed OR failed (a failed tier advertises nothing until close()).
    bool valid() const;
    int device() const;
    bool has(int64_t layer, int32_t expert) const;
    int64_t resident() const;
    size_t bytes() const;   ///< uploaded subset bytes (nine planes per entry, geometry-derived)
    int64_t hidden_width() const;   ///< the subset's uniform expert intermediate width (0 closed)

    /// Test-only inspector: the trellis-layout tag a RESIDENT slot's device descriptors carry.
    /// open()'s fill must propagate each entry's tag into the slot: a slot left at the default
    /// Native described packed planes wrongly, and the first adapt refill (whole-descriptor
    /// copy, tag included) then left the tier mixed-tagged so every launch was refused - the
    /// 2026-11-06 full-model band-8 regression, pinned negative-controlled in
    /// mimo_band8_parity §5b.  Native when the expert is not resident.  No engine path reads this.
    Exl3TrellisLayout slot_layout(int64_t layer, int32_t expert) const;
    /// grouped (multirow) dispatches since open: calls, routed rows, groups (diagnostics/tests;
    /// a call that falls back to indexed - no repeated expert, an unsupported codebook, or a
    /// half-integer-K (frac) expert while STRATA_EXL3_PEER_MULTIROW_FRAC is off - counts nothing
    /// here.  The engine default rejects frac tiles; the env admits the tested K <= 4.5 mul1
    /// range (launcher trial after mixed end-to-end results, 2026-11-03,
    /// notes/MIMO-PEER-FRAC-GROUPED-2026-11-03.md).
    int64_t grouped_launches() const;
    int64_t grouped_routes() const;
    int64_t grouped_groups() const;

    /// STRATA_EXL3_PEER_NATIVE_WIDTH4 (default off), read ONCE at open: a REQUESTED
    /// STRATA_EXL3_PEER_MULTIROW=4 survived the startup gate because the native grouped
    /// core was armed (fused-multirow support + STRATA_EXL3_RDNA_NATIVE=1 +
    /// STRATA_EXL3_NATIVE_GROUPED=1) and the native fp16-slice budget fits where the float
    /// twin's 4-row LDS rule would have downgraded the whole arm to indexed.  The gate OFF
    /// (or the native stack absent) keeps the startup behavior byte-for-byte the old one.
    /// Every launch then re-runs the host-only exl3_native_grouped_plan over the ACTUAL
    /// route descriptors (refills included): admitted groups run at width 4 - counted by
    /// native_width4_calls()/_rows() - and any refusal (native envs flipped off after open,
    /// an unsupported rate or an infeasible shape among the current experts) re-chunks the
    /// same route order at width 2, with the established indexed fallback below that.
    /// Per call, never sticky, never a throw.  Diagnostics/tests only.
    bool native_width4() const;
    int64_t native_width4_calls() const;
    int64_t native_width4_rows() const;

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
    // `k` is the model's top-k (1..64 - a runtime width of the plan/scatter kernels; the Qwen
    // family runs 10, MiMo v2's peer tier 8) and `n_expert` the model's expert count.
    // `defer_device=true` (the peer-loan setup only): prepare everything EXCEPT the four device
    // buffers - host slab, mapped sums, event, workspace reserves, caps - and leave the device
    // buffers to the first lend()'s pf_alloc_device, so startup never holds the full resident
    // set AND the prompt buffers at once.  prefill_enabled() is true immediately;
    // launch_prefill refuses until the buffers exist (never a null-pointer dereference), and a
    // lend that cannot allocate them fails the request honestly.
    bool enable_prefill(int64_t max_tokens, int k, int64_t n_expert, int64_t max_rows, std::string& err,
                        bool defer_device = false, bool* vram_short = nullptr);
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

    // ---- STRATA_EXL3_PEER_PREFILL_BORROW (opt-in): the prompt-time loan of resident planes ----
    //
    // The GGUF PfPart lifecycle and the primary arena's lend_tail/restore_tail, on this tier:
    // a prompt LENDS the tail of the resident subset - the experts' device planes are FREED and
    // their identities saved - so the chunk-sized prefill workspace above fits next to the full
    // resident set without a permanent --peer-slots cut.  A lent expert is not resident (has()
    // false): the prompt path re-reads residency per chunk (Prefill::run) and routes those rows
    // to the primary, whose ownership-scratch path computes them without admission; decode's
    // live has() does the same.  restore_lent() re-allocates the saved planes and re-uploads the
    // bytes from the CALLER's host sources (the model mmap, the same planes open() copied)
    // before any window reads - new plane addresses are safe because every launch rebuilds its
    // descriptors from the tier's own tables and no graph captures peer planes.  When the card
    // is too full to restore, the prefill device buffers are released first (the next lend
    // re-allocates them); if the bytes still do not fit, restore fails HONESTLY (the request
    // errors like a failed arena-loan restore; the still-lent experts stay non-resident and
    // decode computes them on the primary - slower, never wrong).
    //
    // Constraints kept: the plane-arena mode refuses the loan (one allocation has no tail to
    // free); a lend/restore needs the tier idle (no pending launch, no in-flight replacement -
    // the engine's request boundary drains both); no single layer lends more than 32 experts
    // (the primary computes lent rows through its bounded one-wave scratch); the total loan is
    // capped at a quarter of the subset.  Nothing here runs unless the engine opts in.
    bool lend_experts(int count, std::string& err);   ///< lend up to `count` MORE tail experts
    /// STRATA_EXL3_RESIDENT=peer (Qwen peer-complement RAM): the identities lend_experts(count)
    /// WOULD free next, in lend order, without mutating anything.  Pure walk of the same state
    /// lend_experts reads (tail order, the 32-per-layer cap, the quarter-of-subset cap, the
    /// already-lent skip); the caller runs it and the matching lend_experts back to back on one
    /// thread, so the two agree.  A caller that needs the planes' bytes somewhere else before
    /// they are freed (the resident arena's GPU->RAM return) copies them between the two calls.
    bool lend_tail_preview(int count, std::vector<std::pair<int32_t, int32_t>>& out) const;
    bool restore_lent(const std::vector<Entry>& sources, std::string& err);
    int64_t lent_experts() const;          ///< currently lent (0 = no loan out)
    size_t lent_expert_bytes() const;
    std::vector<std::pair<int32_t, int32_t>> lent_list() const;   ///< (layer, expert) still lent
    /// true while the prefill device buffers are allocated (a tight restore releases them;
    /// prefill_ensure_device re-allocates before the next prompt chunk launches).
    bool prefill_device_ready() const;
    bool prefill_release_device(std::string& err);
    bool prefill_ensure_device(std::string& err);

    // Optional adaptation resources, allocated at setup only. Replacements reuse the
    // existing slot planes; larger/different-layer candidates are ineligible.
    struct Replacement { int32_t out = -1; Entry in; };
    bool enable_adaptation(int max_swaps, size_t byte_cap, std::string& err);
    bool can_replace(const Replacement& replacement) const;
    // Writable host targets; drained windows only, on the owned refill stream.
    bool copy_to_host(const Entry& destination, std::string& err);
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
