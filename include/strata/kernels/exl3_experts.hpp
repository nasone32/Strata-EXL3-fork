// include/strata/kernels/exl3_experts.hpp - one layer's routed EXL3 experts resident on the GPU.
//
// The engine's CPU expert pool is blob/quantized-activation oriented; EXL3 experts are per-expert
// trellis/suh/svh tensors (docs/EXL3.md), so this is the GPU-side counterpart: upload a layer's experts
// once and run the MoE (`exl3_moe_ffn`) for a token's routed experts.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "strata/kernels/exl3.hpp"

namespace strata {
class Exl3ExpertBacking;   // artifact/exl3_expert_backing.hpp - optional packed-expert source
}

namespace strata::kernels {

class Exl3ExpertStore {
public:
    // Uploads `n_experts` experts of `layer` from the EXL3 model at `model_dir` to the GPU.  `stream` may
    // be null (default stream).  `layer` is the transformer layer index; expert keys are
    // `<layer_prefix><layer>.mlp.experts.<e>.{gate,up,down}_proj` - the default is the Qwen family's
    // `model.language_model.layers.` layout, a MiMo v2 dir passes `model.layers.` (what
    // strata::mimo_expert_tensor names, see strata/artifact/mimo_pack.hpp).
    Exl3ExpertStore(const std::string& model_dir, int layer, int n_experts, void* stream,
                    const std::string& layer_prefix = "model.language_model.layers.");
    ~Exl3ExpertStore();
    Exl3ExpertStore(const Exl3ExpertStore&) = delete;
    Exl3ExpertStore& operator=(const Exl3ExpertStore&) = delete;

    int n_experts() const { return n_experts_; }
    // Bytes of GPU memory the layer's experts occupy (for the engine's budget).
    size_t bytes() const { return bytes_; }

    // out = sum_i weights[i] * ffn(ids[i], x); x: 2560 fp16, out: 2560 fp16.  ids are expert indices in
    // [0, n_experts), k of them.  `stream` may be null.
    void run(const uint16_t* x, const int* ids, const float* weights, int k, uint16_t* out, void* stream) const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
    int n_experts_ = 0;
    size_t bytes_ = 0;
};

// Streaming EXL3 experts for the engine's pool: reads a token's routed experts straight from the model's
// safetensors (mmap), uploads just those, runs them, and hands back the k UNWEIGHTED rows in fp32 (the
// engine's `moe_combine` applies the router weights).  One instance serves all layers.
class Exl3ExpertStream {
public:
    /// `slots_per_layer`: routed experts kept resident per layer (the same idea as the GGUF path's
    /// `--expert-cache`, counted per layer because the stream is).  0 keeps none: every use reads and
    /// uploads the expert again.  -1 sizes the slots from the free VRAM after this instance's own buffers.
    /// `n_layers` is the model's layer count (the cache spans all of them).
    /// `ram_cache_bytes` is the host-side expert cache (0 = off).  **THE PAGE CACHE IS NOT A CACHE YOU CAN
    /// COUNT ON**: on the development machine a 10.6 GB read of the routed experts did not stay resident, so
    /// every miss was a SATA read (measured 8-91 ms of uploads per token against 1.4 ms of FFN).  A managed
    /// host cache sits outside whatever reclaim policy evicts those pages and is the design the GGUF path
    /// already uses (`--resident-cpu-experts`): a miss is read into RAM, and every later miss of the same
    /// expert is a memcpy at RAM speed (0.2 ms per 1.86 MB against 3.7 ms from the disk).
    ///
    /// `expert_backing` (opt-in, default none): a shared packed-expert source (an Exl3Band8Store through
    /// the abstract Exl3ExpertBacking).  linear()/locate() then serve the packed planes for covered
    /// experts and the raw source for the rest; every uploaded descriptor carries the linear's own
    /// `.layout` tag and this stream's workspaces are switched to the backing's layout exactly once,
    /// drained, at construction.  The ONE canonical format rule: all experts this stream loads must
    /// share the model's expert_layout() - a mixed (Native + packed) record throws instead of computing
    /// on wrong bytes.  A nonzero `ram_cache_bytes` is REFUSED with a backing (a second, managed copy
    /// of the packed planes would break the one-backing memory goal) - pass 0.  Every pre-existing
    /// call site (no backing) is unchanged.
    Exl3ExpertStream(const std::string& model_dir, int n_embd, int n_layers, int slots_per_layer, void* stream,
                     size_t ram_cache_bytes = 0, size_t reserve_device_bytes = (size_t) 512 << 20,
                     int layer_lo = 0, int layer_hi = -1, int prefill_tokens = 0,
                     const std::string& layer_prefix = "model.language_model.layers.",
                     std::shared_ptr<const Exl3ExpertBacking> expert_backing = {},
                     size_t auto_loan_bytes = 0);
    ~Exl3ExpertStream();
    Exl3ExpertStream(const Exl3ExpertStream&) = delete;
    Exl3ExpertStream& operator=(const Exl3ExpertStream&) = delete;

    // out[i*n_embd] = ffn(layer, expert ids[i], x) for i in [0,k).  `x_f32` and `out` are HOST fp32.
    // `cpu_rows` (opt-in STRATA_EXL3_CPU_TIER, host-IO calls only): a k-byte buffer the call
    // ZEROES first; entry i left at 1 marks a selection the stream REFUSED - the cache is full
    // and the tier is on, so nothing was admitted, evicted or uploaded for it - and the caller
    // computes that row itself (the GGUF expert pool's overflow policy, on the CPU).  A
    // refused row is ZEROED in `out` here (the blocking single-token flow; run_multi, whose
    // callers overlap their CPU work with the GPU call, leaves the row untouched).  Null keeps
    // today's evict-and-admit-always behavior.
    bool run(int layer, const float* x_f32, const int* ids, int k, float* out, void* stream,
              uint8_t* cpu_rows = nullptr);
    // Single-token DEVICE input/output with HOST ids, distinct experts only. Keeps the run()
    // cache/admission and hit/miss overlap, but uses pinned indexed plans and enqueues results
    // without an activation/output host round trip. Caller MUST synchronize this compute stream
    // (or observe a downstream publication proving its completion) before the next
    // service/residency mutation (including run_multi/run_device), and before
    // reading output. Requires PRIMARY_PACK_PLAN (default on); no CPU refusal tier.
    bool run_decode_device(int layer, const float* x_device, const int* ids, int k,
                            float* out_device, void* stream, uint8_t* cpu_rows = nullptr);
    // One decode layer in flight. Setup/refresh only on a drained stream. Publish launches
    // resident rows before waiting for the mapped routing record (not FFN completion).
    // Returned payload lives until the next publish; end_host releases admission pins.
    // max_tokens defaults to 1; opt-in multirow capacity is <=8 and max_tokens*k<=64.
    // Multirow publications publish empty GPU hits unless set_window_graph_hits(true)
    // armed the opt-in and the current refresh declared the window's tables valid; they
    // publish F32 x[T,n] + ids/weights[T,k] either way, and the ordinary host planner's
    // admission/observation semantics are retained on the miss-only complement.
    void enable_decode_handoff(int k, int max_tokens = 1);
    // First actual T1 use of a pre-enabled window frame: caller must drain producer and
    // host service first. Matches enable_decode_handoff(k)'s permanent indexed-planner gate.
    void prepare_decode_handoff_t1();
    void decode_handoff(int layer, const float* x, const int32_t* ids, int k,
                        float* out, void* stream, const int32_t*& host_ids,
                        const float*& host_x, const uint8_t*& hits);
    // Split producer/service protocol: enqueue publish + resident FFNs + completion wait.
    // A separate host worker collects and serves on a separate preallocated stream.
    // Complete publishes in service-stream order; the next routing publication proves
    // previous service scratch/plans are reusable. One nonzero ring per forward.
    void enqueue_decode_handoff(int layer, const float* x, const int32_t* ids, int k,
                                float* out, uint32_t ring, void* stream,
                                int tokens = 1, const float* weights = nullptr);
    // Poll only a service-owned stream: querying a producer whose host is in a
    // pageable readback can deadlock on that stream's HIP driver lock.
    void collect_decode_handoff(int layer, uint32_t ring, void* poll_stream,
                                const int32_t*& ids, const float*& x, const uint8_t*& hits,
                                int* tokens = nullptr, const float** weights = nullptr);
    void complete_decode_handoff(int layer, uint32_t ring, void* service_stream = nullptr);
    void abort_decode_handoff();
    // Abort-path recovery for the HOST bookkeeping only. abort_decode_handoff publishes
    // the failfast flag and never touches host state (the service owner owns it). A frame
    // that aborted between begin and complete leaves the bracket open; call this from the
    // SERVICE owner after the unwind, or on a joined/drained frame (the next
    // refresh_verify_graph also closes it), so no later host-IO call can drop groups
    // against a stale record. No device effect, no lock.
    void recover_host_service();
    // HOST x_f32[tokens, n_embd], ids[tokens, k], out[tokens*k, n_embd]. Groups routes by
    // expert, stages each distinct expert once per window, and restores token-major output order.
    // Scratch is bounded by the input and a 4096-route tile, not tokens*k*n_embd device output.
    // `cpu_rows` as in run(), tokens*k entries, indexed by the original route order (token*k+j);
    // refused rows are NOT written (the caller owns them - typically it started computing them
    // before this call via classify()).
    bool run_multi(int layer, const float* x_f32, const int32_t* ids, int tokens, int k,
                    float* out, void* stream, uint8_t* cpu_rows = nullptr);
    // Peer complement: same HOST layout and original token indices, but routes whose experts
    // are in set_owned_elsewhere() are omitted and their output rows left untouched. The
    // caller must finish the peer into those rows before publication. No padding/chunking.
    // For use beside a spinning verifier, construct with STRATA_EXL3_PEER_SPARSE=1 so
    // optional RDNA/FUSE scratch is pre-reserved before the callback can run.
    bool run_multi_complement(int layer, const float* x_f32, const int32_t* ids, int tokens, int k,
                               float* out, void* stream, const uint8_t* cpu_skip = nullptr);
    // Optional cpu_skip is a caller-owned, immutable tokens*k mask of complete
    // expert groups computed concurrently on CPU. Those rows are left untouched.
    // Mixed masks within one expert group are rejected before GPU work.
    // Prefill: x_f32 and out are DEVICE, ids is HOST. Large expert groups use tensor-core GEMM.
    // skip (STRATA_EXL3_PEER_PREFILL) is the same kind of whole-group mask - the peer tier's
    // prompt share: masked groups are dropped from the plan BEFORE staging/compute (their
    // weights are never uploaded), and the surviving rows keep their ORIGINAL destinations,
    // so the caller combines them through the identity slot map with a sentinel for the rest.
    bool run_device(int layer, const float* x_f32, const int32_t* ids, int tokens, int k,
                     float* out, void* stream, const uint8_t* skip = nullptr);
    // STRATA_MIMO_WINDOW_DEVICE_MISSES (default off): the verify-window service's
    // device-input/device-output twin of run_multi_complement - it serves BOTH the
    // complement (peer rows / a live CPU overlay in the same window) and the PLAIN
    // window the host arm ran through run_multi (there the peer-owned / cpu_skip /
    // graph-hit drops are provably no-ops: a plain window holds no peer-owned or CPU
    // row by definition, and armed graph hits drop exactly the rows run_multi's host
    // bracket dropped, so the plan and its arithmetic are identical). x_f32 and out are
    // DEVICE (the producer's activation image and route-row slab, both live while the
    // verifier waits); ids is HOST; cpu_skip is the same whole-group mask. Semantics
    // are exactly run_multi_complement's - peer-owned groups dropped (skip_owned),
    // cpu_skip groups dropped, published graph-hit groups dropped against the live
    // host bracket, misses admitted/staged identically, pending-incoming experts on
    // GPU scratch - and the FFN arithmetic is the host arm's (the same indexed /
    // small-M grouped entry points, the host arm's whole-window tile boundaries; the
    // prefill group arm never runs here).
    // The differences are transport only: the activation is read in place (no F32 H2D),
    // plans upload through the same PINNED/PACKED slabs the T1 device arm requires, and
    // rows scatter straight into `out` on the caller's stream (no result D2H, host memcpy
    // or re-upload). Requires the packed pinned-plan machinery; check
    // window_complement_device_ready() once instead of probing per call.
    bool run_multi_complement_device(int layer, const float* x_device, const int32_t* ids,
                                     int tokens, int k, float* out_device, void* stream,
                                     const uint8_t* cpu_skip = nullptr);
    /// Whether run_multi_complement_device can serve a tokens x k window on this stream:
    /// the packed pinned-plan slabs exist and cover the window's route count. Pure query,
    /// no device work; false means the caller keeps the host-IO complement.
    bool window_complement_device_ready(int tokens, int k) const;

    // Opt-in verifier integration. Enable once before capture; one window/group in flight.
    // refresh runs before launch; publish snapshots/pins the routes before hits and host
    // miss service overlap. copy_misses follows its wait; begin/end bracket host service.
    void enable_verify_graph(int max_tokens, int k);
    // `window_tokens` > 1 additionally arms THIS refresh's tables for multirow handoff
    // windows: set_window_graph_hits(true) must have armed the mode, and the tables are
    // published for windows of <= window_tokens rows only while the ACTIVE cache holds
    // window_tokens*k slots (a whole window's distinct experts fit beside its hits, so
    // miss admission can always find an unpinned victim). When either condition fails the
    // tables stay empty and windows publish the NULL table - every route a host-served
    // miss, byte-identical to the default. The default 0 keeps the T1-only contract:
    // tables for the single-token handoff, windows keep empty hits. window_tokens is
    // validated first: it must be 0..max_tokens (negative or oversized values throw, so
    // the tokens*k gate can never overflow). Called only on a drained frame; that drained
    // seam is also the recovery point after an abort: a stale host-service bracket is
    // closed and the sticky abort flag cleared, so the protocol may resume on a fresh
    // ring (the engine's own worker error stays sticky and never comes through here).
    void refresh_verify_graph(void* stream = nullptr, int window_tokens = 0);
    // Opt-in (default off, common API): MULTIROW handoff windows (T=2..8) may publish
    // RESIDENT-descriptor GPU hits. The producer's stream then computes hit rows inline
    // (exl3_moe_rows_masked over the same cache slots, both trellis physical formats via
    // the descriptors) while the service side resolves ONLY miss/peer/CPU routes: collect
    // runs the begin_host contract for windows with hits (frequency observation and LRU
    // ticks exactly once, hit slots pinned through miss admission, host_active drops the
    // hit groups from run_multi/run_multi_complement). Windows without hits keep the
    // original all-miss statistics and host planner untouched. Requires
    // enable_decode_handoff(k, max_tokens >= 2) first; false restores the default
    // (windows publish no hits) for every later refresh.
    void set_window_graph_hits(bool enabled);
    void verify_graph_hits(int layer, const float* x, const int32_t* ids, int tokens, int k,
                           float* out, uint32_t ring, void* stream);
    const uint32_t* verify_graph_skip() const;
    void verify_graph_publish(int layer, const float* x, const int32_t* ids, int tokens, int k,
                              float* mapped_x, int32_t* mapped_ids, uint32_t* seq, uint32_t ring, void* stream,
                              const float* weights = nullptr);
    void verify_graph_copy_misses(const float* mapped, float* out, int routes, void* stream);
    /// what the LAST refresh_verify_graph declared for multirow windows: the largest T
    /// whose tables are valid (0 = windows publish no hits). Read-only diagnostic.
    int window_hit_tokens() const;
    bool verify_graph_begin_host(int layer);
    void verify_graph_end_host(int layer);
    bool verify_graph_host_active() const;
    void verify_graph_check(int layer, const float* x, const int32_t* ids, int tokens, int k);

    // ---- Ownership exclusion (the peer tier's disjointness hook; policy lives outside) ----
    // A byte-per-(layer, expert) table: entries != 0 are owned by ANOTHER tier (the peer).  The
    // stream then (a) never ADMITS them into this cache (VRAM admit, warmup fill and the loan's
    // restore/refill all skip them), and (b) computes their routes ONLY on the fallback path
    // (prefill, or a caller that cannot use the owner) into the bounded scratch - never into the
    // cache.  Call once at startup before any request; the table must outlive the stream.  A null
    // table/size clears it (the default: no exclusion). An adapter may update its bytes between
    // windows, reserving incoming experts with set_pending_elsewhere until they are ready.
    void set_owned_elsewhere(const uint8_t* table, int64_t n_layers, int64_t n_expert);
    // Between windows: incoming peer refills are excluded from ADMISSION but still computed
    // on the primary scratch until the peer publishes them. Same dimensions as the owner table.
    void set_pending_elsewhere(const uint8_t* table);
    void residency(int layers, int experts, std::vector<int32_t>& out) const;
    using RouteObserver = void (*)(void*, int, const int32_t*, int);
    void set_route_observer(RouteObserver observer, void* user);
    /// whether (layer, expert) is currently RESIDENT in this stream's cache (introspection)
    bool holds(int64_t layer, int32_t expert) const;
    /// whether (layer, expert) is excluded from this stream's ADMISSION by ANY ownership
    /// table: the STATIC peer-owned table OR the PENDING-incoming reservation an in-flight
    /// refill holds (Impl::admission_excluded - the same union the admission loop uses).
    /// Deliberately the union, not owned_elsewhere(): run_multi_complement drops only the
    /// STATIC table, so a pending-incoming group still computes on this GPU's scratch - a
    /// caller partitioning work against the peer side (the verify-window CPU planner) must
    /// treat BOTH states as "not mine", or it would take a group the GPU still computes.
    /// Read-only host introspection: no GPU query, no sync, no lock beyond holds()'s (none);
    /// out-of-range layer/expert reports false (nothing excluded).  Policy lives in the caller.
    bool excluded_elsewhere(int64_t layer, int32_t expert) const;
    /// routes/admissions refused by the ownership exclusion since the last clear (test evidence)
    int64_t owned_exclusions() const;

    /// `--expert-profile` on the EXL3 cache - the GGUF R4.2e fill (generate.cpp's profile fill),
    /// same `profile.bin` and the same `read_expert_profile` file: at startup, before any request,
    /// place each owned layer's hottest ranked pairs into its free slots in dense order, reading
    /// their planes through the ordinary staging path.  Unlike the GGUF cache (which never evicts
    /// and lets the profile BE the policy) this is only the INITIAL state - the measured-faster
    /// LRU keeps adapting afterwards.  Pairs owned by another tier, already resident, duplicated in
    /// the ranking or outside this stream's layers are skipped.  Reads back and compares the first
    /// filled slot's planes against the source before returning (a residency table wrong about
    /// BYTES is the expensive failure mode).  Startup only; takes the run mutex and drains the
    /// staging before returning.  `n_filled`/`n_bytes` report what landed.
    bool warmup_fill(const std::vector<std::pair<int32_t, int32_t>>& ranked,
                     int64_t* n_filled, int64_t* n_bytes, std::string& err);

    /// STRATA_EXL3_CPU_TIER's overlap helper: the refusal decision run()/run_multi would make
    /// right now, WITHOUT side effects - flags[t*k+j] = 1 marks a route the call will refuse
    /// (cache full; nothing admitted, evicted or uploaded for it).  Start the CPU tier on
    /// those rows, then call run/run_multi with the same buffer: the decisions recompute
    /// identically on one calling thread.  The refused rows are NOT written by the calls.
    void classify(int layer, const int32_t* ids, int tokens, int k, uint8_t* flags) const;

    size_t bytes() const;
    struct CacheStats { uint64_t staged_experts = 0, staged_bytes = 0; double stage_host_ms = 0; };
    CacheStats cache_stats() const; // diagnostic counters, enabled by STRATA_EXL3_CACHE_STATS=1
    std::vector<int32_t> cache_lru_order(int layer) const; // diagnostic: oldest first, drained stream
    int slots_per_layer() const;
    /// the host cache's slots per layer and its bytes (0 when it is off)
    int ram_slots_per_layer() const;
    size_t ram_bytes() const;

    // ---- stream loan primitives (experimental STRATA_EXL3_PREFILL_BORROW arena backing only) ----
    //
    // A future prefill coordinator (generate.cpp's PfPart/lend/refill) borrows whole TAIL RANKS of
    // the rank-major arena.  The original nine-per-layer backing and cache == 0 report NO loan;
    // every call below then fails without side effects.  Between calls every owned layer keeps a
    // dense invariant: its mapped slots are exactly [0, filled) (a bijection), filled <= active.
    //
    //   - While a loan is open, run/run_multi/run_device route and evict only in
    //     [0, loan_active_ranks()); a call needing more live experts falls back to the per-call
    //     scratch, so lent addresses are never read or written.
    //   - The lent range is the caller's between lend and restore: it may overwrite the bytes and
    //     MUST be done with them (its own stream syncs) before restore_tail().  lend/restore take
    //     the run mutex and drain this stream's compute+staging streams first; a FAILED drain
    //     rejects the request with no state change - no loan opens on a context whose syncs fail,
    //     and no recovery is claimed for one.
    //   - lend growth re-lends a superset from the same arena end (the address moves down by
    //     rank_bytes per extra rank); a same-size repeat is idempotent; a smaller size, and any
    //     request that would leave no active rank, are rejected with nothing changed.
    //   - restore_tail() is a transaction.  Per owned layer, in a deterministic order (layer order,
    //     then the record's original slot), each saved tail expert that is NOT already resident is
    //     re-read from the model (existing pread/pinned staging) and placed densely - extending
    //     that layer's own occupied prefix into its own tail; an expert saved more than once
    //     (a growth can re-lend a re-admitted expert) is placed once; residents keep their hot
    //     prefix copy (overwritten tail bytes are never trusted).  Admission is bounded: one
    //     loan's records come from the disjoint rank increments of its lends, so they never
    //     exceed the lent ranks and always fit the tail under the dense invariant; if that bound
    //     were ever exceeded anyway, the surplus records in the deterministic order are DROPPED
    //     (not resident; ordinary misses re-admit them) rather than failing the restore.  All
    //     planning - next maps, fill watermarks, placement lists - completes before staging, and
    //     publication happens only after every copy completed, via swap/assign operations that
    //     cannot throw or allocate.  If uploads began and anything fails, no slot is advertised:
    //     every layer's device AND host-cache residency is invalidated (the run error paths'
    //     rule; not a claim that a poisoned context recovers) and the full capacity is still
    //     returned.  A failed PRE-drain returns false with the loan still open.  loan_epoch()
    //     bumps on every successful lend (growth included) and restore, never on a rejection.
    struct Loan {
        void* address = nullptr;     // the lent device range; nullptr <=> the lend failed
        size_t bytes = 0;            // ranks * loan_rank_bytes()
        int ranks = 0;
        uint64_t epoch = 0;          // the loan's epoch; meaningful only on success
    };
    /// the arena backing exists and holds a cache: this stream can lend
    bool loan_available() const;
    bool auto_sized_for_loan() const;
    /// one rank's bytes (this stream's owned layers, all nine planes); 0 when unavailable
    size_t loan_rank_bytes() const;
    /// the largest rank count lend_tail accepts now (at least one active rank must remain)
    int loan_max_ranks() const;
    /// true while a loan is open (an empty saved tail is still an active loan)
    bool loan_active() const;
    /// the cache's routable prefix right now: slots_per_layer() minus any active loan
    int loan_active_ranks() const;
    /// bumps on every successful lend and restore; never on a rejected call
    uint64_t loan_epoch() const;
    /// lend (or grow, or idempotently repeat) a tail-rank loan; a null address means rejected
    Loan lend_tail(int ranks);
    /// end the loan: refill and publish the saved experts (the contract above); false on failure
    bool restore_tail();
    /// validation seam: every owned layer's mapped slots are exactly [0, filled), filled <= active
    bool loan_dense_invariant() const;
    /// validation seam: the ACTIVE loan's saved expert ids for `layer` (unique, ascending; empty
    /// without a loan).  Call between run calls.
    void loan_debug_saved(int layer, std::vector<int>& experts) const;
    /// validation seam: the LAST restore's placements for `layer` (unique, ascending).  A lend
    /// clears it (a loan is open, nothing was restored); a restore that ran and failed clears it;
    /// a REJECTED call (no loan, or a failed pre-drain) leaves the previous value untouched.
    /// Call between run calls.
    void loan_debug_restored(int layer, std::vector<int>& experts) const;
    /// validation seam for STRATA_EXL3_OVERLAP (test evidence, not a control): lifetime counts of
    /// host-IO waves that actually ran the hit/miss split and of waves that took the original
    /// single-pass order (all-hit, all-miss, no cache, or a wave wider than one tile).  Null
    /// pointers are skipped.  Call between run calls.
    void overlap_debug(long long* splits, long long* fallbacks) const;
    /// validation seam for STRATA_EXL3_SMALL_M (test evidence, not a control): lifetime counts of
    /// grouped-FFN calls and the routes they carried, the three indexed-stay reasons (window
    /// tokens<2 / a group wider than the compiled 6 or a tile cutting a group / route reuse
    /// below 1.2), and the FFN workspace's currently allocated bytes - the seam test asserts the
    /// opt-in really dispatched and that no window GREW the workspace after warmup (the
    /// pre-reserve-under-queued-work contract).  Null pointers are skipped.  Call between runs.
    void small_m_debug(long long* calls, long long* routes, long long* stays,
                       size_t* workspace_bytes, bool* enabled = nullptr) const;
    /// validation seam for STRATA_EXL3_RESIDENT_DIRECT_DMA (test evidence, not a control):
    /// lifetime counts of experts uploaded as ONE direct DMA out of the backing's
    /// GPU-registered arena, the gather-memcpy bytes avoided, and the packed-form experts
    /// that kept the gather path while the flag was armed (non-contiguous or unproven
    /// source).  `enabled` reports the flag itself.  Null pointers are skipped.
    void resident_direct_dma_debug(long long* jobs, long long* bytes, long long* fallbacks,
                                   bool* enabled = nullptr) const;

private:
    bool run_single_impl(int layer, const float* x_f32, const int* ids, int k,
                         float* out, void* stream, uint8_t* cpu_rows, bool device_io);
    bool run_multi_impl(int layer, const float* x_f32, const int32_t* ids, int tokens, int k,
                         float* out, void* stream, bool device_io, uint8_t* cpu_rows,
                         bool skip_owned, const uint8_t* cpu_skip, bool window_device);
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace strata::kernels
