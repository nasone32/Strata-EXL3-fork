// include/strata/kernels/exl3_experts.hpp - one layer's routed EXL3 experts resident on the GPU.
//
// The engine's CPU expert pool is blob/quantized-activation oriented; EXL3 experts are per-expert
// trellis/suh/svh tensors (docs/EXL3.md), so this is the GPU-side counterpart: upload a layer's experts
// once and run the MoE (`exl3_moe_ffn`) for a token's routed experts.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "strata/kernels/exl3.hpp"

namespace strata::kernels {

class Exl3ExpertStore {
public:
    // Uploads `n_experts` experts of `layer` from the EXL3 model at `model_dir` to the GPU.  `stream` may
    // be null (default stream).  `layer` is the transformer layer index; expert keys are
    // `model.language_model.layers.<layer>.mlp.experts.<e>.{gate,up,down}_proj`.
    Exl3ExpertStore(const std::string& model_dir, int layer, int n_experts, void* stream);
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
    Exl3ExpertStream(const std::string& model_dir, int n_embd, int n_layers, int slots_per_layer, void* stream,
                     size_t ram_cache_bytes = 0, size_t reserve_device_bytes = (size_t) 512 << 20,
                     int layer_lo = 0, int layer_hi = -1, int prefill_tokens = 0);
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

    // Opt-in verifier integration. Enable once before capture; one window/group in flight.
    // refresh runs before launch; publish snapshots/pins the routes before hits and host
    // miss service overlap. copy_misses follows its wait; begin/end bracket host service.
    void enable_verify_graph(int max_tokens, int k);
    void refresh_verify_graph(void* stream = nullptr);
    void verify_graph_hits(int layer, const float* x, const int32_t* ids, int tokens, int k,
                           float* out, uint32_t ring, void* stream);
    const uint32_t* verify_graph_skip() const;
    void verify_graph_publish(int layer, const float* x, const int32_t* ids, int tokens, int k,
                              float* mapped_x, int32_t* mapped_ids, uint32_t* seq, uint32_t ring, void* stream);
    void verify_graph_copy_misses(const float* mapped, float* out, int routes, void* stream);
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
                       size_t* workspace_bytes) const;

private:
    bool run_multi_impl(int layer, const float* x_f32, const int32_t* ids, int tokens, int k,
                         float* out, void* stream, bool device_io, uint8_t* cpu_rows,
                         bool skip_owned = false, const uint8_t* cpu_skip = nullptr);
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace strata::kernels
