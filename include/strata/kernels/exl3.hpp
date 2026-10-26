// include/strata/kernels/exl3.hpp - GPU EXL3 reconstruct (docs/EXL3.md).
//
// Reconstructs the original-basis weight W = diag(suh) . H128 . W_hat . H128 . diag(svh) from a
// packed EXL3 trellis, on the GPU.  The codebook is procedural (no lookup table), as in ExLlamaV3.
#pragma once

#include <cstdint>

namespace strata::kernels {
// One ordered verifier layer at a time. Portable mapped transport: the peer
// publishes compact fp16 rows; the primary graph scatters directly to parts.
struct Exl3PeerOutputRecord {
    uint32_t ready = 0, expected = 0, aborted = 0;
    int32_t rank[512];
};
void exl3_peer_output_signal(uint32_t* ready, uint32_t sequence, void* stream);
void exl3_peer_output_transfer(const uint16_t* source,uint16_t* target,int64_t elements,
                               uint32_t* ready,uint32_t sequence,int* arrivals,void* stream);
void exl3_peer_output_copy(const Exl3PeerOutputRecord* record, const uint16_t* rows,
                          float* parts, int routes, int embd, const uint32_t* skip,
                          uint32_t ring, void* stream);
struct Exl3Mat;
class Exl3MoeWorkspace;

// Capture-only, capacity-sized resident FFN. Null trellis descriptors mark absent routes;
// those blocks exit before reading weights. Workspace and optional runtime scratch must
// already be reserved. Device descriptors and input_rows remain device-owned throughout.
const uint16_t* exl3_moe_rows_masked(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down,
                          int routes, int n_in, int n_ff, int n_out, const uint16_t* x,
                          const int* input_rows, void* stream,
                          Exl3MoeWorkspace& workspace);

// trellis: [ki, nj, 256*bits/16] packed; suh: ki*16 fp16; svh: nj*16 fp16.  cb: 0=3inst, 1=mcg, 2=mul1.
// out: ki*16 x nj*16 fp16, original basis.  stream may be null.
void exl3_reconstruct_weight(const uint16_t* trellis, int ki, int nj, int bits, int cb,
                             const uint16_t* suh, const uint16_t* svh, uint16_t* out, void* stream);

// out[o][i] = in[i][o] (fp16, 32x32 tiles): the loader's requantization path needs the reconstructed weight
// in the [out][in] form a block quant stores.  stream may be null.
void exl3_transpose_f16(const uint16_t* in, uint16_t* out, int rows, int cols, void* stream);

// Fused decode-time GEMV: y = H(x . suh) @ W_hat . H . svh, without ever materializing W.
// x: ki*16 fp16 (one token); y: nj*16 fp16.  stream may be null.
void exl3_gemv(const uint16_t* x, const uint16_t* suh, const uint16_t* svh, const uint16_t* trellis,
               int ki, int nj, int bits, int cb, uint16_t* y, void* stream);

// The same GEMV for the engine's f32 activations (converts x f32->fp16 and y fp16->f32).  This is the
// entry point `gemv_quantized` uses for an EXL3 dense linear.  stream may be null.
void exl3_gemv_f32(const float* x, const uint16_t* suh, const uint16_t* svh, const uint16_t* trellis,
                   int ki, int nj, int bits, int cb, float* y, void* stream);

// ---- the dense multirow trellis GEMV (A9; gfx11 builds only) ----
// m = 1..8 rows of one matrix through the RDNA decode core: the input rotation runs as its own kernel
// (suh + H128 into an fp32 scratch), the dot kernel decodes B once per run and feeds every row's fp32
// accumulator set, and a counter epilogue Hadamards/scales each row.  Row r of an m-row call is
// bit-identical to an m == 1 call.  `slot` separates concurrent streams (0 main/decode, 1 the
// verifier's forked shared-expert stream); the workspace grows outside graph capture on first use.
// STRATA_EXL3_DENSE_RDNA=1 routes exl3_gemv_f32's m == 1 calls here too.
void exl3_dense_gemv_multi(int slot, const float* x, long x_stride, const uint16_t* trellis,
                           const uint16_t* suh, const uint16_t* svh, int ki, int nj, int bits, int cb,
                           int rows, float* y, long y_stride, void* stream);
// Pre-grow that slot's workspace for one (ki, nj) shape.  The engine walks every trellis role and
// reserves BEFORE any graph capture; a lazy cudaMalloc inside a capture is fatal.
void exl3_dense_mr_reserve_shape(int slot, int ki, int nj);
bool exl3_dense_mr_available();   // false on builds without the gfx11 kernels
bool exl3_dense_rdna_enabled();   // STRATA_EXL3_DENSE_RDNA

// A9's window shared expert: the decode path's fp32 swiglu (silu(g) * u, elementwise; h may alias u)
// and the native multi path's per-row sigmoid gate scale (out_row *= sigmoid(g[row])).
void exl3_silu_mul_f32(const float* g, const float* u, float* h, long n, void* stream);
void exl3_sigmoid_scale_rows(float* out, const float* g, int n, int rows, void* stream);

// One EXL3 linear's tensors (see docs/EXL3.md).  cb: 0=3inst, 1=mcg, 2=mul1.
struct Exl3Mat {
    const uint16_t* trellis = nullptr;
    const uint16_t* suh = nullptr;
    const uint16_t* svh = nullptr;
    int ki = 0, nj = 0, bits = 0, cb = 2;
};

// One expert group in an existing expert-major route plan. Offsets address the wave row table.
struct Exl3Group { Exl3Mat gate, up, down; int first = 0, rows = 0; };

// Scratch belongs to one ordered execution context. Distinct concurrent streams/devices must use
// distinct workspaces; expert weights remain shareable. Reserve before capture or timed execution.
class Exl3MoeWorkspace {
public:
    Exl3MoeWorkspace();
    ~Exl3MoeWorkspace();
    Exl3MoeWorkspace(const Exl3MoeWorkspace&) = delete;
    Exl3MoeWorkspace& operator=(const Exl3MoeWorkspace&) = delete;
    void reserve(int routes, int n_in, int n_ff, int n_out, bool indexed = false);
    // Before a verifier graph can spin: reserve optional indexed RDNA/FUSE scratch too.
    // reserve() alone deliberately leaves those buffers lazy for established callers.
    void reserve_runtime_scratch(int routes, int n_in, int n_ff, int n_out);
    // Group-path scratch only, for callers that queue many group tiles before syncing (the wave
    // barrier); `wmma_scratch` also covers the gfx11 matrix-core arm's buffer when it may run.
    void reserve_group_scratch(int routes, int n_in, int n_ff, int n_out, bool wmma_scratch);
    // True when a group tile of `routes` rows fits the current capacities (base arrays and group
    // scratch): the wave arm's no-growth-under-queued-work invariant.
    bool group_tile_fits(int routes, int n_in, int n_ff, int n_out, bool wmma_scratch) const;
    void reserve_expert_batch(int groups, int routes, int n_in, int n_ff, int n_out);
    // The PROTOTYPE grouped small-M arm's buffers (per-group descriptors + route tables), so a
    // diagnostic caller can pre-reserve before timed/captured execution. Grow-only like reserve().
    void reserve_small_m(int routes, int groups, int n_in, int n_ff, int n_out);
    size_t allocated_bytes() const; // requested device capacities, excluding runtime/BLAS allocations
    struct Impl;
    Impl& implementation();
private:
    Impl* impl_ = nullptr;
};

// One routed expert's FFN for a single token: out = down(silu(gate(x)) * up(x)), accumulated over
// `n_experts` experts weighted by `weights`.  x: ki*16 fp16; out: down[].nj*16 fp16.  stream may be null.
void exl3_moe_ffn(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down, const float* weights,
                  int n_experts, const uint16_t* x, uint16_t* out, void* stream, Exl3MoeWorkspace* workspace = nullptr);

// The same FFN for a subset of the experts, written as `k` rows of `n` fp16 into the caller's buffer (the
// resident/miss split: the resident half can run while the missing experts are still being read).
void exl3_moe_rows_part(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down, int k, const uint16_t* x,
                        uint16_t* out, void* stream, Exl3MoeWorkspace* workspace = nullptr);

// Multi-token counterpart. Matrices are HOST descriptors in route order; input_rows is a DEVICE
// array of route-count token indices. x is DEVICE [tokens, gate[0].ki*16], out is DEVICE
// [routes, down[0].nj*16]. Repeated experts share weight pointers and can consume different tokens.
void exl3_moe_rows_indexed(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down, int routes,
                           const uint16_t* x, const int* input_rows, uint16_t* out, void* stream,
                           Exl3MoeWorkspace& workspace);

// Grouped small-M window FFN - the expert GROUP is the dispatch unit, so a trellis window is
// decoded once for every row of the group instead of once per route.  Per row the float
// arithmetic is byte-identical to exl3_moe_rows_indexed on the same route order: no activation
// quantization, no DP4A, no codebook change.  Validated geometry so far: the real 3bpw experts
// (K3, cb=mul1) plus synthetic K4/mul1 and K3/mcg/3inst parity - other bitrates/codebooks are
// untested.  Contract: gate/up/down are HOST descriptors ONE PER GROUP (a group is ONE expert);
// group_first/group_rows are HOST tables of the groups as CONTIGUOUS slices of the route order
// the engine's Exl3RoutePlan emits (group_first[0]==0, monotone, sum==routes, rows in
// 1..SMALL_M_MAX_ROWS=6; larger groups must be chunked by the caller).  input_rows is the DEVICE
// per-route token table; x is DEVICE [tokens, gate[0].ki*16]; out is DEVICE [routes,
// down[0].nj*16] fp16 route-major, identical layout to the indexed entry.  Malformed tables or
// geometry throw on the host BEFORE anything launches.  The GEMV is instantiated per group
// width and the entry picks the window's widest group, so an all-singleton window runs the
// 1-row body. The unfused prototype remains diagnostic; the fused opt-in peer
// variant requires gfx11 RDNA=1/FUSE=3, mul1, and groups of 1..4 rows. It uses
// three projection launches. Pre-reserve reserve_small_m and (when fused)
// reserve_runtime_scratch before timed/captured execution or a waiting graph.
void exl3_moe_rows_grouped(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down,
                           const int* group_first, const int* group_rows, int groups,
                           int routes, const uint16_t* x, const int* input_rows, uint16_t* out,
                            void* stream, Exl3MoeWorkspace& workspace, bool fused = false);

// Pre-uploaded grouped plan for an ordered caller that owns its device buffers.
// Host descriptors/tables below are still validated; the caller guarantees the
// device plan matches them and remains immutable through stream completion.
struct Exl3MoeDevicePlan {
    const Exl3Mat *gate = nullptr, *up = nullptr, *down = nullptr;
    const int *first = nullptr, *rows = nullptr;
};
// Indexed twin: host descriptors are validated, uploaded device descriptors and
// row indices belong to the caller. Writes directly into out (no output D2D).
void exl3_moe_rows_indexed_device_plan(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down,
                            int routes, const uint16_t* x, const int* input_rows, uint16_t* out,
                            const Exl3MoeDevicePlan& plan, void* stream, Exl3MoeWorkspace& workspace);
// Returns workspace-owned route rows, valid until its next ordered FFN call.
// No plan upload or redundant output D2D copy; reserve before enqueue/capture.
const uint16_t* exl3_moe_rows_grouped_device_plan(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down,
                            const int* group_first, const int* group_rows, int groups,
                            int routes, const uint16_t* x, const int* input_rows,
                            const Exl3MoeDevicePlan& plan, void* stream,
                            Exl3MoeWorkspace& workspace, bool fused = false, uint16_t* direct_out = nullptr);

// Build and environment support for the fused grouped variant.
bool exl3_fused_multirow_supported();

// One expert, many token rows: reconstruct its three weights once and use FP16 tensor-core GEMMs
// with FP32 accumulation. For prefill outside a waiting verifier graph; may grow workspace.
// input_rows is DEVICE [rows]; out is DEVICE float [rows, down.nj*16].
void exl3_moe_group_gemm(const Exl3Mat& gate, const Exl3Mat& up, const Exl3Mat& down, int rows,
                         const uint16_t* x, const int* input_rows, float* out, void* stream,
                         Exl3MoeWorkspace& workspace);

// The same one-expert FFN on the gfx11 (RDNA3) matrix cores: the trellis is decoded straight into
// WMMA fragments, so no FP16 weight is ever materialized (the technique of nasone32's llama.cpp
// EXL3 fork, exl3_wmma_mul1_routed_v4).  Same contract as exl3_moe_group_gemm.  Returns false when
// this build/device/codebook/bitrate cannot take it (the caller falls back to the path above).
bool exl3_moe_group_wmma(const Exl3Mat& gate, const Exl3Mat& up, const Exl3Mat& down, int rows,
                         const uint16_t* x, const int* input_rows, float* out, void* stream,
                         Exl3MoeWorkspace& workspace);

// STRATA_EXL3_PEER_PREFILL's return transport (the peer tier's prompt share): for each of the
// `n` unweighted rows, sum[pair[r] / k] += weights[pair[r]] * rows[r], element-wise over `n_embd`
// columns - one weighted sum per token instead of the raw rows (k as a runtime argument; the
// eddoursul/Strata moe_scatter_add arithmetic, launched once per expert group in a fixed order so
// the sums are repeatable).  `sum` is zeroed by the caller before the first group.
void exl3_pf_scatter_add(float* sum, const float* rows, const float* weights, const int32_t* pair,
                         int64_t n, int k, int64_t n_embd, void* stream);

// Opt-in bounded subgroup: 2..8 homogeneous K3..K6/mul1 experts, >=16 rows/group, total <=4096.
// Groups are contiguous (first=0, then prefix sums) and share a row-tile regime (<24 or >=24).
// Pre-reserve reserve_expert_batch for the largest subgroup BEFORE enqueueing any work. No growth
// occurs here. Host group slices stay immutable/alive through stream completion; device descriptors
// and scratch belong to this ordered stream until its consumers finish (same-stream reuse only).
// input_rows and destinations contain total rows; callers guarantee indices are in bounds and
// destinations are unique valid rows of out. Final postscale scatters directly to destinations.
// All metadata is validated before the first upload/kernel. False means nothing was enqueued.
bool exl3_moe_expert_batch(const Exl3Group* groups, int count, const uint16_t* x,
                           const int* input_rows, const int* destinations, float* out, void* stream,
                           Exl3MoeWorkspace& workspace);

// PROTOTYPE (test-only so far, no engine caller, no performance claim): the resident window's
// hit arm for a future graph-side caller.  Descriptors are gathered on the DEVICE from per-slot
// tables (`d_*_table`, one Exl3Mat per slot) by per-route slot ids, then the existing indexed
// kernels run with no host interaction - capture-friendly UNDER THESE CONTRACTS:
//  - reserve `workspace` for this shape BEFORE building the capture (no allocation under replay);
//  - table geometry must match (n_in/n_ff/n_out) and table CONTENT stay immutable and resident
//    for every replay (the caller mirrors the cache's slots coherently);
//  - the referenced slots must be pinned against eviction until replay completion is observed,
//    exactly like the host arm's wave pinning;
//  - x/input_rows/slot ids are read per replay and must stay alive and unmodified for the replay
//    count.  An INVALID slot id never computes a wrong expert: the entry validates the ids on the
//    host and THROWS before anything launches; a captured graph is built from an already-validated
//    warm call, so replaying with a mutated, invalid id buffer is a caller contract violation.
// Misses stay on the host arm and must be merged exactly once by the caller.
void exl3_moe_rows_slot_table(const Exl3Mat* d_gate_table, const Exl3Mat* d_up_table, const Exl3Mat* d_down_table,
                              int table_slots, const int* d_slot_of_route, int routes, int n_in, int n_ff, int n_out,
                              const uint16_t* x, const int* input_rows, uint16_t* out, void* stream,
                              Exl3MoeWorkspace& workspace);

// Put the two parts' rows back in the token's order: `out[a_row[r]] = a[r]`, `out[b_row[r]] = b[r]`.
void exl3_rows_scatter(const uint16_t* a, const int* a_row, int na, const uint16_t* b, const int* b_row, int nb,
                       int n, uint16_t* out, void* stream);

// One routed expert's FFN, UNWEIGHTED, into its own row: out[i*n] = down(silu(gate(x)) * up(x)) for expert i.
// The engine's expert pool returns k rows and applies the router weights later (`moe_combine`), so this is the
// shape it needs.  x: ki*16 fp16; out: k rows of down[].nj*16 fp16.
void exl3_moe_rows(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down, const int* ids,
                   int k, const uint16_t* x, uint16_t* out, void* stream, Exl3MoeWorkspace* workspace = nullptr);

// Gather one embedding row stored as F16 (bf16=false) or BF16 (bf16=true) into f32.  The EXL3 model's
// token_embd is BF16; the engine's `embed_row` otherwise requires an S2/S4/S8 table.
void exl3_embed_gather(const uint16_t* row, int64_t n, bool bf16, float* out, void* stream);
// Device token ids; invalid ids yield zeros (the host verifier validates request tokens).
void exl3_embed_gather_rows(const uint16_t* table, int64_t n_embd, int64_t n_vocab, bool bf16,
                            const int32_t* tokens, int rows, float* out, void* stream);

// The MoE shared expert for EXL3 weights: out = (down(silu(gate(x)) * up(x))) * sigmoid(w_ginp . x).
void exl3_shared_reserve(int n_ff);
void exl3_shared_expert(const float* x, const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down,
                        const uint16_t* w_ginp, float* out, int n_embd, int n_ff, void* stream);

// Pre-allocate the persistent GEMV workspace (call before graph capture; cudaMalloc is illegal in a capture).
void exl3_gemv_reserve(int max_k, int max_n);

}  // namespace strata::kernels
