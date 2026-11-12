// include/strata/kernels/mimo_attn.hpp - MiMo v2 attention host API (contract for src/mimo).
//
// Implemented in src/kernels/cuda/mimo_attn.cu (compiled by the CUDA and HIP builds; the CPU
// reference lives beside the parity test).  notes/MIMO-V2-PORT.md has the math this follows;
// the layout choices are pinned here so the engine and the kernels cannot drift:
//
//   - KV cache per layer is FLAT (no paging): k_cache [capacity][n_kv][192] F16,
//     v_cache [capacity][n_kv][128] F16.  The checkpoint pads v rows to 192 (zero lanes
//     128..191 per head); we store the real 128 and the attention output is the trimmed
//     [64][128] = 8192-wide vector o_proj expects.
//   - Q rows are [n_q_head][head_dim_qk] F16 with RoPE applied in place on the first n_rot
//     channels (NEOX pairs (i, i + n_rot/2), fp32 rotation).
//   - sm_scale is passed in (192^-0.5 for the target model), never recomputed.
//   - SWA: a query at absolute position p attends keys in [p - (window - 1), p]; window <= 0
//     means full causal.  The sink logit (non-null sinks) joins the softmax denominator RAW -
//     never multiplied by sm_scale - as a zero-value virtual key per q head.
//   - Streams are passed as void* (cudaStream_t under CUDA, hipStream_t under HIP) exactly as
//     s_gemv.hpp does it; null means the default stream.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// In-place NEOX partial RoPE over a batch of qk rows (tokens).
///   qk        [rows][n_heads][head_dim_qk] F16
///   positions absolute position per row
///   n_rot     rotated channels (64): pairs (i, i + n_rot/2) within [0, n_rot)
void mimo_rope_f16(void* qk, const int64_t* positions, int rows, int n_heads, int head_dim_qk,
                   int n_rot, double theta, void* stream);

/// Decode / verify attention: `rows` query tokens, each at its own absolute position with its
/// own history length (verify windows pass per-row lengths).
///   q        [rows][n_q_head][head_dim_qk] F16, rope already applied
///   out      [rows][n_q_head * head_dim_v] F32, trimmed heads, pre-o_proj
///   sinks    [n_q_head] F32 or nullptr (full-attention layers have none)
///   cache_ring 0 keeps flat indexing; otherwise physical KV row = absolute position %
///              cache_ring. The caller must retain every row any query can attend.
void mimo_attn_decode(const void* q, const void* k_cache, const void* v_cache, void* out,
                      const int64_t* positions, const int* seq_lens, int rows, int n_q_head,
                      int n_kv_head, int head_dim_qk, int head_dim_v, int capacity, int window,
                      const void* sinks, double sm_scale, void* stream, int cache_ring = 0);

/// Chunk prefill: `rows` tokens at consecutive positions [pos0, pos0 + rows), causal inside
/// the chunk plus history [0, pos0).  q/out shapes as in decode.
/// Ring storage must cover min(capacity, window + rows - 1) rows, since the whole chunk
/// is appended before attention runs. Positions and RoPE remain absolute.
void mimo_attn_prefill(const void* q, const void* k_cache, const void* v_cache, void* out,
                       int64_t pos0, int rows, int n_q_head, int n_kv_head, int head_dim_qk,
                       int head_dim_v, int capacity, int window, const void* sinks,
                       double sm_scale, void* stream, int cache_ring = 0);

// --------------------------------------------------------------------- split-sequence (opt-in)
//
// OPT-IN "flash-decoding" variant of decode/prefill for small row counts over long key ranges:
// the [lo, hi] scan is partitioned across `n_splits` blocks per (row, kv head) which each
// produce an online-softmax partial (running max m_s, denominator l_s, unnormalized fp32
// accumulator acc_s[G][head_dim_v]); a second kernel launched on the SAME stream merges the
// partials in split order (deterministic, no atomics/counters/spinlocks) and joins the sink
// logit exactly once there.  Gated by STRATA_MIMO_ATTN_SPLIT at runtime:
//
//   - unset/0: both entries below fall back to the single-scan kernels above through the
//     exact same dispatch the plain entries use, so outputs are byte-identical to
//     mimo_attn_decode / mimo_attn_prefill.
//   - "1": split path; a computed split plan of 1 (short ranges, e.g. every SWA window at
//     the default 128 keys/split) also falls back, again byte-identically.
//
// The sink joins ONLY at the merge, RAW (never sm_scale'd), as a zero-value virtual key; an
// all-masked row outputs zeros with or without sinks (the softmax conventions above hold).
// Accumulation is fp32 throughout; the K/V caches are only ever read, never written.
//
// No allocation happens inside: the caller pre-reserves the workspace and passes it by
// pointer; it must stay reserved across queued/captured work that uses these entries.

/// True when STRATA_MIMO_ATTN_SPLIT is set to a non-empty value other than "0".  The engine
/// can use this to skip reserving a workspace at all; the split entries also check it.
bool mimo_attn_split_enabled();

/// Workspace descriptor for the split entries.  `mem` is one caller-owned device buffer of
/// `bytes` bytes whose base MUST already be 256-byte aligned (cudaMalloc/hipMalloc guarantee
/// this; a misaligned base is rejected, not silently fixed up - fixing it up could overrun a
/// `bytes`-exact reservation by the alignment delta).  Reserve once with the geometry's worst
/// case via mimo_attn_split_ws_bytes() and keep it alive for as long as the entries are
/// called.  The layout inside is [rows][n_kv_head][n_splits][n_q_head/n_kv_head] slots of
/// (head_dim_v + 2) fp32 per head (m, l and the head_dim_v accumulator lanes), 256-byte
/// aligned; `max_rows` and `max_splits` are the reservation bounds enforced per call.
struct MimoAttnSplitWs {
    void* mem = nullptr;          ///< device scratch, 256-byte aligned, caller-owned
    uint64_t bytes = 0;           ///< size of mem; must cover mimo_attn_split_ws_bytes(...)
    uint32_t max_rows = 0;        ///< rows must not exceed this
    uint32_t max_splits = 0;      ///< the split plan is clamped to this
};

/// Bytes to reserve for max_rows query rows of the given geometry and up to max_splits
/// splits: align256(rows * n_kv_head * max_splits * G * (head_dim_v + 2) * 4).  Valid only
/// for the compiled GQA groups G in {1, 2, 4, 8, 16} (n_q_head/n_kv_head).  Returns 0 (and
/// reports) on 64-bit overflow or invalid arguments; check before allocating.
uint64_t mimo_attn_split_ws_bytes(uint32_t max_rows, uint32_t n_kv_head, uint32_t n_q_head,
                                  uint32_t head_dim_v, uint32_t max_splits);

/// Pure-host validation of a workspace against one call's shape - no CUDA calls, no device
/// state, safe to unit-test directly.  On success returns true and writes the clamped split
/// plan (>= 1) to *n_splits_out: splits_hint > 0 wins, else the documented heuristic
/// (ceil(keys_bound/128), capped near 512 partial blocks).  On failure returns false after
/// reporting the reason: misaligned or null mem, bytes below the requirement, rows over the
/// reservation, max_rows/max_splits over INT_MAX (they are used as ints downstream), or a
/// geometry mimo_attn_split_ws_bytes rejects.  The split entries call this when the gate is
/// on and exit(1) on failure; callers may use it to validate a reservation once at startup.
bool mimo_attn_split_ws_check(const MimoAttnSplitWs* ws, uint32_t rows, uint32_t n_kv_head,
                              uint32_t n_q_head, uint32_t head_dim_v, int64_t keys_bound,
                              int splits_hint, int* n_splits_out);

/// Split-sequence decode/verify attention: the arguments are mimo_attn_decode's plus
///   keys_hint   upper bound on keys any row can attend (0: derived from capacity/window);
///               only a performance hint - the last split always absorbs any tail, so an
///               underestimate can skew the split widths but never drops keys
///   splits_hint exact split count when > 0 (tests/benches); 0 = the heuristic
///               (~128 keys/split, capped so the partial grid stays near 512 blocks)
///   ws          the pre-reserved workspace (required; no allocation happens here)
/// Gate semantics: STRATA_MIMO_ATTN_SPLIT unset/0, or a computed plan of 1 (short ranges,
/// e.g. every SWA window at the default 128 keys/split), dispatches the single-scan kernels
/// through the plain entries' exact code path - byte-identical outputs.  A null ws (or a
/// ws->mem with max_splits < 2) likewise falls back when the gate is OFF, but with the gate
/// ON a missing/misaligned/too-small workspace is a reported error (exit), never a silent
/// slow path and never a launch into kernel 0.  Needs >= 2 splits after clamping to split.
void mimo_attn_decode_split(const void* q, const void* k_cache, const void* v_cache, void* out,
                            const int64_t* positions, const int* seq_lens, int rows, int n_q_head,
                            int n_kv_head, int head_dim_qk, int head_dim_v, int capacity, int window,
                            const void* sinks, double sm_scale, int keys_hint, int splits_hint,
                            const MimoAttnSplitWs* ws, void* stream, int cache_ring = 0);

/// Split-sequence chunk prefill: mimo_attn_prefill's arguments plus keys_hint/splits_hint/ws
/// exactly as in mimo_attn_decode_split (the natural keys bound here is pos0 + rows).
void mimo_attn_prefill_split(const void* q, const void* k_cache, const void* v_cache, void* out,
                             int64_t pos0, int rows, int n_q_head, int n_kv_head, int head_dim_qk,
                             int head_dim_v, int capacity, int window, const void* sinks,
                             double sm_scale, int keys_hint, int splits_hint,
                             const MimoAttnSplitWs* ws, void* stream, int cache_ring = 0);

/// Block-diffusion attention (the dflash drafter, notes/MIMO-V2-PORT.md §1): like decode but
/// NON-CAUSAL with an explicit right extent.  A query row at position p attends keys in
///   [p - (window - 1), min(p + window_right, seq_lens[row] - 1)]
/// (window <= 0 means no left clamp; window_right < 0 would be decode, so it is rejected).
/// Positions are whatever coordinate the CACHE is indexed in - the drafter's ring passes
/// ring-relative rows (cache row = absolute position - window_beg), which keeps the q-k
/// distances of the window intact.  `seq_lens[row]` is the number of VALID cache rows, so the
/// block's own appended rows are the only keys past p a row can ever see.  Everything else -
/// layouts, device position/length pointers, the sink joining the denominator RAW - is decode's.
/// Compiled for GQA groups 8 (the drafter) and 1 (the parity fixture); anything else exits.
void mimo_attn_block(const void* q, const void* k_cache, const void* v_cache, void* out,
                     const int64_t* positions, const int* seq_lens, int rows, int n_q_head,
                     int n_kv_head, int head_dim_qk, int head_dim_v, int capacity, int window,
                     int window_right, const void* sinks, double sm_scale, void* stream);

}  // namespace strata::kernels
