// include/strata/mimo/dflash.hpp - the DFlash block-diffusion speculative drafter for MiMo v2
// (notes/MIMO-V2-PORT.md §1, verified against exllamav3's architecture/dflash.py,
// modules/arch_specific/dflash.py, modules/sliding_attn.py and the real dflash-bf16 config.json).
//
// WHAT THIS IS.  A 5-layer block-diffusion encoder loaded from the plain-BF16 sibling dir
// (`<model>/dflash-bf16`, ~63 tensors + a separate `mask_embedding.safetensors`) that drafts a
// BLOCK of tokens per round instead of a chain: the input block is [embed(carry), mask_embedding
// x 7] at absolute positions P..P+7, five pre-norm GQA layers attend BIDIRECTIONALLY over a
// per-layer KV ring (window 1023 keys each side of the query, learned per-q-head sinks), and the
// proposals are argmax(TARGET lm_head) over the final-normed state rows 1..7.  After every
// target forward the committed rows' draft K/V are RECOMPUTED from target taps
// (ctx = RMSNorm(fc(concat of the 5 exported target post-block residuals))) and written into the
// ring; the draft's own speculative rows never survive a commit.
//
// WHAT IS NOT HERE (the engine's side, phase B): the target forward that produces the taps, the
// verify pass, the target embedding/lm_head (supplied as hooks below) and sampling.
//
// NUMERICS.  Weights upload as F16 (converted from the checkpoint's BF16; sinks as F32), the
// o_proj weights are pre-multiplied by attention_value_scale (0.612) exactly once AT LOAD, all
// matrix products accumulate in F32, the residual stream is F32 wide, per-head q_norm/k_norm and
// rope run in F16 storage with F32/F64 math as in kernels/mimo_attn.hpp.  Everything runs on the
// default stream (null); a stream-aware variant is a later measured change, not an assumption.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::mimo {

// ------------------------------------------------------------------------------- config (config.json)

/// Everything the drafter needs from `<dir>/config.json`.  Defaults are the real
/// dflash-bf16 values; `from_dir` overwrites them from the file.
struct DflashConfig {
    int n_layers = 5;             ///< num_hidden_layers
    int hidden = 4096;            ///< hidden_size
    int inter = 16384;            ///< intermediate_size (dense SwiGLU)
    int n_q_head = 64;            ///< num_attention_heads
    int n_kv_head = 8;            ///< num_key_value_heads
    int head_dim_qk = 128;        ///< head_dim (q and k width)
    int head_dim_v = 128;         ///< v_head_dim
    int n_rot = 64;               ///< int(head_dim_qk * partial_rotary_factor); NEOX, first n_rot
    double rope_theta = 1e4;
    double rms_eps = 1e-6;        ///< rms_norm_eps
    int block = 8;                ///< block_size: 1 carry row + block-1 mask rows
    int window = 1024;            ///< keys per query INCLUDING the query (left extent window-1)
    int window_right = 1023;      ///< keys AHEAD of the query (is_causal false -> window-1)
    bool causal = false;          ///< is_causal; true would make this a chain drafter (rejected)
    int mask_token = 151675;      ///< dflash_config->mask_token_id (informational: the drafter
                                  ///<   overwrites the mask rows with mask_embedding anyway)
    double value_scale = 0.612;   ///< dflash_config->attention_value_scale, folded into o_proj
    bool has_sinks = true;        ///< attention_sink_bias / add_swa_attention_sink_bias
    int target_hidden = 4096;     ///< target_hidden_size (each tap's width; fc in = taps * this)
    /// Target layers whose OUTPUT states are the taps, in concatenation order, tap_shift
    /// APPLIED (exllamav3's export indices; the real config pins tap_shift 0).
    std::vector<int> taps = {0, 11, 23, 35, 47};
    int page = 256;               ///< ring front-drop granularity (exllamav3 PAGE_SIZE)
    int ring = 0;                 ///< 0 = derive: ceil((window - 1 + 2 * page) / page) * page

    int ring_size() const { return ring > 0 ? ring : ((window - 1 + 2 * page) + page - 1) / page * page; }
    int q_dim() const { return n_q_head * head_dim_qk; }       ///< q_proj rows
    int kv_dim() const { return n_kv_head * head_dim_qk; }     ///< k/v_proj rows
    int fc_in() const { return (int) taps.size() * target_hidden; }  ///< fc in_features
};

/// Parses `<dir>/config.json` into a DflashConfig (no tensor is read).  `*error` is set and a
/// default config returned when the file is missing or does not describe a DFlash drafter the
/// implementation covers (model_type qwen3 + DFlashDraftModel architecture, all layers
/// sliding_attention, is_causal false, GQA group 1 or 8, head_dim 128).
///
/// Recognised keys (the real config's spellings; exllamav3's fallbacks noted):
///   top level      hidden_size intermediate_size num_hidden_layers num_attention_heads
///                  num_key_value_heads head_dim v_head_dim partial_rotary_factor rope_theta
///                  rms_norm_eps block_size sliding_window is_causal layer_types
///   dflash_config  target_layer_ids mask_token_id tap_shift (default 1, exllamav3's)
///                  attention_value_scale attention_sink_bias
///   fallbacks      mask_token_id attention_value_scale attention_sink_bias at top level,
///                  add_swa_attention_sink_bias (the real config carries this one true)
///   strata-only    dflash_ring / dflash_ring_page (scaled parity fixtures shrink the ring;
///                  the real config carries neither)
DflashConfig from_dir(const std::string& dir, std::string* error);

// ------------------------------------------------------------------------------- weights (the dir)

/// One drafter layer's device tensors.  All F16 except `sink` (F32).  Projection layouts are
/// the checkpoint's pytorch [out, in] rows, i.e. `w[o * in + i]` contiguous per output row.
struct DflashLayer {
    uint16_t* ln1 = nullptr;     ///< input_layernorm.weight [hidden]
    uint16_t* ln2 = nullptr;     ///< post_attention_layernorm.weight [hidden]
    uint16_t* q_w = nullptr;     ///< self_attn.q_proj.weight [q_dim][hidden]
    uint16_t* k_w = nullptr;     ///< self_attn.k_proj.weight [kv_dim][hidden]
    uint16_t* v_w = nullptr;     ///< self_attn.v_proj.weight [kv_dim][hidden]
    uint16_t* o_w = nullptr;     ///< self_attn.o_proj.weight [hidden][q_dim], x value_scale AT LOAD
    uint16_t* q_norm = nullptr;  ///< self_attn.q_norm.weight [head_dim_qk]
    uint16_t* k_norm = nullptr;  ///< self_attn.k_norm.weight [head_dim_qk]
    float* sink = nullptr;       ///< self_attn.attention_sink_bias [n_q_head] F32 (BF16 at rest)
    uint16_t* gate_w = nullptr;  ///< mlp.gate_proj.weight [inter][hidden]
    uint16_t* up_w = nullptr;    ///< mlp.up_proj.weight [inter][hidden]
    uint16_t* down_w = nullptr;  ///< mlp.down_proj.weight [hidden][inter]
};

/// The uploaded drafter: config + every device tensor.  Owns nothing else; `dflash_unload`
/// frees (a plain struct so the loader can report partial failures without throwing).
/// `quantized` records which dir shape was loaded (the EXL3 dir reconstructs; the bf16 dir
/// converts) - informational, for logs and tests.
struct Dflash {
    DflashConfig cfg;
    bool loaded = false;
    bool quantized = false;
    bool q8_linears = false;        ///< opt-in: layer projections use Q8_0, fc remains F16
    uint16_t* fc_w = nullptr;        ///< fc.weight [hidden][n_taps * target_hidden]
    uint16_t* hidden_norm = nullptr; ///< hidden_norm.weight [hidden]
    uint16_t* mask_embed = nullptr;  ///< mask_embedding [hidden] (separate safetensors file)
    uint16_t* final_norm = nullptr;  ///< norm.weight [hidden]
    DflashLayer* layers = nullptr;   ///< host array [n_layers] of device pointers
};

/// Loads `dir` in either of the two dir shapes the checkpoints ship:
///
///   EXL3 (PRIMARY, `dflash/`): config.json + model.safetensors + quantization_config.json,
///     where every linear is EXL3 (`<base>.trellis` I16 [in/16, out/16, 16*K], `<base>.suh`/
///     `.svh` F16, `<base>.mul1`) and the RAW tensors - hidden_norm/norm weights, the
///     layernorms, q_norm/k_norm [head_dim] and mask_embedding [hidden] as BF16, the sinks as
///     F16 - live unquantized INSIDE the same file.  The linears are reconstructed to F16 on
///     the host (kernels::cpu::exl3_reconstruct_weight, banded over 128-row blocks like
///     mimo_pack, then transposed to the [out, in] layout the drafter's gemm reads) WITHOUT
///     the o_proj fold: the trellis is PRE-FOLDED (exllamav3's converter quantizes its loaded
///     model, and load_exl3 never applies weight_scale; measured on the real pair, the
///     reconstruction lands at 0.612x of the bf16 dir's raw o_proj row).  Only the bf16 path
///     folds at load.
///
///   plain BF16 (secondary, `dflash-bf16/`): config.json + dflash_draft_model.safetensors (or
///     model.safetensors) + a SEPARATE mask_embedding.safetensors; BF16 -> F16 (sinks -> F32),
///     the same o_proj fold, everything uploaded.
///
/// Everything uploads to the CURRENT device.  Tensor shapes are validated against the config.
/// Returns false with `*error` set on any mismatch; a failed load still calls dflash_unload,
/// so the struct is reusable.
bool dflash_load(Dflash* d, const std::string& dir, std::string* error);
/// Frees every device pointer and resets `d` (safe on a partial load).
void dflash_unload(Dflash* d);

// ------------------------------------------------------------------------------- the drafter

/// The target's embedding row for one token, written to `out_row` (DEVICE, F16, hidden wide).
/// The drafter calls this once per round, for the carry token only - the mask rows use
/// mask_embedding regardless of what the target's table holds for mask_token.
using DflashEmbedFn = void (*)(int32_t token_id, uint16_t* out_row, void* user);

/// argmax over the TARGET lm_head for `n` final-normed state rows: `state_rows` is DEVICE F16
/// [n][hidden], `out_ids` DEVICE int32 [n].  The engine supplies this (its lm_head gemv +
/// argmax); tests supply their own.
using DflashArgmaxFn = void (*)(const uint16_t* state_rows, int n, int32_t* out_ids, void* user);

/// One sequence's drafter state: the per-layer K/V rings plus the window bookkeeping, and the
/// scratch buffers one round needs.  Single-sequence on purpose - the engine's verify loop is a
/// single stream, and batched drafting is a phase-B-plus decision this struct does not block.
///
/// Ring semantics (exllamav3's SWAState, notes §1): rows [window_beg, committed) hold committed
/// K/V (recomputed from taps); a draft round appends its block at [P, P + block) SPECULATIVELY;
/// commit_from_taps overwrites the accepted prefix and erases the rest by moving `committed`.
/// When a write would pass the end of the ring, whole pages drop from the front
/// (shift = ceil(needed/page)*page - ring, rounded to a page multiple).
struct DflashDrafter {
    DflashConfig cfg;
    Dflash weights_view;             ///< borrowed device pointers (NOT freed by reset/free)
    DflashEmbedFn embed_fn = nullptr;
    DflashArgmaxFn argmax_fn = nullptr;
    void* hooks_user = nullptr;

    // ring state (device: k [n_layers][ring][kv_dim], v [n_layers][ring][head_dim_v])
    uint16_t* ring_k = nullptr;
    uint16_t* ring_v = nullptr;
    int64_t window_beg = 0;          ///< absolute position of ring row 0 (page multiple)
    int64_t committed = 0;           ///< absolute position one past the last committed row

    // scratch (device); sizes fixed by init
    uint16_t* sc_blk = nullptr;      ///< [block][hidden] input block (F16)
    float* sc_x = nullptr;           ///< [block][hidden] residual stream (F32)
    uint16_t* sc_nx = nullptr;       ///< [block][hidden] normed input (F16)
    float* sc_g1 = nullptr;          ///< [block][max(q_dim, kv_dim, hidden)] F32 gemm out
    uint16_t* sc_q = nullptr;        ///< [block][q_dim] F16, rope'd in place
    uint16_t* sc_k = nullptr;        ///< [block][kv_dim] F16, k_norm'd + rope'd in place
    uint16_t* sc_v = nullptr;        ///< [block][kv_dim] F16
    float* sc_attn = nullptr;        ///< [block][n_q_head * head_dim_v] attention out (F32)
    uint16_t* sc_o_in = nullptr;     ///< [block][n_q_head * head_dim_v] F16 (o gemm in)
    float* sc_gate = nullptr;        ///< [block][inter] F32 gate
    float* sc_up = nullptr;          ///< [block][inter] F32 up
    uint16_t* sc_act = nullptr;      ///< [block][inter] F16 silu(gate) * up
    int64_t* sc_pos = nullptr;       ///< [max(block, max_commit_rows)] rope positions (H2D)
    int64_t* sc_rpos = nullptr;      ///< [block] ring-relative attention positions
    int* sc_len = nullptr;           ///< [block] attention valid-row counts
    uint16_t* sc_state = nullptr;    ///< [block][hidden] final state rows (F16)
    float* sc_cf = nullptr;          ///< [max_commit_rows][max(hidden, kv_dim)] F32 gemm out
    uint16_t* sc_ctx = nullptr;      ///< [max_commit_rows][hidden] F16 commit ctx
    uint16_t* sc_ckv = nullptr;      ///< [max_commit_rows][kv_dim] F16 commit k (norm + rope'd
                                     ///<   in place) then v
    const uint16_t** sc_taps = nullptr;  ///< [n_taps] device copy of the taps pointer array
    uint16_t* sc_shift = nullptr;    ///< [ring * max(kv_dim, n_kv * head_dim_v)] shift staging
    int max_commit_rows = 0;

    bool ready = false;
    void* q8_scratch = nullptr;
    float* sc_linear_input = nullptr;
    void* q8_stream = nullptr;
    void* q8_ready = nullptr;
    void* q8_done = nullptr;

    /// Allocates the rings and scratch for `w`'s geometry and takes the hooks.  `weights_view`
    /// keeps BORROWED pointers into `w` - keep `w` loaded for the drafter's lifetime.
    bool init(const Dflash& w, DflashEmbedFn embed, DflashArgmaxFn argmax, void* hooks_user,
              int max_commit_rows = 4096);
    /// Frees the rings and scratch (the weights are borrowed, not freed).
    void free();

    /// Drops the sequence state: ring empty, window_beg = committed = 0.
    void reset();

    /// RECOMPUTES the committed K/V rows from target taps and writes them into every layer's
    /// ring, erasing whatever the last draft round left there:
    ///   taps[t] is DEVICE F16 [rows][target_hidden], t indexing cfg.taps (concatenation order)
    ///   ctx = RMSNorm(fc(concat(taps)), hidden_norm)
    ///   per layer: k = rope(k_norm(k_proj(ctx))) and v = v_proj(ctx) at absolute positions
    ///   [position, position + rows), ring slot position - window_beg
    /// Sets committed = position + rows.  `rows` <= max_commit_rows; a longer prefill commits in
    /// chunks (the engine's prefill is chunked anyway).  Mirrors exllamav3's
    /// plan_state_rows/write_state_rows: page-granular front drop when the write would pass the
    /// ring's end, keeping what fits of the old run.
    bool commit_from_taps(const uint16_t* const* taps, int64_t position, int rows);

    /// One draft round at absolute position P (must equal committed(); the engine calls this
    /// right after a commit or a reset).  Builds the block [embed(carry), mask_embedding x
    /// block-1] at positions P..P+block-1, runs the 5 layers bidirectionally over
    /// [committed ring rows + the block's own rows], applies the final norm and calls
    /// argmax_fn on state rows 1..block-1.  `out_ids` (DEVICE, block-1 entries) receives the
    /// proposals; the full state block stays readable in last_state() until the next round.
    bool draft(int32_t carry_token_id, int64_t P, int32_t* out_ids, int block_rows = 0);

    int64_t committed_pos() const { return committed; }
    /// The rings, for inspection and for the drafter's own writes: layer l's K is
    /// ring_k + l * ring * kv_dim (rows [kv_dim] F16), V likewise
    /// ring_v + l * ring * n_kv_head * head_dim_v.
    uint16_t* ring_k_layer(int l) { return ring_k + (size_t) l * cfg.ring_size() * cfg.kv_dim(); }
    uint16_t* ring_v_layer(int l) { return ring_v + (size_t) l * cfg.ring_size() * cfg.n_kv_head * cfg.head_dim_v; }
    /// Moves ring rows [shift, cache_pos) down to [0, cache_pos - shift) in every layer through
    /// the staging buffer (the source and destination overlap, so not in place); the caller
    /// bumps window_beg by shift.  Sets *err on failure.
    void ring_shift_(int shift, int64_t cache_pos, std::string* err);
    /// The last round's final state rows (DEVICE F16 [block][hidden]); row 0 is the carry's.
    const uint16_t* last_state() const { return sc_state; }
    /// The rings, for inspection: layer l's K is ring_k + l * ring * kv_dim (rows [kv_dim] F16),
    /// V likewise ring_v + l * ring * n_kv_head * head_dim_v.
    const uint16_t* ring_k_layer(int l) const { return ring_k + (size_t) l * cfg.ring_size() * cfg.kv_dim(); }
    const uint16_t* ring_v_layer(int l) const { return ring_v + (size_t) l * cfg.ring_size() * cfg.n_kv_head * cfg.head_dim_v; }

};

}  // namespace strata::mimo
