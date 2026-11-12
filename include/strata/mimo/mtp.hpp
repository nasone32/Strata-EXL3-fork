// include/strata/mimo/mtp.hpp - the MiMo v2 3-head MTP drafter (notes/MIMO-V2-PORT.md §1 "MTP" and §2
// "MTP drafter").
//
// Three SWA-geometry heads under model.mtp.layers.{0,1,2}, each with its OWN flat KV cache (K rows
// [n_head_kv_swa][192] F16, V rows [n_head_kv_swa][128] F16 - the layout kernels::mimo_attn_decode /
// mimo_attn_prefill read) at ABSOLUTE positions.  One head cell per position i pairs the token AT i
// with the TARGET's post-model.norm state at i (same index, embedding first through eh_proj) and
// predicts the token at i+1 through the shared lm_head:
//
//   x     = eh_proj([enorm(embed(tok)); hnorm(hidden)])          (concat width 2*n_embd, RMSNorm'd
//                                                                  halves, eps = geometry rms_eps)
//   x    += o_proj(attn(norm_attn(x)))                           (pre-norm SWA attention over the
//                                                                  head's own KV: theta = rope_theta_swa,
//                                                                  n_rot, window INCLUDES the query, sink
//                                                                  RAW in the softmax denominator when the
//                                                                  head has one, sm_scale = hd_qk^-0.5,
//                                                                  v trimmed to 128 lanes)
//   x    += mlp_down(silu(mlp_gate(norm_mlp(x))) * mlp_up(...))  (dense SwiGLU)
//   state = norm_final(x)                                        (feeds the TARGET lm_head, argmax)
//
// Precision policy (mirrored by the parity test's fp64 reference): residuals accumulate in F32,
// every projection input is the F16 image of its F32 activation (the GEMM contract), weights are the
// pack's F16 tensors as loaded - the drafter applies NO load-time folds itself (an o_proj
// attention_value_scale fold is the pack's job, if the checkpoint needs one there).
//
// Chain protocol (draft): step k = 0,1,2 runs heads k..2 on the previous round token (step 0: the
// anchor) at position anchor_pos + k - head k fully (its state feeds the lm_head pick), the deeper
// heads K/V-only so their caches stay complete; EVERY step's hidden input is the anchor's target
// state.  Committed cells never need a head's full forward (the producing cells of a round are the
// speculative ones), which is why catch_up appends K/V only.
//
// Phase B hooks, per request (the engine side):
//   init(g, pack, device, capacity, max_rows, stream, err)        once per model load
//   rewind(0)                                                     between requests, and to drop an
//                                                                  aborted round's speculation
//   prompt:  catch_up(tokens, target_states, n, pos0 = committed())   per prompt chunk; the LAST
//                                                                  chunk ends at the anchor token
//   round:   draft(&anchor_tok_dev, anchor_state_dev, anchor_pos, ids[3])   anchor = the last
//            committed token (anchor_pos == committed() - 1); the target verifies the window
//            [anchor, ids[0..2]] (4 rows <= kVerifyMaxT)
//            catch_up(new_tokens, shifted_states, n_accepted + 1, pos0 = anchor_pos + 1)
//            ... the accepted drafts PLUS the bonus token (the next anchor's own cell), each with
//            the target state of the window row AFTER it - the "hidden = target states shifted by
//            one" of the notes: the token picked at window row r lands at anchor_pos+1+r and its
//            cell pairs with the state of window row r+1.  committed() then moves to
//            anchor_pos + n_accepted + 2 and the next round's anchor is committed() - 1 again.
//            draft() re-runs the anchor cell (the append is idempotent), so the prompt's catch_up
//            may end at the anchor as well.
#include "strata/artifact/mimo_pack.hpp"
#include "strata/core/mimo_geometry.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::prefill {
class Gemm;
}

namespace strata::mimo {

class MimoMtp {
public:
    MimoMtp() = default;
    ~MimoMtp();
    MimoMtp(const MimoMtp&) = delete;
    MimoMtp& operator=(const MimoMtp&) = delete;

    /// Allocates the heads' KV caches (`capacity` rows each; positions are absolute, so capacity is
    /// the context bound) and the scratch for windows of up to `max_rows` tokens, on `device`.  The
    /// multi-row projections go through a prefill::Gemm created on `stream` (null = the default
    /// stream; every launch lands there).  `pack` is borrowed and must outlive the drafter.  Invalid
    /// geometry is refused with `err`; a kernel launch that fails mid-forward reports through `err`
    /// too (the shared kernels this calls exit the process on invalid arguments, as everywhere in
    /// this tree).
    bool init(const core::MimoGeometry& g, const MimoPack* pack, int device, int64_t capacity, int max_rows,
              void* stream, std::string& err);

    /// Appends COMMITTED cells for `n` tokens at consecutive positions [pos0, pos0 + n): `tokens`
    /// (device int32 [n]) and the target's post-model.norm states `hidden_rows` (device F32 [n][n_embd],
    /// row-major), same-index pairing (cell i = token at i, hidden at i).  All heads, K/V only.  pos0
    /// must equal committed() - cells append in order; a rewind() moves that cursor.  Chunked
    /// internally by max_rows.  n = 0 is a no-op.
    bool catch_up(const int32_t* tokens, const float* hidden_rows, int n, int64_t pos0, std::string& err);

    /// One head over `rows` tokens at consecutive positions [pos0, pos0 + rows) - the primitive
    /// behind catch_up / draft, exposed for tests and for any caller that wants a committed cell's
    /// state.  kv_only = true appends K/V (eh_proj, the two input norms, k/v projections, rope) and
    /// skips the attention read, o_proj, the MLP and the final norm; the full path additionally
    /// leaves the head's final_layernorm states in head_state(head) ([rows][n_embd] F32, device).
    /// rows == 1 takes the decode path (per-projection F16 gemv + mimo_attn_decode with per-row
    /// position/length), rows > 1 the prefill path (prefill::Gemm::f16 + mimo_attn_prefill, causal
    /// inside the chunk).  Does not touch the committed cursor.
    bool head_forward(int head, const int32_t* tokens, const float* hidden_rows, int rows, int64_t pos0,
                      bool kv_only, std::string& err);

    /// Marks the cells [0, len) committed and rolls the heads back to them: with a flat
    /// absolute-position cache the rewind moves the cursor only - a stale speculative row past it is
    /// overwritten by the next append at its position before any attention can read it (an append at
    /// position p always precedes the query at p, and rows below p are committed cells).
    void rewind(int64_t len);

    /// One 3-draft round (the chain protocol above).  `anchor_token` (device int32) and
    /// `anchor_hidden` (device F32 [n_embd]) are the last committed token and its target state;
    /// anchor_pos must equal committed() - 1.  out_ids[3] (HOST memory) receives the drafts for
    /// positions anchor_pos + 1 .. +3 - the call syncs the stream once to read them back.
    /// draft_picks() keeps them on the device too.  Appends SPECULATIVE rows (the anchor cell at
    /// anchor_pos in every head, then one cell per step in the deeper heads): rewind() after the
    /// target's verify.
    bool draft(const int32_t* anchor_token, const float* anchor_hidden, int64_t anchor_pos, int32_t out_ids[3],
               std::string& err);

    /// The last head_forward(head)'s final-norm states, [rows][n_embd] F32 on the device (row 0 of
    /// head k is what draft()'s step k pushed through the lm_head).
    const float* head_state(int head) const { return head_out_ + (size_t) head * max_rows_ * n_embd_; }
    /// The last draft()'s three picks (device int32 [3]).
    const int32_t* draft_picks() const { return d_pick_; }

    int64_t committed() const { return committed_; }
    int64_t capacity() const { return capacity_; }
    int64_t n_embd() const { return n_embd_; }
    int64_t vocab() const { return n_vocab_; }
    int n_heads() const { return n_heads_; }
    int device() const { return device_; }
    void* stream() const { return stream_; }
    uint64_t vram_bytes() const { return vram_; }

private:
    /// y[rows][n_out] = W[n_out][n_in] (F16) . x[rows][n_in] (F16 image), F32 out: a plain F16 gemv
    /// at rows == 1, prefill::Gemm::f16 above that.
    bool proj(const void* w, const uint16_t* x16, float* y32, int rows, int n_out, int n_in, std::string& err);

    int64_t n_embd_ = 0, n_vocab_ = 0, inter_ = 0;
    int64_t capacity_ = 0, max_rows_ = 0, committed_ = 0;
    int n_heads_ = 0, n_q_ = 0, n_kv_ = 0, n_rot_ = 0, window_ = 0, hd_qk_ = 0, hd_v_ = 0;
    double theta_ = 0.0, sm_scale_ = 0.0, eps_ = 0.0;
    int device_ = -1;
    void* stream_ = nullptr;
    const MimoPack* pack_ = nullptr;
    prefill::Gemm* gemm_ = nullptr;   ///< new'd / deleted in mimo_mtp.cu (an incomplete type here)

    uint16_t* kv_ = nullptr;   ///< [2][heads][capacity][n_kv][hd] F16: K (hd_qk lanes) then V (hd_v)
    int64_t head_kv_stride_ = 0, plane_kv_stride_ = 0;

    float *eh32_ = nullptr, *x32_ = nullptr, *t32_ = nullptr;
    float *q32_ = nullptr, *k32_ = nullptr, *v32_ = nullptr, *attn32_ = nullptr;
    float *g32_ = nullptr, *u32_ = nullptr, *head_out_ = nullptr, *logits_ = nullptr;
    uint16_t *eh16_ = nullptr, *t16_ = nullptr, *q16_ = nullptr, *k16_ = nullptr, *v16_ = nullptr;
    uint16_t *attn16_ = nullptr, *act16_ = nullptr;
    int64_t* d_pos_ = nullptr;
    int* d_len_ = nullptr;
    int32_t* d_pick_ = nullptr;

    std::vector<void*> owned_;   ///< everything cudaMalloc'd here, freed by the destructor
    uint64_t vram_ = 0;
};

}  // namespace strata::mimo
