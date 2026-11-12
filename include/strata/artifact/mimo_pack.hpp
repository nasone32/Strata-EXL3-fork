// include/strata/artifact/mimo_pack.hpp - the MiMo v2 EXL3 pack: dense tensors uploaded,
// experts described for the arena/stream.  Contract for src/artifact/mimo_pack.cpp and the
// engine driver in src/mimo.
//
// Scope (notes/MIMO-V2-PORT.md): MiMo's dense tensors (attention projections, norms, sinks,
// router gate + bias, layer-0 FFN, embed, lm_head, the 3 MTP heads) are RECONSTRUCTED to F16
// on the host at load and uploaded once - the trellis stays on disk.  Experts stay trellis and
// are served by the existing EXL3 expert machinery (Exl3ExpertStore/stream/arena) through the
// name template below.  Two load-time folds happen here, exactly once each: o_proj weights are
// multiplied by attention_value_scale (0.707; the checkpoint is NOT pre-scaled), and nothing
// else is folded (the layer-47 fp16 quirk is deliberately NOT reproduced - fp32 accumulation
// makes both sides of that fold unnecessary).
#pragma once

#include "strata/core/mimo_geometry.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata {

/// The expert tensor name the expert store streams for (layer, expert, projection).  Shared by
/// the pack validation and the store so the two cannot drift: layers are the MoE layers only.
std::string mimo_expert_tensor(int layer, int expert, const char* proj);

/// The TRELLIS-DENSE mode's per-linear view: the raw EXL3 planes on the device, for the on-device
/// dense kernels (exl3_gemv_f32 at M = 1, exl3_dense_gemv_multi's 8-row tiles at M > 1).  Valid
/// exactly when the corresponding F16 pointer in LayerDense/MimoPack is null - STRATA_MIMO_DENSE_F16=1
/// builds the reconstructed-F16 pack instead (the parity oracle; ~13.2 GiB vs ~4 GiB for the dense
/// set).  The embedding and the MTP heads stay F16 in both modes (no trellis exists for the embed;
/// the MTP module consumes F16 weights directly and is small).
struct MimoTrellis {
    void* trellis = nullptr;
    void* suh = nullptr;
    void* svh = nullptr;
    int ki = 0, nj = 0, bits = 0, cb = 2, frac = 0;
    bool valid() const { return trellis != nullptr; }
};

/// The DENSE-Q8 mode's per-linear view (--dense-q8 / STRATA_MIMO_DENSE_Q8=1): the tensor's
/// reconstructed [out x in] F16 quantized AT LOAD to GGUF Q8_0 blocks - 34 bytes per 32 weights,
/// `{ fp16 d ; int8 qs[32] }`, blocks packed row-major over [out][in/32] - the exact layout
/// `kernels::native_q8_0_f32` (the tree's MMVQ dense-Q8_0 path, llama.cpp-pinned) takes.  The S8
/// canonical form of the same values (codes + fp32 scales, code bias -128; tools/canonical_xcheck.py's
/// Q8_0 mapping) is what the s_gemv family would decode, but those kernels serve ONE activation row
/// per launch while MMVQ serves 1..8 columns in one weight read - measured 325-911 GB/s on gfx1100
/// (build-hip-exl3/q8_gemv_bench) against the trellis dense GEMV's ~180 GB/s.  Valid exactly when
/// both the F16 pointer and the trellis view are null.  The layer-0 MLP keeps its trellis planes in
/// this mode (3 x 25 MB beats 3 x 71 MB of q8 at M = 1); the embed, router gates and the MTP heads
/// stay F16 as in every mode.
struct MimoQ8 {
    void* w = nullptr;
    bool valid() const { return w != nullptr; }
};

struct MimoPack {
    struct LayerDense {
        void* q = nullptr;          ///< F16 [qk_dim x n_embd]      (row-major, out x in)
        void* k = nullptr;          ///< F16 [kv*head_dim_qk x n_embd]  (v rows zero-padded to 192)
        void* v = nullptr;          ///< F16 [kv*head_dim_qk x n_embd]  (lanes 128..191 are zeros)
        void* o = nullptr;          ///< F16 [n_embd x v_dim], attention_value_scale folded in
        void* router = nullptr;     ///< F16 [n_expert x n_embd]    (unquantized mlp.gate.weight)
        void* e_score = nullptr;    ///< F32 [n_expert]             (e_score_correction_bias)
        void* sink = nullptr;       ///< F32 [n_q_head] or nullptr  (SWA layers only)
        void* norm_attn = nullptr;  ///< F32 [n_embd]               (input_layernorm)
        void* norm_mlp = nullptr;   ///< F32 [n_embd]               (post_attention_layernorm)
        void* ffn_gate = nullptr;   ///< F16 [dense_intermediate x n_embd]  (layer 0 only)
        void* ffn_up = nullptr;     ///< F16 [dense_intermediate x n_embd]  (layer 0 only)
        void* ffn_down = nullptr;   ///< F16 [n_embd x dense_intermediate]  (layer 0 only)
        MimoTrellis tq, tk, tv, to, tffn_gate, tffn_up, tffn_down;   ///< the trellis-mode views
        MimoQ8 q8q, q8k, q8v, q8o;   ///< the dense-q8-mode views (the F16 pointers above stay null)
    };
    std::vector<LayerDense> layers;

    struct MtpHead {
        void* eh = nullptr;         ///< F16 [n_embd x 2*n_embd]  (input: [enorm(embed); hnorm(hidden)])
        void* q = nullptr;          ///< F16 [qk_dim x n_embd]    (SWA geometry: kv = n_head_kv_swa)
        void* k = nullptr;          ///< F16 [kv*192 x n_embd]
        void* v = nullptr;          ///< F16 [kv*192 x n_embd]
        void* o = nullptr;          ///< F16 [n_embd x v_dim]
        void* hnorm = nullptr;      ///< F32 [n_embd]
        void* enorm = nullptr;      ///< F32 [n_embd]
        void* norm_attn = nullptr;  ///< F32 [n_embd]             (input_layernorm)
        void* norm_mlp = nullptr;   ///< F32 [n_embd]             (pre_mlp_layernorm)
        void* norm_final = nullptr; ///< F32 [n_embd]             (final_layernorm)
        void* mlp_gate = nullptr;   ///< F16 [dense_intermediate x n_embd]
        void* mlp_up = nullptr;     ///< F16 [dense_intermediate x n_embd]
        void* mlp_down = nullptr;   ///< F16 [n_embd x dense_intermediate]
        void* sink = nullptr;       ///< F32 [n_q_head] or nullptr (per-tensor presence)
    };
    std::vector<MtpHead> mtp;       ///< n_mtp_layers entries

    void* embed = nullptr;          ///< F16 [n_vocab x n_embd]   (checkpoint is BF16; converted)
    void* lm_head = nullptr;        ///< F16 [n_vocab x n_embd]
    void* norm_final = nullptr;     ///< F32 [n_embd]             (model.norm)
    MimoTrellis tlm;                ///< the trellis-mode lm_head view
    MimoQ8 lm_q8;                   ///< the dense-q8-mode lm_head view

    uint64_t vram_bytes = 0;        ///< everything above, summed
};

/// Result of the checkpoint inventory pass: what the loader found, for logs and tests.
struct MimoInventory {
    int layers = 0;
    int moe_layers = 0;
    int experts_per_moe_layer = 0;
    int mtp_layers = 0;
    std::vector<int> trellis_k_counts;   ///< histogram: index = K * 2 (2 -> 4, 2.5 -> 5), value = tensors
    /// The ROUTED experts only, same indexing - what the expert kernels' gates are judged against
    /// (a half-integer or low-K expert set locks the RDNA run-funnel / batched twins out).
    std::vector<int> expert_k_counts;
    std::string summary() const;
};

/// Cross-checks an EXL3 dir against the geometry WITHOUT uploading anything: per-layer kv
/// widths (k_proj rows) agree with the pattern, sink tensors present on SWA layers (and absent
/// on full layers - warning, not error, if a full layer carries one), expert counts, MTP head
/// tensors, embed/lm_head.  Returns false with a human-readable `error` on the first hard
/// mismatch.  Safe to run against the real weights the day they land.
bool mimo_validate_checkpoint(const std::string& exl3_dir, const core::MimoGeometry& g,
                              MimoInventory* inventory, std::string& error);

/// Validates, reconstructs and uploads the dense tensors; experts are NOT touched (the engine
/// registers them with Exl3ExpertStore via mimo_expert_tensor).  `progress` receives one line
/// per stage if non-null.  `inventory`, when non-null, receives the validation pass's findings
/// (the caller does not re-walk the checkpoint).  Returns nullptr with `error` set on failure.
std::unique_ptr<MimoPack> mimo_pack_build(const std::string& exl3_dir, const core::MimoGeometry& g,
                                          int device, std::string& error,
                                          void (*progress)(const char*) = nullptr,
                                          MimoInventory* inventory = nullptr);

}  // namespace strata
