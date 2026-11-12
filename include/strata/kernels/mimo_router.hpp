// include/strata/kernels/mimo_router.hpp - MiMo v2's router in one kernel (contract for src/mimo).
//
// The math is notes/MIMO-V2-PORT.md §1, and it is NOT the Flash-Next router's (native_router.hpp): MiMo scores
// with a SIGMOID, selects the top-8 by the BIASED score s_j + e_score_j (Hunyuan/Doubao "noaux_tc") and combines
// with the UNBIASED scores renormalised over the selected experts only:
//
//     logits_j = <router_j, x>          F16 inputs, F32 accumulation
//     s_j      = sigmoid(logits_j)
//     c_j      = s_j + e_score_j        the SELECTION score (e_score_correction_bias, F32)
//     ids      = top-8 of c_j, descending c_j, ties by lower expert id
//     w_i      = s_i / (sum_i s_i + 1e-20)     over the SELECTED 8 - no bias, no routed_scaling_factor
//
// routed_scaling_factor is 1 on this model and is applied by the caller if a future config says otherwise.  A
// NaN anywhere in a sigmoid score reads as 0 (never selected on its own); a NaN in the biased score reads as
// -FLT_MAX, so a NaN bias cannot poison the argmax.  Output ids are NOT sorted ascending: rank i is the i-th
// expert by descending c_j (the combine order downstream is sum, so the order only has to be deterministic).
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The shapes this kernel takes: 1..256 experts and a top-k up to 32 and to n_expert (both are runtime
/// arguments - the target model's 256/8 and the engine-parity fixture's 4/2 run the same code).
bool mimo_router_supported(int n_expert, int k);

/// Routes `rows` tokens in one launch.
///   router   F16 [n_expert x n_embd] row-major (out x in; mlp.gate.weight, unquantized)
///   x        F16 [rows x n_embd]
///   e_score  F32 [n_expert] (e_score_correction_bias)
///   ids      I32 [rows x k] out, rank i = i-th best by c_j (ties: lower expert id)
///   weights  F32 [rows x k] out, the renormalised unbiased scores
/// n_embd must be a multiple of 8 (the kernel reads uint4 = 8 halves); the F16 buffers 16-byte aligned, the
/// F32/I32 ones 4-byte; outputs disjoint from the inputs.  `stream` null means the default stream.  Throws
/// std::invalid_argument before launching when a requirement does not hold; std::runtime_error when the
/// launch itself is rejected.
void mimo_router_route(const void* router, const void* x, const void* e_score, int32_t* ids, float* weights,
                       int rows, int n_embd, int n_expert, int k, void* stream);

/// The single-token decode arm of the same math: one launch of `mimo_router_route` is ONE block
/// (8 warps) per token, which at rows == 1 leaves the whole card idle while one block reads the
/// n_expert x n_embd weight plane latency-bound (measured 141 us/layer on the target model).
/// This entry splits the same arithmetic over the card - a scores kernel with one 32-thread
/// block per expert (the identical per-warp accumulation order, so every dot is BIT-IDENTICAL
/// to the one-block kernel's), then a one-warp selection kernel with the identical top-k code -
/// and needs one extra device scratch buffer from the caller:
///   scores   F32 [n_expert] scratch (the sigmoid scores between the two kernels)
/// Same alignment/disjointness contract as `mimo_router_route` (scores counts as an output:
/// disjoint from router/x/e_score/ids/weights).  rows must be 1.
void mimo_router_route_decode(const void* router, const void* x, const void* e_score, float* scores,
                              int32_t* ids, float* weights, int n_embd, int n_expert, int k, void* stream);

/// The multi-row window arm of the same math (the verify windows, T = 2..8): the decode arm's
/// two kernels batched over the rows - a scores kernel with one 32-thread block per
/// (expert, row) and a selection block per row.  A T-row one-block launch is T blocks
/// re-reading the whole n_expert x n_embd plane (measured 6.49 ms per T=4 verify round, 47
/// layers, ~138 us per launch, vs 0.63 ms/token at T=1 on the split arm - the 2026-11-09
/// profile); this arm trades the one launch for two with n_expert * rows resp. rows blocks.
/// Every dot, the sigmoid, the NaN clamps, the biased top-k with the lower-id tie rule and
/// the renormalisation are the SAME code as `mimo_router_route`'s (the kernels are shared
/// with the decode arm), so ids and weights are BIT-IDENTICAL to the one-block kernel's on
/// the same inputs.  Two launches instead of one is the cost.
///   scores   F32 [rows x n_expert] scratch (row-major; the sigmoid scores between the kernels)
/// Same alignment contract; disjointness is STRICTER than the decode entry's documentation:
/// scores must ALSO be disjoint from router/x/e_score (it is a kernel-1 output racing those
/// reads - the decode entry's validation does not check this, this one does).  rows must be
/// 1..8 (the verify-window envelope; the engine's T=1 decode keeps `mimo_router_route_decode`).
void mimo_router_route_window(const void* router, const void* x, const void* e_score, float* scores,
                              int32_t* ids, float* weights, int rows, int n_embd, int n_expert, int k,
                              void* stream);

}  // namespace strata::kernels
