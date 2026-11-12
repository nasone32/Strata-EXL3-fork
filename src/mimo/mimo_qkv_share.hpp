// src/mimo/mimo_qkv_share.hpp - PROTOTYPE planning helpers (new file; not in any CMake target yet).
//
// Design companion for the two separable dense-Q8 ideas on the MiMo engine (audit
// notes/QWEN-MIMO-COMPARATIVE-AUDIT-2026-11-05.md §4/§8.3).  CPU-only, dependency-free:
// the engine integration (mimo_engine.cu) is NOT touched by this file; it encodes the
// launch PLANNING and the eligibility rules read straight out of the library sources
// (src/kernels/cuda/native_mmvq.cu, include/strata/kernels/native_mmvq.hpp) so the two
// cannot drift:
//
//   (A) decode/verify/small-M: q|k|v share ONE q8_1 quantization of the shared rmsnorm
//       input `h` (today `linear()` runs native_q8_0_f32 per projection = 3 identical
//       quantizations + 3 launches per 8-row tile), then ONE native_mmvq_bunch launch
//       (ncols 2..8) or, where the bunch is ineligible, 3x native_q8_0_mmvq on the same
//       q8_1 image.  Byte-exact vs today by the small_f32 composition identity
//       (native_mmvq.cu:1734-1746: small_f32 = native_quantize_q8_1 + small_mmvq) and
//       the bunch kernel's per-matrix identity (native_mmvq.cu:1167-1190, 2481-2512).
//
//   (B) large-M prefill: WIDE exact-layout MMVQ tiles - native_mmvq_multi_kernel
//       instantiated at NCOLS 16/32 (new opt-in library entries, default off).  A
//       column's accumulated value is provably independent of NCOLS: per element the
//       kernel runs the identical kbx -> thread mapping (BPI = F::BPI * NW / WARPS is
//       NCOLS-independent at NW = WARPS = 4), identical DP4A order and identical
//       warp-ascending + XOR-tree reduction; the NCOLS loop is innermost over an
//       element-private tmp slot (native_mmvq.cu:1194-1233, 1263-1272).  So a 32-wide
//       tile re-reads the weight block ONCE for 32 activation columns: 64 -> 16 full
//       weight traversals per projection per 512-token chunk.
//
// Scratch/buffer rules this planner assumes (caller owns stream order, as today):
//   - one q8_1 image of native_q8_1_bytes(n_in, ncols) per tile, rewritten only after
//     the previous tile's readers were enqueued earlier on the SAME stream;
//   - tile ncols <= the image's column capacity (8 today; wide needs capacity 32);
//   - per-matrix outputs keep their own row stride: y_m[(row0 + j) * n_out_m + r_m].
#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::mimo {

/// One planned MMVQ launch over a row tile of the shared input.
struct QkvTile {
    int row0 = 0;   // first token row of the tile
    int ncols = 0;  // token rows in the tile (activation columns); 1..8 bunch path, or wide
    bool wide = false;  // true: a 16/32-column exact-layout wide tile (idea B); false: <= 8 (idea A)
};

/// The launch schedule for `rows` token rows.  wide = 0: tiles of at most 8 columns
/// (the current engine loop shape).  wide = 16 or 32: whole `wide` tiles first, then the
/// remainder as <= 8-column tiles (a wide kernel template runs at its exact NCOLS, so a
/// partial wide tile is illegal; the <= 8 remainder keeps today's path and its bytes).
/// Invariants (checked in mimo_qkv_share_test.cpp): every row covered exactly once,
/// tiles ordered, ncols >= 1, wide tiles only with ncols == wide.
inline std::vector<QkvTile> qkv_tile_plan(int rows, int wide) {
    if (rows <= 0) throw std::invalid_argument("qkv_tile_plan: rows must be positive");
    if (wide != 0 && wide != 16 && wide != 32)
        throw std::invalid_argument("qkv_tile_plan: wide must be 0, 16 or 32");
    std::vector<QkvTile> tiles;
    int r = 0;
    if (wide > 0) {
        for (; r + wide <= rows; r += wide) tiles.push_back({r, wide, true});
    }
    for (; r < rows; r += 8) {
        const int nc = rows - r < 8 ? rows - r : 8;
        tiles.push_back({r, nc, false});
    }
    return tiles;
}

/// native_q8_1_bytes (native_mmvq.hpp:26, cu:2170-2177) without the library dependency:
/// 36 bytes per 32 elements per column, no padding.
inline int64_t q8_1_bytes(int n_in, int ncols) {
    return int64_t(ncols) * (n_in / 32) * 36;
}

/// Whether ONE native_mmvq_bunch launch may replace the 3 sequential MMVQ calls for this
/// tile BIT-EXACTLY vs today's native_q8_0_f32 per projection - the engine-side mirror of the
/// library's own query, kernels::native_mmvq_bunch_supported(ggml_type, n_in, ncols).  Mirrors
/// the eligibility the library itself checks (Q8_0, multi-exact layout, 2 <= ncols <= 8,
/// n_in % 32 == 0, n_in / 32 >= BPI = 32) plus the ONE divergence the library's launch cannot
/// see: on a gfx906 wave build the sequential baseline takes native_mmvq_wave_kernel while the
/// bunch always takes native_mmvq_multi_kernel - different partial-sum grouping.  The MiMo
/// target (gfx1100) never compiles that path; native_mmvq_bunch itself is NOT modified, so its
/// existing callers (the Qwen paths) keep their behaviour on every build - the ENGINE calls the
/// query and falls back to the shared-quantization + sequential path where it says no.
///
/// ncols == 1 (decode T=1) is not bunchable: the plan then shares only the quantization
/// (1x native_quantize_q8_1 + 3x native_q8_0_mmvq), still byte-exact.
inline bool qkv_bunch_eligible(int n_in, int ncols, bool multi_exact = true,
                               bool gfx906_wave_active = false) {
    if (ncols < 2 || ncols > 8) return false;
    if (n_in <= 0 || n_in % 32 != 0) return false;
    if (n_in / 32 < 32) return false;  // BPI = 2 * WARPS * WARP / 8 = 32 (SmallTraits<Q80Block, 8>)
    if (!multi_exact) return false;    // upstream layout: the bunch declines, fallback stays bitwise
    if (gfx906_wave_active) return false;
    return true;  // the Q8-packed registry never holds the MiMo pack's views (no register caller)
}

/// Whether a WIDE (idea B) exact-layout tile may run at this width BIT-EXACTLY vs the same
/// columns computed by today's <= 8-column calls - the engine-side mirror of the library's
/// wide entries' declines (kernels::native_q8_0_mmvq_wide / native_mmvq_bunch_wide, opt-in,
/// default off).  Requirements: ncols 16 or 32, the same shape rules as the bunch,
/// STRATA_TSUM off (the transposed butterfly pads to pow2(NCOLS * ROWS), so its grouping
/// CHANGES with NCOLS), the exact multi layout, and not a gfx906 wave build (the baseline's
/// kernel differs there).  Bitwise equality vs the 8-column tiles is the DESIGN and a
/// verify-before-enable property: mmvq_multi_parity's wide section must pass on the target.
inline bool qkv_wide_eligible(int n_in, int ncols, bool tsum_on = false, bool multi_exact = true,
                              bool gfx906_wave_active = false) {
    if (ncols != 16 && ncols != 32) return false;
    if (n_in <= 0 || n_in % 32 != 0) return false;
    if (n_in / 32 < 32) return false;
    if (tsum_on) return false;
    if (!multi_exact) return false;
    if (gfx906_wave_active) return false;
    return true;
}

/// Output element offset for matrix m (its own row count n_out_m) in tile t, column j:
/// the multi kernel writes y[j * n_out + row] with n_out reassigned per matrix
/// (cu:1183-1190), i.e. the engine's row-major [rows][n_out_m] buffers at
/// (t.row0 + j) * n_out_m + row.
inline int64_t qkv_out_offset(const QkvTile& t, int j, int row, int64_t n_out_m) {
    return (int64_t(t.row0 + j)) * n_out_m + row;
}

}  // namespace strata::mimo
