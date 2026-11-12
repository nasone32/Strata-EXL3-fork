#pragma once

#include <cstddef>

namespace strata::kernels {

// Native GGUF Q2_0, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL
// and IQ4_XS / CUDA Q8_1 adapters, pinned to llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d, sm_120 generic MMVQ.
// All pointers are device pointers, at least 4-byte aligned, with no overlap.
// All calls enqueue on the explicit non-null CUDA stream; no allocation or wait.
// The translation unit must use --use_fast_math, as the pinned CUDA oracle does.
//
// Shapes use GGUF order: n_in is the contiguous reduction dimension, n_out is
// the weight row count, and ncols is the activation column/token count, 1..8
// (plan v0.3 P3). Columns are contiguous: activation column j is x + j * n_in
// (its Q8_1 blocks at j * n_in / 32) and output column j is y + j * n_out. See
// native_mmvq_set_multi_exact for how ncols > 1 relates to ncols == 1.
// n_in must be a positive multiple of 32 for quantization/Q4_0/Q5_0/Q8_0/IQ4_NL,
// 64 for Q2_0, or 256 for the other formats. n_out must be positive. Weights remain
// unmodified row-major GGUF blocks: Q3_K=110, Q4_K=144, Q5_K=176, Q6_K=210 and
// IQ4_XS=136 bytes per 256 elements; Q2_0 is 18 bytes per 64 elements.
// Q4_0/IQ4_NL=18, Q5_0=22, Q8_0=34 bytes per 32 elements.
// Q8_1 scratch has 36 bytes per 32 elements, with no extra row padding here.
// Input floats must be finite, and their block scales/sums representable in FP16.
std::size_t native_q8_1_bytes(int n_in, int ncols = 1);

// Layout for ncols > 1. false: llama.cpp's generic multi-column table (upstream), equal to ncols == 1 to
// float rounding, speed not yet measured. true (default): the ncols == 1 layout, every column bitwise equal to a
// single-column call. Set before
// graph capture; captured graphs keep the kernels they captured.
void native_mmvq_set_multi_exact(bool exact);
bool native_mmvq_multi_exact();

// One quantization may serve multiple weight matrices sharing the same input.
// Q8_1 stores FP16 scale and FP16 warp sum of the ORIGINAL float inputs; it does
// not reconstruct that sum from the quantized integers.
void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols,
                          void* stream);

// P2's companion: the SAME quantize kernel and bytes for one 16- or 32-column image.  Every
// pre-existing entry keeps validate_shape's 1..8 cap untouched, so the wide tiles quantize
// through this separate entry instead (exactly 16 or 32; n_in's rules as ever; n_in * ncols
// must fit int, the kernel's index type).  Columns contiguous as always - a wide image's
// column j is bit-identical to a <= 8-column call's column j (mmvq_multi_parity checks the bytes).
void native_quantize_q8_1_wide(const float* x, void* x_q8_1, int n_in, int ncols,
                               void* stream);
// native_q8_1_bytes for the wide image (the same 36 bytes per 32 elements per column; the original
// helper's 1..8 column check stays as it is).
std::size_t native_q8_1_wide_bytes(int n_in, int ncols);

void native_swiglu_quantize_q8_1(const float* gate, const float* up, void* x_q8_1,
                                 int n_in, int ncols, void* stream);

void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

// Convenience composition: caller owns scratch sized by native_q8_1_bytes.
void native_q5_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q2_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q3_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_iq4_xs_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q4_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream);

void native_q6_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream);

void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_q4_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_q5_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

/// STRATA_Q8_PACKED=1 (opt-in): a lossless load-time repack of a Q8_0 matrix into a qs plane (n_out x n_in int8)
/// followed by a d plane (n_out x n_in / 32 fp16), the same bytes (34 per 32 values). A registered matrix's
/// native_q8_0_mmvq calls (keyed by its GGUF-layout device pointer, which stays valid for the prompt path) read the
/// packed copy instead; every output is bitwise equal to the GGUF-layout kernels.
bool native_q8_0_packed_enabled();
bool native_q8_0_packed_eligible(int n_in, int n_out);
void native_q8_0_pack_host(const void* gguf_blocks, void* out, int n_in, int n_out);
void native_q8_0_packed_register(const void* gguf_weights, const void* packed, int n_in, int n_out);
void native_q8_0_packed_unregister(const void* gguf_weights);

/// STRATA_Q6_PACKED=1 (opt-in): a packed copy of a Q6_K matrix (the output heads) - the same bytes as ql / qh /
/// scales / d planes - that native_q6_k_mmvq calls on `weights` read instead (bitwise equal outputs). The copy is
/// owned here: native_q6_k_pack builds it from the device matrix (false: not eligible or failed; nothing changes),
/// native_q6_k_unpack frees it. STRATA_Q6P_SELFTEST=1 checks it bitwise and times it at load.
bool native_q6_k_packed_enabled();
bool native_q6_k_pack(const void* weights, int n_in, int n_out, const char* what);
void native_q6_k_unpack(const void* weights);

void native_q8_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream);

void native_iq4_nl_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream);

// Storage/dispatch helpers take stable GGML type IDs, avoiding a ggml runtime
// dependency in the engine: Q4_0=2, Q5_0=6, Q8_0=8, Q3_K=11, Q4_K=12, Q5_K=13,
// Q6_K=14, IQ4_NL=20, IQ4_XS=23, Q2_0=42. Unsupported IDs throw in the byte-count
// and launch helpers; only the
// capability query returns false.
bool native_mmvq_supported(int ggml_type) noexcept;
std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out);
void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream);
/// S26 STRATA_LFUSE: native_mmvq(w1 -> y1) and native_mmvq(w2 -> y2), same type / shape / input, in ONE launch,
/// bitwise the two calls. Returns false (nothing launched) where that is not the case (Q8_0, 2-8 columns only).
bool native_mmvq_pair(int ggml_type, const void* w1, const void* w2, const void* x_q8_1, float* y1, float* y2,
                      int n_in, int n_out, int ncols, void* stream);
/// A11 STRATA_MMVQ_BUNCH: native_mmvq(w1 -> y1), (w2 -> y2) and optionally (w3 -> y3) - DIFFERENT n_out widths
/// on one shared q8_1 input - in ONE launch, bitwise the sequential calls. n_out3 <= 0 (or w3/y3 null) is a
/// two-matrix bunch. Returns false (nothing launched) where the bundle is not eligible (Q8_0, 2-8 columns only).
bool native_mmvq_bunch(int ggml_type, const void* w1, const void* w2, const void* w3, const void* x_q8_1,
                       float* y1, float* y2, float* y3, int n_in, int n_out1, int n_out2, int n_out3,
                       int ncols, void* stream);

/// Caller-side gate for bunch users that must stay BITWISE the sequential native_mmvq calls (the
/// native_q8_0_f32 baseline).  True exactly where native_mmvq_bunch may launch AND the sequential path on
/// this build runs the same multi kernels: false on a STRATA_HIP_GFX906 build with the wave kernels active
/// (STRATA_MMVQ_WAVE != "0"), where the baseline runs native_mmvq_wave_kernel's 64-lane butterfly while the
/// bunch runs native_mmvq_multi_kernel's LDS/XOR-tree reduction - different partial-sum grouping.  Also
/// mirrors the bunch's own shape rules (Q8_0, the exact multi layout, 2-8 columns, n_in % 32 == 0 and
/// n_in / 32 >= BPI); a Q8_0-packed-registered weight is left to the bunch call itself (it declines).
/// native_mmvq_bunch is NOT modified - its existing callers keep their behaviour on every build.
bool native_mmvq_bunch_supported(int ggml_type, int n_in, int ncols);

// ---- P2 wide-column MMVQ (opt-in, default off in every caller; validate_shape's 1..8 cap is unchanged) ----
//
// The exact-layout multi kernel at NCOLS = 16 or 32: the same q8_1 bytes, the same DP4A integer dots, the
// same per-block F32 products and the same per-element accumulation/reduction order as the ncols <= 8
// launches - one weight traversal serving 16/32 activation columns (columns j at x_q8_1 + j * n_in / 32,
// outputs at y + j * n_out, as ever).  These entries accept ONLY 16 or 32, keep the baseline's geometry
// (4 warps, two rows per block, odd n_out tails guarded per row) and enqueue NOTHING when they return
// false - eligibility is decided first, so a declined call leaves the caller's scratch and outputs
// untouched.  Declines: STRATA_TSUM=1 (the transposed butterfly pads to pow2(NCOLS * ROWS) - not bitwise
// across widths), the upstream multi layout (native_mmvq_set_multi_exact(false)), a STRATA_HIP_GFX906
// build with the wave kernels active, a Q8_0-packed-registered weight (STRATA_Q8_PACKED), a NULL or
// misaligned pointer (the library's 4-byte device-pointer contract; checked before the registry lookup),
// a NULL stream (the library's explicit non-null stream contract - the default stream is not accepted),
// n_out <= 0, and a two-rows-per-block grid that does not fit dim3.x / int when summed in 64 bits
// (n_out3 < 0 is rejected: 0 or positive only).  Bitwise equality with the <= 8-column launches is the
// DESIGN (the per-element order is NCOLS-independent) and a VERIFY-BEFORE-ENABLE property: this TU
// compiles with fast math, and a compiler may contract differently under register pressure
// (NCOLS = 32 keeps 64 accumulator registers) - run mmvq_multi_parity's wide section on the target
// before turning any caller on.  The caller owns one q8_1 image of native_q8_1_bytes(n_in, ncols).

/// One Q8_0 matrix, 16 or 32 PRE-QUANTIZED columns (native_quantize_q8_1's image).
bool native_q8_0_mmvq_wide(const void* weights, const void* x_q8_1, float* y,
                           int n_in, int n_out, int ncols, void* stream);
/// The same launch for a caller that also owns the quantization: eligibility first, then
/// native_quantize_q8_1 + the wide kernel; a false enqueues NOTHING (scratch untouched).
bool native_q8_0_f32_wide(const void* weights, const float* x, void* scratch_q8_1,
                          float* y, int n_in, int n_out, int ncols, void* stream);
/// The q|k|v bundle at 16 or 32 columns - DIFFERENT n_out widths on one shared q8_1 image, one launch,
/// bitwise the sequential per-matrix wide calls (n_out3 <= 0 or w3/y3 null: a two-matrix bunch).  Gated as
/// above; NCOLS = 32's register profile is unmeasured - a parity/test surface until a GPU A/B picks a width.
bool native_mmvq_bunch_wide(int ggml_type, const void* w1, const void* w2, const void* w3, const void* x_q8_1,
                            float* y1, float* y2, float* y3, int n_in, int n_out1, int n_out2, int n_out3,
                            int ncols, void* stream);

/// The wide entries' eligibility as a PURE query (no device, no enqueue): native_q8_0_mmvq_wide's gates
/// minus the pointer checks - ncols 16/32, n_in's shape rules, the exact multi layout, STRATA_TSUM off,
/// not a gfx906 wave build (the packed registry stays per-pointer, with the launch calls themselves).
/// Lets a caller skip the harmless-but-nonzero wide quantize it would otherwise enqueue before a
/// refused launch (flag-on-unsupported would pay an unused quantize per tile without this).
bool native_q8_0_wide_supported(int n_in, int ncols);

/// The wide kernels' ACTUAL compiled resources, read from the loaded code object
/// (cudaFuncGetAttributes): vgprs, shared-memory bytes and max threads per block for NCOLS 16/32
/// (ncols = 8 queries the <= 8 baseline's exact-layout ROWS = 2 kernel).  A bench/diagnostic surface
/// for the width choice - no engine path calls it.
void native_q8_0_wide_resources(int ncols, int* vgprs, int* smem_bytes, int* max_threads);

// Fork F4 (Eddoursul): 2..4 columns also INTERLEAVED, so one 16-byte load reads a position of every column: CP = 2 (2
// columns) or 4 (3-4), zero-padded; nb = n_in / 32 blocks; int32 bm[nb][8][CP] (block-major), int32 pm[8][nb][CP]
// (position-major), float d[nb][CP] (each block's scale as the dots read it, __low2float of ds).
constexpr int native_q8_1_il_cp(int ncols) { return ncols <= 2 ? 2 : ncols <= 4 ? 4 : 8; }
std::size_t native_q8_1_il_bytes(int n_in, int ncols);
// Writes the interleaved copy of 2..4 columns of plain q8_1 blocks (native_quantize_q8_1's, or a fused producer's)
// to x_il (native_q8_1_il_bytes): the same values in the same bytes.
void native_q8_1_interleave(const void* x_q8_1, void* x_il, int n_in, int ncols, void* stream);
// True when native_mmvq_il runs its own kernel for this call (IQ4_XS, Q4_K, Q5_K, Q6_K, 2-4 columns, the exact layout).
bool native_mmvq_il_supported(int ggml_type, int ncols, int n_out);
// native_mmvq for 2..4 columns whose copy native_q8_1_interleave wrote, bitwise native_mmvq's output: kernels in which a
// warp takes 1, 2 or 4 rows and reads the columns from the interleaved copy; native_mmvq (x_q8_1) when not supported.
void native_mmvq_il(int ggml_type, const void* weights, const void* x_q8_1, const void* x_il, float* y, int n_in,
                    int n_out, int ncols, void* stream);
// The table's rows a warp (0, 1, 2, 4; 0 = native_mmvq's kernels) for a card of compute capability `cc` (major * 10 + minor).
// Per-architecture tables plus STRATA_MMVQ_IL_ROWS; see native_mmvq.cu and docs/MMVQ_IL_TABLE.md.
int native_mmvq_il_rows_for(int cc, int ggml_type, int ncols, int n_out);
// Tests and benchmarks: every native_mmvq_il call takes `rows` a warp (0: the table).
void native_mmvq_il_tune(int rows);

} // namespace strata::kernels
