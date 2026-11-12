// include/strata/kernels/cpu/exl3.hpp - CPU reference kernels for the EXL3 quantized format
// (turboderp/ExLlamaV3).  See docs/EXL3.md.  These are the portable/scalar reference the HIP kernels
// are checked against; the arithmetic (codebook, 16-bit sliding windows, tile permutation, Hadamard)
// is bit-for-bit the same as tools/exl3 (the verified numpy specification).
#pragma once

#include <cstdint>

namespace strata::kernels::cpu {

// ExLlamaV3 codebook ids: 0 = 3inst, 1 = mcg, 2 = mul1.
enum class Exl3Codebook : int { ThreeInst = 0, Mcg = 1, Mul1 = 2 };

// Fill `lut` (65536 entries) with the fp16 bit pattern of each 16-bit window's codebook value.
void exl3_codebook_lut(Exl3Codebook cb, uint16_t* lut);

// Half-integer bitrates (K = bits + 0.5, exllamav3 frac.cuh): `frac` selects the alternating
// KA / KA+1-bit trellis steps (MASK 0xAAAA) and grows each tile to 16*K uint16 words.  0 (the
// default, and every integer-K file) keeps the historical behavior exactly.

// Decode one 16x16 tile: `trellis_words` holds 16*K little-endian uint16 words, returns 256
// fp16 values in row-major tile order (the tensor-core interleave is undone here).  `lut` is from
// exl3_codebook_lut.
void exl3_decode_tile(const uint16_t* trellis_words, int bits, const uint16_t* lut,
                      uint16_t* out_row_major, int frac = 0);

// Reconstruct W_hat (ki*16 x nj*16, row-major fp16) from the packed trellis [ki, nj, 16*K].
void exl3_decode_weight_hat(const uint16_t* trellis, int ki, int nj, int bits, Exl3Codebook cb,
                            uint16_t* out, int frac = 0);

// Reconstruct the original-basis weight: W = diag(suh) . H128 . W_hat . H128 . diag(svh), fp16.
// suh has ki*16 entries, svh has nj*16, both fp16 bit patterns.
void exl3_reconstruct_weight(const uint16_t* trellis, int ki, int nj, int bits, Exl3Codebook cb,
                             const uint16_t* suh, const uint16_t* svh, uint16_t* out, int frac = 0);

// Decode-time GEMV without materializing W: y[t] = H(x[t] . suh) @ W_hat . H . svh, in fp32.
// x is (tokens, ki*16) fp16; y is (tokens, nj*16) fp32.
void exl3_folded_gemv(const uint16_t* trellis, int ki, int nj, int bits, Exl3Codebook cb,
                      const uint16_t* suh, const uint16_t* svh, const uint16_t* x, int tokens,
                      float* y, int frac = 0);

// EXL3 n-gram table row (the `exl3_ngram_trellis` ring): one 160-wide tail-biting ring over the mul1
// codebook plus an fp16 scale, decoded to `dim` floats.  `lut` is the mul1 codebook (from
// exl3_codebook_lut(Mul1)).  `bias` is this row's head bias (dim floats) or null.  See docs/EXL3.md.
void exl3_ngram_decode_row(const uint16_t* ring, int K, const uint16_t* lut, int dim,
                           const float* bias, float* out);

}  // namespace strata::kernels::cpu
