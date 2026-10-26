#pragma once

#include "strata/kernels/f16_bits.hpp"
#include <cstddef>
#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#endif

namespace strata::kernels::cpu {

// Exact batch version of f32_from_f16, including signed zero, subnormals and NaN
// payloads. SSE2 is baseline on x86-64; exceptional four-lane blocks use the scalar
// rule, so neither F16C's NaN quieting nor the CPU's floating-point modes matter.
inline void f16_to_f32_exact(const uint16_t* src, float* dst, size_t n) {
    size_t i = 0;
#if defined(__x86_64__) || defined(_M_X64)
    const auto zero = _mm_setzero_si128();
    const auto emask = _mm_set1_epi32(0x7c00);
    for (; i + 4 <= n; i += 4) {
        const auto h = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*) (src + i)), zero);
        const auto ex = _mm_and_si128(h, emask);
        const auto special = _mm_or_si128(_mm_cmpeq_epi32(ex, zero), _mm_cmpeq_epi32(ex, emask));
        if (_mm_movemask_epi8(special)) {
            for (size_t j = 0; j < 4; ++j) dst[i + j] = f32_from_f16(src[i + j]);
        } else {
            const auto sign = _mm_slli_epi32(_mm_and_si128(h, _mm_set1_epi32(0x8000)), 16);
            const auto mag = _mm_add_epi32(_mm_slli_epi32(_mm_and_si128(h, _mm_set1_epi32(0x7fff)), 13),
                                          _mm_set1_epi32(0x38000000));
            _mm_storeu_ps(dst + i, _mm_castsi128_ps(_mm_or_si128(sign, mag)));
        }
    }
#endif
    for (; i < n; ++i) dst[i] = f32_from_f16(src[i]);
}

}  // namespace strata::kernels::cpu
