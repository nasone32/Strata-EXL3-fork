// include/strata/core/exl3_band8_permute.hpp - the ONE native->Band8 tile permutation.
//
// Factored out of src/artifact/exl3_band8_store.cpp so the offline packer and the
// runtime in-memory repack (Exl3ResidentBacking) share the exact same byte-level
// formula and the same four feasibility-verified rates (notes/
// EXL3-BAND8-FEASIBILITY-2026-11-06.md) instead of drifting copies. Header-only,
// dependency-free; keep it byte-identical when touching either producer.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strata {

/// The four feasibility-verified rates (K = bits + frac/2). Anything else (K1, K8,
/// unverified halves like 3.5) is refused by every Band8 producer, never silently packed.
inline bool exl3_band8_verified_rate(int bits, int frac) {
    return (bits == 2 && frac == 0) || (bits == 2 && frac == 1) ||
           (bits == 3 && frac == 0) || (bits == 4 && frac == 0);
}

/// Native -> band-8: tile (kt,nt) moves from ((kt*nj)+nt)*P to (((nt/8)*ki+kt)*8+nt%8)*P;
/// words within a tile are untouched. Pure permutation, size-preserving (the bytes are
/// identical as a multiset). `P` is uint16 words per 256-weight tile (16*K).
inline void exl3_band8_permute(const uint16_t* src, uint16_t* dst, uint32_t ki, uint32_t nj,
                               size_t P) {
    for (uint32_t nt = 0; nt < nj; ++nt) {
        const size_t band = nt >> 3, lane = nt & 7;
        const uint16_t* s = src + (size_t)nt * P;
        uint16_t* d = dst + ((band * ki) * 8 + lane) * P;
        for (uint32_t kt = 0; kt < ki; ++kt)
            std::memcpy(d + (size_t)kt * 8 * P, s + (size_t)kt * nj * P, P * 2);
    }
}

}  // namespace strata
