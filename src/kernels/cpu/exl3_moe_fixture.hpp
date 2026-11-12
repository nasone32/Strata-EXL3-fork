// src/kernels/cpu/exl3_moe_fixture.hpp - TEST/FIXTURE-ONLY band-8 pack entry point.
//
// Declares the one host utility fixture code needs to materialize canonical band-8 trellises
// (notes/results/2026-11-06-cpu-banded-format has the byte-exact contract; the definition in
// src/kernels/cpu/exl3_moe.cpp reuses the file-local repack helper, so fixture bytes can
// never drift from the layout the kernels consume).  NOT installed, NOT part of the public
// API, and never called by the service itself: production consumes canonical bytes from the
// CALLER (Exl3CpuTrellisLayout::Band8); there is deliberately no inverse (unpack), so no
// hidden conversion can be smuggled behind a "service".
#pragma once

#include <cstdint>

namespace strata::kernels::cpu {

/// Pack one projection's trellis (native order, (k/16)*(n/16)*(16*bits+8*hb) u16 words) into
/// the canonical band-8 order, in place-safe separate storage of the same size.  Pure tile
/// permutation; suh/svh are untouched.
void exl3cpu_fixture_pack_band8(uint16_t* dst, const uint16_t* trellis, int k, int n, int bits, int hb);

}  // namespace strata::kernels::cpu
