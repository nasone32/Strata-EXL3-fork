#pragma once
#include <hip/hip_fp16.h>
// CUDA's <cuda_fp16.h> gives host code the fp16 bitcasts too, and Strata's host-only .cpp tests use them
// (src/kernels/exl3_moe_window_bench.cpp).  ROCm's headers define __half_as_ushort & co. only in the
// clang/HIP branch of amd_hip_fp16.h; a plain g++/MSVC host compile lands in hip_fp16_gcc.h, which stops
// at __half itself.  AMD's own definition of the bitcasts is the __half_raw view, so mirror it here.
#if defined(__HIP_PLATFORM_AMD__) && !(defined(__clang__) && defined(__HIP__))
inline unsigned short __half_as_ushort(__half x) { return static_cast<__half_raw>(x).x; }
inline short __half_as_short(__half x) { return static_cast<short>(static_cast<__half_raw>(x).x); }
inline __half __ushort_as_half(unsigned short x) { return __half_raw{x}; }
inline __half __short_as_half(short x) { return __half_raw{static_cast<unsigned short>(x)}; }
#endif
