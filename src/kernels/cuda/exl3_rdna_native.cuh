// RDNA3 low-K expert GEMV, adapted from the register-prefetch / raw mul1 dot2
// design in ExllamaV3-rocm exl3_rdna3_kernel.cuh. Included inside exl3.cu's namespace.
//
// MIT License
// Copyright (c) 2025 Turboderp
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#if defined(STRATA_WMMA_GFX11)
namespace rdna_native {
using HalfPair = _Float16 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(HalfPair, a),
                                  __builtin_bit_cast(HalfPair, b), c, false);
}
__device__ __forceinline__ uint32_t pair(float a, float b) {
    return (uint32_t) __half_as_ushort(__float2half_rn(a)) |
           ((uint32_t) __half_as_ushort(__float2half_rn(b)) << 16);
}
__device__ __forceinline__ uint32_t hash16(uint32_t x) {
    uint32_t hi, product;
    asm("v_pk_mul_lo_u16 %0, %1, %2 op_sel_hi:[0,1]" : "=v"(hi) : "v"(x), "s"(0x83DC0000u));
    asm("v_mad_u32_u16 %0, %1, %2, %3" : "=v"(product) : "v"(x), "s"(0xD12Du), "v"(hi));
    return product;
}
template <bool FastHash>
__device__ __forceinline__ uint32_t raw_pair(uint32_t a, uint32_t b) {
    const uint32_t ha = FastHash ? hash16(a) : a * 0x83DCD12Du;
    const uint32_t hb = FastHash ? hash16(b) : b * 0x83DCD12Du;
    const uint32_t s = __builtin_amdgcn_sad_u8(ha, 0u, 0x64006400u);
    return __builtin_amdgcn_sad_hi_u8(hb, 0u, s);
}

template <bool Frac, bool Band8 = false>
__device__ __forceinline__ void unit(const Exl3Mat& m, const float* xh, float* out,
                                    int parts, float* xs) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int e = blockIdx.z, part = blockIdx.y;
    const int k = m.ki * 16, n = m.nj * 16;
    // Whole H128 blocks, matching the input/output Hadamard boundaries.
    const int kb0 = (m.ki / 8) * part / parts, kb1 = (m.ki / 8) * (part + 1) / parts;
    const int kt0 = kb0 * 8, kt1 = kb1 * 8;
    const int nt = blockIdx.x * 8 + wave;
    for (int i = threadIdx.x; i < (kt1 - kt0) * 16; i += blockDim.x)
        xs[i] = xh[(size_t) e * k + kt0 * 16 + i];
    __syncthreads();
    constexpr int words = Frac ? 20 : 16;
    const auto* tr = reinterpret_cast<const uint32_t*>(m.trellis);
    uint32_t pf0[4], pf1[4], pf2[4], pf3[4];
    auto load = [&](int kt, uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
        a = b = c = d = 0;
        if (kt >= kt1 || nt >= m.nj) return;
        const uint32_t* tw;
        if constexpr (Band8)
            tw = tr + (size_t) blockIdx.x * m.ki * 8 * words + wave * words + (size_t) kt * 8 * words;
        else tw = tr + ((size_t) kt * m.nj + nt) * words;
        if constexpr (Frac) {
            // Two four-window runs per lane. Each span fits in 24 bits at K2.5,
            // so four coalesced dword loads replace sixteen cross-lane gathers.
            const int first = lane * 20 + 642, last = first + 8;
            a = tw[((first - 16) >> 5) % 20]; b = tw[((last - 1) >> 5) % 20];
            c = tw[((first - 6) >> 5) % 20]; d = tw[((last + 9) >> 5) % 20];
        } else {
            const int i = lane >> 1;
            a = tw[(i + 15) & 15]; b = tw[i];
        }
    };
#pragma unroll
    for (int p = 0; p < 4; ++p) load(kt0 + p, pf0[p], pf1[p], pf2[p], pf3[p]);
    float acc0 = 0, acc1 = 0, sum = 0;
    for (int kt = kt0; kt < kt1; ++kt) {
        const int at = (kt - kt0) & 3;
        const uint32_t a = pf0[at], b = pf1[at];
        const uint32_t c = pf2[at], d = pf3[at];
        load(kt + 4, pf0[at], pf1[at], pf2[at], pf3[at]);
        __builtin_amdgcn_sched_barrier(0);
        uint32_t windows[8];
        if constexpr (!Frac) {
            const uint32_t bits = __builtin_amdgcn_alignbit(a, b, ((~(lane << 3)) & 8) << 1);
#pragma unroll
            for (int j = 0; j < 8; ++j) windows[7-j] = (bits >> (2*j)) & 0xffffu;
        } else {
            const int end0 = lane * 20 + 650, end1 = end0 + 10;
            const int shift0 = (((end0 - 1) >> 5) + 1) * 32 - end0;
            const int shift1 = (((end1 - 1) >> 5) + 1) * 32 - end1;
            const uint32_t runs[2] = {__builtin_amdgcn_alignbit(a, b, shift0),
                                      __builtin_amdgcn_alignbit(c, d, shift1)};
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                windows[g*4] = (runs[g] >> 8) & 0xffffu;
                windows[g*4+1] = (runs[g] >> 5) & 0xffffu;
                windows[g*4+2] = (runs[g] >> 3) & 0xffffu;
                windows[g*4+3] = runs[g] & 0xffffu;
            }
        }
        uint32_t w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            w[j] = raw_pair<Frac>(windows[2*j], windows[2*j+1]);
            if constexpr (Frac) {
                union { uint32_t u; __half2 h; } value;
                value.u = w[j];
                value.h = __hfma2(value.h, __half2half2(__ushort_as_half((uint16_t) 0x1eee)),
                                          __half2half2(__ushort_as_half((uint16_t) 0xc931)));
                w[j] = value.u;
            }
        }
        const int i = (kt - kt0) * 16 + (lane & 3) * 2;
        const uint32_t x0 = pair(xs[i], xs[i+1]), x1 = pair(xs[i+8], xs[i+9]);
        acc0 = dot2(w[0], x0, acc0); acc0 = dot2(w[1], x1, acc0);
        acc1 = dot2(w[2], x0, acc1); acc1 = dot2(w[3], x1, acc1);
        if constexpr (!Frac) {
            sum += __half2float(__ushort_as_half((uint16_t) x0)) + __half2float(__ushort_as_half(x0 >> 16)) +
                   __half2float(__ushort_as_half((uint16_t) x1)) + __half2float(__ushort_as_half(x1 >> 16));
        }
    }
#pragma unroll
    for (int delta = 1; delta <= 2; delta <<= 1) {
        acc0 += __shfl_xor_sync(0xffffffffu, acc0, delta);
        acc1 += __shfl_xor_sync(0xffffffffu, acc1, delta);
        if constexpr (!Frac) sum += __shfl_xor_sync(0xffffffffu, sum, delta);
    }
    if constexpr (!Frac) {
        acc0 = fmaf(acc0, 0.00676727294921875f, sum * -10.3828125f);
        acc1 = fmaf(acc1, 0.00676727294921875f, sum * -10.3828125f);
    }
    if ((lane & 3) == 0 && nt < m.nj) {
        float* dst = out + ((size_t) e * parts + part) * n + nt * 16 + lane / 4;
        dst[0] = acc0; dst[8] = acc1;
    }
}

template <bool Frac, bool Band8 = false>
__global__ __launch_bounds__(256) void gemv(const float* xh, const Exl3Mat* mats, float* out, int parts) {
    extern __shared__ float xs[];
    const Exl3Mat m = mats[blockIdx.z];
    if (!m.trellis || (m.frac != 0) != Frac) return;
    unit<Frac, Band8>(m, xh, out, parts, xs);
}

// The GROUPED twin of gemv: the block's z unit is a GROUP (one expert, first..first+rows
// routes) instead of one route, so the trellis words are prefetched, the windows extracted
// and the packed mul1 pair decoded ONCE per (k tile, column tile, lane) and the resulting
// half2 weights are applied to every LIVE row with the indexed native kernel's own dot2
// arithmetic.  The decode core (load lambda, window extraction incl. the K2.5 half-integer
// geometry, raw_pair + the frac hfma2 scale/bias, the !Frac x-sum correction constants and
// the quad butterfly) is unit<Frac>'s code verbatim - only the activation source (an fp16
// pair slice per row instead of unit's on-the-fly pack of one float row; the pair() rounding
// of the SAME float value, so the bits the dot2s consume are unchanged), the accumulator
// count (2 + the !Frac sum per live row, <= 4 rows) and the output loop (one partials row
// per live route) are new.  Per-row accumulation ORDER therefore matches the indexed native
// kernel's lane/quad structure; it is NOT the scalar twin's (i, r, t) FMA order and no byte
// identity between the arms is claimed - parity is measured in exl3_moe_multi_parity.
//
// Grid: (nj/8 column tiles, ps k partitions, groups); 256 threads = 8 waves of one 16-col
// tile.  Dynamic shared = width * widest partition * 2 bytes (fp16 rows; width = the
// compiled row count 1/2/4 >= the window's widest group).  Fused (the peer's FUSE=3 form)
// folds the prescale Hadamard into the prologue (per-row butterfly, writing the fp16
// slice; FusedSilu is the down projection's silu(g)*up product) and STOPS at the partials:
// the output stage is the PROVEN postscale_grouped_kernel, not a last-arrival epilogue.  A
// fused epilogue on the per-(group, 128-column segment) counters was built and REMOVED on
// evidence: even with fence.sc + acq_rel arrivals + volatile partials, gfx1100 still showed
// intermittent one-segment tears where the reducer saw one partition's stores only (the
// fused scalar twin's identical arrival pattern has not shown this in its proven shapes -
// the native core's different instruction mix widens the window).  With postscale doing
// the reduce, no block of this kernel ever reads another block's writes: two launches per
// projection instead of the scalar twin's one, but prescale's global x round-trip still
// disappears.  Mixed-rate windows launch the <false> and <true> instantiations together;
// each returns before any shared state for the groups that are not its rate.
// KBits extends the integer-rate decode beyond K2 (2026-11-05, Qwen3.8-Flash-Next EXL3
// 3bpw/4bpw whose experts are pure K3/K4 mul1): the lane->position assignment is the SAME
// sequential 8L..8L+7 for every rate (proven for K2/K2.5 by simulation), but a K>=3 lane's
// eight 16-bit windows span 7K+16 ring bits (37 at K3, 44 at K4), no longer one funnel, so
// the K3/K4 arm splits them into the SAME two runs of four the K2.5 geometry uses.  Run g
// covers positions 8L+4g..+3: its four windows live in ring bits [K*(8L+4g+1)-16,
// K*(8L+4g+4)), a 3K+16-bit span (<= 28 bits at K4), so ONE alignbit funnel from the two
// span dwords decodes the whole run with window j at shift K*(3-j) - the established
// grouped RDNA twin's own run geometry (gemv_grouped_rdna_kernel), NOT a new bit layout.
// Verified against read_window/window16 on the host before landing: 200k random tiles per
// rate plus a full bit-flip sweep and all-zero/all-one/single-bit tiles (51.6M window
// comparisons per rate, 0 mismatches), and the halves through raw_pair reproduce the
// oracle's h = 0x6400+bytesum(hash(window)) per (row, col) over the full 256-weight tile.
// The integer-K epilogue (the fmaf/sum correction) is rate-independent algebra: every half
// is exactly 1024+S with S <= 1020 whatever K is, so K3/K4 keep the deferred scale/bias
// and the raw_pair<FastHash=false> plain-multiply hash.  K5+ and the K>=3 half-integer
// tiles stay OUT (the host plan refuses them; a K3.5 run's alternating spans would need
// the per-window S() shifts the established frac twin uses only at K2.5).
template <bool Frac, int MAX_ROWS, bool Fused = false, bool FusedSilu = false, int KBits = 2, bool Band8 = false>
__global__ __launch_bounds__(256) void gemv_grouped(const float* __restrict__ xh,
                                    const Exl3Mat* __restrict__ gmats,
                                    const int* __restrict__ gfirst, const int* __restrict__ grows,
                                    float* __restrict__ z, const uint16_t* __restrict__ x_raw = nullptr,
                                    long x_stride = 0, const int* __restrict__ input_rows = nullptr,
                                    const uint16_t* __restrict__ x_up = nullptr) {
    static_assert(KBits >= 2 && KBits <= 4, "the native grouped core decodes integer K2..K4 (plus K2.5 via Frac)");
    static_assert(!Frac || KBits == 2, "the half-integer arm exists only at K2.5");
    const int g = blockIdx.z;
    const Exl3Mat m = gmats[g];
    if (!m.trellis || (m.frac != 0) != Frac || m.bits != KBits) return;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int first = gfirst[g];
    const int rows = grows[g] > MAX_ROWS ? MAX_ROWS : grows[g];
    const int k = m.ki * 16, n = m.nj * 16;
    const int parts = gridDim.y, part = blockIdx.y;
    const int kt0 = m.ki * part / parts, kt1 = m.ki * (part + 1) / parts;
    const int nt = blockIdx.x * 8 + wave;
    const int elems = (kt1 - kt0) * 16;
    extern __shared__ __half2 slices[];
    if constexpr (Fused) {
        // Same partition coverage and per-row Hadamard arithmetic as the fused grouped
        // scalar/RDNA twin's prologue; only the slice store rounds to fp16 here (the pair()
        // rounding the native core would apply on read anyway).
        __shared__ float bfly[2][HAD];
        const int half = threadIdx.x >> 7, t = threadIdx.x & (HAD - 1);
        const int hb0 = kt0 * 16 / HAD, hb1 = (kt1 * 16 + HAD - 1) / HAD;
        for (int mm = 0; mm < rows; ++mm) {
            const long row = input_rows ? input_rows[first + mm] : first + mm;
            for (int p = 0; p < (hb1 - hb0 + 1) / 2; ++p) {
                const int hb = hb0 + 2 * p + half, idx = hb * HAD + t;
                float v = 0;
                if (hb < hb1) {
                    if constexpr (FusedSilu) {
                        const float gv = dh2f(x_raw[row * x_stride + idx]);
                        const float s = gv / (1.0f + expf(-gv));
                        const uint16_t h = __half_as_ushort(__float2half_rn(s * dh2f(x_up[row * x_stride + idx])));
                        v = dh2f(h) * dh2f(m.suh[idx]);
                    } else v = dh2f(x_raw[row * x_stride + idx]) * dh2f(m.suh[idx]);
                }
                for (int step = 1; step < HAD; step <<= 1) {
                    __syncthreads();
                    bfly[half][t] = v;
                    __syncthreads();
                    const float d = bfly[half][t ^ step];
                    v = (t & step) ? d - v : v + d;
                }
                if (hb < hb1 && idx >= kt0 * 16 && idx < kt1 * 16)
                    ((__half*) slices)[(size_t) mm * elems + idx - kt0 * 16] =
                        __float2half_rn(v * 0.08838834764831845f);
            }
        }
        __syncthreads();
    } else {
        // Unfused: pack the prescaled fp32 route rows into the same fp16 slice layout (the
        // pair() rounding applied once at load instead of once per dot2 operand).
        __half* dst = (__half*) slices;
        for (int mm = 0; mm < rows; ++mm) {
            const float* src = xh + (size_t) (first + mm) * k + (size_t) kt0 * 16;
            for (int i = threadIdx.x * 2; i < elems; i += blockDim.x * 2) {
                dst[(size_t) mm * elems + i] = __float2half_rn(src[i]);
                dst[(size_t) mm * elems + i + 1] = __float2half_rn(src[i + 1]);
            }
        }
        __syncthreads();
    }
    constexpr int words = Frac ? 20 : (KBits == 4 ? 32 : KBits == 3 ? 24 : 16);
    const auto* tr = reinterpret_cast<const uint32_t*>(m.trellis);
    uint32_t pf0[4], pf1[4], pf2[4], pf3[4];
    auto load = [&](int kt, uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
        a = b = c = d = 0;
        if (kt >= kt1 || nt >= m.nj) return;
        const uint32_t* tw;
        if constexpr (Band8)
            tw = tr + (size_t) blockIdx.x * m.ki * 8 * words + wave * words + (size_t) kt * 8 * words;
        else tw = tr + ((size_t) kt * m.nj + nt) * words;
        if constexpr (Frac) {
            const int fst = lane * 20 + 642, last = fst + 8;
            a = tw[((fst - 16) >> 5) % 20]; b = tw[((last - 1) >> 5) % 20];
            c = tw[((fst - 6) >> 5) % 20]; d = tw[((last + 9) >> 5) % 20];
        } else if constexpr (KBits >= 3) {
            // the two runs' span dwords (earlier dword first; the ring wraps mod words)
            const int bf0 = KBits * (8 * lane + 1) - 16 + words * 32;
            const int bf1 = bf0 + 4 * KBits;
            a = tw[(bf0 >> 5) % words]; b = tw[((bf0 + 3 * KBits + 15) >> 5) % words];
            c = tw[(bf1 >> 5) % words]; d = tw[((bf1 + 3 * KBits + 15) >> 5) % words];
        } else {
            const int i = lane >> 1;
            a = tw[(i + 15) & 15]; b = tw[i];
        }
    };
#pragma unroll
    for (int p = 0; p < 4; ++p) load(kt0 + p, pf0[p], pf1[p], pf2[p], pf3[p]);
    float acc0[MAX_ROWS], acc1[MAX_ROWS], xsum[MAX_ROWS];
#pragma unroll
    for (int mm = 0; mm < MAX_ROWS; ++mm) acc0[mm] = acc1[mm] = xsum[mm] = 0;
    const int row_pairs = elems >> 1;
    for (int kt = kt0; kt < kt1; ++kt) {
        const int at = (kt - kt0) & 3;
        const uint32_t a = pf0[at], b = pf1[at];
        const uint32_t c = pf2[at], d = pf3[at];
        load(kt + 4, pf0[at], pf1[at], pf2[at], pf3[at]);
        __builtin_amdgcn_sched_barrier(0);
        uint32_t windows[8];
        if constexpr (!Frac && KBits == 2) {
            const uint32_t bits = __builtin_amdgcn_alignbit(a, b, ((~(lane << 3)) & 8) << 1);
#pragma unroll
            for (int j = 0; j < 8; ++j) windows[7-j] = (bits >> (2*j)) & 0xffffu;
        } else if constexpr (!Frac) {
            // integer K3/K4: two runs of four windows, window j of run g at shift K*(3-j)
            // inside the 32-bit funnel ending with the run's last window
            const int bf0 = KBits * (8 * lane + 1) - 16 + words * 32, bf1 = bf0 + 4 * KBits;
            const int shift0 = 32 * (((bf0 + 3 * KBits) + 15) >> 5) + 16 - (bf0 + 3 * KBits);
            const int shift1 = 32 * (((bf1 + 3 * KBits) + 15) >> 5) + 16 - (bf1 + 3 * KBits);
            const uint32_t runs[2] = {__builtin_amdgcn_alignbit(a, b, shift0),
                                      __builtin_amdgcn_alignbit(c, d, shift1)};
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                windows[q*4] = (runs[q] >> (3 * KBits)) & 0xffffu;
                windows[q*4+1] = (runs[q] >> (2 * KBits)) & 0xffffu;
                windows[q*4+2] = (runs[q] >> KBits) & 0xffffu;
                windows[q*4+3] = runs[q] & 0xffffu;
            }
        } else {
            const int end0 = lane * 20 + 650, end1 = end0 + 10;
            const int shift0 = (((end0 - 1) >> 5) + 1) * 32 - end0;
            const int shift1 = (((end1 - 1) >> 5) + 1) * 32 - end1;
            const uint32_t runs[2] = {__builtin_amdgcn_alignbit(a, b, shift0),
                                      __builtin_amdgcn_alignbit(c, d, shift1)};
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                windows[q*4] = (runs[q] >> 8) & 0xffffu;
                windows[q*4+1] = (runs[q] >> 5) & 0xffffu;
                windows[q*4+2] = (runs[q] >> 3) & 0xffffu;
                windows[q*4+3] = runs[q] & 0xffffu;
            }
        }
        uint32_t w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            w[j] = raw_pair<Frac>(windows[2*j], windows[2*j+1]);
            if constexpr (Frac) {
                union { uint32_t u; __half2 h; } value;
                value.u = w[j];
                value.h = __hfma2(value.h, __half2half2(__ushort_as_half((uint16_t) 0x1eee)),
                                          __half2half2(__ushort_as_half((uint16_t) 0xc931)));
                w[j] = value.u;
            }
        }
        const int i = (kt - kt0) * 16 + (lane & 3) * 2;
        const int p2 = i >> 1, p1 = (i + 8) >> 1;
#pragma unroll
        for (int mm = 0; mm < MAX_ROWS; ++mm) {
            if (mm >= rows) break;
            const uint32_t x0 = __builtin_bit_cast(uint32_t, slices[(size_t) mm * row_pairs + p2]);
            const uint32_t x1 = __builtin_bit_cast(uint32_t, slices[(size_t) mm * row_pairs + p1]);
            acc0[mm] = dot2(w[0], x0, acc0[mm]); acc0[mm] = dot2(w[1], x1, acc0[mm]);
            acc1[mm] = dot2(w[2], x0, acc1[mm]); acc1[mm] = dot2(w[3], x1, acc1[mm]);
            if constexpr (!Frac) {
                xsum[mm] += __half2float(__ushort_as_half((uint16_t) x0)) +
                            __half2float(__ushort_as_half(x0 >> 16)) +
                            __half2float(__ushort_as_half((uint16_t) x1)) + __half2float(__ushort_as_half(x1 >> 16));
            }
        }
    }
#pragma unroll
    for (int delta = 1; delta <= 2; delta <<= 1) {
#pragma unroll
        for (int mm = 0; mm < MAX_ROWS; ++mm) {
            if (mm >= rows) break;   // rows is uniform across the block: the shuffle mask holds
            acc0[mm] += __shfl_xor_sync(0xffffffffu, acc0[mm], delta);
            acc1[mm] += __shfl_xor_sync(0xffffffffu, acc1[mm], delta);
            if constexpr (!Frac) xsum[mm] += __shfl_xor_sync(0xffffffffu, xsum[mm], delta);
        }
    }
    if constexpr (!Frac) {
        // unit's own integer-K epilogue: the dot2s accumulated the raw [0x6400+s] fp16 values
        // (1024 + s); this folds the mul1 scale/bias into one fmaf per row
#pragma unroll
        for (int mm = 0; mm < MAX_ROWS; ++mm) {
            if (mm >= rows) break;
            acc0[mm] = fmaf(acc0[mm], 0.00676727294921875f, xsum[mm] * -10.3828125f);
            acc1[mm] = fmaf(acc1[mm], 0.00676727294921875f, xsum[mm] * -10.3828125f);
        }
    }
    if ((lane & 3) == 0 && nt < m.nj) {
#pragma unroll
        for (int mm = 0; mm < MAX_ROWS; ++mm) {
            if (mm >= rows) break;
            float* dst = z + ((size_t) (first + mm) * parts + part) * n + nt * 16 + lane / 4;
            dst[0] = acc0[mm]; dst[8] = acc1[mm];
        }
    }
}
}  // namespace rdna_native
#endif
