// src/kernels/cuda/exl3.cu - EXL3 GPU reconstruct (docs/EXL3.md).
//
// Mirrors ExLlamaV3's reconstruct_had: decode the trellis tiles, apply the 128-point Hadamard on both
// dimensions, and fold in the suh/svh factors.  The codebook is procedural (decode_3inst), computed
// per window with no lookup table, exactly like the reference kernels.  This first version materializes
// W to fp16; the fused decode-GEMV is the follow-up (see docs/EXL3.md, "Decode optimizations").
#include "strata/kernels/exl3.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#if defined(STRATA_WMMA_GFX11)
#include "strata/kernels/gfx_arch.hpp"
#include <mutex>
#include <unordered_map>
#endif

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <cublas_v2.h>
#include <stdexcept>
#include <type_traits>

namespace strata::kernels {
namespace {
__global__ void peer_output_signal_kernel(uint32_t* ready,uint32_t sequence) {
    __threadfence_system();
    *(volatile uint32_t*)ready=sequence;
}
__global__ void peer_output_wait_kernel(const Exl3PeerOutputRecord* frame,const uint32_t* skip,uint32_t ring) {
    if(skip && *(const volatile uint32_t*)skip==ring) return;
    const auto* f=(const volatile Exl3PeerOutputRecord*)frame;
    while(f->ready<f->expected && !f->aborted) {}
    __threadfence_system();
}
__global__ void peer_output_copy_kernel(const Exl3PeerOutputRecord* frame,const uint16_t* rows,
                                       float* parts,int routes,int embd,const uint32_t* skip,uint32_t ring) {
    if(skip && *(const volatile uint32_t*)skip==ring) return;
    if(((const volatile Exl3PeerOutputRecord*)frame)->aborted) return;
    const long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(long)routes*embd) return;
    const int source=frame->rank[i/embd];
    if(source>=0) parts[i]=__half2float(__ushort_as_half(rows[(long)source*embd+i%embd]));
}
__global__ void peer_output_transfer_kernel(const uint4* source,uint4* target,long vectors,
                                           uint32_t* ready,uint32_t sequence,int* arrivals) {
    const long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<vectors) target[i]=source[i];
    __threadfence_system();
    __syncthreads();
    if(threadIdx.x==0 && atomicAdd(arrivals,1)==(int)gridDim.x-1) {
        *arrivals=0;
        __threadfence_system();
        *(volatile uint32_t*)ready=sequence;
    }
}
}
void exl3_peer_output_signal(uint32_t* ready,uint32_t sequence,void* stream) {
    peer_output_signal_kernel<<<1,1,0,(cudaStream_t)stream>>>(ready,sequence);
}
void exl3_peer_output_transfer(const uint16_t* source,uint16_t* target,int64_t elements,
                               uint32_t* ready,uint32_t sequence,int* arrivals,void* stream) {
    const long vectors=(long)elements/8;
    peer_output_transfer_kernel<<<(unsigned)((vectors+255)/256),256,0,(cudaStream_t)stream>>>(
        (const uint4*)source,(uint4*)target,vectors,ready,sequence,arrivals);
}
void exl3_peer_output_copy(const Exl3PeerOutputRecord* frame,const uint16_t* rows,float* parts,
                          int routes,int embd,const uint32_t* skip,uint32_t ring,void* stream) {
    peer_output_wait_kernel<<<1,1,0,(cudaStream_t)stream>>>(frame,skip,ring);
    peer_output_copy_kernel<<<(unsigned)(((long)routes*embd+255)/256),256,0,(cudaStream_t)stream>>>(
        frame,rows,parts,routes,embd,skip,ring);
}
namespace {

constexpr int HAD = 128;

__device__ __forceinline__ float dh2f(uint16_t h) {
    unsigned sign = (unsigned)(h >> 15) << 31, exp = (h >> 10) & 0x1F, man = h & 0x3FF, f;
    if (exp == 0) {
        if (!man) f = sign;
        else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; --exp; } man &= 0x3FF; f = sign | (exp << 23) | (man << 13); }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (man << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float o; __builtin_memcpy(&o, &f, 4); return o;
}

__device__ __forceinline__ uint32_t lop3(uint32_t a, uint32_t b, uint32_t c, uint32_t imm) {
    uint32_t r = 0;
    for (int i = 0; i < 8; ++i) {
        uint32_t t = ((imm >> i) & 1) ? 0xFFFFFFFFu : 0u;
        uint32_t ta = (i & 4) ? a : ~a, tb = (i & 2) ? b : ~b, tc = (i & 1) ? c : ~c;
        r |= t & ta & tb & tc;
    }
    return r;
}

// The window for stored position `pos`: the 16 bits ending at bit (pos+1)*bits, read with ExLlamaV3's
// funnel shift over the packed uint32 words (unpack_trellis / dq).  NOT a linear LSB read - the encoder
// packs per 16-value span, MSB-first, then byte-swaps.
__device__ __forceinline__ uint16_t read_window(const uint16_t* __restrict__ tw, int pos, int bits) {
    const uint32_t* p32 = (const uint32_t*) tw;
    const int nw = bits * 256 / 32;
    const int b0 = pos * bits + bits - 16 + 256 * bits;
    const int b2 = b0 + 16;
    const int i0r = b0 >> 5, i1r = (b2 - 1) >> 5;
    const int s1 = (i1r + 1) * 32 - b2;
    const uint32_t a = p32[i0r % nw], b = p32[i1r % nw];
    const unsigned long long m = ((unsigned long long) a << 32) | (unsigned long long) b;
    return (uint16_t)((m >> s1) & 0xFFFF);
}

// mul1 codebook, folded to pure ALU.  decode_window computes h2f(bytesum(w*C)+0x6400) with the software
// fp16->fp32 helper, but that intermediate always has exponent 25 (s in [0x6400,0x67FC]) so it is exactly
// the integer 1024+(s&0x3FF); the two h2f constants fold to plain floats.  This removes the fp16 helper
// and its denormal loop from the GEMV's innermost op.
__device__ __forceinline__ float decode_mul1(uint32_t w) {
    uint32_t x = w * 0x83DCD12Du;
    uint32_t s = (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + (x >> 24) + 0x6400u;
    return (float)(1024 + (int)(s & 0x3FFu)) * 0.00676727294921875f - 10.3828125f;
}

__device__ __forceinline__ uint16_t decode_window(int cb, uint16_t w) {    uint32_t x = w;
    if (cb == 2) {
        x *= 0x83DCD12Du;
        uint32_t s = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF) + 0x6400u;
        float h = dh2f((uint16_t)(s & 0xFFFF));
        return __half_as_ushort(__float2half_rn(__fmaf_rn(h, dh2f(0x1EEEu), dh2f(0xC931u))));
    }
    x = (cb == 1) ? x * 0xCBAC1FEDu : x * 89226354u + 64248484u;
    x = lop3(x, 0x8FFF8FFFu, 0x3B603B60u, 0x6Au);
    return __half_as_ushort(__float2half_rn(dh2f((uint16_t)(x & 0xFFFF)) + dh2f((uint16_t)(x >> 16))));
}

// One block per (nj, ki) tile grid; 256 threads, thread t decodes stored position t and writes it to
// its row-major destination inside the tile (tensor_core_perm: group g = t/8, slot = t%8).
__global__ void decode_kernel(const uint16_t* __restrict__ trellis, int ki, int nj, int words, int bits,
                              int cb, uint16_t* __restrict__ what, int n) {
    int i = blockIdx.y, j = blockIdx.x, t = threadIdx.x; (void)ki;
    const uint16_t* tw = trellis + (long)(i * nj + j) * words;
    unsigned v = read_window(tw, t, bits);
    int g = t >> 3, s = t & 7;
    int r0 = (g % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9, c0 = g / 4, c1 = c0 + 8;
    int r = (s & 3) == 0 ? r0 : (s & 3) == 1 ? r1 : (s & 3) == 2 ? r2 : r3;
    int c = s < 4 ? c0 : c1;
    what[(long)(i * 16 + r) * n + j * 16 + c] = decode_window(cb, (uint16_t)v);
}

__global__ void widen_kernel(const uint16_t* __restrict__ s, float* __restrict__ d, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = dh2f(s[i]);
}

__global__ void had_rows_kernel(float* __restrict__ w, int k, int n) {
    (void)k;
    // The vocabulary/head output dimension can exceed CUDA's grid.y limit of 65535.
    int rb = blockIdx.y, c = blockIdx.x;
    float b[HAD];
    for (int i = 0; i < HAD; ++i) b[i] = w[(long)(rb * HAD + i) * n + c];
    for (int step = 1; step < HAD; step <<= 1)
        for (int i = 0; i < HAD; ++i)
            if ((i & step) == 0) { float a = b[i], d = b[i | step]; b[i] = a + d; b[i | step] = a - d; }
    const float inv = 0.08838834764831845f;
    for (int i = 0; i < HAD; ++i) w[(long)(rb * HAD + i) * n + c] = b[i] * inv;
}

__global__ void had_cols_kernel(float* __restrict__ w, int k, int n) {
    (void)k;
    int r = blockIdx.x, cblk = blockIdx.y;
    float b[HAD];
    long base = (long)r * n + cblk * HAD;
    for (int j = 0; j < HAD; ++j) b[j] = w[base + j];
    for (int step = 1; step < HAD; step <<= 1)
        for (int j = 0; j < HAD; ++j)
            if ((j & step) == 0) { float a = b[j], d = b[j | step]; b[j] = a + d; b[j | step] = a - d; }
    const float inv = 0.08838834764831845f;
    for (int j = 0; j < HAD; ++j) w[base + j] = b[j] * inv;
}

__global__ void scale_rows_kernel(float* __restrict__ w, int n, const uint16_t* __restrict__ suh) {
    int i = blockIdx.x;
    float s = dh2f(suh[i]);
    for (int c = threadIdx.x; c < n; c += blockDim.x) w[(long)i * n + c] *= s;
}

__global__ void scale_cols_kernel(float* __restrict__ w, int k, int n, const uint16_t* __restrict__ svh) {
    int c = blockIdx.x;
    float s = dh2f(svh[c]);
    for (int r = threadIdx.x; r < k; r += blockDim.x) w[(long)r * n + c] *= s;
}

__global__ void f32_to_f16_kernel(const float* __restrict__ in, uint16_t* __restrict__ out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half_as_ushort(__float2half_rn(in[i]));
}

}  // namespace

void exl3_reconstruct_weight(const uint16_t* trellis, int ki, int nj, int bits, int cb,
                             const uint16_t* suh, const uint16_t* svh, uint16_t* out, void* stream) {
    if (ki <= 0 || nj <= 0 || (ki * 16) % HAD || (nj * 16) % HAD) {
        std::fprintf(stderr, "exl3_reconstruct_weight: dims must be multiples of %d\n", HAD);
        std::abort();
    }
    cudaStream_t st = (cudaStream_t)stream;
    int k = ki * 16, n = nj * 16, words = 256 * bits / 16;
    long nelem = (long)k * n;
    uint16_t* d_what = nullptr;
    float* d_w = nullptr;
    if (cudaMalloc(&d_what, (size_t)nelem * 2) != cudaSuccess ||
        cudaMalloc(&d_w, (size_t)nelem * 4) != cudaSuccess) {
        std::fprintf(stderr, "exl3_reconstruct_weight: cudaMalloc failed\n");
        std::abort();
    }
    auto check_launch = [](const char* name) {
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "exl3_reconstruct_weight %s: %s\n", name, cudaGetErrorString(e));
            std::abort();
        }
    };
    decode_kernel<<<dim3(nj, ki), 256, 0, st>>>(trellis, ki, nj, words, bits, cb, d_what, n);
    check_launch("decode_kernel");
    widen_kernel<<<(int)((nelem + 255) / 256), 256, 0, st>>>(d_what, d_w, nelem);
    check_launch("widen_kernel");
    had_rows_kernel<<<dim3(n, k / HAD), 1, 0, st>>>(d_w, k, n);
    check_launch("had_rows_kernel");
    had_cols_kernel<<<dim3(k, n / HAD), 1, 0, st>>>(d_w, k, n);
    check_launch("had_cols_kernel");
    scale_rows_kernel<<<dim3(k), 128, 0, st>>>(d_w, n, suh);
    check_launch("scale_rows_kernel");
    scale_cols_kernel<<<dim3(n), 128, 0, st>>>(d_w, k, n, svh);
    check_launch("scale_cols_kernel");
    f32_to_f16_kernel<<<(int)((nelem + 255) / 256), 256, 0, st>>>(d_w, out, nelem);
    check_launch("f32_to_f16_kernel");
    (void)cudaFree(d_what);
    (void)cudaFree(d_w);
}

namespace {

__global__ void prescale_had_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ suh,
                                    float* __restrict__ out) {
    __shared__ float b[HAD];
    int t = threadIdx.x, idx = blockIdx.x * HAD + t;
    b[t] = dh2f(x[idx]) * dh2f(suh[idx]);
    __syncthreads();
    for (int step = 1; step < HAD; step <<= 1) {
        if ((t & step) == 0) { float a = b[t], d = b[t | step]; b[t] = a + d; b[t | step] = a - d; }
        __syncthreads();
    }
    out[idx] = b[t] * 0.08838834764831845f;
}

// One block per (output 16-column tile, k-group); 128 threads = 16 columns x 8 k-subchunks.  Thread
// (c, sub) accumulates output column c over its k-slice, the 8 partials are summed in shared, and one
// atomicAdd per column folds the block's k-group into z.  This replaced 16-thread blocks that
// atomicAdd-ed per k-tile: 16-thread blocks cap at ~24 blocks/CU (≈19% occupancy) and every k-tile
// serialized on the same z, which is what pinned the GEMV at ~11 tok/s.
// **4 OUTPUT TILES PER BLOCK, MEASURED.**  The expert FFN is 48 layers x (gate, up, down) over 10 experts
// and it was the largest compute block of a saturated token (18.5 ms/token).  Sweeping the tiles per block on
// the real layer (`strata-exl3-moe`, layer 0, 10 experts): T=1 0.534 ms/layer, T=2 0.385 (the old value),
// T=3 0.321, **T=4 0.308**, T=5 0.319, T=6 0.327, T=7 0.342, T=8 0.325.  Sweeping the k-group count up from
// (ki+15)/16 only lost (x2 0.330, x4 0.543).  So 4, and the split stays as it is: -20% on the expert FFN,
// 3.7 ms/token, on every machine.
constexpr int GEMV_T = 4;

// The dense (single-matrix) GEMV: 256 threads (16 sub-groups of 16 output columns) and `GEMV_T` output
// tiles per block, like the batched kernel, but with the matrix's geometry as scalars.
//
// **THE WINDOW GEOMETRY IS HOISTED OUT OF THE k LOOP.**  The two words a 16-bit window touches and the shift
// between them depend only on the thread's 16 row positions (`tt[r]`, fixed for the block) and `bits`, not on
// the tile the loop is walking.  Computing them per weight costs 3 multiplies and two integer modulos, and
// with ~20 weights per thread (the old 8-way k split over 128 threads) that dominated the kernel: it ran at
// 40-115 GB/s against the ~700 GB/s the card has.  Packed into one int per row: word0 | word1 << 8 | shift << 16.
__global__ void __launch_bounds__(256, 6) gemv_kernel(const float* __restrict__ xh, const uint16_t* __restrict__ trellis,
                                                      int ki, int nj, int words, int bits, int cb,
                                                      const int* __restrict__ pinv, float* __restrict__ z) {
    int c = threadIdx.x & 15, sub = threadIdx.x >> 4;
    int ns = gridDim.y * 16;
    int cid = blockIdx.y * 16 + sub;
    int k0 = (ki * cid) / ns, k1 = (ki * (cid + 1)) / ns;
    int tt[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) tt[r] = pinv[r * 16 + c];
    const int nw = bits * 256 / 32;
    int wo[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int b0 = tt[r] * bits + bits - 16 + 256 * bits;
        const int b2 = b0 + 16;
        const int i0 = b0 >> 5, i1 = (b2 - 1) >> 5;
        wo[r] = (i0 % nw) | ((i1 % nw) << 8) | (((i1 + 1) * 32 - b2) << 16);
    }
    float acc[GEMV_T];
    #pragma unroll
    for (int t = 0; t < GEMV_T; ++t) acc[t] = 0;
    for (int i = k0; i < k1; ++i) {
        const uint32_t* twp[GEMV_T];
        #pragma unroll
        for (int t = 0; t < GEMV_T; ++t) {
            int j = blockIdx.x + t * gridDim.x;
            twp[t] = (j < nj) ? (const uint32_t*) (trellis + (long) (i * nj + j) * words) : nullptr;
        }
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            const float xr = xh[i * 16 + r];
            const int g = wo[r];
            #pragma unroll
            for (int t = 0; t < GEMV_T; ++t) {
                if (twp[t] == nullptr) continue;
                const uint64_t m = ((uint64_t) twp[t][g & 255] << 32) | twp[t][(g >> 8) & 255];
                const uint16_t win = (uint16_t) (m >> (g >> 16));
                const float wv = cb == 2 ? decode_mul1(win)
                                         : __half2float(__ushort_as_half(decode_window(cb, win)));
                acc[t] += xr * wv;
            }
        }
    }
    __shared__ float red[GEMV_T][16][16];
    #pragma unroll
    for (int t = 0; t < GEMV_T; ++t) red[t][sub][c] = acc[t];
    __syncthreads();
    if (sub == 0) {
        #pragma unroll
        for (int t = 0; t < GEMV_T; ++t) {
            const int j = blockIdx.x + t * gridDim.x;
            if (j >= nj) break;
            float s = 0;
            #pragma unroll
            for (int k2 = 0; k2 < 16; ++k2) s += red[t][k2][c];
            atomicAdd(&z[j * 16 + c], s);
        }
    }
}

__global__ void had_postscale_kernel(const float* __restrict__ z, const uint16_t* __restrict__ svh,
                                     uint16_t* __restrict__ y) {
    __shared__ float b[HAD];
    int t = threadIdx.x, idx = blockIdx.x * HAD + t;
    b[t] = z[idx];
    __syncthreads();
    for (int step = 1; step < HAD; step <<= 1) {
        if ((t & step) == 0) { float a = b[t], d = b[t | step]; b[t] = a + d; b[t | step] = a - d; }
        __syncthreads();
    }
    y[idx] = __half_as_ushort(__float2half_rn(b[t] * 0.08838834764831845f * dh2f(svh[idx])));
}

void build_pinv(int* pinv) {
    for (int t = 0; t < 256; ++t) {
        int g = t >> 3, s = t & 7;
        int r0 = (g % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9, c0 = g / 4, c1 = c0 + 8;
        int r = (s & 3) == 0 ? r0 : (s & 3) == 1 ? r1 : (s & 3) == 2 ? r2 : r3;
        int c = s < 4 ? c0 : c1;
        pinv[r * 16 + c] = t;
    }
}

// The GEMV body, with the three scratch buffers supplied so a caller can reuse them across launches.
void gemv_launch(const uint16_t* x, const uint16_t* suh, const uint16_t* svh, const uint16_t* trellis,
                 int ki, int nj, int bits, int cb, uint16_t* y, float* d_xh, float* d_z, const int* d_pinv,
                 cudaStream_t st) {
    int k = ki * 16, n = nj * 16, words = 256 * bits / 16;
    prescale_had_kernel<<<k / HAD, HAD, 0, st>>>(x, suh, d_xh);
    // A k split so the block count fills the machine: at least ~256 blocks, and never more than one row per
    // sub.  gy = 1 (each sub walks ki/16 rows) is the fast case for the big matrices; the small ones (the
    // shared expert's 40 output tiles) need the split or they run on 5k threads.
    int gy = 1;
    while (gy < 16 && ((nj + GEMV_T - 1) / GEMV_T) * gy < 256 && ki >= (gy + 1) * 16) ++gy;
    (void) cudaMemsetAsync(d_z, 0, (size_t)n * 4, st);
    gemv_kernel<<<dim3((nj + GEMV_T - 1) / GEMV_T, gy), 256, 0, st>>>(d_xh, trellis, ki, nj, words, bits, cb, d_pinv, d_z);
    had_postscale_kernel<<<n / HAD, HAD, 0, st>>>(d_z, svh, y);
}

}  // namespace

namespace {
// Persistent GEMV workspace, allocated before any graph capture (cudaMalloc is illegal inside a capture).
struct GemvWs { float* xh = nullptr; float* z = nullptr; int* pinv = nullptr; uint16_t* x16 = nullptr; uint16_t* y16 = nullptr;
                int cxh = 0, cz = 0, cx16 = 0, cy16 = 0; };
GemvWs g_ws;
}  // namespace

// out[o][i] = in[i][o]: a 32x32 tiled transpose, both sides coalesced.  The reconstruct hands back the weight
// in the original basis as [in][out] (the GEMV's `y[j] = sum_i x[i] W[i][j]`), while a ggml block quant - and
// `native_mmvq` - stores a row of `n_in` per output, so the loader transposes before it quantizes.
__global__ void transpose_f16_kernel(const uint16_t* __restrict__ in, uint16_t* __restrict__ out, int rows, int cols) {
    __shared__ uint16_t tile[32][33];
    const int x = blockIdx.x * 32 + threadIdx.x;      // column in `in` == row in `out`
    const int y = blockIdx.y * 32 + threadIdx.y;      // row in `in` == column in `out`
    if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = in[(size_t) y * cols + x];
    __syncthreads();
    const int ox = blockIdx.y * 32 + threadIdx.x;     // column in `out` == row in `in`
    const int oy = blockIdx.x * 32 + threadIdx.y;     // row in `out` == column in `in`
    if (ox < rows && oy < cols) out[(size_t) oy * rows + ox] = tile[threadIdx.x][threadIdx.y];
}

void exl3_transpose_f16(const uint16_t* in, uint16_t* out, int rows, int cols, void* stream) {
    dim3 block(32, 32);
    dim3 grid((cols + 31) / 32, (rows + 31) / 32);
    transpose_f16_kernel<<<grid, block, 0, (cudaStream_t) stream>>>(in, out, rows, cols);
}

void exl3_gemv_reserve(int max_k, int max_n) {
    if (max_k > g_ws.cxh) { (void) cudaFree(g_ws.xh); g_ws.xh = nullptr; if (cudaMalloc(&g_ws.xh, (size_t) max_k * 4) != cudaSuccess) std::abort(); g_ws.cxh = max_k; }
    if (max_n > g_ws.cz) { (void) cudaFree(g_ws.z); g_ws.z = nullptr; if (cudaMalloc(&g_ws.z, (size_t) max_n * 4) != cudaSuccess) std::abort(); g_ws.cz = max_n; }
    if (max_k > g_ws.cx16) { (void) cudaFree(g_ws.x16); g_ws.x16 = nullptr; if (cudaMalloc(&g_ws.x16, (size_t) max_k * 2) != cudaSuccess) std::abort(); g_ws.cx16 = max_k; }
    if (max_n > g_ws.cy16) { (void) cudaFree(g_ws.y16); g_ws.y16 = nullptr; if (cudaMalloc(&g_ws.y16, (size_t) max_n * 2) != cudaSuccess) std::abort(); g_ws.cy16 = max_n; }
    if (g_ws.pinv == nullptr) {
        if (cudaMalloc(&g_ws.pinv, 256 * 4) != cudaSuccess) std::abort();
        int p[256]; build_pinv(p); (void) cudaMemcpy(g_ws.pinv, p, 256 * 4, cudaMemcpyHostToDevice);
    }
}

void exl3_gemv(const uint16_t* x, const uint16_t* suh, const uint16_t* svh, const uint16_t* trellis,
               int ki, int nj, int bits, int cb, uint16_t* y, void* stream) {
    if (ki <= 0 || nj <= 0 || (ki * 16) % HAD || (nj * 16) % HAD) {
        std::fprintf(stderr, "exl3_gemv: dims must be multiples of %d\n", HAD);
        std::abort();
    }
    cudaStream_t st = (cudaStream_t)stream;
    exl3_gemv_reserve(ki * 16, nj * 16);
    gemv_launch(x, suh, svh, trellis, ki, nj, bits, cb, y, g_ws.xh, g_ws.z, g_ws.pinv, st);
}

namespace {

__global__ void f32_to_f16_out_kernel(const float* __restrict__ a, uint16_t* __restrict__ o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = __half_as_ushort(__float2half_rn(a[i]));
}

__global__ void f16_to_f32_kernel(const uint16_t* __restrict__ h, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = dh2f(h[i]);
}

// ---- batched (all-experts) variants: blockIdx.z = expert ----
//
// The engine runs one layer's routed experts per token; doing them one at a time left the GPU idle - each
// expert's prescale->gemv->postscale->silu->down->accum chain is a serial dependency and same-stream
// kernels do not overlap, so throughput scaled exactly linearly with expert count (0.139 ms at k=1 ->
// 1.237 ms at k=10) while the GPU sat ~98% idle.  Making the expert the z dimension lets all experts run
// concurrently inside one kernel and cuts the launch count from 14/expert to ~14/layer.

template <bool Masked = false>
__global__ void prescale_batched_kernel(const uint16_t* __restrict__ x, long x_stride,
                                        const Exl3Mat* __restrict__ mats, float* __restrict__ xh, int k,
                                        float* __restrict__ zero, long nz, const int* input_rows = nullptr,
                                        uint16_t* __restrict__ xh16 = nullptr) {
    __shared__ float b[HAD];
    int e = blockIdx.y, t = threadIdx.x, idx = blockIdx.x * HAD + t;
    if constexpr (Masked) { if (mats[e].trellis == nullptr) return; }
    // **THE ZEROING RIDES ALONG WITH THE PRESCALE.**  The GEMV accumulates with atomics, so its output must be
    // zeroed first, and that was a `cudaMemsetAsync` of its own: a small kernel costs ~14 us of launch and
    // execution, measured as part of the expert FFN's 0.083 ms/layer fixed cost (a quarter of the whole).
    // The prescale already runs immediately before the GEMV on the same stream, so the zeroing is one more
    // strided store here.
    if (zero != nullptr) {
        const long stride = (long) gridDim.x * gridDim.y * HAD;
        for (long i = (long) (blockIdx.y * gridDim.x + blockIdx.x) * HAD + t; i < nz; i += stride) zero[i] = 0.0f;
    }
    const long input_row = input_rows ? input_rows[e] : e;
    b[t] = dh2f(x[input_row * x_stride + idx]) * dh2f(mats[e].suh[idx]);
    __syncthreads();
    for (int step = 1; step < HAD; step <<= 1) {
        if ((t & step) == 0) { float a = b[t], d = b[t | step]; b[t] = a + d; b[t | step] = a - d; }
        __syncthreads();
    }
    const float v = b[t] * 0.08838834764831845f;
    xh[(long)e * k + idx] = v;
    // the fp16 twin for STRATA_EXL3_RDNA=2's fdot2 path (same value rounded once; the fp32 store
    // above is unchanged so every other consumer keeps its exact bytes)
    if (xh16 != nullptr) xh16[(long)e * k + idx] = __half_as_ushort(__float2half_rn(v));
}

template <int T, bool Partials = false, bool Masked = false>
__global__ void __launch_bounds__(256, 6) gemv_batched_kernel(const float* __restrict__ xh, const Exl3Mat* __restrict__ mats,
                                    const int* __restrict__ pinv, float* __restrict__ z) {
    int e = blockIdx.z;
    if constexpr (Masked) { if (mats[e].trellis == nullptr) return; }
    int c = threadIdx.x & 15, sub = threadIdx.x >> 4;
    int ki = mats[e].ki, nj = mats[e].nj, bits = mats[e].bits, cb = mats[e].cb;
    int words = 256 * bits / 16;
    int ns = gridDim.y * 16;
    int cid = blockIdx.y * 16 + sub;
    int k0 = (ki * cid) / ns, k1 = (ki * (cid + 1)) / ns;
    const uint16_t* trellis = mats[e].trellis;
    const float* xhe = xh + (long)e * (ki * 16);
    int tt[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) tt[r] = pinv[r * 16 + c];
    // the window geometry for this thread's 16 rows: fixed for the whole k loop (see gemv_kernel)
    const int nw = bits * 256 / 32;
    int wo[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int b0 = tt[r] * bits + bits - 16 + 256 * bits;
        const int b2 = b0 + 16;
        const int i0 = b0 >> 5, i1 = (b2 - 1) >> 5;
        wo[r] = (i0 % nw) | ((i1 % nw) << 8) | (((i1 + 1) * 32 - b2) << 16);
    }
    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0;
    for (int i = k0; i < k1; ++i) {
        const uint32_t* twp[T];
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            int j = blockIdx.x + t * gridDim.x;
            twp[t] = (j < nj) ? (const uint32_t*) (trellis + (long)(i * nj + j) * words) : nullptr;
        }
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            float xr = xhe[i * 16 + r];
            const int g = wo[r];
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                if (!twp[t]) continue;
                const uint64_t m = ((uint64_t) twp[t][g & 255] << 32) | twp[t][(g >> 8) & 255];
                const uint16_t win = (uint16_t) (m >> (g >> 16));
                float wv = cb == 2 ? decode_mul1(win) : __half2float(__ushort_as_half(decode_window(cb, win)));
                acc[t] += xr * wv;
            }
        }
    }
    __shared__ float red[T][16][16];
    #pragma unroll
    for (int t = 0; t < T; ++t) red[t][sub][c] = acc[t];
    __syncthreads();
    if (sub == 0) {
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            int j = blockIdx.x + t * gridDim.x;
            if (j >= nj) break;
            float s = 0;
            #pragma unroll
            for (int k2 = 0; k2 < 16; ++k2) s += red[t][k2][c];
            if constexpr (Partials)
                z[((long) e * gridDim.y + blockIdx.y) * (nj * 16) + j * 16 + c] = s;
            else atomicAdd(&z[(long)e * (nj * 16) + j * 16 + c], s);
        }
    }
}

template <bool Masked = false>
__global__ void postscale_batched_kernel(const float* __restrict__ z, const Exl3Mat* __restrict__ mats,
                                         uint16_t* __restrict__ y, int n, int parts = 1) {
    __shared__ float b[HAD];
    int e = blockIdx.y, t = threadIdx.x, idx = blockIdx.x * HAD + t;
    if constexpr (Masked) { if (mats[e].trellis == nullptr) return; }
    float value = z[(long) e * parts * n + idx];
    for (int p = 1; p < parts; ++p) value += z[((long) e * parts + p) * n + idx];
    b[t] = value;
    __syncthreads();
    for (int step = 1; step < HAD; step <<= 1) {
        if ((t & step) == 0) { float a = b[t], d = b[t | step]; b[t] = a + d; b[t | step] = a - d; }
        __syncthreads();
    }
    y[(long)e * n + idx] = __half_as_ushort(__float2half_rn(b[t] * 0.08838834764831845f * dh2f(mats[e].svh[idx])));
}

template <bool Masked = false>
__global__ void silu_batched_kernel(const uint16_t* __restrict__ g, const uint16_t* __restrict__ u,
                                    uint16_t* __restrict__ h, long total,
                                    const Exl3Mat* mats = nullptr, int ff = 0) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    if constexpr (Masked) { if (mats[i / ff].trellis == nullptr) return; }
    float gv = __half2float(__ushort_as_half(g[i]));
    float s = gv / (1.0f + expf(-gv));
    h[i] = __half_as_ushort(__float2half_rn(s * __half2float(__ushort_as_half(u[i]))));
}

// ---- RDNA3 instruction-level decode port (opt-in, STRATA_EXL3_RDNA) ----
//
// ExLlamaV3's rdna "tiles" core (exl3_gemv_tiles_rdna.cuh) beats the scalar decode by (a) loading ONE
// 2-dword funnel per run of consecutive stored positions instead of 2 dwords per weight, (b) hashing
// with v_pk_mul_lo_u16 + v_mad_u32_u16 (two full-rate ops replacing the quarter-rate v_mul_lo_u32),
// (c) summing bytes with sad_u8/sad_hi_u8 (full rate, the pair comes out pre-packed for an hfma2),
// and (d) folding the product into v_dot2_f32_f16.  Strata's (c, sub) thread geometry keeps its own
// output/reduction layout; what carries over is the RUN structure: for a fixed output column c, the
// 16 logical rows of a tile sit at stored positions that fall in 4 runs of 4 consecutive positions,
//     run q = rows {2q, 2q+1, 2q+8, 2q+9}  ->  t0 + j,  t0 = 32*(c&7) + 8*q + 4*(c>>3),  j = 0..3
// (build_pinv's tensor_core_perm: g = 4*(c&7) + q, s = j (+4 when c >= 8)).  A run's four 16-bit
// windows live in ring bits [B(t0), B(t0)+3K+16), which is <= 32 bits for K <= 5, so ONE
// v_alignbit_b32 funnel from the two span dwords decodes the whole run; K >= 6 uses two pair
// funnels.  The span/funnel geometry was validated exhaustively against read_window() for every
// K in 1..8 (200k random tiles per K) before this kernel was written.
//
// STRATA_EXL3_RDNA=1 keeps decode_mul1's exact fp32 arithmetic (bit-identical output to the scalar
// kernel: the hash and byte sum are integer-exact replacements, the float expression is the same
// text).  =2 additionally decodes weight PAIRS with sad_u8/sad_hi_u8 + hfma2 and accumulates with
// v_dot2_f32_f16 over fp16 activations (the reference engine's native arithmetic; not bit-identical
// to the fp32 path).

namespace {

// The native forms below need gfx11/gfx12 instructions (v_dot4 = dot8-insts).  STRATA_EXL3_RDNA_DECODE builds
// the RDNA decode kernels for every target in the arch list, so another family in a mixed list (gfx1030, gfx90a)
// gets the integer-exact portable form of the same value instead of a compile error.
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__GFX11__) || defined(__GFX12__)
#define STRATA_EXL3_RDNA_NATIVE 1
#endif

__device__ __forceinline__ uint32_t rdna_hash16_lo(uint32_t src) {
    // src's LOW 16 bits times 0x83DCD12D mod 2^32: x*C = x*C_lo16 + ((x * C_hi16) mod 2^16) << 16
#if defined(STRATA_EXL3_RDNA_NATIVE)
    uint32_t r, p;
    asm ("v_pk_mul_lo_u16 %0, %1, %2 op_sel_hi:[0,1]" : "=v"(r) : "v"(src), "s"(0x83DC0000u));
    asm ("v_mad_u32_u16 %0, %1, %2, %3" : "=v"(p) : "v"(src), "s"(0xD12Du), "v"(r));
    return p;
#else
    return (src & 0xFFFFu) * 0x83DCD12Du;
#endif
}

__device__ __forceinline__ uint32_t rdna_funnel(uint32_t lo, uint32_t hi, uint32_t shift) {
    // ((hi << 32) | lo) >> shift, low 32 bits, one v_alignbit_b32
    return __funnelshift_r(lo, hi, shift);
}

// mul1 decode's byte sum, one v_sudot4 (identical integers to the shift/add chain)
__device__ __forceinline__ uint32_t rdna_bytesum(uint32_t x) {
#if defined(STRATA_EXL3_RDNA_NATIVE)
    return (uint32_t) __builtin_amdgcn_sudot4(false, x, false, 0x01010101u, 0x6400u, false);
#else
    return (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + (x >> 24) + 0x6400u;
#endif
}

// Two hashed windows -> the decoded fp16 weight pair: sad_u8 emits [0x6400+s0 | 0] (the sum fits 16
// bits, so no carry), sad_hi_u8 folds the second sum into the high half, and one hfma2 applies the
// codebook scale/bias to both halves at once (exl3_decode_pair<cb==2>).
__device__ __forceinline__ __half2 rdna_mul1_pair(uint32_t p0, uint32_t p1) {
    uint32_t s = __builtin_amdgcn_sad_u8(p0, 0u, 0x64006400u);
    s = __builtin_amdgcn_sad_hi_u8(p1, 0u, s);
    union { uint32_t u; __half2 h; } v; v.u = s;
    return __hfma2(v.h, __half2half2(__ushort_as_half((uint16_t) 0x1eee)),
                        __half2half2(__ushort_as_half((uint16_t) 0xc931)));
}

__device__ __forceinline__ float rdna_fdot2(__half2 a, __half2 b, float c) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fdot2))
        return __builtin_amdgcn_fdot2(a, b, c, false);
    float lo = __half2float(__low2half(a)) * __half2float(__low2half(b));
    return fmaf(__half2float(__high2half(a)), __half2float(__high2half(b)), c + lo);
}

// Run geometry: per run q, the funnel's two dword indices (ring) and base shift, packed like wo[]
// above (wlo | whi << 8 | sh << 16).  K <= 5: one funnel per run; K >= 6: two pair funnels.
struct RdnaRun { int wlo, whi, sh; };

// STRATA_EXL3_RDNA's level, read once (gfx11 and gfx12 builds, STRATA_EXL3_RDNA_DECODE; see the kernel comments below): shared
// by the indexed twin and the grouped twin's dispatcher.
inline int rdna_level() {
#if defined(STRATA_EXL3_RDNA_DECODE)
    static const int v = [] {
        const char* value = std::getenv("STRATA_EXL3_RDNA");
        return value ? std::max(0, std::min(2, std::atoi(value))) : 0;
    }();
    return v;
#else
    return 0;
#endif
}

__device__ __forceinline__ RdnaRun rdna_span(const int nw, int b_first, int b_last) {
    const int dlo = b_first >> 5, dhi = (b_last + 15) >> 5;
    return { dlo % nw, dhi % nw, 32 * dhi + 16 - b_last };
}

}  // namespace

// The helpers above stay outside the gfx11 guard: gemv_grouped_rdna_kernel (below the guard) uses them,
// so they must compile for every gfx11/gfx12 target (issue #1, gfx1201-only builds).  Without STRATA_EXL3_RDNA_DECODE
// (the build defines it for gfx11 and gfx12 targets) rdna_level() is 0, so the RDNA decode paths are never launched
// there.  The WMMA kernels keep STRATA_WMMA_GFX11 (gfx11 only) and their per-device runtime gate.
#if defined(STRATA_EXL3_RDNA_DECODE)
namespace {

// gemv_batched_kernel's RDNA twin: same grid, same 256-thread (c, sub) geometry, same shared-tile
// reduction and Partials output, same per-(i, r, t) FMA ORDER for RDNA == 1 - only the loads (one
// 2-dword funnel per run of 4 rows instead of 2 dwords per row) and the integer decode change.
// RDNA == 2 walks runs q-major (4 live funnels instead of 16) and is free to do so because its
// hfma2/fdot2 arithmetic is not order-matched to the scalar kernel anyway.
//
// FusedIn (STRATA_EXL3_FUSE, indexed/Partials path only) folds prescale_batched_kernel into this
// kernel's prologue - the block Hadamards its OWN k-slice into shared memory instead of reading a
// separate kernel's global buffer, exactly the reference's exl3_gemv_rotate_in idea.  The butterfly
// arithmetic is prescale_batched_kernel's own, so the activations (and therefore RDNA == 1 outputs)
// stay bit-identical.  The ride-along z zeroing is dropped with it: the Partials path never reads z
// (each partition fully writes its own slot), and that zeroing was already dead work on this path.
// Shared budget: red[T][16][16] static (4 KB) + dynamic slice (<= (ki/gy+1)*16 floats) + butterfly.
//
// FusedOut (STRATA_EXL3_FUSE=2; implies FusedIn) additionally folds postscale_batched_kernel into
// the epilogue, the reference's exl3_gemv_fused_epilogue pattern: tiles become CONTIGUOUS per
// block (j = blockIdx.x*T + t - output values are unchanged, each tile's partials are the same
// numbers whatever block computes them), every block arrives at its (expert, 128-column segment)'s
// counter, and the last of the 2*gy arrivals sums the partials in partition order, Hadamards and
// svh-scales the segment exactly as postscale_batched_kernel does, and writes the fp16 row.  The
// counter self-resets, so the workspace buffer only needs zeroing when it grows.
template <int T, bool Partials, int RDNA, bool FusedIn = false, bool FusedOut = false,
          bool FusedSilu = false, bool Masked = false>
__global__ void __launch_bounds__(256, 4) gemv_batched_rdna_kernel(const float* __restrict__ xh,
                                   const uint16_t* __restrict__ xh16, const Exl3Mat* __restrict__ mats,
                                   const int* __restrict__ pinv, float* __restrict__ z,
                                   const uint16_t* __restrict__ x_raw = nullptr, long x_stride = 0,
                                   const int* __restrict__ input_rows = nullptr,
                                   uint16_t* __restrict__ out16 = nullptr,
                                   int* __restrict__ counters = nullptr,
                                   const uint16_t* __restrict__ x_silu_mul = nullptr) {
    static_assert(RDNA == 1 || RDNA == 2, "RDNA level");
    static_assert(!FusedIn || Partials, "the fused prologue exists for the indexed/Partials path");
    static_assert(!FusedOut || Partials, "the fused epilogue exists for the indexed/Partials path");
    static_assert(!FusedOut || FusedIn, "FUSE=2 implies the fused prologue");
    static_assert(!FusedSilu || FusedIn, "the fused SiLU exists inside the fused prologue");
    int e = blockIdx.z;
    if constexpr (Masked) { if (mats[e].trellis == nullptr) return; }
    int c = threadIdx.x & 15, sub = threadIdx.x >> 4;
    int ki = mats[e].ki, nj = mats[e].nj, bits = mats[e].bits, cb = mats[e].cb;
    int words = 256 * bits / 16;
    int ns = gridDim.y * 16;
    int cid = blockIdx.y * 16 + sub;
    int k0 = (ki * cid) / ns, k1 = (ki * (cid + 1)) / ns;
    const uint16_t* trellis = mats[e].trellis;
    const float* xhe = xh + (long)e * (ki * 16);
    const uint16_t* xhe16 = xh16 + (long)e * (ki * 16);
    const int nw = bits * 256 / 32;
    float* slice = nullptr;                  // FusedIn: this PARTITION's Hadamard'd k-slice
    int part_k0 = 0, part_k1 = 0;            // the k-tile range the whole blockIdx.y covers
    if constexpr (FusedIn) {
        extern __shared__ float dyn[];
        __shared__ float bfly[2][HAD];
        slice = dyn;
        const long row = input_rows ? input_rows[e] : e;
        const uint16_t* xe = x_raw + row * x_stride;
        const uint16_t* xu = x_silu_mul + row * x_stride;   // same route's up row
        const uint16_t* suh = mats[e].suh;
        const int half = threadIdx.x >> 7, t = threadIdx.x & (HAD - 1);
        // A block is 16 independent (c, sub) mini-jobs, each with its OWN k range - the thread's
        // [k0, k1) above.  The prologue therefore Hadamards at PARTITION granularity: the union
        // range [part_k0, part_k1) = all k-tiles of this blockIdx.y, shared by every sub.  A
        // 128-point Hadamard consumes the whole 128-element block around each element, so
        // transform every HAD block covering the partition range and keep the partition's own
        // elements; each sub then reads its sub-range out of the slice (16 | 128 keeps every
        // sub's 16-wide range inside one HAD block).  Both 128-thread halves run the SAME trip
        // count (an odd block count would otherwise deadlock on the shared barrier); a dead half
        // still participates in the butterfly with zero input.
        part_k0 = (ki * (blockIdx.y * 16)) / ns;
        part_k1 = (ki * (blockIdx.y * 16 + 16)) / ns;
        const int hb0 = (part_k0 * 16) / HAD, hb1 = ((part_k1 * 16 - 1) / HAD) + 1;
        for (int p = 0; p < (hb1 - hb0 + 1) / 2; ++p) {
            const int hb = hb0 + 2 * p + half;
            const bool live = hb < hb1;
            float v = 0;
            if (live) {
                const int idx = hb * HAD + t;
                if constexpr (FusedSilu) {
                    // silu_batched_kernel's arithmetic verbatim, then the prescale product: the
                    // half rounding of silu(g)*u is what the separate kernels' d_h buffer carried
                    const float gv = dh2f(xe[idx]);
                    const float s = gv / (1.0f + expf(-gv));
                    const uint16_t h = __half_as_ushort(__float2half_rn(s * dh2f(xu[idx])));
                    v = dh2f(h) * dh2f(suh[idx]);
                } else {
                    v = dh2f(xe[idx]) * dh2f(suh[idx]);
                }
            }
            for (int step = 1; step < HAD; step <<= 1) {
                __syncthreads();
                bfly[half][t] = v;
                __syncthreads();
                const float d = bfly[half][t ^ step];
                v = (t & step) ? d - v : v + d;   // prescale's pair rule: b[lo]=b[lo]+b[hi], b[hi]=b[lo]-b[hi]
            }
            if (live) {
                const int k = hb * HAD + t;       // global element index of this transformed value
                if (k >= part_k0 * 16 && k < part_k1 * 16)
                    slice[k - part_k0 * 16] = v * 0.08838834764831845f;
            }
        }
        __syncthreads();
        xhe = slice - part_k0 * 16;          // xhe[i*16 + r] now reads the partition slice
    }
    // The run t0 formula is build_pinv's permutation stated analytically (see the block comment);
    // the parity tests upload that same pinv, so any drift here fails them loudly.
    const int t0_run = 32 * (c & 7) + 4 * (c >> 3);
    RdnaRun run[4][2];                       // [q][funnel]; bits <= 5 uses only [q][0]
    const bool wide = bits > 5;
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int B0 = (t0_run + 8 * q) * bits + bits - 16 + 256 * bits;
        if (!wide)
            run[q][0] = rdna_span(nw, B0, B0 + 3 * bits);
        else {
            run[q][0] = rdna_span(nw, B0, B0 + bits);
            run[q][1] = rdna_span(nw, B0 + 2 * bits, B0 + 3 * bits);
        }
    }
    // cb != 2 keeps the scalar decoder (the packed-hash/sad/hfma2 tricks are mul1-only): this is
    // gemv_batched_kernel's own inner loop, pinv table and all.
    int tt[16], wo[16];
    if (cb != 2) {
        #pragma unroll
        for (int r = 0; r < 16; ++r) tt[r] = pinv[r * 16 + c];
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            const int b0 = tt[r] * bits + bits - 16 + 256 * bits;
            const int b2 = b0 + 16;
            const int i0 = b0 >> 5, i1 = (b2 - 1) >> 5;
            wo[r] = (i0 % nw) | ((i1 % nw) << 8) | (((i1 + 1) * 32 - b2) << 16);
        }
    }
    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0;
    if (cb != 2) {
        for (int i = k0; i < k1; ++i) {
            const uint32_t* twp[T];
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                int j = FusedOut ? blockIdx.x * T + t : blockIdx.x + t * gridDim.x;
                twp[t] = (j < nj) ? (const uint32_t*) (trellis + (long)(i * nj + j) * words) : nullptr;
            }
            #pragma unroll
            for (int r = 0; r < 16; ++r) {
                float xr = xhe[i * 16 + r];
                const int g = wo[r];
                #pragma unroll
                for (int t = 0; t < T; ++t) {
                    if (!twp[t]) continue;
                    const uint64_t m = ((uint64_t) twp[t][g & 255] << 32) | twp[t][(g >> 8) & 255];
                    const uint16_t win = (uint16_t) (m >> (g >> 16));
                    acc[t] += xr * __half2float(__ushort_as_half(decode_window(cb, win)));
                }
            }
        }
    } else if (RDNA == 1) {
        // bit-identical to gemv_batched_kernel: same r-major (i, r, t) accumulation order, the same
        // fp32 decode expression, integer-exact hash/byte-sum substitutes
        for (int i = k0; i < k1; ++i) {
            const uint32_t* twp[T];
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                int j = FusedOut ? blockIdx.x * T + t : blockIdx.x + t * gridDim.x;
                twp[t] = (j < nj) ? (const uint32_t*) (trellis + (long)(i * nj + j) * words) : nullptr;
            }
            uint32_t f[T][4];
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                if (!twp[t]) continue;
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    if (!wide)
                        f[t][q] = rdna_funnel(twp[t][run[q][0].whi], twp[t][run[q][0].wlo], run[q][0].sh);
                    else {
                        // pack the two pair funnels as f[q] = funnel2:bits | funnel1:0 is not
                        // possible in one register; store funnel1, recompute funnel2 on demand
                        f[t][q] = rdna_funnel(twp[t][run[q][0].whi], twp[t][run[q][0].wlo], run[q][0].sh);
                    }
                }
            }
            #pragma unroll
            for (int r = 0; r < 16; ++r) {
                const float xr = xhe[i * 16 + r];
                const int q = (r >> 1) & 3, j = (r & 1) + ((r >> 3) << 1);
                #pragma unroll
                for (int t = 0; t < T; ++t) {
                    if (!twp[t]) continue;
                    uint32_t win;
                    if (!wide)
                        win = f[t][q] >> (bits * (3 - j));
                    else if (j < 2)       // funnel 0: j=0 sits `bits` above j=1
                        win = j == 0 ? (f[t][q] >> bits) : f[t][q];
                    else {                // funnel 1: recomputed from the same dwords
                        const uint32_t fb = rdna_funnel(twp[t][run[q][1].whi], twp[t][run[q][1].wlo], run[q][1].sh);
                        win = j == 2 ? (fb >> bits) : fb;
                    }
                    const uint32_t x = rdna_hash16_lo(win);
                    const uint32_t s = rdna_bytesum(x);
                    const float wv = (float) (1024 + (int) (s & 0x3FFu)) * 0.00676727294921875f - 10.3828125f;
                    acc[t] += xr * wv;
                }
            }
        }
    } else {
        // RDNA == 2: q-major runs, fp16 activation pairs (xh16), weight pairs via sad + hfma2,
        // accumulation via v_dot2_f32_f16 - the reference engine's native arithmetic
        for (int i = k0; i < k1; ++i) {
            const uint32_t* twp[T];
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                int j = FusedOut ? blockIdx.x * T + t : blockIdx.x + t * gridDim.x;
                twp[t] = (j < nj) ? (const uint32_t*) (trellis + (long)(i * nj + j) * words) : nullptr;
            }
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                __half2 x01, x89;
                if constexpr (FusedIn) {   // round the slice once, exactly as xh16's store would
                    x01 = __floats2half2_rn(xhe[i * 16 + 2 * q], xhe[i * 16 + 2 * q + 1]);
                    x89 = __floats2half2_rn(xhe[i * 16 + 2 * q + 8], xhe[i * 16 + 2 * q + 9]);
                } else {
                    x01 = *((const __half2*) (xhe16 + i * 16 + 2 * q));
                    x89 = *((const __half2*) (xhe16 + i * 16 + 2 * q + 8));
                }
                #pragma unroll
                for (int t = 0; t < T; ++t) {
                    if (!twp[t]) continue;
                    __half2 w01, w89;
                    if (!wide) {
                        const uint32_t f = rdna_funnel(twp[t][run[q][0].whi], twp[t][run[q][0].wlo], run[q][0].sh);
                        // j=0 at shift 3*bits, j=1 at 2*bits, j=2 at bits, j=3 at 0
                        w01 = rdna_mul1_pair(rdna_hash16_lo(f >> (3 * bits)), rdna_hash16_lo(f >> (2 * bits)));
                        w89 = rdna_mul1_pair(rdna_hash16_lo(f >> bits), rdna_hash16_lo(f));
                    } else {
                        const uint32_t fa = rdna_funnel(twp[t][run[q][0].whi], twp[t][run[q][0].wlo], run[q][0].sh);
                        const uint32_t fb = rdna_funnel(twp[t][run[q][1].whi], twp[t][run[q][1].wlo], run[q][1].sh);
                        w01 = rdna_mul1_pair(rdna_hash16_lo(fa >> bits), rdna_hash16_lo(fa));
                        w89 = rdna_mul1_pair(rdna_hash16_lo(fb >> bits), rdna_hash16_lo(fb));
                    }
                    acc[t] = rdna_fdot2(x01, w01, acc[t]);
                    acc[t] = rdna_fdot2(x89, w89, acc[t]);
                }
            }
        }
    }
    __shared__ float red[T][16][16];
    #pragma unroll
    for (int t = 0; t < T; ++t) red[t][sub][c] = acc[t];
    __syncthreads();
    if (sub == 0) {
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            int j = FusedOut ? blockIdx.x * T + t : blockIdx.x + t * gridDim.x;
            if (j >= nj) break;
            float s = 0;
            #pragma unroll
            for (int k2 = 0; k2 < 16; ++k2) s += red[t][k2][c];
            if constexpr (Partials)
                z[((long) e * gridDim.y + blockIdx.y) * (nj * 16) + j * 16 + c] = s;
            else atomicAdd(&z[(long)e * (nj * 16) + j * 16 + c], s);
        }
    }
    if constexpr (FusedOut) {
        // Arrive at this (expert, segment)'s counter; the last of the 2*gy blocks covering the
        // segment finalizes it (partials sum in partition order + Hadamard + svh + fp16 store,
        // postscale_batched_kernel's arithmetic verbatim) and resets the counter for the next call.
        // Fences are outside every predicate, as in the reference's epilogue.
        __syncthreads();
        __threadfence();
        __shared__ int arrival;
        if (threadIdx.x == 0)
            arrival = (int) atomicAdd(&counters[e * (nj / 8) + (blockIdx.x >> 1)], 1u);
        __syncthreads();
        const int target = 2 * (int) gridDim.y;
        if (arrival != target - 1) return;
        __threadfence();
        if (threadIdx.x == 0) counters[e * (nj / 8) + (blockIdx.x >> 1)] = 0;
        // postscale_batched_kernel's arithmetic verbatim, on this block's low 128 threads; the
        // high half stays in the barriers (no early return - __syncthreads must see every thread)
        __shared__ float b2[HAD];
        const int idx = (blockIdx.x >> 1) * HAD + threadIdx.x;
        const int n = nj * 16, parts = (int) gridDim.y;
        float value = 0;
        if (threadIdx.x < HAD) {
            value = z[(long) e * parts * n + idx];
            for (int p = 1; p < parts; ++p) value += z[((long) e * parts + p) * n + idx];
            b2[threadIdx.x] = value;
        }
        __syncthreads();
        for (int step = 1; step < HAD; step <<= 1) {
            if (threadIdx.x < HAD && (threadIdx.x & step) == 0) {
                float a = b2[threadIdx.x], d = b2[threadIdx.x | step];
                b2[threadIdx.x] = a + d; b2[threadIdx.x | step] = a - d;
            }
            __syncthreads();
        }
        if (threadIdx.x < HAD)
            out16[(long)e * n + idx] = __half_as_ushort(__float2half_rn(
                b2[threadIdx.x] * 0.08838834764831845f * dh2f(mats[e].svh[idx])));
    }
}

// ---- the dense multirow GEMV: one matrix, m = 1..8 rows (A9) ----
//
// The dense linears staged as trellis (STRATA_EXL3_DENSE=exl3) had only the m == 1 decode GEMV; the
// verify window (spec/batch/prefill) needs native_data and refuses to start, which is why every dense
// linear is requantized to Q8_0 by default (~2.8x weight traffic).  This is the window's stand-in: the
// reference engine's exl3_gemv_multirow design on this file's kernel geometry -
//
//   * the input rotation is a SEPARATE kernel (one 128-thread block per (row, 128-block): suh + H128
//     into a global fp32 scratch that stays L2-resident; 8 rows x 2560 floats would not fit LDS),
//   * the dot kernel keeps the batched kernel's 256-thread (c, sub) mini-jobs and k-partitions, but
//     holds M accumulator sets, one per row, behind the SAME single B decode - so row r of an m-row
//     call accumulates exactly the m == 1 call's terms in the m == 1 order and is bit-identical to it,
//   * the counter epilogue (arrivals per 128-column segment, last block sums the partitions in order,
//     Hadamards, svh-scales) is the fused epilogue's, run once per row, storing fp32.
//
// Rows past rows_valid recompute the last valid row's data and are never stored (the reference's rule),
// so the tail tile of a large batch costs nothing but the padded lanes.
__global__ void dense_had_in_kernel(const float* __restrict__ x, long x_stride, const uint16_t* __restrict__ suh,
                                    float* __restrict__ xh, int k) {
    __shared__ float b[HAD];
    const int t = threadIdx.x, idx = blockIdx.x * HAD + t, row = blockIdx.y;
    b[t] = x[(long) row * x_stride + idx] * dh2f(suh[idx]);
    __syncthreads();
    for (int step = 1; step < HAD; step <<= 1) {
        if ((t & step) == 0) { float a = b[t], d = b[t | step]; b[t] = a + d; b[t | step] = a - d; }
        __syncthreads();
    }
    xh[(long) row * k + idx] = b[t] * 0.08838834764831845f;
}

template <int T, int M>
__global__ void __launch_bounds__(256, 2) gemv_dense_mr_kernel(const float* __restrict__ xh,
                                   const uint16_t* __restrict__ trellis, const uint16_t* __restrict__ svh,
                                   const int* __restrict__ pinv, int ki, int nj, int bits, int cb,
                                   float* __restrict__ z, float* __restrict__ y, long y_stride,
                                   int rows_valid, int* __restrict__ counters) {
    const int c = threadIdx.x & 15, sub = threadIdx.x >> 4;
    const int ns = gridDim.y * 16;
    const int cid = blockIdx.y * 16 + sub;
    const int k0 = (ki * cid) / ns, k1 = (ki * (cid + 1)) / ns;
    const int words = 256 * bits / 16, nw = bits * 256 / 32, n = nj * 16;
    // row r's base in xh: rows past rows_valid re-read the last valid row (their outputs are dropped)
    const float* xrow[M];
    #pragma unroll
    for (int r = 0; r < M; ++r) xrow[r] = xh + (long) (r < rows_valid ? r : rows_valid - 1) * (ki * 16);
    const int t0_run = 32 * (c & 7) + 4 * (c >> 3);
    RdnaRun run[4][2];
    const bool wide = bits > 5;
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int B0 = (t0_run + 8 * q) * bits + bits - 16 + 256 * bits;
        if (!wide)
            run[q][0] = rdna_span(nw, B0, B0 + 3 * bits);
        else {
            run[q][0] = rdna_span(nw, B0, B0 + bits);
            run[q][1] = rdna_span(nw, B0 + 2 * bits, B0 + 3 * bits);
        }
    }
    int tt[16], wo[16];
    if (cb != 2) {   // the scalar decoder, as in the batched kernel (pinv + window tables)
        #pragma unroll
        for (int r = 0; r < 16; ++r) tt[r] = pinv[r * 16 + c];
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            const int b0 = tt[r] * bits + bits - 16 + 256 * bits;
            const int b2 = b0 + 16;
            const int i0 = b0 >> 5, i1 = (b2 - 1) >> 5;
            wo[r] = (i0 % nw) | ((i1 % nw) << 8) | (((i1 + 1) * 32 - b2) << 16);
        }
    }
    float acc[T][M];
    #pragma unroll
    for (int t = 0; t < T; ++t)
        #pragma unroll
        for (int r = 0; r < M; ++r) acc[t][r] = 0;
    for (int i = k0; i < k1; ++i) {
        const uint32_t* twp[T];
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const int j = blockIdx.x * T + t;
            twp[t] = (j < nj) ? (const uint32_t*) (trellis + (long) (i * nj + j) * words) : nullptr;
        }
        if (cb != 2) {
            #pragma unroll
            for (int r = 0; r < 16; ++r) {
                const int g = wo[r];
                #pragma unroll
                for (int t = 0; t < T; ++t) {
                    if (!twp[t]) continue;
                    const uint64_t m = ((uint64_t) twp[t][g & 255] << 32) | twp[t][(g >> 8) & 255];
                    const uint16_t win = (uint16_t) (m >> (g >> 16));
                    const float wv = __half2float(__ushort_as_half(decode_window(cb, win)));
                    #pragma unroll
                    for (int r2 = 0; r2 < M; ++r2)
                        acc[t][r2] += xrow[r2][i * 16 + r] * wv;
                }
            }
        } else {
            // the batched kernel's RDNA == 1 body, one B decode per run, M rows fed from it: per
            // (t, r2) the (i, r) term order is the m == 1 order, so each row stays bit-identical to it
            uint32_t f[T][4];
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                if (!twp[t]) continue;
                #pragma unroll
                for (int q = 0; q < 4; ++q)
                    f[t][q] = rdna_funnel(twp[t][run[q][0].whi], twp[t][run[q][0].wlo], run[q][0].sh);
            }
            #pragma unroll
            for (int r = 0; r < 16; ++r) {
                const int q = (r >> 1) & 3, j = (r & 1) + ((r >> 3) << 1);
                #pragma unroll
                for (int t = 0; t < T; ++t) {
                    if (!twp[t]) continue;
                    uint32_t win;
                    if (!wide)
                        win = f[t][q] >> (bits * (3 - j));
                    else if (j < 2)
                        win = j == 0 ? (f[t][q] >> bits) : f[t][q];
                    else {
                        const uint32_t fb = rdna_funnel(twp[t][run[q][1].whi], twp[t][run[q][1].wlo], run[q][1].sh);
                        win = j == 2 ? (fb >> bits) : fb;
                    }
                    const uint32_t x = rdna_hash16_lo(win);
                    const uint32_t s = rdna_bytesum(x);
                    const float wv = (float) (1024 + (int) (s & 0x3FFu)) * 0.00676727294921875f - 10.3828125f;
                    #pragma unroll
                    for (int r2 = 0; r2 < M; ++r2)
                        acc[t][r2] += xrow[r2][i * 16 + r] * wv;
                }
            }
        }
    }
    // stage the M rows through the batched kernel's 16-sub reduction, one row at a time: per (t, r2)
    // the sub sum order is unchanged
    __shared__ float red[T][16][16];
    #pragma unroll
    for (int r2 = 0; r2 < M; ++r2) {
        #pragma unroll
        for (int t = 0; t < T; ++t) red[t][sub][c] = acc[t][r2];
        __syncthreads();
        if (sub == 0) {
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                const int j = blockIdx.x * T + t;
                if (j >= nj) break;
                float s = 0;
                #pragma unroll
                for (int k2 = 0; k2 < 16; ++k2) s += red[t][k2][c];
                z[((long) blockIdx.y * M + r2) * n + j * 16 + c] = s;
            }
        }
        __syncthreads();
    }
    // the fused epilogue's arrival handshake; the last block of the segment finalizes every row
    __threadfence();
    __shared__ int arrival;
    if (threadIdx.x == 0)
        arrival = (int) atomicAdd(&counters[blockIdx.x >> 1], 1u);
    __syncthreads();
    const int target = 2 * (int) gridDim.y;
    if (arrival != target - 1) return;
    __threadfence();
    if (threadIdx.x == 0) counters[blockIdx.x >> 1] = 0;
    __shared__ float b2[HAD];
    const int parts = (int) gridDim.y;
    const int seg0 = (blockIdx.x >> 1) * HAD;
    for (int r2 = 0; r2 < rows_valid; ++r2) {
        float value = 0;
        if (threadIdx.x < HAD) {
            const int idx = seg0 + threadIdx.x;
            value = z[(long) r2 * n + idx];
            for (int p = 1; p < parts; ++p) value += z[((long) p * M + r2) * n + idx];
            b2[threadIdx.x] = value;
        }
        __syncthreads();
        for (int step = 1; step < HAD; step <<= 1) {
            if (threadIdx.x < HAD && (threadIdx.x & step) == 0) {
                float a = b2[threadIdx.x], d = b2[threadIdx.x | step];
                b2[threadIdx.x] = a + d; b2[threadIdx.x | step] = a - d;
            }
            __syncthreads();
        }
        if (threadIdx.x < HAD)
            y[(long) r2 * y_stride + seg0 + threadIdx.x] =
                b2[threadIdx.x] * 0.08838834764831845f * dh2f(svh[seg0 + threadIdx.x]);
        __syncthreads();
    }
}

}  // namespace
#endif  // STRATA_EXL3_RDNA_DECODE

__global__ void accum_out_kernel(const uint16_t* __restrict__ d, const float* __restrict__ weights,
                                 int n_experts, int n, uint16_t* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0;
    for (int e = 0; e < n_experts; ++e) s += weights[e] * __half2float(__ushort_as_half(d[(long)e * n + i]));
    out[i] = __half_as_ushort(__float2half_rn(s));
}

// ---- PROTOTYPE grouped small-M arm (diagnostic-only, no engine caller) ----
//
// Constraint this arm is built under: the GROUP is the blockIdx.z unit, so a trellis window is
// extracted and decoded once per (k row, output tile) and applied to every row of the group, and
// the group expert's suh/svh loads once per group - but the per-row FLOAT ARITHMETIC MUST MATCH
// gemv_batched_kernel/prescale_batched_kernel/postscale_batched_kernel EXACTLY (k-chunk
// boundaries, (i, r) FMA order, 16-sub shared reduce order, parts order, Hadamards), so a row's
// output is byte-identical to the indexed arm's on the same route order.  No activation
// quantization, no DP4A, no codebook change.  Measurements live in docs/EXL3-HIP-BENCH.md.
constexpr int SMALL_M_MAX_ROWS = 6;   // widest group the entry accepts; callers chunk larger ones

// One block per (HAD block of k, GROUP), looping the group's rows; row m of group g is route
// first+m so the prescaled activations stay ROUTE-major like the indexed arm's.
__global__ void prescale_grouped_kernel(const uint16_t* __restrict__ x, long x_stride,
                                        const Exl3Mat* __restrict__ gmats,
                                        const int* __restrict__ gfirst, const int* __restrict__ grows,
                                        float* __restrict__ xh, int k, float* __restrict__ zero, long nz,
                                        const int* input_rows) {
    __shared__ float b[HAD];
    int g = blockIdx.y, t = threadIdx.x, idx = blockIdx.x * HAD + t;
    if (zero != nullptr) {   // same ride-along zeroing as the indexed prescale, same element set
        const long stride = (long) gridDim.x * gridDim.y * HAD;
        for (long i = (long) (blockIdx.y * gridDim.x + blockIdx.x) * HAD + t; i < nz; i += stride) zero[i] = 0.0f;
    }
    const int first = gfirst[g], rows = grows[g];
    const float s = dh2f(gmats[g].suh[idx]);   // the group's one expert: the scale loads once
    for (int m = 0; m < rows; ++m) {
        const long input_row = input_rows ? input_rows[first + m] : (first + m);
        b[t] = dh2f(x[input_row * x_stride + idx]) * s;
        __syncthreads();
        for (int step = 1; step < HAD; step <<= 1) {
            if ((t & step) == 0) { float a = b[t], d = b[t | step]; b[t] = a + d; b[t | step] = a - d; }
            __syncthreads();
        }
        xh[(long) (first + m) * k + idx] = b[t] * 0.08838834764831845f;
        // End-of-row barrier: every thread's own-element read above is complete before ANY
        // thread overwrites b for the next row.  Without it the next iteration's write races
        // a slower warp's store operand read; the Hadamard's own barriers only order the
        // butterfly, not this trailing read.
        __syncthreads();
    }
}

// The grouped twin of gemv_batched_kernel: same 16x16 thread geometry, same chunking (cid over
// gridDim.y*16), same window-geometry hoist, same Partials output layout - but the trellis word is
// extracted and decoded once per (i, r, tile) and the weight multiplies every row's activation.
template <int T, int MAX_ROWS, bool Partials = false>
__global__ void __launch_bounds__(256, 6) gemv_grouped_kernel(const float* __restrict__ xh,
                                    const Exl3Mat* __restrict__ gmats,
                                    const int* __restrict__ gfirst, const int* __restrict__ grows,
                                    const int* __restrict__ pinv, float* __restrict__ z) {
    int g = blockIdx.z;
    int c = threadIdx.x & 15, sub = threadIdx.x >> 4;
    int first = gfirst[g], rows = grows[g];
    int ki = gmats[g].ki, nj = gmats[g].nj, bits = gmats[g].bits, cb = gmats[g].cb;
    int words = 256 * bits / 16;
    int ns = gridDim.y * 16;
    int cid = blockIdx.y * 16 + sub;
    int k0 = (ki * cid) / ns, k1 = (ki * (cid + 1)) / ns;
    const uint16_t* trellis = gmats[g].trellis;
    int tt[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) tt[r] = pinv[r * 16 + c];
    const int nw = bits * 256 / 32;
    int wo[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int b0 = tt[r] * bits + bits - 16 + 256 * bits;
        const int b2 = b0 + 16;
        const int i0 = b0 >> 5, i1 = (b2 - 1) >> 5;
        wo[r] = (i0 % nw) | ((i1 % nw) << 8) | (((i1 + 1) * 32 - b2) << 16);
    }
    // one hoisted row base (the indexed arm's xhe); the unrolled m offsets are compile-time strides
    const float* xrow = xh + (long) first * (ki * 16);
    float acc[T][MAX_ROWS];
    #pragma unroll
    for (int t = 0; t < T; ++t)
        #pragma unroll
        for (int m = 0; m < MAX_ROWS; ++m) acc[t][m] = 0;
    for (int i = k0; i < k1; ++i) {
        const uint32_t* twp[T];
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            int j = blockIdx.x + t * gridDim.x;
            twp[t] = (j < nj) ? (const uint32_t*) (trellis + (long)(i * nj + j) * words) : nullptr;
        }
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            float xr[MAX_ROWS];
            #pragma unroll
            for (int m = 0; m < MAX_ROWS; ++m) {
                if constexpr (MAX_ROWS > 1) { if (m >= rows) break; }   // width 1 has no guard
                xr[m] = xrow[(long) m * (ki * 16) + i * 16 + r];
            }
            const int wgeo = wo[r];
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                if (!twp[t]) continue;
                const uint64_t m2 = ((uint64_t) twp[t][wgeo & 255] << 32) | twp[t][(wgeo >> 8) & 255];
                const uint16_t win = (uint16_t) (m2 >> (wgeo >> 16));
                float wv = cb == 2 ? decode_mul1(win) : __half2float(__ushort_as_half(decode_window(cb, win)));
                #pragma unroll
                for (int m = 0; m < MAX_ROWS; ++m) {
                    if constexpr (MAX_ROWS > 1) { if (m >= rows) break; }
                    acc[t][m] += xr[m] * wv;
                }
            }
        }
    }
    // Same two-level reduction as the indexed kernel (subs in k2 order inside the block), one row
    // at a time so the shared tile stays [T][16][16] and occupancy is not taxed by MAX_ROWS.
    __shared__ float red[T][16][16];
    #pragma unroll
    for (int m = 0; m < MAX_ROWS; ++m) {
        if constexpr (MAX_ROWS > 1) { if (m >= rows) break; }
        #pragma unroll
        for (int t = 0; t < T; ++t) red[t][sub][c] = acc[t][m];
        __syncthreads();
        if (sub == 0) {
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                int j = blockIdx.x + t * gridDim.x;
                if (j >= nj) break;
                float s = 0;
                #pragma unroll
                for (int k2 = 0; k2 < 16; ++k2) s += red[t][k2][c];
                if constexpr (Partials)
                    z[((long) (first + m) * gridDim.y + blockIdx.y) * (nj * 16) + j * 16 + c] = s;
                else
                    atomicAdd(&z[(long) (first + m) * (nj * 16) + j * 16 + c], s);
            }
        }
        __syncthreads();
    }
}

// gemv_grouped_kernel's RDNA twin (the A4/A5 follow-up the ledger named): same grid, same
// (c, sub) geometry, same per-row shared reduction and Partials layout, and the same per-row
// (i, r, t) FMA ORDER as BOTH arms - the decoded weight wv depends only on (i, r, t), so the
// group's rows consume ONE run-funnel extraction and ONE hash/byte-sum decode, and each row's
// float accumulation sequence is the indexed RDNA arm's own.  Every row therefore stays
// byte-identical to both arms while the trellis bytes and the decode ALU are paid once per
// GROUP instead of once per route.  mul1 only (the packed tricks are cb == 2): the dispatcher
// keeps the scalar prototype for every other codebook, and bits <= 5 uses the single-funnel
// runs exactly like the indexed kernel.
template <int T, int MAX_ROWS, bool Partials = false, bool Fused = false, bool FusedSilu = false>
__global__ void __launch_bounds__(256, 4) gemv_grouped_rdna_kernel(const float* __restrict__ xh,
                                    const Exl3Mat* __restrict__ gmats,
                                    const int* __restrict__ gfirst, const int* __restrict__ grows,
                                    float* __restrict__ z, const uint16_t* x_raw = nullptr,
                                    long x_stride = 0, const int* input_rows = nullptr,
                                    uint16_t* out = nullptr, int* counters = nullptr,
                                    const uint16_t* x_up = nullptr) {
    static_assert(!Fused || (Partials && (T == 2 || T == 4)), "fused segments require two- or four-tile blocks");
    int g = blockIdx.z;
    int c = threadIdx.x & 15, sub = threadIdx.x >> 4;
    const int first = gfirst[g], rows = grows[g];
    const int ki = gmats[g].ki, nj = gmats[g].nj, bits = gmats[g].bits;
    const int words = 256 * bits / 16;
    const int ns = gridDim.y * 16;
    const int cid = blockIdx.y * 16 + sub;
    const int k0 = (ki * cid) / ns, k1 = (ki * (cid + 1)) / ns;
    const uint16_t* trellis = gmats[g].trellis;
    const int nw = bits * 256 / 32;
    const float* xrow = xh + (long) first * (ki * 16);
    int row_stride = ki * 16;
    if constexpr (Fused) {
        // Same partition/Hadamard coverage and per-row arithmetic as indexed FUSE=3.
        // Only the transformed partition is retained, for every live row in the group.
        extern __shared__ float slices[];
        __shared__ float bfly[2][HAD];
        const int pk0 = ki * (int) blockIdx.y / (int) gridDim.y;
        const int pk1 = ki * ((int) blockIdx.y + 1) / (int) gridDim.y;
        const int hb0 = pk0 * 16 / HAD, hb1 = (pk1 * 16 + HAD - 1) / HAD;
        const int half = threadIdx.x >> 7, t = threadIdx.x & (HAD - 1);
        row_stride = (pk1 - pk0) * 16;
        for (int m = 0; m < rows; ++m) {
            const long row = input_rows ? input_rows[first + m] : first + m;
            for (int p = 0; p < (hb1 - hb0 + 1) / 2; ++p) {
                const int hb = hb0 + 2 * p + half, idx = hb * HAD + t;
                float v = 0;
                if (hb < hb1) {
                    if constexpr (FusedSilu) {
                        const float gv = dh2f(x_raw[row * x_stride + idx]);
                        const float s = gv / (1.0f + expf(-gv));
                        const uint16_t h = __half_as_ushort(__float2half_rn(s * dh2f(x_up[row * x_stride + idx])));
                        v = dh2f(h) * dh2f(gmats[g].suh[idx]);
                    } else v = dh2f(x_raw[row * x_stride + idx]) * dh2f(gmats[g].suh[idx]);
                }
                for (int step = 1; step < HAD; step <<= 1) {
                    __syncthreads();
                    bfly[half][t] = v;
                    __syncthreads();
                    const float d = bfly[half][t ^ step];
                    v = (t & step) ? d - v : v + d;
                }
                if (hb < hb1 && idx >= pk0 * 16 && idx < pk1 * 16)
                    slices[m * row_stride + idx - pk0 * 16] = v * 0.08838834764831845f;
            }
        }
        __syncthreads();
        xrow = slices - pk0 * 16;
    }
    const int t0_run = 32 * (c & 7) + 4 * (c >> 3);
    RdnaRun run[4][2];
    const bool wide = bits > 5;
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int B0 = (t0_run + 8 * q) * bits + bits - 16 + 256 * bits;
        if (!wide)
            run[q][0] = rdna_span(nw, B0, B0 + 3 * bits);
        else {
            run[q][0] = rdna_span(nw, B0, B0 + bits);
            run[q][1] = rdna_span(nw, B0 + 2 * bits, B0 + 3 * bits);
        }
    }
    float acc[T][MAX_ROWS];
    #pragma unroll
    for (int t = 0; t < T; ++t)
        #pragma unroll
        for (int m = 0; m < MAX_ROWS; ++m) acc[t][m] = 0;
    for (int i = k0; i < k1; ++i) {
        const uint32_t* twp[T];
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            int j = Fused ? blockIdx.x * T + t : blockIdx.x + t * gridDim.x;
            twp[t] = (j < nj) ? (const uint32_t*) (trellis + (long)(i * nj + j) * words) : nullptr;
        }
        uint32_t f[T][4];
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            if (!twp[t]) continue;
            #pragma unroll
            for (int q = 0; q < 4; ++q)
                f[t][q] = rdna_funnel(twp[t][run[q][0].whi], twp[t][run[q][0].wlo], run[q][0].sh);
        }
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            float xr[MAX_ROWS];
            #pragma unroll
            for (int m = 0; m < MAX_ROWS; ++m) {
                if constexpr (MAX_ROWS > 1) { if (m >= rows) break; }
                xr[m] = xrow[(long) m * row_stride + i * 16 + r];
            }
            const int q = (r >> 1) & 3, j = (r & 1) + ((r >> 3) << 1);
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                if (!twp[t]) continue;
                uint32_t win;
                if (!wide)
                    win = f[t][q] >> (bits * (3 - j));
                else if (j < 2)
                    win = j == 0 ? (f[t][q] >> bits) : f[t][q];
                else {
                    const uint32_t fb = rdna_funnel(twp[t][run[q][1].whi], twp[t][run[q][1].wlo], run[q][1].sh);
                    win = j == 2 ? (fb >> bits) : fb;
                }
                const uint32_t xw = rdna_hash16_lo(win);
                const uint32_t s = rdna_bytesum(xw);
                const float wv = (float) (1024 + (int) (s & 0x3FFu)) * 0.00676727294921875f - 10.3828125f;
                #pragma unroll
                for (int m = 0; m < MAX_ROWS; ++m) {
                    if constexpr (MAX_ROWS > 1) { if (m >= rows) break; }
                    acc[t][m] += xr[m] * wv;
                }
            }
        }
    }
    __shared__ float red[T][16][16];
    #pragma unroll
    for (int m = 0; m < MAX_ROWS; ++m) {
        if constexpr (MAX_ROWS > 1) { if (m >= rows) break; }
        #pragma unroll
        for (int t = 0; t < T; ++t) red[t][sub][c] = acc[t][m];
        __syncthreads();
        if (sub == 0) {
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                int j = Fused ? blockIdx.x * T + t : blockIdx.x + t * gridDim.x;
                if (j >= nj) break;
                float s = 0;
                #pragma unroll
                for (int k2 = 0; k2 < 16; ++k2) s += red[t][k2][c];
                if constexpr (Partials)
                    z[((long) (first + m) * gridDim.y + blockIdx.y) * (nj * 16) + j * 16 + c] = s;
                else
                    atomicAdd(&z[(long) (first + m) * (nj * 16) + j * 16 + c], s);
            }
        }
        __syncthreads();
    }
    if constexpr (Fused) {
        // All rows' partials precede one group/segment arrival. The final block
        // reduces partitions in the indexed order and publishes every live row.
        constexpr int segment_blocks = (HAD / 16) / T;
        const int segment = blockIdx.x / segment_blocks;
        __threadfence();
        __shared__ int arrival;
        if (threadIdx.x == 0)
            arrival = (int) atomicAdd(&counters[g * (nj / 8) + segment], 1u);
        __syncthreads();
        if (arrival != segment_blocks * (int) gridDim.y - 1) return;
        __threadfence();
        if (threadIdx.x == 0) counters[g * (nj / 8) + segment] = 0;
        __shared__ float b[HAD];
        const int idx = segment * HAD + threadIdx.x;
        const int n = nj * 16, parts = gridDim.y;
        for (int m = 0; m < rows; ++m) {
            if (threadIdx.x < HAD) {
                float v = z[(long) (first + m) * parts * n + idx];
                for (int p = 1; p < parts; ++p) v += z[((long) (first + m) * parts + p) * n + idx];
                b[threadIdx.x] = v;
            }
            __syncthreads();
            for (int step = 1; step < HAD; step <<= 1) {
                if (threadIdx.x < HAD && !(threadIdx.x & step)) {
                    const float a = b[threadIdx.x], d = b[threadIdx.x | step];
                    b[threadIdx.x] = a + d; b[threadIdx.x | step] = a - d;
                }
                __syncthreads();
            }
            if (threadIdx.x < HAD)
                out[(long) (first + m) * n + idx] = __half_as_ushort(__float2half_rn(
                    b[threadIdx.x] * 0.08838834764831845f * dh2f(gmats[g].svh[idx])));
            __syncthreads();
        }
    }
}

// The grouped twin of postscale_batched_kernel: parts summed in p order, then the 128-point
// Hadamard, then the group expert's svh - loaded once per group instead of per route.
__global__ void postscale_grouped_kernel(const float* __restrict__ z, const Exl3Mat* __restrict__ gmats,
                                         const int* __restrict__ gfirst, const int* __restrict__ grows,
                                         uint16_t* __restrict__ y, int n, int parts) {
    __shared__ float b[HAD];
    int g = blockIdx.y, t = threadIdx.x, idx = blockIdx.x * HAD + t;
    const int first = gfirst[g], rows = grows[g];
    const float svh = dh2f(gmats[g].svh[idx]);
    for (int m = 0; m < rows; ++m) {
        float value = z[(long) (first + m) * parts * n + idx];
        for (int p = 1; p < parts; ++p) value += z[((long) (first + m) * parts + p) * n + idx];
        b[t] = value;
        __syncthreads();
        for (int step = 1; step < HAD; step <<= 1) {
            if ((t & step) == 0) { float a = b[t], d = b[t | step]; b[t] = a + d; b[t | step] = a - d; }
            __syncthreads();
        }
        y[(long) (first + m) * n + idx] =
            __half_as_ushort(__float2half_rn(b[t] * 0.08838834764831845f * svh));
        // End-of-row barrier: same contract as prescale_grouped_kernel's - the trailing
        // own-element read is ordered against the next row's shared reuse.
        __syncthreads();
    }
}

}  // namespace


// Mutable MoE scratch belongs to an execution context, not to the process. Element capacities
// include the route count so a 3-expert hit partition followed by 10 misses cannot overrun scratch.
namespace {
template <typename T> struct MoeBuffer {
    T* p = nullptr;
    size_t capacity = 0;
    void reserve(size_t count) {
        if (count <= capacity) return;
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::overflow_error("exl3 MoE workspace size overflow");
        T* next = nullptr;
        if (cudaMalloc((void**) &next, count * sizeof(T)) != cudaSuccess)
            throw std::runtime_error("exl3 MoE workspace allocation failed");
        (void) cudaFree(p);
        p = next;
        capacity = count;
    }
    ~MoeBuffer() { if (p) (void) cudaFree(p); }
};
}

struct Exl3MoeWorkspace::Impl {
    int device = -1;
    MoeBuffer<Exl3Mat> gate, up, down;
    MoeBuffer<uint16_t> g, u, h, d;
    MoeBuffer<float> xh, z, weights, partials;
    MoeBuffer<uint16_t> xh16;   // STRATA_EXL3_RDNA=2's fp16 activation twin of xh
    MoeBuffer<int> pinv;
    MoeBuffer<int> fuse_counters;   // STRATA_EXL3_FUSE=2's per-(expert, segment) arrival counters
    MoeBuffer<uint16_t> group_x, group_weight;
    MoeBuffer<float> group_y, group_reconstruct;
    MoeBuffer<uint16_t> wmma_x;   // the gfx11 matrix-core group FFN's gathered h128 slab
    MoeBuffer<Exl3Group> expert_groups;
    // the PROTOTYPE grouped small-M arm's route-plan buffers (diagnostic-only; see the kernels above)
    MoeBuffer<Exl3Mat> smallm_gate, smallm_up, smallm_down;
    MoeBuffer<int> smallm_first, smallm_rows;
    cublasHandle_t blas = nullptr;
    Impl() { if (cudaGetDevice(&device) != cudaSuccess) throw std::runtime_error("exl3: no CUDA device"); }
    ~Impl() { if (blas) (void) cublasDestroy(blas); }
};

Exl3MoeWorkspace::Exl3MoeWorkspace() : impl_(new Impl) {}
Exl3MoeWorkspace::~Exl3MoeWorkspace() {
    int previous = -1;
    (void) cudaGetDevice(&previous);
    (void) cudaSetDevice(impl_->device);
    delete impl_;
    if (previous != -1) (void) cudaSetDevice(previous);
}
Exl3MoeWorkspace::Impl& Exl3MoeWorkspace::implementation() { return *impl_; }
void Exl3MoeWorkspace::reserve(int routes, int n_in, int n_ff, int n_out, bool indexed) {
    int device = -1;
    if (routes < 1 || n_in < 1 || n_ff < 1 || n_out < 1 ||
        cudaGetDevice(&device) != cudaSuccess || device != impl_->device)
        throw std::invalid_argument("exl3: workspace geometry/device mismatch");
    auto& w = *impl_;
    const size_t r = (size_t) routes;
    w.gate.reserve(r); w.up.reserve(r); w.down.reserve(r); w.weights.reserve(r);
    w.g.reserve(r * (size_t) n_ff); w.u.reserve(r * (size_t) n_ff); w.h.reserve(r * (size_t) n_ff);
    w.d.reserve(r * (size_t) n_out);
    w.xh.reserve(r * (size_t) std::max(n_in, n_ff));
    // xh16 / fuse_counters are allocated lazily by moe_run_experts_dev's opt-in paths only: the
    // flag-off path must keep the EXACT allocation pattern of the established code (its layout is
    // load-bearing for the concurrent-workspace parity tests).
    w.z.reserve(r * (size_t) std::max(n_ff, n_out));
    if (indexed) {
        const size_t gate_parts = ((size_t) n_in + 255) / 256;
        const size_t down_parts = ((size_t) n_ff + 255) / 256;
        w.partials.reserve(r * std::max(gate_parts * (size_t) n_ff, down_parts * (size_t) n_out));
    }
    if (!w.pinv.p) {
        w.pinv.reserve(256);
        int pinv[256]; build_pinv(pinv);
        if (cudaMemcpy(w.pinv.p, pinv, sizeof pinv, cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("exl3: permutation upload failed");
    }
}

void Exl3MoeWorkspace::reserve_runtime_scratch(int routes, int n_in, int n_ff, int n_out) {
    reserve(routes, n_in, n_ff, n_out, true);
    const int rdna = rdna_level();
    const char* v = std::getenv("STRATA_EXL3_FUSE");
    const int fuse = rdna >= 1 && v ? std::max(0, std::min(3, std::atoi(v))) : 0;
    auto& w = *impl_;
    if (rdna == 2 && fuse == 0)
        w.xh16.reserve((size_t) routes * std::max(n_in, n_ff));
    if (fuse >= 2) {
        const size_t count = (size_t) routes * ((size_t) std::max(n_ff, n_out) / 128 + 1);
        if (count > w.fuse_counters.capacity) {
            w.fuse_counters.reserve(count);
            if (cudaMemset(w.fuse_counters.p, 0, count * sizeof(int)) != cudaSuccess ||
                cudaStreamSynchronize(nullptr) != cudaSuccess)
                throw std::runtime_error("exl3: fused-epilogue setup counter zeroing failed");
        }
    }
}

// The group paths' own scratch (group_x/group_weight/group_reconstruct/group_y, and the matrix-core
// arm's wmma_x when it may run), kept OUT of reserve() so ordinary indexed callers never allocate it.
// A caller that batches many group tiles before syncing calls this once with its largest count: the
// entry points' grow-only reserves would otherwise cudaFree buffers that queued tiles still read.
void Exl3MoeWorkspace::reserve_group_scratch(int routes, int n_in, int n_ff, int n_out, bool wmma_scratch) {
    int device = -1;
    if (routes < 1 || n_in < 1 || n_ff < 1 || n_out < 1 ||
        cudaGetDevice(&device) != cudaSuccess || device != impl_->device)
        throw std::invalid_argument("exl3: workspace geometry/device mismatch");
    auto& w = *impl_;
    const size_t r = (size_t) routes, wide = (size_t) std::max(n_in, n_ff);
    w.group_x.reserve(r * (size_t) n_in);
    w.group_weight.reserve(std::max((size_t) n_in * n_ff, (size_t) n_ff * n_out));
    w.group_reconstruct.reserve(std::max((size_t) n_in * n_ff, (size_t) n_ff * n_out));
    w.group_y.reserve(r * (size_t) std::max(n_ff, n_out));
    if (wmma_scratch) w.wmma_x.reserve(r * wide);
}

void Exl3MoeWorkspace::reserve_expert_batch(int groups, int routes, int n_in, int n_ff, int n_out) {
    if (groups < 1 || groups > 8 || routes < 1 || routes > 4096)
        throw std::invalid_argument("exl3: invalid expert batch reserve geometry");
    reserve(routes, n_in, n_ff, n_out);
    reserve_group_scratch(routes, n_in, n_ff, n_out, true);
    impl_->expert_groups.reserve((size_t) groups);
}

// The PROTOTYPE grouped small-M arm's buffers only, so a diagnostic caller can pre-reserve before
// timed/captured execution like the other arms do. Same grow-only contract as reserve().
void Exl3MoeWorkspace::reserve_small_m(int routes, int groups, int n_in, int n_ff, int n_out) {
    if (groups < 1 || groups > routes || routes < 1 || routes > 65535)
        throw std::invalid_argument("exl3: invalid small-M reserve geometry");
    reserve(routes, n_in, n_ff, n_out, true);
    auto& w = *impl_;
    w.smallm_gate.reserve((size_t) groups); w.smallm_up.reserve((size_t) groups);
    w.smallm_down.reserve((size_t) groups);
    w.smallm_first.reserve((size_t) groups); w.smallm_rows.reserve((size_t) groups);
}

size_t Exl3MoeWorkspace::allocated_bytes() const {
    const auto& w=*impl_;
    return (w.gate.capacity+w.up.capacity+w.down.capacity)*sizeof(Exl3Mat) +
        (w.g.capacity+w.u.capacity+w.h.capacity+w.d.capacity+w.group_x.capacity+w.group_weight.capacity+w.wmma_x.capacity+w.xh16.capacity)*sizeof(uint16_t) +
        (w.xh.capacity+w.z.capacity+w.weights.capacity+w.partials.capacity+w.group_y.capacity+w.group_reconstruct.capacity)*sizeof(float) +
        (w.pinv.capacity+w.fuse_counters.capacity)*sizeof(int)+w.expert_groups.capacity*sizeof(Exl3Group) +
        (w.smallm_gate.capacity+w.smallm_up.capacity+w.smallm_down.capacity)*sizeof(Exl3Mat) +
        (w.smallm_first.capacity+w.smallm_rows.capacity)*sizeof(int);
}

// Capacity query for the wave arm's allocation invariant: true when every buffer the group entry
// points touch already holds `routes` rows (base arrays and group scratch, wmma_x when it may run),
// i.e. dispatching this tile cannot grow - and free - anything under queued work.
bool Exl3MoeWorkspace::group_tile_fits(int routes, int n_in, int n_ff, int n_out, bool wmma_scratch) const {
    if (routes < 1) return false;
    const auto& w = *impl_;
    const size_t r = (size_t) routes, wide = (size_t) std::max(n_in, n_ff);
    return w.gate.capacity >= r && w.up.capacity >= r && w.down.capacity >= r && w.weights.capacity >= r &&
           w.g.capacity >= r * (size_t) n_ff && w.u.capacity >= r * (size_t) n_ff &&
           w.h.capacity >= r * (size_t) n_ff && w.d.capacity >= r * (size_t) n_out &&
           w.xh.capacity >= r * wide && w.z.capacity >= r * (size_t) std::max(n_ff, n_out) &&
           w.group_x.capacity >= r * (size_t) n_in && w.group_y.capacity >= r * (size_t) std::max(n_ff, n_out) &&
           w.group_weight.capacity >= (size_t) std::max((size_t) n_in * n_ff, (size_t) n_ff * n_out) &&
           w.group_reconstruct.capacity >= (size_t) std::max((size_t) n_in * n_ff, (size_t) n_ff * n_out) &&
           (!wmma_scratch || w.wmma_x.capacity >= r * wide);
}

namespace {

// Compact the window's routes out of a DEVICE-resident slot table (one descriptor per slot per
// projection).  This is the only step a graph-side hit arm needs beyond the existing kernels: no
// host memory, no pageable upload, capture-friendly.  Slot ids are validated ON THE HOST by the
// entry below before anything launches - an invalid id throws before a kernel can run, so the
// gather itself never needs a fallback path (a wrong expert is never computed).
__global__ void gather_slot_mats_kernel(const Exl3Mat* __restrict__ table, const int* __restrict__ slots,
                                        int table_slots, Exl3Mat* __restrict__ out, int routes) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= routes) return;
    out[i] = table[slots[i]];
}

// Standalone compatibility callers get distinct scratch for each device, stream and host thread.
// The engine uses explicitly owned workspaces, whose lifetime matches its expert stream.
Exl3MoeWorkspace& fallback_moe_workspace(cudaStream_t stream) {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) throw std::runtime_error("exl3: no CUDA device");
    using Key = std::pair<int, uintptr_t>;
    static thread_local std::map<Key, std::unique_ptr<Exl3MoeWorkspace>> workspaces;
    auto& item = workspaces[{device, reinterpret_cast<uintptr_t>(stream)}];
    if (!item) item = std::make_unique<Exl3MoeWorkspace>();
    return *item;
}

// The device-only tail of the FFN: descriptors are ALREADY in device memory (uploaded by the host
// callers, or gathered from a device slot table by the graph-side entry below).  Everything from
// here on is launch-only, so a captured graph can replay it without host interaction.
template <bool Masked = false>
uint16_t* moe_run_experts_dev(Exl3Mat* d_gate, Exl3Mat* d_up, Exl3Mat* d_down, int n_experts,
                              const uint16_t* x, cudaStream_t st, Exl3MoeWorkspace& owner,
                              long x_stride, const int* input_rows, int k, int ff, int n,
                              uint16_t* direct_out = nullptr) {
    auto& w = owner.implementation();
    uint16_t* d_g = w.g.p; uint16_t* d_u = w.u.p; uint16_t* d_h = w.h.p;
    uint16_t* d_d = direct_out ? direct_out : w.d.p;
    float* d_xh = w.xh.p; float* d_z = w.z.p; int* d_pinv = w.pinv.p;
    int gy_g = (k / 16 + 15) / 16; if (gy_g < 1) gy_g = 1;
    int gy_d = (ff / 16 + 15) / 16; if (gy_d < 1) gy_d = 1;
    // Indexed windows already expose route-level parallelism. An optional smaller K split
    // amortizes trellis-window setup and reduces atomics; keep the established path by default.
    if (input_rows) {
        static const int cap = [] {
            const char* value = std::getenv("STRATA_EXL3_GEMV_SPLIT");
            return value ? std::max(1, std::min(16, std::atoi(value))) : std::numeric_limits<int>::max();
        }();
        gy_g = std::min(gy_g, cap); gy_d = std::min(gy_d, cap);
    }
    // STRATA_EXL3_RDNA (gfx11 and gfx12 builds): 1 = run-grouped funnel loads + packed-hash decode,
    // bit-identical to the scalar kernel; 2 = additionally fp16 weight pairs + v_dot2_f32_f16 over
    // fp16 activations (the reference engine's arithmetic).  0/absent = the established kernel.
#if defined(STRATA_EXL3_RDNA_DECODE)
    static const int rdna = [] {
        const char* value = std::getenv("STRATA_EXL3_RDNA");
        return value ? std::max(0, std::min(2, std::atoi(value))) : 0;
    }();
#else
    static constexpr int rdna = 0;
#endif
    // STRATA_EXL3_FUSE (indexed/Partials path, RDNA kernel on): 1 folds the prescale Hadamard
    // into the GEMV prologue (3 launches per projection -> 2, and the dead z-zeroing disappears);
    // 2 additionally folds postscale into the GEMV epilogue (-> 1 launch per projection);
    // 3 additionally folds the silu*up elementwise product into the down projection's prologue
    // (-> 3 launches per expert FFN: gate, up, down).
#if defined(STRATA_EXL3_RDNA_DECODE)
    const int fuse = input_rows != nullptr && rdna >= 1 ? [] {
        const char* value = std::getenv("STRATA_EXL3_FUSE");
        return value ? std::max(0, std::min(3, std::atoi(value))) : 0;
    }() : 0;
#else
    static constexpr int fuse = 0;
#endif
    // lazy opt-in scratch (see reserve): xh16 only when RDNA=2 without fusion reads it, the
    // arrival counters only when the fused epilogue (FUSE >= 2) actually runs.  Growth happens on
    // the first (warm, non-captured - the slot-table contract) call only; captured replays find
    // the capacity already there, so no allocation ever happens against a spinning graph.
    uint16_t* d_xh16 = nullptr;
    int* d_counters = nullptr;
    if (rdna == 2 && fuse == 0) {
        w.xh16.reserve((size_t) n_experts * std::max(k, ff));
        d_xh16 = w.xh16.p;
    }
    if (fuse >= 2) {
        const size_t segs = std::max((size_t) ff, (size_t) n) / 128 + 1;
        if ((size_t) n_experts * segs > w.fuse_counters.capacity) {
            w.fuse_counters.reserve((size_t) n_experts * segs);
            // A default-stream memset is not ordered before these nonblocking execution
            // streams. Zero on the owning stream, including when concurrent workspaces grow.
            if (cudaMemsetAsync(w.fuse_counters.p, 0, (size_t) n_experts * segs * sizeof(int), st) != cudaSuccess)
                throw std::runtime_error("exl3: fused-epilogue counter zeroing failed");
        }
        d_counters = w.fuse_counters.p;
    }
    auto project = [&](Exl3Mat* mats, int outputs, int parts, uint16_t* out, int ki16,
                       bool is_down = false) {
        const dim3 grid((outputs / 16 + GEMV_T - 1) / GEMV_T, parts, n_experts);
        size_t smem = 0;
        if (fuse) {
            // dynamic shared bytes for the fused prologue: the widest PARTITION's slice, in the
            // kernel's own arithmetic (part k-range = [ki16*p/parts, ki16*(p+1)/parts) k-tiles)
            int widest = 0;
            for (int p = 0; p < parts; ++p)
                widest = std::max(widest, ((ki16 * (p + 1)) / parts - (ki16 * p) / parts) * 16);
            smem = (size_t) widest * 4;
        }
        if (input_rows) {
            // Each K partition has its own output. The Hadamard pass sums them in a fixed order,
            // so a route's result does not depend on the other slots or CUDA block scheduling.
#if defined(STRATA_EXL3_RDNA_DECODE)
            if (fuse >= 2) {
                const uint16_t* xin = x;
                const uint16_t* silu_mul = nullptr;
                const long xin_stride = is_down ? (long) ff : x_stride;   // down reads the FFN slab
                if (fuse == 3 && is_down) { xin = d_g; silu_mul = d_u; }
                else if (is_down) xin = d_h;
                if (rdna == 1) {
                    if (fuse == 3 && is_down)
                        gemv_batched_rdna_kernel<GEMV_T, true, 1, true, true, true, Masked><<<grid, 256, smem, st>>>(d_xh, d_xh16, mats, d_pinv, w.partials.p, xin, xin_stride, is_down ? nullptr : input_rows, out, d_counters, silu_mul);
                    else
                        gemv_batched_rdna_kernel<GEMV_T, true, 1, true, true, false, Masked><<<grid, 256, smem, st>>>(d_xh, d_xh16, mats, d_pinv, w.partials.p, xin, xin_stride, is_down ? nullptr : input_rows, out, d_counters);
                } else {
                    if (fuse == 3 && is_down)
                        gemv_batched_rdna_kernel<GEMV_T, true, 2, true, true, true, Masked><<<grid, 256, smem, st>>>(d_xh, d_xh16, mats, d_pinv, w.partials.p, xin, xin_stride, is_down ? nullptr : input_rows, out, d_counters, silu_mul);
                    else
                        gemv_batched_rdna_kernel<GEMV_T, true, 2, true, true, false, Masked><<<grid, 256, smem, st>>>(d_xh, d_xh16, mats, d_pinv, w.partials.p, xin, xin_stride, is_down ? nullptr : input_rows, out, d_counters);
                }
                return;   // the epilogue already wrote `out`
            }
            if (fuse == 1) {
                const uint16_t* xin = is_down ? d_h : x;
                const long xin_stride = is_down ? (long) ff : x_stride;
                if (rdna == 1)
                    gemv_batched_rdna_kernel<GEMV_T, true, 1, true, false, false, Masked><<<grid, 256, smem, st>>>(d_xh, d_xh16, mats, d_pinv, w.partials.p, xin, xin_stride, is_down ? nullptr : input_rows);
                else
                    gemv_batched_rdna_kernel<GEMV_T, true, 2, true, false, false, Masked><<<grid, 256, smem, st>>>(d_xh, d_xh16, mats, d_pinv, w.partials.p, xin, xin_stride, is_down ? nullptr : input_rows);
            } else if (rdna == 1)
                gemv_batched_rdna_kernel<GEMV_T, true, 1, false, false, false, Masked><<<grid, 256, 0, st>>>(d_xh, d_xh16, mats, d_pinv, w.partials.p);
            else if (rdna == 2)
                gemv_batched_rdna_kernel<GEMV_T, true, 2, false, false, false, Masked><<<grid, 256, 0, st>>>(d_xh, d_xh16, mats, d_pinv, w.partials.p);
            else
#endif
            gemv_batched_kernel<GEMV_T, true, Masked><<<grid, 256, 0, st>>>(d_xh, mats, d_pinv, w.partials.p);
            postscale_batched_kernel<Masked><<<dim3(outputs / HAD, n_experts), HAD, 0, st>>>(w.partials.p, mats, out, outputs, parts);
        } else {
#if defined(STRATA_EXL3_RDNA_DECODE)
            if (rdna == 1)
                gemv_batched_rdna_kernel<GEMV_T, false, 1><<<grid, 256, 0, st>>>(d_xh, d_xh16, mats, d_pinv, d_z);
            else if (rdna == 2)
                gemv_batched_rdna_kernel<GEMV_T, false, 2><<<grid, 256, 0, st>>>(d_xh, d_xh16, mats, d_pinv, d_z);
            else
#endif
            gemv_batched_kernel<GEMV_T><<<grid, 256, 0, st>>>(d_xh, mats, d_pinv, d_z);
            postscale_batched_kernel<<<dim3(outputs / HAD, n_experts), HAD, 0, st>>>(d_z, mats, out, outputs);
        }
    };
    auto prescale = [&](const uint16_t* xin, long stride, Exl3Mat* mats, int kk, long nz,
                        const int* rows) {
        if (fuse) return;   // the GEMV prologue Hadamards its own slice
        // the DOWN projection reads the route-major FFN slab: no token gather there (the
        // established code passes no input_rows on that call - keep that byte-for-byte)
        prescale_batched_kernel<Masked><<<dim3(kk / HAD, n_experts), HAD, 0, st>>>(xin, stride, mats, d_xh, kk, d_z, nz, rows, d_xh16);
    };
    prescale(x, x_stride, d_gate, k, (long) n_experts * ff, input_rows);
    project(d_gate, ff, gy_g, d_g, k / 16);
    prescale(x, x_stride, d_up, k, (long) n_experts * ff, input_rows);
    project(d_up, ff, gy_g, d_u, k / 16);
    if (fuse != 3) {
        long tot = (long)n_experts * ff;
        silu_batched_kernel<Masked><<<(unsigned)((tot + 255) / 256), 256, 0, st>>>(d_g, d_u, d_h, tot, d_gate, ff);
    }
    prescale(d_h, ff, d_down, ff, (long) n_experts * n, nullptr);
    project(d_down, n, gy_d, d_d, ff / 16, true);
    return d_d;
}

uint16_t* moe_run_experts(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down, int n_experts,
                          const uint16_t* x, cudaStream_t st, Exl3MoeWorkspace* workspace,
                          long x_stride = 0, const int* input_rows = nullptr,
                          const Exl3MoeDevicePlan* plan = nullptr, uint16_t* direct_out = nullptr) {
    if (!gate || !up || !down || !x || n_experts <= 0 || n_experts > 65535)
        throw std::invalid_argument("exl3: invalid FFN arguments");
    if (plan && (!plan->gate || !plan->up || !plan->down))
        throw std::invalid_argument("exl3: invalid indexed FFN device plan");
    for (int e = 0; e < n_experts; ++e) {
        for (const Exl3Mat* m : {&gate[e], &up[e], &down[e]})
            if (!m->trellis || !m->suh || !m->svh || m->ki <= 0 || m->ki > INT32_MAX / 16 ||
                m->nj <= 0 || m->nj > INT32_MAX / 16 || m->ki % 8 || m->nj % 8 ||
                m->bits < 1 || m->bits > 8 || m->cb < 0 || m->cb > 2)
                throw std::invalid_argument("exl3: invalid FFN matrix geometry");
        if (gate[e].ki != gate[0].ki || gate[e].nj != gate[0].nj || up[e].ki != gate[0].ki ||
            up[e].nj != gate[0].nj || down[e].ki != gate[0].nj || down[e].nj != down[0].nj)
            throw std::invalid_argument("exl3: mixed FFN dimensions");
    }
    const int k = gate[0].ki * 16;
    const int ff = gate[0].nj * 16;
    const int n = down[0].nj * 16;
    auto& owner = workspace ? *workspace : fallback_moe_workspace(st);
    owner.reserve(n_experts, k, ff, n, input_rows != nullptr);
    auto& w = owner.implementation();
    Exl3Mat* d_gate = plan ? const_cast<Exl3Mat*>(plan->gate) : w.gate.p;
    Exl3Mat* d_up = plan ? const_cast<Exl3Mat*>(plan->up) : w.up.p;
    Exl3Mat* d_down = plan ? const_cast<Exl3Mat*>(plan->down) : w.down.p;
    const size_t msz = (size_t) n_experts * sizeof(Exl3Mat);
    // checked: a failed descriptor upload must surface rather than leave the kernels reading
    // stale descriptors
    if (!plan && (cudaMemcpyAsync(d_gate, gate, msz, cudaMemcpyHostToDevice, st) != cudaSuccess ||
        cudaMemcpyAsync(d_up, up, msz, cudaMemcpyHostToDevice, st) != cudaSuccess ||
        cudaMemcpyAsync(d_down, down, msz, cudaMemcpyHostToDevice, st) != cudaSuccess)) {
        (void) cudaGetLastError();
        throw std::runtime_error("exl3: descriptor upload failed");
    }
    return moe_run_experts_dev(d_gate, d_up, d_down, n_experts, x, st, owner,
                              x_stride, input_rows, k, ff, n, direct_out);
}

}  // namespace

void exl3_moe_ffn(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down, const float* weights,
                  int n_experts, const uint16_t* x, uint16_t* out, void* stream, Exl3MoeWorkspace* workspace) {
    if (n_experts <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
    const int n = down[0].nj * 16;
    uint16_t* d_d = moe_run_experts(gate, up, down, n_experts, x, st, workspace);
    auto& w = (workspace ? *workspace : fallback_moe_workspace(st)).implementation();
    float* d_weights = w.weights.p;
    (void)cudaMemcpyAsync(d_weights, weights, (size_t)n_experts * 4, cudaMemcpyHostToDevice, st);
    // one kernel, not two: the accumulator's f32 buffer is never read by anything else
    accum_out_kernel<<<(n + 255) / 256, 256, 0, st>>>(d_d, d_weights, n_experts, n, out);
}

void exl3_moe_rows_part(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down, int k, const uint16_t* x,
                        uint16_t* out, void* stream, Exl3MoeWorkspace* workspace) {
    if (k <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
    const int n = down[0].nj * 16;
    uint16_t* d_d = moe_run_experts(gate, up, down, k, x, st, workspace);
    (void) cudaMemcpyAsync(out, d_d, (size_t)k * n * 2, cudaMemcpyDeviceToDevice, st);
}

void exl3_moe_rows_slot_table(const Exl3Mat* d_gate_table, const Exl3Mat* d_up_table, const Exl3Mat* d_down_table,
                              int table_slots, const int* d_slot_of_route, int routes, int n_in, int n_ff, int n_out,
                              const uint16_t* x, const int* input_rows, uint16_t* out, void* stream,
                              Exl3MoeWorkspace& workspace) {
    if (!d_gate_table || !d_up_table || !d_down_table || !d_slot_of_route || !x || !input_rows || !out ||
        table_slots <= 0 || routes <= 0 || routes > 65535 ||
        n_in <= 0 || n_in % 128 || n_ff <= 0 || n_ff % 128 || n_out <= 0 || n_out % 128)
        throw std::invalid_argument("exl3: invalid slot-table FFN arguments");
    const auto st = (cudaStream_t) stream;
    // Validate the slot ids on the host BEFORE anything launches: an id outside
    // [0, table_slots) throws here (no kernel runs, nothing computes a wrong expert, no device
    // error state).  Inside stream capture a sync is illegal - the caller validates on a warm,
    // non-captured call first (the contract in the header says so).
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(st, &capturing) != cudaSuccess) capturing = cudaStreamCaptureStatusActive;
    if (capturing == cudaStreamCaptureStatusNone) {
        std::vector<int> ids((size_t) routes);
        if (cudaMemcpyAsync(ids.data(), d_slot_of_route, (size_t) routes * sizeof(int), cudaMemcpyDeviceToHost, st) != cudaSuccess ||
            cudaStreamSynchronize(st) != cudaSuccess)
            throw std::runtime_error("exl3: slot-table FFN: slot id readback failed");
        for (int r = 0; r < routes; ++r)
            if (ids[(size_t) r] < 0 || ids[(size_t) r] >= table_slots)
                throw std::runtime_error("exl3: slot-table FFN: invalid slot id " + std::to_string(ids[(size_t) r]) +
                                         " at route " + std::to_string(r));
    }
    workspace.reserve(routes, n_in, n_ff, n_out, true);
    auto& w = workspace.implementation();
    const unsigned gb = (unsigned) ((routes + 63) / 64);
    gather_slot_mats_kernel<<<gb, 64, 0, st>>>(d_gate_table, d_slot_of_route, table_slots, w.gate.p, routes);
    gather_slot_mats_kernel<<<gb, 64, 0, st>>>(d_up_table, d_slot_of_route, table_slots, w.up.p, routes);
    gather_slot_mats_kernel<<<gb, 64, 0, st>>>(d_down_table, d_slot_of_route, table_slots, w.down.p, routes);
    // x is the dense window token buffer [tokens, n_in]: the gather stride is the row pitch n_in
    // (the host callers pass the same value as x_stride).
    uint16_t* rows = moe_run_experts_dev(w.gate.p, w.up.p, w.down.p, routes, x, st, workspace, (long) n_in,
                                         input_rows, n_in, n_ff, n_out);
    if (cudaMemcpyAsync(out, rows, (size_t) routes * n_out * sizeof(uint16_t), cudaMemcpyDeviceToDevice, st) != cudaSuccess)
        throw std::runtime_error("exl3: slot-table FFN output copy failed");
}

void exl3_moe_rows_indexed(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down, int routes,
                           const uint16_t* x, const int* input_rows, uint16_t* out, void* stream,
                           Exl3MoeWorkspace& workspace) {
    if (routes == 0) return;
    if (routes < 0 || routes > 65535 || !gate || !up || !down || !x || !input_rows || !out)
        throw std::invalid_argument("exl3: invalid indexed FFN arguments");
    const auto st = (cudaStream_t) stream;
    uint16_t* rows = moe_run_experts(gate, up, down, routes, x, st, &workspace,
                                     (long) gate[0].ki * 16, input_rows);
    if (cudaMemcpyAsync(out, rows, (size_t) routes * down[0].nj * 16 * sizeof(uint16_t),
                        cudaMemcpyDeviceToDevice, st) != cudaSuccess)
        throw std::runtime_error("exl3: indexed FFN output copy failed");
}

void exl3_moe_rows_indexed_device_plan(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down,
                            int routes, const uint16_t* x, const int* input_rows, uint16_t* out,
                            const Exl3MoeDevicePlan& plan, void* stream, Exl3MoeWorkspace& workspace) {
    if (routes == 0) return;
    if (!input_rows || !out) throw std::invalid_argument("exl3: invalid indexed device-plan output/rows");
    (void) moe_run_experts(gate, up, down, routes, x, (cudaStream_t)stream, &workspace,
                           gate ? (long)gate[0].ki * 16 : 0, input_rows, &plan, out);
}

const uint16_t* exl3_moe_rows_masked(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down,
                          int routes, int n_in, int n_ff, int n_out, const uint16_t* x,
                          const int* input_rows, void* stream,
                          Exl3MoeWorkspace& workspace) {
    // No reserve, readback or synchronization here: all storage belongs to the graph owner.
    return moe_run_experts_dev<true>(const_cast<Exl3Mat*>(gate), const_cast<Exl3Mat*>(up),
        const_cast<Exl3Mat*>(down), routes, x, (cudaStream_t) stream, workspace,
        n_in, input_rows, n_in, n_ff, n_out);
}

namespace {

// Device tail of the prototype grouped FFN.  Constraint: same launch cadence and the same
// ROUTE-major scratch layout as moe_run_experts_dev's indexed arm (prescale -> gemv -> postscale
// -> silu -> down), so the two arms are drop-in comparable on one route order.  d_g* are
// per-GROUP descriptors resident on the device; d_gfirst/d_grows are the per-group route table.
// `max_rows` is the window's LARGEST group (host-validated) and picks the GEMV instantiation,
// so only windows that really hold m-row groups pay the wider accumulators.
uint16_t* moe_run_experts_grouped_dev(const Exl3Mat* d_gate, const Exl3Mat* d_up, const Exl3Mat* d_down,
                                      const int* d_gfirst, const int* d_grows, int groups, int routes,
                                      int max_rows, const uint16_t* x, cudaStream_t st, Exl3MoeWorkspace& owner,
                                      long x_stride, const int* input_rows, int k, int ff, int n, int cb,
                                      bool fused, uint16_t* direct_out = nullptr) {
    auto& w = owner.implementation();
    uint16_t* d_g = w.g.p; uint16_t* d_u = w.u.p; uint16_t* d_h = w.h.p;
    uint16_t* d_d = direct_out ? direct_out : w.d.p;
    float* d_xh = w.xh.p; float* d_z = w.z.p; int* d_pinv = w.pinv.p;
    int gy_g = (k / 16 + 15) / 16; if (gy_g < 1) gy_g = 1;
    int gy_d = (ff / 16 + 15) / 16; if (gy_d < 1) gy_d = 1;
    // Same optional K split as the indexed arm (STRATA_EXL3_GEMV_SPLIT): the default is uncapped,
    // and the two arms must be configured identically for a like-for-like comparison.
    static const int cap = [] {
        const char* value = std::getenv("STRATA_EXL3_GEMV_SPLIT");
        return value ? std::max(1, std::min(16, std::atoi(value))) : std::numeric_limits<int>::max();
    }();
    gy_g = std::min(gy_g, cap); gy_d = std::min(gy_d, cap);
    if (fused) {
        // Tile width changes CTA ownership, not a row's FMA/reduction order.
        // The smaller opt-in trades extra prologues/arrivals for fewer registers.
        static const int tiles = [] {
            const char* v = std::getenv("STRATA_EXL3_MULTIROW_TILES");
            return v && std::atoi(v) == 2 ? 2 : GEMV_T;
        }();
        auto project_fused = [&](const Exl3Mat* mats, int outputs, int parts, int ki, bool down) {
            int widest = 0;
            for (int p = 0; p < parts; ++p)
                widest = std::max(widest, (ki * (p + 1) / parts - ki * p / parts) * 16);
            const int width = max_rows <= 2 ? max_rows : 4;
            const size_t smem = (size_t) widest * width * sizeof(float);
            const dim3 grid((outputs / 16 + tiles - 1) / tiles, parts, groups);
            auto launch = [&](auto tag, auto tile_tag) {
                constexpr int R = decltype(tag)::value;
                constexpr int T = decltype(tile_tag)::value;
                if (down)
                    gemv_grouped_rdna_kernel<T, R, true, true, true><<<grid, 256, smem, st>>>(
                        d_xh, mats, d_gfirst, d_grows, w.partials.p, d_g, ff, nullptr,
                        d_d, w.fuse_counters.p, d_u);
                else
                    gemv_grouped_rdna_kernel<T, R, true, true><<<grid, 256, smem, st>>>(
                        d_xh, mats, d_gfirst, d_grows, w.partials.p, x, x_stride, input_rows,
                        mats == d_gate ? d_g : d_u, w.fuse_counters.p);
            };
            auto by_width = [&](auto tile_tag) {
                if (width == 1) launch(std::integral_constant<int, 1>{}, tile_tag);
                else if (width == 2) launch(std::integral_constant<int, 2>{}, tile_tag);
                else launch(std::integral_constant<int, 4>{}, tile_tag);
            };
            if (tiles == 2) by_width(std::integral_constant<int, 2>{});
            else by_width(std::integral_constant<int, 4>{});
        };
        project_fused(d_gate, ff, gy_g, k / 16, false);
        project_fused(d_up, ff, gy_g, k / 16, false);
        project_fused(d_down, n, gy_d, ff / 16, true);
        return d_d;
    }
    // STRATA_EXL3_RDNA >= 1 with the mul1 codebook runs the RDNA twin (run funnels + packed
    // hash decode paid once per GROUP); anything else keeps the scalar prototype.  The cb
    // check reads the HOST descriptors the caller validated (the device copies are not
    // host-readable here).
    const bool rdna = rdna_level() >= 1 && cb == 2;
    auto project = [&](const Exl3Mat* gmats, int outputs, int parts, uint16_t* out) {
        const dim3 grid((outputs / 16 + GEMV_T - 1) / GEMV_T, parts, groups);
        if (rdna) {
            switch (max_rows) {
                case 1: gemv_grouped_rdna_kernel<GEMV_T, 1, true><<<grid, 256, 0, st>>>(
                            d_xh, gmats, d_gfirst, d_grows, w.partials.p); break;
                case 2: gemv_grouped_rdna_kernel<GEMV_T, 2, true><<<grid, 256, 0, st>>>(
                            d_xh, gmats, d_gfirst, d_grows, w.partials.p); break;
                case 3: gemv_grouped_rdna_kernel<GEMV_T, 3, true><<<grid, 256, 0, st>>>(
                            d_xh, gmats, d_gfirst, d_grows, w.partials.p); break;
                case 4: gemv_grouped_rdna_kernel<GEMV_T, 4, true><<<grid, 256, 0, st>>>(
                            d_xh, gmats, d_gfirst, d_grows, w.partials.p); break;
                case 5: gemv_grouped_rdna_kernel<GEMV_T, 5, true><<<grid, 256, 0, st>>>(
                            d_xh, gmats, d_gfirst, d_grows, w.partials.p); break;
                default: gemv_grouped_rdna_kernel<GEMV_T, SMALL_M_MAX_ROWS, true><<<grid, 256, 0, st>>>(
                            d_xh, gmats, d_gfirst, d_grows, w.partials.p); break;
            }
        } else {
            switch (max_rows) {
                case 1: gemv_grouped_kernel<GEMV_T, 1, true><<<grid, 256, 0, st>>>(
                            d_xh, gmats, d_gfirst, d_grows, d_pinv, w.partials.p); break;
                case 2: gemv_grouped_kernel<GEMV_T, 2, true><<<grid, 256, 0, st>>>(
                            d_xh, gmats, d_gfirst, d_grows, d_pinv, w.partials.p); break;
                case 3: gemv_grouped_kernel<GEMV_T, 3, true><<<grid, 256, 0, st>>>(
                            d_xh, gmats, d_gfirst, d_grows, d_pinv, w.partials.p); break;
                case 4: gemv_grouped_kernel<GEMV_T, 4, true><<<grid, 256, 0, st>>>(
                            d_xh, gmats, d_gfirst, d_grows, d_pinv, w.partials.p); break;
                case 5: gemv_grouped_kernel<GEMV_T, 5, true><<<grid, 256, 0, st>>>(
                            d_xh, gmats, d_gfirst, d_grows, d_pinv, w.partials.p); break;
                default: gemv_grouped_kernel<GEMV_T, SMALL_M_MAX_ROWS, true><<<grid, 256, 0, st>>>(
                            d_xh, gmats, d_gfirst, d_grows, d_pinv, w.partials.p); break;
            }
        }
        postscale_grouped_kernel<<<dim3(outputs / HAD, groups), HAD, 0, st>>>(
            w.partials.p, gmats, d_gfirst, d_grows, out, outputs, parts);
    };
    prescale_grouped_kernel<<<dim3(k / HAD, groups), HAD, 0, st>>>(
        x, x_stride, d_gate, d_gfirst, d_grows, d_xh, k, d_z, (long) routes * ff, input_rows);
    project(d_gate, ff, gy_g, d_g);
    prescale_grouped_kernel<<<dim3(k / HAD, groups), HAD, 0, st>>>(
        x, x_stride, d_up, d_gfirst, d_grows, d_xh, k, d_z, (long) routes * ff, input_rows);
    project(d_up, ff, gy_g, d_u);
    { long tot = (long) routes * ff;
      silu_batched_kernel<<<(unsigned)((tot + 255) / 256), 256, 0, st>>>(d_g, d_u, d_h, tot); }
    prescale_grouped_kernel<<<dim3(ff / HAD, groups), HAD, 0, st>>>(
        d_h, ff, d_down, d_gfirst, d_grows, d_xh, ff, d_z, (long) routes * n, nullptr);
    project(d_down, n, gy_d, d_d);
    return d_d;
}

}  // namespace

// Grouped small-M window FFN (unfused diagnostic or opt-in fused peer variant).
// gate/up/down are HOST descriptors, ONE PER GROUP (a group is ONE expert);
// group_first/group_rows are HOST tables of the groups as CONTIGUOUS SLICES of the route order
// (group_first[0]==0, monotone, sum(group_rows)==routes, 1<=group_rows<=SMALL_M_MAX_ROWS; the
// engine's Exl3RoutePlan emits exactly this).  input_rows is the DEVICE per-route token table;
// x is DEVICE [tokens, gate[0].ki*16]; out is DEVICE [routes, down[0].nj*16] fp16 in the SAME
// route-major layout as exl3_moe_rows_indexed, so the arms are substitutable downstream.
// The route table and matrix geometry are validated ON THE HOST before anything launches; a
// malformed plan throws and no kernel runs.  Pre-reserve with reserve_small_m before timed or
// captured execution.
bool exl3_fused_multirow_supported() {
    const char* f = std::getenv("STRATA_EXL3_FUSE");
    return rdna_level() == 1 && f && std::atoi(f) == 3;
}

static uint16_t* moe_rows_grouped_impl(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down,
                           const int* group_first, const int* group_rows, int groups,
                           int routes, const uint16_t* x, const int* input_rows, uint16_t* out,
                            void* stream, Exl3MoeWorkspace& workspace, bool fused,
                            const Exl3MoeDevicePlan* plan, bool direct_output = false) {
    if (routes == 0) return nullptr;
    if (routes < 0 || routes > 65535 || groups < 1 || groups > routes ||
        !gate || !up || !down || !group_first || !group_rows || !x || !input_rows || (!out && !plan))
        throw std::invalid_argument("exl3: invalid grouped FFN arguments");
    if (plan && (!plan->gate || !plan->up || !plan->down || !plan->first || !plan->rows))
        throw std::invalid_argument("exl3: invalid grouped FFN device plan");
    int max_rows = 1;   // the window's widest group: picks the tightest GEMV instantiation
    // The route table is validated ON THE HOST before anything launches, so a malformed plan can
    // never reach a kernel (a wrong expert or row is never computed).  Rows above the compiled
    // group width are rejected: the caller chunks such groups or stays on the indexed entry.
    {
        int prefix = 0, widest = 1;
        for (int g = 0; g < groups; ++g) {
            if (group_first[g] != prefix || group_rows[g] < 1 || group_rows[g] > SMALL_M_MAX_ROWS)
                throw std::invalid_argument("exl3: grouped FFN route table is not a valid "
                                            "contiguous 1.." + std::to_string(SMALL_M_MAX_ROWS) +
                                            "-row slice of the route order");
            prefix += group_rows[g];
            widest = std::max(widest, group_rows[g]);
        }
        if (prefix != routes)
            throw std::invalid_argument("exl3: grouped FFN route table does not cover the routes");
        max_rows = widest;
    }
    for (int g = 0; g < groups; ++g) {
        for (const Exl3Mat* m : {&gate[g], &up[g], &down[g]})
            if (!m->trellis || !m->suh || !m->svh || m->ki <= 0 || m->ki > INT32_MAX / 16 ||
                m->nj <= 0 || m->nj > INT32_MAX / 16 || m->ki % 8 || m->nj % 8 ||
                m->bits < 1 || m->bits > 8 || m->cb < 0 || m->cb > 2)
                throw std::invalid_argument("exl3: invalid grouped FFN matrix geometry");
        if (gate[g].ki != gate[0].ki || gate[g].nj != gate[0].nj || up[g].ki != gate[0].ki ||
            up[g].nj != gate[0].nj || down[g].ki != gate[0].nj || down[g].nj != down[0].nj)
            throw std::invalid_argument("exl3: mixed grouped FFN dimensions");
    }
    const int k = gate[0].ki * 16, ff = gate[0].nj * 16, n = down[0].nj * 16;
    if (fused) {
        const char* f = std::getenv("STRATA_EXL3_FUSE");
        if (rdna_level() != 1 || !f || std::atoi(f) != 3 || max_rows > 4 ||
            (size_t) std::max(k, ff) * (max_rows <= 2 ? max_rows : 4) * sizeof(float) > 40 * 1024)
            throw std::invalid_argument("exl3: fused grouped FFN requires gfx11 RDNA=1/FUSE=3, <=4 rows and bounded LDS");
        for (int g = 0; g < groups; ++g)
            if (gate[g].cb != 2 || up[g].cb != 2 || down[g].cb != 2)
                throw std::invalid_argument("exl3: fused grouped FFN requires mul1 codebooks");
        workspace.reserve_runtime_scratch(routes, k, ff, n);
    }
    workspace.reserve_small_m(routes, groups, k, ff, n);
    auto& w = workspace.implementation();
    const cudaStream_t st = (cudaStream_t) stream;
    const size_t msz = (size_t) groups * sizeof(Exl3Mat);
    // checked: a failed plan/descriptor upload must surface, not leave kernels reading stale data
    if (!plan && (cudaMemcpyAsync(w.smallm_gate.p, gate, msz, cudaMemcpyHostToDevice, st) != cudaSuccess ||
        cudaMemcpyAsync(w.smallm_up.p, up, msz, cudaMemcpyHostToDevice, st) != cudaSuccess ||
        cudaMemcpyAsync(w.smallm_down.p, down, msz, cudaMemcpyHostToDevice, st) != cudaSuccess ||
        cudaMemcpyAsync(w.smallm_first.p, group_first, (size_t) groups * sizeof(int),
                        cudaMemcpyHostToDevice, st) != cudaSuccess ||
        cudaMemcpyAsync(w.smallm_rows.p, group_rows, (size_t) groups * sizeof(int),
                        cudaMemcpyHostToDevice, st) != cudaSuccess)) {
        (void) cudaGetLastError();
        throw std::runtime_error("exl3: grouped FFN plan upload failed");
    }
    uint16_t* rows = moe_run_experts_grouped_dev(plan ? plan->gate : w.smallm_gate.p,
                                                 plan ? plan->up : w.smallm_up.p,
                                                 plan ? plan->down : w.smallm_down.p,
                                                 plan ? plan->first : w.smallm_first.p,
                                                 plan ? plan->rows : w.smallm_rows.p, groups, routes,
                                                 max_rows, x, st, workspace, (long) k, input_rows, k, ff, n,
                                                  gate[0].cb, fused, direct_output ? out : nullptr);
    if (out && !direct_output && cudaMemcpyAsync(out, rows, (size_t) routes * n * sizeof(uint16_t),
                        cudaMemcpyDeviceToDevice, st) != cudaSuccess)
        throw std::runtime_error("exl3: grouped FFN output copy failed");
    return rows;
}

void exl3_moe_rows_grouped(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down,
                           const int* group_first, const int* group_rows, int groups,
                           int routes, const uint16_t* x, const int* input_rows, uint16_t* out,
                           void* stream, Exl3MoeWorkspace& workspace, bool fused) {
    (void) moe_rows_grouped_impl(gate, up, down, group_first, group_rows, groups, routes,
                               x, input_rows, out, stream, workspace, fused, nullptr);
}

const uint16_t* exl3_moe_rows_grouped_device_plan(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down,
                           const int* group_first, const int* group_rows, int groups,
                           int routes, const uint16_t* x, const int* input_rows,
                           const Exl3MoeDevicePlan& plan, void* stream,
                           Exl3MoeWorkspace& workspace, bool fused, uint16_t* direct_out) {
    return moe_rows_grouped_impl(gate, up, down, group_first, group_rows, groups, routes,
                                x, input_rows, direct_out, stream, workspace, fused, &plan, direct_out != nullptr);
}

namespace {
__global__ void decode_group_weight(Exl3Mat m, float* out) {
    const int i = blockIdx.y, j = blockIdx.x, t = threadIdx.x;
    const uint16_t* tw = m.trellis + (size_t) (i * m.nj + j) * 16 * m.bits;
    const int group = t >> 3, slot = t & 7;
    const int r0 = (group % 4) * 2, c0 = group / 4;
    const int r = r0 + ((slot & 3) >= 2 ? 8 : 0) + (slot & 1);
    const int c = c0 + (slot >= 4 ? 8 : 0);
    out[(size_t) (i * 16 + r) * (m.nj * 16) + j * 16 + c] =
        dh2f(decode_window(m.cb, (uint16_t) read_window(tw, t, m.bits)));
}
// Transform 32 columns together: contiguous loads/stores instead of a one-thread block's
// 128 strided values. No allocation or global synchronisation between expert groups.
__global__ void group_had_rows(float* w, int n) {
    __shared__ float tile[HAD][32];
    const int col = blockIdx.x * 32 + threadIdx.x, rb = blockIdx.y * HAD;
    for (int r = threadIdx.y; r < HAD; r += blockDim.y) tile[r][threadIdx.x] = w[(size_t) (rb + r) * n + col];
    __syncthreads();
    for (int step = 1; step < HAD; step <<= 1) {
        for (int r = threadIdx.y; r < HAD; r += blockDim.y)
            if (!(r & step)) {
                const float a = tile[r][threadIdx.x], b = tile[r | step][threadIdx.x];
                tile[r][threadIdx.x] = a + b; tile[r | step][threadIdx.x] = a - b;
            }
        __syncthreads();
    }
    for (int r = threadIdx.y; r < HAD; r += blockDim.y)
        w[(size_t) (rb + r) * n + col] = tile[r][threadIdx.x] * 0.08838834764831845f;
}
__global__ void group_had_cols_pack(const float* w, int n, const uint16_t* suh, const uint16_t* svh, uint16_t* out) {
    __shared__ float tile[HAD];
    const int r = blockIdx.x, col = blockIdx.y * HAD + threadIdx.x, t = threadIdx.x;
    tile[t] = w[(size_t) r * n + col];
    __syncthreads();
    for (int step = 1; step < HAD; step <<= 1) {
        if (!(t & step)) {
            const float a = tile[t], b = tile[t | step];
            tile[t] = a + b; tile[t | step] = a - b;
        }
        __syncthreads();
    }
    out[(size_t) r * n + col] = __half_as_ushort(__float2half_rn(
        ((tile[t] * 0.08838834764831845f) * dh2f(suh[r])) * dh2f(svh[col])));
}
__global__ void gather_group_rows(const uint16_t* x, const int* input_rows, uint16_t* out, int rows, int n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t) rows * n) out[i] = x[(int64_t) input_rows[i / n] * n + i % n];
}
__global__ void round_group_output(const float* in, float* out, int64_t count) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = __half2float(__float2half_rn(in[i]));
}
}

void exl3_moe_group_gemm(const Exl3Mat& gate, const Exl3Mat& up, const Exl3Mat& down, int rows,
                         const uint16_t* x, const int* input_rows, float* out, void* stream,
                         Exl3MoeWorkspace& workspace) {
    for (const Exl3Mat* matrix : {&gate, &up, &down})
        if (!matrix->trellis || !matrix->suh || !matrix->svh || matrix->bits < 1 || matrix->bits > 8 ||
            matrix->cb < 0 || matrix->cb > 2 || matrix->ki <= 0 || matrix->ki > INT32_MAX / 16 ||
            matrix->nj <= 0 || matrix->nj > INT32_MAX / 16 || matrix->ki % 8 || matrix->nj % 8)
            throw std::invalid_argument("exl3: invalid grouped GEMM matrix");
    if (rows <= 0 || rows > 65535 || !x || !input_rows || !out || gate.ki != up.ki || gate.nj != up.nj ||
        down.ki != gate.nj || gate.ki <= 0 || gate.nj <= 0 || down.nj <= 0)
        throw std::invalid_argument("exl3: invalid grouped GEMM geometry");
    const int k = gate.ki * 16, ff = gate.nj * 16, n = down.nj * 16;
    workspace.reserve(rows, k, ff, n);
    auto& w = workspace.implementation();
    w.group_x.reserve((size_t) rows * k);
    w.group_weight.reserve(std::max((size_t) k * ff, (size_t) ff * n));
    w.group_reconstruct.reserve(std::max((size_t) k * ff, (size_t) ff * n));
    w.group_y.reserve((size_t) rows * std::max(ff, n));
    const auto st = (cudaStream_t) stream;
    if (!w.blas && cublasCreate(&w.blas) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("exl3: cannot create grouped GEMM handle");
    if (cublasSetStream(w.blas, st) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("exl3: cannot set grouped GEMM stream");
    gather_group_rows<<<(unsigned) (((int64_t) rows * k + 255) / 256), 256, 0, st>>>(x, input_rows, w.group_x.p, rows, k);
    auto project = [&](const Exl3Mat& m, const uint16_t* input) {
        const int in = m.ki * 16, outputs = m.nj * 16;
        decode_group_weight<<<dim3(m.nj, m.ki), 256, 0, st>>>(m, w.group_reconstruct.p);
        group_had_rows<<<dim3(outputs / 32, in / HAD), dim3(32, 4), 0, st>>>(w.group_reconstruct.p, outputs);
        group_had_cols_pack<<<dim3(in, outputs / HAD), HAD, 0, st>>>(
            w.group_reconstruct.p, outputs, m.suh, m.svh, w.group_weight.p);
        const float alpha = 1.0f, beta = 0.0f;
        // Row-major W[in,out] is column-major W[out,in]; X[rows,in] is X[in,rows].
        if (cublasGemmEx(w.blas, CUBLAS_OP_N, CUBLAS_OP_N, outputs, rows, in, &alpha,
                         w.group_weight.p, CUDA_R_16F, outputs, input, CUDA_R_16F, in, &beta,
                         w.group_y.p, CUDA_R_32F, outputs, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("exl3: grouped tensor-core GEMM failed");
    };
    project(gate, w.group_x.p);
    const int64_t hidden = (int64_t) rows * ff;
    f32_to_f16_out_kernel<<<(unsigned) ((hidden + 255) / 256), 256, 0, st>>>(w.group_y.p, w.g.p, (int) hidden);
    project(up, w.group_x.p);
    f32_to_f16_out_kernel<<<(unsigned) ((hidden + 255) / 256), 256, 0, st>>>(w.group_y.p, w.u.p, (int) hidden);
    silu_batched_kernel<<<(unsigned) ((hidden + 255) / 256), 256, 0, st>>>(w.g.p, w.u.p, w.h.p, hidden);
    project(down, w.h.p);
    round_group_output<<<(unsigned) (((int64_t) rows * n + 255) / 256), 256, 0, st>>>(w.group_y.p, out, (int64_t) rows * n);
}

// ---- the gfx11 (RDNA3) matrix-core group FFN ----
//
// The technique is nasone32's llama.cpp EXL3 fork (`exl3_wmma_mul1_routed_v4`, MIT; itself carrying
// turboderp's ExLlamaV3 decode, MIT): the trellis windows are decoded straight into the B operand of
// v_wmma_f32_16x16x16_f16 - the packed tile's tensor-core interleave is exactly the fragment layout -
// so no FP16 weight is ever materialized.  Strata calls this per (expert group, tile) like
// exl3_moe_group_gemm, so the fork's CSR route table is unnecessary: this expert's routes are the
// contiguous slab the gather writes.  A tiles are staged in LDS once per k-chunk and reused by all
// 16 column warps.
#if defined(STRATA_WMMA_GFX11)
namespace {

__device__ __forceinline__ uint32_t exl3_funnel(uint32_t lo, uint32_t hi, int shift) {
    return (uint32_t) (((uint64_t(hi) << 32) | lo) >> shift);
}

// sum of the four UNSIGNED bytes of x plus `add`, one v_dot4_i32_iu8 on gfx11: the mul1 codebook's
// byte sum without extracting the bytes.
__device__ __forceinline__ uint32_t exl3_bytesum(uint32_t x, uint32_t add) {
    return (uint32_t) __builtin_amdgcn_sudot4(false, x, false, 0x01010101u, add, false);
}

// Two 16-bit windows -> two f16 weights, the hfma2 form of the mul1 codebook (decode_window's cb==2
// branch; the exponent-25 intermediate makes the bytesum fold exact).
__device__ __forceinline__ __half2 exl3_mul1_h2(uint32_t x0, uint32_t x1) {
    x0 *= 0x83DCD12Du;
    x1 *= 0x83DCD12Du;
    const uint16_t s0 = (uint16_t) exl3_bytesum(x0, 0x6400);
    const uint16_t s1 = (uint16_t) exl3_bytesum(x1, 0x6400);
    return __hfma2(__halves2half2(__ushort_as_half(s0), __ushort_as_half(s1)),
                   __half2half2(__ushort_as_half((uint16_t) 0x1eee)),
                   __half2half2(__ushort_as_half((uint16_t) 0xc931)));
}

struct Exl3Frag { __half2 v[2]; };

// The fragment decode of the fork's exl3.cu (MIT): 8 consecutive stored positions per lane from two
// 32-bit words of the packed tile, K3 special-cased like the reference (llama.cpp exl3_decode_mul1).
__device__ __forceinline__ void exl3_decode_k3_mul1(const uint32_t* ptr, int offset, Exl3Frag& f0, Exl3Frag& f1) {
    const int b1 = (offset + 257) * 3;
    const int b0 = b1 - 16;
    const int b2 = b1 + 3 * 7;
    const int i0 = b0 / 32, i2 = (b2 - 1) / 32;
    const int shift = (i2 + 1) * 32 - b2;
    const uint32_t lo = ptr[i2 % 24], hi = ptr[i0 % 24];
    const uint32_t w7 = exl3_funnel(lo, hi, shift);
    const uint32_t w6 = w7 >> 3, w5 = w6 >> 3, w4 = w5 >> 3;
    const uint32_t w3 = exl3_funnel(lo, hi, shift + 12);
    const uint32_t w2 = w3 >> 3, w1 = w2 >> 3, w0 = w1 >> 3;
    f0.v[0] = exl3_mul1_h2(w0 & 0xffffu, w1 & 0xffffu);
    f0.v[1] = exl3_mul1_h2(w2 & 0xffffu, w3 & 0xffffu);
    f1.v[0] = exl3_mul1_h2(w4 & 0xffffu, w5 & 0xffffu);
    f1.v[1] = exl3_mul1_h2(w6 & 0xffffu, w7 & 0xffffu);
}

template <int bits>
__device__ __forceinline__ void exl3_decode_mul1_4(const uint32_t* ptr, int offset, Exl3Frag& frag) {
    constexpr int words = 8 * bits;
    const int b0 = (offset + 257) * bits - 16;
    const int b2 = b0 + 3 * bits + 16;
    const int i0 = b0 / 32, i2 = (b2 - 1) / 32;
    const int shift = (i2 + 1) * 32 - b2;
    const uint32_t lo = ptr[i2 % words], hi = ptr[i0 % words];
    const uint32_t w3 = exl3_funnel(lo, hi, shift) & 0xffffu;
    const uint32_t w2 = exl3_funnel(lo, hi, shift + bits) & 0xffffu;
    const uint32_t w1 = exl3_funnel(lo, hi, shift + 2 * bits) & 0xffffu;
    const uint32_t w0 = exl3_funnel(lo, hi, shift + 3 * bits) & 0xffffu;
    frag.v[0] = exl3_mul1_h2(w0, w1);
    frag.v[1] = exl3_mul1_h2(w2, w3);
}

template <int bits>
__device__ __forceinline__ void exl3_decode_mul1(const uint32_t* ptr, int offset, Exl3Frag& f0, Exl3Frag& f1) {
    if constexpr (bits == 3) {
        exl3_decode_k3_mul1(ptr, offset, f0, f1);
    } else {
        static_assert(bits == 4 || bits == 5 || bits == 6, "the fragment decode covers K3..K6");
        exl3_decode_mul1_4<bits>(ptr, offset, f0);
        exl3_decode_mul1_4<bits>(ptr, offset + 4, f1);
    }
}

// The routed projection on the matrix cores: y[r, n] = sum_k A[r, k] * What[k, n], A = the gathered
// h128(suh . x) slab in f16, B decoded from the trellis per (k-tile, 16-column subtile).  One block
// per (16*row_tiles row tile, 256-column band); 512 threads = 16 column warps.  The k loop pairs two
// k-tiles per step (the fork's K=32 form): both B fragments decode up front, then two chained mma per
// row tile.  Admissible shapes (ki % 8) make rows_total a multiple of 8, so kct is always even; the
// odd-tail step below is a guard, not a live path.
template <int bits, int row_tiles, bool batched = false>
__global__ void __launch_bounds__(512) routed_wmma_kernel(const uint16_t* __restrict__ input16,
                                                          const uint16_t* __restrict__ trellis,
                                                          float* __restrict__ output,
                                                          int rows, int in_features, int out_features,
                                                          const Exl3Group* groups = nullptr, int projection = 0) {
    if constexpr (batched) {
        const Exl3Group group = groups[blockIdx.z];
        const Exl3Mat m = projection == 0 ? group.gate : projection == 1 ? group.up : group.down;
        input16 += (size_t) group.first * in_features;
        output += (size_t) group.first * out_features;
        trellis = m.trellis;
        rows = group.rows;
    }
    constexpr int warps = 16;
    constexpr int n_group_cols = warps * 16;   // 256 columns per pass
    constexpr int k_chunk = 16;                // A rows staged in LDS per pass (32 measured -8% in the fork)
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int lo = lane & 15, hi = lane >> 4;
    const int rows_total = in_features / 16;
    const int n_groups = (out_features + n_group_cols - 1) / n_group_cols;
    __shared__ __half2 sh_a[k_chunk][16 * row_tiles][8];
    using v16f16 = _Float16 __attribute__((ext_vector_type(16)));
    using v8f32 = float __attribute__((ext_vector_type(8)));
    {   // blockIdx.y is the 256-column band (grid dim `bands`); each band grid-strides the row tiles.
        const int subtile = (int) blockIdx.y * warps + wave;
        const bool active = subtile * 16 < out_features;
        (void) n_groups;
        for (int base_m = blockIdx.x * 16 * row_tiles; base_m < rows; base_m += gridDim.x * 16 * row_tiles) {
            const int match_count = min(16 * row_tiles, rows - base_m);
            v8f32 acc[row_tiles];
            #pragma unroll
            for (int rt = 0; rt < row_tiles; ++rt) acc[rt] = {};
            for (int kc = 0; kc < rows_total; kc += k_chunk) {
                const int kct = min(k_chunk, rows_total - kc);
                for (int i = threadIdx.x; i < kct * 16 * row_tiles * 8; i += 512) {
                    const int p = i & 7, m = (i >> 3) % (16 * row_tiles), k = i / (16 * row_tiles * 8);
                    const int m_idx = m < match_count ? base_m + m : -1;
                    const __half2 zero2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));
                    sh_a[k][m][p] = m_idx < 0 ? zero2 :
                        *reinterpret_cast<const __half2*>(input16 + (size_t) m_idx * in_features + (kc + k) * 16 + p * 2);
                }
                __syncthreads();
                int k = 0;
                for (; k + 1 < kct; k += 2) {   // K=32 pairing: two k-tiles per step
                    uint32_t d00[2] = {0, 0}, d01[2] = {0, 0}, d10[2] = {0, 0}, d11[2] = {0, 0};
                    if (active) {
                        #pragma unroll
                        for (int step = 0; step < 2; ++step) {
                            const uint32_t* packed = reinterpret_cast<const uint32_t*>(
                                trellis + ((size_t)(kc + k + step) * (out_features / 16) + (size_t) subtile) * (16 * bits));
                            Exl3Frag f0, f1;
                            exl3_decode_mul1<bits>(packed, lane * 8, f0, f1);
                            d00[step] = __builtin_bit_cast(uint32_t, f0.v[0]);
                            d01[step] = __builtin_bit_cast(uint32_t, f0.v[1]);
                            d10[step] = __builtin_bit_cast(uint32_t, f1.v[0]);
                            d11[step] = __builtin_bit_cast(uint32_t, f1.v[1]);
                        }
                    }
                    // The packed tile's tensor-core interleave, redistributed to the wmma B layout.
                    // The mapping is the fork's empirically verified one (llama.cpp exl3_wmma_routed_rows),
                    // and this card agrees: a probe that solves for the matrix the mma actually applies
                    // (128 basis rows against a CPU-decoded What) recovers What's rows through exactly
                    // this shuffle - slot 2g/2g+1 of lane lo carries What[2g+9/2g+8][lo] and slots
                    // 8+2g/9+2g carry What[2g+1/2g][lo] (lo<8; +8 columns take s=4..7), which is the
                    // fragment order v_wmma_f32_16x16x16_f16 pairs A's sequential k slots against.
                    #pragma unroll
                    for (int step = 0; step < 2; ++step) {
                        v16f16 b;
                        uint32_t* bp = reinterpret_cast<uint32_t*>(&b);
                        #pragma unroll
                        for (int group = 0; group < 4; ++group) {
                            const int source_lane = 4 * (lo & 7) + group;
                            const uint32_t v00 = __shfl_sync(0xffffffffu, d00[step], source_lane);
                            const uint32_t v01 = __shfl_sync(0xffffffffu, d01[step], source_lane);
                            const uint32_t v10 = __shfl_sync(0xffffffffu, d10[step], source_lane);
                            const uint32_t v11 = __shfl_sync(0xffffffffu, d11[step], source_lane);
                            bp[group] = lo < 8 ? v00 : v10;
                            bp[4 + group] = lo < 8 ? v01 : v11;
                        }
                        #pragma unroll
                        for (int rt = 0; rt < row_tiles; ++rt) {
                            v16f16 a;
                            uint32_t* ap = reinterpret_cast<uint32_t*>(&a);
                            const int m = rt * 16 + lo;
                            #pragma unroll
                            for (int pair = 0; pair < 8; ++pair)
                                ap[pair] = __builtin_bit_cast(uint32_t, sh_a[k + step][m][pair]);
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
                            acc[rt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[rt]);
#else
                            __builtin_trap();   // the runtime gate never launches this on another arch
#endif
                        }
                    }
                }
                for (; k < kct; ++k) {   // odd-tail guard; unreachable for ki % 8 == 0 shapes
                    uint32_t d00 = 0, d01 = 0, d10 = 0, d11 = 0;
                    if (active) {
                        const uint32_t* packed = reinterpret_cast<const uint32_t*>(
                            trellis + ((size_t)(kc + k) * (out_features / 16) + (size_t) subtile) * (16 * bits));
                        Exl3Frag f0, f1;
                        exl3_decode_mul1<bits>(packed, lane * 8, f0, f1);
                        d00 = __builtin_bit_cast(uint32_t, f0.v[0]);
                        d01 = __builtin_bit_cast(uint32_t, f0.v[1]);
                        d10 = __builtin_bit_cast(uint32_t, f1.v[0]);
                        d11 = __builtin_bit_cast(uint32_t, f1.v[1]);
                    }
                    v16f16 b;
                    uint32_t* bp = reinterpret_cast<uint32_t*>(&b);
                    #pragma unroll
                    for (int group = 0; group < 4; ++group) {
                        const int source_lane = 4 * (lo & 7) + group;
                        const uint32_t v00 = __shfl_sync(0xffffffffu, d00, source_lane);
                        const uint32_t v01 = __shfl_sync(0xffffffffu, d01, source_lane);
                        const uint32_t v10 = __shfl_sync(0xffffffffu, d10, source_lane);
                        const uint32_t v11 = __shfl_sync(0xffffffffu, d11, source_lane);
                        bp[group] = lo < 8 ? v00 : v10;
                        bp[4 + group] = lo < 8 ? v01 : v11;
                    }
                    #pragma unroll
                    for (int rt = 0; rt < row_tiles; ++rt) {
                        v16f16 a;
                        uint32_t* ap = reinterpret_cast<uint32_t*>(&a);
                        const int m = rt * 16 + lo;
                        #pragma unroll
                        for (int pair = 0; pair < 8; ++pair) ap[pair] = __builtin_bit_cast(uint32_t, sh_a[k][m][pair]);
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
                        acc[rt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[rt]);
#else
                        __builtin_trap();   // the runtime gate never launches this on another arch
#endif
                    }
                }
                __syncthreads();
            }
            #pragma unroll
            for (int rt = 0; rt < row_tiles; ++rt) {
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int rel = rt * 16 + 2 * e + hi;
                    if (active && rel < match_count)
                        output[(size_t)(base_m + rel) * out_features + subtile * 16 + lo] = acc[rt][e];
                }
            }
        }
    }
}

// Gather the group's input rows, fold this projection's suh and the input-side Hadamard, emit f16
// (the fork's exl3_routed_input_h128, shuffle form).  `input_rows` indexes the TOKEN rows of `x`
// (gate/up); the down projection's input is already route-major (the silu's slab), so it passes
// null and each route reads its own row.
template <bool batched = false>
__global__ void gather_h128_f16_kernel(const uint16_t* __restrict__ x, const int* __restrict__ input_rows,
                                       const uint16_t* __restrict__ suh, uint16_t* __restrict__ out, int in_features,
                                       const Exl3Group* groups = nullptr, int projection = 0) {
    int route = blockIdx.y;
    if constexpr (batched) {
        const Exl3Group group = groups[blockIdx.z];
        if (route >= group.rows) return;
        route += group.first;
        suh = (projection == 0 ? group.gate : projection == 1 ? group.up : group.down).suh;
    }
    const int lane = threadIdx.x;
    const uint16_t* src = x + (size_t) (input_rows ? input_rows[route] : route) * in_features;
    uint16_t* dst = out + (size_t) route * in_features;
    const int base = blockIdx.x * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = dh2f(src[base + i]) * dh2f(suh[base + i]);
    float h0 = v[0] + v[1], h1 = v[0] - v[1], h2 = v[2] + v[3], h3 = v[2] - v[3];
    float s0 = h0 + h2, d0 = h1 + h3, s1 = h0 - h2, d1 = h1 - h3;
    h0 = s0; h1 = d0; h2 = s1; h3 = d1;
    #pragma unroll
    for (int mask = 1; mask < 32; mask <<= 1) {
        const float p0 = __shfl_xor_sync(0xffffffffu, h0, mask);
        const float p1 = __shfl_xor_sync(0xffffffffu, h1, mask);
        const float p2 = __shfl_xor_sync(0xffffffffu, h2, mask);
        const float p3 = __shfl_xor_sync(0xffffffffu, h3, mask);
        h0 = (lane & mask ? -h0 : h0) + p0;
        h1 = (lane & mask ? -h1 : h1) + p1;
        h2 = (lane & mask ? -h2 : h2) + p2;
        h3 = (lane & mask ? -h3 : h3) + p3;
    }
    dst[base + 0] = __half_as_ushort(__float2half_rn(h0 * 0.08838834764831845f));
    dst[base + 1] = __half_as_ushort(__float2half_rn(h1 * 0.08838834764831845f));
    dst[base + 2] = __half_as_ushort(__float2half_rn(h2 * 0.08838834764831845f));
    dst[base + 3] = __half_as_ushort(__float2half_rn(h3 * 0.08838834764831845f));
}

// The output-side Hadamard + svh for the group's rows: the f16 form the silu consumes, and the f32
// form the down projection writes (the same math as postscale_batched_kernel, without the per-expert
// mat descriptor: this path is one expert).
template <bool batched = false>
__global__ void postscale_rows_f16_kernel(const float* __restrict__ z, const uint16_t* __restrict__ svh,
                                          uint16_t* __restrict__ y, int n,
                                          const Exl3Group* groups = nullptr, int projection = 0) {
    __shared__ float b[HAD];
    int r = blockIdx.y;
    if constexpr (batched) {
        const Exl3Group group = groups[blockIdx.z];
        if (r >= group.rows) return;
        r += group.first;
        svh = (projection == 0 ? group.gate : group.up).svh;
    }
    const int t = threadIdx.x, idx = blockIdx.x * HAD + t;
    b[t] = z[(size_t) r * n + idx];
    __syncthreads();
    for (int step = 1; step < HAD; step <<= 1) {
        if ((t & step) == 0) { float a = b[t], d = b[t | step]; b[t] = a + d; b[t | step] = a - d; }
        __syncthreads();
    }
    y[(size_t) r * n + idx] = __half_as_ushort(__float2half_rn(b[t] * 0.08838834764831845f * dh2f(svh[idx])));
}

template <bool batched = false>
__global__ void postscale_rows_f32_kernel(const float* __restrict__ z, const uint16_t* __restrict__ svh,
                                          float* __restrict__ y, int n,
                                          const Exl3Group* groups = nullptr, const int* destinations = nullptr) {
    __shared__ float b[HAD];
    int r = blockIdx.y;
    if constexpr (batched) {
        const Exl3Group group = groups[blockIdx.z];
        if (r >= group.rows) return;
        r += group.first;
        svh = group.down.svh;
    }
    const int t = threadIdx.x, idx = blockIdx.x * HAD + t;
    b[t] = z[(size_t) r * n + idx];
    __syncthreads();
    for (int step = 1; step < HAD; step <<= 1) {
        if ((t & step) == 0) { float a = b[t], d = b[t | step]; b[t] = a + d; b[t | step] = a - d; }
        __syncthreads();
    }
    const int dst = batched ? destinations[r] : r;
    y[(size_t) dst * n + idx] = b[t] * 0.08838834764831845f * dh2f(svh[idx]);
}

// The answer is per device (a mixed-gfx machine must not run gfx11 fragments elsewhere) and must not
// freeze a failed query: cache successful queries keyed by the current device id, retry failures.
bool exl3_wmma_device() {
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    static std::mutex mu;
    static std::unordered_map<int, bool> cache;
    {
        std::lock_guard<std::mutex> lock(mu);
        const auto hit = cache.find(dev);
        if (hit != cache.end()) return hit->second;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
    const bool ok = strata::kernels::gfx_arch_is_gfx11_wmma(prop.gcnArchName);
    std::lock_guard<std::mutex> lock(mu);
    cache.emplace(dev, ok);
    return ok;
}

}  // namespace

bool exl3_moe_group_wmma(const Exl3Mat& gate, const Exl3Mat& up, const Exl3Mat& down, int rows,
                         const uint16_t* x, const int* input_rows, float* out, void* stream,
                         Exl3MoeWorkspace& workspace) {
    if (!exl3_wmma_device()) return false;
    for (const Exl3Mat* matrix : {&gate, &up, &down})
        if (!matrix->trellis || !matrix->suh || !matrix->svh || matrix->bits < 3 || matrix->bits > 6 ||
            matrix->cb != 2 || matrix->ki <= 0 || matrix->ki > INT32_MAX / 16 ||
            matrix->nj <= 0 || matrix->nj > INT32_MAX / 16 || matrix->ki % 8 || matrix->nj % 8)
            return false;   // the mcg/3inst codebooks, K outside 3..6 and non-128-multiple dims stay on the reconstruct path
    if (rows <= 0 || rows > 65535 || !x || !input_rows || !out || gate.ki != up.ki || gate.nj != up.nj ||
        down.ki != gate.nj || gate.ki <= 0 || gate.nj <= 0 || down.nj <= 0)
        return false;
    const int k = gate.ki * 16, ff = gate.nj * 16, n = down.nj * 16;
    const auto st = (cudaStream_t) stream;
    workspace.reserve(rows, k, ff, n);
    auto& w = workspace.implementation();
    w.wmma_x.reserve((size_t) rows * std::max(k, ff));
    w.group_y.reserve((size_t) rows * std::max(ff, n));
    uint16_t* slab = w.wmma_x.p;
    float* y = w.group_y.p;
    const int row_tiles = rows >= 24 ? 2 : 1;
    const int m_blocks = std::min((rows + 16 * row_tiles - 1) / (16 * row_tiles), 32);
    auto project = [&](const Exl3Mat& m, const uint16_t* input, const int* rows_of, uint16_t* f16_out, float* f32_out) {
        const int in = m.ki * 16, outputs = m.nj * 16;
        gather_h128_f16_kernel<<<dim3(in / 128, rows), 32, 0, st>>>(input, rows_of, m.suh, slab, in);
        const int bands = (outputs + 255) / 256;
        const int bits = m.bits;
        if (bits == 3) {
            if (row_tiles == 1) routed_wmma_kernel<3, 1><<<dim3(m_blocks, bands), 512, 0, st>>>(slab, m.trellis, y, rows, in, outputs);
            else routed_wmma_kernel<3, 2><<<dim3(m_blocks, bands), 512, 0, st>>>(slab, m.trellis, y, rows, in, outputs);
        } else if (bits == 4) {
            if (row_tiles == 1) routed_wmma_kernel<4, 1><<<dim3(m_blocks, bands), 512, 0, st>>>(slab, m.trellis, y, rows, in, outputs);
            else routed_wmma_kernel<4, 2><<<dim3(m_blocks, bands), 512, 0, st>>>(slab, m.trellis, y, rows, in, outputs);
        } else if (bits == 5) {
            if (row_tiles == 1) routed_wmma_kernel<5, 1><<<dim3(m_blocks, bands), 512, 0, st>>>(slab, m.trellis, y, rows, in, outputs);
            else routed_wmma_kernel<5, 2><<<dim3(m_blocks, bands), 512, 0, st>>>(slab, m.trellis, y, rows, in, outputs);
        } else {
            if (row_tiles == 1) routed_wmma_kernel<6, 1><<<dim3(m_blocks, bands), 512, 0, st>>>(slab, m.trellis, y, rows, in, outputs);
            else routed_wmma_kernel<6, 2><<<dim3(m_blocks, bands), 512, 0, st>>>(slab, m.trellis, y, rows, in, outputs);
        }
        if (f16_out)
            postscale_rows_f16_kernel<<<dim3(outputs / HAD, rows), HAD, 0, st>>>(y, m.svh, f16_out, outputs);
        else
            postscale_rows_f32_kernel<<<dim3(outputs / HAD, rows), HAD, 0, st>>>(y, m.svh, f32_out, outputs);
    };
    project(gate, x, input_rows, w.g.p, nullptr);
    project(up, x, input_rows, w.u.p, nullptr);
    const long hidden = (long) rows * ff;
    silu_batched_kernel<<<(unsigned) ((hidden + 255) / 256), 256, 0, st>>>(w.g.p, w.u.p, w.h.p, hidden);
    project(down, w.h.p, nullptr, nullptr, out);   // the silu slab is route-major: no row table
    return true;
}

bool exl3_moe_expert_batch(const Exl3Group* groups, int count, const uint16_t* x,
                           const int* input_rows, const int* destinations, float* out, void* stream,
                           Exl3MoeWorkspace& workspace) {
    if (!groups || count < 2 || count > 8 || !x || !input_rows || !destinations || !out ||
        !exl3_wmma_device()) return false;
    int rows = 0, max_rows = 0;
    for (int i = 0; i < count; ++i) {
        const auto& g = groups[i];
        if (g.first != rows || g.rows < 16 || g.rows > 4096 - rows ||
            (g.rows >= 24) != (groups[0].rows >= 24)) return false;
        for (int p = 0; p < 3; ++p) {
            const Exl3Mat& m = p == 0 ? g.gate : p == 1 ? g.up : g.down;
            const Exl3Mat& ref = p == 0 ? groups[0].gate : p == 1 ? groups[0].up : groups[0].down;
            if (!m.trellis || !m.suh || !m.svh || m.cb != 2 || m.bits < 3 || m.bits > 6 ||
                m.bits != ref.bits || m.ki != ref.ki || m.nj != ref.nj ||
                m.ki <= 0 || m.ki > INT32_MAX / 16 || m.nj <= 0 || m.nj > INT32_MAX / 16 ||
                m.ki % 8 || m.nj % 8) return false;
        }
        if (g.up.ki != g.gate.ki || g.up.nj != g.gate.nj || g.down.ki != g.gate.nj) return false;
        rows += g.rows;
        max_rows = std::max(max_rows, g.rows);
    }
    auto& w = workspace.implementation();
    const int k = groups[0].gate.ki * 16, ff = groups[0].gate.nj * 16, n = groups[0].down.nj * 16;
    if (w.expert_groups.capacity < (size_t) count || !workspace.group_tile_fits(rows, k, ff, n, true))
        throw std::runtime_error("exl3: expert batch workspace would grow under queued work");
    const auto st = (cudaStream_t) stream;
    if (cudaMemcpyAsync(w.expert_groups.p, groups, count * sizeof(Exl3Group), cudaMemcpyHostToDevice, st) != cudaSuccess)
        throw std::runtime_error("exl3: expert batch descriptor upload failed");
    const int rt = groups[0].rows >= 24 ? 2 : 1;
    const int mb = std::min((max_rows + 16 * rt - 1) / (16 * rt), 32);
    auto project = [&](int p, const uint16_t* input, const int* ids, uint16_t* h, float* f) {
        const Exl3Mat& m = p == 0 ? groups[0].gate : p == 1 ? groups[0].up : groups[0].down;
        const int in = m.ki * 16, outputs = m.nj * 16;
        gather_h128_f16_kernel<true><<<dim3(in / 128, max_rows, count), 32, 0, st>>>(
            input, ids, nullptr, w.wmma_x.p, in, w.expert_groups.p, p);
        const dim3 grid(mb, (outputs + 255) / 256, count);
        auto launch = [&]<int bits>() {
            if (rt == 1)
                routed_wmma_kernel<bits, 1, true><<<grid, 512, 0, st>>>(
                    w.wmma_x.p, nullptr, w.group_y.p, 0, in, outputs, w.expert_groups.p, p);
            else
                routed_wmma_kernel<bits, 2, true><<<grid, 512, 0, st>>>(
                    w.wmma_x.p, nullptr, w.group_y.p, 0, in, outputs, w.expert_groups.p, p);
        };
        switch (m.bits) {
            case 3: launch.template operator()<3>(); break;
            case 4: launch.template operator()<4>(); break;
            case 5: launch.template operator()<5>(); break;
            case 6: launch.template operator()<6>(); break;
        }
        if (h) postscale_rows_f16_kernel<true><<<dim3(outputs / HAD, max_rows, count), HAD, 0, st>>>(
            w.group_y.p, nullptr, h, outputs, w.expert_groups.p, p);
        else postscale_rows_f32_kernel<true><<<dim3(outputs / HAD, max_rows, count), HAD, 0, st>>>(
            w.group_y.p, nullptr, f, outputs, w.expert_groups.p, destinations);
    };
    project(0, x, input_rows, w.g.p, nullptr);
    project(1, x, input_rows, w.u.p, nullptr);
    const long hidden = (long) rows * ff;
    silu_batched_kernel<<<(unsigned) ((hidden + 255) / 256), 256, 0, st>>>(w.g.p, w.u.p, w.h.p, hidden);
    project(2, w.h.p, nullptr, nullptr, out);
    return true;
}

#else   // !STRATA_WMMA_GFX11

bool exl3_moe_expert_batch(const Exl3Group*, int, const uint16_t*, const int*, const int*, float*,
                           void*, Exl3MoeWorkspace&) { return false; }

bool exl3_moe_group_wmma(const Exl3Mat&, const Exl3Mat&, const Exl3Mat&, int,
                         const uint16_t*, const int*, float*, void*, Exl3MoeWorkspace&) {
    return false;
}

#endif

// STRATA_EXL3_PEER_PREFILL's return transport - see the exl3.hpp declaration.  The arithmetic is
// peer_scatter_add's (src/prefill/kernels.cu, the eddoursul/Strata sums transport): one fmaf per
// element into the token's sum row; the peer launches it once per expert group in expert order,
// so the accumulation order is fixed and the sums repeatable.
namespace {
__global__ void pf_scatter_add_kernel(float* __restrict__ sum, const float* __restrict__ rows,
                                      const float* __restrict__ weights, const int32_t* __restrict__ pair,
                                      int64_t n, int k, int64_t n_embd) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * n_embd) return;
    const int64_t r = i / n_embd, d = i % n_embd;
    const int32_t p = pair[r];
    float* o = sum + (int64_t) (p / k) * n_embd + d;
    *o = fmaf(weights[p], rows[i], *o);
}
}

void exl3_pf_scatter_add(float* sum, const float* rows, const float* weights, const int32_t* pair,
                         int64_t n, int k, int64_t n_embd, void* stream) {
    if (n <= 0) return;
    if (!sum || !rows || !weights || !pair || !stream || k <= 0 || n_embd <= 0 ||
        n > INT32_MAX / 4 || n_embd > INT32_MAX / 4)
        throw std::invalid_argument("exl3: peer prefill scatter-add needs rows, weights, pairs, k, width");
    pf_scatter_add_kernel<<<(unsigned) ((n * n_embd + 255) / 256), 256, 0, (cudaStream_t) stream>>>(
        sum, rows, weights, pair, n, k, n_embd);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("exl3: peer prefill scatter-add: ") +
                                                   cudaGetErrorString(e));
}



void exl3_moe_rows(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down, const int* ids,
                   int k, const uint16_t* x, uint16_t* out, void* stream, Exl3MoeWorkspace* workspace) {
    (void) ids;   // the caller orders gate/up/down to match ids, so row i is expert ids[i]
    exl3_moe_rows_part(gate, up, down, k, x, out, stream, workspace);
}

// Scatter two f16 row blocks into the token's k rows (the order `moe_combine` expects): `dst[row]` takes
// block `b`'s next row, `n` values each.  One block per row: k is 10 at the real geometry.
__global__ void rows_scatter_kernel(const uint16_t* __restrict__ a, const uint16_t* __restrict__ b,
                                    const int* __restrict__ a_row, int na, const int* __restrict__ b_row, int nb,
                                    int n, uint16_t* __restrict__ out) {
    const int i = blockIdx.x;
    const bool second = i >= na;
    const int r = second ? i - na : i;
    const int row = second ? b_row[r] : a_row[r];
    const uint16_t* src = (second ? b : a) + (size_t) r * n;
    uint16_t* dst = out + (size_t) row * n;
    for (int j = threadIdx.x; j < n; j += blockDim.x) dst[j] = src[j];
}

void exl3_rows_scatter(const uint16_t* a, const int* a_row, int na, const uint16_t* b, const int* b_row, int nb,
                       int n, uint16_t* out, void* stream) {
    if (na + nb <= 0) return;
    rows_scatter_kernel<<<na + nb, 256, 0, (cudaStream_t) stream>>>(a, b, a_row, na, b_row, nb, n, out);
}

__global__ void embed_gather_kernel(const uint16_t* __restrict__ row, long n, int bf16, float* __restrict__ out) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint16_t h = row[i];
    if (bf16) { uint32_t f = (uint32_t) h << 16; float o; __builtin_memcpy(&o, &f, 4); out[i] = o; }
    else out[i] = __half2float(*(const __half*) &h);
}

__global__ void embed_gather_rows_kernel(const uint16_t* table, int64_t n, int64_t vocab, bool bf16,
                                        const int32_t* tokens, int rows, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) rows * n) return;
    const int32_t token = tokens[i / n];
    float value = 0;
    if (token >= 0 && token < vocab) {
        const uint16_t h = table[(int64_t) token * n + i % n];
        value = bf16 ? __uint_as_float((uint32_t) h << 16) : dh2f(h);
    }
    out[i] = value;
}

void exl3_embed_gather_rows(const uint16_t* table, int64_t n_embd, int64_t n_vocab, bool bf16,
                            const int32_t* tokens, int rows, float* out, void* stream) {
    if (!table || !tokens || !out || n_embd <= 0 || n_vocab <= 0 || rows <= 0)
        throw std::invalid_argument("exl3: invalid embedding gather");
    embed_gather_rows_kernel<<<(unsigned) (((int64_t) rows * n_embd + 255) / 256), 256, 0,
                               (cudaStream_t) stream>>>(table, n_embd, n_vocab, bf16, tokens, rows, out);
}


__global__ void silu_mul_f32_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ h, long n) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = g[i];
    h[i] = (gv / (1.0f + expf(-gv))) * u[i];
}

__global__ void shared_scalar_gate_kernel(float* __restrict__ out, const float* __restrict__ x,
                                          const uint16_t* __restrict__ w_ginp, int n_embd) {
    // one block: sigmoid(w_ginp . x) over n_embd, applied to every out element
    __shared__ float red[256];
    const int t = threadIdx.x;
    float s = 0;
    for (int i = t; i < n_embd; i += blockDim.x) {
        const uint16_t h = w_ginp[i];
        uint32_t f = (uint32_t) h << 16; float w; __builtin_memcpy(&w, &f, 4);
        s += w * x[i];
    }
    red[t] = s; __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) { if (t < step) red[t] += red[t + step]; __syncthreads(); }
    if (t == 0) red[0] = 1.0f / (1.0f + expf(-red[0]));
    __syncthreads();
    const float gate = red[0];
    for (int i = t; i < n_embd; i += blockDim.x) out[i] *= gate;
}

static float* g_sh_g = nullptr; static float* g_sh_u = nullptr; static float* g_sh_h = nullptr;
static int g_sh_ff = 0;

void exl3_shared_reserve(int n_ff) {
    if (n_ff <= g_sh_ff) return;
    (void) cudaFree(g_sh_g); (void) cudaFree(g_sh_u); (void) cudaFree(g_sh_h);
    g_sh_g = g_sh_u = g_sh_h = nullptr;
    if (cudaMalloc((void**) &g_sh_g, (size_t) n_ff * 4) || cudaMalloc((void**) &g_sh_u, (size_t) n_ff * 4) ||
        cudaMalloc((void**) &g_sh_h, (size_t) n_ff * 4)) std::abort();
    g_sh_ff = n_ff;
}

void exl3_shared_expert(const float* x, const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down,
                        const uint16_t* w_ginp, float* out, int n_embd, int n_ff, void* stream) {
    cudaStream_t st = (cudaStream_t) stream;
    exl3_shared_reserve(n_ff);
    float* const d_g = g_sh_g; float* const d_u = g_sh_u; float* const d_h = g_sh_h;
    exl3_gemv_f32(x, gate->suh, gate->svh, gate->trellis, gate->ki, gate->nj, gate->bits, gate->cb, d_g, stream);
    exl3_gemv_f32(x, up->suh, up->svh, up->trellis, up->ki, up->nj, up->bits, up->cb, d_u, stream);
    silu_mul_f32_kernel<<<(unsigned)((n_ff + 255) / 256), 256, 0, st>>>(d_g, d_u, d_h, n_ff);
    exl3_gemv_f32(d_h, down->suh, down->svh, down->trellis, down->ki, down->nj, down->bits, down->cb, out, stream);
    shared_scalar_gate_kernel<<<1, 256, 0, st>>>(out, x, w_ginp, n_embd);
}

void exl3_embed_gather(const uint16_t* row, int64_t n, bool bf16, float* out, void* stream) {
    cudaStream_t st = (cudaStream_t) stream;
    embed_gather_kernel<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(row, n, bf16 ? 1 : 0, out);
}

// A9's window shared expert: the decode path's fp32 swiglu and the native multi path's sigmoid gate
// scale, exposed so the trellis arm produces the same values the native arm would.
__global__ void sigmoid_scale_rows_kernel(float* __restrict__ out, const float* __restrict__ g, int n) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float gt = __fdividef(1.0f, 1.0f + __expf(-g[t]));
        out[(size_t) t * n + i] *= gt;
    }
}

void exl3_silu_mul_f32(const float* g, const float* u, float* h, long n, void* stream) {
    cudaStream_t st = (cudaStream_t) stream;
    silu_mul_f32_kernel<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(g, u, h, n);
}

void exl3_sigmoid_scale_rows(float* out, const float* g, int n, int rows, void* stream) {
    cudaStream_t st = (cudaStream_t) stream;
    sigmoid_scale_rows_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) rows), 256, 0, st>>>(out, g, n);
}


void exl3_gemv_f32(const float* x, const uint16_t* suh, const uint16_t* svh, const uint16_t* trellis,
                   int ki, int nj, int bits, int cb, float* y, void* stream) {
#if defined(STRATA_EXL3_RDNA_DECODE)
    // STRATA_EXL3_DENSE_RDNA=1: the dense m == 1 GEMV takes the multirow kernel's row tile of 1
    // (2 launches, the RDNA decode core, fp32 end to end) instead of the f32->f16->f32 old pipeline
    if (exl3_dense_rdna_enabled()) {
        exl3_dense_gemv_multi(0, x, (long) ki * 16, trellis, suh, svh, ki, nj, bits, cb, 1, y,
                              (long) nj * 16, stream);
        return;
    }
#endif
    cudaStream_t st = (cudaStream_t) stream;
    const int k = ki * 16, n = nj * 16;
    exl3_gemv_reserve(k, n);
    f32_to_f16_out_kernel<<<(k + 255) / 256, 256, 0, st>>>(x, g_ws.x16, k);
    exl3_gemv(g_ws.x16, suh, svh, trellis, ki, nj, bits, cb, g_ws.y16, stream);
    f16_to_f32_kernel<<<(n + 255) / 256, 256, 0, st>>>(g_ws.y16, y, n);
}

// ---- the dense multirow GEMV's host side: workspace, split heuristic, row-tile dispatch ----
namespace {
// One workspace per stream that runs the kernel concurrently: slot 0 is the main/decode stream (the
// window's projections and the head), slot 1 the verifier's forked shared-expert stream.
struct DenseMrWs { float* xh = nullptr; float* z = nullptr; int* cnt = nullptr; long cxh = 0, cz = 0, ccnt = 0; };
DenseMrWs g_dense_mr[2];
}  // namespace

bool exl3_dense_mr_available() {
#if defined(STRATA_EXL3_RDNA_DECODE)
    return true;
#else
    return false;
#endif
}

bool exl3_dense_rdna_enabled() {
    static const bool on = [] {
#if defined(STRATA_EXL3_RDNA_DECODE)
        const char* e = std::getenv("STRATA_EXL3_DENSE_RDNA");
        return e != nullptr && std::atoi(e) > 0;
#else
        return false;
#endif
    }();
    return on;
}

void exl3_dense_mr_reserve_shape(int slot, int ki, int nj) {
#if defined(STRATA_EXL3_RDNA_DECODE)
    DenseMrWs& w = g_dense_mr[slot & 1];
    const int k = ki * 16, n = nj * 16;
    int parts = 1;
    while (parts < 16 && ((nj + GEMV_T - 1) / GEMV_T) * parts < 256 && ki >= (parts + 1) * 16) ++parts;
    if (8L * k > w.cxh) {
        (void) cudaFree(w.xh); w.xh = nullptr;
        if (cudaMalloc(&w.xh, (size_t) 8 * k * 4) != cudaSuccess) std::abort();
        w.cxh = 8L * k;
    }
    const long z_need = (long) parts * 8 * n;   // [partition][M row][column], the worst M = 8
    if (z_need > w.cz) {
        (void) cudaFree(w.z); w.z = nullptr;
        if (cudaMalloc(&w.z, (size_t) z_need * 4) != cudaSuccess) std::abort();
        w.cz = z_need;
    }
    if (n / HAD > w.ccnt) {
        (void) cudaFree(w.cnt); w.cnt = nullptr;
        if (cudaMalloc(&w.cnt, (size_t) (n / HAD) * 4) != cudaSuccess) std::abort();
        if (cudaMemset(w.cnt, 0, (size_t) (n / HAD) * 4) != cudaSuccess) std::abort();
        w.ccnt = n / HAD;
    }
#else
    (void) slot; (void) ki; (void) nj;
#endif
}

void exl3_dense_gemv_multi(int slot, const float* x, long x_stride, const uint16_t* trellis,
                           const uint16_t* suh, const uint16_t* svh, int ki, int nj, int bits, int cb,
                           int rows, float* y, long y_stride, void* stream) {
#if defined(STRATA_EXL3_RDNA_DECODE)
    const int k = ki * 16, n = nj * 16;
    if (ki <= 0 || nj <= 0 || rows < 1 || k % HAD || n % HAD) {
        std::fprintf(stderr, "exl3_dense_gemv_multi: dims must be multiples of %d (ki=%d nj=%d rows=%d)\n",
                     HAD, ki, nj, rows);
        std::abort();
    }
    exl3_gemv_reserve(k, n);   // the pinv table (and the old path's scratch), allocated outside capture
    // gemv_launch's split rule: enough blocks to fill the machine, never a row-per-sub overkill
    int parts = 1;
    while (parts < 16 && ((nj + GEMV_T - 1) / GEMV_T) * parts < 256 && ki >= (parts + 1) * 16) ++parts;
    DenseMrWs& w = g_dense_mr[slot & 1];
    if (8L * k > w.cxh || (long) parts * 8 * n > w.cz || n / HAD > w.ccnt)
        exl3_dense_mr_reserve_shape(slot, ki, nj);   // an unreserved shape: grow now (never inside a capture)
    cudaStream_t st = (cudaStream_t) stream;
    const dim3 grid((nj + GEMV_T - 1) / GEMV_T, parts);
    for (int r0 = 0; r0 < rows; ) {
        const int tile = rows - r0 <= 1 ? 1 : (rows - r0 <= 2 ? 2 : (rows - r0 <= 4 ? 4 : 8));
        const int rv = std::min(tile, rows - r0);
        dense_had_in_kernel<<<dim3(k / HAD, rv), HAD, 0, st>>>(x + (long) r0 * x_stride, x_stride, suh, w.xh, k);
        float* yv = y + (long) r0 * y_stride;
        switch (tile) {
            case 1: gemv_dense_mr_kernel<GEMV_T, 1><<<grid, 256, 0, st>>>(w.xh, trellis, svh, g_ws.pinv, ki, nj, bits, cb, w.z, yv, y_stride, rv, w.cnt); break;
            case 2: gemv_dense_mr_kernel<GEMV_T, 2><<<grid, 256, 0, st>>>(w.xh, trellis, svh, g_ws.pinv, ki, nj, bits, cb, w.z, yv, y_stride, rv, w.cnt); break;
            case 4: gemv_dense_mr_kernel<GEMV_T, 4><<<grid, 256, 0, st>>>(w.xh, trellis, svh, g_ws.pinv, ki, nj, bits, cb, w.z, yv, y_stride, rv, w.cnt); break;
            default: gemv_dense_mr_kernel<GEMV_T, 8><<<grid, 256, 0, st>>>(w.xh, trellis, svh, g_ws.pinv, ki, nj, bits, cb, w.z, yv, y_stride, rv, w.cnt); break;
        }
        r0 += rv;
    }
#else
    (void) slot; (void) x; (void) x_stride; (void) trellis; (void) suh; (void) svh; (void) ki; (void) nj;
    (void) bits; (void) cb; (void) rows; (void) y; (void) y_stride; (void) stream;
    std::fprintf(stderr, "exl3_dense_gemv_multi: this build has no gfx11 kernels\n");
    std::abort();
#endif
}

}  // namespace strata::kernels
