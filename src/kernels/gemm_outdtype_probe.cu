// src/kernels/gemm_outdtype_probe.cu - bounded rocBLAS GEMM output-dtype probe (no engine change).
//
// Question (candidate #3 from the exllamav3-amd source survey): does gfx1100 + this rocBLAS
// (core-7.14, cublas shim) run f16-in GEMMs with an f32 C meaningfully slower than with an f16 C,
// both at CUBLAS_COMPUTE_32F?  On gfx1151 + hipblaslt the reference fork measured f32-out
// 5.2-5.6x SLOWER at prefill shapes (fa32aaa, +23% prefill by GEMM-ing into an f16 buffer and
// widening; their PPL was bit-identical because their consumers round to f16 anyway).  gfx1100
// here is UNMEASURED - this probe measures it at Strata's real shapes:
//
//   grouped/recon expert GEMM layout (exl3.cu project(): OP_N/OP_N, W column-major [out,in],
//     ld = out): gate/up N=2048 K=4096 and down N=4096 K=2048, rows 2/4/8/16/32/64
//   dense/lm_head layout (Gemm::f16: OP_T/OP_N, W row-major [N,K] ld = K): lm_head
//     N=152576 K=4096 rows 4/8/16/32/64/512, plus a dense-projection shape N=8192 K=4096
//
// It also compares VALUES: f32-C vs f16-C-widened (different kernel selection can change the
// reduction/accumulation order - byte identity is REPORTED, never assumed; f16 C can also
// overflow to inf where f32 C would not - max |y| is printed so the f16 headroom is explicit).
//
// Usage: gemm_outdtype_probe [device]
// Output: one line per (shape, rows): ms f32-out, ms f16-out, speedup, byte-equal count, max|y|.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

float one_f = 1.0f, zero_f = 0.0f;

#define CK(x) do { if ((x) != cudaSuccess) { std::fprintf(stderr, "CUDA error %s:%d\n", __FILE__, __LINE__); std::exit(1); } } while (0)
#define BK(x) do { if ((x) != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "cublas error %s:%d\n", __FILE__, __LINE__); std::exit(1); } } while (0)

__global__ void fill_f16_kernel(uint16_t* p, int64_t n, uint32_t seed) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // xorshift -> [-1, 1) f16: deterministic, no host round-trip for the 1.25 GiB lm_head weight
    uint32_t s = seed ^ (uint32_t) (i * 2654435761u);
    s ^= s >> 13; s *= 0x5bd1e995u; s ^= s >> 15;
    const float v = ((int) (s & 0xFFFF) - 32768) * (1.0f / 32768.0f);
    p[i] = __half_as_ushort(__float2half(v));
}

__global__ void widen_f16_kernel(const uint16_t* in, float* out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(__ushort_as_half(in[i]));
}

double time_gemm(cublasHandle_t blas, cudaStream_t st, bool bench_f32,
                 int n, int rows, int k, bool op_t, const uint16_t* w, const uint16_t* x,
                 float* y32, uint16_t* y16, int reps) {
    for (int i = 0; i < 3; ++i) {
        if (bench_f32)
            BK(cublasGemmEx(blas, op_t ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, n, rows, k, &one_f,
                            w, CUDA_R_16F, op_t ? k : n, x, CUDA_R_16F, k, &zero_f, y32, CUDA_R_32F, n,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        else
            BK(cublasGemmEx(blas, op_t ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, n, rows, k, &one_f,
                            w, CUDA_R_16F, op_t ? k : n, x, CUDA_R_16F, k, &zero_f, y16, CUDA_R_16F, n,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    CK(cudaStreamSynchronize(st));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a, st));
    for (int i = 0; i < reps; ++i) {
        if (bench_f32)
            BK(cublasGemmEx(blas, op_t ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, n, rows, k, &one_f,
                            w, CUDA_R_16F, op_t ? k : n, x, CUDA_R_16F, k, &zero_f, y32, CUDA_R_32F, n,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        else
            BK(cublasGemmEx(blas, op_t ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, n, rows, k, &one_f,
                            w, CUDA_R_16F, op_t ? k : n, x, CUDA_R_16F, k, &zero_f, y16, CUDA_R_16F, n,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    CK(cudaEventRecord(b, st));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return (double) ms / reps;
}

}  // namespace

int main(int argc, char** argv) {
    const int device = argc > 1 ? std::atoi(argv[1]) : 0;
    CK(cudaSetDevice(device));
    cublasHandle_t blas;
    BK(cublasCreate(&blas));
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    BK(cublasSetStream(blas, st));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, device));
    std::printf("GEMM_OUTDTYPE device=%d %s\n", device, prop.name);

    struct Shape { const char* what; int n, k; bool op_t; };
    const Shape shapes[] = {
        {"expert_gate_up", 2048, 4096, false},
        {"expert_down", 4096, 2048, false},
        {"dense_n8192", 8192, 4096, true},
        {"lm_head", 152576, 4096, true},
    };
    const int rows_set[] = {2, 4, 8, 16, 32, 64, 512};
    for (const Shape& s : shapes) {
        uint16_t *w = nullptr, *x = nullptr, *y16 = nullptr;
        float* y32 = nullptr;
        CK(cudaMalloc(&w, (size_t) s.n * s.k * 2));
        CK(cudaMalloc(&x, (size_t) 512 * s.k * 2));
        CK(cudaMalloc(&y32, (size_t) 512 * s.n * 4));
        CK(cudaMalloc(&y16, (size_t) 512 * s.n * 2));
        fill_f16_kernel<<<(unsigned) (((int64_t) s.n * s.k + 255) / 256), 256, 0, st>>>(w, (int64_t) s.n * s.k, 1234u);
        fill_f16_kernel<<<(unsigned) (((int64_t) 512 * s.k + 255) / 256), 256, 0, st>>>(x, (int64_t) 512 * s.k, 99u);
        CK(cudaStreamSynchronize(st));
        for (int rows : rows_set) {
            if (std::strcmp(s.what, "lm_head") == 0 && rows == 2) continue;   // decode rows measured at 4+
            const int reps = rows <= 64 ? 200 : 50;
            const double ms32 = time_gemm(blas, st, true, s.n, rows, s.k, s.op_t, w, x, y32, y16, reps);
            const double ms16 = time_gemm(blas, st, false, s.n, rows, s.k, s.op_t, w, x, y32, y16, reps);
            // values: one more pass each, then three comparisons.  NO assumption that the two
            // GemmEx calls share a reduction order: (a) float_equal = exact f32 equality of the
            // raw f32-C result vs the widened f16-C result (a BITS comparison would be
            // memcmp-style; this is value ==, labelled as such); (b) round_match = whether the
            // f16-C half equals round_to_nearest_even(f32-C) BIT-FOR-BIT per element - the exact
            // question "is the f16 output just the rounded f32 result"; (c) nonfinite counters on
            // both outputs (f16 can overflow where f32 does not).
            BK(cublasGemmEx(blas, s.op_t ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, s.n, rows, s.k, &one_f,
                            w, CUDA_R_16F, s.op_t ? s.k : s.n, x, CUDA_R_16F, s.k, &zero_f, y32, CUDA_R_32F, s.n,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            CK(cudaStreamSynchronize(st));
            std::vector<float> a((size_t) rows * s.n);
            CK(cudaMemcpy(a.data(), y32, a.size() * 4, cudaMemcpyDeviceToHost));
            BK(cublasGemmEx(blas, s.op_t ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, s.n, rows, s.k, &one_f,
                            w, CUDA_R_16F, s.op_t ? s.k : s.n, x, CUDA_R_16F, s.k, &zero_f, y16, CUDA_R_16F, s.n,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            CK(cudaStreamSynchronize(st));
            std::vector<uint16_t> h((size_t) rows * s.n);
            CK(cudaMemcpy(h.data(), y16, h.size() * 2, cudaMemcpyDeviceToHost));
            widen_f16_kernel<<<(unsigned) (((int64_t) rows * s.n + 255) / 256), 256, 0, st>>>(y16, y32, (int64_t) rows * s.n);
            CK(cudaStreamSynchronize(st));
            std::vector<float> b2((size_t) rows * s.n);
            CK(cudaMemcpy(b2.data(), y32, b2.size() * 4, cudaMemcpyDeviceToHost));
            size_t float_equal = 0, round_match = 0, nonfinite32 = 0, nonfinite16 = 0;
            double max_abs = 0, max_diff = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                if (a[i] == b2[i]) ++float_equal;
                if (!std::isfinite(a[i])) ++nonfinite32;
                if ((h[i] & 0x7C00u) == 0x7C00u) ++nonfinite16;   // inf/nan half exponent, either sign
                if (h[i] == __half_as_ushort(__float2half(a[i]))) ++round_match;
                if (std::fabs(a[i]) > max_abs) max_abs = std::fabs(a[i]);
                const double d = std::fabs((double) a[i] - (double) b2[i]);
                if (d > max_diff) max_diff = d;
            }
            std::printf("GEMM_OUTDTYPE %-14s op=%s N=%-6d K=%-5d rows=%-3d | f32out=%.4f ms f16out=%.4f ms "
                        "speedup=%.2fx | float_equal=%zu/%zu round_match=%zu/%zu nonfinite(f32=%zu,f16=%zu) "
                        "max|y|=%.1f maxdiff=%.3g\n",
                        s.what, s.op_t ? "T" : "N", s.n, s.k, rows, ms32, ms16, ms32 / ms16,
                        float_equal, a.size(), round_match, a.size(), nonfinite32, nonfinite16,
                        max_abs, max_diff);
            std::fflush(stdout);
        }
        CK(cudaFree(w)); CK(cudaFree(x)); CK(cudaFree(y32)); CK(cudaFree(y16));
    }
    BK(cublasDestroy(blas));
    CK(cudaStreamDestroy(st));
    return 0;
}
