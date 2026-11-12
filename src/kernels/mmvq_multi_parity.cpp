// src/kernels/mmvq_multi_parity.cpp - the multi_exact contract of native_mmvq, tested bitwise.
//
//     build/mmvq_multi_parity
//
// WHY IT EXISTS.  `include/strata/kernels/native_mmvq.hpp` says of `native_mmvq_set_multi_exact`:
//
//     "true (default): the ncols == 1 layout, every column bitwise equal to a single-column call"
//
// and `include/strata/core/verify.hpp` counts "multi-column MMVQ in exact mode" among the reasons the verify window
// produces what greedy decode would, bit for bit.  Nothing tested that sentence: `iq_parity` calls `native_mmvq`
// with two columns and compares against a float64 reference at 2e-2, which cannot see a difference between one
// column and T columns.
//
// WHAT IT COMPARES.  The SAME Q8_1 bytes: one T-column call against T single-column calls, each reading column j
// from the same buffer (the header: "its Q8_1 blocks at j * n_in / 32").  The quantization is shared, so the only
// thing under test is how the columns are laid out.
//
// THE DATA.  The weights are random bytes, except each block's fp16 scales (`d`, and `dmin` where the format has
// one), which are rewritten as normal fp16 values in +-[2^-10, 2^-5): random bytes there would put inf and NaN in
// the outputs, and on the GPU every NaN comes out as the same canonical bit pattern, so a NaN output compares equal
// whatever operations produced it and proves nothing.  With finite scales and finite normal activations (the
// precondition the header puts on the input) every output is finite, and a non-finite one fails the test.
//
// THE NEGATIVE CONTROL.  The same comparison with `multi_exact` off must find differences, or the test has no power
// and its pass means nothing.  Where it has power depends on T: in `native_mmvq.cu` the generic layout uses
// `NW = NCOLS <= 4 ? 4 : 2` warps, and with `WARPS = 4` and `BPI = F::BPI * NW / WARPS` it coincides with the
// exact layout at T <= 4.  So the control is ASSERTED on T > 4 for every case, and at T <= 4 its silence is
// reported as that coincidence rather than skipped.  The control needs a row long enough for the two layouts to
// group blocks differently: IQ4_XS takes 16 blocks per iteration with 4 warps and 8 with 2, so at n_in = 2048 (8
// blocks of 256) each thread holds at most one block in both layouts and they coincide; its case uses n_in = 4096.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct Case {
    const char* name;
    int type;          // stable GGML type id, from the list in native_mmvq.hpp
    int n_in;
    int n_out;
    int block_elems;   // elements per weight block
    int block_bytes;
    int scale_at[2];   // byte offsets of the block's fp16 scales in native_mmvq.cu's block structs; -1 = none
};

// the types a head or a projection can have in this model
const Case CASES[] = {
    {"Q4_0", 2, 2048, 512, 32, 18, {0, -1}},          // Q40Block: half d
    {"Q8_0", 8, 2048, 512, 32, 34, {0, -1}},          // Q80Block: half d
    {"Q4_K", 12, 2048, 512, 256, 144, {0, 2}},        // Q4KBlock: half2 dm
    {"Q5_K", 13, 2048, 512, 256, 176, {0, 2}},        // Q5KBlock: half2 dm
    {"Q6_K", 14, 2048, 512, 256, 210, {208, -1}},     // Q6KBlock: half d after ql, qh, scales
    {"IQ4_XS", 23, 4096, 512, 256, 136, {0, -1}},     // IQ4XSBlock: half d; n_in 4096, see THE NEGATIVE CONTROL
    {"Q5_K wide", 13, 4096, 2048, 256, 176, {0, 2}},
};

// a normal fp16 in +-[2^-10, 2^-5): exponent field 5..9 (bias 15), any mantissa, either sign
uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

// T = 4 is the last width where the two layouts coincide, T = 5 the first where they differ
const int WIDTHS[] = {1, 2, 3, 4, 5, 6, 8};
constexpr int COINCIDE_MAX_T = 4;

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

// Returns the number of outputs that differ bit for bit; counts the non-finite outputs and the differences on
// outputs that are finite in both paths.
long long compare(const Case& c, int T, cudaStream_t s, long long& nonfinite, long long& diff_finite, bool& ran) {
    ran = false;
    if (!strata::kernels::native_mmvq_supported(c.type)) return -1;

    const std::size_t wbytes = strata::kernels::native_mmvq_weight_bytes(c.type, c.n_in, c.n_out);
    const std::size_t qcol = strata::kernels::native_q8_1_bytes(c.n_in, 1);   // Q8_1 bytes of ONE column
    const std::size_t qall = strata::kernels::native_q8_1_bytes(c.n_in, T);
    if (qall != qcol * (std::size_t) T) {
        std::printf("%-12s T=%d: the Q8_1 bytes of T columns (%zu) are not T times those of one (%zu): the test "
                    "assumes contiguous columns and they are not\n", c.name, T, qall, qcol);
        return -1;
    }

    std::mt19937 rng(1234u + (unsigned) c.type * 97u + (unsigned) T);
    std::vector<uint8_t> w(wbytes);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    const std::size_t n_blocks = (std::size_t) c.n_out * (std::size_t) (c.n_in / c.block_elems);
    if (n_blocks * (std::size_t) c.block_bytes != wbytes) {
        std::printf("%-12s T=%d: %zu blocks of %d bytes are not the %zu weight bytes native_mmvq expects\n", c.name,
                    T, n_blocks, c.block_bytes, wbytes);
        return -1;
    }
    for (std::size_t k = 0; k < n_blocks; ++k)
        for (int at : c.scale_at)
            if (at >= 0) {
                const uint16_t h = sane_half(rng);
                std::memcpy(&w[k * (std::size_t) c.block_bytes + (std::size_t) at], &h, 2);
            }
    std::vector<float> x((std::size_t) T * c.n_in);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);

    void* dw = nullptr;
    void* xq = nullptr;
    float* dx = nullptr;
    float* dmulti = nullptr;
    float* dsingle = nullptr;
    const std::size_t ybytes = (std::size_t) T * c.n_out * 4;
    if (!ck(cudaMalloc(&dw, wbytes), "malloc w") || !ck(cudaMalloc(&xq, qall), "malloc xq") ||
        !ck(cudaMalloc(&dx, x.size() * 4), "malloc x") || !ck(cudaMalloc(&dmulti, ybytes), "malloc multi") ||
        !ck(cudaMalloc(&dsingle, ybytes), "malloc single"))
        return -1;
    if (!ck(cudaMemcpy(dw, w.data(), wbytes, cudaMemcpyHostToDevice), "copy w") ||
        !ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x"))
        return -1;

    // ONE quantization, shared by both paths: the quantization is not under test
    strata::kernels::quantize_q8_1_rows(dx, T, c.n_in, xq, s);
    if (!ck(cudaStreamSynchronize(s), "quantize")) return -1;

    try {
        strata::kernels::native_mmvq(c.type, dw, xq, dmulti, c.n_in, c.n_out, T, s);
        for (int j = 0; j < T; ++j)
            strata::kernels::native_mmvq(c.type, dw, (const uint8_t*) xq + (std::size_t) j * qcol,
                                         dsingle + (std::size_t) j * c.n_out, c.n_in, c.n_out, 1, s);
    } catch (const std::exception& e) {
        std::printf("%-12s T=%d: %s\n", c.name, T, e.what());
        return -1;
    }
    if (!ck(cudaStreamSynchronize(s), "mmvq")) return -1;

    std::vector<uint32_t> a((std::size_t) T * c.n_out), b(a.size());
    if (!ck(cudaMemcpy(a.data(), dmulti, ybytes, cudaMemcpyDeviceToHost), "read multi") ||
        !ck(cudaMemcpy(b.data(), dsingle, ybytes, cudaMemcpyDeviceToHost), "read single"))
        return -1;

    long long diff = 0;
    nonfinite = 0;
    diff_finite = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        float fa, fb;
        std::memcpy(&fa, &a[i], 4);
        std::memcpy(&fb, &b[i], 4);
        const bool both_finite = std::isfinite(fa) && std::isfinite(fb);
        if (!both_finite) ++nonfinite;
        if (a[i] != b[i]) {
            ++diff;
            if (both_finite) ++diff_finite;
        }
    }
    cudaFree(dw); cudaFree(xq); cudaFree(dx); cudaFree(dmulti); cudaFree(dsingle);
    ran = true;
    return diff;
}

struct Totals {
    long long checked = 0, diff = 0, diff_finite = 0, nonfinite = 0;   // all widths
    long long diff_finite_wide = 0;                                     // T > COINCIDE_MAX_T only
    int powerless_cases = 0;   // cases where the control found no difference at any T > COINCIDE_MAX_T
};

int sweep(bool exact, cudaStream_t s, Totals& t) {
    strata::kernels::native_mmvq_set_multi_exact(exact);
    if (strata::kernels::native_mmvq_multi_exact() != exact) {
        std::printf("native_mmvq_multi_exact() does not reflect what was set\n");
        return 1;
    }
    std::printf("\n=== multi_exact %s ===\n", exact ? "ON (the default)" : "off (negative control)");
    std::printf("%-12s %3s %10s %10s %14s %10s\n", "type", "T", "outputs", "bits diff", "finite diff", "nonfinite");
    int bad = 0;
    for (const Case& c : CASES) {
        long long case_wide = 0;
        for (int T : WIDTHS) {
            long long nf = 0, dfin = 0;
            bool ran = false;
            const long long diff = compare(c, T, s, nf, dfin, ran);
            if (!ran) { ++bad; continue; }
            const long long n = (long long) T * c.n_out;
            t.checked += n;
            t.diff += diff;
            t.diff_finite += dfin;
            t.nonfinite += nf;
            if (T > COINCIDE_MAX_T) { t.diff_finite_wide += dfin; case_wide += dfin; }
            const char* verdict;
            if (exact) verdict = diff == 0 ? "ok" : "FAIL";
            else if (dfin > 0) verdict = "control sees it";
            else verdict = T <= COINCIDE_MAX_T ? "layouts coincide here (NW = 4)" : "no power at this T";
            std::printf("%-12s %3d %10lld %10lld %14lld %10lld  %s\n", c.name, T, n, diff, dfin, nf, verdict);
            if (exact && diff != 0) ++bad;
        }
        if (!exact && case_wide == 0) ++t.powerless_cases;
    }
    return bad;
}

}  // namespace

// A11 STRATA_MMVQ_BUNCH: the q|k|v (three widths) and qkv|z (two widths) bundles against the sequential
// native_mmvq calls on the SAME q8_1 bytes - bitwise, every output, 2-8 columns.  The MiMo v2 shapes ride
// along (2026-11-05 dense-prefill design): q 12288 rows, k|v 768 (full) and 1536 (swa) rows, all at
// n_in = 4096; the engine's verify windows reach T = 8, so the sweep does too.
int bunch_parity(cudaStream_t s) {
    struct B { int n_out; };
    const struct { int n_in; int m; B b[3]; } cases[] = {
        {2560, 3, {{2048}, {512}, {512}}},   // QSA q|k|v
        {2560, 2, {{10240}, {640}, {0}}},    // GDN qkv|z
        {4096, 3, {{12288}, {768}, {768}}},   // MiMo q|k|v, full-attention layer
        {4096, 3, {{12288}, {1536}, {1536}}}, // MiMo q|k|v, SWA layer
    };
    int fails = 0;
    for (const auto& c : cases)
        for (int T = 2; T <= 8; ++T) {
            std::mt19937 rng(777u + (unsigned) c.n_in + (unsigned) T);
            const std::size_t qall = strata::kernels::native_q8_1_bytes(c.n_in, T);
            void* xq = nullptr;
            if (!ck(cudaMalloc(&xq, qall), "bunch malloc xq")) return 1;
            std::vector<float> x((std::size_t) T * c.n_in);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (auto& v : x) v = nd(rng);
            float* dx = nullptr;
            if (!ck(cudaMalloc(&dx, x.size() * 4), "bunch malloc x")) return 1;
            if (!ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "bunch copy x")) return 1;
            strata::kernels::native_quantize_q8_1(dx, xq, c.n_in, T, s);
            const void* w[3] = {nullptr, nullptr, nullptr};
            float* yseq[3] = {nullptr, nullptr, nullptr};
            float* ybun[3] = {nullptr, nullptr, nullptr};
            std::vector<std::vector<uint8_t>> hw(c.m);
            for (int m = 0; m < c.m; ++m) {
                const std::size_t wb = strata::kernels::native_mmvq_weight_bytes(8, c.n_in, c.b[m].n_out);
                hw[m].resize(wb);
                for (auto& b : hw[m]) b = (uint8_t) (rng() & 0xff);
                const std::size_t nb = (std::size_t) c.b[m].n_out * (std::size_t) (c.n_in / 32);
                for (std::size_t k = 0; k < nb; ++k) {   // the Q80Block's half d at offset 0, finite scales
                    const uint16_t h = sane_half(rng);
                    std::memcpy(&hw[m][k * 34], &h, 2);
                }
                void* dwm = nullptr;
                if (!ck(cudaMalloc(&dwm, wb), "bunch malloc w")) return 1;
                if (!ck(cudaMemcpy(dwm, hw[m].data(), wb, cudaMemcpyHostToDevice), "bunch copy w")) return 1;
                w[m] = dwm;
                if (!ck(cudaMalloc(&yseq[m], (std::size_t) T * c.b[m].n_out * 4), "bunch malloc yseq")) return 1;
                if (!ck(cudaMalloc(&ybun[m], (std::size_t) T * c.b[m].n_out * 4), "bunch malloc ybun")) return 1;
                strata::kernels::native_mmvq(8, w[m], xq, yseq[m], c.n_in, c.b[m].n_out, T, s);
            }
            if (!strata::kernels::native_mmvq_bunch(8, w[0], w[1], c.m > 2 ? w[2] : nullptr, xq, ybun[0], ybun[1],
                                                    c.m > 2 ? ybun[2] : nullptr, c.n_in, c.b[0].n_out,
                                                    c.b[1].n_out, c.m > 2 ? c.b[2].n_out : 0, T, s)) {
                std::printf("bunch n_in=%d m=%d T=%d: declined\n", c.n_in, c.m, T);
                ++fails;
            } else {
                long long diff = 0, nonf = 0;
                for (int m = 0; m < c.m; ++m) {
                    std::vector<float> a((std::size_t) T * c.b[m].n_out), b((std::size_t) T * c.b[m].n_out);
                    if (!ck(cudaMemcpy(a.data(), yseq[m], a.size() * 4, cudaMemcpyDeviceToHost), "bunch cy a")) return 1;
                    if (!ck(cudaMemcpy(b.data(), ybun[m], b.size() * 4, cudaMemcpyDeviceToHost), "bunch cy b")) return 1;
                    for (size_t i = 0; i < a.size(); ++i) {
                        if (!std::isfinite(a[i]) && !std::isfinite(b[i])) { ++nonf; continue; }
                        if (std::memcmp(&a[i], &b[i], 4) != 0) ++diff;
                    }
                }
                std::printf("bunch n_in=%d m=%d T=%d: %lld differ bitwise, %lld non-finite\n", c.n_in, c.m, T, diff, nonf);
                if (diff != 0 || nonf != 0) ++fails;
            }
            for (int m = 0; m < c.m; ++m) {
                (void) cudaFree((void*) w[m]); (void) cudaFree(yseq[m]); (void) cudaFree(ybun[m]);
            }
            (void) cudaFree(xq); (void) cudaFree(dx);
        }
    return fails;
}

// ---- P2/P1 (2026-11-05 dense-prefill design) -------------------------------------------------------
//
// wide_parity covers, all bitwise on finite outputs:
//   * native_q8_0_mmvq_wide at NCOLS 16/32 vs the engine's TODAY path (8-column native_mmvq tiles) on the
//     SAME q8_1 image - the MiMo dense shapes q 12288, kv 768/1536 (n_in 4096) and o 4096 (n_in 8192), with
//     bounded normals and a saturation-scale magnitude pass, every output column checked written (sentinel);
//   * native_mmvq_bunch_wide (q|k|v, one launch) vs those same per-matrix tiles;
//   * P1's shared image: quantizing 32 columns at once == 8 at a time == 1 at a time, byte for byte;
//   * P1's decode fallback: quantize once + native_q8_0_mmvq(1) == native_q8_0_f32(1);
//   * native_q8_0_f32_wide == native_quantize_q8_1 + native_q8_0_mmvq_wide (and its quantized bytes);
//   * the decline guards: multi_exact off (a declined f32_wide leaves the caller's scratch UNTOUCHED - no
//     enqueue happens before eligibility is decided), a Q8_0-packed-registered weight, wrong widths (8/24),
//     and STRATA_TSUM=1 / a gfx906 wave build, where the wide entries MUST decline (the exact-layout
//     comparisons run only where the exact layout is the baseline).
namespace {

bool wide_env_allows() {
    const char* t = std::getenv("STRATA_TSUM");
    if (t && std::string(t) == "1") return false;
#if defined(STRATA_HIP_GFX906)
    const char* w = std::getenv("STRATA_MMVQ_WAVE");
    if (!(w && std::string(w) == "0")) return false;   // the wave kernels are the baseline there
#endif
    return true;
}

std::vector<uint8_t> q8_0_weights(std::mt19937& rng, int n_in, int n_out) {
    const std::size_t wb = strata::kernels::native_mmvq_weight_bytes(8, n_in, n_out);
    std::vector<uint8_t> w(wb);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    const std::size_t nb = (std::size_t) n_out * (std::size_t) (n_in / 32);
    for (std::size_t k = 0; k < nb; ++k) {   // Q80Block: a finite fp16 d at offset 0 (the file's data policy)
        const uint16_t h = sane_half(rng);
        std::memcpy(&w[k * 34], &h, 2);
    }
    return w;
}

}  // namespace

int wide_parity(cudaStream_t s) {
    using strata::kernels::native_mmvq;
    using strata::kernels::native_mmvq_bunch_wide;
    using strata::kernels::native_mmvq_set_multi_exact;
    using strata::kernels::native_q8_0_f32_wide;
    using strata::kernels::native_q8_0_mmvq_wide;
    using strata::kernels::native_q8_1_bytes;
    using strata::kernels::native_quantize_q8_1;
    constexpr uint32_t SENTINEL = 0x7f7f7f7fu;   // cudaMemset 0x7f: ~3.4e38, unreachable for this data
    int fails = 0;
    const bool allow = wide_env_allows();
    if (!allow)
        std::printf("\nwide: STRATA_TSUM=1 or a gfx906 wave build - the wide entries must DECLINE here; the "
                    "bitwise sections need the exact layout (on gfx906: STRATA_MMVQ_WAVE=0).\n");

    // ---- the single-matrix wide launch vs today's 8-column tiles -------------------------------
    struct S { int n_in, n_out; const char* what; };
    std::vector<S> shapes = {{4096, 12288, "q"}, {4096, 768, "kv-full"}, {4096, 1536, "kv-swa"},
                             {8192, 4096, "o"}};
    const char* lm = std::getenv("STRATA_PARITY_LM");
    if (lm && lm[0] == '1') shapes.push_back({4096, 152576, "lm_head"});   // ~633 MB of weights: opt-in
    for (const S& sh : shapes)
        for (int ncols : {16, 32})
            for (float scale : {1.0f, 1000.0f}) {
                std::mt19937 rng(91u + (unsigned) sh.n_in + (unsigned) sh.n_out + (unsigned) ncols +
                                     (unsigned) int(scale));
                const std::vector<uint8_t> w = q8_0_weights(rng, sh.n_in, sh.n_out);
                std::vector<float> x((std::size_t) ncols * sh.n_in);
                std::normal_distribution<float> nd(0.f, 1.f);
                for (auto& v : x) v = nd(rng) * scale;   // bounded, then saturation-scale magnitudes
                const std::size_t qall = strata::kernels::native_q8_1_wide_bytes(sh.n_in, ncols);
                const std::size_t qcol = native_q8_1_bytes(sh.n_in, 1);
                const std::size_t ybytes = (std::size_t) ncols * sh.n_out * 4;
                void* dw = nullptr;
                void* xq = nullptr;
                float *dx = nullptr, *ywide = nullptr, *ybase = nullptr;
                if (!ck(cudaMalloc(&dw, w.size()), "wide malloc w") || !ck(cudaMalloc(&xq, qall), "wide malloc xq") ||
                    !ck(cudaMalloc(&dx, x.size() * 4), "wide malloc x") ||
                    !ck(cudaMalloc(&ywide, ybytes), "wide malloc ywide") ||
                    !ck(cudaMalloc(&ybase, ybytes), "wide malloc ybase"))
                    return 1;
                if (!ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "wide copy w") ||
                    !ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "wide copy x"))
                    return 1;
                if (!ck(cudaMemset(ywide, 0x7f, ybytes), "wide sentinel") ||
                    !ck(cudaMemset(ybase, 0x7f, ybytes), "base sentinel"))
                    return 1;
                strata::kernels::native_quantize_q8_1_wide(dx, xq, sh.n_in, ncols, s);   // ONE image, shared by both paths
                for (int r = 0; r < ncols; r += 8)   // today's engine loop: 8-column native_mmvq tiles
                    native_mmvq(8, dw, (const uint8_t*) xq + (std::size_t) r * qcol,
                                ybase + (std::size_t) r * sh.n_out, sh.n_in, sh.n_out, 8, s);
                const bool ran = native_q8_0_mmvq_wide(dw, xq, ywide, sh.n_in, sh.n_out, ncols, s);
                if (!ck(cudaStreamSynchronize(s), "wide sync")) return 1;
                bool bad = ran != allow;
                long long diff = 0, nonf = 0, sentinel = 0;
                if (ran) {
                    std::vector<uint32_t> a((std::size_t) ncols * sh.n_out), b(a.size());
                    if (!ck(cudaMemcpy(a.data(), ywide, ybytes, cudaMemcpyDeviceToHost), "wide read") ||
                        !ck(cudaMemcpy(b.data(), ybase, ybytes, cudaMemcpyDeviceToHost), "base read"))
                        return 1;
                    for (std::size_t i = 0; i < a.size(); ++i) {
                        float f;
                        std::memcpy(&f, &a[i], 4);
                        if (!std::isfinite(f)) ++nonf;
                        if (a[i] == SENTINEL) ++sentinel;
                        if (a[i] != b[i]) ++diff;
                    }
                }
                bad = bad || diff != 0 || nonf != 0 || sentinel != 0;
                if (bad) ++fails;
                std::printf("wide %-8s n_in=%5d n_out=%6d nc=%2d scale=%5g: %s%lld differ, %lld non-finite, "
                            "%lld unwritten\n",
                            sh.what, sh.n_in, sh.n_out, ncols, scale,
                            ran != allow ? (allow ? "DECLINED-BAD " : "RAN-BAD ") : "", diff, nonf, sentinel);
                cudaFree(dw);
                cudaFree(xq);
                cudaFree(dx);
                cudaFree(ywide);
                cudaFree(ybase);
            }

    // ---- the q|k|v bundle at 16/32: one launch vs the same per-matrix 8-column tiles -------------
    const struct { int n_out[3]; const char* what; } bq[] = {{{12288, 768, 768}, "full"},
                                                             {{12288, 1536, 1536}, "swa"}};
    for (const auto& c : bq)
        for (int ncols : {16, 32}) {
            std::mt19937 rng(4242u + (unsigned) c.n_out[1] + (unsigned) ncols);
            std::vector<std::vector<uint8_t>> w;
            for (int m = 0; m < 3; ++m) w.push_back(q8_0_weights(rng, 4096, c.n_out[m]));
            std::vector<float> x((std::size_t) ncols * 4096);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (auto& v : x) v = nd(rng);
            const std::size_t qall = strata::kernels::native_q8_1_wide_bytes(4096, ncols);
            const std::size_t qcol = native_q8_1_bytes(4096, 1);
            void* dw[3] = {nullptr, nullptr, nullptr};
            void* xq = nullptr;
            float* dx = nullptr;
            float *yb[3] = {nullptr, nullptr, nullptr}, *yu[3] = {nullptr, nullptr, nullptr};
            bool alloc_ok = ck(cudaMalloc(&xq, qall), "bw malloc xq") &&
                            ck(cudaMalloc(&dx, x.size() * 4), "bw malloc x") &&
                            ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "bw copy x");
            for (int m = 0; m < 3; ++m)
                alloc_ok = alloc_ok && ck(cudaMalloc(&dw[m], w[m].size()), "bw malloc w") &&
                           ck(cudaMemcpy(dw[m], w[m].data(), w[m].size(), cudaMemcpyHostToDevice), "bw copy w") &&
                           ck(cudaMalloc(&yb[m], (std::size_t) ncols * c.n_out[m] * 4), "bw malloc yb") &&
                           ck(cudaMalloc(&yu[m], (std::size_t) ncols * c.n_out[m] * 4), "bw malloc yu") &&
                           ck(cudaMemset(yb[m], 0x7f, (std::size_t) ncols * c.n_out[m] * 4), "bw sentinel b") &&
                           ck(cudaMemset(yu[m], 0x7f, (std::size_t) ncols * c.n_out[m] * 4), "bw sentinel u");
            if (!alloc_ok) return 1;
            strata::kernels::native_quantize_q8_1_wide(dx, xq, 4096, ncols, s);
            for (int m = 0; m < 3; ++m)
                for (int r = 0; r < ncols; r += 8)
                    native_mmvq(8, dw[m], (const uint8_t*) xq + (std::size_t) r * qcol,
                                yb[m] + (std::size_t) r * c.n_out[m], 4096, c.n_out[m], 8, s);
            const bool ran = native_mmvq_bunch_wide(8, dw[0], dw[1], dw[2], xq, yu[0], yu[1], yu[2], 4096,
                                                    c.n_out[0], c.n_out[1], c.n_out[2], ncols, s);
            if (!ck(cudaStreamSynchronize(s), "bw sync")) return 1;
            bool bad = ran != allow;
            long long diff = 0, nonf = 0, sentinel = 0;
            if (ran)
                for (int m = 0; m < 3; ++m) {
                    std::vector<uint32_t> a((std::size_t) ncols * c.n_out[m]), b(a.size());
                    if (!ck(cudaMemcpy(a.data(), yu[m], a.size() * 4, cudaMemcpyDeviceToHost), "bw read u") ||
                        !ck(cudaMemcpy(b.data(), yb[m], b.size() * 4, cudaMemcpyDeviceToHost), "bw read b"))
                        return 1;
                    for (std::size_t i = 0; i < a.size(); ++i) {
                        float f;
                        std::memcpy(&f, &a[i], 4);
                        if (!std::isfinite(f)) ++nonf;
                        if (a[i] == SENTINEL) ++sentinel;
                        if (a[i] != b[i]) ++diff;
                    }
                }
            bad = bad || diff != 0 || nonf != 0 || sentinel != 0;
            if (bad) ++fails;
            std::printf("wide bunch %-4s nc=%2d: %s%lld differ, %lld non-finite, %lld unwritten\n", c.what, ncols,
                        ran != allow ? (allow ? "DECLINED-BAD " : "RAN-BAD ") : "", diff, nonf, sentinel);
            for (int m = 0; m < 3; ++m) {
                if (dw[m]) cudaFree(dw[m]);
                if (yb[m]) cudaFree(yb[m]);
                if (yu[m]) cudaFree(yu[m]);
            }
            cudaFree(xq);
            cudaFree(dx);
        }

    // ---- P1's shared image: 32 columns at once == 8 at a time == 1 at a time, byte for byte -------
    if (allow) {
        const int n_in = 4096, ncols = 32;
        std::mt19937 rng(31337u);
        std::vector<float> x((std::size_t) ncols * n_in);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(rng);
        const std::size_t qall = strata::kernels::native_q8_1_wide_bytes(n_in, ncols);
        const std::size_t qcol = native_q8_1_bytes(n_in, 1);
        float* dx = nullptr;
        void *q32 = nullptr, *q8 = nullptr, *q1 = nullptr;
        if (!ck(cudaMalloc(&dx, x.size() * 4), "qi malloc x") || !ck(cudaMalloc(&q32, qall), "qi malloc q32") ||
            !ck(cudaMalloc(&q8, qall), "qi malloc q8") || !ck(cudaMalloc(&q1, qall), "qi malloc q1"))
            return 1;
        if (!ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "qi copy x")) return 1;
        strata::kernels::native_quantize_q8_1_wide(dx, q32, n_in, ncols, s);
        for (int r = 0; r < ncols; r += 8)
            native_quantize_q8_1(dx + (std::size_t) r * n_in, (uint8_t*) q8 + (std::size_t) r * qcol, n_in, 8, s);
        for (int j = 0; j < ncols; ++j)
            native_quantize_q8_1(dx + (std::size_t) j * n_in, (uint8_t*) q1 + (std::size_t) j * qcol, n_in, 1, s);
        if (!ck(cudaStreamSynchronize(s), "qi sync")) return 1;
        std::vector<uint8_t> h32(qall), h8(qall), h1(qall);
        if (!ck(cudaMemcpy(h32.data(), q32, qall, cudaMemcpyDeviceToHost), "qi read 32") ||
            !ck(cudaMemcpy(h8.data(), q8, qall, cudaMemcpyDeviceToHost), "qi read 8") ||
            !ck(cudaMemcpy(h1.data(), q1, qall, cudaMemcpyDeviceToHost), "qi read 1"))
            return 1;
        const bool same = h32 == h8 && h32 == h1;
        if (!same) ++fails;
        std::printf("wide q8_1 image: 32-at-once vs 8-at-a-time vs 1-at-a-time: %s\n",
                    same ? "identical bytes" : "DIFFER");
        cudaFree(dx);
        cudaFree(q32);
        cudaFree(q8);
        cudaFree(q1);
    }

    // ---- P1's decode fallback: quantize once + mmvq(1) == native_q8_0_f32(1) ---------------------
    if (allow)
        for (const S& sh : std::vector<S>{{4096, 12288, "q"}, {4096, 768, "k/v-full"}, {4096, 1536, "k/v-swa"},
                                          {8192, 4096, "o"}}) {
            std::mt19937 rng(5150u + (unsigned) sh.n_out);
            const std::vector<uint8_t> w = q8_0_weights(rng, sh.n_in, sh.n_out);
            std::vector<float> x((std::size_t) sh.n_in);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (auto& v : x) v = nd(rng);
            const std::size_t q1 = native_q8_1_bytes(sh.n_in, 1);
            void* dw = nullptr;
            void* qa = nullptr;
            void* qb = nullptr;
            float *dx = nullptr, *ya = nullptr, *yb = nullptr;
            if (!ck(cudaMalloc(&dw, w.size()), "nc1 malloc w") || !ck(cudaMalloc(&qa, q1), "nc1 malloc qa") ||
                !ck(cudaMalloc(&qb, q1), "nc1 malloc qb") || !ck(cudaMalloc(&dx, x.size() * 4), "nc1 malloc x") ||
                !ck(cudaMalloc(&ya, (std::size_t) sh.n_out * 4), "nc1 malloc ya") ||
                !ck(cudaMalloc(&yb, (std::size_t) sh.n_out * 4), "nc1 malloc yb"))
                return 1;
            if (!ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "nc1 copy w") ||
                !ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "nc1 copy x"))
                return 1;
            strata::kernels::native_q8_0_f32(dw, dx, qa, ya, sh.n_in, sh.n_out, 1, s);   // today's call
            native_quantize_q8_1(dx, qb, sh.n_in, 1, s);                                 // the shared quantization
            strata::kernels::native_q8_0_mmvq(dw, qb, yb, sh.n_in, sh.n_out, 1, s);
            if (!ck(cudaStreamSynchronize(s), "nc1 sync")) return 1;
            std::vector<uint32_t> a((std::size_t) sh.n_out), b(a.size());
            std::vector<uint8_t> ha(q1), hb(q1);
            if (!ck(cudaMemcpy(a.data(), ya, a.size() * 4, cudaMemcpyDeviceToHost), "nc1 read ya") ||
                !ck(cudaMemcpy(b.data(), yb, b.size() * 4, cudaMemcpyDeviceToHost), "nc1 read yb") ||
                !ck(cudaMemcpy(ha.data(), qa, q1, cudaMemcpyDeviceToHost), "nc1 read qa") ||
                !ck(cudaMemcpy(hb.data(), qb, q1, cudaMemcpyDeviceToHost), "nc1 read qb"))
                return 1;
            const bool same = a == b && ha == hb;
            if (!same) ++fails;
            std::printf("wide nc=1 share %-9s (n_in %d, n_out %d): %s\n", sh.what, sh.n_in, sh.n_out,
                        same ? "identical bytes" : "DIFFER");
            cudaFree(dw);
            cudaFree(qa);
            cudaFree(qb);
            cudaFree(dx);
            cudaFree(ya);
            cudaFree(yb);
        }

    // ---- the composed entry == its parts, including the quantized image --------------------------
    if (allow) {
        const int n_in = 4096, n_out = 12288, ncols = 32;
        std::mt19937 rng(6060u);
        const std::vector<uint8_t> w = q8_0_weights(rng, n_in, n_out);
        std::vector<float> x((std::size_t) ncols * n_in);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(rng);
        const std::size_t qall = strata::kernels::native_q8_1_wide_bytes(n_in, ncols);
        void *dw = nullptr, *qa = nullptr, *qb = nullptr;
        float *dx = nullptr, *ya = nullptr, *yb = nullptr;
        if (!ck(cudaMalloc(&dw, w.size()), "cw malloc w") || !ck(cudaMalloc(&qa, qall), "cw malloc qa") ||
            !ck(cudaMalloc(&qb, qall), "cw malloc qb") || !ck(cudaMalloc(&dx, x.size() * 4), "cw malloc x") ||
            !ck(cudaMalloc(&ya, (std::size_t) ncols * n_out * 4), "cw malloc ya") ||
            !ck(cudaMalloc(&yb, (std::size_t) ncols * n_out * 4), "cw malloc yb"))
            return 1;
        if (!ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "cw copy w") ||
            !ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "cw copy x"))
            return 1;
        strata::kernels::native_quantize_q8_1_wide(dx, qa, n_in, ncols, s);
        if (!native_q8_0_mmvq_wide(dw, qa, ya, n_in, n_out, ncols, s)) {
            std::printf("wide composed: the parts ran but mmvq_wide declined - FAIL\n");
            ++fails;
        } else if (!native_q8_0_f32_wide(dw, dx, qb, yb, n_in, n_out, ncols, s)) {
            std::printf("wide composed: f32_wide declined - FAIL\n");
            ++fails;
        } else {
            if (!ck(cudaStreamSynchronize(s), "cw sync")) return 1;
            std::vector<uint32_t> a((std::size_t) ncols * n_out), b(a.size());
            std::vector<uint8_t> ha(qall), hb(qall);
            if (!ck(cudaMemcpy(a.data(), ya, a.size() * 4, cudaMemcpyDeviceToHost), "cw read ya") ||
                !ck(cudaMemcpy(b.data(), yb, b.size() * 4, cudaMemcpyDeviceToHost), "cw read yb") ||
                !ck(cudaMemcpy(ha.data(), qa, qall, cudaMemcpyDeviceToHost), "cw read qa") ||
                !ck(cudaMemcpy(hb.data(), qb, qall, cudaMemcpyDeviceToHost), "cw read qb"))
                return 1;
            const bool same = a == b && ha == hb;
            if (!same) ++fails;
            std::printf("wide composed f32_wide == quantize + mmvq_wide: %s\n",
                        same ? "identical bytes" : "DIFFER");
        }
        cudaFree(dw);
        cudaFree(qa);
        cudaFree(qb);
        cudaFree(dx);
        cudaFree(ya);
        cudaFree(yb);
    }

    // ---- decline guards ---------------------------------------------------------------------------
    {
        // multi_exact off: every new entry declines, bunch_supported reports false, and a declined
        // f32_wide does not even enqueue the quantize (the caller's scratch bytes stay as they were)
        native_mmvq_set_multi_exact(false);
        if (strata::kernels::native_mmvq_bunch_supported(8, 4096, 4)) {
            std::printf("guard: bunch_supported true under multi_exact off - FAIL\n");
            ++fails;
        }
        const int n_in = 4096, n_out = 768;
        std::mt19937 rng(77u);
        const std::vector<uint8_t> w = q8_0_weights(rng, n_in, n_out);
        std::vector<float> x((std::size_t) 32 * n_in);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(rng);
        const std::size_t qall = strata::kernels::native_q8_1_wide_bytes(n_in, 32);
        void *dw = nullptr, *scratch = nullptr;
        float *dx = nullptr, *y = nullptr;
        if (!ck(cudaMalloc(&dw, w.size()), "gd malloc w") || !ck(cudaMalloc(&scratch, qall), "gd malloc s") ||
            !ck(cudaMalloc(&dx, x.size() * 4), "gd malloc x") ||
            !ck(cudaMalloc(&y, (std::size_t) 32 * n_out * 4), "gd malloc y"))
            return 1;
        if (!ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "gd copy w") ||
            !ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "gd copy x") ||
            !ck(cudaMemset(scratch, 0xa5, qall), "gd sentinel"))
            return 1;
        if (native_q8_0_mmvq_wide(dw, scratch, y, n_in, n_out, 32, s) ||
            native_q8_0_f32_wide(dw, dx, scratch, y, n_in, n_out, 16, s) ||
            native_mmvq_bunch_wide(8, dw, dw, nullptr, scratch, y, y, nullptr, n_in, n_out, n_out, 0, 32, s)) {
            std::printf("guard: a wide entry launched under multi_exact off - FAIL\n");
            ++fails;
        }
        if (!ck(cudaStreamSynchronize(s), "gd sync")) return 1;
        std::vector<uint8_t> h(qall);
        if (!ck(cudaMemcpy(h.data(), scratch, qall, cudaMemcpyDeviceToHost), "gd read s")) return 1;
        bool untouched = true;
        for (uint8_t v : h)
            if (v != 0xa5) untouched = false;
        if (!untouched) {
            std::printf("guard: declined f32_wide left the scratch untouched: FAIL (bytes were written)\n");
            ++fails;
        } else {
            std::printf("guard: multi_exact off - all wide entries decline, scratch untouched\n");
        }
        cudaFree(dw);
        cudaFree(scratch);
        cudaFree(dx);
        cudaFree(y);
        native_mmvq_set_multi_exact(true);
    }
    {
        // a Q8_0-packed-registered weight (STRATA_Q8_PACKED): the wide entries and the bunch decline
        const int n_in = 1024, n_out = 2048;   // inside native_q8_0_packed_eligible's window
        std::mt19937 rng(78u);
        const std::vector<uint8_t> w = q8_0_weights(rng, n_in, n_out);
        const std::size_t qall = strata::kernels::native_q8_1_wide_bytes(n_in, 32);
        void *dw = nullptr, *packed = nullptr, *xq = nullptr;
        float *y1 = nullptr, *y2 = nullptr;
        if (!ck(cudaMalloc(&dw, w.size()), "gp malloc w") || !ck(cudaMalloc(&packed, w.size()), "gp malloc p") ||
            !ck(cudaMalloc(&xq, qall), "gp malloc xq") ||
            !ck(cudaMalloc(&y1, (std::size_t) 32 * n_out * 4), "gp malloc y1") ||
            !ck(cudaMalloc(&y2, (std::size_t) 32 * n_out * 4), "gp malloc y2"))
            return 1;
        if (!ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "gp copy w")) return 1;
        strata::kernels::native_q8_0_packed_register(dw, packed, n_in, n_out);
        const bool declined =
            !native_q8_0_mmvq_wide(dw, xq, y1, n_in, n_out, 32, s) &&
            !native_q8_0_f32_wide(dw, y1, xq, y1, n_in, n_out, 16, s) &&
            !native_mmvq_bunch_wide(8, dw, dw, nullptr, xq, y1, y2, nullptr, n_in, n_out, n_out, 0, 32, s) &&
            !strata::kernels::native_mmvq_bunch(8, dw, dw, nullptr, xq, y1, y2, nullptr, n_in, n_out, n_out, 0,
                                                 4, s);
        strata::kernels::native_q8_0_packed_unregister(dw);
        if (!declined) {
            std::printf("guard: a packed-registered weight must decline every fused/wide entry - FAIL\n");
            ++fails;
        } else {
            std::printf("guard: Q8_0-packed weight - wide/bunch/bunch_wide all decline\n");
        }
        cudaFree(dw);
        cudaFree(packed);
        cudaFree(xq);
        cudaFree(y1);
        cudaFree(y2);
    }
    {   // misaligned pointers, n_out3 < 0 and a grid sum past dim3.x/int all decline BEFORE any enqueue -
        // the bool contract, not a throw; odd pointers never reach the registry lookup or CUDA
        void* odd = reinterpret_cast<void*>(0x10002);   // 2-byte aligned only
        void* aligned = reinterpret_cast<void*>(0x10000);
        float* odd_y = reinterpret_cast<float*>(0x1000a);
        const bool ok = !native_q8_0_mmvq_wide(odd, aligned, odd_y, 4096, 768, 32, s) &&
                        !native_q8_0_mmvq_wide(aligned, odd, odd_y, 4096, 768, 32, s) &&
                        !native_q8_0_mmvq_wide(aligned, aligned, odd_y, 4096, 768, 32, s) &&
                        !native_q8_0_f32_wide(odd, (const float*) aligned, aligned, odd_y, 4096, 768, 16, s) &&
                        !native_q8_0_f32_wide(aligned, (const float*) odd, aligned, odd_y, 4096, 768, 16, s) &&
                        !native_q8_0_f32_wide(aligned, (const float*) aligned, odd, odd_y, 4096, 768, 16, s) &&
                        !native_mmvq_bunch_wide(8, odd, aligned, nullptr, aligned, odd_y, odd_y, nullptr, 4096,
                                                768, 768, 0, 32, s) &&
                        !native_mmvq_bunch_wide(8, aligned, aligned, nullptr, odd, odd_y, odd_y, nullptr, 4096,
                                                768, 768, 0, 32, s) &&
                        !native_mmvq_bunch_wide(8, aligned, aligned, nullptr, aligned, odd_y, odd_y, nullptr,
                                                4096, 768, 768, -1, 32, s) &&   // n_out3 < 0: rejected
                        !native_q8_0_mmvq_wide(aligned, aligned, odd_y, 4096, 0, 32, s) &&   // n_out <= 0
                        !native_q8_0_mmvq_wide(aligned, aligned, reinterpret_cast<float*>(aligned), 4096, 768,
                                                32, nullptr) &&   // null stream
                        !native_mmvq_bunch_wide(8, aligned, aligned, aligned, aligned,
                                                reinterpret_cast<float*>(0x10000),
                                                reinterpret_cast<float*>(0x10000),
                                                reinterpret_cast<float*>(0x10000), 4096, 2147483647,
                                                2147483647, 2147483647, 32,
                                                s);   // summed grid > dim3.x: rejected in 64 bits
        if (!ok) {
            std::printf("guard: alignment / n_out3 / overflow rejections wrong - FAIL\n");
            ++fails;
        } else {
            std::printf("guard: misaligned ptrs, n_out3 < 0, null stream and grid overflow all decline\n");
        }
    }
    {   // the width table: 8 and 24 are not wide (the 1..8 cap elsewhere is untouched); bunch_supported
        // answers its own table under the default layout
        void* dummy = reinterpret_cast<void*>(16);   // eligibility-only calls never enqueue
        const bool widths_ok = !native_q8_0_mmvq_wide(dummy, dummy, nullptr, 4096, 768, 8, s) &&
                               !native_q8_0_mmvq_wide(dummy, dummy, nullptr, 4096, 768, 24, s);
        bool table_ok = widths_ok;
        for (int nc = 2; nc <= 8; ++nc)
            table_ok = strata::kernels::native_mmvq_bunch_supported(8, 4096, nc) && table_ok;
        table_ok = table_ok && !strata::kernels::native_mmvq_bunch_supported(8, 4096, 1) &&
                   !strata::kernels::native_mmvq_bunch_supported(8, 4096, 9) &&
                   !strata::kernels::native_mmvq_bunch_supported(8, 64, 4) &&
                   !strata::kernels::native_mmvq_bunch_supported(2, 4096, 4);
        if (!table_ok) {
            std::printf("guard: width/eligibility table wrong - FAIL\n");
            ++fails;
        } else {
            std::printf("guard: widths 8/24 decline; bunch_supported table ok (nc 2..8 yes, 1/9/short/type no)\n");
        }
    }
    return fails;
}

int main() {
    cudaStream_t s;
    if (!ck(cudaStreamCreate(&s), "stream create")) return 1;

    const int bad_bunch = bunch_parity(s);
    Totals on, off;
    const int bad = sweep(true, s, on);
    const int bad_off = sweep(false, s, off);
    strata::kernels::native_mmvq_set_multi_exact(true);   // back to the default
    const int bad_wide = wide_parity(s);   // the default layout restored: wide needs it

    std::printf("\nmulti_exact on:   %lld outputs compared, %lld differ bitwise, %lld non-finite\n",
                on.checked, on.diff, on.nonfinite);
    std::printf("multi_exact off:  %lld outputs compared, %lld differ bitwise (%lld on finite outputs, %lld at T > %d)\n",
                off.checked, off.diff, off.diff_finite, off.diff_finite_wide, COINCIDE_MAX_T);

    if (on.nonfinite != 0 || off.nonfinite != 0) {
        std::printf("\nmmvq_multi_parity: FAILED. %lld non-finite outputs from finite scales and activations: the "
                    "bit comparison is only meaningful on finite outputs.\n", on.nonfinite + off.nonfinite);
        return 1;
    }
    if (bad_bunch != 0) {
        std::printf("\nmmvq_multi_parity: FAILED. %d bunch case(s) differ from the sequential calls or were declined.\n",
                    bad_bunch);
        return 1;
    }
    if (bad_wide != 0) {
        std::printf("\nmmvq_multi_parity: FAILED. %d wide/fused case(s) differ from the 8-column tiles, were "
                    "declined where they must run, ran where they must decline, or wrote bytes on a declined "
                    "call.\n",
                    bad_wide);
        return 1;
    }
    if (bad != 0 || bad_off != 0) {
        std::printf("\nmmvq_multi_parity: FAILED. %d case(s) with multi_exact on differ from single-column calls "
                    "or did not run, %d case(s) did not run with it off. If they differ, the sentence in "
                    "native_mmvq.hpp does not hold, and neither does the bit-exactness verify.hpp claims.\n",
                    bad, bad_off);
        return 1;
    }
    // THE NEGATIVE CONTROL, asserted where it can see something: past T = 4 the generic layout is a different one,
    // so if it still finds no difference the comparison cannot tell the layouts apart and the pass proves nothing.
#if defined(STRATA_HIP_GFX906)
    // gfx906: the wave64 layout (one wavefront per row) is the same kernel for every column count, so multi_exact
    // has nothing to switch and the control cannot see a difference.  The control with power is the CUDA layout:
    // ctest's mmvq_multi_parity_cuda_layout runs this with STRATA_MMVQ_WAVE=0.
    if (off.powerless_cases != 0 && !(std::getenv("STRATA_MMVQ_WAVE") && std::string(std::getenv("STRATA_MMVQ_WAVE")) == "0")) {
        std::printf("\nmmvq_multi_parity: ok (gfx906 wave64 layout: every column count runs the same exact kernel, "
                    "so the negative control has no other layout to find; mmvq_multi_parity_cuda_layout checks it).\n");
        return 0;
    }
#endif
    if (off.powerless_cases != 0) {
        std::printf("\nmmvq_multi_parity: FAILED (no power). In %d case(s) no output differs with multi_exact off at "
                    "any T > %d, where the generic layout is not the exact one: for those cases this comparison "
                    "cannot tell the two layouts apart, so their pass above proves nothing.\n", off.powerless_cases,
                    COINCIDE_MAX_T);
        return 1;
    }
    std::printf("\nmmvq_multi_parity: ok. With multi_exact on every column is bitwise equal to a single-column "
                "call; with it off the control finds differences at T > %d in every case, so the comparison has "
                "power.\n",
                COINCIDE_MAX_T);
    return 0;
}
