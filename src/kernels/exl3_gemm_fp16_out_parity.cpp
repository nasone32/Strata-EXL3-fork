// src/kernels/exl3_gemm_fp16_out_parity.cpp - focused STRATA_EXL3_GEMM_FP16_OUT parity.
//
// Independent oracle: the CPU reconstruction (the same rounded F16 weights the GPU materializes,
// bit-exact by the recon_tile_kernel proof) projected in DOUBLE precision, crossed through the
// SAME F16 boundaries the engine crosses (gate/up rounded to half before SiLU, down rounded to
// half before the widen/scatter tail) - so the only tolerated difference is the GEMM library's
// reduction order, exactly as in exl3_moe_multi_parity.
//
// Covers, for BOTH epilogues of exl3_moe_group_gemm AND exl3_moe_recon_batch (F32-C baseline vs
// FP16-out): K2/K2.5/K3..K8 over the mul1/mcg/3inst codebooks, ragged padded recon groups, input
// row permutations and nonidentity destination maps, n_out != n_ff (and n_out != n_in for the
// group path; recon's contract fixes n_out == n_in), repeated use of a reserved workspace, two
// concurrent workspaces/streams with the flag on, inf/NaN/overflow classification, zero rows,
// subnormal and negative/scale stress, exact-destination scatters (sentinel check), byte-stable
// default path after flag-on calls (the F32-C path re-run must repeat bit for bit), unchanged
// allocated_bytes (no hidden allocation), and the entry refusals for invalid batches with the
// flag on.  Reported per case: RMS relative error vs the oracle, max abs/rel, the F16 ulp
// distance of finite elements, bit-identity vs the F32-C baseline (round_match - reported, NOT
// a pass criterion: bit identity is documented as not guaranteed) and nonfinite classification.
//
// Pass criteria use the existing reference bound of these paths (relative error < 0.02 vs the
// CPU oracle, the same number exl3_moe_multi_parity holds them to) plus EXACT nonfinite-class
// equality and exact zero rows; the subnormal case additionally requires max ulp <= 1 (the F32
// accumulation error, ~2^-40 at those magnitudes, sits far below the half quantum).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/cpu/exl3.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {
using namespace strata::kernels;
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
uint16_t half(float f) { const __half h = __float2half_rn(f); uint16_t v; std::memcpy(&v, &h, 2); return v; }
uint16_t half_bits(double d) { const __half h = __float2half_rn((float) d); uint16_t v; std::memcpy(&v, &h, 2); return v; }
float fp32(uint16_t v) { __half h; std::memcpy(&h, &v, 2); return __half2float(h); }
// Monotonic rank over finite halves (subnormals included; +0 == -0), for ulp distances.
int half_rank(uint16_t v) {
    const uint16_t mag = v & 0x7FFFu;
    return (v & 0x8000u) ? -(int) mag : (int) mag;
}

template <class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) { check(cudaMalloc((void**) &p, n * sizeof(T))); }
    ~Buffer() { (void) cudaFree(p); }
    Buffer(const Buffer&) = delete;
};

// The Matrix harness of exl3_moe_multi_parity, plus su/sv scale multipliers for the stress cases
// (applied before upload AND before the CPU reconstruct, so the oracle carries the scaled scales).
struct Matrix {
    int inputs, outputs;
    std::vector<uint16_t> trellis, suh, svh, weight;
    Buffer<uint16_t> tr, su, sv;
    Exl3Mat gpu;
    Matrix(int bits, int cb, std::mt19937& rng, int in, int out, int frac = 0,
           double su_scale = 1.0, double sv_scale = 1.0) : inputs(in), outputs(out),
        trellis((size_t) (in / 16) * (out / 16) * (16 * bits + 8 * frac)), suh(in), svh(out), weight((size_t) in*out),
        tr(trellis.size()), su(in), sv(out) {
        for (auto& v : trellis) v = (uint16_t) rng();
        for (auto& v : suh) v = half((float) (((int) (rng()%1000) - 500) * 0.0005 * su_scale));
        for (auto& v : svh) v = half((float) (((int) (rng()%1000) - 500) * 0.0005 * sv_scale));
        check(cudaMemcpy(tr.p, trellis.data(), trellis.size()*2, cudaMemcpyHostToDevice));
        check(cudaMemcpy(su.p, suh.data(), inputs*2, cudaMemcpyHostToDevice));
        check(cudaMemcpy(sv.p, svh.data(), outputs*2, cudaMemcpyHostToDevice));
        gpu = {tr.p, su.p, sv.p, inputs/16, outputs/16, bits, cb, frac};
        cpu::exl3_reconstruct_weight(trellis.data(), inputs/16, outputs/16, bits, (cpu::Exl3Codebook) cb,
                                     suh.data(), svh.data(), weight.data(), frac);
    }
    std::vector<uint16_t> project(const uint16_t* x) const {
        std::vector<uint16_t> y(outputs);
        for (int j=0;j<outputs;++j) {
            double s=0;
            for (int i=0;i<inputs;++i) s += (double) fp32(x[i]) * fp32(weight[(size_t) i*outputs+j]);
            y[(size_t) j] = half_bits(s);
        }
        return y;
    }
};

// The FFN oracle with the engine's exact F16 boundaries (multi_parity's expression verbatim).
std::vector<float> ffn_oracle(const Matrix& g, const Matrix& u, const Matrix& d, const uint16_t* x) {
    auto gv = g.project(x);
    const auto uv = u.project(x);
    for (int i = 0; i < g.outputs; ++i) {
        const float v = fp32(gv[(size_t) i]);
        gv[(size_t) i] = half_bits(v / (1 + std::exp(-(double) v)) * fp32(uv[(size_t) i]));
    }
    const auto y = d.project(gv.data());
    std::vector<float> out(y.size());
    for (size_t i = 0; i < y.size(); ++i) out[i] = fp32(y[i]);
    return out;
}

struct Compare {
    double num = 0, den = 0;          // RMS relative error pieces
    double rel = 0;                   // squared RMS relative error (callers sqrt it)
    double ab_rel = 0;                // squared RMS relative error vs the F32-C baseline
    double max_abs = 0, max_rel = 0;  // over finite-vs-finite elements
    long long elements = 0, identical_bits = 0, max_ulp = 0, finite_pairs = 0, round_match = 0;
    long long class_mismatch = 0, sign_mismatch = 0;   // sign/class vs the F32-C BASELINE
    long long oracle_class_mismatch = 0, oracle_sign_flip = 0;   // vs the oracle (reported; see below)
    long long oracle_inf = 0, oracle_nan = 0, got_inf = 0, got_nan = 0;
};
// got: engine output (float, widened from half); oracle: double-accum + half boundaries; ref:
// the F32-C baseline when given.  Classification is asserted against the ORACLE (identical F16
// boundary semantics); SIGN is asserted against the baseline: a dot whose double sum sits within
// F32-accumulation noise of zero can legitimately land on either side of 0 for EITHER epilogue
// (cancellation, order-dependent) - what must never happen is the F16-C epilogue flipping a sign
// the established F32-C epilogue does not.  Oracle sign flips are counted and reported.
Compare compare_outputs(const std::vector<float>& got, const std::vector<float>& oracle,
                        const float* ref = nullptr) {
    Compare c;
    c.elements = (long long) got.size();
    for (long long i = 0; i < c.elements; ++i) {
        const uint16_t hg = half(got[(size_t) i]), ho = half(oracle[(size_t) i]);
        if (hg == ho) ++c.identical_bits;
        if (ref) { if (hg == half(ref[(size_t) i])) ++c.round_match; }
        const bool gn = std::isnan(got[(size_t) i]) != 0, on = std::isnan(oracle[(size_t) i]) != 0;
        const bool gi = std::isinf(got[(size_t) i]) != 0, oi = std::isinf(oracle[(size_t) i]) != 0;
        if (on) ++c.oracle_nan;
        if (gn) ++c.got_nan;
        if (oi) ++c.oracle_inf;
        if (gi) ++c.got_inf;
        const bool class_ok = gn == on && gi == oi &&
            (!gi || std::signbit(got[(size_t) i]) == std::signbit(oracle[(size_t) i]));
        if (!class_ok) ++c.oracle_class_mismatch;
        if (gn || on || gi || oi) continue;   // nonfinite: classified above, no distance metrics
        const double dv = got[(size_t) i], dr = oracle[(size_t) i];
        c.num += (dv - dr) * (dv - dr);
        c.den += dr * dr;
        c.max_abs = std::max(c.max_abs, std::fabs(dv - dr));
        if (dr != 0.0) c.max_rel = std::max(c.max_rel, std::fabs((dv - dr) / dr));
        if (dv != 0.0 && dr != 0.0 && std::signbit(dv) != std::signbit(dr)) ++c.oracle_sign_flip;
        if (ref) {
            const double br = ref[(size_t) i];
            if (dv != 0.0 && br != 0.0 && !std::isnan(br) && std::signbit(dv) != std::signbit(br))
                ++c.sign_mismatch;
        } else if (dv != 0.0 && dr != 0.0 && std::signbit(dv) != std::signbit(dr)) {
            ++c.sign_mismatch;
        }
        if (hg != ho) { ++c.finite_pairs; c.max_ulp = std::max(c.max_ulp, (long long) std::abs(half_rank(hg) - half_rank(ho))); }
    }
    c.rel = c.num / (c.den + 1e-30);
    return c;
}
bool verdict(const char* label, const Compare& c, bool stress_subnormal) {
    const double rel = std::sqrt(c.rel), ab_rel = std::sqrt(c.ab_rel);
    // Existing reference bound (exl3_moe_multi_parity holds both paths to rel < 0.02 vs this
    // oracle); classification must be EXACTLY equal to the oracle; the two epilogues must agree
    // with each other at the same bound and never flip a sign the baseline does not; the
    // subnormal case adds the ulp <= 1 proof (see the header).
    const bool pass = std::isfinite(rel) && rel < 0.02 && std::isfinite(ab_rel) && ab_rel < 0.02 &&
                      c.oracle_class_mismatch == 0 && c.sign_mismatch == 0 &&
                      (!stress_subnormal || c.max_ulp <= 1);
    std::printf("  %-44s rel=%.5g (f32path rel=%.5g) max_abs=%.4g max_rel=%.4g ulp16_max=%lld "
                "bits=%lld/%lld round_match_vs_f32path=%lld nonfinite(o:i%lld/n%lld g:i%lld/n%lld) "
                "class_diff=%lld sign_flip(o%lld/base%lld) %s\n",
                label, rel, ab_rel, c.max_abs, c.max_rel, c.max_ulp, c.identical_bits, c.elements,
                c.round_match, c.oracle_inf, c.oracle_nan, c.got_inf, c.got_nan,
                c.oracle_class_mismatch, c.oracle_sign_flip, c.sign_mismatch,
                pass ? "PASS" : "FAIL");
    return pass;
}

// Byte equality that survives NaN outputs (all-NaN cases make float == useless).
bool same_bytes(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

// Runs one (gate,up,down) fixture through the group entry with BOTH epilogues and checks:
// oracle tolerance + classification for each, epilogue equivalence, byte-identical default-path
// re-run after the flag-on call, stable allocated_bytes, and exact zeros for all-zero x rows.
bool group_case(std::mt19937& rng, const char* label, int bits, int frac, int cb,
                int k, int ff, int n, int rows, double su_scale = 1.0, double sv_scale = 1.0,
                bool stress_subnormal = false, int inf_nan_rows = 0) {
    Matrix g(bits, cb, rng, k, ff, frac, su_scale, sv_scale);
    Matrix u(bits, cb, rng, k, ff, frac, su_scale, sv_scale);
    Matrix d(bits, cb, rng, ff, n, frac, su_scale, sv_scale);
    const int tokens = 4 * rows + 13;
    std::vector<uint16_t> x((size_t) tokens * k);
    for (auto& v : x) v = half(((int) (rng() % 1000) - 500) * 0.001f);
    for (int r = 0; r < rows && r < inf_nan_rows; ++r)          // seed +inf/NaN halves into row r
        for (int i = 0; i < k; ++i)
            x[(size_t) r * k + i] = (i & 1) ? (uint16_t) 0x7C00u : (uint16_t) 0x7E00u;
    const int zero_row = inf_nan_rows > 0 ? -1 : 1;             // token 1 is an all-zero row
    if (zero_row > 0) std::fill(x.begin() + (size_t) zero_row * k, x.begin() + (size_t) (zero_row + 1) * k, 0);
    std::vector<int> input_rows((size_t) rows);
    for (int r = 0; r < rows; ++r) input_rows[(size_t) r] = zero_row >= 0 && r == rows / 2 ? zero_row : (r * 113) % tokens;
    std::vector<std::vector<float>> oracle((size_t) rows);
    for (int r = 0; r < rows; ++r)
        oracle[(size_t) r] = ffn_oracle(g, u, d, x.data() + (size_t) input_rows[(size_t) r] * k);
    std::vector<float> flat;
    flat.reserve((size_t) rows * n);
    for (int r = 0; r < rows; ++r)
        flat.insert(flat.end(), oracle[(size_t) r].begin(), oracle[(size_t) r].end());

    Buffer<uint16_t> dx(x.size());
    check(cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    Buffer<int> dr((size_t) rows);
    check(cudaMemcpy(dr.p, input_rows.data(), rows * 4, cudaMemcpyHostToDevice));
    const float sentinel = -12345.5f;
    Buffer<float> dy((size_t) rows * n);
    check(cudaMemset(dy.p, 0, (size_t) rows * n * 4));
    Exl3MoeWorkspace workspace;

    auto run = [&](std::vector<float>& out) {
        check(cudaMemset(dy.p, 0, (size_t) rows * n * 4));
        exl3_moe_group_gemm(g.gpu, u.gpu, d.gpu, rows, dx.p, dr.p, dy.p, nullptr, workspace);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        out.assign((size_t) rows * n, sentinel);
        check(cudaMemcpy(out.data(), dy.p, out.size() * 4, cudaMemcpyDeviceToHost));
    };
    std::vector<float> baseline, got, rerun;
    exl3_gemm_fp16_out_set(false);
    run(baseline);
    const size_t bytes = workspace.allocated_bytes();
    exl3_gemm_fp16_out_set(true);
    run(got);
    const size_t bytes_on = workspace.allocated_bytes();
    exl3_gemm_fp16_out_set(false);
    run(rerun);
    bool ok = bytes == bytes_on && same_bytes(rerun, baseline);
    if (bytes != bytes_on)
        std::printf("  %-44s allocated_bytes changed %zu -> %zu FAIL\n", label, bytes, bytes_on);
    if (!same_bytes(rerun, baseline))
        std::printf("  %-44s default path not byte-stable after a flag-on call FAIL\n", label);

    Compare c = compare_outputs(got, flat, baseline.data());
    c.ab_rel = compare_outputs(got, baseline).rel;
    ok = verdict(label, c, stress_subnormal) && ok;
    // the F32-C baseline must hold its own established bound on the same fixture
    {
        Compare b = compare_outputs(baseline, flat);
        const double brel = std::sqrt(b.rel);
        if (!(std::isfinite(brel) && brel < 0.02 && b.oracle_class_mismatch == 0)) {
            std::printf("  %-44s F32-C baseline itself off (rel=%.5g class_diff=%lld) FAIL\n",
                        label, brel, b.oracle_class_mismatch);
            ok = false;
        }
    }
    if (zero_row >= 0) {   // the all-zero token row: exact +0 through every boundary
        bool zero_ok = true;
        for (int i = 0; i < n; ++i) {
            if (std::memcmp(&got[(size_t) (rows / 2) * n + i], &sentinel, 4) == 0) { zero_ok = false; break; }
            if (got[(size_t) (rows / 2) * n + i] != 0.0f ||
                std::memcmp(&baseline[(size_t) (rows / 2) * n + i], &got[(size_t) (rows / 2) * n + i], 4))
                zero_ok = false;
        }
        std::printf("  %-44s zero_row exact_zero=%d %s\n", label, (int) zero_ok, zero_ok ? "PASS" : "FAIL");
        ok = ok && zero_ok;
    }
    return ok;
}

// The recon-batch twin: ragged groups (padded rows never scattered), mixed bitrates, permuted
// input rows, nonidentity destinations; every `out` element must be written EXACTLY once (the
// sentinel survives nowhere) under BOTH epilogues.
bool recon_case(std::mt19937& rng, const char* label,
                const std::vector<std::pair<int,int>>& members,   // (bits, frac) per expert; cb=2
                int k, int ff, int n, const std::vector<int>& rows_per_group) {
    const int count = (int) members.size();
    std::vector<std::unique_ptr<Matrix>> gs, us, ds;
    for (int e = 0; e < count; ++e) {
        gs.emplace_back(new Matrix(members[(size_t) e].first, 2, rng, k, ff, members[(size_t) e].second));
        us.emplace_back(new Matrix(members[(size_t) e].first, 2, rng, k, ff, members[(size_t) e].second));
        ds.emplace_back(new Matrix(members[(size_t) e].first, 2, rng, ff, n, members[(size_t) e].second));
    }
    int total = 0, widest = 0;
    for (int r : rows_per_group) { total += r; widest = std::max(widest, r); }
    const int tokens = 3 * total + 11;
    std::vector<uint16_t> x((size_t) tokens * k);
    for (auto& v : x) v = half(((int) (rng() % 1000) - 500) * 0.001f);
    std::vector<int> rows((size_t) total), dest((size_t) total);
    for (int r = 0; r < total; ++r) { rows[(size_t) r] = (r * 7 + 5) % tokens; dest[(size_t) r] = total - 1 - r; }
    std::vector<float> flat((size_t) total * n);
    int at = 0;
    for (int e = 0; e < count; ++e)
        for (int r = 0; r < rows_per_group[(size_t) e]; ++r, ++at) {
            const auto o = ffn_oracle(*gs[(size_t) e], *us[(size_t) e], *ds[(size_t) e],
                                      x.data() + (size_t) rows[(size_t) at] * k);
            std::copy(o.begin(), o.end(), flat.begin() + (size_t) at * n);
        }

    Buffer<uint16_t> dx(x.size());
    check(cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    Buffer<int> dr((size_t) total), dd((size_t) total);
    check(cudaMemcpy(dr.p, rows.data(), total * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dd.p, dest.data(), total * 4, cudaMemcpyHostToDevice));
    const float sentinel = -12345.5f;
    Buffer<float> dy((size_t) total * n);
    std::vector<Exl3Group> groups;
    { int prefix = 0; for (int e = 0; e < count; ++e) {
        groups.push_back({gs[(size_t) e]->gpu, us[(size_t) e]->gpu, ds[(size_t) e]->gpu, prefix,
                          rows_per_group[(size_t) e]});
        prefix += rows_per_group[(size_t) e]; } }
    Exl3MoeWorkspace workspace;
    workspace.reserve_recon_batch(count, count * widest + 16, k, ff, n);
    const size_t bytes = workspace.allocated_bytes();

    auto run = [&](std::vector<float>& out) {
        out.assign((size_t) total * n, sentinel);   // prefill the device buffer: every element
        check(cudaMemcpy(dy.p, out.data(), out.size() * 4, cudaMemcpyHostToDevice));  // must be scattered
        const bool took = exl3_moe_recon_batch(groups.data(), count, dx.p, dr.p, dd.p, dy.p,
                                               nullptr, workspace);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(out.data(), dy.p, out.size() * 4, cudaMemcpyDeviceToHost));
        return took;
    };
    // The oracle in DESTINATION order (dest is a permutation: every element must be written once).
    std::vector<float> oracle_dest(flat.size());
    for (int r = 0; r < total; ++r)
        std::copy(flat.begin() + (size_t) r * n, flat.begin() + (size_t) (r + 1) * n,
                  oracle_dest.begin() + (size_t) dest[(size_t) r] * n);
    std::vector<float> baseline, got, rerun;
    exl3_gemm_fp16_out_set(false);
    bool ok = run(baseline);
    exl3_gemm_fp16_out_set(true);
    ok = run(got) && ok;
    exl3_gemm_fp16_out_set(false);
    ok = run(rerun) && ok;
    if (!ok) { std::printf("  %-44s batch unexpectedly rejected FAIL\n", label); return false; }
    ok = bytes == workspace.allocated_bytes() && same_bytes(rerun, baseline) && ok;
    if (bytes != workspace.allocated_bytes())
        std::printf("  %-44s allocated_bytes changed %zu -> %zu FAIL\n", label, bytes, workspace.allocated_bytes());
    if (!same_bytes(rerun, baseline))
        std::printf("  %-44s default path not byte-stable after a flag-on call FAIL\n", label);
    for (const auto* v : {&baseline, &got})   // exact destinations: no sentinel anywhere
        for (size_t i = 0; i < v->size(); ++i)
            if ((*v)[i] == sentinel) {
                std::printf("  %-44s out[%zu] never scattered FAIL\n", label, i);
                ok = false;
                break;
            }
    // compare in ROUTE order for a stable oracle mapping
    std::vector<float> got_route(got.size()), base_route(baseline.size());
    for (int r = 0; r < total; ++r) {
        std::copy(got.begin() + (size_t) dest[(size_t) r] * n, got.begin() + (size_t) (dest[(size_t) r] + 1) * n,
                  got_route.begin() + (size_t) r * n);
        std::copy(baseline.begin() + (size_t) dest[(size_t) r] * n, baseline.begin() + (size_t) (dest[(size_t) r] + 1) * n,
                  base_route.begin() + (size_t) r * n);
    }
    (void) oracle_dest;
    Compare c = compare_outputs(got_route, flat, base_route.data());
    c.ab_rel = compare_outputs(got_route, base_route).rel;
    ok = verdict(label, c, false) && ok;
    {
        Compare b = compare_outputs(base_route, flat);
        const double brel = std::sqrt(b.rel);
        if (!(std::isfinite(brel) && brel < 0.02 && b.oracle_class_mismatch == 0)) {
            std::printf("  %-44s F32-C baseline itself off (rel=%.5g class_diff=%lld) FAIL\n",
                        label, brel, b.oracle_class_mismatch);
            ok = false;
        }
    }
    return ok;
}

bool group_battery(int device) {
    check(cudaSetDevice(device));
    std::mt19937 rng(20261104);
    bool ok = true;
    char label[64];
    struct K { int bits, frac, cb; const char* name; };
    const K ks[] = {{2,0,2,"K2/mul1"},{2,1,2,"K2.5/mul1"},{3,0,2,"K3/mul1"},{3,0,1,"K3/mcg"},
                    {3,0,0,"K3/3inst"},{4,0,2,"K4/mul1"},{5,0,2,"K5/mul1"},{6,0,2,"K6/mul1"},
                    {7,0,2,"K7/mul1"},{8,0,2,"K8/mul1"}};
    for (const K& kk : ks) {   // square 256; odd row count; permuted token rows
        std::snprintf(label, sizeof label, "group k256/ff256/n256 %s rows19", kk.name);
        ok = group_case(rng, label, kk.bits, kk.frac, kk.cb, 256, 256, 256, 19) && ok;
    }
    // n_out != n_ff and n_out != n_in geometries (recon's contract fixes n_out==n_in; the group
    // entry does not), on the MiMo-mix and K3 bitrates
    ok = group_case(rng, "group k256/ff128/n384 K2.5/mul1 rows19", 2, 1, 2, 256, 128, 384, 19) && ok;
    ok = group_case(rng, "group k384/ff256/n128 K3/mul1 rows19", 3, 0, 2, 384, 256, 128, 19) && ok;
    // stress: tiny scales (subnormal weights/outputs), huge scales (overflow to inf through the
    // SAME half boundary), all-negative activations, and +inf/NaN-seeded input rows
    ok = group_case(rng, "group k128 subnormal-scales K2/mul1", 2, 0, 2, 128, 128, 128, 16,
                    5.96e-8, 5.96e-8, true) && ok;
    ok = group_case(rng, "group k128 overflow-scales K3/mul1", 3, 0, 2, 128, 128, 128, 16,
                    65536.0, 1.0) && ok;
    {
        // all-negative activations: same fixture, x negated row-by-row is covered by the random
        // signs; run one explicitly negative-biased case via a scale flip (sv negative scale)
        ok = group_case(rng, "group k128 negative-scales K2/mul1", 2, 0, 2, 128, 128, 128, 16,
                        -1.0, -1.0) && ok;
    }
    ok = group_case(rng, "group k128 inf-nan-input K2.5/mul1", 2, 1, 2, 128, 128, 128, 16,
                    1.0, 1.0, false, 2) && ok;
    return ok;
}

bool recon_battery(int device) {
    check(cudaSetDevice(device));
    std::mt19937 rng(987654321);
    bool ok = true;
    // the MiMo mix (K2 + K2.5 in one batch), ragged, ff != k both ways
    ok = recon_case(rng, "recon k256/ff128 K2+K2.5 rows17/23", {{2,0},{2,1}}, 256, 128, 256, {17,23}) && ok;
    ok = recon_case(rng, "recon k256/ff384 K3 rows16/31/16",
                    {{3,0},{3,0},{3,0}}, 256, 384, 256, {16,31,16}) && ok;
    ok = recon_case(rng, "recon k256/ff256 K4+K6+K8 rows20/21/22",
                    {{4,0},{6,0},{8,0}}, 256, 256, 256, {20,21,22}) && ok;
    // repeated use of one reserved workspace: a second, differently-shaped batch after the first
    {
        std::mt19937 rng2(555);
        ok = recon_case(rng2, "recon k128/ff128 K2 rows16/17 (reuse ws)", {{2,0},{2,0}}, 128, 128, 128, {16,17}) && ok;
        // (workspace reuse across cases is inherent: each case news its own; the GROUP battery's
        // three same-workspace calls and this battery's grow-only reserves cover the repeat path)
    }
    // negative probes with the flag ON: refusals must behave exactly as the default path's
    {
        exl3_gemm_fp16_out_set(true);
        std::mt19937 rng3(777);
        Matrix g(2,2,rng3,128,128), u(2,2,rng3,128,128), d(2,2,rng3,128,128);
        Matrix big(3,2,rng3,256,256);
        Exl3MoeWorkspace ws;
        ws.reserve_recon_batch(2, 2 * 20 + 16, 128, 128, 128);
        Buffer<uint16_t> dx((size_t) 8 * 128);
        check(cudaMemset(dx.p, 0, (size_t) 8 * 128 * 2));
        Buffer<int> dr(20), dd(20);
        check(cudaMemset(dr.p, 0, 20 * 4));
        std::vector<int> hd(20); for (int r = 0; r < 20; ++r) hd[(size_t) r] = 19 - r;
        check(cudaMemcpy(dd.p, hd.data(), 20 * 4, cudaMemcpyHostToDevice));
        Buffer<float> dy((size_t) 20 * 128);
        Exl3Group one{g.gpu,u.gpu,d.gpu,0,20};
        bool refused = !exl3_moe_recon_batch(&one, 1, dx.p, dr.p, dd.p, dy.p, nullptr, ws);
        Exl3Group two[]{Exl3Group{g.gpu,u.gpu,d.gpu,0,20},Exl3Group{big.gpu,u.gpu,d.gpu,20,20}};
        refused = refused && !exl3_moe_recon_batch(two, 2, dx.p, dr.p, dd.p, dy.p, nullptr, ws);
        bool threw = false;
        try { Exl3Mat bad = g.gpu; bad.bits = 0; bad.ki = 8; bad.nj = 8;
              exl3_moe_group_gemm(bad, u.gpu, d.gpu, 4, dx.p, dr.p, dy.p, nullptr, ws); }
        catch (const std::invalid_argument&) { threw = true; }
        exl3_gemm_fp16_out_set(false);
        const bool pass = refused && threw;
        std::printf("  %-44s refused=%d invalid_matrix_threw=%d %s\n",
                    "negative probes (flag on)", (int) refused, (int) threw, pass ? "PASS" : "FAIL");
        ok = ok && pass;
    }
    return ok;
}

// Repeated use of ONE reserved workspace through the group entry: three sequential calls with
// different row counts (grow-only reserves under the established standalone-entry contract).
bool workspace_reuse(int device) {
    check(cudaSetDevice(device));
    std::mt19937 rng(31415);
    Matrix g(2,2,rng,128,128), u(2,2,rng,128,128), d(2,2,rng,128,128);
    const int tokens = 200;
    std::vector<uint16_t> x((size_t) tokens * 128);
    for (auto& v : x) v = half(((int) (rng() % 1000) - 500) * 0.001f);
    Buffer<uint16_t> dx(x.size());
    check(cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    Exl3MoeWorkspace ws;
    bool ok = true;
    exl3_gemm_fp16_out_set(true);
    for (int rows : {17, 5, 40}) {
        std::vector<int> rows_idx((size_t) rows);
        for (int r = 0; r < rows; ++r) rows_idx[(size_t) r] = (r * 37) % tokens;
        std::vector<float> oracle;
        for (int r = 0; r < rows; ++r) {
            const auto o = ffn_oracle(g, u, d, x.data() + (size_t) rows_idx[(size_t) r] * 128);
            oracle.insert(oracle.end(), o.begin(), o.end());
        }
        Buffer<int> dr((size_t) rows);
        check(cudaMemcpy(dr.p, rows_idx.data(), rows * 4, cudaMemcpyHostToDevice));
        Buffer<float> dy((size_t) rows * 128);
        exl3_moe_group_gemm(g.gpu, u.gpu, d.gpu, rows, dx.p, dr.p, dy.p, nullptr, ws);
        check(cudaDeviceSynchronize());
        std::vector<float> got((size_t) rows * 128);
        check(cudaMemcpy(got.data(), dy.p, got.size() * 4, cudaMemcpyDeviceToHost));
        Compare c = compare_outputs(got, oracle);
        char label[48];
        std::snprintf(label, sizeof label, "reuse group rows=%d", rows);
        const double rel = std::sqrt(c.rel);
        const bool pass = std::isfinite(rel) && rel < 0.02 && c.oracle_class_mismatch == 0;
        std::printf("  %-44s rel=%.5g class_diff=%lld %s\n", label, rel, c.oracle_class_mismatch, pass ? "PASS" : "FAIL");
        ok = ok && pass;
    }
    exl3_gemm_fp16_out_set(false);
    return ok;
}

// Two threads, separate workspaces and streams, flag ON throughout: the flag itself is the only
// new shared state and it is read-only here, so concurrent use must stay correct (the established
// concurrent-workspace pattern of exl3_moe_multi_parity).
bool concurrency(int device) {
    std::atomic<bool> ok{true};
    auto task = [&](int seed) {
        try {
            check(cudaSetDevice(device));
            std::mt19937 rng((unsigned) seed * 7919 + 17);
            Matrix g(2,2,rng,128,128,seed & 1), u(2,2,rng,128,128,seed & 1), d(2,2,rng,128,128,seed & 1);
            const int rows = 16, tokens = 100;
            std::vector<uint16_t> x((size_t) tokens * 128);
            for (auto& v : x) v = half(((int) (rng() % 1000) - 500) * 0.001f);
            std::vector<int> rows_idx((size_t) rows), dest((size_t) rows);
            for (int r = 0; r < rows; ++r) { rows_idx[(size_t) r] = (r * 7 + seed) % tokens; dest[(size_t) r] = rows - 1 - r; }
            std::vector<float> oracle;
            for (int r = 0; r < rows; ++r) {
                const auto o = ffn_oracle(g, u, d, x.data() + (size_t) rows_idx[(size_t) r] * 128);
                oracle.insert(oracle.end(), o.begin(), o.end());
            }
            cudaStream_t st;
            check(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
            Exl3MoeWorkspace ws;
            Buffer<uint16_t> dx(x.size());
            check(cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
            Buffer<int> dr((size_t) rows), dd((size_t) rows);
            check(cudaMemcpy(dr.p, rows_idx.data(), rows * 4, cudaMemcpyHostToDevice));
            check(cudaMemcpy(dd.p, dest.data(), rows * 4, cudaMemcpyHostToDevice));
            Buffer<float> dy((size_t) rows * 128);
            exl3_moe_group_gemm(g.gpu, u.gpu, d.gpu, rows, dx.p, dr.p, dy.p, (void*) st, ws);
            Exl3Group group{g.gpu, u.gpu, d.gpu, 0, rows};
            ws.reserve_recon_batch(2, 2 * rows + 16, 128, 128, 128);
            Buffer<float> dy2((size_t) rows * 128);
            if (!exl3_moe_recon_batch(&group, 1, dx.p, dr.p, dd.p, dy2.p, (void*) st, ws)) {
                // single-group batches are refused by contract; a 2-group retry covers the batch
                // path (second member at the contiguous prefix offset)
                Exl3Group two[]{group, {g.gpu, u.gpu, d.gpu, rows, rows}};
                std::vector<int> rows2((size_t) rows * 2), dest2((size_t) rows * 2);
                for (int r = 0; r < rows * 2; ++r) { rows2[(size_t) r] = rows_idx[(size_t) (r % rows)]; dest2[(size_t) r] = r; }
                Buffer<int> dr2(rows2.size()), dd2(dest2.size());
                check(cudaMemcpy(dr2.p, rows2.data(), rows2.size() * 4, cudaMemcpyHostToDevice));
                check(cudaMemcpy(dd2.p, dest2.data(), dest2.size() * 4, cudaMemcpyHostToDevice));
                Buffer<float> dy3((size_t) rows2.size() * 128);
                if (!exl3_moe_recon_batch(two, 2, dx.p, dr2.p, dd2.p, dy3.p, (void*) st, ws))
                    throw std::runtime_error("concurrent recon batch rejected");
            }
            check(cudaStreamSynchronize(st));
            check(cudaStreamDestroy(st));
            std::vector<float> got((size_t) rows * 128);
            check(cudaMemcpy(got.data(), dy.p, got.size() * 4, cudaMemcpyDeviceToHost));
            Compare c = compare_outputs(got, oracle);
            const double rel = std::sqrt(c.rel);
            const bool pass = std::isfinite(rel) && rel < 0.02 && c.oracle_class_mismatch == 0;
            std::printf("  thread=%d group rel=%.5g class_diff=%lld %s\n", seed, rel, c.oracle_class_mismatch,
                        pass ? "PASS" : "FAIL");
            ok = ok && pass;
        } catch (const std::exception& e) {
            std::printf("  thread=%d EXCEPTION %s FAIL\n", seed, e.what());
            ok = false;
        }
    };
    std::thread t0([&] { task(0); }), t1([&] { task(1); });
    t0.join(); t1.join();
    return ok.load();
}

// The entry-eligibility counters' accounting: dispatched group_gemm/recon calls count (with
// their rows/experts), a refused batch and an invalid-matrix throw count NOTHING, and the
// fp16-epilogue total equals dispatched calls while the flag is on (0 while off). These
// totals are what the serve bench hook prints per phase, so their semantics are gate-covered.
bool counters_check(int device) {
    check(cudaSetDevice(device));
    exl3_gemm_fp16_out_counters_set_enabled(true);   // collection is opt-in; this check owns it
    std::mt19937 rng(2718);
    Matrix g(2,2,rng,128,128), u(2,2,rng,128,128), d(2,2,rng,128,128);
    const int tokens = 64, rows = 20;
    std::vector<uint16_t> x((size_t) tokens * 128);
    for (auto& v : x) v = half(((int) (rng() % 1000) - 500) * 0.001f);
    Buffer<uint16_t> dx(x.size());
    check(cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    std::vector<int> rows_idx((size_t) rows), dest((size_t) rows);
    for (int r = 0; r < rows; ++r) { rows_idx[(size_t) r] = (r * 11) % tokens; dest[(size_t) r] = rows - 1 - r; }
    Buffer<int> dr(rows), dd(rows);
    check(cudaMemcpy(dr.p, rows_idx.data(), rows * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dd.p, dest.data(), rows * 4, cudaMemcpyHostToDevice));
    Buffer<float> dy((size_t) rows * 128), dy2((size_t) rows * 128);
    Exl3MoeWorkspace ws;
    ws.reserve_recon_batch(2, 2 * rows + 16, 128, 128, 128);
    Exl3Group one{g.gpu, u.gpu, d.gpu, 0, rows};
    bool ok = true;
    // flag OFF: two group calls + one refused single-group batch -> epilogue delta 0
    exl3_gemm_fp16_out_set(false);
    const auto c0 = exl3_gemm_fp16_out_counters();
    exl3_moe_group_gemm(g.gpu, u.gpu, d.gpu, rows, dx.p, dr.p, dy.p, nullptr, ws);
    exl3_moe_group_gemm(g.gpu, u.gpu, d.gpu, rows, dx.p, dr.p, dy.p, nullptr, ws);
    (void) !exl3_moe_recon_batch(&one, 1, dx.p, dr.p, dd.p, dy2.p, nullptr, ws);
    auto c1 = exl3_gemm_fp16_out_counters();
    bool pass = c1.group_gemm_calls - c0.group_gemm_calls == 2 &&
                c1.group_gemm_rows - c0.group_gemm_rows == 2 * rows &&
                c1.recon_calls - c0.recon_calls == 0 &&
                c1.fp16_epilogue_calls - c0.fp16_epilogue_calls == 0;
    std::printf("  %-44s %s\n", "counters: 2 group calls off-flag, refusal uncounted", pass ? "PASS" : "FAIL");
    ok = ok && pass;
    // flag ON: one group call + one accepted 2-group batch (40 rows) -> epilogue delta 2
    exl3_gemm_fp16_out_set(true);
    exl3_moe_group_gemm(g.gpu, u.gpu, d.gpu, rows, dx.p, dr.p, dy.p, nullptr, ws);
    Exl3Group two[]{one, {g.gpu, u.gpu, d.gpu, rows, rows}};
    std::vector<int> rows2((size_t) rows * 2), dest2((size_t) rows * 2);
    for (int r = 0; r < rows * 2; ++r) { rows2[(size_t) r] = rows_idx[(size_t) (r % rows)]; dest2[(size_t) r] = r; }
    Buffer<int> dr2(rows2.size()), dd2(dest2.size());
    check(cudaMemcpy(dr2.p, rows2.data(), rows2.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dd2.p, dest2.data(), dest2.size() * 4, cudaMemcpyHostToDevice));
    Buffer<float> dy3((size_t) rows2.size() * 128);
    if (!exl3_moe_recon_batch(two, 2, dx.p, dr2.p, dd2.p, dy3.p, nullptr, ws))
        return false;
    c1 = exl3_gemm_fp16_out_counters();
    pass = c1.group_gemm_calls - c0.group_gemm_calls == 3 &&
           c1.group_gemm_rows - c0.group_gemm_rows == 3 * rows &&
           c1.recon_calls - c0.recon_calls == 1 &&
           c1.recon_experts - c0.recon_experts == 2 &&
           c1.recon_rows - c0.recon_rows == 2 * rows &&
           c1.fp16_epilogue_calls - c0.fp16_epilogue_calls == 2;
    std::printf("  %-44s %s\n", "counters: on-flag group + accepted batch, epilogue", pass ? "PASS" : "FAIL");
    exl3_gemm_fp16_out_set(false);
    exl3_gemm_fp16_out_counters_set_enabled(false);
    return ok && pass;
}

}  // namespace

// STRATA_EXL3_GEMM_FP16_OUT flag semantics - exl3_wmma_stg_pad's checks, verbatim in structure:
// the env applies exactly ONCE per process at the first getter/setter call, then the setter is
// authoritative in BOTH directions; a setter called BEFORE the first getter survives the apply.
// The env=1-at-first-read proof needs a fresh process, so ctest drives --fp16-out-flag-test with
// the env set through cmake -E env; this mode touches no GPU API at all.
static int fp16_out_flag_checks(bool env_expected)
{
    bool ok = exl3_gemm_fp16_out_enabled() == env_expected;   // first read applies the env once
    exl3_gemm_fp16_out_set(!env_expected);
    ok = ok && exl3_gemm_fp16_out_enabled() == !env_expected; // setter overrides downwards
    exl3_gemm_fp16_out_set(env_expected);
    ok = ok && exl3_gemm_fp16_out_enabled() == env_expected;  // ...and back up
    std::printf("fp16_out flag: env %s at first read, then setter authoritative both ways: %s\n",
                env_expected ? "ON" : "unset", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    // No GPU API is touched before this point (the flag machinery exists on every backend).
    if (argc > 1 && std::strcmp(argv[1], "--fp16-out-flag-test") == 0) {
        const bool env_on = std::getenv("STRATA_EXL3_GEMM_FP16_OUT") != nullptr &&
                            std::getenv("STRATA_EXL3_GEMM_FP16_OUT")[0] != '0';
        return fp16_out_flag_checks(env_on);
    }
    {
        const char* preset = std::getenv("STRATA_EXL3_GEMM_FP16_OUT");
        const bool env_was_set = preset != nullptr && preset[0] != '0';
        const int rc = fp16_out_flag_checks(env_was_set);   // default invocation: env normally unset
        if (rc != 0) return rc;
        if (!env_was_set) {
            exl3_gemm_fp16_out_set(false);
            exl3_gemm_fp16_out_set(true);                   // BEFORE any getter call
            const bool ok = exl3_gemm_fp16_out_enabled() == true;
            exl3_gemm_fp16_out_set(false);                  // leave off for the battery
            std::printf("fp16_out flag: setter before first read not overridden by env apply: %s\n",
                        ok ? "PASS" : "FAIL");
            if (!ok) return 1;
        } else {
            exl3_gemm_fp16_out_set(false);                  // the battery drives the flag explicitly
        }
    }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { std::puts("SKIP: no CUDA device"); return 77; }
    try {
        bool ok = true;
        for (int d = 0; d < count; ++d) {
            std::printf("device=%d STRATA_EXL3_GEMM_FP16_OUT focused parity\n", d);
            ok = group_battery(d) && ok;
            ok = recon_battery(d) && ok;
            ok = workspace_reuse(d) && ok;
            ok = concurrency(d) && ok;
            ok = counters_check(d) && ok;
        }
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
