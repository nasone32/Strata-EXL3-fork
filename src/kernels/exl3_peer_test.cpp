// src/kernels/exl3_peer_test.cpp - Exl3Peer's test battery (ctest exl3_peer_test, SKIP 77).
//
// Oracle: expected rows are computed ROUTE BY ROUTE through the independent single-expert entry
// (exl3_moe_rows, k=1) on GPU0 from device-resident planes - never through Exl3Peer - so a wrong
// route->row mapping cannot pass on both sides.  Cross-path comparison uses the EXISTING
// wave_check compare_rel rule PER COMPUTED ROW (5e-3 norm vs that row's own scale - the GEMV
// accumulates with atomics and the two call shapes differ in the last ULPs); byte-identity is
// NOT claimed across call shapes.  Sentinels on non-peer rows are compared EXACTLY; oracle and
// actual computed rows are each checked finite + non-constant first.  The launch/finish wall
// split and busy() are printed as OBSERVATIONAL enqueue-only evidence, never as pass/fail.
// Sections: synthetic (K3/K4, two geometries, layers 1/5 and 2/3), real model (explicitly all
// of layers 0/7/23), contract matrix, open() pre-check matrix, allocation/copy fault matrix
// (STRATA_EXL3_PEER_FAULT).
#include "strata/kernels/exl3_peer.hpp"
#include "strata/kernels/exl3_experts.hpp"
#include "strata/kernels/exl3_peer_adapt.hpp"
#include "strata/artifact/exl3_model.hpp"
#include "strata/core/exl3_flags.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using strata::kernels::Exl3Mat;
using strata::kernels::Exl3Peer;

namespace {

struct On {
    int prev = 0;
    explicit On(int dev) { cudaGetDevice(&prev); if (prev != dev) cudaSetDevice(dev); }
    ~On() { int cur = 0; cudaGetDevice(&cur); if (cur != prev) cudaSetDevice(prev); }
};

double ms_now() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

std::vector<uint8_t> pattern(size_t n, uint32_t seed) {
    std::vector<uint8_t> out(n);
    uint32_t s = seed & 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) { s = 1103515245u * s + 12345u; out[i] = (uint8_t) (s >> 16); }
    return out;
}
std::vector<uint8_t> scales(size_t count, uint32_t seed) {   // finite f16 in the calibrated 0x2800 band
    const std::vector<uint8_t> raw = pattern(count * 2, seed);
    std::vector<uint8_t> out(count * 2);
    for (size_t i = 0; i < count; ++i) {
        const uint16_t w = (uint16_t) (((uint16_t) raw[2 * i] | ((uint16_t) raw[2 * i + 1] << 8)) & 0x83FFu | 0x2800u);
        out[2 * i] = (uint8_t) (w & 0xFF);
        out[2 * i + 1] = (uint8_t) (w >> 8);
    }
    return out;
}

struct HostExpert {
    std::vector<uint8_t> buf[9];
    Exl3Mat g, u, d;
    Exl3Peer::Entry entry(int64_t layer, int32_t expert) const {
        Exl3Peer::Entry e;
        e.layer = layer;
        e.expert = expert;
        e.g = g;
        e.u = u;
        e.d = d;
        return e;
    }
};

HostExpert make_expert(int ki, int nj, int bits, int cb, uint32_t seed, int frac = 0) {
    HostExpert h;
    auto fill = [&](int slot, int k, int n, uint32_t sd) {
        h.buf[slot + 0] = pattern((size_t) k * (size_t) n * (size_t) (16 * bits + 8 * frac) * 2, sd);
        h.buf[slot + 1] = scales((size_t) k * 16, sd + 1);
        h.buf[slot + 2] = scales((size_t) n * 16, sd + 2);
    };
    fill(0, ki, nj, seed);
    fill(3, ki, nj, seed + 7000);
    fill(6, nj, ki, seed + 14000);
    auto set = [&](Exl3Mat& m, int slot, int k, int n) {
        m.trellis = (const uint16_t*) h.buf[slot + 0].data();
        m.suh = (const uint16_t*) h.buf[slot + 1].data();
        m.svh = (const uint16_t*) h.buf[slot + 2].data();
        m.ki = k;
        m.nj = n;
        m.bits = bits;
        m.frac = frac;
        m.cb = cb;
    };
    set(h.g, 0, ki, nj);
    set(h.u, 3, ki, nj);
    set(h.d, 6, nj, ki);
    return h;
}

// per row: finite AND non-constant (a row of zeros or NaNs must never pass a byte compare)
bool row_sane(const float* p, size_t n) {
    if (n == 0) return false;
    size_t finite = 0;
    bool varied = false;
    for (size_t i = 0; i < n; ++i) { if (std::isfinite(p[i])) ++finite; if (p[i] != p[0]) varied = true; }
    return finite == n && varied;
}

// The EXISTING wave_check rule (src/kernels/exl3_wave_check.cpp compare_rel, tol 5e-3), applied
// PER COMPUTED ROW and normalized by THAT row's scale - a whole-output normalization could let a
// strong row mask a wrong low-amplitude one, and sentinels must not dilute it.
bool row_rel(const float* a, const float* b, size_t n, double* norm_out) {
    double scale = 0, worst = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { *norm_out = 1e300; return false; }
        scale = std::max(scale, (double) std::fabs(b[i]));
        const double d = std::fabs((double) a[i] - (double) b[i]);
        if (d > worst) worst = d;
    }
    *norm_out = worst / std::max(1e-6, scale);
    return *norm_out <= 5e-3;
}

int g_fail = 0;
void check(bool pass, const char* what) {
    std::printf("%s: %s\n", what, pass ? "PASS" : "FAIL");
    if (!pass) ++g_fail;
}

// device-anchored: the tier allocates on device 1, so leak evidence must read THAT device
size_t free_vram() {
    int prev = 0;
    (void) cudaGetDevice(&prev);
    size_t f = 0, t = 0;
    if (cudaSetDevice(1) == cudaSuccess) (void) cudaMemGetInfo(&f, &t);
    (void) cudaSetDevice(prev);
    return f;
}

// The independent oracle on GPU0: its OWN device copies of the planes (the kernels need device
// pointers) and per-route exl3_moe_rows (k=1) calls - never Exl3Peer's code path.
struct Oracle {
    int n_embd = 0;
    float* d_x32 = nullptr;
    uint16_t *d_x16 = nullptr, *d_row = nullptr;
    cudaStream_t stream = nullptr;
    std::unique_ptr<strata::kernels::Exl3MoeWorkspace> ws;
    std::vector<uint16_t> h_row;
    std::vector<uint8_t*> planes;      // 9 device planes per registered expert
    std::vector<Exl3Mat> g, u, d;      // device-pointer mats, one per registered expert

    bool init(int n_embd_, int max_tokens, std::string& err) {
        n_embd = n_embd_;
        h_row.assign((size_t) n_embd, 0);
        On on(0);
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaMalloc((void**) &d_x32, (size_t) max_tokens * n_embd * 4) != cudaSuccess ||
            cudaMalloc((void**) &d_x16, (size_t) max_tokens * n_embd * 2) != cudaSuccess ||
            cudaMalloc((void**) &d_row, (size_t) n_embd * 2) != cudaSuccess) {
            err = "oracle alloc";
            return false;
        }
        ws = std::make_unique<strata::kernels::Exl3MoeWorkspace>();
        ws->reserve(1, n_embd, 1024, n_embd, true);
        return true;
    }
    ~Oracle() {
        On on(0);
        ws.reset();
        for (uint8_t* p : planes) (void) cudaFree(p);
        if (d_x32) (void) cudaFree(d_x32);
        if (d_x16) (void) cudaFree(d_x16);
        if (d_row) (void) cudaFree(d_row);
        if (stream) (void) cudaStreamDestroy(stream);
    }
    int add(const Exl3Mat& hg, const Exl3Mat& hu, const Exl3Mat& hd, std::string& err) {
        On on(0);
        const Exl3Mat host[3] = {hg, hu, hd};
        Exl3Mat devm[3] = {};
        for (int p = 0; p < 3; ++p) {
            const size_t tr = (size_t) host[p].ki * (size_t) host[p].nj *
                              (size_t) (16 * host[p].bits + 8 * host[p].frac) * 2;
            const size_t sizes[3] = {tr, (size_t) host[p].ki * 16 * 2, (size_t) host[p].nj * 16 * 2};
            const void* src[3] = {host[p].trellis, host[p].suh, host[p].svh};
            void* dst[3] = {};
            for (int q = 0; q < 3; ++q) {
                // single ownership: published into `planes` the moment it allocates; the
                // destructor frees each exactly once (no local frees on the failure path)
                if (cudaMalloc(&dst[q], sizes[q]) != cudaSuccess) {
                    err = "oracle plane alloc";
                    return -1;
                }
                planes.push_back((uint8_t*) dst[q]);
                if (cudaMemcpy(dst[q], src[q], sizes[q], cudaMemcpyHostToDevice) != cudaSuccess) {
                    err = "oracle plane upload";
                    return -1;
                }
            }
            devm[p].trellis = (const uint16_t*) dst[0];
            devm[p].suh = (const uint16_t*) dst[1];
            devm[p].svh = (const uint16_t*) dst[2];
            devm[p].ki = host[p].ki;
            devm[p].nj = host[p].nj;
            devm[p].bits = host[p].bits;
            devm[p].frac = host[p].frac;
            devm[p].cb = host[p].cb;
        }
        g.push_back(devm[0]);
        u.push_back(devm[1]);
        d.push_back(devm[2]);
        return (int) g.size() - 1;
    }
    // expected[route * n_embd] = ffn(x[route/k], expert ridx[route]) for kind==2 routes
    bool expected(const std::vector<float>& x, const std::vector<int>& ridx, const std::vector<int32_t>& kind,
                  int tokens, int k, std::vector<float>& out, std::string& err) {
        On on(0);
        out.assign((size_t) tokens * k * n_embd, 3.25f);
        if (cudaMemcpy(d_x32, x.data(), (size_t) tokens * n_embd * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = std::string("oracle x upload: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        strata::kernels::f32_to_f16_bulk(d_x32, d_x16, (int64_t) tokens * n_embd, stream);
        if (cudaGetLastError() != cudaSuccess) { err = "oracle convert"; return false; }
        const int routes = tokens * k;
        for (int r = 0; r < routes; ++r) {
            if (kind[(size_t) r] != 2) continue;
            const int tok = r / k;
            const int e = ridx[(size_t) r];
            strata::kernels::exl3_moe_rows(&g[(size_t) e], &u[(size_t) e], &d[(size_t) e], nullptr, 1,
                                           d_x16 + (size_t) tok * n_embd, d_row, stream, ws.get());
            if (cudaGetLastError() != cudaSuccess) { err = "oracle rows"; return false; }
            if (cudaStreamSynchronize(stream) != cudaSuccess ||   // order against the non-blocking stream
                cudaMemcpy(h_row.data(), d_row, (size_t) n_embd * 2, cudaMemcpyDeviceToHost) != cudaSuccess) {
                err = "oracle readback";
                return false;
            }
            for (int i = 0; i < n_embd; ++i)
                out[(size_t) r * n_embd + i] = strata::kernels::f32_from_f16(h_row[(size_t) i]);
        }
        return true;
    }
};

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 2) {
        std::printf("SKIP: needs two visible devices (have %d)\n", ndev);
        return 77;
    }
    std::string err;
    std::printf("exl3_peer_test: 2 devices; independent oracle = per-route exl3_moe_rows on GPU0 "
                "(device planes, k=1)\n");

    // Exact SAME-FUSE comparison isolates grouping/scatter from the existing
    // independent oracle's different reduction path. Exercise chunk tails,
    // duplicate token rows, mixed route kinds, and unsupported-codebook fallback.
    if (strata::kernels::exl3_fused_multirow_supported()) {
        const char* old = std::getenv("STRATA_EXL3_PEER_MULTIROW");
        const std::string saved = old ? old : "";
        auto set_width = [](const char* value) {
#if defined(_WIN32)
            _putenv_s("STRATA_EXL3_PEER_MULTIROW", value);
#else
            setenv("STRATA_EXL3_PEER_MULTIROW", value, 1);
#endif
        };
        std::vector<HostExpert> host;
        std::vector<Exl3Peer::Entry> subset;
        for (int i=0;i<5;++i) {
            host.push_back(make_expert(160,40,i==4 ? 3 : 1+i*2,i==4 ? 0 : 2,821u+i*197u));
            subset.push_back(host.back().entry(0,100+i));
        }
        Exl3Peer control;
        set_width("0");
        check(control.open(1,2560,subset,8,10,err),"multirow exact: indexed setup");
        for (const char* width : {"2","3","4"}) {
            set_width(width);
            Exl3Peer candidate;
            check(candidate.open(1,2560,subset,8,10,err),"multirow exact: grouped setup");
            for (int tokens : {1,2,3,4,7,8}) for (int mode : {0,1,2}) {
                const int k=10, routes=tokens*k;
                std::vector<float> x((size_t) tokens*2560), want((size_t) routes*2560,3.25f), got=want;
                for (size_t i=0;i<x.size();++i) x[i]=((int)((i*113+tokens*19)%2000)-1000)*0.001f;
                std::vector<int32_t> ids(routes),kind(routes);
                for (int r=0;r<routes;++r) {
                    ids[r]=100+(r*3+r/10)%(mode==2 ? 5 : 4);
                    kind[r]=mode==1 && r%3==0 ? 0 : 2;
                }
                bool ok=true;
                for (int repeat=0;repeat<3 && ok;++repeat) {
                    ok=control.launch(0,x.data(),ids.data(),tokens,k,kind.data(),err) && control.finish(want.data(),err) &&
                       candidate.launch(0,x.data(),ids.data(),tokens,k,kind.data(),err) && candidate.finish(got.data(),err) &&
                       std::memcmp(want.data(),got.data(),got.size()*sizeof(float))==0;
                }
                if (!ok) std::printf("multirow exact width=%s tokens=%d mode=%d: %s\n",width,tokens,mode,err.c_str());
                check(ok,"multirow exact: repeated indexed/grouped bytes and non-peer sentinels");
            }
        }
        set_width(saved.c_str());
    }

    // ============ frac (half-integer K, the MiMo K2/K2.5 mix) through the grouped gate ============
    // On the parity fixtures (exl3_moe_multi_parity's fused mixed-frac battery) the grouped RDNA
    // twin decodes the TESTED frac configurations - K2/K2.5 mul1 tiles - with fused output
    // byte-identical to the indexed entry at the same call shape.  That is a tested-fixture fact,
    // not a universal guarantee over frac widths or geometries.  Pinned HERE, per arm of the
    // opt-in STRATA_EXL3_PEER_MULTIROW_FRAC (engine default OFF; restored 2026-11-03 at the
    // user's request after a negative-result removal - the end-to-end evidence was mixed, see
    // notes/MIMO-PEER-FRAC-GROUPED-2026-11-03.md):
    //   flag OFF: a launch holding any frac expert stays on the indexed entry (dispatch counters
    //             do not move) and its bytes equal the width-0 control exactly;
    //   flag ON : the same launch dispatches GROUPED and STILL equals the control bytes exactly -
    //             grouping changes transport, never the row arithmetic;
    //   both    : integer-K repeats still group, non-mul1 falls back, and an independent
    //             per-route oracle agrees within this file's cross-shape tolerance.
    if (strata::kernels::exl3_fused_multirow_supported()) {
        const char* old_w = std::getenv("STRATA_EXL3_PEER_MULTIROW");
        const char* old_f = std::getenv("STRATA_EXL3_PEER_MULTIROW_FRAC");
        const std::string saved_w = old_w ? old_w : "", saved_f = old_f ? old_f : "";
        auto set_env = [](const char* name, const char* value) {
#if defined(_WIN32)
            _putenv_s(name, value);
#else
            if (value) setenv(name, value, 1); else unsetenv(name);
#endif
        };
        // experts 0..3: K2/K2.5 alternating (bits 2, frac 0/1, mul1); expert 4: K2 INTEGER
        // non-mul1 (the codebook-fallback fixture - a non-mul1 FRAC tile cannot run anywhere
        // under RDNA=1, the indexed entry rejects it too, so integer K is the runnable probe).
        std::vector<HostExpert> host;
        std::vector<Exl3Peer::Entry> subset;
        for (int i = 0; i < 5; ++i) {
            const int frac = (i < 4) ? (i & 1) : 0;
            const int bits = 2;
            const int cb = i == 4 ? 0 : 2;
            host.push_back(make_expert(160, 40, bits, cb, 911u + 131u * (uint32_t) i, frac));
            subset.push_back(host.back().entry(0, 200 + i));
        }
        Exl3Peer control;
        set_env("STRATA_EXL3_PEER_MULTIROW", "0");
        set_env("STRATA_EXL3_PEER_MULTIROW_FRAC", nullptr);
        check(control.open(1, 2560, subset, 8, 10, err), "multirow frac: indexed control setup");
        // STRATA_EXL3_NATIVE_GROUPED=1 (default off) puts the grouped arm on the native
        // core (packed mul1 decode + fdot2) whose per-row accumulation/partitioning is a
        // DIFFERENT arithmetic from the indexed arm's - byte equality is then reported,
        // not enforced (the same self-disable the indexed byte check applies under
        // STRATA_EXL3_RDNA_NATIVE=1); the flag-off default stays byte-exact.
        const char* ng_env = std::getenv("STRATA_EXL3_NATIVE_GROUPED");
        const char* rn_env = std::getenv("STRATA_EXL3_RDNA_NATIVE");
        const bool ng_on = (ng_env && ng_env[0] != '\0' && ng_env[0] != '0') ||
                           (rn_env && rn_env[0] == '1');
        set_env("STRATA_EXL3_PEER_MULTIROW", "2");
        for (int arm = 0; arm < 2; ++arm) {   // 0: flag off (indexed expected), 1: flag on (grouped)
            const char* const arm_name = arm ? "grouped" : "indexed";
            set_env("STRATA_EXL3_PEER_MULTIROW_FRAC", arm ? "1" : nullptr);
            Exl3Peer candidate;
            check(candidate.open(1, 2560, subset, 8, 10, err), "multirow frac: candidate setup");
            for (int tokens : {2, 4, 8}) {
                const int k = 10, routes = tokens * k;
                std::vector<float> x((size_t) tokens * 2560), want((size_t) routes * 2560, 3.25f), got = want;
                for (size_t i = 0; i < x.size(); ++i) x[i] = ((int) ((i * 113 + tokens * 19) % 2000) - 1000) * 0.001f;
                for (int mode = 0; mode < 2; ++mode) {   // 0: frac experts repeat (DFlash-window shape), 1: mixed kinds
                    std::vector<int32_t> ids(routes), kind(routes);
                    for (int r = 0; r < routes; ++r) {
                        ids[r] = 200 + (mode == 0 ? (r * 3 + r / 10) % 4 : (r % 5 == 0 ? 4 : (r * 3) % 4));
                        kind[r] = mode == 1 && r % 3 == 0 ? 0 : 2;
                    }
                    auto outputs_agree = [&](const std::vector<float>& a, const std::vector<float>& b) {
                        if (a.size() != b.size()) return false;
                        if (!ng_on) return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
                        double num = 0, den = 0;   // native core: closeness, not byte identity
                        for (size_t i = 0; i < a.size(); ++i) {
                            num += (double) (a[i] - b[i]) * (a[i] - b[i]);
                            den += (double) b[i] * b[i];
                        }
                        return std::isfinite(num) && std::sqrt(num / (den + 1e-30)) < 0.02;
                    };
                    bool ok = true;
                    for (int repeat = 0; repeat < 3 && ok; ++repeat) {
                        ok = control.launch(0, x.data(), ids.data(), tokens, k, kind.data(), err) &&
                             control.finish(want.data(), err) &&
                             candidate.launch(0, x.data(), ids.data(), tokens, k, kind.data(), err) &&
                             candidate.finish(got.data(), err) &&
                             outputs_agree(want, got);
                    }
                    if (!ok) std::printf("multirow frac arm=%s tokens=%d mode=%d: %s\n", arm_name, tokens, mode, err.c_str());
                    check(ok, ng_on
                          ? "multirow frac: indexed/native-grouped outputs close on the K2/K2.5 mix"
                          : "multirow frac: indexed/grouped bytes equal on the K2/K2.5 mix");
                }
            }
            {   // dispatch classification, pinned exactly
                const int tokens = 2, k = 2, routes = tokens * k;
                std::vector<float> x((size_t) tokens * 2560, 0.05f), got((size_t) routes * 2560, 3.25f);
                std::vector<int32_t> kind(routes, 2);
                const int64_t c0 = candidate.grouped_launches();
                std::vector<int32_t> ids{200, 201, 200, 201};   // two would-be 2-row groups: K2 + K2.5
                check(candidate.launch(0, x.data(), ids.data(), tokens, k, kind.data(), err) &&
                      candidate.finish(got.data(), err), "multirow frac: classified launch");
                if (arm == 0)
                    check(candidate.grouped_launches() == c0,
                          "multirow frac: flag off keeps repeated frac launches indexed");
                else
                    check(candidate.grouped_launches() > c0,
                          "multirow frac: flag on dispatches repeated frac launches grouped");
                const int64_t ci = candidate.grouped_launches();
                std::vector<int32_t> integer{200, 200};   // INTEGER K2 repeats group in BOTH arms;
                check(candidate.launch(0, x.data(), integer.data(), 1, 2, kind.data(), err) &&   // frac is the only
                      candidate.finish(got.data(), err), "multirow frac: integer repeat launch");  // gated tile
                check(candidate.grouped_launches() == ci + 1, "multirow frac: integer repeats still group");
                const int64_t c1 = candidate.grouped_launches();
                std::vector<int32_t> wide{204, 204};   // non-mul1 INTEGER K2: grouped falls back
                check(candidate.launch(0, x.data(), wide.data(), 1, 2, kind.data(), err) &&
                      candidate.finish(got.data(), err), "multirow frac: non-mul1 launch");
                check(candidate.grouped_launches() == c1, "multirow frac: non-mul1 falls back to indexed");
            }
            {   // independent per-route oracle on a repeated frac shape (both K2 and K2.5 rows)
                Oracle oracle;
                check(oracle.init(2560, 4, err), "multirow frac: oracle init");
                for (int i = 0; i < 4; ++i)
                    check(oracle.add(host[(size_t) i].g, host[(size_t) i].u, host[(size_t) i].d, err) == i,
                          "multirow frac: oracle add");
                const int tokens = 4, k = 8, routes = tokens * k;
                std::vector<float> x((size_t) tokens * 2560), got((size_t) routes * 2560, 3.25f), ref;
                for (size_t i = 0; i < x.size(); ++i) x[i] = ((int) ((i * 71 + 5) % 2000) - 1000) * 0.001f;
                std::vector<int32_t> ids(routes), kind(routes, 2);
                std::vector<int> table(routes);
                for (int r = 0; r < routes; ++r) { ids[r] = 200 + (r % 4); table[(size_t) r] = r % 4; }
                check(candidate.launch(0, x.data(), ids.data(), tokens, k, kind.data(), err) &&
                      candidate.finish(got.data(), err), "multirow frac: oracle launch");
                check(oracle.expected(x, table, kind, tokens, k, ref, err), "multirow frac: oracle rows");
                // the oracle's per-route call shape (atomics reduction) differs from the peer's
                // Partials shape in the last ULPs - this file's documented rule is per-row 5e-3
                // tolerance for cross-shape checks; byte identity across arms was proven above
                // at the SAME call shape
                double worst_norm = 0;
                bool rel_ok = true;
                for (int r = 0; r < routes; ++r)
                    rel_ok = row_rel(got.data() + (size_t) r * 2560, ref.data() + (size_t) r * 2560,
                                     2560, &worst_norm) && rel_ok;
                std::printf("multirow frac arm=%s vs independent oracle worst row rel %.3g %s\n",
                            arm_name, worst_norm, rel_ok ? "PASS" : "FAIL");
                check(rel_ok, "multirow frac: rows match the independent per-route oracle");
            }
            candidate.close();
        }
        set_env("STRATA_EXL3_PEER_MULTIROW", saved_w.empty() ? nullptr : saved_w.c_str());
        set_env("STRATA_EXL3_PEER_MULTIROW_FRAC", saved_f.empty() ? nullptr : saved_f.c_str());
        control.close();
        std::puts("PASS: multirow frac - flag off stays indexed / flag on groups, bytes exact "
                  "both arms, oracle-equal; integer repeats still group, non-mul1 falls back");
    }

    // ============ STRATA_EXL3_PEER_NATIVE_WIDTH4 (default off): width-4 native groups ============
    // MiMo-shaped wide geometry (n_embd 4096, hidden 2048) where the float twin's width-4 LDS
    // rule (64 KiB > 40 KiB) downgraded the whole arm to indexed.  Pinned per arm:
    //   gate OFF: the startup downgrade is EXACTLY the old one - native_width4() false, no
    //             grouped dispatch, bytes equal the indexed control;
    //   gate ON : width 4 survives open only with the native stack armed, every launch
    //             re-plans (admitted K2/K2.5/K3/K4 mixes keep width 4; a K5 population, a
    //             post-open native-env flip or a non-mul1 tile degrades 4 -> 2 -> indexed
    //             per call, never sticky, refused populations byte-equal the control);
    //   refills : descriptors are re-read per call - a smaller-rate replacement keeps width
    //             4 admissible, a plane-growing replacement is refused by the slot capacity.
    if (strata::kernels::exl3_fused_multirow_supported()) {
        const char* names[6] = {"STRATA_EXL3_PEER_MULTIROW", "STRATA_EXL3_PEER_MULTIROW_FRAC",
                                "STRATA_EXL3_NATIVE_GROUPED", "STRATA_EXL3_RDNA_NATIVE",
                                "STRATA_EXL3_PEER_NATIVE_WIDTH4", "STRATA_EXL3_GEMV_SPLIT"};
        std::string saved[6];
        for (int i = 0; i < 6; ++i) { const char* v = std::getenv(names[i]); saved[i] = v ? v : ""; }
        auto set_env = [](const char* name, const char* value) {
#if defined(_WIN32)
            _putenv_s(name, value);
#else
            if (value) setenv(name, value, 1); else unsetenv(name);
#endif
        };
        auto restore = [&]() {
            for (int i = 0; i < 6; ++i) set_env(names[i], saved[i].empty() ? nullptr : saved[i].c_str());
        };
        // ---- host-only plan unit: a mid-list half-integer K3.5 (bits 3, frac 1) descriptor
        // refuses the WHOLE window and the refusal zeroes every output (the audited
        // commit(false) contract; nothing is dereferenced, no GPU needed)
        {
            HostExpert k2 = make_expert(256, 128, 2, 2, 555u), k35 = make_expert(256, 128, 3, 2, 557u, 1);
            const Exl3Mat gate[2] = {k2.g, k35.g}, up[2] = {k2.u, k35.u}, down[2] = {k2.d, k35.d};
            int rates[3] = {0x7f, 0x7f, 0x7f}, parts[3] = {0x7f, 0x7f, 0x7f};
            check(!strata::kernels::exl3_native_grouped_plan(gate, up, down,
                                                             2, 4, true, true, rates, parts) &&
                  (rates[0] | rates[1] | rates[2] | parts[0] | parts[1] | parts[2]) == 0,
                  "native width-4: mid-rate K3.5 refuses and zeroes all plan outputs");
        }
        // experts: 300/302 K2, 301/303 K2.5 (frac), 304 K2 non-mul1 (indexed probe),
        // 305 K5 integer mul1 (plan-refuses probe), 306 K3, 307 K4 (mixed rates).
        std::vector<HostExpert> host;
        std::vector<Exl3Peer::Entry> subset;
        const int n_embd = 4096;
        for (int i = 0; i < 8; ++i) {
            const int bits = i == 5 ? 5 : (i == 6 ? 3 : i == 7 ? 4 : 2);
            const int frac = (i == 1 || i == 3) ? 1 : 0;
            const int cb = i == 4 ? 0 : 2;
            host.push_back(make_expert(256, 128, bits, cb, 4177u + 601u * (uint32_t) i, frac));
            subset.push_back(host.back().entry(0, 300 + i));
        }
        Exl3Peer control;
        set_env("STRATA_EXL3_PEER_MULTIROW", "0");
        check(control.open(1, n_embd, subset, 8, 10, err), "native width-4: indexed control setup");

        // ---- arm A: the new gate default-off keeps the OLD startup downgrade
        {
            set_env("STRATA_EXL3_PEER_MULTIROW", "4");
            set_env("STRATA_EXL3_PEER_MULTIROW_FRAC", "1");
            set_env("STRATA_EXL3_NATIVE_GROUPED", "1");
            set_env("STRATA_EXL3_RDNA_NATIVE", "1");
            set_env("STRATA_EXL3_PEER_NATIVE_WIDTH4", nullptr);
            Exl3Peer cand;
            check(cand.open(1, n_embd, subset, 8, 10, err), "native width-4: gate-off open");
            check(!cand.native_width4(), "native width-4: gate off -> old float-gate downgrade state");
            const int tokens = 4, k = 8, routes = tokens * k;
            std::vector<float> x((size_t) tokens * n_embd), want((size_t) routes * n_embd, 3.25f), got = want;
            for (size_t i = 0; i < x.size(); ++i) x[i] = ((int) ((i * 113 + tokens * 19) % 2000) - 1000) * 0.001f;
            std::vector<int32_t> ids(routes), kind(routes, 2);
            for (int r = 0; r < routes; ++r) ids[r] = 300 + (r % 4);
            bool ok = true;
            for (int repeat = 0; repeat < 2 && ok; ++repeat)
                ok = control.launch(0, x.data(), ids.data(), tokens, k, kind.data(), err) &&
                     control.finish(want.data(), err) &&
                     cand.launch(0, x.data(), ids.data(), tokens, k, kind.data(), err) &&
                     cand.finish(got.data(), err) &&
                     std::memcmp(want.data(), got.data(), got.size() * sizeof(float)) == 0;
            check(ok && cand.grouped_launches() == 0,
                  "native width-4: gate off stays indexed and byte-equal (default unchanged)");
            cand.close();
        }

        // ---- arm B: gate on - per-call width 4, degradation 4 -> 2 -> indexed, refills
        {
            set_env("STRATA_EXL3_PEER_MULTIROW", "4");
            set_env("STRATA_EXL3_PEER_MULTIROW_FRAC", "1");
            set_env("STRATA_EXL3_NATIVE_GROUPED", "1");
            set_env("STRATA_EXL3_RDNA_NATIVE", "1");
            set_env("STRATA_EXL3_PEER_NATIVE_WIDTH4", "1");
            Exl3Peer cand;
            check(cand.open(1, n_embd, subset, 8, 10, err), "native width-4: gate-on open");
            check(cand.native_width4(), "native width-4: armed open keeps width 4");
            // Match the fallback's arithmetic and grouping. Indexed K2 uses native dot2,
            // whereas an env-off grouped call uses the established FP32 grouped kernel.
            Exl3Peer fallback_control;
            set_env("STRATA_EXL3_PEER_MULTIROW", "2");
            check(fallback_control.open(1, n_embd, subset, 8, 10, err),
                  "native width-4: width-2 fallback control setup");
            set_env("STRATA_EXL3_PEER_MULTIROW", "4");
            const int tokens = 4, k = 8, routes = tokens * k;
            std::vector<float> x((size_t) tokens * n_embd), want((size_t) routes * n_embd, 3.25f), got = want;
            for (size_t i = 0; i < x.size(); ++i) x[i] = ((int) ((i * 113 + tokens * 19) % 2000) - 1000) * 0.001f;
            std::vector<int32_t> ids(routes), kind(routes, 2);
            for (int r = 0; r < routes; ++r) ids[r] = 300 + (r % 4);   // K2/K2.5/K2/K2.5 repeats
            bool ok = control.launch(0, x.data(), ids.data(), tokens, k, kind.data(), err) &&
                      control.finish(want.data(), err) &&
                      cand.launch(0, x.data(), ids.data(), tokens, k, kind.data(), err) &&
                      cand.finish(got.data(), err);
            const int64_t n0 = cand.native_width4_calls();
            check(ok && n0 > 0, "native width-4: admissible mix dispatches at width 4");
            {   // the native core's row arithmetic: closeness vs the indexed control (the
                // established convention under the native flags - bytes are NOT asserted)
                double worst = 0;
                bool close = true;
                for (int r = 0; r < routes; ++r)
                    close = row_rel(got.data() + (size_t) r * n_embd,
                                    want.data() + (size_t) r * n_embd, n_embd, &worst) && close;
                check(close,
                      "native width-4: native-core rows close to the indexed control");
            }
            {   // mixed K2/K2.5/K3/K4 rates keep width 4 (per-group rate mixing is admitted)
                std::vector<int32_t> mixed(routes);
                for (int r = 0; r < routes; ++r) mixed[r] = 300 + (r % 8 == 6 ? 6 : r % 8 == 7 ? 7 : r % 2);
                const int64_t m0 = cand.native_width4_calls();
                check(cand.launch(0, x.data(), mixed.data(), tokens, k, kind.data(), err) &&
                      cand.finish(got.data(), err) && cand.native_width4_calls() == m0 + 1,
                      "native width-4: mixed K2/K2.5/K3/K4 groups keep width 4");
            }
            {   // group-count sweep: the parts plan's occupancy floor moves with the group
                // count (ceil(288 / ((outputs/128) * groups))), so pin width 4 at BOTH ends -
                // one expert repeated 32x (8 groups of 4) and six admissible experts cycling
                // (12 groups of mixed 2-4 rows) - and under explicit STRATA_EXL3_GEMV_SPLIT
                // caps 1 and 2 (ps=1: 4096x4x2B = 32 KiB fits; ps=2: 16 KiB fits)
                const int64_t split0 = cand.native_width4_calls();
                std::vector<int32_t> single(routes, 300);   // 1 expert x 32 rows -> 8 groups
                bool sweep = cand.launch(0, x.data(), single.data(), tokens, k, kind.data(), err) &&
                             cand.finish(got.data(), err);
                std::vector<int32_t> many(routes);          // 6 admissible experts -> 12 groups
                const int32_t admissible[6] = {300, 301, 302, 303, 306, 307};
                for (int r = 0; r < routes; ++r) many[r] = admissible[r % 6];
                sweep = sweep && cand.launch(0, x.data(), many.data(), tokens, k, kind.data(), err) &&
                        cand.finish(got.data(), err);
                check(sweep && cand.native_width4_calls() == split0 + 2,
                      "native width-4: 8-group and 12-group windows both admit at width 4");
                for (const char* split : {"1", "2"}) {
                    set_env("STRATA_EXL3_GEMV_SPLIT", split);
                    const int64_t s0 = cand.native_width4_calls();
                    bool cap_ok = cand.launch(0, x.data(), single.data(), tokens, k, kind.data(), err) &&
                                  cand.finish(got.data(), err) &&
                                  cand.native_width4_calls() == s0 + 1;
                    std::printf("native width-4: GEMV_SPLIT=%s %s\n", split, cap_ok ? "PASS" : "FAIL");
                    check(cap_ok, "native width-4: explicit GEMV_SPLIT cap keeps width 4");
                }
                set_env("STRATA_EXL3_GEMV_SPLIT", saved[5].empty() ? nullptr : saved[5].c_str());
            }
            {   // K5 population: the plan refuses -> width-2 re-chunk; the dispatcher refuses
                // native there too, so the established twin computes and bytes equal control
                std::vector<int32_t> wide(routes, 305);
                std::fill(want.begin(), want.end(), 3.25f), std::fill(got.begin(), got.end(), 3.25f);
                const int64_t w0 = cand.native_width4_calls(), g0 = cand.grouped_launches();
                bool k5 = control.launch(0, x.data(), wide.data(), tokens, k, kind.data(), err) &&
                          control.finish(want.data(), err) &&
                          cand.launch(0, x.data(), wide.data(), tokens, k, kind.data(), err) &&
                          cand.finish(got.data(), err);
                check(k5 && cand.native_width4_calls() == w0 && cand.grouped_launches() == g0 + 1 &&
                      std::memcmp(want.data(), got.data(), got.size() * sizeof(float)) == 0,
                      "native width-4: unsupported rate re-chunks at width 2, byte-equal, not sticky");
            }
            {   // non-mul1 tile: the existing per-route eligibility keeps it indexed
                std::vector<int32_t> cb0(routes, 304);
                const int64_t g1 = cand.grouped_launches();
                check(cand.launch(0, x.data(), cb0.data(), tokens, k, kind.data(), err) &&
                      cand.finish(got.data(), err) && cand.grouped_launches() == g1,
                      "native width-4: non-mul1 tile still falls back indexed");
            }
            {   // native env flipped OFF after open: width-2 established twin, bytes equal;
                // flipped back ON: width 4 resumes - per-call, never sticky
                std::vector<int32_t> again(routes);
                for (int r = 0; r < routes; ++r) again[r] = 300 + (r % 4);
                set_env("STRATA_EXL3_NATIVE_GROUPED", "0");
                std::fill(want.begin(), want.end(), 3.25f), std::fill(got.begin(), got.end(), 3.25f);
                const int64_t f0 = cand.native_width4_calls(), g2 = cand.grouped_launches();
                bool flip = fallback_control.launch(0, x.data(), again.data(), tokens, k, kind.data(), err) &&
                             fallback_control.finish(want.data(), err) &&
                            cand.launch(0, x.data(), again.data(), tokens, k, kind.data(), err) &&
                            cand.finish(got.data(), err);
                check(flip && cand.native_width4_calls() == f0 && cand.grouped_launches() == g2 + 1 &&
                      std::memcmp(want.data(), got.data(), got.size() * sizeof(float)) == 0,
                      "native width-4: env off after open degrades to width 2, byte-equal");
                set_env("STRATA_EXL3_NATIVE_GROUPED", "1");
                check(cand.launch(0, x.data(), again.data(), tokens, k, kind.data(), err) &&
                      cand.finish(got.data(), err) && cand.native_width4_calls() == f0 + 1,
                      "native width-4: env back on resumes width 4 (never sticky)");
            }
            {   // DESCRIPTOR-IDENTITY probes (the AAAABBBBCCCC bug class): three experts with
                // distinct weights, NON-UNIFORM rows per expert (4/3/5), through both refusal
                // triggers - the plan refusal (a 4-row K5 group mixed in) and the env flip -
                // byte-equal to the indexed control, plus the positive width-4 admit.  Wrong
                // descriptors (a stale group's mats after the re-chunk) would compute the
                // wrong experts' rows and break the memcmp.  tokens=4 x k=3 -> 12 routes.  Runs BEFORE
                // the refill block: 306 is still the K3 expert here (the refill below swaps it).
                const int tokens3 = 4, k3 = 3, routes3 = tokens3 * k3;
                std::vector<float> x3((size_t) tokens3 * n_embd), w3((size_t) routes3 * n_embd, 3.25f),
                    g3 = w3;
                for (size_t i = 0; i < x3.size(); ++i) x3[i] = ((int) ((i * 71 + 3) % 2000) - 1000) * 0.001f;
                std::vector<int32_t> kind3(routes3, 2);
                std::vector<int32_t> nonuniform{300, 300, 300, 300, 301, 301, 301, 306, 306, 306, 306, 306};
                std::vector<int32_t> shuffled = nonuniform;   // the peer sorts routes itself
                for (size_t i = 1; i < shuffled.size(); ++i)
                    if (i % 3 == 0) std::swap(shuffled[i], shuffled[i - 1]);
                for (int variant = 0; variant < 2; ++variant) {
                    const std::vector<int32_t>& idsv = variant ? shuffled : nonuniform;
                    const char* const vname = variant ? "shuffled" : "sorted";
                    const int64_t p0 = cand.native_width4_calls();
                    bool pos = cand.launch(0, x3.data(), idsv.data(), tokens3, k3, kind3.data(), err) &&
                               cand.finish(g3.data(), err);
                    char what[128];
                    std::snprintf(what, sizeof what, "native width-4: %s 4/3/5 non-uniform admits at width 4", vname);
                    check(pos && cand.native_width4_calls() == p0 + 1, what);
                    std::vector<int32_t> withk5 = idsv;   // replace one expert's rows with a
                    for (int32_t& id : withk5) if (id == 300) id = 305;   // 4-row K5 group
                    const int64_t q0 = cand.native_width4_calls();
                    bool k5mix = fallback_control.launch(0, x3.data(), withk5.data(), tokens3, k3, kind3.data(), err) &&
                                 fallback_control.finish(w3.data(), err) &&
                                 cand.launch(0, x3.data(), withk5.data(), tokens3, k3, kind3.data(), err) &&
                                 cand.finish(g3.data(), err);
                    std::snprintf(what, sizeof what, "native width-4: %s 3-expert + K5 group plan-refusal refill byte-equal", vname);
                    check(k5mix && cand.native_width4_calls() == q0 &&
                          std::memcmp(w3.data(), g3.data(), g3.size() * sizeof(float)) == 0, what);
                    set_env("STRATA_EXL3_NATIVE_GROUPED", "0");
                    std::fill(w3.begin(), w3.end(), 3.25f), std::fill(g3.begin(), g3.end(), 3.25f);
                    const int64_t e0 = cand.native_width4_calls();
                    bool envoff = fallback_control.launch(0, x3.data(), idsv.data(), tokens3, k3, kind3.data(), err) &&
                                  fallback_control.finish(w3.data(), err) &&
                                  cand.launch(0, x3.data(), idsv.data(), tokens3, k3, kind3.data(), err) &&
                                  cand.finish(g3.data(), err);
                    std::snprintf(what, sizeof what, "native width-4: %s 4/3/5 env-flip refill byte-equal", vname);
                    check(envoff && cand.native_width4_calls() == e0 &&
                          std::memcmp(w3.data(), g3.data(), g3.size() * sizeof(float)) == 0, what);
                    set_env("STRATA_EXL3_NATIVE_GROUPED", "1");
                }
            }
            {   // Refill after the descriptor-identity probes, while expert 306 still exists.
                HostExpert k25 = make_expert(256, 128, 2, 2, 9973u, 1);
                HostExpert k5 = make_expert(256, 128, 5, 2, 9977u);
                Exl3Peer::Replacement grow{}, shrink{};
                grow.out = 306; grow.in = k5.entry(0, 310);
                shrink.out = 306; shrink.in = k25.entry(0, 309);
                check(!cand.can_replace(grow), "native width-4: plane-growing refill refused (capacity)");
                check(cand.enable_adaptation(2, 0, err) && cand.can_replace(shrink),
                      "native width-4: smaller-rate refill admissible");
                std::vector<Exl3Peer::Replacement> batch{shrink}, landed;
                const bool ref = cand.replace(batch, err) && cand.apply_refill(true, landed, err);
                check(ref && !landed.empty() && cand.has(0, 309) && !cand.has(0, 306),
                      "native width-4: refill lands (ownership transition)");
                std::vector<int32_t> after(routes, 309);
                const int64_t r0 = cand.native_width4_calls();
                check(cand.launch(0, x.data(), after.data(), tokens, k, kind.data(), err) &&
                      cand.finish(got.data(), err) && cand.native_width4_calls() == r0 + 1,
                      "native width-4: post-refill descriptors re-planned at width 4");
            }
            {   // independent per-route oracle on a width-4 window (this file's 5e-3 rule)
                Oracle oracle;
                check(oracle.init(n_embd, 8, err), "native width-4: oracle init");
                for (int i = 0; i < 8; ++i)
                    check(oracle.add(host[(size_t) i].g, host[(size_t) i].u, host[(size_t) i].d, err) == i,
                          "native width-4: oracle add");
                std::vector<int> table(routes);
                for (int r = 0; r < routes; ++r) { ids[r] = 300 + (r % 4); table[(size_t) r] = r % 4; }
                std::vector<float> ref;
                check(cand.launch(0, x.data(), ids.data(), tokens, k, kind.data(), err) &&
                      cand.finish(got.data(), err) && oracle.expected(x, table, kind, tokens, k, ref, err),
                      "native width-4: oracle window");
                double worst_norm = 0;
                bool rel_ok = true;
                for (int r = 0; r < routes; ++r)
                    rel_ok = row_rel(got.data() + (size_t) r * n_embd, ref.data() + (size_t) r * n_embd,
                                     n_embd, &worst_norm) && rel_ok;
                std::printf("native width-4 vs independent oracle worst row rel %.3g %s\n",
                            worst_norm, rel_ok ? "PASS" : "FAIL");
                check(rel_ok, "native width-4: rows match the independent per-route oracle");
            }
            cand.close();
        }
        control.close();
        restore();
        std::puts("PASS: native width-4 - gate off byte-identical default; gate on plans per "
                  "call (4 -> 2 -> indexed, refills re-read, never sticky)");
    }

    // ================= SYNTHETIC: K3/K4, two geometries, layers 1/5 and 2/3 =================
    const int geoms[2][2] = {{160, 40}, {96, 24}};
    const int64_t layersA[2] = {1, 5}, layersB[2] = {2, 3};
    for (int gi = 0; gi < 2; ++gi) {
        const int ki = geoms[gi][0], nj = geoms[gi][1], n_embd = ki * 16, max_tokens = 8, max_k = 10;
        const int64_t* layers = gi == 0 ? layersA : layersB;
        std::vector<HostExpert> experts;
        std::vector<Exl3Peer::Entry> subset;
        for (int i = 0; i < 8; ++i) {
            experts.push_back(make_expert(ki, nj, (i % 2) ? 4 : 3, i % 3,
                                          7919u * (uint32_t) (i + 1) + (uint32_t) gi * 31u));
            subset.push_back(experts.back().entry(layers[i < 4 ? 0 : 1], 100 + i));
        }
        Exl3Peer cand;
        if (!cand.open(1, n_embd, subset, max_tokens, max_k, err)) {
            std::printf("candidate open failed: %s\n", err.c_str());
            return 1;
        }
        Oracle oracle;
        if (!oracle.init(n_embd, max_tokens, err)) {
            std::printf("oracle init failed: %s\n", err.c_str());
            return 1;
        }
        for (int i = 0; i < 8; ++i)
            if (oracle.add(experts[(size_t) i].g, experts[(size_t) i].u, experts[(size_t) i].d, err) != i) {
                std::printf("oracle add failed: %s\n", err.c_str());
                return 1;
            }
        char what[176];
        std::snprintf(what, sizeof what, "geom %d (%d->%d) K3/K4 open/has/resident/bytes", gi, n_embd, nj * 16);
        check(cand.valid() && cand.resident() == 8 && cand.bytes() > 0 &&
              cand.has(layers[0], 100) && cand.has(layers[1], 107) &&
              !cand.has(layers[0], 108) && !cand.has(0, 100), what);

        for (int tokens : {1, 2, 4, 8}) {
            for (int k : {1, 10}) {
                for (int set = 0; set < 2; ++set) {   // 0: duplicates, all peer; 1: mixed kinds
                    const int routes = tokens * k;
                    const int64_t layer = layers[(tokens + k) % 2];
                    const int32_t base = (int32_t) (layer == layers[0] ? 100 : 104);
                    std::vector<float> x((size_t) tokens * n_embd), got((size_t) routes * n_embd, 3.25f), want;
                    std::vector<int32_t> ids((size_t) routes), kind((size_t) routes);
                    std::vector<int> ridx((size_t) routes);
                    uint32_t s = 12345u + (uint32_t) tokens * 100 + (uint32_t) k;
                    for (auto& v : x) { s = 1103515245u * s + 12345u; v = (float) ((int) (s >> 16) % 2000 - 1000) * 0.001f; }
                    for (int r = 0; r < routes; ++r) {
                        ids[(size_t) r] = base + (set == 0 ? (r / 2) : (r * 3)) % 4;
                        kind[(size_t) r] = set == 1 && (r % 3 == 2) ? (r % 2 ? 0 : -1) : 2;
                        ridx[(size_t) r] = (ids[(size_t) r] - 100) % 8;
                    }
                    if (!oracle.expected(x, ridx, kind, tokens, k, want, err)) {
                        std::printf("oracle failed: %s\n", err.c_str());
                        ++g_fail;
                        continue;
                    }
                    const bool launched = cand.launch(layer, x.data(), ids.data(), tokens, k, kind.data(), err);
                    if (!launched) std::printf("  launch err: %s\n", err.c_str());
                    const bool finished = launched && cand.finish(got.data(), err);
                    if (launched && !finished) std::printf("  finish err: %s\n", err.c_str());
                    bool rows_ok = true, want_ok = true, sentinel_ok = true, computed_seen = false;
                    double worst_norm = 0;
                    bool rel_ok = launched && finished;
                    for (int r = 0; r < routes; ++r) {
                        const float* gr = got.data() + (size_t) r * n_embd;
                        const float* wr = want.data() + (size_t) r * n_embd;
                        if (kind[(size_t) r] == 2) {
                            computed_seen = true;
                            rows_ok = rows_ok && row_sane(gr, (size_t) n_embd);
                            want_ok = want_ok && row_sane(wr, (size_t) n_embd);
                            double nr = 0;
                            rel_ok = rel_ok && row_rel(gr, wr, (size_t) n_embd, &nr);
                            worst_norm = std::max(worst_norm, nr);
                        } else {
                            for (int i = 0; i < n_embd; ++i)
                                if (gr[i] != 3.25f || wr[i] != 3.25f) sentinel_ok = false;
                        }
                    }
                    std::printf("  cross-path worst per-row norm %.3e (tol 5e-3, established atomic "
                                "accumulation)\n", worst_norm);
                    std::snprintf(what, sizeof what, "geom %d T=%d k=%d set=%d %s vs independent oracle: %s",
                                  gi, tokens, k, set, computed_seen ? "peer" : "zero-peer",
                                  (rel_ok && rows_ok && want_ok && sentinel_ok) ? "PASS" : "FAIL");
                    check(rel_ok && rows_ok && want_ok && sentinel_ok, what);
                }
            }
        }
        // ---- input lifetime: the request is copied before launch returns
        {
            const int tokens = 4, k = 10, routes = tokens * k;
            std::vector<float> x((size_t) tokens * n_embd, 0.31f), pristine = x;
            std::vector<int32_t> ids((size_t) routes), kind((size_t) routes, 2);
            std::vector<int> ridx((size_t) routes);
            for (int r = 0; r < routes; ++r) {
                ids[(size_t) r] = 100 + (r / 2) % 4;
                ridx[(size_t) r] = (ids[(size_t) r] - 100) % 8;
            }
            std::vector<float> want;
            if (!oracle.expected(pristine, ridx, kind, tokens, k, want, err)) ++g_fail;
            std::vector<float> got((size_t) routes * n_embd, 3.25f);
            check(cand.launch(layers[0], x.data(), ids.data(), tokens, k, kind.data(), err), "input-lifetime launch");
            std::fill(x.begin(), x.end(), -777.0f);
            std::fill(ids.begin(), ids.end(), 103);
            check(cand.finish(got.data(), err), "input-lifetime finish");
            bool life_ok = true;
            for (int r = 0; r < routes && life_ok; ++r) {
                double nr = 0;
                life_ok = row_rel(got.data() + (size_t) r * n_embd, want.data() + (size_t) r * n_embd,
                                  (size_t) n_embd, &nr);
            }
            check(life_ok, "inputs copied before launch returns (clobbering x/ids changes nothing)");
        }
        // ---- zero peer rows / non-resident (seam counter) / null finish
        {
            std::vector<float> x((size_t) 2 * n_embd, 0.5f), out((size_t) 8 * n_embd, 3.25f);
            std::vector<int32_t> ids(8, 100), kind(8, 0);
            const int64_t e0 = cand.enqueues();
            check(cand.launch(layers[0], x.data(), ids.data(), 2, 4, kind.data(), err) && !cand.pending() &&
                  cand.enqueues() == e0, "zero peer rows: no-op, enqueue seam NOT reached");
            check(cand.finish(out.data(), err), "zero peer rows: finish no-op");
            bool untouched = true;
            for (float v : out) untouched = untouched && v == 3.25f;
            check(untouched, "zero peer rows: out untouched");
        }
        {
            std::vector<float> x((size_t) n_embd, 0.5f), out((size_t) 10 * n_embd, 3.25f);
            std::vector<int32_t> ids(10, 100), kind(10, 2);
            ids[9] = 199;
            const int64_t e0 = cand.enqueues();
            check(!cand.launch(layers[0], x.data(), ids.data(), 1, 10, kind.data(), err) &&
                  cand.enqueues() == e0 && !cand.pending(),
                  "non-resident id: rejected, enqueue seam NOT reached");
            std::printf("  (rejection: %s)\n", err.c_str());
            bool untouched = true;
            for (float v : out) untouched = untouched && v == 3.25f;
            check(untouched, "non-resident id: out untouched");
        }
        {
            std::vector<float> x((size_t) n_embd, 0.5f), out((size_t) 10 * n_embd, 3.25f);
            std::vector<int32_t> ids(10, 100), kind(10, 2);
            check(cand.launch(layers[0], x.data(), ids.data(), 1, 10, kind.data(), err) && cand.pending(),
                  "null-finish: launch pending");
            check(!cand.finish(nullptr, err) && cand.pending(), "finish(nullptr) errors, keeps pending");
            check(cand.finish(out.data(), err) && !cand.pending(), "finish(out) then completes");
            check(!cand.finish(nullptr, err), "finish(nullptr) is ALWAYS an error (even with nothing pending)");
        }
        {
            std::vector<float> x((size_t) 64 * n_embd, 0.5f);
            std::vector<int32_t> ids(640, 100), kind(640, 2);
            const int64_t e0 = cand.enqueues();
            bool all_rejected = true;
            all_rejected = all_rejected && !cand.launch(layers[0], x.data(), ids.data(), 9, 10, kind.data(), err);
            all_rejected = all_rejected && !cand.launch(layers[0], x.data(), ids.data(), 1, 11, kind.data(), err);
            all_rejected = all_rejected && !cand.launch(layers[0], x.data(), ids.data(), -1, 10, kind.data(), err);
            all_rejected = all_rejected && !cand.launch(layers[0], x.data(), ids.data(), 1, -1, kind.data(), err);
            all_rejected = all_rejected && !cand.launch(layers[0], nullptr, ids.data(), 1, 10, kind.data(), err);
            all_rejected = all_rejected && !cand.launch(layers[0], x.data(), nullptr, 1, 10, kind.data(), err);
            all_rejected = all_rejected && !cand.launch(layers[0], x.data(), ids.data(), 1, 10, nullptr, err);
            all_rejected = all_rejected && !cand.launch(99, x.data(), ids.data(), 1, 10, kind.data(), err);
            check(all_rejected && cand.enqueues() == e0 && !cand.pending(),
                  "invalid dims/caps/nulls/layer: all rejected, enqueue seam NOT reached");
        }
        // ---- nested pending, repeat (observational walls), caller device, close-while-pending
        {
            std::vector<float> x((size_t) 8 * n_embd, 0.25f), out((size_t) 80 * n_embd, 3.25f);
            std::vector<int32_t> ids(80), kind(80, 2);
            for (int r = 0; r < 80; ++r) ids[(size_t) r] = 104 + (r / 2) % 4;
            double launch_ms = 0, finish_ms = 0;
            bool busy_seen = false;
            for (int rep = 0; rep < 20; ++rep) {
                double t0 = ms_now();
                check(cand.launch(layers[1], x.data(), ids.data(), 8, 10, kind.data(), err), "repeated launch");
                launch_ms += ms_now() - t0;
                if (rep == 0 && cand.busy()) busy_seen = true;
                if (rep == 0) {
                    const int64_t l1 = cand.launches();
                    check(!cand.launch(layers[1], x.data(), ids.data(), 8, 10, kind.data(), err) &&
                          cand.launches() == l1 && cand.pending(), "nested launch rejected, state intact");
                }
                t0 = ms_now();
                check(cand.finish(out.data(), err), "repeated finish");
                finish_ms += ms_now() - t0;
            }
            std::printf("  OBSERVATIONAL enqueue evidence: launch total %.2f ms, finish total %.2f ms, "
                        "busy() after launch: %s, enqueues=%lld\n",
                        launch_ms, finish_ms, busy_seen ? "yes" : "no", (long long) cand.enqueues());
            check(row_sane(out.data(), out.size()), "repeated rows finite+non-constant");
            bool restored = true;
            for (int caller : {0, 1}) {
                int cur = -1;
                { On on(caller);
                  std::vector<float> o2((size_t) 10 * n_embd, 3.25f);
                  (void) cand.launch(layers[1], x.data(), ids.data(), 1, 10, kind.data(), err);
                  cudaGetDevice(&cur); restored = restored && cur == caller;
                  (void) cand.finish(o2.data(), err);
                  cudaGetDevice(&cur); restored = restored && cur == caller; }
            }
            check(restored, "caller device restored after launch/finish (callers 0 and 1)");
            check(cand.launch(layers[1], x.data(), ids.data(), 8, 10, kind.data(), err) && cand.pending(),
                  "close-while-pending: launch pending");
            cand.close();
            check(!cand.valid() && !cand.pending() && cand.resident() == 0,
                  "close drains and closes with work pending");
            check(cand.open(1, n_embd, subset, max_tokens, max_k, err) && cand.valid(), "reopen after close");
        }
        // ---- open() pre-check matrix (reject + empty state + VRAM unchanged)
        {
            const size_t base_free = free_vram();
            auto expect_open_fail = [&](const std::vector<Exl3Peer::Entry>& s, int n_in, int mt, int mk,
                                        const char* what) {
                // VRAM equality is banded: the runtime allocator oscillates its cached region on
                // a ~32 MiB quantum (observed both directions); "no allocation" evidence is the
                // rejection before the resource phase plus in-band VRAM.
                const size_t band = (size_t) 32 << 20;
                Exl3Peer p;
                const bool okp = p.open(1, n_in, s, mt, mk, err);
                const size_t now_free = free_vram();
                const size_t d = now_free > base_free ? now_free - base_free : base_free - now_free;
                check(!okp && !p.valid() && p.resident() == 0 && !p.has(0, 0) && d <= band, what);
                if (okp) p.close();
            };
            {
                std::vector<Exl3Peer::Entry> bad;
                expect_open_fail(bad, n_embd, max_tokens, max_k, "open precheck: empty subset");
                bad = subset;
                bad.push_back(subset[0]);
                expect_open_fail(bad, n_embd, max_tokens, max_k, "open precheck: duplicate entry");
                bad = subset;
                bad[0].g.ki += 1;
                expect_open_fail(bad, n_embd, max_tokens, max_k, "open precheck: geometry relation");
                bad = subset;
                bad[0].g.bits = 9;
                expect_open_fail(bad, n_embd, max_tokens, max_k, "open precheck: bits out of range");
                bad = subset;
                bad[0].u.cb = 3;
                expect_open_fail(bad, n_embd, max_tokens, max_k, "open precheck: cb out of range");
                bad = subset;
                bad[0].d.nj = -1;
                expect_open_fail(bad, n_embd, max_tokens, max_k, "open precheck: negative dim");
                bad = subset;
                bad[0].d.nj = INT32_MAX / 16 + 1;
                expect_open_fail(bad, n_embd, max_tokens, max_k, "open precheck: dim overflow bound");
                bad = subset;
                bad[0].g.trellis = nullptr;
                expect_open_fail(bad, n_embd, max_tokens, max_k, "open precheck: null plane");
                expect_open_fail(subset, n_embd, 0, max_k, "open precheck: max_tokens 0");
                expect_open_fail(subset, n_embd, max_tokens, 0, "open precheck: max_k 0");
                expect_open_fail(subset, n_embd, 65536, 1, "open precheck: routes cap >65535");
                expect_open_fail(subset, n_embd + 8, max_tokens, max_k, "open precheck: n_embd mismatch");
                // coherent-but-unaligned geometry with FAKE non-null plane pointers: rejection
                // must happen before any read (a deref would crash here), allocation or enqueue.
                // Baseline re-captured HERE: this check owns "this open() allocated nothing",
                // not "nothing since the geometry block started" - the runtime pool legitimately
                // drifts when earlier sections allocate (observed: sections in between grew it,
                // failing the exact equality without any leak in open itself).
                const size_t precheck_free = free_vram();
                {
                    auto fake = [](uint16_t v) { return (const uint16_t*) (uintptr_t) v; };
                    std::vector<Exl3Peer::Entry> bad(1);
                    bad[0].layer = 1;
                    bad[0].expert = 100;
                    auto rel = [&](int ki, int nj) {
                        Exl3Mat g, u, d;
                        g.trellis = fake(0x10); g.suh = fake(0x20); g.svh = fake(0x30);
                        u = g; d = g;
                        g.ki = u.ki = ki; g.nj = u.nj = nj; d.ki = nj; d.nj = ki;
                        g.bits = u.bits = d.bits = 3; g.cb = u.cb = d.cb = 2;
                        bad[0].g = g; bad[0].u = u; bad[0].d = d;
                    };
                    rel(161, 40);   // relations hold, n_embd 2576 valid, ki % 8 != 0
                    Exl3Peer p;
                    check(!p.open(1, 2576, bad, max_tokens, max_k, err) && !p.valid() &&
                          free_vram() == precheck_free,
                          "open precheck: ki not multiple of 8 (fake pointers, no read/alloc)");
                    rel(160, 39);   // hidden width not multiple of 8
                    Exl3Peer p2;
                    check(!p2.open(1, 2560, bad, max_tokens, max_k, err) && !p2.valid() &&
                          free_vram() == precheck_free,
                          "open precheck: hidden width not multiple of 8 (fake pointers, no read/alloc)");
                }
            }
        }
        // ---- allocation/copy fault matrix over the INJECTABLE positions.
        // Scope: the tier's OWN operations only (stream create + 4 device buffers + 6 pinned
        // buffers + 9 planes x alloc/copy per entry).  NOT injected: the workspace's MoeBuffer
        // allocations, the row_of/map/vector allocations, and the upstream descriptor upload in
        // exl3.cu (now checked and throwing - covered by contract, not by injection).  Evidence,
        // stated exactly: the runs SURVIVE the allocator (a double free would abort inside
        // malloc - surviving is evidence, not proof that no invalid free exists); empty-state
        // introspection is exact; device-VRAM is checked within the runtime pool's documented
        // band (banded VRAM can miss small leaks); pinned-host/heap release is by code
        // inspection of free_all, NOT directly observed.
        {
            Exl3Peer limited;
            bool vram_short = false;
            check(!limited.open(1, n_embd, subset, max_tokens, max_k, err, SIZE_MAX, &vram_short) &&
                      vram_short && !limited.valid(),
                  "peer sizing: an unmet VRAM reserve is retryable and leaves no live tier");
            check(!limited.open(-1, n_embd, subset, max_tokens, max_k, err, 0, &vram_short) &&
                      !vram_short,
                  "peer sizing: an invalid device is not a retryable VRAM failure");
            check(limited.open(1, n_embd, subset, max_tokens, max_k, err, 0, &vram_short) &&
                      !vram_short && limited.resident() == (int64_t) subset.size(),
                  "peer sizing: reopen succeeds after a rejected attempt");
            limited.close();
            const char* mr = std::getenv("STRATA_EXL3_PEER_MULTIROW");
            const bool extra_pinned = mr && std::atoi(mr) > 0 && strata::kernels::exl3_fused_multirow_supported();
            const char* pack = std::getenv("STRATA_EXL3_PEER_PACK_PLAN");
            const bool packed_plan = extra_pinned && pack && pack[0] && pack[0] != '0';
            const bool pooled = strata::core::exl3_flag("STRATA_EXL3_PEER_ARENA", nullptr, false);
            const int expected_sites = 11 + 18 * 8 + (extra_pinned ? 2 : 0) + (packed_plan ? 2 : 0) +
                                       (pooled ? 1 : 0);
            // Pooling retains each borrowed-plane injection seam and adds one owner allocation.
            // stream + 4 dev + 6 pinned + optional 2 group tables / 2 packed buffers + planes
            // Two passes: the FIRST primes the runtime allocator (its pooling caches freed
            // blocks and coalesces late), so the second pass's exact-equality invariant is
            // about OUR unwind, not the runtime's caching granularity.
            // Two full passes over the positions.  Leak evidence, stated exactly: the runtime
            // allocator oscillates its cached region on a fixed ~32 MiB quantum (observed in
            // BOTH directions), so per-position free VRAM is checked within that documented
            // band and ACCUMULATION is checked pass-to-pass (a per-cycle leak of any plane
            // would move the pass-final free by positions x bytes between passes - though
            // banded VRAM can still miss small leaks).
            const size_t base_free = free_vram();
            const size_t band = (size_t) 32 << 20;
            size_t pass_final[2] = {0, 0};
            size_t max_drift = 0;
            int failures = 0;
            for (int pass = 0; pass < 2; ++pass) {
                for (int n = 1; n <= expected_sites + 2; ++n) {
                    char buf[16];
                    std::snprintf(buf, sizeof buf, "%d", n);
                    setenv("STRATA_EXL3_PEER_FAULT", buf, 1);
                    Exl3Peer p;
                    const bool okp = p.open(1, n_embd, subset, max_tokens, max_k, err);
                    unsetenv("STRATA_EXL3_PEER_FAULT");
                    if (okp) {
                        p.close();
                        break;
                    }
                    if (p.valid() || p.resident() != 0 || p.has(1, 100)) {
                        std::printf("  fault %d kept state (valid %d)\n", n, (int) p.valid());
                        ++g_fail;
                    }
                    const size_t f = free_vram();
                    const size_t d = f > base_free ? f - base_free : base_free - f;
                    max_drift = std::max(max_drift, d);
                    if (d > band) {
                        std::printf("  fault %d out of band: free %zu vs baseline %zu\n", n, f, base_free);
                        ++g_fail;
                    }
                    if (pass == 1) ++failures;
                    pass_final[pass] = f;
                }
            }
            {   // accumulation between identical passes = leak (any plane or buffer leaks
                // positions x bytes per pass, far beyond the runtime's one-time quantum)
                const size_t d = pass_final[1] > pass_final[0] ? pass_final[1] - pass_final[0]
                                                              : pass_final[0] - pass_final[1];
                if (d > band) {
                    std::printf("  matrix not pass-stable: final free %zu then %zu\n", pass_final[0], pass_final[1]);
                    ++g_fail;
                }
            }
            std::printf("  fault matrix evidence: 2 x %d cycles survived the allocator (no crash/abort), empty "
                        "state exact, pass-final free within band (%zu vs %zu), max one-time runtime-pool "
                        "drift %zu B (band %zu)\n", expected_sites, pass_final[1], pass_final[0], max_drift, band);
            std::snprintf(what, sizeof what,
                          "fault matrix: %d failed opens unwound (expected %d injectable sites)", failures,
                          expected_sites);
            check(failures == expected_sites, what);
            Exl3Peer p;
            check(p.open(1, n_embd, subset, max_tokens, max_k, err), "reopen after fault matrix");
            for (int n = 1; n <= 4; ++n) {   // 4 injectable launch seams (x H2D, idx H2D, rows kernel, rows D2H)
                char buf[16];
                std::snprintf(buf, sizeof buf, "%d", n);
                setenv("STRATA_EXL3_PEER_FAULT_LAUNCH", buf, 1);
                std::vector<float> x((size_t) n_embd, 0.5f), out((size_t) 10 * n_embd, 3.25f);
                std::vector<int32_t> ids(10, 100), kind(10, 2);
                const bool launched = p.launch(layers[0], x.data(), ids.data(), 1, 10, kind.data(), err);
                unsetenv("STRATA_EXL3_PEER_FAULT_LAUNCH");
                std::snprintf(what, sizeof what, "launch fault %d: launch rejected at the seam", n);
                check(!launched && p.failed() && !p.valid() && !p.has(1, 100), what);
                std::snprintf(what, sizeof what, "launch fault %d: failed tier advertises nothing", n);
                check(!p.finish(out.data(), err) && !p.launch(layers[0], x.data(), ids.data(), 1, 10, kind.data(), err), what);
                p.close();
                check(p.open(1, n_embd, subset, max_tokens, max_k, err) && p.valid(), "close clears the failed tier");
            }
            p.close();
        }
        cand.close();
    }

    // ================= REAL MODEL: explicit loop over all of layers 0/7/23 =================
    if (const char* model = std::getenv("STRATA_EXL3_TEST_MODEL"); model && *model) {
        On on0(0);
        strata::Exl3Model mdl(model);
        std::vector<Exl3Peer::Entry> subset;
        for (int64_t layer : {0, 7, 23})
            for (int32_t e : {100, 101, 102, 103}) {
                const std::string b = "model.language_model.layers." + std::to_string(layer) +
                                      ".mlp.experts." + std::to_string(e) + ".";
                const strata::Exl3Linear gl = mdl.linear(b + "gate_proj");
                const strata::Exl3Linear ul = mdl.linear(b + "up_proj");
                const strata::Exl3Linear dl = mdl.linear(b + "down_proj");
                Exl3Peer::Entry en;
                en.layer = layer;
                en.expert = e;
                auto set = [](Exl3Mat& m, const strata::Exl3Linear& l) {
                    m.trellis = l.trellis; m.suh = l.suh; m.svh = l.svh;
                    m.ki = l.ki; m.nj = l.nj; m.bits = l.bits; m.cb = l.cb;
                };
                set(en.g, gl);
                set(en.u, ul);
                set(en.d, dl);
                subset.push_back(en);
            }
        const int n_embd = 2560, max_tokens = 8, max_k = 10;
        {
            // Fixed CPU ownership mask composes with peer ownership. GPU rows
            // stay exact, and neither owner may have its output overwritten.
            using strata::kernels::Exl3ExpertStream;
            std::vector<float> x(4*2560),want(8*2560),got(8*2560,3.25f);
            for(size_t i=0;i<x.size();++i) x[i]=((int)((i*113)%2000)-1000)*0.001f;
            int32_t ids[]={100,101,100,102,103,101,104,105};
            uint8_t mask[]={0,0,0,1,0,0,1,0};
            Exl3ExpertStream all(model,2560,48,0,nullptr,0,(size_t)512<<20,0,1,8);
            check(all.run_multi(0,x.data(),ids,4,2,want.data(),nullptr),"CPU hybrid mask: GPU oracle");
            Exl3ExpertStream gpu(model,2560,48,2,nullptr,0,(size_t)512<<20,0,1,8);
            std::vector<uint8_t> owned(48*512,0); owned[100]=1;
            gpu.set_owned_elsewhere(owned.data(),48,512);
            bool ok=gpu.run_multi_complement(0,x.data(),ids,4,2,got.data(),nullptr,mask);
            for(int r=0;r<8;++r) for(int c=0;c<2560;++c)
                ok=ok && got[(size_t)r*2560+c]==((mask[r] || ids[r]==100) ? 3.25f : want[(size_t)r*2560+c]);
            check(ok && !gpu.holds(0,102) && !gpu.holds(0,104) && gpu.loan_dense_invariant(),
                  "CPU hybrid mask: exact GPU rows, peer/CPU sentinels, no CPU admission");
            std::fill(got.begin(),got.end(),3.25f); mask[0]=1;
            check(!gpu.run_multi_complement(0,x.data(),ids,4,2,got.data(),nullptr,mask) &&
                  std::all_of(got.begin(),got.end(),[](float v){return v==3.25f;}),
                  "CPU hybrid mask: split expert group rejected before GPU output");
            std::fill(std::begin(mask),std::end(mask),1);
            check(gpu.run_multi_complement(0,x.data(),ids,4,2,got.data(),nullptr,mask) &&
                  std::all_of(got.begin(),got.end(),[](float v){return v==3.25f;}),
                  "CPU hybrid mask: all-CPU call leaves output untouched");
        }
        Exl3Peer cand;
        if (!cand.open(1, n_embd, subset, max_tokens, max_k, err)) {
            std::printf("real open failed: %s\n", err.c_str());
            return 1;
        }
        Oracle oracle;
        if (!oracle.init(n_embd, max_tokens, err)) {
            std::printf("real oracle init failed: %s\n", err.c_str());
            return 1;
        }
        for (size_t i = 0; i < subset.size(); ++i)
            if (oracle.add(subset[i].g, subset[i].u, subset[i].d, err) != (int) i) {
                std::printf("real oracle add failed: %s\n", err.c_str());
                return 1;
            }
        check(cand.resident() == 12 && cand.has(23, 103) && !cand.has(23, 104),
              "real model: 12 experts over layers 0/7/23");
        for (int64_t layer : {0, 7, 23}) {
            for (int tokens : {1, 4, 8}) {
                const int k = 10, routes = tokens * k;
                std::vector<float> x((size_t) tokens * n_embd), got((size_t) routes * n_embd, 3.25f), want;
                std::vector<int32_t> ids((size_t) routes), kind((size_t) routes);
                std::vector<int> ridx((size_t) routes);
                const int base = layer == 0 ? 0 : layer == 7 ? 4 : 8;
                uint32_t s = 999u + (uint32_t) tokens * 10 + (uint32_t) layer;
                for (auto& v : x) { s = 1103515245u * s + 12345u; v = (float) ((int) (s >> 16) % 2000 - 1000) * 0.001f; }
                for (int r = 0; r < routes; ++r) {
                    ids[(size_t) r] = 100 + (r * 5 + tokens) % 4;
                    kind[(size_t) r] = (r % 4 == 3) ? 0 : 2;        // mixed kinds
                    ridx[(size_t) r] = base + (ids[(size_t) r] - 100);
                }
                if (!oracle.expected(x, ridx, kind, tokens, k, want, err)) {
                    std::printf("real oracle failed: %s\n", err.c_str());
                    ++g_fail;
                    continue;
                }
                const bool launched = cand.launch(layer, x.data(), ids.data(), tokens, k, kind.data(), err);
                const bool finished = launched && cand.finish(got.data(), err);
                bool rows_ok = true, want_ok = true, sentinel_ok = true;
                double worst_norm = 0;
                bool rel_ok = launched && finished;
                for (int r = 0; r < routes; ++r) {
                    const float* gr = got.data() + (size_t) r * n_embd;
                    const float* wr = want.data() + (size_t) r * n_embd;
                    if (kind[(size_t) r] == 2) {
                        rows_ok = rows_ok && row_sane(gr, (size_t) n_embd);
                        want_ok = want_ok && row_sane(wr, (size_t) n_embd);
                        double nr = 0;
                        rel_ok = rel_ok && row_rel(gr, wr, (size_t) n_embd, &nr);
                        worst_norm = std::max(worst_norm, nr);
                    } else {
                        for (int i = 0; i < n_embd; ++i) if (gr[i] != 3.25f || wr[i] != 3.25f) sentinel_ok = false;
                    }
                }
                std::printf("real T=%d layer=%lld: cross-path worst per-row norm %.3e - %s\n", tokens,
                            (long long) layer, worst_norm, (rel_ok && rows_ok && want_ok && sentinel_ok) ? "PASS" : "FAIL");
                check(rel_ok && rows_ok && want_ok && sentinel_ok, "real model: rows vs independent oracle");
            }
        }
        for(int mode : {0,1,2}) {
            const bool p2p=mode>0;
            int access=0; cudaDeviceCanAccessPeer(&access,1,0);
            if(p2p && !access) {std::printf("SKIP: direct peer output unavailable\n");continue;}
            cudaSetDevice(0);
            strata::kernels::Exl3PeerOutput output;
            Exl3Peer service;
            check(service.open(1,2560,subset,8,10,err) && output.open(0,1,2560,80,err,p2p,mode==2),
                  p2p ? "device output: P2P setup" : "device output: staged setup");
            float* parts=nullptr; uint32_t* skip=nullptr; cudaStream_t stream=nullptr;
            cudaMalloc((void**)&parts,80*2560*sizeof(float)); cudaMalloc((void**)&skip,sizeof(uint32_t));
            cudaMemset(skip,0,sizeof(uint32_t)); cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);
            std::vector<float> x(8*2560),sentinel(80*2560,3.25f),want=sentinel,got=sentinel;
            for(size_t i=0;i<x.size();++i) x[i]=((int)((i*97)%2000)-1000)*0.001f;
            std::vector<int32_t> ids(80),kind(80);
            for(int r=0;r<80;++r) {ids[r]=100+(r*3+r/10)%4;kind[r]=r%3 ? 2 : 0;}
            cudaGraph_t graph=nullptr; cudaGraphExec_t exec=nullptr;
            cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal);
            output.copy(parts,80,skip,1,stream);
            cudaStreamEndCapture(stream,&graph); cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0);
            bool exact=true;
            for(int repeat=0;repeat<3;++repeat) for(int tokens:{1,4,8}) for(int layer:{0,7,23}) {
                want=sentinel;
                bool ok=service.launch(layer,x.data(),ids.data(),tokens,10,kind.data(),err) && service.finish(want.data(),err);
                ok=ok && service.launch(layer,x.data(),ids.data(),tokens,10,kind.data(),err,&output) && service.finish_deferred(err);
                cudaMemcpyAsync(parts,sentinel.data(),sentinel.size()*sizeof(float),cudaMemcpyHostToDevice,stream);
                ok=ok && cudaGraphLaunch(exec,stream)==cudaSuccess && cudaStreamSynchronize(stream)==cudaSuccess;
                ok=ok && cudaMemcpy(got.data(),parts,got.size()*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess;
                exact=exact && ok && got==want;
                if(!ok) std::printf("device output failure: %s\n",err.c_str());
            }
            check(exact,"device output: repeated captured replay, mixed routes and tails exactly match host finish");
            service.close(); // transport frame must outlive peer weights/stream
            output.empty();
            cudaMemcpyAsync(parts,sentinel.data(),sentinel.size()*sizeof(float),cudaMemcpyHostToDevice,stream);
            output.copy(parts,80,skip,1,stream); cudaStreamSynchronize(stream);
            cudaMemcpy(got.data(),parts,got.size()*sizeof(float),cudaMemcpyDeviceToHost);
            check(got==sentinel,"device output: empty frame after peer close preserves primary rows");
            output.release();
            output.copy(parts,80,skip,1,stream);
            check(cudaStreamSynchronize(stream)==cudaSuccess,"device output: watchdog release drains captured waits");
            // Hold the producer behind a separate mapped gate, so release must
            // drain a genuinely incomplete frame rather than an already-ready one.
            strata::kernels::Exl3PeerOutput blocked;
            check(blocked.open(0,1,8,1,err,p2p,mode==2),"device output: blocked-frame setup");
            strata::kernels::Exl3PeerOutputRecord *gate=nullptr,*device_gate=nullptr;
            uint16_t* source=nullptr;
            float* discard=nullptr;
            cudaStream_t producer=nullptr;
            cudaSetDevice(1);
            cudaHostAlloc((void**)&gate,sizeof(*gate),cudaHostAllocPortable|cudaHostAllocMapped);
            cudaHostGetDevicePointer((void**)&device_gate,gate,0);
            *gate=strata::kernels::Exl3PeerOutputRecord{};
            gate->expected=1;
            cudaMalloc((void**)&source,8*sizeof(uint16_t));
            cudaMalloc((void**)&discard,8*sizeof(float));
            cudaMemset(source,0,8*sizeof(uint16_t));
            cudaStreamSynchronize(nullptr);
            cudaStreamCreateWithFlags(&producer,cudaStreamNonBlocking);
            strata::kernels::exl3_peer_output_copy(device_gate,source,discard,1,8,nullptr,0,producer);
            const int destination=0;
            check(blocked.enqueue(source,&destination,1,producer,err),"device output: enqueue behind blocked producer");
            cudaSetDevice(0);
            blocked.copy(parts,1,nullptr,0,stream);
            check(cudaStreamQuery(stream)==cudaErrorNotReady,"device output: consumer cannot finish before publication");
            blocked.release();
            const double deadline=ms_now()+2000;
            cudaError_t status;
            do {status=cudaStreamQuery(stream);} while(status==cudaErrorNotReady && ms_now()<deadline);
            check(status==cudaSuccess,"device output: watchdog drains an unpublished frame without producer completion");
            // Always unblock the producer before draining/freeing its resources.
            *(volatile uint32_t*)&gate->aborted=1;
            cudaSetDevice(1);
            cudaStreamSynchronize(producer);
            cudaStreamDestroy(producer);cudaFree(source);cudaFree(discard);cudaFreeHost(gate);
            cudaSetDevice(0);
            cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream); cudaFree(parts);cudaFree(skip);
        }
        cand.close();
    } else {
        std::printf("real-model section SKIPPED (set STRATA_EXL3_TEST_MODEL=<model-dir>)\n");
    }

    // ============ ENGINE SEAM: stream ownership exclusion + mixed-route split vs full compute =
    if (const char* model = std::getenv("STRATA_EXL3_TEST_MODEL"); model && *model) {
        using strata::kernels::Exl3ExpertStream;
        const int n_embd = 2560, n_layers = 48, n_expert = 512;
        std::vector<uint8_t> owned((size_t) n_layers * n_expert, 0);
        for (int32_t e = 100; e < 104; ++e) owned[(size_t) 0 * n_expert + e] = 1;
        std::unique_ptr<Exl3ExpertStream> prim;
        std::unique_ptr<Exl3Peer> peer;
        { On on(0);
          // the arena backing (and thus the loan machinery) needs the opt-in at construction
          setenv("STRATA_EXL3_PREFILL_BORROW", "1", 1);
          prim = std::make_unique<Exl3ExpertStream>(model, n_embd, n_layers, 8, nullptr);   // 8 slots: churny
          unsetenv("STRATA_EXL3_PREFILL_BORROW");
          prim->set_owned_elsewhere(owned.data(), n_layers, n_expert); }
        {
            On on1(1);
            strata::Exl3Model mdl(model);
            std::vector<Exl3Peer::Entry> entries;
            for (int32_t e = 100; e < 104; ++e) {
                const std::string b = "model.language_model.layers.0.mlp.experts." + std::to_string(e) + ".";
                Exl3Peer::Entry en;
                en.layer = 0;
                en.expert = e;
                auto set = [](Exl3Mat& m, const strata::Exl3Linear& l) {
                    m.trellis = l.trellis; m.suh = l.suh; m.svh = l.svh;
                    m.ki = l.ki; m.nj = l.nj; m.bits = l.bits; m.frac = l.frac; m.cb = l.cb;
                };
                set(en.g, mdl.linear(b + "gate_proj"));
                set(en.u, mdl.linear(b + "up_proj"));
                set(en.d, mdl.linear(b + "down_proj"));
                entries.push_back(en);
            }
            peer = std::make_unique<Exl3Peer>();
            if (!peer->open(1, n_embd, entries, 8, 64, err)) {
                std::printf("seam peer open failed: %s\n", err.c_str());
                ++g_fail;
            }
        }
        // churn the primary cache with many distinct experts, INCLUDING the peer-owned ones
        std::vector<float> x((size_t) 2 * n_embd), ref((size_t) 20 * n_embd, 0.0f), got((size_t) 20 * n_embd, 3.25f);
        uint32_t s2 = 4242u;
        for (auto& v : x) { s2 = 1103515245u * s2 + 12345u; v = (float) ((int) (s2 >> 16) % 2000 - 1000) * 0.001f; }
        std::vector<int32_t> churn(20);
        for (int r = 0; r < 20; ++r) churn[(size_t) r] = (r * 7) % 80;   // experts 0..79: forces evictions
        for (int rep = 0; rep < 6; ++rep) {
            if (!prim->run_multi(0, x.data(), churn.data(), 2, 10, ref.data(), nullptr)) ++g_fail;
            std::vector<int32_t> mix(20);
            for (int r = 0; r < 20; ++r) mix[(size_t) r] = 100 + (r % 4);   // peer-owned traffic too
            if (!prim->run_multi(0, x.data(), mix.data(), 2, 10, ref.data(), nullptr)) ++g_fail;
        }
        check(prim->owned_exclusions() > 0, "ownership exclusion engaged during churn");
        bool disjoint = true;
        for (int32_t e = 100; e < 104; ++e) disjoint = disjoint && !prim->holds(0, e);
        check(disjoint, "peer-owned experts never resident after churn");
        check(prim->holds(0, 0) || prim->holds(0, 7) || prim->holds(0, 14), "non-owned experts still admitted");
        // loan lend/restore must not resurrect peer-owned weights
        {
            auto loan = prim->lend_tail(2);
            check(loan.address != nullptr, "loan opens over the primary cache");
            std::vector<int32_t> mix(20);
            for (int r = 0; r < 20; ++r) mix[(size_t) r] = 100 + (r % 4);
            (void) prim->run_multi(0, x.data(), mix.data(), 2, 10, ref.data(), nullptr);
            check(prim->restore_tail(), "loan restores");
            disjoint = true;
            for (int32_t e = 100; e < 104; ++e) disjoint = disjoint && !prim->holds(0, e);
            check(disjoint, "peer-owned experts still absent after lend/restore");
        }
        // ---- mixed-route split vs the full primary compute (exact by destination)
        for (int shape = 0; shape < 4; ++shape) {
            // categories: peer-owned, primary (resident or scratch), duplicates, and a zero-peer set
            std::vector<int32_t> ids(20), kind(20);
            if (shape == 2) for (int r = 0; r < 20; ++r) { ids[(size_t) r] = (r * 7) % 80; kind[(size_t) r] = -1; }
            else for (int r = 0; r < 20; ++r) {
                ids[(size_t) r] = (r % 5 == 0) ? 100 + (r % 4) : (r % 3 == 0 ? 5 : (r * 7) % 80);
                kind[(size_t) r] = 0;
            }
            if (shape == 3) for (int r = 0; r < 20; ++r) ids[(size_t) r] = 100 + r % 4;
            // the split: classify -> peer.launch -> complement (k=1 layout) -> peer.finish -> scatter
            int64_t n_peer = 0;
            for (int r = 0; r < 20; ++r) {
                kind[(size_t) r] = peer->has(0, ids[(size_t) r]) ? 2 : -1;
                n_peer += kind[(size_t) r] == 2;
            }
            std::vector<float> xin((size_t) 20 * n_embd), outc((size_t) 20 * n_embd);
            std::vector<int32_t> idc(20);
            std::vector<int64_t> rowmap(20);
            int64_t c = 0;
            for (int r = 0; r < 20; ++r) {
                if (kind[(size_t) r] == 2) continue;
                std::memcpy(xin.data() + (size_t) c * n_embd, x.data() + (size_t) (r / 10) * n_embd,
                            (size_t) n_embd * sizeof(float));
                idc[(size_t) c] = ids[(size_t) r];
                rowmap[(size_t) c] = r;
                ++c;
            }
            bool ok = true;
            if (n_peer > 0) ok = peer->launch(0, x.data(), ids.data(), 2, 10, kind.data(), err);
            if (ok) ok = prim->run_multi(0, xin.data(), idc.data(), (int) c, 1, outc.data(), nullptr);
            if (ok && n_peer > 0) ok = peer->finish(got.data(), err);
            if (ok)
                for (int64_t j = 0; j < c; ++j)
                    std::memcpy(got.data() + (size_t) rowmap[(size_t) j] * n_embd,
                                outc.data() + (size_t) j * n_embd, (size_t) n_embd * sizeof(float));
            // the reference: the FULL primary compute of the same routes (owned rows via scratch)
            const bool okref = prim->run_multi(0, x.data(), ids.data(), 2, 10, ref.data(), nullptr);
            bool rows_ok = true;
            for (int r = 0; r < 20; ++r) rows_ok = rows_ok && row_sane(got.data() + (size_t) r * n_embd, (size_t) n_embd);
            const bool identical = ok && okref && rows_ok &&
                std::memcmp(got.data(), ref.data(), (size_t) 20 * n_embd * sizeof(float)) == 0;
            std::printf("seam shape=%d (%s): %s\n", shape,
                        shape == 2 ? "zero-peer" : (n_peer == 20 ? "all-peer" : "mixed"),
                        identical ? "split == full primary, byte-identical PASS" : "FAIL");
            check(identical, "mixed-route split vs full compute (exact by destination)");
            // Sparse service must preserve original token/route indices and leave ALL peer
            // destinations untouched until finish(), including the empty-complement case.
            std::fill(got.begin(), got.end(), 3.25f);
            bool sparse_ok = prim->run_multi_complement(0, x.data(), ids.data(), 2, 10, got.data(), nullptr);
            for (int r = 0; r < 20; ++r) {
                const float* row = got.data() + (size_t) r * n_embd;
                if (kind[(size_t) r] == 2) {
                    for (int i = 0; i < n_embd; ++i) sparse_ok = sparse_ok && row[i] == 3.25f;
                } else {
                    sparse_ok = sparse_ok && std::memcmp(row, ref.data() + (size_t) r * n_embd,
                                                         (size_t) n_embd * sizeof(float)) == 0;
                }
            }
            check(sparse_ok, "sparse complement: exact primary rows, peer sentinels untouched");
            if (n_peer > 0)
                sparse_ok = sparse_ok && peer->launch(0, x.data(), ids.data(), 2, 10, kind.data(), err) &&
                            peer->finish(got.data(), err);
            check(sparse_ok && std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) == 0,
                  "sparse complement + peer == full primary (mixed/zero/all-peer)");
        }
        // ---- already-warmed then exclude: enabling the table AFTER residency must invalidate
        // the affected layer's whole cache (dense [0,filled) preserved for the loan machinery)
        {
            setenv("STRATA_EXL3_PREFILL_BORROW", "1", 1);
            std::unique_ptr<Exl3ExpertStream> warm;
            { On on(0); warm = std::make_unique<Exl3ExpertStream>(model, n_embd, n_layers, 8, nullptr); }
            unsetenv("STRATA_EXL3_PREFILL_BORROW");
            std::vector<int32_t> wids(20);
            for (int r = 0; r < 20; ++r) wids[(size_t) r] = 100 + (r % 4);   // would-be-owned experts
            std::vector<float> tmp((size_t) 20 * n_embd, 0.0f);
            bool warmed = true;
            for (int rep = 0; rep < 3; ++rep)
                warmed = warmed && warm->run_multi(0, x.data(), wids.data(), 2, 10, tmp.data(), nullptr);
            check(warmed && warm->holds(0, 100), "warmed: owned candidates resident before exclusion");
            warm->set_owned_elsewhere(owned.data(), n_layers, n_expert);
            bool inv = true;
            for (int32_t e = 0; e < 80; ++e) inv = inv && !warm->holds(0, e);
            check(inv, "warm-then-exclude invalidates the affected layer entirely (dense layout)");
            // capacity is RETURNED, not merely hidden from lookup: the invalidated layer (filled=0,
            // dense [0,filled) restart) must refill its full 8 slots with non-owned experts
            {
                std::vector<int32_t> rid(16);
                for (int r = 0; r < 16; ++r) rid[(size_t) r] = r % 8;   // 8 distinct non-owned
                const bool refilled = warm->run_multi(0, x.data(), rid.data(), 2, 8, tmp.data(), nullptr);
                int residents = 0;
                for (int32_t e = 0; e < 80; ++e) residents += warm->holds(0, e) ? 1 : 0;
                for (int32_t e = 100; e < 104; ++e) residents += warm->holds(0, e) ? 1 : 0;
                check(refilled && residents == 8,
                      "exclusion-invalidated layer refills to full slot capacity (no dead slots)");
            }
            auto loan2 = warm->lend_tail(2);
            check(loan2.address != nullptr && warm->restore_tail(),
                  "loan lend/restore intact after warm-then-exclude");
            for (int32_t e = 100; e < 104; ++e) inv = inv && !warm->holds(0, e);
            check(inv, "owned experts still absent after lend/restore post-exclusion");
            warm.reset();
        }
        // ---- scratch-vs-cache parity: the SAME window on a table-OFF stream (cache path) must
        // match the table-ON stream (excluded experts compute via the bounded scratch)
        {
            setenv("STRATA_EXL3_PREFILL_BORROW", "1", 1);
            std::unique_ptr<Exl3ExpertStream> plain;
            { On on(0); plain = std::make_unique<Exl3ExpertStream>(model, n_embd, n_layers, 8, nullptr); }
            unsetenv("STRATA_EXL3_PREFILL_BORROW");
            std::vector<int32_t> wids(20);
            for (int r = 0; r < 20; ++r) wids[(size_t) r] = 100 + (r % 4);   // all owned by the peer
            std::vector<float> a((size_t) 20 * n_embd, 0.0f), b((size_t) 20 * n_embd, 0.0f);
            bool ok1 = true;
            for (int rep = 0; rep < 4; ++rep) {   // churn both caches first, same traffic
                ok1 = ok1 && plain->run_multi(0, x.data(), wids.data(), 2, 10, a.data(), nullptr);
                ok1 = ok1 && prim->run_multi(0, x.data(), wids.data(), 2, 10, b.data(), nullptr);
            }
            const bool same = ok1 && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
            double worst = 0;
            for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, (double) std::fabs((double) a[i] - (double) b[i]));
            std::printf("scratch-vs-cache rows: %s (max |diff| %.3e)\n",
                        same ? "byte-identical" : "DIFFER", worst);
            check(same, "excluded experts: scratch fallback == cache path rows");
            plain.reset();
        }
        // ---- launch_prefill seam: both batch arms (PEER_RECON_BATCH and PEER_BATCH alone) take
        // the same masked layer and must return the same weighted sums to fp16-rounding noise.
        // The PEER_BATCH-only combination indexes the recon tables nowhere (a crash here was the
        // first thing the WMMA unlock's WEBPB arm found on the real model), and >=16-row groups
        // force both arms' batching to engage, not just the standalone group path.
        {
            const int64_t T = 40, k = 10;
            std::vector<uint16_t> x16((size_t) T * n_embd);
            std::vector<int32_t> ids((size_t) T * k);
            std::vector<uint8_t> mask((size_t) T * k, 0);
            std::vector<float> w((size_t) T * k, 0.0f);
            uint32_t s3 = 97531u;
            auto rnd = [&] { s3 = 1103515245u * s3 + 12345u; return (int) (s3 >> 16); };
            for (auto& v : x16)
                v = strata::kernels::f16_from_f32((float) (rnd() % 2000 - 1000) * 0.001f);
            for (int64_t r = 0; r < T * k; ++r) {
                const bool mine = (rnd() % 4) != 0;   // ~75% of routes on the peer's 4 experts
                ids[(size_t) r] = mine ? 100 + (rnd() % 4) : (rnd() % 80);
                mask[(size_t) r] = mine ? 1 : 0;
                w[(size_t) r] = (float) (rnd() % 1000) / 4000.0f + 0.05f;
            }
            auto seam = [&](bool recon) -> std::vector<float> {
                if (recon) {
                    setenv("STRATA_EXL3_PEER_RECON_BATCH", "1", 1);
                    unsetenv("STRATA_EXL3_PEER_BATCH");
                } else {
                    unsetenv("STRATA_EXL3_PEER_RECON_BATCH");
                    setenv("STRATA_EXL3_PEER_BATCH", "1", 1);
                }
                std::vector<float> sums;
                std::unique_ptr<Exl3Peer> p;
                { On on1(1);
                  strata::Exl3Model mdl(model);
                  std::vector<Exl3Peer::Entry> entries;
                  for (int32_t e = 100; e < 104; ++e) {
                      const std::string bname =
                          "model.language_model.layers.0.mlp.experts." + std::to_string(e) + ".";
                      Exl3Peer::Entry en;
                      en.layer = 0;
                      en.expert = e;
                      auto set = [](Exl3Mat& m, const strata::Exl3Linear& l) {
                          m.trellis = l.trellis; m.suh = l.suh; m.svh = l.svh;
                          m.ki = l.ki; m.nj = l.nj; m.bits = l.bits; m.frac = l.frac; m.cb = l.cb;
                      };
                      set(en.g, mdl.linear(bname + "gate_proj"));
                      set(en.u, mdl.linear(bname + "up_proj"));
                      set(en.d, mdl.linear(bname + "down_proj"));
                      entries.push_back(en);
                  }
                  p = std::make_unique<Exl3Peer>();
                  std::string err;
                  if (!p->open(1, n_embd, entries, 8, 64, err) ||
                      !p->enable_prefill(T, k, 512, T * k, err)) {
                      std::printf("prefill seam open (%s): %s\n", recon ? "recon" : "batch", err.c_str());
                      ++g_fail;
                      unsetenv("STRATA_EXL3_PEER_RECON_BATCH");
                      unsetenv("STRATA_EXL3_PEER_BATCH");
                      return sums;
                  }
                }
                unsetenv("STRATA_EXL3_PEER_RECON_BATCH");
                unsetenv("STRATA_EXL3_PEER_BATCH");
                std::string err;
                const bool launched = p->launch_prefill(0, x16.data(), T, ids.data(), mask.data(),
                                                        w.data(), err);
                check(launched && p->finish_prefill(err), recon ? "prefill seam recon arm" : "prefill seam batch arm");
                if (!launched) { std::printf("prefill seam launch(%d): %s\n", (int) recon, err.c_str()); return sums; }
                const float* s = p->prefill_sums_host();
                sums.assign(s, s + (size_t) T * n_embd);
                p.reset();
                (void) cudaGetLastError();   // a closed tier may leave a stale ring state: setup hygiene
                return sums;
            };
            const std::vector<float> a = seam(true), b = seam(false);
            check(!a.empty() && !b.empty(), "prefill seam: both arms produced sums");
            {   // the minimal probe first: ONE token, ONE route, weight 1 - the sums must be the
                // expert's plain FFN row, isolating the row from any summing/weights semantics
                const int64_t one_T = 1, one_k = 1;
                std::vector<uint16_t> ox16((size_t) one_T * n_embd);
                std::vector<int32_t> oids{(int32_t) 100};
                std::vector<uint8_t> omask{1};
                const float oweight = 0.25f;   // not 1: the scatter-add's multiply is in the check
                std::vector<float> ow{oweight};
                std::vector<float> xr((size_t) one_T * n_embd), rows((size_t) one_T * one_k * n_embd);
                for (int i = 0; i < n_embd; ++i) {
                    ox16[(size_t) i] = x16[(size_t) i];   // reuse the random activations
                    xr[(size_t) i] = strata::kernels::f32_from_f16(ox16[(size_t) i]);
                }
                const bool ran = prim->run_multi(0, xr.data(), oids.data(), (int) one_T, (int) one_k,
                                                 rows.data(), nullptr);
                auto one_seam = [&](bool recon) -> std::vector<float> {
                    if (recon) { setenv("STRATA_EXL3_PEER_RECON_BATCH", "1", 1); unsetenv("STRATA_EXL3_PEER_BATCH"); }
                    else { unsetenv("STRATA_EXL3_PEER_RECON_BATCH"); setenv("STRATA_EXL3_PEER_BATCH", "1", 1); }
                    std::vector<float> sums;
                    std::unique_ptr<Exl3Peer> p;
                    { On on1(1);
                      strata::Exl3Model mdl(model);
                      std::vector<Exl3Peer::Entry> entries;
                      const std::string bname = "model.language_model.layers.0.mlp.experts.100.";
                      Exl3Peer::Entry en;
                      en.layer = 0;
                      en.expert = 100;
                      auto set = [](Exl3Mat& m, const strata::Exl3Linear& l) {
                          m.trellis = l.trellis; m.suh = l.suh; m.svh = l.svh;
                          m.ki = l.ki; m.nj = l.nj; m.bits = l.bits; m.frac = l.frac; m.cb = l.cb;
                      };
                      set(en.g, mdl.linear(bname + "gate_proj"));
                      set(en.u, mdl.linear(bname + "up_proj"));
                      set(en.d, mdl.linear(bname + "down_proj"));
                      entries.push_back(en);
                      p = std::make_unique<Exl3Peer>();
                      std::string err;
                      if (!p->open(1, n_embd, entries, 8, 64, err) ||
                          !p->enable_prefill(one_T, one_k, 512, one_T * one_k, err)) {
                          unsetenv("STRATA_EXL3_PEER_RECON_BATCH");
                          unsetenv("STRATA_EXL3_PEER_BATCH");
                          return sums;
                      }
                    }
                    unsetenv("STRATA_EXL3_PEER_RECON_BATCH");
                    unsetenv("STRATA_EXL3_PEER_BATCH");
                    std::string err;
                    if (!p->launch_prefill(0, ox16.data(), one_T, oids.data(), omask.data(),
                                           ow.data(), err) || !p->finish_prefill(err))
                        return sums;
                    const float* s = p->prefill_sums_host();
                    sums.assign(s, s + (size_t) one_T * n_embd);
                    p.reset();
                    (void) cudaGetLastError();   // same setup-boundary hygiene as the engine
                    return sums;
                };
                if (ran) {
                    const std::vector<float> oa = one_seam(true), ob = one_seam(false);
                    double ra = 0, rb = 0, den = 0;
                    for (int i = 0; i < n_embd; ++i) {
                        const double d = oweight * (double) rows[(size_t) i];
                        ra += ((double) oa[(size_t) i] - d) * ((double) oa[(size_t) i] - d);
                        rb += ((double) ob[(size_t) i] - d) * ((double) ob[(size_t) i] - d);
                        den += d * d;
                    }
                    const double rrel = std::sqrt(ra / (den + 1e-30)), brel = std::sqrt(rb / (den + 1e-30));
                    const bool opass = !oa.empty() && !ob.empty() && std::isfinite(rrel) && rrel < 0.02 &&
                                       std::isfinite(brel) && brel < 0.02;
                    std::printf("prefill seam one-route row (w=%.2f): recon rel=%.3g batch rel=%.3g %s\n",
                                oweight, rrel, brel, opass ? "PASS" : "FAIL");
                    check(opass, "prefill seam: the one-route weighted row matches the primary");
                }
            }
            // the multi-route layer stays STRUCTURAL (finite, zero-token zeros, per-token
            // magnitudes) plus the one-route numeric probe above: a full multi-route numeric
            // oracle needs its own per-route reference (run_multi's row order across a mixed
            // 40x10 window does not line up with the caller's route order trivially) - a
            // follow-up, noted in notes/MIMO-BRINGUP.md
            if (!a.empty() && !b.empty()) {
                bool finite_ok = true, zero_ok = true;
                for (int64_t t = 0; t < T; ++t) {
                    bool any = false;
                    for (int j = 0; j < k; ++j) any = any || mask[(size_t) (t * k + j)] != 0;
                    for (int i = 0; i < n_embd; ++i) {
                        const size_t at = (size_t) t * n_embd + i;
                        finite_ok = finite_ok && std::isfinite(a[at]) && std::isfinite(b[at]);
                        if (!any) zero_ok = zero_ok && a[at] == 0.0f && b[at] == 0.0f;
                    }
                }
                const bool pass = finite_ok && zero_ok;
                std::printf("prefill seam recon-vs-batch multi-route: finite=%d zero-tokens=%d "
                            "(numeric cross-check: the one-route probe above) %s\n",
                            (int) finite_ok, (int) zero_ok, pass ? "PASS" : "FAIL");
                check(pass, "prefill seam: both batch arms produce structural sums");
            }
        }
        peer.reset();
        prim.reset();
    }

    // Adaptive peer: pending ownership must reserve incoming experts without dropping their
    // primary scratch results; old ownership must be released, and hot primary experts stay out.
    if (const char* model = std::getenv("STRATA_EXL3_TEST_MODEL"); model && *model) {
        On on(0);
        constexpr int n = 2560;
        strata::Exl3Model source(model);
        auto entry = [&](int e) {
            Exl3Peer::Entry en; en.layer = 0; en.expert = e;
            const std::string b = "model.language_model.layers.0.mlp.experts." + std::to_string(e) + ".";
            auto mat = [&](const char* name) {
                const auto l = source.linear(b + name);
                return Exl3Mat{l.trellis, l.suh, l.svh, l.ki, l.nj, l.bits, l.cb, l.frac};
            };
            en.g=mat("gate_proj"); en.u=mat("up_proj"); en.d=mat("down_proj"); return en;
        };
        strata::kernels::Exl3ExpertStream primary(model, n, 1, 80, nullptr, 0, 0);
        strata::kernels::Exl3ExpertStream reference(model, n, 1, 80, nullptr, 0, 0);
        Exl3Peer peer;
        std::vector<uint8_t> owned(512, 0); owned[0]=owned[1]=1;
        primary.set_owned_elsewhere(owned.data(), 1, 512);
        check(peer.open(1,n,{entry(0),entry(1)},8,10,err), "adapt real: open");
        strata::kernels::Exl3PeerAdapt::Options opt;
        opt.every=1; opt.max_swaps=1; opt.byte_cap=2u<<20;
        strata::kernels::Exl3PeerAdapt adapt(peer,primary,model,1,512,owned,opt);
        std::vector<float> x(n), got(n), want(n);
        for(int i=0;i<n;++i)x[i]=(float)(i%37-18)*.01f;
        int32_t id=90, hot=1;
        check(adapt.before_window(err), "adapt real: first window does not refill");
        for(int i=0;i<8;++i)adapt.observe(0,&id,1);
        for(int i=0;i<4;++i)adapt.observe(0,&hot,1);
        check(adapt.before_window(err) && peer.refilling() && !peer.has(0,0) && !peer.has(0,90) && peer.has(0,1),
              "adapt real: victim withdrawn, incoming unpublished, unrelated resident available");
        check(primary.run_multi_complement(0,x.data(),&id,1,1,got.data(),nullptr) && !primary.holds(0,90) &&
              reference.run_multi(0,x.data(),&id,1,1,want.data(),nullptr) &&
              std::memcmp(got.data(),want.data(),n*sizeof(float))==0,
              "adapt real: pending incoming computed on scratch, not skipped or admitted");
        int32_t kind=2;
        check(peer.launch(0,x.data(),&hot,1,1,&kind,err) && peer.finish(got.data(),err),
              "adapt real: unrelated resident computes during refill");
        check(adapt.drain(err) && peer.has(0,90) && !peer.has(0,0) && owned[90] && !owned[0] && !primary.holds(0,90),
              "adapt real: landed incoming disjoint and old ownership released");
        check(peer.launch(0,x.data(),&id,1,1,&kind,err) && peer.finish(got.data(),err) &&
              std::memcmp(got.data(),want.data(),n*sizeof(float))==0,
              "adapt real: replaced expert rows match independent primary FFN bytes");
        id=0;
        check(primary.run_multi(0,x.data(),&id,1,1,got.data(),nullptr) && primary.holds(0,0),
              "adapt real: outgoing expert becomes primary-admissible");
        for(int i=0;i<100;++i)adapt.observe(0,&id,1);
        id=91; for(int i=0;i<12;++i)adapt.observe(0,&id,1);
        check(adapt.before_window(err) && adapt.drain(err) && peer.has(0,91) && !peer.has(0,0) &&
              primary.holds(0,0) && !primary.holds(0,91) && adapt.swaps()==2 && adapt.upload_bytes() <= (4u<<20),
              "adapt real: hottest primary resident excluded, two bounded swaps remain disjoint");
        check(primary.loan_dense_invariant(), "adapt real: dense primary slot invariant retained");

        // ---- exclusive-residency regression (2026-11-01 ownership audit): a multi-round
        // timeline of observe -> before_window (submit) -> primary traffic DURING the pending
        // refill -> drain, asserting after EVERY stage that (a) no (layer,expert) is resident
        // on both cards, (b) no primary-resident expert carries a live owned flag, (c) the
        // owned table equals the peer's actual residency exactly (no phantom rows), and (d)
        // the primary keeps its dense slot invariant (usable capacity, no leaked slots).
        {
            std::vector<int32_t> pres, qres;
            auto disjoint = [&](const char* stage) {
                primary.residency(1, 512, pres);
                peer.residency(1, 512, qres);
                int64_t np = 0, nq = 0, both = 0, table_bad = 0;
                for (int32_t e = 0; e < 512; ++e) {
                    np += pres[e] >= 0; nq += qres[e] >= 0;
                    if (pres[e] >= 0 && qres[e] >= 0) ++both;              // two GPU residents
                    if (pres[e] >= 0 && owned[(size_t) e]) ++both;         // resident + owned
                    if ((owned[(size_t) e] != 0) != (qres[e] >= 0)) ++table_bad;
                }
                const bool ok = both == 0 && table_bad == 0 && np <= 80 && primary.loan_dense_invariant();
                if (!ok)
                    std::printf("ownership timeline (%s): primary=%lld peer=%lld overlaps=%lld "
                                "table-mismatch=%lld dense=%d\n", stage, (long long) np,
                                (long long) nq, (long long) both, (long long) table_bad,
                                (int) primary.loan_dense_invariant());
                return ok;
            };
            check(disjoint("steady"), "ownership timeline: disjoint at rest");
            std::vector<float> rout((size_t) 3 * 10 * n);
            bool ran = true;
            for (int round = 0; round < 6 && ran; ++round) {
                const int32_t fresh = 300 + round;   // never routed on either card before
                for (int i = 0; i < 16; ++i) adapt.observe(0, &fresh, 1);
                ran = adapt.before_window(err);
                check(ran, "ownership timeline: window submitted");
                if (!ran) break;
                check(!primary.holds(0, fresh),
                      "ownership timeline: pending incoming not admitted while in flight");
                // churn the primary mid-flight: admissions + evictions under the pending refill
                std::vector<int32_t> churn(30);
                for (int r = 0; r < 30; ++r) churn[(size_t) r] = 200 + r;
                ran = primary.run_multi(0, x.data(), churn.data(), 3, 10, rout.data(), nullptr);
                check(ran, "ownership timeline: pending-window churn ran");
                if (!ran) break;
                check(disjoint("pending"), "ownership timeline: disjoint while a refill is in flight");
                ran = adapt.drain(err);
                check(ran, "ownership timeline: refill drained");
                if (!ran) break;
                check(disjoint("landed"), "ownership timeline: disjoint after publication");
            }
        }
        // ---- the runtime tripwire, on its own tier (the corruption must not poison the main
        // one): a live ownership row dropped behind the table's back makes the peer-owned
        // expert primary-admissible, and before_window's snapshot check must report the
        // two-card overlap loudly instead of letting a duplicate compute silently
        {
            Exl3Peer peer2;
            std::vector<uint8_t> owned2(512, 0); owned2[0] = owned2[1] = 1;
            strata::kernels::Exl3ExpertStream primary2(model, n, 1, 80, nullptr, 0, 0);
            primary2.set_owned_elsewhere(owned2.data(), 1, 512);
            check(peer2.open(1, n, {entry(0), entry(1)}, 8, 10, err), "ownership tripwire: open");
            strata::kernels::Exl3PeerAdapt::Options o2;
            o2.every = 1; o2.max_swaps = 1; o2.byte_cap = 2u << 20;
            strata::kernels::Exl3PeerAdapt adapt2(peer2, primary2, model, 1, 512, owned2, o2);
            check(adapt2.before_window(err), "ownership tripwire: the first window does not plan");
            owned2[1] = 0;   // the table loses a live row; the peer still holds expert 1
            const int32_t leaked = 1;
            const bool ran = primary2.run_multi(0, x.data(), &leaked, 1, 1, got.data(), nullptr);
            check(ran && primary2.holds(0, 1),
                  "ownership tripwire: the dropped row became primary-admissible (both cards hold it)");
            check(!adapt2.before_window(err) && err.find("overlap") != std::string::npos,
                  "ownership tripwire: before_window reports primary/peer overlap");
        }
        // ---- a failed tier must not leave a stuck exclusion: fault the in-flight refill, the
        // tier goes sticky-failed, the next window clears owned/incoming, and the primary
        // re-admits a formerly peer-owned expert (full capacity, no scratch-only purgatory)
        {
            int32_t former = -1;
            for (int32_t e = 0; e < 512 && former < 0; ++e) if (owned[(size_t) e]) former = e;
            check(former >= 0, "ownership failure: a peer-owned expert exists before the fault");
            const int32_t hot3 = 320;
            for (int i = 0; i < 16; ++i) adapt.observe(0, &hot3, 1);
            setenv("STRATA_EXL3_PEER_FAULT_REFILL", "2", 1);
            const bool submitted = adapt.before_window(err);   // replace() faults mid-copy
            unsetenv("STRATA_EXL3_PEER_FAULT_REFILL");
            check(!submitted && !peer.valid(),
                  "ownership failure: refill fault fails the window and the tier");
            check(adapt.before_window(err), "ownership failure: the next window proceeds on the dead tier");
            bool cleared = true;
            for (int32_t e = 0; e < 512; ++e) cleared = cleared && owned[(size_t) e] == 0;
            check(cleared, "ownership failure: the exclusion table is cleared with the tier");
            const bool primed = primary.run_multi(0, x.data(), &former, 1, 1, got.data(), nullptr);
            check(primed && primary.holds(0, former),
                  "ownership failure: formerly peer-owned expert admitted on the primary");
        }
    }

    // Pooled planes preserve row bytes, refill capacities and transactional cleanup. Keep the
    // original allocator/fault matrix above unchanged and explicitly exercise this opt-in arm.
    {
        auto a = make_expert(8, 8, 3, 2, 711), b = make_expert(8, 8, 4, 2, 713);
        const char* old = std::getenv("STRATA_EXL3_PEER_ARENA");
        const bool had_old = old != nullptr;
        const std::string saved = old ? old : "";
        std::vector<float> x(128, 0.125f), flat(256), pooled(256);
        int32_t ids[2] = {0, 1}, kind[2] = {2, 2};
        Exl3Peer p;
        unsetenv("STRATA_EXL3_PEER_ARENA");
        check(p.open(1, 128, {a.entry(0, 0), b.entry(0, 1)}, 1, 2, err) &&
              p.launch(0, x.data(), ids, 1, 2, kind, err) && p.finish(flat.data(), err),
              "peer arena: independent-plane oracle");
        const size_t flat_bytes = p.bytes();
        p.close();
        setenv("STRATA_EXL3_PEER_ARENA", "1", 1);
        check(p.open(1, 128, {a.entry(0, 0), b.entry(0, 1)}, 1, 2, err) && p.bytes() == flat_bytes &&
              p.launch(0, x.data(), ids, 1, 2, kind, err) && p.finish(pooled.data(), err) &&
              std::memcmp(flat.data(), pooled.data(), flat.size() * sizeof(float)) == 0,
              "peer arena: mixed-K FFN rows byte-identical");
        check(p.enable_adaptation(1, 0, err), "peer arena: refill setup");
        Exl3Peer::Replacement r; r.out = 1; r.in = b.entry(0, 2);
        std::vector<Exl3Peer::Replacement> landed;
        check(p.replace({r}, err) && p.apply_refill(true, landed, err) && p.has(0, 2) && !p.has(0, 1),
              "peer arena: replacement publishes borrowed plane addresses");
        ids[1] = 2;
        check(p.launch(0, x.data(), ids, 1, 2, kind, err) && p.finish(pooled.data(), err) &&
              std::memcmp(flat.data(), pooled.data(), flat.size() * sizeof(float)) == 0,
              "peer arena: replacement preserves row bytes");
        p.close();
        check(!p.open(1, 128, {a.entry(0, 0)}, 1, 1, err, SIZE_MAX) && !p.valid() && p.bytes() == 0 &&
              err.find("reserve") != std::string::npos, "peer reserve: impossible floor rejects before allocation");
        setenv("STRATA_EXL3_PEER_FAULT", "16", 1);
        check(!p.open(1, 128, {a.entry(0, 0)}, 1, 1, err) && !p.valid() && p.bytes() == 0,
              "peer arena: partial plane copy failure unwinds single owner");
        unsetenv("STRATA_EXL3_PEER_FAULT");
        check(p.open(1, 128, {a.entry(0, 0)}, 1, 1, err), "peer arena: reopen after failure");
        p.close();
        if (had_old) setenv("STRATA_EXL3_PEER_ARENA", saved.c_str(), 1);
        else unsetenv("STRATA_EXL3_PEER_ARENA");
    }

    // Resource/validation seam: reject invalid batches before withdrawal and make a partial
    // upload failure sticky, including close() while the refill stream still has work.
    {
        auto a=make_expert(8,8,3,2,101), b=make_expert(8,8,3,2,303), big=make_expert(8,8,4,2,505);
        Exl3Peer p;
        check(p.open(1,128,{a.entry(0,0)},1,1,err) && p.enable_adaptation(1,0,err), "adapt synthetic: setup");
        Exl3Peer::Replacement r; r.out=0; r.in=big.entry(0,1);
        check(!p.can_replace(r) && !p.replace({r},err) && p.has(0,0) && !p.refilling(),
              "adapt synthetic: larger planes rejected before mutation");
        r.in=b.entry(0,1);
        check(!p.replace({r,r},err) && p.has(0,0), "adapt synthetic: batch cap/duplicates rejected");
        setenv("STRATA_EXL3_PEER_FAULT_REFILL","2",1);
        check(!p.replace({r},err) && p.failed() && !p.has(0,0) && !p.has(0,1),
              "adapt synthetic: partial upload failure advertises no residents");
        unsetenv("STRATA_EXL3_PEER_FAULT_REFILL");
        p.close(); check(!p.valid() && !p.refilling(), "adapt synthetic: close drains and clears refill");
        Exl3Peer q;
        check(q.open(1,128,{a.entry(0,0)},1,1,err) && q.enable_adaptation(1,1,err) &&
              !q.replace({r},err) && q.has(0,0) && !q.refilling(),
              "adapt synthetic: byte cap rejects oversized batch without withdrawal");
    }

    // ---- lend_tail_preview (STRATA_EXL3_RESIDENT=peer's borrow): the preview must mirror
    // lend_experts' selection exactly - same tail order, same quarter cap, same skips - and
    // mutate nothing.  The engine copies the previewed experts' planes GPU->RAM between the
    // preview and the lend; a divergence would free planes with no host copy (the post-lend
    // invariant in generate.cpp catches that loudly; this pins the selection itself).
    {
        auto e0 = make_expert(8, 8, 3, 2, 911), e1 = make_expert(8, 8, 3, 2, 913),
             e2 = make_expert(8, 8, 3, 2, 917), e3 = make_expert(8, 8, 3, 2, 919),
             e4 = make_expert(8, 8, 3, 2, 923), e5 = make_expert(8, 8, 3, 2, 929),
             e6 = make_expert(8, 8, 3, 2, 931), e7 = make_expert(8, 8, 3, 2, 937);
        Exl3Peer p;
        const char* old_arena = std::getenv("STRATA_EXL3_PEER_ARENA");
        const std::string saved_arena = old_arena ? old_arena : "";
        unsetenv("STRATA_EXL3_PEER_ARENA");   // the loan refuses the pooled mode
        check(p.open(1, 128, {e0.entry(0, 0), e1.entry(0, 1), e2.entry(0, 2), e3.entry(0, 3),
                              e4.entry(1, 0), e5.entry(1, 1), e6.entry(1, 2), e7.entry(1, 3)},
                     1, 2, err), "loan preview: open 8 experts over 2 layers");
        std::vector<std::pair<int32_t, int32_t>> preview;
        check(p.lend_tail_preview(3, preview) && preview.size() == 2,
              "loan preview: the quarter cap (8/4 = 2) bounds the request");
        check(p.lent_experts() == 0 && p.has(1, 3) && p.has(1, 2) && p.resident() == 8,
              "loan preview: mutates nothing (all 8 still resident)");
        check(p.lend_experts(3, err), "loan preview: the lend itself succeeds");
        const auto lent = p.lent_list();
        check(lent.size() == preview.size() && lent.size() == 2 &&
              lent[0] == preview[0] && lent[1] == preview[1],
              "loan preview: the lend freed exactly the previewed experts, same order");
        check(p.lend_tail_preview(3, preview) && preview.empty(),
              "loan preview: at the cap nothing further is previewed");
        // restore, then the cycle again (the engine's per-prompt rhythm)
        std::vector<Exl3Peer::Entry> sources;
        for (const auto& key : lent)
            sources.push_back(key.first == 0
                ? (key.second == 2 ? e2.entry(0, 2) : e3.entry(0, 3))
                : (key.second == 3 ? e7.entry(1, 3) : e6.entry(1, 2)));
        check(p.restore_lent(sources, err) && p.lent_experts() == 0 && p.has(1, 3) && p.has(1, 2),
              "loan preview: restore returns the loan to resident");
        check(p.lend_tail_preview(1, preview) && preview.size() == 1 &&
              p.lend_experts(1, err) && p.lent_list()[0] == preview[0],
              "loan preview: the cycle repeats after a restore");
        p.close();
        if (!saved_arena.empty()) setenv("STRATA_EXL3_PEER_ARENA", saved_arena.c_str(), 1);
    }

    std::printf("exl3_peer_test: %s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
