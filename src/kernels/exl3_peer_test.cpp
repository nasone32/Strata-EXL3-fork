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

HostExpert make_expert(int ki, int nj, int bits, int cb, uint32_t seed) {
    HostExpert h;
    auto fill = [&](int slot, int k, int n, uint32_t sd) {
        h.buf[slot + 0] = pattern((size_t) k * (size_t) n * (size_t) (256 * bits / 16) * 2, sd);
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
            const size_t tr = (size_t) host[p].ki * (size_t) host[p].nj * (size_t) (256 * host[p].bits / 16) * 2;
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
                // must happen before any read (a deref would crash here), allocation or enqueue
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
                          free_vram() == base_free,
                          "open precheck: ki not multiple of 8 (fake pointers, no read/alloc)");
                    rel(160, 39);   // hidden width not multiple of 8
                    Exl3Peer p2;
                    check(!p2.open(1, 2560, bad, max_tokens, max_k, err) && !p2.valid() &&
                          free_vram() == base_free,
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
            const char* mr = std::getenv("STRATA_EXL3_PEER_MULTIROW");
            const bool extra_pinned = mr && std::atoi(mr) > 0 && strata::kernels::exl3_fused_multirow_supported();
            const char* pack = std::getenv("STRATA_EXL3_PEER_PACK_PLAN");
            const bool packed_plan = extra_pinned && pack && pack[0] && pack[0] != '0';
            const int expected_sites = 11 + 18 * 8 + (extra_pinned ? 2 : 0) + (packed_plan ? 2 : 0);
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
                    m.ki = l.ki; m.nj = l.nj; m.bits = l.bits; m.cb = l.cb;
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
                return Exl3Mat{l.trellis, l.suh, l.svh, l.ki, l.nj, l.bits, l.cb};
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

    std::printf("exl3_peer_test: %s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
