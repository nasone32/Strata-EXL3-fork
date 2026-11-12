// src/kernels/exl3_band8_joint_partition.cu - the JOINT CPU + 2-GPU offload service-unit bench.
//
// PROTOTYPE, isolated like exl3_band8_gpu_parity: includes the production CUDA/HIP TU
// (cuda/exl3.cu) for the grouped GEMV kernels and links the REAL CPU service TU
// (src/kernels/cpu/exl3_moe.cpp, freshly compiled against the 48-byte Exl3CpuMatrix ABI).
// No engine, loader, cache, routing or default changes; this file is built only inside the
// STRATA_EXPERIMENTAL_BAND8_TEST focused branch (EXCLUDE_FROM_ALL).
//
// What one "service unit" measures (model-independent, representative window shapes only):
//   - ONE host canonical corpus in ONE trellis layout for its whole lifetime (packed band8 by
//     default, native with --layout native): 32 experts x MiMo 4096/2048 at a REAL layer rate
//     (K2 or K2.5 - layers are uniform per the safetensors header survey). The native plane
//     used for packing is freed immediately after each expert packs; there is never a full
//     native host shadow. Scales ride a separate vec, uploaded once.
//   - TWO GPUs with DISJOINT expert ownership (e % 2): each device pre-reserves one
//     corpus-sized buffer plus its own stream + Exl3MoeWorkspace; only the owned half of each
//     buffer ever holds bytes (no expert is resident on both cards). GPU0 = primary, GPU1 = peer.
//   - Windows: T1 k8 (8 unique experts, 1 row each) and DFlash T4 k8 (32 routes over 24
//     unique experts: 8 two-row + 16 one-row groups), rotating over the corpus with a fixed
//     plan; t4worst (4 rows of one expert) is the K2 4-row subset probe. A group is a WHOLE
//     expert: the CPU never duplicates GPU compute.
//   - Cold/staging: the first `--cold` GPU0-OWNED experts of a window are treated as misses.
//     The joint arms stage them EVERY iteration with REAL cudaMemcpyAsync (pageable host
//     canonical -> GPU0 packed slot) INSIDE the timed region, before GPU0 compute. The device
//     slot may already hold correct bytes: the copy is forced to count the bytes, so this arm
//     is labeled SYNTHETIC STAGING, not a cache policy. Scales stay resident (copied once);
//     staging bytes = trellis planes only. `--cpu S` moves the first S of those cold groups to
//     the CPU tier and avoids those copies (S <= cold; never GPU1-peer-owned experts).
//     `--steal` is the optional separate label where the CPU takes RESIDENT GPU0 experts.
//   - Orchestration order (timed): GPU1 enqueue -> CPU run_rows_bg submit -> GPU0 staging H2D
//     + grouped FFN -> F16 widen of GPU0 rows -> CPU join -> peer result D2H(pinned)/H2D +
//     widen -> CPU F32 rows H2D directly into their parts rows -> deterministic
//     route-ordered 1/k token sum -> stream sync. Overlap is ACTUAL: wall is compared against
//     the per-arm event spans (each timed on its own stream, non-additive).
//   - Numerics: GPU-kept rows must be BIT-EXACT against an all-GPU resident baseline run with
//     the SAME forced parts; CPU rows must be byte-exact against a standalone blocking CPU job
//     with the identical participants; CPU (int8) vs GPU (half) rows are held to the EXISTING
//     real-model parity gate (rel RMS < 3%, rel max < 12%; that reference measures ~0.9% RMS).
//     The band8-vs-native CPU byte proof is the existing 39-check battery, not re-run here.
//
// Scope guard: --smoke is the audited checkpoint ONLY. The full matrix (2 layouts x T1/T4 x
// cold 0/2/4 x CPU 0/1/2/4 x W 8/12) runs only after the personal audit.
#include "cuda/exl3.cu"
#include "strata/kernels/cpu/exl3_moe.hpp"
#include "kernels/cpu/exl3_moe_fixture.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(STRATA_WMMA_GFX11)
int main() { std::puts("joint partition bench needs a gfx11 WMMA build"); return 77; }
#else

using strata::kernels::cpu::Exl3CpuMatrix;
using strata::kernels::cpu::Exl3CpuMoe;
using strata::kernels::cpu::Exl3CpuTrellisLayout;
using namespace strata::kernels;
using Clock = std::chrono::steady_clock;

namespace {

void jck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
int jtrace() { static const int t = std::getenv("PARTRACE") ? 1 : 0; return t; }
#define JTR(x) do { if (jtrace()) { std::fprintf(stderr, "TRACE %s\n", x); std::fflush(stderr); } } while (0)
void jreq(bool x, const char* s) { if (!x) throw std::runtime_error(s); }
uint16_t jhalf(float x) { return __half_as_ushort(__float2half_rn(x)); }
float jreal(uint16_t x) { return __half2float(__ushort_as_half(x)); }
double jms(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
double jus(Clock::time_point t) { return std::chrono::duration<double, std::micro>(Clock::now() - t).count(); }
double jpct(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[(size_t) std::min<double>((double) (v.size() - 1), std::ceil(p * (v.size() - 1)))];
}

// Device-tagged buffer: every operation runs with `dev` current, so two cards never alias.
template <class T> struct JDevBuf {
    int dev = 0; T* p = nullptr; size_t count = 0;
    JDevBuf() = default;
    JDevBuf(int d, size_t n) : dev(d), count(n) {
        int prev = -1; jck(cudaGetDevice(&prev)); jck(cudaSetDevice(d));
        jck(cudaMalloc((void**) &p, n * sizeof(T)));
        jck(cudaSetDevice(prev));
    }
    ~JDevBuf() {
        if (p) { int prev = -1; if (cudaGetDevice(&prev) == cudaSuccess) { (void) cudaSetDevice(dev); (void) cudaFree(p); (void) cudaSetDevice(prev); } }
    }
    JDevBuf(const JDevBuf&) = delete; JDevBuf& operator=(const JDevBuf&) = delete;
    void put(const std::vector<T>& v) {
        jreq(v.size() <= count, "devbuf upload capacity");
        int prev = -1; jck(cudaGetDevice(&prev)); jck(cudaSetDevice(dev));
        jck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
        jck(cudaSetDevice(prev));
    }
};

template <class T> struct JPinBuf {
    T* p = nullptr; size_t count = 0;
    explicit JPinBuf(size_t n) : count(n) { jck(cudaHostAlloc((void**) &p, n * sizeof(T), cudaHostAllocDefault)); }
    ~JPinBuf() { if (p) (void) cudaFreeHost(p); }
    JPinBuf(const JPinBuf&) = delete; JPinBuf& operator=(const JPinBuf&) = delete;
};

// Benchmark-only merge helpers, bounds-checked: widen F16 rows into F32 parts at mapped rows,
// and the deterministic route-ordered 1/k token sum (ascending j, one multiply at the end).
__global__ void jw_widen_scatter(const uint16_t* __restrict__ src, const int* __restrict__ row_map,
                                 int rows, int n, float* __restrict__ parts, int parts_rows) {
    const int idx = blockIdx.x * (int) blockDim.x + threadIdx.x;
    if (idx >= rows * n) return;
    const int r = idx / n, c = idx - r * n;
    const int dst = row_map[r];
    if (dst < 0 || dst >= parts_rows) return;             // bounded: a bad map row never writes
    parts[(size_t) dst * n + c] = __half2float(__ushort_as_half(src[(size_t) r * n + c]));
}
__global__ void jw_row_sum(const float* __restrict__ parts, const int* __restrict__ token_route_row,
                           int tokens, int k, int routes_cap, int n, float w, float* __restrict__ out) {
    const int idx = blockIdx.x * (int) blockDim.x + threadIdx.x;
    if (idx >= tokens * n) return;
    const int t = idx / n;
    float acc = 0.f;
    for (int j = 0; j < k; ++j) {                          // deterministic ascending route order
        const int rr = token_route_row[t * k + j];
        if (rr < 0 || rr >= routes_cap) continue;
        acc += parts[(size_t) rr * n + (idx - t * n)];
    }
    out[idx] = w * acc;
}

struct JGeo { const char* name; int k, ff, bits, hb; };    // UNIFORM trio (real MiMo layers are uniform)
const JGeo kJGeos[] = { {"k2", 4096, 2048, 2, 0}, {"k25", 4096, 2048, 2, 1} };

// One window over the corpus: group-major route order, token map, token-major sum table.
struct JWin {
    const char* pattern = "";
    int first_expert = 0, groups = 0, routes = 0, widest = 1, tokens = 1, k_top = 1;
    std::vector<int> experts, rows, first;      // per group
    std::vector<int> input_rows;                // route -> token
    std::vector<int> token_route_row;           // [t * k_top + j] -> route
};

struct JCorpus {                               // ONE host canonical layout, whole lifetime
    const JGeo* G = nullptr; int experts = 0; bool packed = false;
    size_t wg = 0, wu = 0, wd = 0, per_expert = 0;   // u16 words
    std::vector<uint16_t> tre;                 // PAGEABLE host canonical (the RAM source)
    std::vector<uint16_t> scales;
    double pack_ms = 0;
    size_t plane_off(int e, int plane) const { // 0 gate 1 up 2 down
        const size_t base = (size_t) e * per_expert;
        return plane == 2 ? base + wg + wu : plane == 1 ? base + wg : base;
    }
    size_t scale_off(int e, int plane) const { return ((size_t) e * 3 + plane) * (size_t) (G->k + G->ff); }
};

// GPU-proto fixture order (BenchCorpus): per expert gate/up/down trellis words from
// mt19937(20261106 + geo), then all scales +- (0.10 + rng%32*0.001). The native plane used by
// the packer is a per-expert temporary, freed right after its pack - never a corpus shadow.
void jcorpus_build(JCorpus& c, const JGeo& g, int experts, bool packed, int geo_idx) {
    c.G = &g; c.experts = experts; c.packed = packed;
    const int pw = 16 * g.bits + 8 * g.hb;
    c.wg = (size_t) (g.k / 16) * (g.ff / 16) * pw;
    c.wu = c.wg;
    c.wd = (size_t) (g.ff / 16) * (g.k / 16) * pw;
    c.per_expert = c.wg + c.wu + c.wd;
    std::mt19937 rng(20261106 + geo_idx);
    c.tre.assign((size_t) experts * c.per_expert, 0);
    std::vector<uint16_t> native;
    const int dk[3] = { g.k, g.k, g.ff }, dn[3] = { g.ff, g.ff, g.k };
    const auto pack0 = Clock::now();
    for (int e = 0; e < experts; ++e)
        for (int plane = 0; plane < 3; ++plane) {
            const size_t words = (size_t) (dk[plane] / 16) * (dn[plane] / 16) * pw;
            native.assign(words, 0);
            for (auto& v : native) v = (uint16_t) rng();
            uint16_t* dst = c.tre.data() + c.plane_off(e, plane);
            if (packed)
                strata::kernels::cpu::exl3cpu_fixture_pack_band8(dst, native.data(),
                                                                 dk[plane], dn[plane], g.bits, g.hb);
            else
                std::memcpy(dst, native.data(), words * 2);
            native.clear();                                  // temporary native plane freed at once
            native.shrink_to_fit();
        }
    c.pack_ms = jms(pack0);
    c.scales.resize((size_t) experts * 3 * (g.k + g.ff));
    for (auto& v : c.scales) v = jhalf((0.10f + (rng() % 32) * 0.001f) * (rng() % 2 ? 1 : -1));
}

// Fixed window plans over the corpus.
JWin jmake_t1(const JCorpus& c, int first) {
    JWin w; w.pattern = "t1"; w.first_expert = first; w.groups = 8; w.routes = 8; w.widest = 1;
    w.tokens = 1; w.k_top = 8;
    w.experts.resize(8); w.rows.assign(8, 1); w.first = {0,1,2,3,4,5,6,7};
    for (int i = 0; i < 8; ++i) w.experts[(size_t) i] = (first + i) % c.experts;
    w.input_rows.assign(8, 0);
    w.token_route_row.resize(8); for (int i = 0; i < 8; ++i) w.token_route_row[(size_t) i] = i;
    return w;
}
JWin jmake_t4(const JCorpus& c, int first, std::mt19937& rr) {
    JWin w; w.pattern = "t4"; w.first_expert = first; w.groups = 24;
    w.rows.assign(24, 1); w.experts.resize(24);
    for (int i = 0; i < 24; ++i) w.experts[(size_t) i] = (first + i) % c.experts;
    std::vector<int> idx(24); for (int i = 0; i < 24; ++i) idx[(size_t) i] = i;
    std::shuffle(idx.begin(), idx.end(), rr);
    for (int i = 0; i < 8; ++i) w.rows[(size_t) idx[(size_t) i]] = 2;   // 8 two-row + 16 one-row
    w.routes = 32; w.widest = 2; w.tokens = 4; w.k_top = 8;
    int prefix = 0; w.first.clear();
    for (int g = 0; g < 24; ++g) { w.first.push_back(prefix); prefix += w.rows[(size_t) g]; }
    w.input_rows.resize(32); for (int r = 0; r < 32; ++r) w.input_rows[(size_t) r] = r % 4;
    // token-major table: token t owns routes r with r % 4 == t, ascending (deterministic order)
    std::vector<std::vector<int>> by_token(4);
    for (int r = 0; r < 32; ++r) by_token[(size_t) (r % 4)].push_back(r);
    w.token_route_row.clear();
    for (int t = 0; t < 4; ++t) for (int j = 0; j < 8; ++j) w.token_route_row.push_back(by_token[(size_t) t][(size_t) j]);
    return w;
}
JWin jmake_worst(const JCorpus& c, int expert) {
    JWin w; w.pattern = "t4worst"; w.first_expert = expert; w.groups = 1; w.rows = {4}; w.first = {0};
    w.experts = {expert}; w.routes = 4; w.widest = 4; w.tokens = 4; w.k_top = 1;
    w.input_rows = {0, 1, 2, 3};
    w.token_route_row = {0, 1, 2, 3};
    return w;
}

// One device's descriptor subset for a window variant. Host mirrors stay alive for the run.
struct JDevPlan {
    std::vector<Exl3Mat> hg, hu, hd;
    std::vector<int> first, rows, input_rows, route_row;   // route_row: local route -> GLOBAL route
    int groups = 0, routes = 0, rates[3] = {0,0,0}, parts[3] = {0,0,0};
    std::unique_ptr<JDevBuf<Exl3Mat>> dg, du, dd;
    std::unique_ptr<JDevBuf<int>> dfirst, drows, dinput, droute;
};

struct JGpu {                                   // per-device context: stream + workspace + buffers
    int dev = 0; bool band8 = true; cudaStream_t st = nullptr;
    std::unique_ptr<Exl3MoeWorkspace> ws;
    std::unique_ptr<JDevBuf<uint16_t>> corpus, scales, x16, out16, stage16;
    std::unique_ptr<JDevBuf<float>> parts, sum;             // GPU0 only
    // GPU0 only, PER-WINDOW merge maps (a shared buffer uploaded once would serve window 1's
    // map to window 0 - the t4 windows genuinely differ):
    //   sum_map[w]:  token-major route table of window w for the 1/k sum
    //   peer_map[w]: the PEER plan's local->global route map of window w, mirrored on dev0
    //                (a dev1 pointer is unreadable from a dev0 kernel without peer mapping)
    std::vector<std::unique_ptr<JDevBuf<int>>> sum_map, peer_map;
    ~JGpu() {
        if (!st) return;
        int prev = -1; if (cudaGetDevice(&prev) == cudaSuccess) {
            (void) cudaSetDevice(dev);
            (void) cudaStreamSynchronize(st);   // drain before destroy: the fault-injection
            (void) cudaStreamDestroy(st);       // unwind frees pinned staging while the peer's
            (void) cudaSetDevice(prev);         // D2H may still reference it
        }
    }
};

using JDevFn = uint16_t* (*)(const Exl3Mat*, const Exl3Mat*, const Exl3Mat*, const int*, const int*,
                             int, int, int, const uint16_t*, cudaStream_t, Exl3MoeWorkspace&, long,
                             const int*, int, int, int, int, bool, uint16_t*, int, int, int, int, int, int);

Exl3Mat jmat(const JCorpus& c, const JGpu& g, int e, int plane) {
    const JGeo& G = *c.G;
    const int in = plane == 2 ? G.ff : G.k, out = plane == 2 ? G.k : G.ff;
    Exl3Mat m{};
    m.trellis = g.corpus->p + c.plane_off(e, plane);
    m.suh = g.scales->p + c.scale_off(e, plane);
    m.svh = m.suh + in;
    m.ki = in / 16; m.nj = out / 16; m.bits = G.bits; m.cb = 2; m.frac = G.hb;
    return m;
}

// Build + upload one device subset; forced parts validated against the LDS budget (the same
// check win_plan_upload applies) so every arm of a comparison launches the SAME kernel shape.
void jplan_build(JDevPlan& p, const JCorpus& c, const JGpu& g, const JWin& w,
                 const std::vector<int>& group_subset, int forced, bool fused) {
    p.groups = (int) group_subset.size();
    p.hg.clear(); p.hu.clear(); p.hd.clear(); p.first.clear(); p.rows.clear();
    p.input_rows.clear(); p.route_row.clear();
    p.rates[0] = p.rates[1] = p.rates[2] = 0;
    p.parts[0] = p.parts[1] = p.parts[2] = 0;
    int prefix = 0;
    for (int gi : group_subset) {
        p.hg.push_back(jmat(c, g, w.experts[(size_t) gi], 0));
        p.hu.push_back(jmat(c, g, w.experts[(size_t) gi], 1));
        p.hd.push_back(jmat(c, g, w.experts[(size_t) gi], 2));
        p.first.push_back(prefix);
        p.rows.push_back(w.rows[(size_t) gi]);
        prefix += w.rows[(size_t) gi];
        for (int j = 0; j < w.rows[(size_t) gi]; ++j) {
            const int route = w.first[(size_t) gi] + j;
            p.input_rows.push_back(w.input_rows[(size_t) route]);
            p.route_row.push_back(route);
        }
    }
    p.routes = prefix;
    if (prefix == 0) {                       // an EMPTY subset (e.g. t4worst's idle peer) is
        p.dg.reset(); p.du.reset(); p.dd.reset();   // legal: no plan, no buffers, nothing launched
        p.dfirst.reset(); p.drows.reset(); p.dinput.reset(); p.droute.reset();
        return;
    }
    jreq(exl3_native_grouped_plan(p.hg.data(), p.hu.data(), p.hd.data(), p.groups, w.widest,
                                  true, fused, p.rates, p.parts), "joint plan refused");
    if (forced) {
        const int ki[3] = { p.hg[0].ki, p.hu[0].ki, p.hd[0].ki };
        for (int b = 0; b < 3; ++b) {
            jreq(forced <= ki[b] &&
                 native_grouped_widest(ki[b], forced) * w.widest * 2 +
                     (fused ? kNativeGroupedStaticFused : 0) <= kNativeGroupedLdsBudget,
                 "forced parts LDS budget");
            p.parts[b] = forced;
        }
    }
    p.dg.reset(new JDevBuf<Exl3Mat>(g.dev, (size_t) p.groups));
    p.du.reset(new JDevBuf<Exl3Mat>(g.dev, (size_t) p.groups));
    p.dd.reset(new JDevBuf<Exl3Mat>(g.dev, (size_t) p.groups));
    p.dfirst.reset(new JDevBuf<int>(g.dev, (size_t) p.groups));
    p.drows.reset(new JDevBuf<int>(g.dev, (size_t) p.groups));
    p.dinput.reset(new JDevBuf<int>(g.dev, (size_t) p.routes));
    p.droute.reset(new JDevBuf<int>(g.dev, (size_t) p.routes));
    p.dg->put(p.hg); p.du->put(p.hu); p.dd->put(p.hd);
    p.dfirst->put(p.first); p.drows->put(p.rows);
    p.dinput->put(p.input_rows); p.droute->put(p.route_row);
}

struct JCellCfg {
    const char* geo = "k2"; const char* pattern = "t1";
    int cold = 0, cpu = 0, workers = 8, iters = 12, warmup = 3, experts = 32, parts = 1;
    int sweeps = 2;
    bool steal = false, packed = true;
};

// Partition metadata for one window - the ownership proof that runs BEFORE any GPU work.
struct JPartition {
    std::vector<int> g0_groups, g1_groups, cpu_groups;       // window group indices per role
    std::vector<int> cold_groups;                            // GPU0-owned miss groups, window order
    std::vector<int> staged_groups;                          // cold minus CPU-taken
    std::vector<int> served_by;                              // route -> 0 G0 / 1 G1 / 2 CPU
    std::vector<int> g0_joint;                               // GPU0's joint compute set
    int cpu_routes = 0;
};
JPartition jpartition(const JWin& w, const JCellCfg& cfg) {
    JPartition p;
    for (int g = 0; g < w.groups; ++g) {
        const bool gpu0 = w.experts[(size_t) g] % 2 == 0;
        (gpu0 ? p.g0_groups : p.g1_groups).push_back(g);
    }
    jreq((int) p.g0_groups.size() >= cfg.cold, "cold exceeds GPU0-owned groups in window");
    for (int i = 0; i < cfg.cold; ++i) p.cold_groups.push_back(p.g0_groups[(size_t) i]);
    if (cfg.steal) {
        // resident-steal label: CPU takes GPU0 RESIDENT experts (never the cold ones)
        std::vector<int> resident;
        for (int g : p.g0_groups)
            if (std::find(p.cold_groups.begin(), p.cold_groups.end(), g) == p.cold_groups.end())
                resident.push_back(g);
        jreq((int) resident.size() >= cfg.cpu, "cpu share exceeds GPU0 resident groups");
        for (int i = 0; i < cfg.cpu; ++i) p.cpu_groups.push_back(resident[(size_t) i]);
        p.staged_groups = p.cold_groups;
    } else {
        jreq(cfg.cpu <= cfg.cold, "cpu share exceeds cold groups (non-steal)");
        for (int i = 0; i < cfg.cpu; ++i) p.cpu_groups.push_back(p.cold_groups[(size_t) i]);
        for (int i = cfg.cpu; i < cfg.cold; ++i) p.staged_groups.push_back(p.cold_groups[(size_t) i]);
    }
    for (int g : p.g0_groups)
        if (std::find(p.cpu_groups.begin(), p.cpu_groups.end(), g) == p.cpu_groups.end())
            p.g0_joint.push_back(g);
    p.served_by.assign((size_t) w.routes, -1);
    auto mark = [&](const std::vector<int>& gs, int who) {
        for (int g : gs) for (int j = 0; j < w.rows[(size_t) g]; ++j) {
            const int r = w.first[(size_t) g] + j;
            jreq(p.served_by[(size_t) r] == -1, "route assigned twice");
            p.served_by[(size_t) r] = who;
        }
    };
    mark(p.g0_joint, 0); mark(p.g1_groups, 1); mark(p.cpu_groups, 2);
    for (int who : p.served_by) jreq(who >= 0, "route left unassigned");
    for (int g : p.cpu_groups) jreq(w.experts[(size_t) g] % 2 == 0, "CPU took a peer-owned expert");
    for (int g : p.cpu_groups) p.cpu_routes += w.rows[(size_t) g];
    return p;
}

// Per-iteration metrics. `cpu_us` is SUBMISSION-TO-JOIN latency (completion OBSERVED by the
// caller), measured from just after run_rows_bg returns (i.e. AFTER the GPU1 enqueue block) to
// the join's return: it spans the host's own GPU0 staging/FFN/widen enqueues - including any
// pageable-H2D host blocking - plus the residual CPU wait, and the job may have finished
// before the join was reached. It is an UPPER BOUND on CPU service, never CPU-hardware
// compute time; policy budgeting should use the join residual and the end wall.
// `join_us` is the residual wait inside run_rows_join itself.
struct JIter { double wall_ms = 0, submit_us = 0, join_us = 0, cpu_us = 0, g0_ms = 0, g1_ms = 0; };

// RAII drain for a background CPU job: if anything between run_rows_bg and run_rows_join
// throws (a GPU enqueue/copy error unwinding this stack), the selection tables ids/t/r and the
// pinned in/out buffers MUST stay alive until the pool has stopped writing them. Declare this
// guard AFTER those locals: it unwinds first, joins the job while they are still alive, and
// never swallows the in-flight exception (the join itself is best-effort in the destructor).
struct JBgpDrain {
    Exl3CpuMoe* svc = nullptr; bool pending = false;
    ~JBgpDrain() {
        if (pending) {
            pending = false;
            try { svc->run_rows_join(); } catch (...) {}   // keep the original exception
        }
    }
};
// Scoped GPU drain: declared FIRST in jrun_unit (destroyed LAST, after JBgpDrain has joined
// the CPU job and after the selection tables freed), so an exception anywhere in the unit
// still synchronizes BOTH device-owned streams BEFORE control returns to jrun_cell/caller,
// where pinned staging (g1's queued D2H target) and device buffers would be freed during
// outer unwinding. The bench has no device-side spin kernels, and the CPU job is joined
// FIRST, so these host stream syncs cannot deadlock on the pool. Disabled on the normal path
// right after the final g0 stream sync (the g1 done event was already host-confirmed), so a
// normal iteration pays NO extra synchronization. A LOST GPU context is not recovery-tested:
// the guarantee is unwind safety up to process exit, not context repair. Best-effort only -
// the original exception is preserved.
struct JStreamDrain {
    JGpu* g0 = nullptr; JGpu* g1 = nullptr; bool enabled = false;
    ~JStreamDrain() {
        if (!enabled || !g0 || !g1 || !g0->st || !g1->st) return;
        int prev = -1;
        if (cudaGetDevice(&prev) == cudaSuccess) {
            (void) cudaSetDevice(g1->dev);
            (void) cudaStreamSynchronize(g1->st);
            (void) cudaSetDevice(g0->dev);
            (void) cudaStreamSynchronize(g0->st);
            (void) cudaSetDevice(prev);
        }
    }
};
// Bench-only fault injection (PARFAULT=after_peer_enqueue | after_submit): throws on the host
// at the named point, so the RAII drain paths can be verified (join ran, both streams drained,
// no use after free, process exits with this known error instead of hanging). Never in sweeps.
bool jfault_is(const char* which) {
    const char* f = std::getenv("PARFAULT");
    return f && !std::strcmp(f, which);
}

struct JResult {                                // host copies when validating
    bool have = false;
    std::vector<float> parts, sum;
    std::vector<float> cpu_rows;                // pinned CPU rows snapshot
};

// The whole joint service unit. baseline=true runs the all-GPU resident reference instead
// (no staging, no CPU; GPU0 computes every owned group; the peer merge path is identical).
// validate=true prefills parts with a 0x7f byte sentinel and proves every row was written.
JResult jrun_unit(const JWin& w, const JPartition& part, JGpu& g0, JGpu& g1,
                  const JDevPlan& p0, const JDevPlan& p1, const JCorpus& c, Exl3CpuMoe* svc,
                  float* cpu_x, JPinBuf<float>* cpu_out, JPinBuf<uint16_t>* g1_stage,
                  cudaEvent_t ev_g1_compute[2], cudaEvent_t ev_g1_done,
                  cudaEvent_t ev_g0[2], const JDevBuf<int>& sum_map,
                  const JDevBuf<int>& peer_map, bool baseline, bool validate, JIter* it) {
    const int H = c.G->k, FF = c.G->ff, N = c.G->k;
    const JDevFn dev = g0.band8 ? moe_run_experts_grouped_dev<true> : moe_run_experts_grouped_dev<false>;
    JResult res;
    // FIRST-declared local: on unwind this drains both GPU streams AFTER JBgpDrain has joined
    // the CPU job and BEFORE any outer scope frees the pinned staging or device buffers.
    JStreamDrain sd{&g0, &g1, true};
    const auto t0 = Clock::now();
    // ---- GPU1 (peer) first: grouped FFN + D2H into pinned staging + done event ----
    // (an empty peer plan - a one-expert window like t4worst - launches nothing and the
    //  done event records immediately, so the merge path below is unchanged)
    if (p1.routes > 0) {
        jck(cudaSetDevice(g1.dev));
        JTR("g1 begin");
        if (it) jck(cudaEventRecord(ev_g1_compute[0], g1.st));
        jreq(dev(p1.dg->p, p1.du->p, p1.dd->p, p1.dfirst->p, p1.drows->p, p1.groups, p1.routes,
                 w.widest, g1.x16->p, g1.st, *g1.ws, H, p1.dinput->p, H, FF, N, 2, true,
                 g1.out16->p, p1.rates[0], p1.rates[1], p1.rates[2], p1.parts[0], p1.parts[1],
                 p1.parts[2]) != nullptr, "peer grouped FFN refused");
        jck(cudaGetLastError());
        JTR("g1 ffn enqueued");
        if (it) jck(cudaEventRecord(ev_g1_compute[1], g1.st));
        jck(cudaMemcpyAsync(g1_stage->p, g1.out16->p, (size_t) p1.routes * N * 2,
                            cudaMemcpyDeviceToHost, g1.st));
        jck(cudaEventRecord(ev_g1_done, g1.st));
        JTR("g1 d2h+event enqueued");
    } else {
        jck(cudaSetDevice(g1.dev));
        jck(cudaEventRecord(ev_g1_done, g1.st));   // nothing to wait for; keep the flow uniform
        if (it) jck(cudaEventRecord(ev_g1_compute[0], g1.st)), jck(cudaEventRecord(ev_g1_compute[1], g1.st));
    }
    if (jfault_is("after_peer_enqueue"))
        throw std::runtime_error("PARFAULT(after_peer_enqueue): bench-only injected fault");
    // ---- CPU submit (bg; the ids/t/r tables live until the join below) ----
    Clock::time_point ts0{}, ts1{}, tj0{}, tj1{};
    const bool cpu_job = !baseline && part.cpu_routes > 0;
    std::vector<int> ids, tt, rr;
    JBgpDrain bgp{svc};          // declared after ids/tt/rr: unwinds first, drains on throw
    if (cpu_job) {
        for (int g : part.cpu_groups)
            for (int j = 0; j < w.rows[(size_t) g]; ++j) {
                const int route = w.first[(size_t) g] + j;
                ids.push_back(w.experts[(size_t) g]);
                tt.push_back(w.input_rows[(size_t) route]);
                rr.push_back(route);
            }
        ts0 = Clock::now();
        svc->run_rows_bg(0, cpu_x, H, ids.data(), tt.data(), rr.data(), (int) ids.size(),
                         cpu_out->p, H);
        bgp.pending = true;
        ts1 = Clock::now();
        if (jfault_is("after_submit"))
            throw std::runtime_error("PARFAULT(after_cpu_submit): bench-only injected fault");
    }
    // ---- GPU0: SYNTHETIC STAGING (forced copy each iteration) + grouped FFN + widen ----
    jck(cudaSetDevice(g0.dev));
    JTR("g0 begin");
    if (validate)
        jck(cudaMemsetAsync(g0.parts->p, 0x7f, (size_t) w.routes * N * sizeof(float), g0.st));
    JTR("g0 memset done");
    if (it) jck(cudaEventRecord(ev_g0[0], g0.st));
    if (!baseline)
        for (int g : part.staged_groups) {
            const int e = w.experts[(size_t) g];
            jck(cudaMemcpyAsync(g0.corpus->p + (size_t) e * c.per_expert,
                                c.tre.data() + (size_t) e * c.per_expert,
                                c.per_expert * 2, cudaMemcpyHostToDevice, g0.st));
        }
    JTR("g0 staging enqueued");
    // An EMPTY primary plan (every GPU0-owned group taken by the CPU, e.g. t1 cold4 cpu4 or
    // t4worst cold1 cpu1) launches nothing; the span events still record back-to-back (~0) and
    // the merge below reads only peer/CPU rows - never GPU0's out16.
    if (p0.routes > 0) {
        jreq(dev(p0.dg->p, p0.du->p, p0.dd->p, p0.dfirst->p, p0.drows->p, p0.groups, p0.routes,
                 w.widest, g0.x16->p, g0.st, *g0.ws, H, p0.dinput->p, H, FF, N, 2, true,
                 g0.out16->p, p0.rates[0], p0.rates[1], p0.rates[2], p0.parts[0], p0.parts[1],
                 p0.parts[2]) != nullptr, "primary grouped FFN refused");
        jck(cudaGetLastError());
    }
    JTR("g0 ffn enqueued");
    if (it) jck(cudaEventRecord(ev_g0[1], g0.st));
    if (p0.routes > 0) {
        const int threads = 256, total = p0.routes * N;
        jw_widen_scatter<<<(total + threads - 1) / threads, threads, 0, g0.st>>>(
            g0.out16->p, p0.droute->p, p0.routes, N, g0.parts->p, w.routes);
        jck(cudaGetLastError());
    }
    JTR("g0 widen enqueued");
    // ---- CPU join, then peer transfer + widen, then CPU F32 rows, then the 1/k sum ----
    if (cpu_job) {
        tj0 = Clock::now();
        svc->run_rows_join();
        bgp.pending = false;
        tj1 = Clock::now();
    }
    jck(cudaEventSynchronize(ev_g1_done));
    JTR("g1 done synced");
    if (p1.routes > 0) {
        jck(cudaMemcpyAsync(g0.stage16->p, g1_stage->p, (size_t) p1.routes * N * 2,
                            cudaMemcpyHostToDevice, g0.st));
        {
            const int threads = 256, total = p1.routes * N;
            jw_widen_scatter<<<(total + threads - 1) / threads, threads, 0, g0.st>>>(
                g0.stage16->p, peer_map.p, p1.routes, N, g0.parts->p, w.routes);
            jck(cudaGetLastError());
        }
    }
    JTR("g1 transfer+widen enqueued");
    if (!baseline)
        for (int g : part.cpu_groups)
            for (int j = 0; j < w.rows[(size_t) g]; ++j) {
                const int route = w.first[(size_t) g] + j;
                jck(cudaMemcpyAsync(g0.parts->p + (size_t) route * N,
                                    cpu_out->p + (size_t) route * N, N * sizeof(float),
                                    cudaMemcpyHostToDevice, g0.st));
            }
    {
        const int threads = 256, total = w.tokens * N;
        jw_row_sum<<<(total + threads - 1) / threads, threads, 0, g0.st>>>(
            g0.parts->p, sum_map.p, w.tokens, w.k_top, w.routes, N,
            1.0f / w.k_top, g0.sum->p);
        jck(cudaGetLastError());
    }
    jck(cudaStreamSynchronize(g0.st));
    sd.enabled = false;   // g0 drained; g1's done event was host-confirmed above - normal path
                          // pays no extra sync, and the guard goes quiet for this return
    JTR("g0 stream synced");
    if (it) {
        it->wall_ms = jms(t0);
        if (cpu_job) {
            it->submit_us = std::chrono::duration<double, std::micro>(ts1 - ts0).count();
            it->join_us = std::chrono::duration<double, std::micro>(tj1 - tj0).count();
            it->cpu_us = std::chrono::duration<double, std::micro>(tj1 - ts0).count();
        }
        float a = 0, b = 0;
        jck(cudaSetDevice(g1.dev)); jck(cudaEventElapsedTime(&a, ev_g1_compute[0], ev_g1_compute[1]));
        jck(cudaSetDevice(g0.dev)); jck(cudaEventElapsedTime(&b, ev_g0[0], ev_g0[1]));
        it->g1_ms = a; it->g0_ms = b;
    }
    if (validate) {
        res.have = true;
        res.parts.resize((size_t) w.routes * N);
        res.sum.resize((size_t) w.tokens * N);
        jck(cudaSetDevice(g0.dev));
        jck(cudaMemcpy(res.parts.data(), g0.parts->p, res.parts.size() * 4, cudaMemcpyDeviceToHost));
        jck(cudaMemcpy(res.sum.data(), g0.sum->p, res.sum.size() * 4, cudaMemcpyDeviceToHost));
        if (cpu_job) {
            res.cpu_rows.resize((size_t) part.cpu_routes * N);
            int i = 0;
            for (int g : part.cpu_groups)
                for (int j = 0; j < w.rows[(size_t) g]; ++j) {
                    const int route = w.first[(size_t) g] + j;
                    std::memcpy(res.cpu_rows.data() + (size_t) i * N, cpu_out->p + (size_t) route * N, N * 4);
                    ++i;
                }
        }
    }
    jck(cudaSetDevice(g0.dev));
    return res;
}

struct JCellReport {
    double wall_p50 = 0, wall_p95 = 0, submit_p50 = 0, join_p50 = 0, cpu_p50 = 0, g0_p50 = 0, g1_p50 = 0;
    double tail_p50 = 0;                       // wall - max(g0 span, g1 span): the non-hidden
                                               // merge/transfer/sync tail (NOT a utilization proof)
    double staging_mib = 0, avoided_mib = 0, xfer_kib = 0;
    size_t gpu_bitdiff = 0, cpu_bitdiff = 0, sum_bitdiff = 0;
    double cpu_gpu_rms = 0, cpu_gpu_max = 0;
    bool finite = true, coverage = true, bg_ok = true;
    int joins = 0, bg_workers = 0;
    long long fills = 0, cache_bytes = 0;
    std::vector<double> it_wall, it_g0, it_g1, it_cpu, it_submit, it_join;   // per-iteration
};

// One cell: per-window plans (joint + all-GPU baseline), warmup with full validation on the
// first window, then the timed loop rotating windows over the corpus.
JCellReport jrun_cell(const JCellCfg& cfg, const JCorpus& c, JGpu& g0, JGpu& g1,
                      std::vector<JWin>& wins, Exl3CpuMoe* svc, float* cpu_x,
                      JPinBuf<float>* cpu_out, JPinBuf<uint16_t>* g1_stage,
                      cudaEvent_t ev_g1c[2], cudaEvent_t ev_g1d, cudaEvent_t ev_g0[2],
                      const char* label) {
    const int H = c.G->k, N = c.G->k;
    JCellReport rep;
    // The merge maps are PER-CELL state on the shared JGpu: drop any a previous cell left
    // (all its GPU work is synced), or this cell would widen through the stale window maps -
    // the matrix exposed exactly that after t1 cells ran before t4 in one process.
    g0.sum_map.clear();
    g0.peer_map.clear();
    std::vector<std::unique_ptr<JDevPlan>> p1(wins.size()), p0j(wins.size()), p0b(wins.size());
    std::vector<JPartition> parts(wins.size());
    for (size_t wi = 0; wi < wins.size(); ++wi) {
        const JWin& w = wins[wi];
        parts[wi] = jpartition(w, cfg);
        p1[wi].reset(new JDevPlan());
        jplan_build(*p1[wi], c, g1, w, parts[wi].g1_groups, cfg.parts, true);
        p0j[wi].reset(new JDevPlan());
        jplan_build(*p0j[wi], c, g0, w, parts[wi].g0_joint, cfg.parts, true);
        p0b[wi].reset(new JDevPlan());
        jplan_build(*p0b[wi], c, g0, w, parts[wi].g0_groups, cfg.parts, true);
        // PER-WINDOW GPU0 merge maps: the token-major sum table and the peer plan's local ->
        // global route map (mirrored on dev0; uploaded once per window, outside all timing).
        // An empty peer plan (t4worst) gets a 1-int placeholder that is never dereferenced.
        g0.sum_map.emplace_back(new JDevBuf<int>(g0.dev, std::max<size_t>(1, w.token_route_row.size())));
        if (!w.token_route_row.empty()) g0.sum_map.back()->put(w.token_route_row);
        g0.peer_map.emplace_back(new JDevBuf<int>(g0.dev, std::max<size_t>(1, p1[wi]->route_row.size())));
        if (!p1[wi]->route_row.empty()) g0.peer_map.back()->put(p1[wi]->route_row);
    }
    jck(cudaStreamSynchronize(g0.st));
    const double per_expert_mib = c.per_expert * 2.0 / 1048576.0;
    rep.staging_mib = 0; rep.avoided_mib = 0; rep.xfer_kib = 0;
    for (size_t wi = 0; wi < wins.size(); ++wi) {
        rep.staging_mib += parts[wi].staged_groups.size() * per_expert_mib;
        // H2D bytes AVOIDED by the CPU tier = cold experts NOT staged. This is the CPU row
        // count only in the miss-offload arms; a resident-steal group was never a miss, so
        // cold0 steal avoids ZERO copies (its earlier cpu_groups.size() reporting was wrong).
        rep.avoided_mib += (parts[wi].cold_groups.size() - parts[wi].staged_groups.size())
                           * per_expert_mib;
        rep.xfer_kib += (p1[wi]->routes * N * 2 * 2 + parts[wi].cpu_routes * N * 4) / 1024.0;
    }
    rep.staging_mib /= wins.size(); rep.avoided_mib /= wins.size(); rep.xfer_kib /= wins.size();
    svc->set_threads(cfg.workers);
    // warmup; the first iteration validates against the all-GPU baseline of its window
    for (int i = 0; i < cfg.warmup; ++i) {
        const size_t wi = (size_t) i % wins.size();
        JIter dummy;
        const bool val = i == 0;
        JResult jr = jrun_unit(wins[wi], parts[wi], g0, g1, *p0j[wi], *p1[wi], c, svc, cpu_x,
                               cpu_out, g1_stage, ev_g1c, ev_g1d, ev_g0, *g0.sum_map[wi],
                               *g0.peer_map[wi], false, val, &dummy);
        if (val) {
            JResult base = jrun_unit(wins[wi], parts[wi], g0, g1, *p0b[wi], *p1[wi], c, svc,
                                     cpu_x, cpu_out, g1_stage, ev_g1c, ev_g1d, ev_g0,
                                     *g0.sum_map[wi], *g0.peer_map[wi], true, true, nullptr);
            const JWin& w = wins[wi];
            const JPartition& pp = parts[wi];
            for (float v : jr.parts) {                    // coverage: 0x7f7f7f7f sentinel gone
                uint32_t u; std::memcpy(&u, &v, 4);
                if (u == 0x7f7f7f7fu) rep.coverage = false;
                if (!std::isfinite(v)) rep.finite = false;
            }
            if (!rep.coverage) {
                std::printf("COVERAGE-MISS wi=%zu", wi);
                for (int r = 0; r < w.routes; ++r) {
                    uint32_t u; std::memcpy(&u, &jr.parts.data()[(size_t) r * N], 4);
                    if (u == 0x7f7f7f7fu)
                        std::printf(" row=%d served_by=%d", r, pp.served_by[(size_t) r]);
                }
                std::printf(" | g0j_routes=%d g1_routes=%d cpu_routes=%d\n",
                            p0j[wi]->routes, p1[wi]->routes, pp.cpu_routes);
                std::printf("WIN groups:");
                for (int g = 0; g < w.groups; ++g)
                    std::printf(" g%d(e%d,f%d,r%d)", g, w.experts[(size_t) g], w.first[(size_t) g], w.rows[(size_t) g]);
                std::printf("\nPEER map local->global:");
                for (int r = 0; r < p1[wi]->routes; ++r)
                    std::printf(" %d->%d", r, p1[wi]->route_row[(size_t) r]);
                std::printf("\n");
                std::fflush(stdout);
            }
            for (float v : jr.sum) if (!std::isfinite(v)) rep.finite = false;
            for (int r = 0; r < w.routes; ++r)            // GPU-kept rows BIT EXACT vs baseline
                if (pp.served_by[(size_t) r] != 2)
                    rep.gpu_bitdiff += std::memcmp(jr.parts.data() + (size_t) r * N,
                                                   base.parts.data() + (size_t) r * N, N * 4) != 0;
            if (pp.cpu_routes > 0) {
                // CPU rows byte-exact vs a standalone blocking job, identical participants.
                // solo is indexed by GLOBAL route id (run_rows writes at r[i]*out_stride),
                // so it must span the whole window's route capacity, not the CPU row count.
                std::vector<int> ids, tt, rr;
                for (int g : pp.cpu_groups)
                    for (int j = 0; j < w.rows[(size_t) g]; ++j) {
                        const int route = w.first[(size_t) g] + j;
                        ids.push_back(w.experts[(size_t) g]);
                        tt.push_back(w.input_rows[(size_t) route]);
                        rr.push_back(route);
                    }
                std::vector<float> solo((size_t) w.routes * N, -1e30f);
                svc->run_rows(0, cpu_x, H, ids.data(), tt.data(), rr.data(), (int) ids.size(),
                              solo.data(), H);
                int i2 = 0;
                for (int g : pp.cpu_groups)
                    for (int j = 0; j < w.rows[(size_t) g]; ++j) {
                        const int route = w.first[(size_t) g] + j;
                        rep.cpu_bitdiff += std::memcmp(solo.data() + (size_t) route * N,
                                                       jr.cpu_rows.data() + (size_t) i2 * N, N * 4) != 0;
                        ++i2;
                    }
                // CPU (int8) vs GPU baseline rows: the EXISTING real-model gate, 3% RMS / 12% max
                double worst_rms = 0, worst_max = 0;
                int i3 = 0;
                for (int g : pp.cpu_groups)
                    for (int j = 0; j < w.rows[(size_t) g]; ++j) {
                        const int route = w.first[(size_t) g] + j;
                        double se = 0, sr = 0, maxabs = 0;
                        const float* a = jr.cpu_rows.data() + (size_t) i3 * N;
                        const float* b = base.parts.data() + (size_t) route * N;
                        for (int q = 0; q < N; ++q) {
                            const double err = (double) a[(size_t) q] - (double) b[(size_t) q];
                            se += err * err; sr += (double) b[(size_t) q] * b[(size_t) q];
                            maxabs = std::max(maxabs, std::fabs(err));
                        }
                        worst_rms = std::max(worst_rms, std::sqrt(se / N) / std::max(std::sqrt(sr / N), 1e-9));
                        worst_max = std::max(worst_max, maxabs / std::max(std::sqrt(sr / N), 1e-9));
                        ++i3;
                    }
                rep.cpu_gpu_rms = worst_rms; rep.cpu_gpu_max = worst_max;
            }
            // final sum: host recompute in the same deterministic order (bit-equal expected)
            for (int t = 0; t < w.tokens; ++t)
                for (int q = 0; q < N; ++q) {
                    float acc = 0.f;
                    for (int j2 = 0; j2 < w.k_top; ++j2)
                        acc += jr.parts[(size_t) w.token_route_row[(size_t) (t * w.k_top + j2)] * N + q];
                    const float want = (1.0f / w.k_top) * acc;
                    rep.sum_bitdiff += std::memcmp(&want, &jr.sum[(size_t) t * N + q], 4) != 0;
                }
        }
    }
    // timed loop (windows rotate; the fixed plan decides who is cold/CPU each window)
    std::vector<double> wall, submit, join, cpus, g0s, g1s;
    rep.joins = 0;
    for (int i = 0; i < cfg.iters; ++i) {
        const size_t wi = (size_t) i % wins.size();
        JIter it;
        jrun_unit(wins[wi], parts[wi], g0, g1, *p0j[wi], *p1[wi], c, svc, cpu_x, cpu_out,
                  g1_stage, ev_g1c, ev_g1d, ev_g0, *g0.sum_map[wi], *g0.peer_map[wi],
                  false, false, &it);
        wall.push_back(it.wall_ms); g0s.push_back(it.g0_ms); g1s.push_back(it.g1_ms);
        rep.it_wall.push_back(it.wall_ms); rep.it_g0.push_back(it.g0_ms);
        rep.it_g1.push_back(it.g1_ms); rep.it_submit.push_back(it.submit_us);
        rep.it_join.push_back(it.join_us); rep.it_cpu.push_back(it.cpu_us);
        if (parts[wi].cpu_routes > 0) {
            rep.joins++;
            submit.push_back(it.submit_us); join.push_back(it.join_us); cpus.push_back(it.cpu_us);
        } else { submit.push_back(0); join.push_back(0); cpus.push_back(0); }
    }
    rep.wall_p50 = jpct(wall, 0.50); rep.wall_p95 = jpct(wall, 0.95);
    rep.submit_p50 = submit.empty() ? 0 : jpct(submit, 0.50);
    rep.join_p50 = join.empty() ? 0 : jpct(join, 0.50);
    rep.cpu_p50 = cpus.empty() ? 0 : jpct(cpus, 0.50);
    rep.g0_p50 = jpct(g0s, 0.50); rep.g1_p50 = jpct(g1s, 0.50);
    rep.tail_p50 = rep.wall_p50 - std::max(rep.g0_p50, rep.g1_p50);
    const auto st = svc->stats();
    rep.bg_workers = st.bg_workers; rep.fills = st.repack_fills; rep.cache_bytes = st.cache_bytes;
    if (rep.joins > 0) rep.bg_ok = st.bg_workers == cfg.workers - 1;
    const bool pass = rep.finite && rep.coverage && rep.bg_ok && rep.gpu_bitdiff == 0 &&
                      rep.cpu_bitdiff == 0 && rep.sum_bitdiff == 0 && rep.fills == 0 &&
                      rep.cache_bytes == 0 && rep.cpu_gpu_rms < 0.03 && rep.cpu_gpu_max < 0.12;
    std::printf("PARCELL %s geo=%s pat=%s cold=%d cpu=%d%s W=%d bg=%d parts=%d experts=%d "
                "windows=%zu iters=%d joins=%d SYNTHETIC-STAGING(host=pageable,forced-every-iter) "
                "staging_MiB=%.2f avoided_MiB=%.2f xfer_KiB=%.1f scales=copied-once(trellis-only-"
                "on-miss;engine-uploads-svh-suh-too-small-delta) | "
                "wall_p50_us=%.0f wall_p95_us=%.0f submit_p50_us=%.1f join_p50_us=%.0f "
                "cpu_complete_obs_p50_us=%.0f(cpu-submit-to-join-upper-bound) "
                "g0_span_p50_us=%.0f g1_span_p50_us=%.0f tail_beyond_max_span_p50_us=%.0f | "
                "gpu_rows_bitdiff=%zu cpu_rows_bitdiff=%zu sum_bitdiff=%zu cpu_vs_gpu_rms=%.4f%% "
                "cpu_vs_gpu_max=%.2f%% finite=%d coverage=%d fills=%lld cache_bytes=%lld %s\n",
                label, cfg.geo, cfg.pattern, cfg.cold, cfg.cpu, cfg.steal ? "(steal)" : "",
                cfg.workers, rep.bg_workers, cfg.parts, cfg.experts, wins.size(), cfg.iters, rep.joins,
                rep.staging_mib, rep.avoided_mib, rep.xfer_kib,
                1000 * rep.wall_p50, 1000 * rep.wall_p95, rep.submit_p50, rep.join_p50, rep.cpu_p50,
                1000 * rep.g0_p50, 1000 * rep.g1_p50, 1000 * std::max(0.0, rep.tail_p50),
                rep.gpu_bitdiff, rep.cpu_bitdiff, rep.sum_bitdiff, 100 * rep.cpu_gpu_rms,
                100 * rep.cpu_gpu_max, (int) rep.finite, (int) rep.coverage, rep.fills, rep.cache_bytes,
                pass ? "PASS" : "FAIL");
    std::fflush(stdout);
    if (!pass) throw std::runtime_error("cell checks failed");
    return rep;
}

void jsetup_gpu(const JCorpus& c, JGpu& g, int dev, bool primary, const std::vector<uint16_t>& hx16) {
    g.dev = dev; g.band8 = c.packed;
    jck(cudaSetDevice(dev));
    jck(cudaStreamCreate(&g.st));
    g.ws.reset(new Exl3MoeWorkspace());
    g.ws->reserve_small_m(32, 24, c.G->k, c.G->ff, c.G->k);
    g.corpus.reset(new JDevBuf<uint16_t>(dev, (size_t) c.experts * c.per_expert));
    g.scales.reset(new JDevBuf<uint16_t>(dev, c.scales.size()));
    g.scales->put(c.scales);                       // scales resident once, whole run
    g.x16.reset(new JDevBuf<uint16_t>(dev, hx16.size()));
    g.x16->put(hx16);
    g.out16.reset(new JDevBuf<uint16_t>(dev, (size_t) 32 * c.G->k));
    if (primary) {
        g.stage16.reset(new JDevBuf<uint16_t>(dev, (size_t) 32 * c.G->k));
        g.parts.reset(new JDevBuf<float>(dev, (size_t) 32 * c.G->k));
        g.sum.reset(new JDevBuf<float>(dev, (size_t) 4 * c.G->k));
    }
    // only the OWNED half of the corpus buffer is written (disjoint residency: e % 2 == dev)
    for (int e = dev; e < c.experts; e += 2)
        jck(cudaMemcpy(g.corpus->p + (size_t) e * c.per_expert, c.tre.data() + (size_t) e * c.per_expert,
                       c.per_expert * 2, cudaMemcpyHostToDevice));
}

struct JEvents {
    cudaEvent_t g1c[2] = {nullptr, nullptr}, g1d = nullptr, g0[2] = {nullptr, nullptr};
    JEvents(int dev0, int dev1) {
        jck(cudaSetDevice(dev1));
        jck(cudaEventCreate(&g1c[0])); jck(cudaEventCreate(&g1c[1])); jck(cudaEventCreate(&g1d));
        jck(cudaSetDevice(dev0));
        jck(cudaEventCreate(&g0[0])); jck(cudaEventCreate(&g0[1]));
    }
    ~JEvents() {
        (void) cudaEventDestroy(g1c[0]); (void) cudaEventDestroy(g1c[1]); (void) cudaEventDestroy(g1d);
        (void) cudaEventDestroy(g0[0]); (void) cudaEventDestroy(g0[1]);
    }
};

// CPU service registered on the SAME canonical host bytes (layout follows the corpus).
void jregister_cpu(Exl3CpuMoe& svc, const JCorpus& c) {
    const JGeo& G = *c.G;
    const int dk[3] = { G.k, G.k, G.ff }, dn[3] = { G.ff, G.ff, G.k };
    std::vector<Exl3CpuMatrix> gm((size_t) c.experts), um((size_t) c.experts), dm((size_t) c.experts);
    const Exl3CpuTrellisLayout lay = c.packed ? Exl3CpuTrellisLayout::Band8
                                              : Exl3CpuTrellisLayout::Native;
    for (int e = 0; e < c.experts; ++e)
        for (int p = 0; p < 3; ++p) {
            Exl3CpuMatrix m{c.tre.data() + c.plane_off(e, p),
                            c.scales.data() + c.scale_off(e, p),
                            c.scales.data() + c.scale_off(e, p) + dk[p],
                            dk[p], dn[p], G.bits, G.hb, lay};
            (p == 0 ? gm : p == 1 ? um : dm)[(size_t) e] = m;
        }
    svc.add_layer(0, gm.data(), um.data(), dm.data(), c.experts);
}

int joint_main(int argc, char** argv) {
    bool smoke = false, matrix = false;
    JCellCfg cfg; cfg.geo = "k2"; cfg.pattern = "t1"; cfg.cold = 2; cfg.cpu = 0; cfg.workers = 8;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> const char* { jreq(i + 1 < argc, "missing argument value"); return argv[++i]; };
        if (!std::strcmp(argv[i], "--smoke")) smoke = true;
        else if (!std::strcmp(argv[i], "--matrix")) matrix = true;
        else if (!std::strcmp(argv[i], "--geo")) cfg.geo = next();
        else if (!std::strcmp(argv[i], "--pattern")) cfg.pattern = next();
        else if (!std::strcmp(argv[i], "--cold")) cfg.cold = std::atoi(next());
        else if (!std::strcmp(argv[i], "--cpu")) cfg.cpu = std::atoi(next());
        else if (!std::strcmp(argv[i], "--steal")) cfg.steal = true;
        else if (!std::strcmp(argv[i], "--workers")) cfg.workers = std::atoi(next());
        else if (!std::strcmp(argv[i], "--iters")) cfg.iters = std::atoi(next());
        else if (!std::strcmp(argv[i], "--warmup")) cfg.warmup = std::atoi(next());
        else if (!std::strcmp(argv[i], "--experts")) cfg.experts = std::atoi(next());
        else if (!std::strcmp(argv[i], "--parts")) cfg.parts = std::atoi(next());
        else if (!std::strcmp(argv[i], "--sweeps")) cfg.sweeps = std::atoi(next());
        else if (!std::strcmp(argv[i], "--layout")) cfg.packed = std::strcmp(next(), "native") != 0;
        else jreq(false, "unknown joint bench flag");
    }
    jreq(cfg.iters >= 4 && cfg.iters <= 4096 && cfg.warmup >= 1 && cfg.warmup <= 64, "iters/warmup bounds");
    jreq(cfg.workers == 8 || cfg.workers == 12, "workers must be 8 or 12 (the audited configs)");
    jreq(cfg.experts == 32 || cfg.experts == 64, "experts must be 32 or 64");
    jreq(cfg.parts >= 1 && cfg.parts <= 4, "parts 1..4");
    jreq(cfg.sweeps >= 1 && cfg.sweeps <= 4, "sweeps 1..4");
    jreq(!smoke || !matrix, "--smoke and --matrix are exclusive");

    int count = 0; jck(cudaGetDeviceCount(&count));
    jreq(count >= 2, "joint bench needs two GPUs");
    const int devs[2] = {0, 1};
    for (int d : devs) {
        hipDeviceProp_t prop{}; jck(cudaSetDevice(d)); jck(hipGetDeviceProperties(&prop, d));
        jreq(gfx_arch_is_gfx11_wmma(prop.gcnArchName), "both devices must be gfx11 wmma");
    }
    jreq(Exl3CpuMoe::supported(), "CPU service unsupported here");

    auto make_windows = [](const JCorpus& c, const char* pattern) {
        std::vector<JWin> wins;
        std::mt19937 rr(20261107);
        if (!std::strcmp(pattern, "t1"))
            for (int f = 0; f + 8 <= c.experts; f += 8) wins.push_back(jmake_t1(c, f));
        else if (!std::strcmp(pattern, "t4"))
            for (int f = 0; f + 24 <= c.experts; f += 8) wins.push_back(jmake_t4(c, f, rr));
        else if (!std::strcmp(pattern, "t4worst"))
            for (int f = 0; f < c.experts; f += 8) wins.push_back(jmake_worst(c, f % c.experts));
        else jreq(false, "pattern must be t1|t4|t4worst");
        jreq(!wins.empty(), "no windows for this corpus/pattern");
        return wins;
    };
    auto find_geo = [](const char* name) -> const JGeo* {
        for (const auto& g : kJGeos) if (!std::strcmp(g.name, name)) return &g;
        return nullptr;
    };

    // half-exact activations: the GPU proto f16 pattern; the CPU consumes the SAME values as f32
    const int H = kJGeos[0].k;
    std::vector<uint16_t> hx16((size_t) 4 * H);
    for (size_t i = 0; i < hx16.size(); ++i) hx16[i] = jhalf(((int) (i % 97) - 48) * 0.001f);
    std::vector<float> cpu_x(hx16.size());
    for (size_t i = 0; i < hx16.size(); ++i) cpu_x[i] = jreal(hx16[i]);

    JEvents ev(devs[0], devs[1]);
    JPinBuf<float> cpu_out((size_t) 32 * H);
    JPinBuf<uint16_t> g1_stage((size_t) 32 * H);
    std::printf("PARHEAD devices=%d,%d host_src=pageable corpus_layout=%s "
                "cpu_layout=%s fixtures=scales +-(0.10..0.131) act=((i%%97)-48)*0.001f16 "
                "mt19937(20261106+geo) merge=widen_f16_to_f32+cpu_f32_rows_direct "
                "sum=route_ordered_1/k | notes: cpu_out=PINNED(engine-vec-is-pageable;2rows~32KiB"
                "-difference-recorded-not-inferred) staging=trellis-only(scales-resident;engine-"
                "miss-uploads-suh/svh-too,~0.5%%-delta) cpu_complete_obs=CPUsubmit-to-join-UPPER-BOUND"
                "(measured-after-GPU1-enqueue;includes-host-GPU0-enqueue+pageable-H2D-blocking;"
                "not-cpu-hardware-time) spans=per-device-events non-additive avoided=cold-not-staged"
                "(resident-steal=0)\n",
                devs[0], devs[1], cfg.packed ? "band8" : "native",
                cfg.packed ? "Band8(cache0,swizzle0)" : "Native");

    if (smoke) {
        // The audited smoke set ONLY - STOP before the full matrix. Includes the EMPTY-PRIMARY
        // (t1 cold4 cpu4), EMPTY-PEER (t4worst) and FULL-CPU (t4worst cold1 cpu1: both empty)
        // cells, each in both layouts.
        struct Smoke { const char* geo; const char* pat; int cold, cpu, W; bool packed; bool steal; };
        const Smoke cells[] = {
            {"k2", "t1", 2, 0, 8, true, false}, {"k2", "t1", 2, 1, 8, true, false},
            {"k2", "t1", 2, 2, 8, true, false},
            {"k25", "t4", 2, 0, 12, true, false}, {"k25", "t4", 2, 1, 12, true, false},
            {"k2", "t1", 4, 4, 8, true, false},       // all four primary groups on CPU
            {"k2", "t4worst", 1, 1, 12, true, false}, // full CPU: primary AND peer both empty
            {"k2", "t1", 4, 4, 8, false, false},      // native-layout counterparts
            {"k2", "t4worst", 1, 1, 12, false, false},
            {"k2", "t1", 0, 1, 8, true, true},        // hot steal: must report avoided_MiB=0.00
        };
        const char* cur_geo = "";
        bool cur_packed = true;
        std::unique_ptr<JCorpus> corpus;
        std::unique_ptr<JGpu> g0, g1;
        int failures = 0;
        for (const Smoke& s : cells) {
            if (std::strcmp(cur_geo, s.geo) != 0 || cur_packed != s.packed) {
                // per (geo, layout) corpus + GPU residency, reused by its cells
                cur_geo = s.geo; cur_packed = s.packed;
                const JGeo* G = find_geo(s.geo);
                jreq(G != nullptr, "unknown geo");
                corpus.reset(new JCorpus());
                jcorpus_build(*corpus, *G, cfg.experts, s.packed, (int) (G - kJGeos));
                std::printf("PARCORPUS geo=%s layout=%s experts=%d corpus_MiB=%.1f pack_ms=%.1f "
                            "rates=%d+%d per_expert_MiB=%.3f\n", s.geo, s.packed ? "band8" : "native",
                            cfg.experts, corpus->tre.size() * 2.0 / 1048576, corpus->pack_ms,
                            G->bits, G->hb, corpus->per_expert * 2.0 / 1048576);
                g0.reset(new JGpu()); g1.reset(new JGpu());
                jsetup_gpu(*corpus, *g0, devs[0], true, hx16);
                jsetup_gpu(*corpus, *g1, devs[1], false, hx16);
            }
            JCellCfg cc = cfg; cc.geo = s.geo; cc.pattern = s.pat;
            cc.cold = s.cold; cc.cpu = s.cpu; cc.workers = s.W; cc.steal = s.steal; cc.packed = s.packed;
            auto wins = make_windows(*corpus, s.pat);
            try {
                Exl3CpuMoe svc;
                jregister_cpu(svc, *corpus);
                jrun_cell(cc, *corpus, *g0, *g1, wins, &svc, cpu_x.data(), &cpu_out, &g1_stage,
                          ev.g1c, ev.g1d, ev.g0, "SMOKE");
            } catch (const std::exception& e) {
                std::printf("PARCELL SMOKE geo=%s pat=%s cold=%d cpu=%d W=%d ERROR %s FAIL\n",
                            s.geo, s.pat, s.cold, s.cpu, s.W, e.what());
                ++failures;
            }
        }
        if (failures) { std::printf("PARSMOKE FAILED=%d\n", failures); return 1; }
        std::puts("PARSMOKE PASS (smoke checkpoint only - STOP before the full matrix; "
                  "no engine/default/commit changes)");
        return 0;
    }
    if (matrix) {
        // ---- the bounded full matrix, ONE persistent process ----
        // Groups are (geo, layout); each group materializes its OWN corpus (packed cells never
        // see a native shadow and vice versa), sets up both GPUs once, and runs every cell with
        // a fresh CPU service. Sweep 0 runs groups and cells in build order; sweep 1 runs BOTH
        // reversed, so first-position effects cannot alias onto a layout/rate. Every cell logs a
        // PARMXCSV summary plus one PARITCSV line per timed iteration; nothing is dropped.
        struct Mx { const char* geo; const char* pat; int cold, cpu, W; bool steal; };
        auto build_cells = [](const char* geo) {
            std::vector<Mx> v;
            for (int W : {8, 12}) {
                for (const char* pat : {"t1", "t4"}) {
                    v.push_back({geo, pat, 0, 0, W, false});
                    for (int cold : {2, 4})
                        for (int cpu = 0; cpu <= cold; ++cpu) v.push_back({geo, pat, cold, cpu, W, false});
                    v.push_back({geo, pat, 0, 1, W, true});   // resident-steal label (0 miss)
                    v.push_back({geo, pat, 0, 2, W, true});
                }
                v.push_back({geo, "t4worst", 1, 0, W, false});
                v.push_back({geo, "t4worst", 1, 1, W, false});
            }
            return v;
        };
        struct Group { const char* geo; bool packed; };
        const Group groups[] = {{"k2", true}, {"k2", false}, {"k25", true}, {"k25", false}};
        const int ngroups = 4;
        int failures = 0;
        for (int sweep = 0; sweep < cfg.sweeps; ++sweep) {
            for (int gi = 0; gi < ngroups; ++gi) {
                const int gsel = sweep == 0 ? gi : ngroups - 1 - gi;
                const Group& gr = groups[gsel];
                const JGeo* G = find_geo(gr.geo);
                jreq(G != nullptr, "unknown geo");
                JCorpus corpus;
                jcorpus_build(corpus, *G, cfg.experts, gr.packed, (int) (G - kJGeos));
                std::printf("PARCORPUS sweep=%d geo=%s layout=%s experts=%d corpus_MiB=%.1f "
                            "pack_ms=%.1f rates=%d+%d per_expert_MiB=%.3f\n", sweep, gr.geo,
                            gr.packed ? "band8" : "native", cfg.experts,
                            corpus.tre.size() * 2.0 / 1048576, corpus.pack_ms, G->bits, G->hb,
                            corpus.per_expert * 2.0 / 1048576);
                std::fflush(stdout);
                JGpu g0, g1;
                jsetup_gpu(corpus, g0, devs[0], true, hx16);
                jsetup_gpu(corpus, g1, devs[1], false, hx16);
                auto wins_by_pat = [&](const char* pat) { return make_windows(corpus, pat); };
                std::vector<Mx> cells = build_cells(gr.geo);
                if (sweep == 1) std::reverse(cells.begin(), cells.end());
                for (size_t ci = 0; ci < cells.size(); ++ci) {
                    const Mx& m = cells[ci];
                    JCellCfg cc = cfg; cc.geo = m.geo; cc.pattern = m.pat; cc.cold = m.cold;
                    cc.cpu = m.cpu; cc.workers = m.W; cc.steal = m.steal; cc.packed = gr.packed;
                    auto wins = wins_by_pat(m.pat);
                    try {
                        Exl3CpuMoe svc;
                        jregister_cpu(svc, corpus);
                        JCellReport rep = jrun_cell(cc, corpus, g0, g1, wins, &svc, cpu_x.data(),
                                                    &cpu_out, &g1_stage, ev.g1c, ev.g1d, ev.g0,
                                                    sweep == 0 ? "MX-A" : "MX-B");
                        std::printf("PARMXCSV %d,%s,%s,%s,%d,%d,%d,%d,%zu,%d,%d,%d,%.3f,%.3f,%.1f,"
                                    "%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%zu,%zu,%zu,%.4f,%.2f,%d,%d\n",
                                    sweep, m.geo, gr.packed ? "band8" : "native", m.pat, m.cold,
                                    m.cpu, (int) m.steal, m.W, ci, cfg.iters, rep.joins,
                                    rep.bg_workers, rep.staging_mib, rep.avoided_mib, rep.xfer_kib,
                                    1000 * rep.wall_p50, 1000 * rep.wall_p95, 1000 * rep.g0_p50,
                                    1000 * rep.g1_p50, rep.submit_p50, rep.join_p50, rep.cpu_p50,
                                    1000 * std::max(0.0, rep.tail_p50), rep.gpu_bitdiff,
                                    rep.cpu_bitdiff, rep.sum_bitdiff, 100 * rep.cpu_gpu_rms,
                                    100 * rep.cpu_gpu_max, (int) rep.coverage, (int) rep.finite);
                        for (size_t k = 0; k < rep.it_wall.size(); ++k)
                            std::printf("PARITCSV %d,%s,%s,%s,%d,%d,%d,%d,%zu,%zu,%.1f,%.1f,%.1f,"
                                        "%.1f,%.1f,%.1f\n",
                                        sweep, m.geo, gr.packed ? "band8" : "native", m.pat, m.cold,
                                        m.cpu, (int) m.steal, m.W, ci, k, 1000 * rep.it_wall[k],
                                        1000 * rep.it_g0[k], 1000 * rep.it_g1[k], rep.it_submit[k],
                                        rep.it_join[k], rep.it_cpu[k]);
                        std::fflush(stdout);
                    } catch (const std::exception& e) {
                        std::printf("PARMXCSV-ERROR sweep=%d geo=%s pat=%s cold=%d cpu=%d W=%d "
                                    "steal=%d ERROR %s FAIL\n", sweep, m.geo, m.pat, m.cold, m.cpu,
                                    m.W, (int) m.steal, e.what());
                        ++failures;
                    }
                }
            }
        }
        if (failures) { std::printf("PARMATRIX FAILED=%d\n", failures); return 1; }
        std::printf("PARMATRIX DONE sweeps=%d iters=%d cells_per_group=%zu (bounded sweep "
                    "complete; STOP for audit)\n", cfg.sweeps, cfg.iters, build_cells("k2").size());
        return 0;
    }
    // single-cell mode (the post-audit matrix driver)
    const JGeo* G = find_geo(cfg.geo);
    jreq(G != nullptr, "unknown geo");
    JCorpus corpus;
    jcorpus_build(corpus, *G, cfg.experts, cfg.packed, (int) (G - kJGeos));
    std::printf("PARCORPUS geo=%s experts=%d corpus_MiB=%.1f pack_ms=%.1f rates=%d+%d "
                "per_expert_MiB=%.3f\n", cfg.geo, cfg.experts,
                corpus.tre.size() * 2.0 / 1048576, corpus.pack_ms, G->bits, G->hb,
                corpus.per_expert * 2.0 / 1048576);
    JGpu g0, g1;
    jsetup_gpu(corpus, g0, devs[0], true, hx16);
    jsetup_gpu(corpus, g1, devs[1], false, hx16);
    auto wins = make_windows(corpus, cfg.pattern);
    Exl3CpuMoe svc;
    jregister_cpu(svc, corpus);
    jrun_cell(cfg, corpus, g0, g1, wins, &svc, cpu_x.data(), &cpu_out, &g1_stage,
              ev.g1c, ev.g1d, ev.g0, "CELL");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return joint_main(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}

#endif  // STRATA_WMMA_GFX11
