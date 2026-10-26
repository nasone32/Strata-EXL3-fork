// src/kernels/exl3_small_m_bench.cpp - NON-PRODUCTION diagnostic (MTP small-M verification windows).
//
// ISOLATES ONE QUESTION on device-RESIDENT real 3bpw expert weights: does expert trellis-decode
// reuse across the 2..6 activation rows that share an expert beat the current per-route indexed
// GEMV (`exl3_moe_rows_indexed`, the path every decode-window M=1..6 route uses today), INCLUDING
// the per-call route-plan uploads and descriptor uploads - not the inner kernel alone.
//
//   A (current)  : the public indexed entry, one call per layer-window exactly like the engine's
//                   hit arm (host per-ROUTE descriptors, device input_rows).
//   B (candidate): the PROTOTYPE `exl3_moe_rows_grouped` - same route order, same buffers, same
//                   outputs, one block per expert GROUP; the trellis window is decoded once and
//                   applied to every row of the group.  The ggml kernel this idea comes from
//                   (exl3_mul1_rows_body) switches to prequantized DP4A activations - a different
//                   numeric contract than this engine's float path - so B keeps the float
//                   arithmetic bit for bit.
//
// FIXTURES ARE REAL ROUTING SHAPES: every window is a VALID token-major top-10 assignment (each
// token picks 10 DISTINCT experts), the groups are derived with the engine's own Exl3RoutePlan
// counting sort, and the harness asserts: exactly 10 routes per token, no repeated expert within a
// token, group rows <= window M, distinct token rows within every group, groups = contiguous
// slices covering the routes.  Mixed histograms are CONSTRUCTED to approximate the measured prose
// MTP fractions (M4 matches the measured 17/5/3/1 per-layer-window; M2/M3/M5 approximate the
// measured 44/52/60% rows-in-multigroup; M6 is extrapolated) - they are NOT captured routes.
//
// Parity: B vs A must be BYTE-IDENTICAL on EVERY trial of every scenario (all 30 plans, outside
// the timed region), plus finite/non-constant outputs; an independent CPU oracle (trellis
// reconstructed on the CPU, projection in double) runs 2 probes on exactly 3 scenarios
// (homo-M1, mix-M4, homo-M6) - that is the stated oracle coverage, no more.  Synthetic-parity
// probes cover K4/mul1 and K3/mcg/K3/3inst (real-weight scenarios are K3/mul1 only).
//
// Cost boundaries (printed with every line): "e2e" - host wall around enqueue+sync, the PRIMARY
// per-call resident-API cost; "gpu_span" - CUDA-event interval around the enqueued work (includes
// queue-idle gaps while the host stages pageable uploads - not pure kernel time); "issue" - host
// wall until the call returns.  "chain" is a 48-call back-to-back batch on one stream - a queue
// DIAGNOSTIC ONLY, NOT engine cadence (the engine consumes each layer's activations before the
// next window): no end-to-end gain may be inferred from it and it is not preferred over e2e.
// The host route-plan build cost is measured separately (checksummed so the compiler cannot
// elide it) and never folded into a GPU span.  Weight upload happens once at setup.  Microbench
// evidence, NOT an engine TG claim.
//
// Usage: exl3_small_m_bench <model_dir> [device=0] [trials=30]
#include "strata/artifact/exl3_model.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/exl3_route_plan.hpp"
#include "strata/kernels/cpu/exl3.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using strata::Exl3Linear;
using strata::Exl3Model;
using strata::kernels::Exl3Mat;
using strata::kernels::Exl3MoeWorkspace;
using strata::kernels::Exl3RoutePlan;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int WIDTH = 2560, HIDDEN = 640;      // the real expert geometry (verified at load)
constexpr int X_ROWS = 8;                      // the padded verify-window token buffer (worst window)
constexpr int TOP_K = 10;                      // the model's experts per token
constexpr int SMALL_M_MAX_ROWS = 6;           // must match the prototype entry's compiled width

void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
uint16_t f16(float f) { const __half h = __float2half_rn(f); uint16_t v; std::memcpy(&v, &h, 2); return v; }
float fp32(uint16_t v) { __half h; std::memcpy(&h, &v, 2); return __half2float(h); }

double median(std::vector<double> v) {
    if (v.empty()) return -1;
    std::sort(v.begin(), v.end());
    return v[(v.size() - 1) / 2];
}
// interquartile spread: (q3 - q1), the middle half of the samples
double iqr(std::vector<double> v) {
    if (v.size() < 4) return -1;
    std::sort(v.begin(), v.end());
    const size_t q = v.size() / 4;
    return v[v.size() - 1 - q] - v[q];
}

[[noreturn]] void fixture_fail(const std::string& what) {
    throw std::runtime_error("smallm fixture invalid: " + what);
}

// One resident expert: its three projections in ONE device allocation (planes packed back to
// back; every plane size is a multiple of 4 so the trellis stays uint32-readable), plus the
// mmap'd host linears the CPU oracle re-reads.
struct ResidentSlot {
    uint8_t* blob = nullptr;
    size_t bytes = 0;
    Exl3Mat g, u, d;
    Exl3Linear hg, hu, hd;   // HOST pointers (mmap of the safetensors) for the CPU oracle
};
struct ResidentSet {
    std::vector<int> layers;                       // distinct REAL layers
    int per_layer = 0;
    std::vector<ResidentSlot> slots;               // layers.size() * per_layer
    std::vector<std::vector<int>> expert_ids;      // expert id per (layer index, slot)
    ~ResidentSet() { for (auto& s : slots) if (s.blob) (void) cudaFree(s.blob); }
};

void load_resident(Exl3Model& model, ResidentSet& rs, std::mt19937& rng) {
    rs.layers = {0, 12, 24, 36};   // distinct real layers spread through the 48-layer model
    rs.per_layer = 48;             // enough distinct experts for every scenario's group count
    const size_t total = rs.layers.size() * (size_t) rs.per_layer;
    rs.slots.resize(total);
    rs.expert_ids.assign(rs.layers.size(), std::vector<int>((size_t) rs.per_layer, 0));
    for (size_t li = 0; li < rs.layers.size(); ++li) {
        std::vector<int> ids(512);                  // deterministic distinct expert ids per layer
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), rng);
        for (int s = 0; s < rs.per_layer; ++s) rs.expert_ids[li][(size_t) s] = ids[(size_t) s];
    }
    for (size_t li = 0; li < rs.layers.size(); ++li) {
        for (int s = 0; s < rs.per_layer; ++s) {
            const std::string base = "model.language_model.layers." + std::to_string(rs.layers[li]) +
                                     ".mlp.experts." + std::to_string(rs.expert_ids[li][(size_t) s]) + ".";
            ResidentSlot& r = rs.slots[li * (size_t) rs.per_layer + (size_t) s];
            r.hg = model.linear(base + "gate_proj");
            r.hu = model.linear(base + "up_proj");
            r.hd = model.linear(base + "down_proj");
            const Exl3Linear* lin[3] = {&r.hg, &r.hu, &r.hd};
            Exl3Mat* mats[3] = {&r.g, &r.u, &r.d};
            size_t bytes = 0, off[9] = {};
            for (int i = 0; i < 3; ++i) {
                off[(size_t) i * 3 + 0] = bytes; bytes += (size_t) lin[i]->ki * lin[i]->nj * 32u * (unsigned) lin[i]->bits;
                off[(size_t) i * 3 + 1] = bytes; bytes += (size_t) lin[i]->in() * 2;
                off[(size_t) i * 3 + 2] = bytes; bytes += (size_t) lin[i]->out() * 2;
            }
            r.bytes = bytes;
            check(cudaMalloc((void**) &r.blob, bytes));
            for (int i = 0; i < 3; ++i) {
                const size_t tr = (size_t) lin[i]->ki * lin[i]->nj * 32u * (unsigned) lin[i]->bits;
                check(cudaMemcpy(r.blob + off[(size_t) i * 3 + 0], lin[i]->trellis, tr, cudaMemcpyHostToDevice));
                check(cudaMemcpy(r.blob + off[(size_t) i * 3 + 1], lin[i]->suh, (size_t) lin[i]->in() * 2, cudaMemcpyHostToDevice));
                check(cudaMemcpy(r.blob + off[(size_t) i * 3 + 2], lin[i]->svh, (size_t) lin[i]->out() * 2, cudaMemcpyHostToDevice));
                mats[i]->trellis = (const uint16_t*) (r.blob + off[(size_t) i * 3 + 0]);
                mats[i]->suh = (const uint16_t*) (r.blob + off[(size_t) i * 3 + 1]);
                mats[i]->svh = (const uint16_t*) (r.blob + off[(size_t) i * 3 + 2]);
                mats[i]->ki = lin[i]->ki; mats[i]->nj = lin[i]->nj;
                mats[i]->bits = lin[i]->bits; mats[i]->cb = lin[i]->cb;
            }
        }
    }
}

// One window's route plan, in BOTH arms' formats, derived from a VALID token-major assignment.
struct TrialPlan {
    int layer_index = 0;
    int window_m = 0;
    int routes = 0, groups = 0;
    std::vector<int> input_rows;                       // per route: the token row in the window's x
    std::vector<Exl3Mat> route_g, route_u, route_d;    // per-route host descriptors (arm A)
    std::vector<Exl3Mat> group_g, group_u, group_d;    // per-group host descriptors (arm B)
    std::vector<int> group_first, group_rows;
    std::vector<int> group_slot;                       // pool slot per group (oracle resolution)
};

// CONSTRUCTED rows-per-group targets.  mix-M4 matches the measured prose histogram (91964/27953/
// 11842/6951 groups at 1/2/3/4 rows over 110 windows => 17.4/5.3/2.25/1.3 per layer-window);
// mix-M2/M3/M5 approximate the measured rows-in-multigroup fractions (44/52/60%) - M2 alternates
// between the 40% and 50% arrangements, averaging ~45%; M6 is an extrapolation, no measurement.
// `variant` alternates arrangements per trial where a measured fraction admits more than one
// valid integer split.  All targets: every size <= M and sum == TOP_K*M (a token never repeats an
// expert, so no group can exceed the window's token count).
std::vector<int> target_sizes(const char* kind, int m, int variant) {
    const std::string k = kind;
    auto rep = [](int rows, int times) { return std::vector<int>((size_t) times, rows); };
    std::vector<int> p;
    auto add = [&p](std::vector<int> extra) { p.insert(p.end(), extra.begin(), extra.end()); };
    if (k == "homo") return rep(m, TOP_K);            // 10 experts, each picked by all M tokens
    switch (m) {
        case 1: return rep(1, TOP_K);                  // M1: measured, all singletons
        case 2: if ((variant & 1) == 0) { add(rep(1, 12)); add(rep(2, 4)); }        // 40% shared
                else { add(rep(1, 10)); add(rep(2, 5)); }                           // 50% shared
                break;
        case 3: if ((variant & 1) == 0) { add(rep(1, 14)); add(rep(2, 5)); add(rep(3, 2)); }  // 53%
                else { add(rep(1, 15)); add(rep(2, 6)); add(rep(3, 1)); }                     // 50%
                break;
        case 4: add(rep(1, 17)); add(rep(2, 5)); add(rep(3, 3)); add(rep(4, 1)); break;  // measured M4 shape
        case 5: add(rep(1, 20)); add(rep(2, 4)); add(rep(3, 3)); add(rep(4, 2)); add(rep(5, 1)); break;  // 60%
        case 6: add(rep(1, 23)); add(rep(2, 6)); add(rep(3, 4)); add(rep(4, 2)); add(rep(5, 1)); break;  // extrapolated
        default: fixture_fail("target_sizes: m out of range");
    }
    if ((int) std::accumulate(p.begin(), p.end(), 0) != TOP_K * m) fixture_fail("target sizes do not sum to 10*M");
    for (int s : p) if (s > m) fixture_fail("target size exceeds the window token count");
    return p;
}

// Build one VALID window: assign the target group sizes to distinct pool slots, let each token
// pick its 10 experts by largest remaining demand (so the token-major assignment realizes the
// targets while every token holds 10 DISTINCT experts), then derive the expert-major groups with
// the ENGINE'S OWN Exl3RoutePlan counting sort and assert the whole contract.
void build_valid_plan(const Exl3Mat* pool_g, const Exl3Mat* pool_u, const Exl3Mat* pool_d,
                      int pool_size, int window_m, const std::vector<int>& targets,
                      std::mt19937& rng, int layer_index, TrialPlan& out) {
    const int n_groups = (int) targets.size();
    if (pool_size < n_groups) fixture_fail("pool smaller than the target group count");
    std::vector<int> order((size_t) pool_size);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<int> demand((size_t) pool_size, 0);
    for (int g = 0; g < n_groups; ++g) demand[(size_t) order[(size_t) g]] = targets[(size_t) g];
    // token-major picks: 10 distinct slots per token, largest remaining demand first
    std::vector<int32_t> ids((size_t) window_m * TOP_K);
    for (int t = 0; t < window_m; ++t) {
        std::set<int> used;
        for (int k = 0; k < TOP_K; ++k) {
            int best = -1;
            for (int s = 0; s < pool_size; ++s)
                if (demand[(size_t) s] > 0 && used.count(s) == 0 &&
                    (best < 0 || demand[(size_t) s] > demand[(size_t) best]))
                    best = s;
            if (best < 0) fixture_fail("cannot complete a token's 10 distinct experts");
            used.insert(best);
            ids[(size_t) (t * TOP_K + k)] = best;
            --demand[(size_t) best];
        }
    }
    for (int s = 0; s < pool_size; ++s)
        if (demand[(size_t) s] != 0) fixture_fail("greedy assignment did not realize the targets");
    // the engine's own plan: expert-major groups over the token-major ids
    Exl3RoutePlan plan;
    plan.build(ids.data(), window_m, TOP_K, pool_size);
    out.layer_index = layer_index;
    out.window_m = window_m;
    out.routes = window_m * TOP_K;
    out.groups = (int) plan.experts.size();
    out.input_rows.clear(); out.group_first.clear(); out.group_rows.clear(); out.group_slot.clear();
    out.route_g.clear(); out.route_u.clear(); out.route_d.clear();
    out.group_g.clear(); out.group_u.clear(); out.group_d.clear();
    int prefix = 0;
    for (int g = 0; g < out.groups; ++g) {
        const int slot = plan.experts[(size_t) g];
        const int rows = plan.offsets[(size_t) g + 1] - plan.offsets[(size_t) g];
        if (plan.offsets[(size_t) g] != prefix) fixture_fail("plan groups are not contiguous");
        if (rows < 1 || rows > window_m) fixture_fail("group rows out of [1, window_m]");
        if (rows > SMALL_M_MAX_ROWS) fixture_fail("group rows exceed the prototype width");
        std::set<int> group_tokens;
        for (int i = 0; i < rows; ++i) {
            const int r = plan.destinations[(size_t) (prefix + i)];
            const int token = r / TOP_K;
            if (!group_tokens.insert(token).second) fixture_fail("group repeats a token row");
            out.input_rows.push_back(token);
            out.route_g.push_back(pool_g[(size_t) slot]);
            out.route_u.push_back(pool_u[(size_t) slot]);
            out.route_d.push_back(pool_d[(size_t) slot]);
        }
        out.group_first.push_back(prefix);
        out.group_rows.push_back(rows);
        out.group_slot.push_back(slot);
        out.group_g.push_back(pool_g[(size_t) slot]);
        out.group_u.push_back(pool_u[(size_t) slot]);
        out.group_d.push_back(pool_d[(size_t) slot]);
        prefix += rows;
    }
    if (prefix != out.routes) fixture_fail("plan does not cover the routes");
    // the achieved group-size multiset must equal the target multiset exactly
    std::vector<int> got(out.group_rows), want(targets);
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    if (got != want) fixture_fail("achieved group sizes differ from the targets");
}

// Independent CPU oracle for one (expert, token) row: reconstruct the three real matrices on the
// CPU, project in double, mirror the fp16 rounding points of the GPU path.  Same reference shape
// as exl3_moe_multi_parity; the real K3/mul1 planes go through the same cpu decoder.
double cpu_oracle_row(const ResidentSlot& s, const uint16_t* x_row, std::vector<float>& y) {
    const int in = s.hg.in(), ff = s.hg.out(), n = s.hd.out();
    std::vector<uint16_t> wg((size_t) in * ff), wu((size_t) in * ff), wd((size_t) ff * n);
    strata::kernels::cpu::exl3_reconstruct_weight(s.hg.trellis, s.hg.ki, s.hg.nj, s.hg.bits,
        (strata::kernels::cpu::Exl3Codebook) s.hg.cb, s.hg.suh, s.hg.svh, wg.data());
    strata::kernels::cpu::exl3_reconstruct_weight(s.hu.trellis, s.hu.ki, s.hu.nj, s.hu.bits,
        (strata::kernels::cpu::Exl3Codebook) s.hu.cb, s.hu.suh, s.hu.svh, wu.data());
    strata::kernels::cpu::exl3_reconstruct_weight(s.hd.trellis, s.hd.ki, s.hd.nj, s.hd.bits,
        (strata::kernels::cpu::Exl3Codebook) s.hd.cb, s.hd.suh, s.hd.svh, wd.data());
    auto project = [](const std::vector<uint16_t>& w, int o, const std::vector<uint16_t>& x,
                      std::vector<uint16_t>& yr) {
        const int kk = (int) x.size();
        yr.assign((size_t) o, 0);
        for (int j = 0; j < o; ++j) {
            double sum = 0;
            for (int i = 0; i < kk; ++i) sum += (double) fp32(x[(size_t) i]) * fp32(w[(size_t) i * o + j]);
            yr[(size_t) j] = f16((float) sum);
        }
    };
    std::vector<uint16_t> xg(x_row, x_row + in), g, u, h, yy;
    project(wg, ff, xg, g);
    project(wu, ff, xg, u);
    h.assign((size_t) ff, 0);
    for (int i = 0; i < ff; ++i) {
        const float v = fp32(g[(size_t) i]);
        h[(size_t) i] = f16(v / (1.0f + std::exp(-v)) * fp32(u[(size_t) i]));
    }
    project(wd, n, h, yy);
    y.assign((size_t) n, 0.0);
    double den = 0;
    for (int i = 0; i < n; ++i) { y[(size_t) i] = fp32(yy[(size_t) i]); den += (double) y[(size_t) i] * y[(size_t) i]; }
    return std::sqrt(den);
}

struct Stats { std::vector<double> span, issue, e2e; std::vector<double> chain; };

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model_dir> [device=0] [trials=30]\n", argv[0]); return 2; }
    const std::string model_dir = argv[1];
    const int device = argc > 2 ? std::atoi(argv[2]) : 0;
    const int trials = argc > 3 ? std::atoi(argv[3]) : 30;
    if (trials < 4) { std::fprintf(stderr, "trials must be >= 4\n"); return 2; }
    try {
        check(cudaSetDevice(device));
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, device));
        char pci[16] = {};
        (void) cudaDeviceGetPCIBusId(pci, sizeof pci, device);
        std::printf("smallm device=%d gpu=%s pci=%s trials=%d\n", device, prop.name, pci, trials);
        std::printf("smallm limit: resident-microbench evidence only - NOT an engine TG measurement; "
                    "commit/rollback and MTP acceptance gates are engine-side and untouched\n");

        // ---- setup (outside every timed region): resident real weights, window x, workspace ----
        Exl3Model model(model_dir);
        ResidentSet rs;
        std::mt19937 rng(20261018u);
        const Clock::time_point load_t0 = Clock::now();
        load_resident(model, rs, rng);
        const double load_ms = ms_between(load_t0, Clock::now());
        size_t resident_bytes = 0;
        for (const auto& s : rs.slots) resident_bytes += s.bytes;
        {   // verify the geometry we claim; refuse anything else
            const ResidentSlot& s0 = rs.slots[0];
            if (s0.hg.in() != WIDTH || s0.hg.out() != HIDDEN || s0.hd.out() != WIDTH ||
                s0.hg.bits != 3 || s0.hg.cb != 2 || s0.hd.bits != 3 || s0.hd.cb != 2)
                throw std::runtime_error("smallm: unexpected expert geometry (expected 2560/640/2560 K3 mul1)");
            for (const auto& s : rs.slots)
                if (s.hg.ki != s0.hg.ki || s.hg.nj != s0.hg.nj || s.hg.bits != 3 || s.hg.cb != 2 ||
                    s.hd.ki != s0.hd.ki || s.hd.nj != s0.hd.nj)
                    throw std::runtime_error("smallm: mixed resident geometry");
            std::printf("smallm resident layers=%d per_layer=%d experts=%zu bytes=%.1f MiB upload_ms=%.0f "
                        "geometry=2560/640/2560 K3 cb=mul1\n",
                        (int) rs.layers.size(), rs.per_layer, rs.slots.size(),
                        (double) resident_bytes / 1048576.0, load_ms);
        }
        // the padded verify-window activation buffer: deterministic varied non-zero fp16
        std::vector<uint16_t> x_host((size_t) X_ROWS * WIDTH);
        for (size_t i = 0; i < x_host.size(); ++i) {
            const float v = ((int) (rng() % 2000) - 1000) * 0.001f;
            x_host[i] = f16((i % 16 == 0) ? v * 4.0f : v);   // a mild outlier cadence like real activations
        }
        uint16_t* d_x = nullptr;
        check(cudaMalloc((void**) &d_x, x_host.size() * 2));
        check(cudaMemcpy(d_x, x_host.data(), x_host.size() * 2, cudaMemcpyHostToDevice));

        cudaStream_t cs = nullptr;
        cudaEvent_t ev0 = nullptr, ev1 = nullptr;
        check(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
        check(cudaEventCreate(&ev0));
        check(cudaEventCreate(&ev1));
        const int max_routes = TOP_K * 6;   // M6
        uint16_t *d_out_a = nullptr, *d_out_b = nullptr;
        int* d_rows = nullptr;
        check(cudaMalloc((void**) &d_out_a, (size_t) max_routes * WIDTH * 2));
        check(cudaMalloc((void**) &d_out_b, (size_t) max_routes * WIDTH * 2));
        check(cudaMalloc((void**) &d_rows, (size_t) max_routes * sizeof(int)));
        struct Guard { ~Guard() { (void) cudaEventDestroy(e0); (void) cudaEventDestroy(e1);
                                  (void) cudaStreamDestroy(s); (void) cudaFree(x);
                                  (void) cudaFree(a); (void) cudaFree(b); (void) cudaFree(r); }
            cudaEvent_t e0, e1; cudaStream_t s; uint16_t *x, *a, *b; int* r; }
        guard{ev0, ev1, cs, d_x, d_out_a, d_out_b, d_rows};

        Exl3MoeWorkspace ws;
        ws.reserve(max_routes, WIDTH, HIDDEN, WIDTH, true);       // both arms' per-route scratch
        ws.reserve_small_m(max_routes, TOP_K * 6, WIDTH, HIDDEN, WIDTH);   // the grouped arm's plan buffers
        std::printf("smallm workspace_bytes=%zu split=%s\n", ws.allocated_bytes(),
                    std::getenv("STRATA_EXL3_GEMV_SPLIT") ? std::getenv("STRATA_EXL3_GEMV_SPLIT") : "default");

        auto run_indexed = [&](const TrialPlan& p) {
            check(cudaMemcpyAsync(d_rows, p.input_rows.data(), (size_t) p.routes * sizeof(int),
                                  cudaMemcpyHostToDevice, cs));
            strata::kernels::exl3_moe_rows_indexed(p.route_g.data(), p.route_u.data(), p.route_d.data(),
                                                   p.routes, d_x, d_rows, d_out_a, (void*) cs, ws);
        };
        auto run_grouped = [&](const TrialPlan& p) {
            check(cudaMemcpyAsync(d_rows, p.input_rows.data(), (size_t) p.routes * sizeof(int),
                                  cudaMemcpyHostToDevice, cs));
            strata::kernels::exl3_moe_rows_grouped(p.group_g.data(), p.group_u.data(), p.group_d.data(),
                                                   p.group_first.data(), p.group_rows.data(), p.groups,
                                                   p.routes, d_x, d_rows, d_out_b, (void*) cs, ws);
        };

        // ---- negative precheck: the prototype must REJECT malformed plans BEFORE enqueueing ----
        {
            TrialPlan plan;
            std::mt19937 neg_rng(4242u);
            std::vector<Exl3Mat> pg((size_t) rs.per_layer), pu((size_t) rs.per_layer), pd((size_t) rs.per_layer);
            for (int s = 0; s < rs.per_layer; ++s) {
                pg[(size_t) s] = rs.slots[(size_t) s].g;
                pu[(size_t) s] = rs.slots[(size_t) s].u;
                pd[(size_t) s] = rs.slots[(size_t) s].d;
            }
            build_valid_plan(pg.data(), pu.data(), pd.data(), rs.per_layer, 2,
                             target_sizes("mix", 2, 0), neg_rng, 0, plan);
            run_indexed(plan);
            check(cudaStreamSynchronize(cs));
            int rejections = 0, enqueues_after_reject = 0;
            auto must_throw = [&](const char* what, std::vector<Exl3Mat> g, std::vector<Exl3Mat> u,
                                  std::vector<Exl3Mat> d, std::vector<int> gf, std::vector<int> gr,
                                  int groups, int routes) {
                bool threw = false;
                try {
                    strata::kernels::exl3_moe_rows_grouped(g.data(), u.data(), d.data(), gf.data(),
                                                           gr.data(), groups, routes, d_x, d_rows,
                                                           d_out_b, (void*) cs, ws);
                } catch (const std::exception&) { threw = true; }
                if (!threw) { std::printf("smallm negative %s: ACCEPTED FAIL\n", what); throw std::runtime_error("negative precheck"); }
                ++rejections;
                if (cudaStreamQuery(cs) == cudaSuccess) return;   // nothing was enqueued
                ++enqueues_after_reject;
            };
            {   // a heap of malformed variants of the one VALID plan
                const int G = plan.groups, R = plan.routes;
                std::vector<int> gf = plan.group_first, gr = plan.group_rows;
                std::vector<Exl3Mat> g = plan.group_g, u = plan.group_u, d = plan.group_d;
                gf[0] = 1;                       must_throw("group_first[0]!=0", g, u, d, gf, gr, G, R);
                gf = plan.group_first; gf[1] = gf[1] + 1;   // non-contiguous: overlaps the next group
                must_throw("non-contiguous-slice", g, u, d, gf, gr, G, R);
                gf = plan.group_first; gr[0] = gr[0] + 1;   // rows no longer sum to routes
                must_throw("rows-do-not-cover-routes", g, u, d, gf, gr, G, R);
                gr = plan.group_rows; gr[0] = 0;           must_throw("zero-row-group", g, u, d, gf, gr, G, R);
                gr = plan.group_rows; gr[0] = SMALL_M_MAX_ROWS + 1;
                must_throw("rows-exceed-compiled-width", g, u, d, gf, gr, G, R);
                {   // mixed geometry: group 1's gate swapped for a down-projection descriptor
                    std::vector<Exl3Mat> bad = g;
                    bad[1] = plan.group_d[1];
                    must_throw("mixed-geometry", bad, u, d, plan.group_first, plan.group_rows, G, R);
                }
                {   // null route table
                    bool threw = false;
                    try {
                        strata::kernels::exl3_moe_rows_grouped(g.data(), u.data(), d.data(), nullptr,
                                                               gr.data(), G, R, d_x, d_rows,
                                                               d_out_b, (void*) cs, ws);
                    } catch (const std::exception&) { threw = true; }
                    if (!threw) { std::printf("smallm negative null-route-table: ACCEPTED FAIL\n");
                                  throw std::runtime_error("negative precheck"); }
                    ++rejections;
                    if (cudaStreamQuery(cs) != cudaSuccess) ++enqueues_after_reject;
                }
            }
            // the workspace must still work after the rejections: the valid call matches arm A
            run_grouped(plan);
            check(cudaStreamSynchronize(cs));
            std::vector<uint16_t> ya((size_t) plan.routes * WIDTH), yb((size_t) plan.routes * WIDTH);
            check(cudaMemcpy(ya.data(), d_out_a, ya.size() * 2, cudaMemcpyDeviceToHost));
            check(cudaMemcpy(yb.data(), d_out_b, yb.size() * 2, cudaMemcpyDeviceToHost));
            const bool identical = std::memcmp(ya.data(), yb.data(), ya.size() * 2) == 0;
            const bool pass = rejections == 7 && enqueues_after_reject == 0 && identical &&
                              cudaStreamQuery(cs) == cudaSuccess;
            std::printf("smallm negative malformed-plans rejected=%d/7 stream_still_empty_after_reject=%d "
                        "workspace_usable=%s %s\n",
                        rejections, enqueues_after_reject == 0 ? 1 : 0, identical ? "byte-identical" : "FAIL",
                        pass ? "PASS" : "FAIL");
            if (!pass) return 1;
        }

        // ---- synthetic-parity probes: the prototype API allows bits 1..8 / cb 0..2, but the real
        // weights are K3/mul1; cover K4/mul1 and K3 mcg/3inst with synthetic planes so the claim is
        // tested, not assumed.  Arm A is itself CPU-validated for these codebooks by
        // exl3_moe_multi_parity; this probe checks the grouped twin is byte-identical to it.
        for (auto [bits, cb] : std::initializer_list<std::pair<int, int>>{{4, 2}, {3, 1}, {3, 0}}) {
            const int pool = 24;
            const size_t tr_words = (size_t) (WIDTH / 16) * (HIDDEN / 16) * (256 * bits / 16);
            const size_t tr_bytes = tr_words * 2;
            std::vector<ResidentSlot> synth((size_t) pool);
            std::mt19937 srng(977u + (unsigned) bits * 31u + (unsigned) cb);
            bool ok = true;
            for (auto& s : synth) {
                s.hg.ki = WIDTH / 16; s.hg.nj = HIDDEN / 16; s.hg.bits = bits; s.hg.cb = cb;
                s.hu = s.hd = s.hg;
                s.hd.ki = HIDDEN / 16; s.hd.nj = WIDTH / 16;
                const size_t bytes = 3 * (tr_bytes + (size_t) WIDTH * 2 + (size_t) HIDDEN * 2);
                s.bytes = bytes;
                check(cudaMalloc((void**) &s.blob, bytes));
                std::vector<uint16_t> packed(tr_words), scales_w((size_t) WIDTH), scales_h((size_t) WIDTH);
                for (auto& v : packed) v = (uint16_t) srng();
                // both scale vectors are WIDTH long: gate/up want 2560 suh, down wants 2560 svh.
                // Magnitudes are small because random weights at the full geometry must stay
                // finite in fp16 through the down projection (real weights are calibrated).
                for (auto& v : scales_w) v = f16(((int) (srng() % 1000) - 500) * 0.001f);
                for (auto& v : scales_h) v = f16(((int) (srng() % 1000) - 500) * 0.001f);
                Exl3Mat* mats[3] = {&s.g, &s.u, &s.d};
                const int in[3] = {WIDTH, WIDTH, HIDDEN}, out[3] = {HIDDEN, HIDDEN, WIDTH};
                size_t off = 0;
                for (int i = 0; i < 3; ++i) {
                    check(cudaMemcpy(s.blob + off, packed.data(), tr_bytes, cudaMemcpyHostToDevice));
                    mats[i]->trellis = (const uint16_t*) (s.blob + off);
                    off += tr_bytes;
                    check(cudaMemcpy(s.blob + off, scales_w.data(), (size_t) in[i] * 2, cudaMemcpyHostToDevice));
                    mats[i]->suh = (const uint16_t*) (s.blob + off);
                    off += (size_t) in[i] * 2;
                    check(cudaMemcpy(s.blob + off, scales_h.data(), (size_t) out[i] * 2, cudaMemcpyHostToDevice));
                    mats[i]->svh = (const uint16_t*) (s.blob + off);
                    off += (size_t) out[i] * 2;
                    mats[i]->ki = in[i] / 16; mats[i]->nj = out[i] / 16;
                    mats[i]->bits = bits; mats[i]->cb = cb;
                }
            }
            struct FreeSynth { std::vector<ResidentSlot>& v; ~FreeSynth() { for (auto& s : v) (void) cudaFree(s.blob); } }
            free_on_exit{synth};
            std::vector<Exl3Mat> pg((size_t) pool), pu((size_t) pool), pd((size_t) pool);
            for (int s = 0; s < pool; ++s) {
                pg[(size_t) s] = synth[(size_t) s].g;
                pu[(size_t) s] = synth[(size_t) s].u;
                pd[(size_t) s] = synth[(size_t) s].d;
            }
            TrialPlan plan;
            build_valid_plan(pg.data(), pu.data(), pd.data(), pool, 3, target_sizes("mix", 3, 0),
                            srng, 0, plan);
            run_indexed(plan);
            check(cudaStreamSynchronize(cs));
            run_grouped(plan);
            check(cudaStreamSynchronize(cs));
            std::vector<uint16_t> ya((size_t) plan.routes * WIDTH), yb((size_t) plan.routes * WIDTH);
            check(cudaMemcpy(ya.data(), d_out_a, ya.size() * 2, cudaMemcpyDeviceToHost));
            check(cudaMemcpy(yb.data(), d_out_b, yb.size() * 2, cudaMemcpyDeviceToHost));
            bool identical = std::memcmp(ya.data(), yb.data(), ya.size() * 2) == 0;
            bool finite = true;
            for (size_t i = 0; i < ya.size(); ++i) if (!std::isfinite(fp32(ya[i]))) finite = false;
            ok = identical && finite;
            std::printf("smallm synthetic K%d cb=%d routes=%d groups=%d grouped_vs_indexed=%s "
                        "finite=%d %s\n",
                        bits, cb, plan.routes, plan.groups, identical ? "byte-identical" : "FAIL",
                        finite, ok ? "PASS" : "FAIL");
            if (!ok) return 1;
        }

        // ---- scenarios ----
        struct Scenario { const char* kind; int m; };
        const std::vector<Scenario> scenarios = {
            {"homo", 1}, {"mix", 1},
            {"homo", 2}, {"mix", 2},
            {"homo", 3}, {"mix", 3},
            {"homo", 4}, {"mix", 4},
            {"homo", 5}, {"mix", 5},
            {"homo", 6}, {"mix", 6},
        };
        for (const Scenario& sc : scenarios) {
            std::vector<TrialPlan> plans((size_t) trials);
            for (int t = 0; t < trials; ++t) {
                const int li = t % (int) rs.layers.size();
                std::vector<Exl3Mat> pg((size_t) rs.per_layer), pu((size_t) rs.per_layer), pd((size_t) rs.per_layer);
                for (int s = 0; s < rs.per_layer; ++s) {
                    pg[(size_t) s] = rs.slots[(size_t) li * (size_t) rs.per_layer + (size_t) s].g;
                    pu[(size_t) s] = rs.slots[(size_t) li * (size_t) rs.per_layer + (size_t) s].u;
                    pd[(size_t) s] = rs.slots[(size_t) li * (size_t) rs.per_layer + (size_t) s].d;
                }
                std::mt19937 prng(7919u + 104729u * (unsigned) sc.m + (sc.kind[0] == 'h' ? 0u : 13u) +
                                  31337u * (unsigned) t);
                build_valid_plan(pg.data(), pu.data(), pd.data(), rs.per_layer, sc.m,
                                 target_sizes(sc.kind, sc.m, t), prng, li, plans[(size_t) t]);
            }
            const TrialPlan& p0 = plans[0];
            const int routes = p0.routes, groups = p0.groups;
            const double reuse = (double) routes / groups;
            int shared = 0;                       // rows in multi-row groups, vs the measured profile
            for (int r : p0.group_rows) if (r > 1) shared += r;
            char hist[96] = {};
            {   int at = 0; bool first = true;
                for (int rows = 1; rows <= 6; ++rows) {
                    const int c = (int) std::count(p0.group_rows.begin(), p0.group_rows.end(), rows);
                    if (!c) continue;
                    at += std::snprintf(hist + at, sizeof hist - at, "%s%dx%d", first ? "" : "/", rows, c);
                    first = false;
                }
            }
            const char* target_note =
                sc.m == 6 ? " (M6 extrapolated, no measurement)" :
                (sc.kind[0] == 'm' && sc.m == 4) ? " (matches the measured prose M4 histogram)" :
                (sc.kind[0] == 'm' && (sc.m == 2 || sc.m == 3)) ?
                    " (two valid arrangements alternated across trials, approximating the measured fraction)" :
                (sc.kind[0] == 'm' && sc.m > 1) ? " (constructed to approximate the measured fraction)" : "";
            std::printf("smallm scenario=%s-M%d routes=%d groups=%d reuse=%.2f rows_in_multigroup=%.1f%% "
                        "hist=%s%s\n",
                        sc.kind, sc.m, routes, groups, reuse, 100.0 * shared / routes, hist, target_note);

            // ---- parity on EVERY trial (outside the timed region) + the stated CPU-oracle probes ----
            const bool oracle_scenario = (sc.m == 1 && sc.kind[0] == 'h') ||
                                         (sc.m == 4 && sc.kind[0] == 'm') ||
                                         (sc.m == 6 && sc.kind[0] == 'h');
            {
                std::vector<uint16_t> ya((size_t) routes * WIDTH), yb((size_t) routes * WIDTH);
                bool all_identical = true, finite = true, varied = false;
                double worst_rel = 0;
                bool oracle_ok = true;
                for (int t = 0; t < trials; ++t) {
                    const TrialPlan& p = plans[(size_t) t];
                    run_indexed(p);
                    check(cudaStreamSynchronize(cs));
                    run_grouped(p);
                    check(cudaStreamSynchronize(cs));
                    check(cudaMemcpy(ya.data(), d_out_a, ya.size() * 2, cudaMemcpyDeviceToHost));
                    check(cudaMemcpy(yb.data(), d_out_b, yb.size() * 2, cudaMemcpyDeviceToHost));
                    all_identical = all_identical && std::memcmp(ya.data(), yb.data(), ya.size() * 2) == 0;
                    for (size_t i = 0; i < ya.size(); ++i) {
                        if (!std::isfinite(fp32(ya[i]))) finite = false;
                        if (i && ya[i] != ya[0]) varied = true;
                    }
                    if (oracle_scenario && t == 0) {   // exactly 2 probes on exactly these scenarios
                        for (int probe : {0, routes / 2}) {
                            // `probe` is a ROUTE; the group CONTAINING it is the last group whose
                            // first is <= probe (group_first is ascending)
                            const int g = (int) (std::upper_bound(p.group_first.begin(),
                                                                  p.group_first.end(), probe) -
                                                 p.group_first.begin()) - 1;
                            const ResidentSlot& slot =
                                rs.slots[(size_t) p.layer_index * (size_t) rs.per_layer +
                                         (size_t) p.group_slot[(size_t) g]];
                            std::vector<float> ref;
                            const double den = cpu_oracle_row(slot,
                                x_host.data() + (size_t) p.input_rows[(size_t) probe] * WIDTH, ref);
                            double num = 0;
                            for (int i = 0; i < WIDTH; ++i) {
                                const double d = fp32(ya[(size_t) probe * WIDTH + i]) - ref[(size_t) i];
                                num += d * d;
                            }
                            const double rel = std::sqrt(num / (den * den + 1e-30));
                            worst_rel = std::max(worst_rel, rel);
                            oracle_ok = oracle_ok && std::isfinite(rel) && rel < 0.02;
                        }
                    }
                }
                const bool pass = all_identical && finite && varied && oracle_ok;
                if (oracle_scenario)
                    std::printf("smallm scenario=%s-M%d parity grouped_vs_indexed=%s on_all=%d trials "
                                "outputs_finite=%d nonconstant=%d cpu_oracle_rel=%.3g (2 probes, this "
                                "scenario) %s\n",
                                sc.kind, sc.m, all_identical ? "byte-identical" : "FAIL", trials,
                                finite, varied, worst_rel, pass ? "PASS" : "FAIL");
                else
                    std::printf("smallm scenario=%s-M%d parity grouped_vs_indexed=%s on_all=%d trials "
                                "outputs_finite=%d nonconstant=%d cpu_oracle=none-this-scenario %s\n",
                                sc.kind, sc.m, all_identical ? "byte-identical" : "FAIL", trials,
                                finite, varied, pass ? "PASS" : "FAIL");
                if (!pass) return 1;
            }

            // ---- interleaved per-call timing: uploads + API call, arms alternating ----
            for (int i = 0; i < 4; ++i) { run_indexed(plans[0]); run_grouped(plans[0]); }
            check(cudaStreamSynchronize(cs));
            Stats a, b;
            for (int t = 0; t < trials; ++t) {
                const TrialPlan& p = plans[(size_t) t];
                const bool grouped_first = (t & 1) != 0;   // order-varied to cancel drift
                for (int arm = 0; arm < 2; ++arm) {
                    const bool grouped = (arm == 0) == grouped_first;
                    const Clock::time_point t0 = Clock::now();
                    check(cudaEventRecord(ev0, cs));
                    if (grouped) run_grouped(p); else run_indexed(p);
                    check(cudaEventRecord(ev1, cs));
                    const Clock::time_point t1 = Clock::now();
                    check(cudaStreamSynchronize(cs));
                    const Clock::time_point t2 = Clock::now();
                    float span = -1;
                    check(cudaEventElapsedTime(&span, ev0, ev1));
                    (grouped ? b : a).span.push_back(span);
                    (grouped ? b : a).issue.push_back(ms_between(t0, t1));
                    (grouped ? b : a).e2e.push_back(ms_between(t0, t2));
                }
            }
            // 48-call back-to-back batches, arm order ALTERNATED AB/BA/AB... (6 per arm).  A
            // queue diagnostic only: the engine consumes each layer's activations before the next
            // window, so this cadence is NOT the engine's and no end-to-end number may be read
            // from it.
            for (int rep = 0; rep < 12; ++rep) {
                const bool grouped_first = (rep & 1) != 0;
                for (int arm = 0; arm < 2; ++arm) {
                    const bool grouped = (arm == 0) == grouped_first;
                    check(cudaEventRecord(ev0, cs));
                    for (int i = 0; i < 48; ++i) {
                        const TrialPlan& p = plans[(size_t) ((rep * 48 + i) % trials)];
                        if (grouped) run_grouped(p); else run_indexed(p);
                    }
                    check(cudaEventRecord(ev1, cs));
                    check(cudaStreamSynchronize(cs));
                    float span = -1;
                    check(cudaEventElapsedTime(&span, ev0, ev1));
                    (grouped ? b : a).chain.push_back(span / 48.0);
                }
            }
            const double am = median(a.e2e), bm = median(b.e2e);
            std::printf("smallm scenario=%s-M%d arm=indexed  e2e_ms=%.4f [iqr %.4f] gpu_span_ms=%.4f "
                        "[iqr %.4f] issue_ms=%.4f batch48_ms=%.4f [iqr %.4f]\n",
                        sc.kind, sc.m, am, iqr(a.e2e), median(a.span), iqr(a.span), median(a.issue),
                        median(a.chain), iqr(a.chain));
            std::printf("smallm scenario=%s-M%d arm=grouped e2e_ms=%.4f [iqr %.4f] gpu_span_ms=%.4f "
                        "[iqr %.4f] issue_ms=%.4f batch48_ms=%.4f [iqr %.4f]\n",
                        sc.kind, sc.m, bm, iqr(b.e2e), median(b.span), iqr(b.span), median(b.issue),
                        median(b.chain), iqr(b.chain));
            std::printf("smallm scenario=%s-M%d verdict grouped_vs_indexed (per-call, PRIMARY) "
                        "e2e=%+.1f%% gpu_span=%+.1f%%\n",
                        sc.kind, sc.m, (bm - am) / am * 100.0,
                        (median(b.span) - median(a.span)) / median(a.span) * 100.0);
        }

        // ---- host route-plan build cost (both arms' per-call HOST work, measured separately).
        // A checksum over EVERY built element is folded in and printed, so the compiler cannot
        // elide the builds and the us-scale numbers are work, not dead stores.
        {
            const int window_m = 4, routes = window_m * TOP_K;
            std::mt19937 hb_rng(4242u);
            std::vector<int32_t> ids((size_t) routes);
            for (int t = 0; t < window_m; ++t) {
                std::vector<int> pick((size_t) rs.per_layer);
                std::iota(pick.begin(), pick.end(), 0);
                std::shuffle(pick.begin(), pick.end(), hb_rng);
                for (int k = 0; k < TOP_K; ++k) ids[(size_t) (t * TOP_K + k)] = pick[(size_t) k];
            }
            std::vector<Exl3Mat> rg((size_t) routes), ru((size_t) routes), rd((size_t) routes);
            std::vector<Exl3Mat> gg((size_t) routes), gu((size_t) routes), gd((size_t) routes);
            std::vector<int> gfirst((size_t) routes), grows((size_t) routes), in_rows((size_t) routes);
            uint64_t sum_idx = 0, sum_grp = 0;
            double t_idx = 0, t_grp = 0;
            int last_groups = 0;
            for (int rep = 0; rep < 400; ++rep) {
                {   // arm A: the plan (counting sort) then per-ROUTE descriptor arrays in the
                    // plan's group-major row order - what the engine's hit arm builds per window
                    const Clock::time_point t0 = Clock::now();
                    Exl3RoutePlan plan;
                    plan.build(ids.data(), window_m, TOP_K, rs.per_layer);
                    for (int i = 0; i < routes; ++i) {
                        const ResidentSlot& slot = rs.slots[(size_t) ids[(size_t) plan.destinations[(size_t) i]]];
                        rg[(size_t) i] = slot.g; ru[(size_t) i] = slot.u; rd[(size_t) i] = slot.d;
                        in_rows[(size_t) i] = plan.token_at((size_t) i);
                    }
                    uint64_t sum = 0;
                    for (int i = 0; i < routes; ++i)
                        sum += (uint64_t) (uintptr_t) rg[(size_t) i].trellis + (uint64_t) in_rows[(size_t) i];
                    sum_idx += sum;
                    last_groups = (int) plan.experts.size();
                    t_idx += ms_between(t0, Clock::now());
                }
                {   // arm B: the same plan, then GROUP tables (they ARE the plan's offsets) +
                    // per-GROUP descriptor arrays - reuse-factor fewer copies
                    const Clock::time_point t0 = Clock::now();
                    Exl3RoutePlan plan;
                    plan.build(ids.data(), window_m, TOP_K, rs.per_layer);
                    const int gcount = (int) plan.experts.size();
                    for (int g = 0; g < gcount; ++g) {
                        const ResidentSlot& slot = rs.slots[(size_t) plan.experts[(size_t) g]];
                        gg[(size_t) g] = slot.g; gu[(size_t) g] = slot.u; gd[(size_t) g] = slot.d;
                        gfirst[(size_t) g] = plan.offsets[(size_t) g];
                        grows[(size_t) g] = plan.offsets[(size_t) g + 1] - plan.offsets[(size_t) g];
                    }
                    for (int i = 0; i < routes; ++i) in_rows[(size_t) i] = plan.token_at((size_t) i);
                    uint64_t sum = 0;
                    for (int g = 0; g < gcount; ++g)
                        sum += (uint64_t) (uintptr_t) gg[(size_t) g].trellis + (uint64_t) gfirst[(size_t) g] +
                               (uint64_t) grows[(size_t) g];
                    sum_grp += sum;
                    t_grp += ms_between(t0, Clock::now());
                }
            }
            std::printf("smallm hostbuild scenario=router-M4 routes=%d groups=%d indexed_us=%.2f "
                        "grouped_us=%.2f (per window, 400 reps, the counting-sort plan included in "
                        "both arms, full-array checksums %llu/%llu nonzero=%d)\n",
                        routes, last_groups, t_idx * 1000.0 / 400.0, t_grp * 1000.0 / 400.0,
                        (unsigned long long) sum_idx, (unsigned long long) sum_grp,
                        sum_idx != 0 && sum_grp != 0 ? 1 : 0);
        }

        std::printf("smallm done (resident microbench, NOT engine TG)\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "smallm FAIL: %s\n", e.what());
        return 1;
    }
}
