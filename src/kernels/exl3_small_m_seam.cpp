// src/kernels/exl3_small_m_seam.cpp - STRATA_EXL3_SMALL_M engine-seam check (opt-in grouped small-M).
//
// Validates the OPT-IN integration of exl3_moe_rows_grouped into the host-IO expert service
// (run_multi - the verifier decode windows' path) on a synthetic model directory
// (tools/test_exl3_loader.py writes it: layer 0 = 80 experts 2560->640->2560 K4 mul1 with
// DISTINCT weights per (expert, projection); layer 1 = 16 experts with a DIFFERENT FFN width,
// 2560->128->2560, so successive passes with different descriptor geometry are exercised).
//
// Four arms run the SAME window battery in the SAME order, so their caches evolve identically:
//   OFF    - no flags (the untouched default)
//   SM     - STRATA_EXL3_SMALL_M=1
//   OV     - STRATA_EXL3_OVERLAP=1 (hit/miss split waves)
//   SM+OV  - both
// Every window's output must be BYTE-IDENTICAL across all four arms (the grouped FFN is
// arithmetic-identical to the indexed GEMV; a window that stays on the indexed path is identical
// by construction).  With a 4-slot cache a window decomposes into 4-group waves and each wave
// (or split pass) makes its own dispatch decision, so the dispatch assertions are ranges, with
// every stay reason required to have fired: the M1 window (window<2), the M=7 homogeneous
// window (groups wider than the compiled 6) and pure singleton passes (reuse < 1.2) stay indexed,
// while eligible M2..M6 mixed passes (valid token-major top-10 assignments built by
// largest-remaining-demand, the kernel bench's measured shapes) must dispatch grouped many
// times over their waves.  The flag-off arms must never dispatch.  The workspace must not GROW
// after warmup (the pre-reserve-under-queued-work contract).
//
// Shape coverage: cold (all-miss) first window, warm repeats (partial hits), a primed
// guaranteed all-hit window (4 experts warmed through the 4-slot cache, M=4 reuse-4 shape),
// mixed hit/miss windows via cache thrash, M1 indexed regression, M2..M6 grouped mixes,
// an M=7 window whose groups hold 7 rows (the safe indexed fallback), duplicate expert ids
// inside one token (a group with a repeated token row), a zero-route call, layer-1 windows
// (different FFN geometry) after layer-0 windows, and repeated windows.  Exits 77 when no CUDA
// device is present; every other failure fails.
//
// Usage: exl3_small_m_seam <model_dir>
#include "strata/kernels/exl3_experts.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr int kEmb = 2560, kSlots = 4, kTop = 10;

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    (void) _putenv_s(name, value);
#else
    (void) setenv(name, value, 1);
#endif
}
void clear_env(const char* name) {
#if defined(_WIN32)
    (void) _putenv_s(name, "=");
#else
    (void) unsetenv(name);
#endif
}

// CONSTRUCTED rows-per-group targets (the kernel bench's measured shapes).  Every size <= M and
// the sum is TOP*M, so a valid token-major top-10 assignment realizes them.
std::vector<int> mix_sizes(int m) {
    auto rep = [](int rows, int times) { return std::vector<int>((size_t) times, rows); };
    std::vector<int> p;
    auto add = [&p](std::vector<int> e) { p.insert(p.end(), e.begin(), e.end()); };
    switch (m) {
        case 2: add(rep(1, 12)); add(rep(2, 4)); break;                        // reuse 1.25: eligible
        case 3: add(rep(1, 14)); add(rep(2, 5)); add(rep(3, 2)); break;         // reuse 1.43: grouped
        case 4: add(rep(1, 17)); add(rep(2, 5)); add(rep(3, 3)); add(rep(4, 1)); break;   // 1.54
        case 5: add(rep(1, 20)); add(rep(2, 4)); add(rep(3, 3)); add(rep(4, 2)); add(rep(5, 1)); break;  // 1.67
        case 6: add(rep(1, 23)); add(rep(2, 6)); add(rep(3, 4)); add(rep(4, 2)); add(rep(5, 1)); break;  // 1.71
        default: break;
    }
    return p;
}

// A VALID token-major top-10 assignment realizing `targets` over [0, span): each token picks 10
// DISTINCT experts by largest remaining demand.  Same construction as the kernel bench.  An
// infeasible target set (more groups than slots, or a token that cannot complete 10 distinct
// picks) throws instead of corrupting memory - the fixture is the checker's own input.
std::vector<int32_t> build_ids(const std::vector<int>& targets, int span, unsigned seed) {
    if ((int) targets.size() > span)
        throw std::runtime_error("build_ids: more target groups than expert slots");
    std::mt19937_64 rng(seed);
    std::vector<int> order((size_t) span);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<int> demand((size_t) span, 0);
    for (size_t g = 0; g < targets.size(); ++g) demand[(size_t) order[g]] = targets[g];
    const int m = std::accumulate(targets.begin(), targets.end(), 0) / kTop;
    std::vector<int32_t> ids((size_t) m * kTop);
    for (int t = 0; t < m; ++t) {
        std::set<int> used;
        for (int j = 0; j < kTop; ++j) {
            int best = -1;
            for (int s = 0; s < span; ++s)
                if (demand[(size_t) s] > 0 && used.count(s) == 0 &&
                    (best < 0 || demand[(size_t) s] > demand[(size_t) best]))
                    best = s;
            if (best < 0) throw std::runtime_error("build_ids: infeasible target histogram");
            used.insert(best);
            ids[(size_t) (t * kTop + j)] = best;
            --demand[(size_t) best];
        }
    }
    return ids;
}

// The priming window: every token repeats the SAME four experts, so the 4-slot cache holds
// exactly them afterwards and the FOLLOWING identical window is all-hit.  Reuse 16/4 = 4.
std::vector<int32_t> prime_ids() {
    std::vector<int32_t> ids((size_t) 4 * kTop);
    for (int t = 0; t < 4; ++t)
        for (int j = 0; j < kTop; ++j) ids[(size_t) (t * kTop + j)] = 7 + (j % 4);
    return ids;
}

std::vector<float> build_x(int rows, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::vector<float> x((size_t) rows * kEmb);
    for (auto& v : x) v = ((int) (rng() % 2000) - 1000) * 0.001f;
    return x;
}

struct Arm {
    std::string name;
    std::vector<std::vector<float>> out;   // one entry per battery step
    long long sm_calls = 0, sm_routes = 0, sm_stays[3] = {0, 0, 0};
    size_t ws_after_warmup = 0;
};

bool battery(const std::string& dir, Arm& arm, std::string* err) {
    clear_env("STRATA_EXL3_SMALL_M");
    clear_env("STRATA_EXL3_OVERLAP");
    clear_env("STRATA_EXL3_PRIMARY_MULTIROW");
    clear_env("STRATA_EXL3_STAGE_INLINE");
    clear_env("STRATA_EXL3_STAGE_WAKE_LIMIT");
    clear_env("STRATA_EXL3_PRIMARY_PACK_PLAN");
    if (arm.name.find("PACK") != std::string::npos) set_env("STRATA_EXL3_PRIMARY_PACK_PLAN", "1");
    if (arm.name.find("WAKE") != std::string::npos) set_env("STRATA_EXL3_STAGE_WAKE_LIMIT", "1");
    if (const auto at = arm.name.find("INLINE"); at != std::string::npos) {
        const auto cap = arm.name.substr(at + 6);
        set_env("STRATA_EXL3_STAGE_INLINE", cap.c_str());
    }
    if (arm.name.rfind("OV+", 0) == 0) set_env("STRATA_EXL3_OVERLAP", "1");
    if (arm.name == "SM" || arm.name.rfind("SM+",0)==0) set_env("STRATA_EXL3_SMALL_M", "1");
    if (arm.name == "OV" || arm.name.find("+OV")!=std::string::npos) set_env("STRATA_EXL3_OVERLAP", "1");
    if (arm.name.rfind("F2",0)==0 || arm.name.rfind("F4",0)==0) {
        set_env("STRATA_EXL3_PRIMARY_MULTIROW", arm.name[1]=='2' ? "2" : "4");
        if (arm.name.find("OV")!=std::string::npos) set_env("STRATA_EXL3_OVERLAP","1");
    }
    strata::kernels::Exl3ExpertStream s(dir, kEmb, 2, kSlots, nullptr);
    const std::vector<float> x = build_x(9, 4242u);
    long long calls0 = 0, routes0 = 0, stays0[3] = {0, 0, 0};
    size_t ws0 = 0;
    s.small_m_debug(&calls0, &routes0, stays0, &ws0);
    std::vector<int32_t> ids_dup = build_ids(mix_sizes(3), 80, 99u);
    ids_dup[1] = ids_dup[0];   // duplicate expert inside token 0: a group row set with a repeat
    const std::vector<int32_t> prime = prime_ids();
    struct Step { const char* name; int layer; const std::vector<int32_t>* ids; int tokens; };
    const std::vector<int32_t> m1 = build_ids(std::vector<int>(kTop, 1), 80, 1u);
    const std::vector<int32_t> m2 = build_ids(mix_sizes(2), 80, 2u);
    const std::vector<int32_t> m3 = build_ids(mix_sizes(3), 80, 3u);
    const std::vector<int32_t> m4 = build_ids(mix_sizes(4), 80, 4u);
    const std::vector<int32_t> m5 = build_ids(mix_sizes(5), 80, 5u);
    const std::vector<int32_t> m6 = build_ids(mix_sizes(6), 80, 6u);
    const std::vector<int32_t> m7 = build_ids(std::vector<int>(kTop, 7), 80, 5u);
    // Layer 1 has only 16 experts, so its windows use feasible shapes: homogeneous (all tokens
    // pick the same 10 experts - always realizable) plus one 13-group M3 mix (six 3s, five 2s,
    // two 1s - hand-checked feasible for 3 tokens x 10 picks over 16 slots).
    const std::vector<int32_t> l1h4 = build_ids(std::vector<int>(kTop, 4), 16, 8u);
    const std::vector<int32_t> l1h6 = build_ids(std::vector<int>(kTop, 6), 16, 9u);
    std::vector<int> l1mix{3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1};
    const std::vector<int32_t> l1m3 = build_ids(l1mix, 16, 10u);
    const Step steps[] = {
        {"warm/cold-M4", 0, &m4, 4},   // cold: every group a miss; also the workspace baseline
        {"warm-M4", 0, &m4, 4},        // warm repeat of the same window (partial hits)
        {"M1", 0, &m1, 1},             // stays indexed: window tokens < 2
        {"M2", 0, &m2, 2},             // mixed passes: eligible when reuse >= 1.2
        {"M3", 0, &m3, 3},             // grouped over the window's waves
        {"M4", 0, &m4, 4},
        {"M5", 0, &m5, 5},
        {"M6", 0, &m6, 6},
        {"M7-homo", 0, &m7, 7},        // stays indexed: 7-row groups exceed the compiled width
        {"M3-dup", 0, &ids_dup, 3},   // grouped, one group row set repeats a token
        {"L1-homo-M4", 1, &l1h4, 4},   // a different FFN geometry after layer 0, grouped
        {"L1-homo-M6", 1, &l1h6, 6},
        {"L1-M3-mix", 1, &l1m3, 3},    // a mixed histogram at the other geometry, grouped
        {"prime-4", 0, &prime, 4},     // reuse-4 shape; leaves exactly 4 experts resident
        {"all-hit", 0, &prime, 4},     // guaranteed all-hit window on the warmed 4 experts
        {"repeat-M4", 0, &m4, 4},      // repeated window after everything else
    };
    for (const Step& st : steps) {
        std::vector<float> y((size_t) st.tokens * kTop * kEmb, -1.0f);
        if (!s.run_multi(st.layer, x.data(), st.ids->data(), st.tokens, kTop, y.data(), nullptr)) {
            *err = std::string(st.name) + ": run_multi failed";
            return false;
        }
        arm.out.push_back(std::move(y));
    }
    if (!s.run_multi(0, x.data(), nullptr, 0, kTop, nullptr, nullptr)) {   // zero routes: a no-op
        *err = "zero-route window failed";
        return false;
    }
    // The pre-reserve contract where it binds: a shape's FIRST call may grow the workspace at its
    // call start (drained stream, before anything queues) - the M6 window's 60-route reserve does
    // exactly that, legally.  Once a shape has been seen, no later window may grow anything:
    // repeat the three largest shapes and require the byte count to hold EXACTLY.
    s.small_m_debug(nullptr, nullptr, nullptr, &arm.ws_after_warmup);
    const Step repeats[] = {
        {"post-M6", 0, &m6, 6},
        {"post-all-hit", 0, &prime, 4},
        {"post-M4", 0, &m4, 4},
    };
    for (const Step& st : repeats) {
        std::vector<float> y((size_t) st.tokens * kTop * kEmb, -1.0f);
        if (!s.run_multi(st.layer, x.data(), st.ids->data(), st.tokens, kTop, y.data(), nullptr)) {
            *err = std::string(st.name) + ": run_multi failed";
            return false;
        }
        arm.out.push_back(std::move(y));
    }
    long long calls1 = 0, routes1 = 0, stays1[3] = {0, 0, 0};
    size_t ws1 = 0;
    s.small_m_debug(&calls1, &routes1, stays1, &ws1);
    arm.sm_calls = calls1 - calls0;
    arm.sm_routes = routes1 - routes0;
    for (int i = 0; i < 3; ++i) arm.sm_stays[i] = stays1[i] - stays0[i];
    if (ws1 != arm.ws_after_warmup) {   // the pre-reserve contract: repeats must not allocate
        char buf[128];
        std::snprintf(buf, sizeof buf, "workspace grew on repeated shapes: %zu -> %zu",
                      arm.ws_after_warmup, ws1);
        *err = buf;
        return false;
    }
    return true;
}

bool compare(const std::vector<float>& a, const std::vector<float>& b, const char* what) {
    if (a.size() != b.size()) { std::printf("seam %s: size mismatch FAIL\n", what); return false; }
    for (size_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { std::printf("seam %s: non-finite FAIL\n", what); return false; }
    if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0) {
        std::printf("seam %s: NOT byte-identical FAIL\n", what);
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model_dir>\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { std::puts("SKIP: no CUDA device"); return 77; }
    try {
        std::vector<Arm> arms = {{"OFF", {}}, {"SM", {}}, {"OV", {}}, {"SM+OV", {}}};
        if (strata::kernels::exl3_fused_multirow_supported()) {
            arms.push_back({"F2",{}}); arms.push_back({"F2+OV",{}});
            arms.push_back({"F4",{}}); arms.push_back({"F4+OV",{}});
        }
        bool ok = true;
        for (auto& arm : arms) {
            std::string err;
            if (!battery(dir, arm, &err)) {
                std::printf("seam arm %s: FAIL (%s)\n", arm.name.c_str(), err.c_str());
                ok = false;
            }
        }
        if (!ok) return 1;
        const std::vector<const char*> names = {"warm/cold-M4", "warm-M4", "M1", "M2", "M3", "M4", "M5",
                                                "M6", "M7-homo", "M3-dup", "L1-homo-M4", "L1-homo-M6",
                                                "L1-M3-mix", "prime-4", "all-hit", "repeat-M4",
                                                "post-M6", "post-all-hit", "post-M4"};
        if (arms[0].out.size() != names.size()) { std::printf("seam: step count mismatch FAIL\n"); return 1; }
        for (size_t st = 0; st < names.size(); ++st) {
            bool same = true;
            for (size_t a = 1; a < arms.size(); ++a)
                same = compare(arms[0].out[st], arms[a].out[st], names[st]) && same;
            if (!same) return 1;
            std::printf("seam %s: %zu arms byte-identical PASS\n", names[st], arms.size());
        }
        std::printf("seam dispatch: OFF grouped=%lld, SM grouped=%lld routes=%lld, OV grouped=%lld, "
                    "SM+OV grouped=%lld; SM stays window<2:%lld group>6:%lld reuse<1.2:%lld\n",
                    arms[0].sm_calls, arms[1].sm_calls, arms[1].sm_routes, arms[2].sm_calls,
                    arms[3].sm_calls, arms[1].sm_stays[0], arms[1].sm_stays[1], arms[1].sm_stays[2]);
        if (arms[0].sm_calls != 0 || arms[2].sm_calls != 0) {
            std::printf("seam dispatch: a flag-off arm dispatched the grouped path FAIL\n");
            return 1;
        }
        // The M3..M6/L1/prime windows must dispatch grouped many times over their waves (each is
        // 6-8 waves of 4 groups at reuse >= 1.43); at least 8 grouped dispatches is the floor.
        if (arms[1].sm_calls < 8 || arms[1].sm_routes <= 0) {
            std::printf("seam dispatch: SM arm dispatched too rarely (%lld calls) FAIL\n", arms[1].sm_calls);
            return 1;
        }
        if (arms[1].sm_stays[0] < 1 || arms[1].sm_stays[1] < 1 || arms[1].sm_stays[2] < 1) {
            std::printf("seam dispatch: window/group-width/reuse stay reasons did not all fire FAIL\n");
            return 1;
        }
        if (arms[3].sm_calls <= 0) {   // the overlap arm's split passes also dispatch grouped
            std::printf("seam dispatch: SM+OV arm never dispatched FAIL\n");
            return 1;
        }
        for (size_t i=4;i<arms.size();++i)
            if (arms[i].sm_calls<8 || arms[i].sm_routes<=0) {
                std::printf("seam fused primary %s did not dispatch FAIL\n",arms[i].name.c_str()); return 1;
            }
        // Compare the scheduler against the pool, with packed experts so one
        // job really is one expert. Cold and warm batteries alternate inline
        // and worker waves; cap8 covers every four-slot cold wave.
        set_env("STRATA_EXL3_PACK_STAGE", "1");
        set_env("STRATA_EXL3_PREFILL_BORROW", "1");
        std::vector<Arm> stages = {{"OFF",{}}, {"INLINE1",{}}, {"INLINE2",{}},
                                  {"INLINE4",{}}, {"INLINE8",{}}, {"OV+INLINE2",{}}, {"OV+INLINE8",{}},
                                  {"WAKE",{}}, {"WAKE+INLINE2",{}}, {"OV+WAKE+INLINE2",{}},
                                  {"PACK",{}}, {"OV+PACK",{}}, {"OV+PACK+INLINE2",{}}, {"SM+OV+PACK",{}}};
        if (strata::kernels::exl3_fused_multirow_supported()) {
            stages.push_back({"F2+OV+PACK",{}});
            stages.push_back({"F4+OV+PACK+INLINE2",{}});
        }
        for (auto& arm : stages) {
            std::string err;
            if (!battery(dir, arm, &err)) {
                std::printf("stage seam arm %s: FAIL (%s)\n", arm.name.c_str(), err.c_str());
                return 1;
            }
        }
        for (size_t st = 0; st < stages[0].out.size(); ++st)
            for (size_t a = 1; a < stages.size(); ++a)
                if (!compare(stages[0].out[st], stages[a].out[st], names[st])) return 1;
        std::printf("stage seam: %zu pool/inline/wake/overlap arms byte-identical, workspace stable PASS\n", stages.size());
        std::printf("seam all: OPT-IN grouped small-M engine seam PASS\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "seam FAIL: %s\n", e.what());
        return 1;
    }
}
