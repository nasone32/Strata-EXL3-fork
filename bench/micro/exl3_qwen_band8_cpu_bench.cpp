// bench/micro/exl3_qwen_band8_cpu_bench.cpp - canonical band-8 vs native CPU expert service at
// the REAL Qwen3.8-Flash-Next EXL3 4 bpw geometry (2026-11-08 stage-1 feasibility microbench).
//
//     (a) g++ -std=c++20 -O3 -pthread -I include -I src -c src/kernels/cpu/exl3_moe.cpp
//         with -o /tmp/opencode/qwen-band8/exl3_moe.o
//     (b) g++ -std=c++20 -O3 -Wall -pthread -I include -I src
//         bench/micro/exl3_qwen_band8_cpu_bench.cpp plus (a)'s object, linked to
//         /tmp/opencode/qwen-band8/exl3_qwen_band8_cpu_bench
//     Run: /tmp/opencode/qwen-band8/exl3_qwen_band8_cpu_bench [--smoke]
//
// WHY THIS BENCH EXISTS. The 2026-11-06 rotating-corpus CPU numbers
// (notes/EXL3-BAND8-FEASIBILITY-2026-11-06.md, notes/results/2026-11-06-packed-cpu-partition/)
// measured MiMo's K2/K2.5 rates at 4096/2048; the Qwen K3/K4 CPU sample was a cache-hot
// three-expert check only ("no meaningful speedup, 0.989-1.013x"). Nothing measured the
// band-8 vs native CPU service at THIS model's actual geometry (k=2560->640->2560, all
// projections K4 - verified from all nine safetensors headers, 2026-11-08: 48 layers x 512
// experts, gate/up trellis [160,40,64], down [40,160,64], all mul1, no frac tiles; a 49th
// mtp.layers.0 block of 512 experts exists in the file and stays outside the engine's
// model.language_model.layers.* expert stream).
//
// WHAT IT TIMES. The CPU overflow tier's real job shapes on a rotating corpus LARGER than
// this 7950X's 64 MiB L3, so trellis bytes actually stream from DRAM like a cold miss does:
//   t1g1/t1g2/t1g4/t1g8 - decode-shaped: 1 token, G unique experts (t1g2 = the bounded
//                          CPU hybrid policy's "quota 2 groups" shape)
//   t4g1/t4g2/t4g4      - MTP verify-window shapes (5-token spec4 windows: 2-row groups,
//                          2+2, one 2-row + three 1-row)
//   t4worst             - 4 rows of ONE expert (the feasibility table's counterexample
//                          shape where band-8 lost 8-12% at K2)
// run_rows_bg submit->join, NO fake GPU sleep (the joint CPU+GPU staging question is already
// answered by the 2026-11-06 joint matrix; this is the CPU kernel/stream question only).
//
// ARMS are counterbalanced: at EVERY worker config BOTH orders (native->band8 and
// band8->native) run, one format resident at a time, identical corpus bytes per arm (same
// mt19937 seed; band-8 is a pure permutation of the native stream). Workers: W=8 (the MiMo
// launcher's pool) and W=16 (the Qwen engine's default: one per physical core on the
// 7950X). Bitwise native==band-8 parity is asserted once per worker config (already proven
// wholesale by exl3_cpu_band8_check at this exact geometry; this is a cheap tripwire).
//
// Scope limits, deliberately: CPU-only (no GPU context), synthetic trellis bytes (random
// u16 words; the band-8 vs native delta is a memory-layout effect, not a value effect -
// parity is bitwise), 96-expert corpus (~227 MiB) so peak RSS stays well under 1 GiB. No
// model files are read; nothing here is a model-throughput claim.
#include "strata/kernels/cpu/exl3_moe.hpp"
#include "kernels/cpu/exl3_moe_fixture.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

using strata::kernels::cpu::Exl3CpuMatrix;
using strata::kernels::cpu::Exl3CpuMoe;
using strata::kernels::cpu::Exl3CpuTrellisLayout;
using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point t)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

uint16_t f32_to_f16_bits(float v)
{
    uint32_t x;
    std::memcpy(&x, &v, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00u);
    if (((x >> 23) & 0xff) == 0) return (uint16_t)sign;
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) return (uint16_t)sign;
    man >>= 13;
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | man);
    const uint32_t dropped = x & 0x1fffu;
    if (dropped > 0x1000u || (dropped == 0x1000u && (h & 1))) ++h;
    return h;
}

float f16_bits_to_f32(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t man = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0)
    {
        if (man == 0) bits = sign;
        else
        {
            uint32_t e = 0, m = man;
            while ((m & 0x400u) == 0) { m <<= 1; ++e; }
            bits = sign | ((127 - 15 - e + 1) << 23) | ((m & 0x3ffu) << 13);
        }
    }
    else if (exp == 0x1fu)
        bits = sign | 0x7f800000u | (man << 13);
    else
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

// ---- the real 4 bpw geometry (headers, 2026-11-08): hidden 2560, FF 640, every routed
// projection K4 mul1. gate/up k=2560 n=640 (trellis [160,40,64]); down k=640 n=2560 ([40,160,64]).
constexpr int H = 2560, FF = 640, EXPERTS = 96;   // 96 x 2.4768 MB ~ 227 MiB rotating corpus
constexpr int BITS = 4, HB = 0;                    // uniform K4 across all 48x512 routed experts

size_t plane_words(int k, int n) { return (size_t)(k / 16) * (n / 16) * (16 * BITS + 8 * HB); }

struct Corpus
{
    std::vector<std::vector<uint16_t>> t[3];      // per expert, per plane
    std::vector<std::vector<uint16_t>> suh[3], svh[3];
    double pack_ms = 0.0;
    double bytes_per_expert = 0.0;
};

void materialize(Corpus& c, bool packed)
{
    const int dk[3] = { H, H, FF }, dn[3] = { FF, FF, H };
    std::mt19937 rng(20261108);                   // same bytes for both arms; band-8 permutes them
    c.pack_ms = 0.0;
    c.bytes_per_expert = 0.0;
    for (int p = 0; p < 3; ++p)
        c.bytes_per_expert += plane_words(dk[p], dn[p]) * 2.0 + (double)(dk[p] + dn[p]) * 2.0;
    for (int p = 0; p < 3; ++p)
    {
        c.t[p].assign(EXPERTS, {});
        c.suh[p].assign(EXPERTS, {});
        c.svh[p].assign(EXPERTS, {});
        for (int e = 0; e < EXPERTS; ++e)
        {
            const size_t words = plane_words(dk[p], dn[p]);
            c.t[p][(size_t)e].resize(words);
            for (auto& v : c.t[p][(size_t)e]) v = (uint16_t)rng();
            if (packed)
            {
                std::vector<uint16_t> staged = std::move(c.t[p][(size_t)e]);
                c.t[p][(size_t)e].resize(words);
                const auto tp = Clock::now();
                strata::kernels::cpu::exl3cpu_fixture_pack_band8(c.t[p][(size_t)e].data(),
                                                                 staged.data(), dk[p], dn[p], BITS, HB);
                c.pack_ms += ms_since(tp);
            }
            c.suh[p][(size_t)e].resize((size_t)dk[p]);
            c.svh[p][(size_t)e].resize((size_t)dn[p]);
            // the GPU prototype's scale fixture: +- (0.10 + (rng%32)*0.001)
            for (auto& v : c.suh[p][(size_t)e]) v = f32_to_f16_bits((0.10f + (rng() % 32) * 0.001f) * (rng() % 2 ? 1 : -1));
            for (auto& v : c.svh[p][(size_t)e]) v = f32_to_f16_bits((0.10f + (rng() % 32) * 0.001f) * (rng() % 2 ? 1 : -1));
        }
    }
}

// ---- job patterns (t1: G unique experts; t4: whole-group MTP window shapes)
template <int G>
void build_t1(int i, int round, std::vector<int>& ids, std::vector<int>& t, std::vector<int>& r)
{
    const int base = (i * G + round * 3) % EXPERTS;
    ids.resize(G); t.resize(G); r.resize(G);
    for (int g = 0; g < G; ++g) { ids[(size_t)g] = (base + g) % EXPERTS; t[(size_t)g] = 0; r[(size_t)g] = g; }
}
void build_t4g1(int i, int round, std::vector<int>& ids, std::vector<int>& t, std::vector<int>& r)
{
    const int base = (i * 7 + round * 3) % EXPERTS;
    ids = { base }; t = { 0, 1 }; r = { 0, 1 };                    // one whole group, 2 rows
}
void build_t4g2(int i, int round, std::vector<int>& ids, std::vector<int>& t, std::vector<int>& r)
{
    const int base = (i * 7 + round * 3) % EXPERTS;
    ids = { base, (base + 1) % EXPERTS }; t = { 0, 1, 2, 3 }; r = { 0, 1, 2, 3 };   // two 2-row groups
}
void build_t4g4(int i, int round, std::vector<int>& ids, std::vector<int>& t, std::vector<int>& r)
{
    const int base = (i * 7 + round * 3) % EXPERTS;
    ids = { base, (base + 1) % EXPERTS, (base + 2) % EXPERTS, (base + 3) % EXPERTS, base };
    t = { 0, 1, 1, 2, 3 }; r = { 0, 1, 2, 3, 4 };                  // one 2-row + three 1-row groups
}
void build_t4worst(int i, int round, std::vector<int>& ids, std::vector<int>& t, std::vector<int>& r)
{
    const int base = (i * 7 + round * 3) % EXPERTS;
    ids.assign(4, base); t = { 0, 1, 2, 3 }; r = { 0, 1, 2, 3 };   // 4 rows of one expert
}

struct Pattern
{
    const char* name;
    void (*build)(int, int, std::vector<int>&, std::vector<int>&, std::vector<int>&);
    int experts_per_job;
    int rows_per_job;
};
const Pattern PATTERNS[] = {
    { "t1g1",    build_t1<1>,  1, 1 },
    { "t1g2",    build_t1<2>,  2, 2 },
    { "t1g4",    build_t1<4>,  4, 4 },
    { "t1g8",    build_t1<8>,  8, 8 },
    { "t4g1",    build_t4g1,   1, 2 },
    { "t4g2",    build_t4g2,   2, 4 },
    { "t4g4",    build_t4g4,   4, 5 },
    { "t4worst", build_t4worst, 1, 4 },
};

double pct(std::vector<double> v, double p)
{
    std::sort(v.begin(), v.end());
    const size_t idx = (size_t)std::min<double>((double)(v.size() - 1), std::ceil(p * (v.size() - 1)));
    return v[(size_t)idx];
}

long peak_rss_kib()
{
    std::ifstream f("/proc/self/status");
    std::string key; long val = 0; std::string unit;
    while (f >> key) if (key == "VmHWM:") { f >> val >> unit; return val; }
    return 0;
}

long mem_available_kib()
{
    std::ifstream f("/proc/meminfo");
    std::string key; long val = 0;
    while (f >> key) if (key == "MemAvailable:") { f >> val; return val; }
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    const bool smoke = argc > 1 && std::strcmp(argv[1], "--smoke") == 0;
    const int worker_cfgs[] = { 8, 16 };
    const int jobs_per_round = smoke ? 8 : 48;
    const int rounds = smoke ? 1 : 3;

    // The declared PHYSICAL layout must be canonical regardless of these flags (they gate
    // optional conversions of NATIVE bytes); forcing them off makes the native arm the
    // engine's "native layout / swizzle0 / cache0" CPU configuration, matching the recorded
    // single-GPU+CPU 4 bpw run.
    setenv("STRATA_EXL3_CPU_SWIZZLE", "0", 1);
    setenv("STRATA_EXL3_CPU_CACHE_MIB", "0", 1);

    const long avail = mem_available_kib();
    std::printf("QWEN-BAND8-CPU-BENCH env=(CACHE_MIB=0,SWIZZLE=0) geometry=Qwen3.8-NEXT-exl3-4bpw "
                "H=%d FF=%d K%d.%s experts=%d(corpus %0.1f MiB, model has 48x512+MTP) "
                "isa_note=band-8 needs AVX-512BW MemAvailable_GiB=%.1f\n",
                H, FF, BITS, HB ? "5" : "0", EXPERTS,
                EXPERTS * (3.0 * (plane_words(H, FF) * 2.0 + (H + FF) * 2.0)) / 1048576.0,
                avail / 1048576.0);
    if (avail < 4LL << 20)
    {
        std::printf("REFUSED: less than 4 GiB MemAvailable\n");
        return 1;
    }

    // activations: the GPU prototype's f16 pattern, fed to the CPU service as fp32
    std::vector<float> x((size_t)4 * H);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = f16_bits_to_f32(f32_to_f16_bits(((int)(i % 97) - 48) * 0.001f));

    std::unique_ptr<Exl3CpuMoe> svc;
    Corpus corpus;
    std::vector<Exl3CpuMatrix> gmat((size_t)EXPERTS), umat((size_t)EXPERTS), dmat((size_t)EXPERTS);
    std::vector<float> out((size_t)8 * H, 0.0f);
    std::vector<int> ids, tk, rr;

    struct Cell { double submit, total, join, p25, p75; long long native_reads; };
    std::map<std::string, std::map<std::string, Cell>> cells;   // [pattern][arm] pooled over orders
    int parity_failures = 0;

    for (int W : worker_cfgs)
    {
        if (smoke && W != 8) continue;
        for (int order = 0; order < 2; ++order)                 // BOTH orderings at EVERY config
        {
            for (int arm_pass = 0; arm_pass < 2; ++arm_pass)
            {
                const bool native_arm = order == 0 ? arm_pass == 0 : arm_pass == 1;
                corpus = Corpus{};                              // one format resident at a time
                materialize(corpus, !native_arm);
                if (!native_arm)
                    std::printf("PACK corpus_MiB=%.1f pack_only_ms=%.1f (untimed, once per sweep)\n",
                                EXPERTS * corpus.bytes_per_expert / 1048576.0, corpus.pack_ms);
                const Exl3CpuTrellisLayout layout = native_arm ? Exl3CpuTrellisLayout::Native
                                                                : Exl3CpuTrellisLayout::Band8;
                const int dk[3] = { H, H, FF }, dn[3] = { FF, FF, H };
                for (int e = 0; e < EXPERTS; ++e)
                    for (int p = 0; p < 3; ++p)
                    {
                        Exl3CpuMatrix m{ corpus.t[p][(size_t)e].data(), corpus.suh[p][(size_t)e].data(),
                                         corpus.svh[p][(size_t)e].data(), dk[p], dn[p], BITS, HB, layout };
                        (p == 0 ? gmat : p == 1 ? umat : dmat)[(size_t)e] = m;
                    }
                svc = std::make_unique<Exl3CpuMoe>();            // fresh service: no layer accumulation
                svc->add_layer(0, gmat.data(), umat.data(), dmat.data(), EXPERTS);
                svc->set_threads(W);

                for (const Pattern& pat : PATTERNS)
                {
                    if (smoke && std::strcmp(pat.name, "t1g2") != 0 && std::strcmp(pat.name, "t4worst") != 0)
                        continue;
                    for (int w = 0; w < 2; ++w)                  // untimed: pool resize + output pages
                    {
                        pat.build(w, 0, ids, tk, rr);
                        svc->run_rows_bg(0, x.data(), H, ids.data(), tk.data(), rr.data(),
                                        (int)ids.size(), out.data(), H);
                        svc->run_rows_join();
                    }
                    std::vector<double> submit_ms, total_ms, join_ms;
                    submit_ms.reserve((size_t)jobs_per_round * rounds);
                    bool finite = true;
                    for (int round = 0; round < rounds; ++round)
                        for (int i = 0; i < jobs_per_round; ++i)
                        {
                            pat.build(i, round, ids, tk, rr);
                            const auto t0 = Clock::now();
                            svc->run_rows_bg(0, x.data(), H, ids.data(), tk.data(), rr.data(),
                                            (int)ids.size(), out.data(), H);
                            const auto t1 = Clock::now();
                            svc->run_rows_join();
                            const auto t2 = Clock::now();
                            submit_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                            join_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
                            total_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t0).count());
                            if ((round * jobs_per_round + i) % 8 == 0)
                                for (int rw = 0; rw < pat.rows_per_job; ++rw)
                                    for (int cxi = 0; cxi < H; cxi += 16)
                                        finite = finite && std::isfinite(out[(size_t)rw * H + cxi]);
                        }
                    if (!finite) { std::printf("FAIL non-finite output (%s)\n", pat.name); return 1; }
                    const auto st = svc->stats();
                    if (native_arm && st.repack_fills != 0)
                    { std::printf("FAIL native arm repacked (%lld fills)\n", (long long)st.repack_fills); return 1; }
                    if (!native_arm && (st.repack_fills != 0 || st.cache_bytes != 0))
                    { std::printf("FAIL band8 arm converted bytes (fills=%lld cache=%lld)\n",
                                  (long long)st.repack_fills, (long long)st.cache_bytes); return 1; }
                    Cell c{ pct(submit_ms, .5), pct(total_ms, .5), pct(join_ms, .5),
                            pct(total_ms, .25), pct(total_ms, .75), st.native_reads };
                    const std::string arm = native_arm ? "native" : "band8";
                    std::printf("W=%2d order=%d %-7s %-7s submit=%7.3f join=%7.3f total_med=%7.3f ms "
                                "[p25 %7.3f p75 %7.3f] isa=%d bg=%2d native_reads=%lld\n",
                                W, order, arm.c_str(), pat.name, c.submit, c.join, c.total,
                                c.p25, c.p75, st.isa, st.bg_workers, c.native_reads);
                    // pooled over orders into the same (pattern, arm) bucket
                    auto& bucket = cells[pat.name][arm];
                    if (bucket.total == 0.0) bucket = c;
                    else { bucket.total = (bucket.total + c.total) / 2; bucket.submit = (bucket.submit + c.submit) / 2;
                           bucket.join = (bucket.join + c.join) / 2; }
                }

                // parity tripwire (once per arm pair per W, on this sweep's corpus bytes):
                // t1g2 and t4worst outputs must match the OTHER arm's bitwise. The other
                // arm ran in the previous arm_pass with identical corpus bytes; snapshot
                // its outputs then, compare now.
                static std::vector<float> prev_t1g2, prev_t4worst;
                if (arm_pass == 0)
                {
                    build_t1<2>(3, 1, ids, tk, rr);
                    prev_t1g2.assign((size_t)2 * H, -1e30f);
                    svc->run_rows(0, x.data(), H, ids.data(), tk.data(), rr.data(), (int)ids.size(),
                                 prev_t1g2.data(), H);
                    build_t4worst(3, 1, ids, tk, rr);
                    prev_t4worst.assign((size_t)4 * H, -1e30f);
                    svc->run_rows(0, x.data(), H, ids.data(), tk.data(), rr.data(), (int)ids.size(),
                                 prev_t4worst.data(), H);
                }
                else
                {
                    std::vector<float> now;
                    build_t1<2>(3, 1, ids, tk, rr);
                    now.assign((size_t)2 * H, -1e30f);
                    svc->run_rows(0, x.data(), H, ids.data(), tk.data(), rr.data(), (int)ids.size(), now.data(), H);
                    if (std::memcmp(now.data(), prev_t1g2.data(), now.size() * sizeof(float)) != 0) ++parity_failures;
                    build_t4worst(3, 1, ids, tk, rr);
                    now.assign((size_t)4 * H, -1e30f);
                    svc->run_rows(0, x.data(), H, ids.data(), tk.data(), rr.data(), (int)ids.size(), now.data(), H);
                    if (std::memcmp(now.data(), prev_t4worst.data(), now.size() * sizeof(float)) != 0) ++parity_failures;
                    std::printf("W=%2d parity tripwire (t1g2, t4worst): %s\n", W,
                                parity_failures == 0 ? "bitwise equal across arms" : "MISMATCH");
                }
            }
        }
    }

    std::printf("\nPOOLED MEDIANS (mean of the two orders), band8/native ratio of total medians:\n");
    for (const auto& [name, arms] : cells)
    {
        const auto n = arms.find("native"), b = arms.find("band8");
        if (n == arms.end() || b == arms.end()) continue;
        const double logical_mib = 3.0 * (plane_words(H, FF) * 2.0 + (H + FF) * 2.0) / 1048576.0;
        std::printf("  %-7s native=%7.3f ms band8=%7.3f ms ratio=%5.3fx  (one expert streams "
                    "%.2f MiB; <1.0 = band-8 faster)\n",
                    name.c_str(), n->second.total, b->second.total,
                    n->second.total > 0 ? b->second.total / n->second.total : 0.0, logical_mib);
    }
    std::printf("VmHWM_KiB=%ld parity_failures=%d\n", peak_rss_kib(), parity_failures);
    return parity_failures == 0 ? 0 : 1;
}
