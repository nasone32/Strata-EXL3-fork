// src/kernels/exl3_cpu_parity.cpp - the ported EXL3 CPU expert tier against the oracle.
//
// Three checks on the REAL 3bpw model (default path below, argv[1] overrides):
//   1. ORACLE: one expert's end-to-end row (gate -> silu*up -> down, unweighted) vs
//      cpu::exl3_folded_gemv chains.  The service quantizes activations to int8 per GEMV, so
//      this is a tolerance check (the reference measures ~0.9% per-call RMS for the int8
//      class); the gate is 3% RMS / 12% max.
//   2. BIT-EXACT ROWS: row r of a multi-selection call is bitwise the single-selection call of
//      the same expert, and deduped rows (one expert serving two tokens) are bitwise the solo
//      rows - the property the verify path relies on when it groups routes by expert.
//   3. LAYOUT A/B: swizzled (repack cache) vs native trellis bytes, same tier, bitwise equal.
//
// With --sum the outputs of checks 1-3 are folded into one FNV-1a hash printed at the end, so
// runs under different STRATA_EXL3_CPU_MAX_ISA caps (separate processes: the tier is chosen by
// a static initializer) can be compared for the tiers' claimed cross-bit-exactness.  --bench
// replaces pass/fail with a throughput report (decode and verify shapes, warm).
#include "strata/kernels/cpu/exl3_moe.hpp"
#include "strata/kernels/cpu/exl3.hpp"
#include "strata/artifact/exl3_model.hpp"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

uint64_t fnv1a(const void* data, size_t bytes, uint64_t h = 1469598103934665603ull)
{
    const uint8_t* p = (const uint8_t*) data;
    for (size_t i = 0; i < bytes; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

uint16_t f32_to_f16_bits(float v)
{
    // round-to-nearest-even, no subnormal finesse needed for N(0,1)-scale activations
    uint32_t x;
    std::memcpy(&x, &v, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00u);   // inf/nan
    if (((x >> 23) & 0xff) == 0) return (uint16_t)sign;                   // zero (flush)
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);                    // overflow -> inf
    if (exp <= 0) return (uint16_t)sign;                                   // underflow -> 0
    man >>= 13;
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | man);
    // rounding from the dropped bits
    const uint32_t dropped = x & 0x1fffu;
    if (dropped > 0x1000u || (dropped == 0x1000u && (h & 1))) ++h;
    return h;
}

int failures = 0;

void check(bool ok, const char* what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

// One expert's reference row: exact fp decode through the CPU oracle, silu*up, down
void oracle_expert(const strata::Exl3Linear& g, const strata::Exl3Linear& u, const strata::Exl3Linear& d,
                   const uint16_t* x_f16, float* y_out, int H, int I)
{
    using strata::kernels::cpu::Exl3Codebook;
    std::vector<float> gg((size_t) I), uu((size_t) I), act((size_t) I);
    strata::kernels::cpu::exl3_folded_gemv(g.trellis, g.ki, g.nj, g.bits, Exl3Codebook::Mul1, g.suh, g.svh, x_f16, 1, gg.data());
    strata::kernels::cpu::exl3_folded_gemv(u.trellis, u.ki, u.nj, u.bits, Exl3Codebook::Mul1, u.suh, u.svh, x_f16, 1, uu.data());
    for (int i = 0; i < I; ++i) act[(size_t) i] = gg[(size_t) i] / (1.0f + std::exp(-gg[(size_t) i])) * uu[(size_t) i];
    std::vector<uint16_t> act_f16((size_t) I);
    for (int i = 0; i < I; ++i) act_f16[(size_t) i] = f32_to_f16_bits(act[(size_t) i]);
    strata::kernels::cpu::exl3_folded_gemv(d.trellis, d.ki, d.nj, d.bits, Exl3Codebook::Mul1, d.suh, d.svh, act_f16.data(), 1, y_out);
}

}  // namespace

int main(int argc, char** argv)
{
    const char* model_arg = nullptr;
    bool sum_only = false, bench = false, costs = false, background = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--sum") == 0) sum_only = true;
        else if (std::strcmp(argv[i], "--bench") == 0) bench = true;
        else if (std::strcmp(argv[i], "--costs") == 0) costs = true;
        else if (std::strcmp(argv[i], "--background") == 0) background = true;
        else if (model_arg == nullptr) model_arg = argv[i];
    }
    if (model_arg == nullptr)
    {
        std::printf("usage: strata-exl3-cpu-parity <model-dir> [--sum] [--bench] [--costs] [--background]\n"
                    "       <model-dir>: an EXL3 model directory (config.json + safetensors shards)\n");
        return 2;
    }
    const std::string model_dir = model_arg;

    strata::Exl3Model model(model_dir);
    const int n_layers = 48, n_experts = 512, H = 2560, I = 640;
    const int test_layers[2] = { 5, 30 };
    const int test_experts[4] = { 0, 7, 123, 511 };

    strata::kernels::cpu::Exl3CpuMoe svc;
    if (!strata::kernels::cpu::Exl3CpuMoe::supported())
    {
        std::printf("SKIP: CPU expert tier not supported here\n");
        return 0;
    }
    for (int l : test_layers)
    {
        std::vector<strata::kernels::cpu::Exl3CpuMatrix> gs((size_t) n_experts), us((size_t) n_experts), ds((size_t) n_experts);
        for (int e = 0; e < n_experts; ++e)
        {
            const std::string base = "model.language_model.layers." + std::to_string(l) +
                                     ".mlp.experts." + std::to_string(e);
            auto fill = [](const strata::Exl3Linear& L, strata::kernels::cpu::Exl3CpuMatrix& M)
            {
                M.trellis = L.trellis; M.suh = L.suh; M.svh = L.svh;
                M.k = L.in(); M.n = L.out(); M.bits = L.bits; M.hb = 0;
            };
            fill(model.linear(base + ".gate_proj"), gs[(size_t) e]);
            fill(model.linear(base + ".up_proj"), us[(size_t) e]);
            fill(model.linear(base + ".down_proj"), ds[(size_t) e]);
        }
        svc.add_layer(l, gs.data(), us.data(), ds.data(), n_experts);
    }

    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    uint64_t sum = 1469598103934665603ull;
    double worst_rms = 0.0, worst_max = 0.0;

    if (costs) {
        // Rotating 256 real experts (~450 MiB) exceeds both CCDs' L3 caches.
        // This is CPU service latency, not a DRAM-bandwidth or end-to-end claim.
        constexpr int working_set = 256, iterations = 128;
        std::vector<float> x((size_t) 4 * H), out((size_t) 40 * H);
        for (auto& v : x) v = dist(rng);
        int primer = 511, zero = 0;
        svc.run_rows(test_layers[0], x.data(), H, &primer, &zero, &zero, 1, out.data(), H);
        for (int tokens : {1, 4}) for (int groups : {1, 2, 4, 8, 10}) {
            const int rows = groups * tokens;
            std::vector<int> ids(rows), tok(rows), dst(rows);
            for (int r=0;r<rows;++r) { tok[r]=r/groups; dst[r]=r; }
            auto call = [&](int base) {
                for (int r=0;r<rows;++r) ids[r]=(base+r%groups)%working_set;
                if (background) {
                    svc.run_rows_bg(test_layers[0],x.data(),H,ids.data(),tok.data(),dst.data(),rows,out.data(),H);
                    svc.run_rows_join();
                } else svc.run_rows(test_layers[0],x.data(),H,ids.data(),tok.data(),dst.data(),rows,out.data(),H);
            };
            // Measure the first traversal separately: first layout sees repack fills here.
            const auto before = svc.stats();
            auto start = std::chrono::steady_clock::now();
            const int cold_calls = (working_set+groups-1)/groups;
            for (int i=0;i<cold_calls;++i) call(i*groups);
            const double first_us = std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/cold_calls;
            const auto warmed = svc.stats();
            start = std::chrono::steady_clock::now();
            for (int i=0;i<iterations;++i) call(i*groups);
            const double rotate_us = std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/iterations;
            start = std::chrono::steady_clock::now();
            for (int i=0;i<iterations;++i) call(0);
            const double hot_us = std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/iterations;
            bool finite = true;
            for (int r=0;r<rows;++r) for(int c=0;c<H;++c) finite = finite && std::isfinite(out[(size_t)r*H+c]);
            if (!finite) return 1;
            sum=fnv1a(out.data(),(size_t)rows*H*sizeof(float),sum);
            std::printf("CPU_COST tokens=%d groups=%d rows=%d background=%d threads=%d isa=%d first_us=%.3f rotate_us=%.3f hot_us=%.3f fills=%lld evictions=%lld cache_mib=%.2f checksum=%016llx\n",
                tokens,groups,rows,(int)background,warmed.threads,warmed.isa,first_us,rotate_us,hot_us,
                (long long)(warmed.repack_fills-before.repack_fills),(long long)warmed.repack_evictions,
                warmed.cache_bytes/1048576.0,(unsigned long long)sum);
            std::fflush(stdout);
        }
        return 0;
    }

    // ---- 1. oracle tolerance + 2. row bit-exactness ----
    for (int l : test_layers)
    {
        for (int e : test_experts)
        {
            std::vector<float> x((size_t) H);
            for (int i = 0; i < H; ++i) x[(size_t) i] = dist(rng);
            std::vector<uint16_t> x16((size_t) H);
            for (int i = 0; i < H; ++i) x16[(size_t) i] = f32_to_f16_bits(x[(size_t) i]);

            std::vector<float> ref((size_t) H);
            const std::string base = "model.language_model.layers." + std::to_string(l) +
                                     ".mlp.experts." + std::to_string(e);
            oracle_expert(model.linear(base + ".gate_proj"), model.linear(base + ".up_proj"),
                          model.linear(base + ".down_proj"), x16.data(), ref.data(), H, I);

            std::vector<float> got((size_t) H, -1e30f);
            int ids[1] = { e }, t[1] = { 0 }, r[1] = { 0 };
            svc.run_rows(l, x.data(), H, ids, t, r, 1, got.data(), H);

            double se = 0.0, sr = 0.0, maxabs = 0.0;
            for (int i = 0; i < H; ++i)
            {
                const double err = (double) got[(size_t) i] - (double) ref[(size_t) i];
                se += err * err;
                sr += (double) ref[(size_t) i] * (double) ref[(size_t) i];
                maxabs = std::max(maxabs, std::fabs(err));
            }
            const double rms = std::sqrt(se / H) / std::max(std::sqrt(sr / H), 1e-9);
            const double relmax = maxabs / std::max(std::sqrt(sr / H), 1e-9);
            worst_rms = std::max(worst_rms, rms);
            worst_max = std::max(worst_max, relmax);
            if (!sum_only)
                std::printf("oracle layer %2d expert %3d: rel RMS %.4f%%, rel max %.2f%%\n", l, e, 100 * rms, 100 * relmax);
            sum = fnv1a(got.data(), got.size() * sizeof(float), sum);

            // bit-exact: the same expert inside a 4-row call (other rows are other experts).
            // Two token rows in the buffer: token 1 selects e alone (solo comparison target),
            // e2 twice on token 0 (a dedup chunk with m = 2).
            {
                std::vector<float> x2((size_t) 2 * H);
                for (int i = 0; i < 2 * H; ++i) x2[(size_t) i] = dist(rng);
                std::vector<float> solo2((size_t) H, -1e30f);
                int sid[1] = { e }, st[1] = { 1 }, sr[1] = { 0 };
                svc.run_rows(l, x2.data(), H, sid, st, sr, 1, solo2.data(), H);

                std::vector<float> multi((size_t) 4 * H, -1e30f);
                const int e2 = (e + 1) % n_experts;
                int mid[4] = { e2, e, (e + 2) % n_experts, e2 };
                int mt[4] = { 0, 1, 1, 0 };
                int mr[4] = { 0, 1, 2, 3 };
                svc.run_rows(l, x2.data(), H, mid, mt, mr, 4, multi.data(), H);
                // row 1 (expert e, token 1) must equal the solo call of (e, token 1) bitwise
                if (std::memcmp(multi.data() + (size_t) H, solo2.data(), (size_t) H * sizeof(float)) != 0)
                {
                    int ndiff = 0, first = -1;
                    double maxd = 0.0;
                    for (int i = 0; i < H; ++i)
                        if (std::memcmp(&multi[(size_t) H + (size_t) i], &solo2[(size_t) i], 4) != 0)
                        {
                            ++ndiff;
                            if (first < 0) first = i;
                            const float a = multi[(size_t) H + (size_t) i], b = solo2[(size_t) i];
                            maxd = std::max(maxd, (double) std::fabs(a - b));
                        }
                    std::printf("     diff rows: %d of %d floats differ, first at %d, max |d| %.6g "
                                "(vals %.6g vs %.6g)\n", ndiff, H, first, maxd,
                                first >= 0 ? (double) multi[(size_t) H + (size_t) first] : 0.0,
                                first >= 0 ? (double) solo2[(size_t) first] : 0.0);
                }
                check(std::memcmp(multi.data() + (size_t) H, solo2.data(), (size_t) H * sizeof(float)) == 0,
                      "multi-call solo row is bitwise the single call");
                sum = fnv1a(multi.data(), multi.size() * sizeof(float), sum);
                sum = fnv1a(solo2.data(), solo2.size() * sizeof(float), sum);
                // rows 0 and 3: same expert, same token -> identical outputs
                check(std::memcmp(multi.data(), multi.data() + (size_t) 3 * H, (size_t) H * sizeof(float)) == 0,
                      "same (expert, token) rows agree bitwise");
            }
        }
    }
    if (!sum_only)
    {
        char what[128];
        std::snprintf(what, sizeof what, "oracle tolerance: worst rel RMS %.3f%% < 3%%, worst rel max %.2f%% < 12%%",
                      100 * worst_rms, 100 * worst_max);
        check(worst_rms < 0.03 && worst_max < 0.12, what);
    }

    // ---- 3. layout A/B: the same computation with the repack cache disabled ----
    {
        setenv("STRATA_EXL3_CPU_SWIZZLE", "0", 1);
        strata::kernels::cpu::Exl3CpuMoe native;
        for (int l : test_layers)
        {
            std::vector<strata::kernels::cpu::Exl3CpuMatrix> gs((size_t) n_experts), us((size_t) n_experts), ds((size_t) n_experts);
            for (int e = 0; e < n_experts; ++e)
            {
                const std::string base = "model.language_model.layers." + std::to_string(l) +
                                         ".mlp.experts." + std::to_string(e);
                auto fill = [](const strata::Exl3Linear& L, strata::kernels::cpu::Exl3CpuMatrix& M)
                {
                    M.trellis = L.trellis; M.suh = L.suh; M.svh = L.svh;
                    M.k = L.in(); M.n = L.out(); M.bits = L.bits; M.hb = 0;
                };
                fill(model.linear(base + ".gate_proj"), gs[(size_t) e]);
                fill(model.linear(base + ".up_proj"), us[(size_t) e]);
                fill(model.linear(base + ".down_proj"), ds[(size_t) e]);
            }
            native.add_layer(l, gs.data(), us.data(), ds.data(), n_experts);
        }
        bool same = true;
        for (int l : test_layers)
            for (int e : test_experts)
            {
                std::vector<float> x((size_t) H);
                for (int i = 0; i < H; ++i) x[(size_t) i] = dist(rng);
                std::vector<float> a((size_t) H, -1e30f), b((size_t) H, -1e30f);
                int ids[1] = { e }, t[1] = { 0 }, r[1] = { 0 };
                svc.run_rows(l, x.data(), H, ids, t, r, 1, a.data(), H);
                native.run_rows(l, x.data(), H, ids, t, r, 1, b.data(), H);
                if (std::memcmp(a.data(), b.data(), (size_t) H * sizeof(float)) != 0) same = false;
                sum = fnv1a(b.data(), b.size() * sizeof(float), sum);
            }
        unsetenv("STRATA_EXL3_CPU_SWIZZLE");
        check(same, "swizzled (repacked) and native trellis bytes agree bitwise");
    }

    if (sum_only)
    {
        std::printf("SUM %016llx\n", (unsigned long long) sum);
        return failures == 0 ? 0 : 1;
    }

    // ---- 4. the overlapping dispatch vs the blocking one, bitwise, many iterations: the bg
    // phase barrier is the one piece of scheduling the other checks cannot reach, and its
    // failure mode is intermittent (a laggard worker credited to the next phase) ----
    {
        const int tokens = 6, k = 10, iters = 300;
        std::vector<float> x((size_t) tokens * H);
        std::vector<int> ids((size_t) tokens * k), t((size_t) tokens * k), rr((size_t) tokens * k);
        std::vector<float> a((size_t) tokens * k * H), b((size_t) tokens * k * H);
        std::mt19937 r5(777);
        int bad = 0;
        for (int it = 0; it < iters; ++it)
        {
            for (int i = 0; i < tokens * H; ++i) x[(size_t) i] = dist(r5);
            for (int i = 0; i < tokens * k; ++i)
            {
                ids[(size_t) i] = (int) (r5() % 64);   // dense: many duplicate experts -> m>1 chunks
                t[(size_t) i] = i / k;
                rr[(size_t) i] = i;
            }
            const int l = test_layers[it & 1];
            std::fill(a.begin(), a.end(), -1e30f);
            std::fill(b.begin(), b.end(), -1e30f);
            svc.run_rows(l, x.data(), H, ids.data(), t.data(), rr.data(), tokens * k, a.data(), H);
            svc.run_rows_bg(l, x.data(), H, ids.data(), t.data(), rr.data(), tokens * k, b.data(), H);
            svc.run_rows_join();
            if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0)
            {
                ++bad;
                int rows = 0, first = -1;
                for (int i = 0; i < tokens * k; ++i)
                    if (std::memcmp(&a[(size_t) i * H], &b[(size_t) i * H], H * sizeof(float)) != 0)
                    { ++rows; if (first < 0) first = i; }
                std::printf("     bg diff iter %d: %d rows differ, first row %d (expert %d) "
                            "a0=%.6f b0=%.6f\n",
                            it, rows, first, first >= 0 ? ids[(size_t) first] : -1,
                            first >= 0 ? (double) a[(size_t) first * H] : 0.0,
                            first >= 0 ? (double) b[(size_t) first * H] : 0.0);
            }
            sum = fnv1a(b.data(), b.size() * sizeof(float), sum);
        }
        char what[96];
        std::snprintf(what, sizeof what, "bg dispatch is bitwise the blocking one over %d window jobs", iters);
        check(bad == 0, what);
    }

    // Small repack cache must not evict bytes already referenced by this job.
    // Destroying the service must retire its detached pool workers too.
    {
        const char* old_budget = std::getenv("STRATA_EXL3_CPU_CACHE_MIB");
        const std::string saved = old_budget ? old_budget : "";
        setenv("STRATA_EXL3_CPU_CACHE_MIB","4",1);
        {
            strata::kernels::cpu::Exl3CpuMoe tiny;
            tiny.set_threads(4);
            std::vector<strata::kernels::cpu::Exl3CpuMatrix> gs(4),us(4),ds(4);
            for (int i=0;i<4;++i) {
                const std::string base="model.language_model.layers.5.mlp.experts."+std::to_string(test_experts[i]);
                auto matrix=[](const strata::Exl3Linear& l) {
                    return strata::kernels::cpu::Exl3CpuMatrix{l.trellis,l.suh,l.svh,l.in(),l.out(),l.bits,0};
                };
                gs[i]=matrix(model.linear(base+".gate_proj")); us[i]=matrix(model.linear(base+".up_proj"));
                ds[i]=matrix(model.linear(base+".down_proj"));
            }
            tiny.add_layer(5,gs.data(),us.data(),ds.data(),4);
            std::vector<float> x(2*H),want(4*H),got(4*H);
            for(auto& v:x) v=dist(rng);
            int ids[]={0,1,2,3}, tok[]={0,1,0,1}, row[]={0,1,2,3};
            svc.run_rows(5,x.data(),H,test_experts,tok,row,4,want.data(),H);
            bool ok=true;
            for(int i=0;i<4;++i) {
                tiny.run_rows(5,x.data(),H,ids,tok,row,4,got.data(),H);
                ok=ok && got==want;
                tiny.run_rows_bg(5,x.data(),H,ids,tok,row,4,got.data(),H); tiny.run_rows_join();
                ok=ok && got==want;
            }
            tiny.set_threads(1);
            tiny.run_rows_bg(5,x.data(),H,ids,tok,row,4,got.data(),H); tiny.run_rows_join();
            check(got==want,"single-worker background join runs inline with valid worker index");
            const auto stats=tiny.stats();
            check(ok && stats.repack_evictions>0 && stats.cache_bytes<=4*1048576,
                  "small repack cache pins live job bytes across eviction (blocking/background)");
        }
        if(old_budget) setenv("STRATA_EXL3_CPU_CACHE_MIB",saved.c_str(),1);
        else unsetenv("STRATA_EXL3_CPU_CACHE_MIB");
    }

    // ---- bench: warm throughput on decode and verify shapes ----
    if (bench)
    {
        using clock = std::chrono::steady_clock;
        const int l = test_layers[0];
        auto one_set = [&](int seed)
        {
            std::mt19937 r2((unsigned) seed);
            std::vector<float> x((size_t) H);
            for (int i = 0; i < H; ++i) x[(size_t) i] = dist(r2);
            return x;
        };
        // decode: 1 token, 10 experts (all misses by construction of the ids below)
        {
            const int rep = 200;
            std::vector<float> out((size_t) 10 * H);
            int ids[10], t[10], rr[10];
            for (int i = 0; i < 10; ++i) { ids[i] = 100 + i * 7; t[i] = 0; rr[i] = i; }
            const auto input = one_set(7);
            const float* x = input.data();
            const auto t0 = clock::now();
            for (int it = 0; it < rep; ++it) svc.run_rows(l, x, H, ids, t, rr, 10, out.data(), H);
            const double ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count() / rep;
            const double gb = 10.0 * 1843200.0 / 1e9;   // 3 x 614400 trellis bytes per expert
            std::printf("bench decode (1 token x 10 experts): %.3f ms/call, %.1f GB/s -> %.4f ms/expert-layer\n",
                        ms, gb / (ms / 1000.0), ms / 10.0);
        }
        // verify window: 6 tokens x 10 experts, deduped
        {
            const int tokens = 6, k = 10, rep = 100;
            std::vector<float> x((size_t) tokens * H);
            {
                std::mt19937 r3(99);
                for (int i = 0; i < tokens * H; ++i) x[(size_t) i] = dist(r3);
            }
            std::vector<int> ids((size_t) tokens * k), t((size_t) tokens * k), rr((size_t) tokens * k);
            std::mt19937 r4(1234);
            for (int i = 0; i < tokens * k; ++i)
            {
                ids[(size_t) i] = (int)(r4() % n_experts);
                t[(size_t) i] = i / k;
                rr[(size_t) i] = i;
            }
            std::vector<int> uniq = ids;
            std::sort(uniq.begin(), uniq.end());
            uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
            std::vector<float> out((size_t) tokens * k * H);
            const auto t0 = clock::now();
            for (int it = 0; it < rep; ++it) svc.run_rows(l, x.data(), H, ids.data(), t.data(), rr.data(),
                                                          tokens * k, out.data(), H);
            const double ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count() / rep;
            const double gb = (double) uniq.size() * 1843200.0 / 1e9;
            std::printf("bench window (6 tokens x 10 experts, %zu unique): %.3f ms/call, %.1f GB/s\n",
                        uniq.size(), ms, gb / (ms / 1000.0));
        }
    }

    if (failures > 0)
    {
        std::printf("%d FAILURES\n", failures);
        return 1;
    }
    std::printf("all exl3 cpu parity checks passed\n");
    return 0;
}
