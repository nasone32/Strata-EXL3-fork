// src/kernels/exl3_cpu_band8_check.cpp - canonical caller-packed band-8 trellis (B1 API).
//
// CPU-only, fully synthetic (no model files, no GPU).  Verifies the Exl3CpuTrellisLayout
// registration path against the native-layout service with the PUBLIC API only:
//   1. PARITY: native vs Band8, same inputs, bitwise, at a FIXED participant count
//      (blocking vs blocking, run_rows_bg/join vs run_rows_bg/join), for the four verified
//      rates (K2, K2.5, K3, K4) on the real model geometries (MiMo 4096/2048, Qwen 2560/640),
//      including shuffled output row maps, repeated expert rows (2/3/4-row groups -> the
//      5/6-phase path), scattered token indices with a non-unit x_stride, and the bg
//      dispatch's actual participant count (stats().bg_workers).
//   2. NO HIDDEN CONVERSIONS: the band-8 service holds repack fills/evictions/cache_bytes at
//      0 (default cache ON - nothing repacked or copied), and stays bitwise correct with
//      STRATA_EXL3_CPU_SWIZZLE=0 and STRATA_EXL3_CPU_CACHE_MIB=0 (those gate optional
//      conversions of NATIVE bytes, not a declared physical layout).
//   3. CALLER OWNERSHIP: overwriting the caller's packed bytes changes the output, restoring
//      them restores it bitwise - the service reads the caller's buffer live, no snapshot.
//   4. REGISTRATION REFUSALS, decided from metadata only (garbage non-null trellis pointers
//      prove nothing is dereferenced): unknown enum value, mixed-projection expert, K8,
//      unverified rates (K5, K3.5).  A native registration of the same garbage pointers is
//      the negative control.
//   5. --reject-isa (separate process, started with STRATA_EXL3_CPU_MAX_ISA=avx2): a valid
//      band-8 registration on a CPU/tier without the AVX-512BW band kernels must throw.
//
// Build/run: see notes/results/2026-11-06-packed-cpu-partition/README.md (plain g++, no
// CMake/GPU).
#include "strata/kernels/cpu/exl3_moe.hpp"
#include "kernels/cpu/exl3_moe_fixture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using strata::kernels::cpu::Exl3CpuMatrix;
using strata::kernels::cpu::Exl3CpuMoe;
using strata::kernels::cpu::Exl3CpuTrellisLayout;

namespace {

int failures = 0;

void check(bool ok, const char* what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
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

struct Projection
{
    std::vector<uint16_t> native, packed;   // same size, band-8 is a pure permutation
    std::vector<uint16_t> suh, svh;
    int k, n, bits, hb;
    void make(int k_, int n_, int bits_, int hb_, std::mt19937& rng)
    {
        k = k_; n = n_; bits = bits_; hb = hb_;
        const size_t words = (size_t)(k / 16) * (n / 16) * (16 * bits + 8 * hb);
        native.resize(words); packed.resize(words);
        for (auto& v : native) v = (uint16_t)rng();
        strata::kernels::cpu::exl3cpu_fixture_pack_band8(packed.data(), native.data(), k, n, bits, hb);
        suh.resize(k); svh.resize(n);
        std::uniform_real_distribution<float> sc(0.25f, 1.75f);
        for (auto& v : suh) v = f32_to_f16_bits(sc(rng));
        for (auto& v : svh) v = f32_to_f16_bits(sc(rng));
    }
    Exl3CpuMatrix mat(Exl3CpuTrellisLayout layout) const
    {
        return Exl3CpuMatrix{ layout == Exl3CpuTrellisLayout::Band8 ? packed.data() : native.data(),
                              suh.data(), svh.data(), k, n, bits, hb, layout };
    }
};

struct Case
{
    int H, I, bits, hb;
    std::vector<Projection> g, u, d;   // 8 experts each
};

// One job description; run on both services, all four dispatch shapes compared
struct Job
{
    std::vector<int> ids, t, r;
    int tokens;
};

bool run_job(Exl3CpuMoe& svc, int layer, const Job& j, const std::vector<float>& x, int64_t x_stride,
             int H, bool bg, std::vector<float>& out)
{
    out.assign((size_t)j.ids.size() * H, -1e30f);
    if (bg)
    {
        svc.run_rows_bg(layer, x.data(), x_stride, j.ids.data(), j.t.data(), j.r.data(),
                        (int)j.ids.size(), out.data(), H);
        // complete the job before anything else touches this service (one outstanding bg job
        // per calling thread - the caller-owned protocol the engine follows too)
        svc.run_rows_join();
    }
    else
        svc.run_rows(layer, x.data(), x_stride, j.ids.data(), j.t.data(), j.r.data(),
                     (int)j.ids.size(), out.data(), H);
    bool finite = true;
    for (float v : out) finite = finite && std::isfinite(v);
    return finite;
}

void parity_case(Case& c, const char* name, int threads)
{
    const int E = 8;
    std::vector<Exl3CpuMatrix> gn, un, dn, gb, ub, db;
    for (int e = 0; e < E; ++e)
    {
        gn.push_back(c.g[(size_t)e].mat(Exl3CpuTrellisLayout::Native));
        un.push_back(c.u[(size_t)e].mat(Exl3CpuTrellisLayout::Native));
        dn.push_back(c.d[(size_t)e].mat(Exl3CpuTrellisLayout::Native));
        gb.push_back(c.g[(size_t)e].mat(Exl3CpuTrellisLayout::Band8));
        ub.push_back(c.u[(size_t)e].mat(Exl3CpuTrellisLayout::Band8));
        db.push_back(c.d[(size_t)e].mat(Exl3CpuTrellisLayout::Band8));
    }
    Exl3CpuMoe nat, band;
    nat.set_threads(threads);  band.set_threads(threads);
    nat.add_layer(0, gn.data(), un.data(), dn.data(), E);
    band.add_layer(0, gb.data(), ub.data(), db.data(), E);

    std::mt19937 rng(1234 + c.bits * 7 + c.hb * 3 + c.H);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    const int64_t stride = (int64_t)2 * c.H;   // token rows are scattered, stride > H
    const int tokens = 8;
    std::vector<float> x((size_t)tokens * stride + c.H);
    for (auto& v : x) v = dist(rng);

    // decode: 1 expert; decode: 3 unique experts with a shuffled row map; verify mix with
    // repeated expert rows 2/3/4 across varied token indices; worst case: one 4-row group
    std::vector<Job> jobs;
    jobs.push_back({ { 2 }, { 5 }, { 0 }, 8 });
    jobs.push_back({ { 1, 3, 5 }, { 5, 5, 5 }, { 2, 0, 1 }, 8 });
    {
        Job j; j.tokens = 8;
        j.ids = { 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 4 };
        j.t   = { 1, 4,  1, 4, 2,  7, 2, 4, 5,  5, 2 };
        j.r   = { 3, 0,  6, 1, 8,  2, 7, 4, 10, 9, 5 };
        jobs.push_back(j);
    }
    jobs.push_back({ { 5, 5, 5, 5 }, { 1, 4, 2, 7 }, { 3, 2, 1, 0 }, 8 });

    bool parity = true, bg_parity = true, shape_parity = true, finite = true;
    for (const Job& j : jobs)
    {
        std::vector<float> a, b, a2, b2;
        finite = run_job(nat, 0, j, x, stride, c.H, false, a) && finite;
        finite = run_job(band, 0, j, x, stride, c.H, false, b) && finite;
        finite = run_job(nat, 0, j, x, stride, c.H, true, a2) && finite;
        finite = run_job(band, 0, j, x, stride, c.H, true, b2) && finite;
        parity = parity && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        bg_parity = bg_parity && std::memcmp(a2.data(), b2.data(), a2.size() * sizeof(float)) == 0;
        shape_parity = shape_parity && std::memcmp(a.data(), a2.data(), a.size() * sizeof(float)) == 0
                                  && std::memcmp(b.data(), b2.data(), b.size() * sizeof(float)) == 0;
    }
    char what[160];
    std::snprintf(what, sizeof what, "%s: band-8 bitwise == native (blocking, %d participants)", name, threads);
    check(parity && finite, what);
    std::snprintf(what, sizeof what, "%s: band-8 bitwise == native (run_rows_bg/join, %d bg participants)",
                  name, band.stats().bg_workers);
    check(bg_parity, what);
    check(shape_parity, "    bg dispatch bitwise == blocking on both layouts");
    std::snprintf(what, sizeof what, "    bg participant count is %d of %d threads (caller excluded)",
                  band.stats().bg_workers, band.stats().threads);
    check(band.stats().bg_workers == threads - 1 && band.stats().threads == threads, what);

    // no hidden conversion: cache ON (default set below) yet zero fills/evictions/bytes
    const auto bs = band.stats();
    std::snprintf(what, sizeof what, "%s: band-8 service holds fills=0 evictions=0 cache_bytes=0 (cache ON, native svc filled %lld)",
                  name, (long long)nat.stats().repack_fills);
    check(bs.repack_fills == 0 && bs.repack_evictions == 0 && bs.cache_bytes == 0 &&
          nat.stats().repack_fills > 0, what);

    // caller ownership: the service reads the caller's packed bytes live
    {
        Job j = jobs[1];
        std::vector<float> before, during, after;
        run_job(band, 0, j, x, stride, c.H, false, before);
        Projection& p = c.g[1];
        const std::vector<uint16_t> saved = p.packed;
        for (auto& v : p.packed) v = (uint16_t)(v ^ 0xa5a5u);
        run_job(band, 0, j, x, stride, c.H, false, during);
        std::memcpy(p.packed.data(), saved.data(), saved.size() * 2);   // in place: no realloc, pointer stays live
        run_job(band, 0, j, x, stride, c.H, false, after);
        check(std::memcmp(before.data(), during.data(), before.size() * sizeof(float)) != 0 &&
              std::memcmp(before.data(), after.data(), before.size() * sizeof(float)) == 0,
              "    caller-owned packed bytes are read live (mutate->different, restore->bitwise same)");
    }
}

// Registration refusals on garbage (non-null, never dereferenced) pointers
void rejection_checks(int isa)
{
    const uint16_t* junk = reinterpret_cast<const uint16_t*>(0x1);
    using Trio = std::array<Exl3CpuMatrix, 3>;
    auto trio = [&](int bits, int hb, Exl3CpuTrellisLayout layout) -> Trio
    {
        return { Exl3CpuMatrix{ junk, junk, junk, 256, 128, bits, hb, layout },
                 Exl3CpuMatrix{ junk, junk, junk, 256, 128, bits, hb, layout },
                 Exl3CpuMatrix{ junk, junk, junk, 128, 256, bits, hb, layout } };
    };
    auto must_throw = [&](const char* what, std::vector<Exl3CpuMatrix> g, std::vector<Exl3CpuMatrix> u,
                          std::vector<Exl3CpuMatrix> d)
    {
        bool threw = false;
        try
        {
            Exl3CpuMoe svc;
            svc.add_layer(0, g.data(), u.data(), d.data(), 1);
        }
        catch (const std::runtime_error&) { threw = true; }
        catch (...) {}
        check(threw, what);
    };

    std::vector<Exl3CpuMatrix> g = { trio(2, 0, Exl3CpuTrellisLayout::Band8)[0] };
    std::vector<Exl3CpuMatrix> u = { trio(2, 0, Exl3CpuTrellisLayout::Band8)[1] };
    std::vector<Exl3CpuMatrix> d = { trio(2, 0, Exl3CpuTrellisLayout::Band8)[2] };
    must_throw("unknown layout enum value refused at registration",
               { { junk, junk, junk, 256, 128, 2, 0, static_cast<Exl3CpuTrellisLayout>(7) } },
               u, d);
    must_throw("mixed-projection expert refused (gate Band8, up/down native)",
               g,
               { trio(2, 0, Exl3CpuTrellisLayout::Native)[1] },
               { trio(2, 0, Exl3CpuTrellisLayout::Native)[2] });
    must_throw("K8 band-8 refused (dword kernel reads native only)",
               { trio(8, 0, Exl3CpuTrellisLayout::Band8)[0] }, u, d);
    must_throw("unverified rate K5 band-8 refused",
               { trio(5, 0, Exl3CpuTrellisLayout::Band8)[0] }, u, d);
    must_throw("unverified rate K3.5 band-8 refused",
               { trio(3, 1, Exl3CpuTrellisLayout::Band8)[0] }, u, d);
    if (isa < 2)
        must_throw("band-8 refused below the AVX-512BW tier",
                   { trio(2, 0, Exl3CpuTrellisLayout::Band8)[0] }, u, d);
    else
        check(true, "    (AVX-512BW tier present: ISA refusal covered by --reject-isa)");

    // per-expert shape-chain safety: expert-0 dims define the layer; a DEEPER expert with a
    // different shape would read out of bounds at job time (prepare uses layer-wide H/I), so
    // registration must refuse it - from metadata only, before any pointer dereference.
    // Base trio is 256->128 / 128->256 (H=256, I=128); expert 1 breaks it two ways.
    auto base_gud = [](Exl3CpuTrellisLayout layout)
    {
        return Trio{ Exl3CpuMatrix{ nullptr, nullptr, nullptr, 256, 128, 2, 0, layout },
                     Exl3CpuMatrix{ nullptr, nullptr, nullptr, 256, 128, 2, 0, layout },
                     Exl3CpuMatrix{ nullptr, nullptr, nullptr, 128, 256, 2, 0, layout } };
    };
    auto two_experts = [&](const Trio& bad, Exl3CpuTrellisLayout layout)
    {
        const Trio good = base_gud(layout);
        std::vector<Exl3CpuMatrix> bg{ good[0], bad[0] }, bu{ good[1], bad[1] }, bd{ good[2], bad[2] };
        return std::make_tuple(bg, bu, bd);
    };
    for (Exl3CpuTrellisLayout layout : { Exl3CpuTrellisLayout::Native, Exl3CpuTrellisLayout::Band8 })
    {
        const char* tag = layout == Exl3CpuTrellisLayout::Band8 ? "band-8" : "native";
        // (a) internally inconsistent trio at expert 1: gate 256x256 (valid matrix, wrong n)
        {
            auto [bg, bu, bd] = two_experts(Trio{ Exl3CpuMatrix{ junk, junk, junk, 256, 256, 2, 0, layout },
                                                  Exl3CpuMatrix{ junk, junk, junk, 256, 128, 2, 0, layout },
                                                  Exl3CpuMatrix{ junk, junk, junk, 128, 256, 2, 0, layout } }, layout);
            char what[128];
            std::snprintf(what, sizeof what, "expert 1 inconsistent gate shape refused at registration (%s)", tag);
            must_throw(what, bg, bu, bd);
        }
        // (b) SELF-CONSISTENT but different H/I at expert 1 (gate/up 384->128, down 128->384:
        // every matrix individually valid and the trio chains - just not the layer's H/I)
        {
            auto [bg, bu, bd] = two_experts(Trio{ Exl3CpuMatrix{ junk, junk, junk, 384, 128, 2, 0, layout },
                                                  Exl3CpuMatrix{ junk, junk, junk, 384, 128, 2, 0, layout },
                                                  Exl3CpuMatrix{ junk, junk, junk, 128, 384, 2, 0, layout } }, layout);
            char what[128];
            std::snprintf(what, sizeof what, "expert 1 self-consistent different H/I refused at registration (%s)", tag);
            must_throw(what, bg, bu, bd);
        }
    }

    // negative control: the SAME garbage pointers register fine as native (nothing was read)
    bool threw = false;
    try
    {
        std::vector<Exl3CpuMatrix> ng(1), nu(1), nd(1);
        ng[0] = trio(2, 0, Exl3CpuTrellisLayout::Native)[0];
        nu[0] = trio(2, 0, Exl3CpuTrellisLayout::Native)[1];
        nd[0] = trio(2, 0, Exl3CpuTrellisLayout::Native)[2];
        Exl3CpuMoe svc;
        svc.add_layer(0, ng.data(), nu.data(), nd.data(), 1);
    }
    catch (...) { threw = true; }
    check(!threw, "    native registration of the same garbage pointers throws nothing (no pointer read)");
}

}  // namespace

int main(int argc, char** argv)
{
    // --reject-isa: run ONLY in a process started with STRATA_EXL3_CPU_MAX_ISA=avx2 (the ISA
    // tier is chosen by a static initializer; no fork after threads)
    if (argc > 1 && std::strcmp(argv[1], "--reject-isa") == 0)
    {
        Exl3CpuMoe svc;
        const int isa = svc.stats().isa;
        const uint16_t* junk = reinterpret_cast<const uint16_t*>(0x1);
        std::vector<Exl3CpuMatrix> g(1), u(1), d(1);
        g[0] = { junk, junk, junk, 256, 128, 2, 0, Exl3CpuTrellisLayout::Band8 };
        u[0] = { junk, junk, junk, 256, 128, 2, 0, Exl3CpuTrellisLayout::Band8 };
        d[0] = { junk, junk, junk, 128, 256, 2, 0, Exl3CpuTrellisLayout::Band8 };
        bool threw = false;
        const char* msg = "";
        try { svc.add_layer(0, g.data(), u.data(), d.data(), 1); }
        catch (const std::runtime_error& e) { threw = true; msg = e.what(); }
        std::printf("%s  --reject-isa: isa=%d add_layer(Band8) %s (%s)\n",
                    threw && isa < 2 ? "ok  " : "FAIL", isa, threw ? "threw" : "accepted", msg);
        return (threw && isa < 2) ? 0 : 1;
    }

    Exl3CpuMoe probe;
    const int isa = probe.stats().isa;
    std::printf("isa=%d (0 scalar, 1 avx2, 2 bw, 3 vnni, 4 vbmi)\n", isa);
    if (!Exl3CpuMoe::supported())
    {
        std::printf("SKIP: CPU expert tier not supported here\n");
        return 0;
    }
    rejection_checks(isa);
    if (isa < 2)
    {
        std::printf("SKIP: band-8 parity needs the AVX-512BW tier (this machine reports %d)\n", isa);
        return failures == 0 ? 0 : 1;
    }

    // Deterministic default policy for the battery: cache ON so a hidden conversion would be
    // VISIBLE (fills/cache_bytes) and the native control actually exercises the legacy repack.
    setenv("STRATA_EXL3_CPU_CACHE_MIB", "16384", 1);
    setenv("STRATA_EXL3_CPU_SWIZZLE", "1", 1);

    struct Shape { int H, I, bits, hb; const char* name; };
    const Shape shapes[] = {
        { 4096, 2048, 2, 0, "MiMo 4096/2048 K2" },
        { 4096, 2048, 2, 1, "MiMo 4096/2048 K2.5" },
        { 2560,  640, 3, 0, "Qwen 2560/640 K3" },
        { 2560,  640, 4, 0, "Qwen 2560/640 K4" },
    };
    for (const Shape& s : shapes)
    {
        Case c; c.H = s.H; c.I = s.I; c.bits = s.bits; c.hb = s.hb;
        std::mt19937 rng(4242 + s.bits * 11 + s.hb * 5 + s.H % 97);
        for (int e = 0; e < 8; ++e)
        {
            Projection pg, pu, pd;
            pg.make(s.H, s.I, s.bits, s.hb, rng);
            pu.make(s.H, s.I, s.bits, s.hb, rng);
            pd.make(s.I, s.H, s.bits, s.hb, rng);
            c.g.push_back(std::move(pg));
            c.u.push_back(std::move(pu));
            c.d.push_back(std::move(pd));
        }
        parity_case(c, s.name, 8);
    }

    // flags-off canonical: SWIZZLE=0 / CACHE_MIB=0 must not reinterpret a declared Band8
    // (services constructed AFTER the env change; the env is read once per service)
    setenv("STRATA_EXL3_CPU_SWIZZLE", "0", 1);
    setenv("STRATA_EXL3_CPU_CACHE_MIB", "0", 1);
    {
        Case c; c.H = 2560; c.I = 640; c.bits = 3; c.hb = 0;
        std::mt19937 rng(777);
        for (int e = 0; e < 8; ++e)
        {
            Projection pg, pu, pd;
            pg.make(c.H, c.I, 3, 0, rng);
            pu.make(c.H, c.I, 3, 0, rng);
            pd.make(c.I, c.H, 3, 0, rng);
            c.g.push_back(std::move(pg));
            c.u.push_back(std::move(pu));
            c.d.push_back(std::move(pd));
        }
        std::vector<Exl3CpuMatrix> gb, ub, db;
        for (int e = 0; e < 8; ++e)
        {
            gb.push_back(c.g[(size_t)e].mat(Exl3CpuTrellisLayout::Band8));
            ub.push_back(c.u[(size_t)e].mat(Exl3CpuTrellisLayout::Band8));
            db.push_back(c.d[(size_t)e].mat(Exl3CpuTrellisLayout::Band8));
        }
        Exl3CpuMoe band; band.set_threads(8);
        band.add_layer(0, gb.data(), ub.data(), db.data(), 8);
        std::mt19937 rng2(1234 + 3 * 7 + 2560);
        std::normal_distribution<float> dist(0.0f, 0.5f);
        std::vector<float> x((size_t)8 * 2 * c.H + c.H);
        for (auto& v : x) v = dist(rng2);
        const int ids[3] = { 1, 3, 5 }, t[3] = { 5, 5, 5 }, r[3] = { 2, 0, 1 };
        std::vector<float> out((size_t)3 * c.H);
        band.run_rows(0, x.data(), 2 * (int64_t)c.H, ids, t, r, 3, out.data(), c.H);
        bool finite = true;
        for (float v : out) finite = finite && std::isfinite(v);
        // Correctness under flags-off is bitwise vs the same computation with flags-on.  The
        // checksum pins it: register the same bytes on a flags-on service and compare.
        setenv("STRATA_EXL3_CPU_SWIZZLE", "1", 1);
        setenv("STRATA_EXL3_CPU_CACHE_MIB", "16384", 1);
        Exl3CpuMoe band_on; band_on.set_threads(8);
        band_on.add_layer(0, gb.data(), ub.data(), db.data(), 8);
        std::vector<float> out2((size_t)3 * c.H, -1e30f);
        band_on.run_rows(0, x.data(), 2 * (int64_t)c.H, ids, t, r, 3, out2.data(), c.H);
        check(std::memcmp(out.data(), out2.data(), out.size() * sizeof(float)) == 0 && finite &&
              band.stats().repack_fills == 0 && band.stats().cache_bytes == 0,
              "SWIZZLE=0 + CACHE_MIB=0 keep a declared Band8 canonical and copy-free");
    }
    setenv("STRATA_EXL3_CPU_SWIZZLE", "1", 1);
    setenv("STRATA_EXL3_CPU_CACHE_MIB", "16384", 1);

    // single-worker background join runs inline on the band-8 path (degenerate pool shape)
    {
        Case c; c.H = 2560; c.I = 640; c.bits = 4; c.hb = 0;
        std::mt19937 rng(99);
        for (int e = 0; e < 2; ++e)
        {
            Projection pg, pu, pd;
            pg.make(c.H, c.I, 4, 0, rng);
            pu.make(c.H, c.I, 4, 0, rng);
            pd.make(c.I, c.H, 4, 0, rng);
            c.g.push_back(std::move(pg));
            c.u.push_back(std::move(pu));
            c.d.push_back(std::move(pd));
        }
        std::vector<Exl3CpuMatrix> gb, ub, db;
        for (int e = 0; e < 2; ++e)
        {
            gb.push_back(c.g[(size_t)e].mat(Exl3CpuTrellisLayout::Band8));
            ub.push_back(c.u[(size_t)e].mat(Exl3CpuTrellisLayout::Band8));
            db.push_back(c.d[(size_t)e].mat(Exl3CpuTrellisLayout::Band8));
        }
        Exl3CpuMoe band; band.set_threads(1);
        band.add_layer(0, gb.data(), ub.data(), db.data(), 2);
        std::mt19937 rng2(5);
        std::normal_distribution<float> dist(0.0f, 0.5f);
        std::vector<float> x((size_t)c.H);
        for (auto& v : x) v = dist(rng2);
        const int ids[2] = { 0, 1 }, t[2] = { 0, 0 }, r[2] = { 0, 1 };
        std::vector<float> a((size_t)2 * c.H, -1e30f), b((size_t)2 * c.H, -1e30f);
        band.run_rows(0, x.data(), c.H, ids, t, r, 2, a.data(), c.H);
        band.run_rows_bg(0, x.data(), c.H, ids, t, r, 2, b.data(), c.H);
        band.run_rows_join();
        check(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
              "single-worker bg join runs inline, band-8, bitwise the blocking call");
    }

    if (failures > 0)
    {
        std::printf("%d FAILURES\n", failures);
        return 1;
    }
    std::printf("all band-8 canonical checks passed\n");
    return 0;
}
