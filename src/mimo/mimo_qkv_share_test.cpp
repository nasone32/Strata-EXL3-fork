// src/mimo/mimo_qkv_share_test.cpp - CPU unit test for the PROTOTYPE planner (new file; not in
// any CMake target yet; compiled standalone: g++ -O2 -std=c++17 -ffp-contract=off this file).
//
// What it falsifies, and what it deliberately does NOT:
//
//  1. The central idea-B claim: a column's output in native_mmvq_multi_kernel does not
//     depend on NCOLS (nor on other columns' data).  Emulated here is the EXACT-layout
//     instantiation the Q8_0 path takes on the MiMo target (NW = WARPS = 4 warps, ROWS
//     1 or 2, no TS/PAIR/MULTI), thread-for-thread: kbx = tid/4 stepping by BPI = 32,
//     kqs = 2*(tid%4), per-element tmp accumulation, warp-ascending partial adds, XOR
//     tree - the structure of native_mmvq.cu:1180-1272.  The device compiler may
//     contract differently than the host, but any such choice is NCOLS-independent, so
//     host-side equality of the structure is the falsifiable core of the claim; the
//     device-level bitwise check is mmvq_multi_parity's job (extend it with MiMo shapes
//     when a GPU slot frees).
//
//  2. Tile-plan coverage: every token row launched exactly once, wide tiles only at
//     their exact width, remainders on the <= 8 path, for the engine's real T values
//     (decode 1, verify 3..8, prompt chunks up to 512 and beyond).
//
//  3. Eligibility rules mirrored from the library (bunch decline cases, wide gates).
//
//  4. Output stride math per projection (q 12288, k/v 768 and 1536 rows at n_in 4096).
#include "mimo_qkv_share.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::mimo;

// ------------------------------------------------------------------ the kernel emulator
namespace {

int32_t dp4a(int32_t a, int32_t b, int32_t c) {  // exact integer dot of 4 byte lanes + c
    const uint8_t* pa = reinterpret_cast<const uint8_t*>(&a);
    const uint8_t* pb = reinterpret_cast<const uint8_t*>(&b);
    for (int i = 0; i < 4; ++i) {
        const int32_t xa = int32_t(int8_t(pa[i])), xb = int32_t(int8_t(pb[i]));
        c += xa * xb;
    }
    return c;
}

struct Q80 { float d; int8_t qs[32]; };          // { fp16 d ; int8 qs[32] }, 34 B
struct Q81 { float d; int8_t qs[32]; };          // ds low half (the sum half is unused by the dot)

int32_t lane_int(const int8_t* qs, int k) {      // load_int_b2(w->qs, k): int at int-index k
    int32_t v;
    std::memcpy(&v, qs + 4 * k, 4);
    return v;
}

/// y[j * n_out + row] for j < ncols, exactly the multi kernel's thread/reduction structure.
void multi_emulate(const std::vector<Q80>& w, const std::vector<Q81>& x, int n_in, int n_out,
                   int ncols, int rows_per_block, std::vector<float>& y) {
    const int blocks_per_row = n_in / 32, x_stride = n_in / 32;
    const int NW = 4, WARP = 32, T = 4, BPI = 32;  // SmallTraits<Q80Block, 8> at NW = WARPS
    y.assign(size_t(ncols) * n_out, 0.0f);
    for (int bx = 0; bx < (n_out + rows_per_block - 1) / rows_per_block; ++bx) {
        const int row0 = rows_per_block * bx;
        for (int j0 = 0; j0 < ncols; ++j0) {  // (per-column, for the emulator's convenience;
            //  the device loops j innermost - value-equal per element)
            // per thread, ONE accumulator per (block-)row: the kernel's tmp[j][i]
            static float tpart[128][2];
            for (int tid = 0; tid < NW * WARP; ++tid) {
                float t[2] = {0.0f, 0.0f};
                const int kqs = 2 * (tid % 4);
                for (int kbx = tid / T; kbx < blocks_per_row; kbx += BPI) {
                    for (int i = 0; i < rows_per_block; ++i) {
                        if (row0 + i >= n_out) continue;
                        const Q80& wb = w[size_t(row0 + i) * blocks_per_row + kbx];
                        const Q81& xb = x[size_t(j0) * x_stride + kbx];
                        int32_t sumi = 0;
                        sumi = dp4a(lane_int(wb.qs, kqs), lane_int(xb.qs, kqs), sumi);
                        sumi = dp4a(lane_int(wb.qs, kqs + 1), lane_int(xb.qs, kqs + 1), sumi);
                        t[i] += (wb.d * xb.d) * float(sumi);
                    }
                }
                tpart[tid][0] = t[0];
                tpart[tid][1] = t[1];
            }
            // warp 0 sums the other warps' partials warp-ascending, then the XOR tree
            for (int i = 0; i < rows_per_block; ++i) {
                if (row0 + i >= n_out) continue;
                float v[WARP];
                for (int lane = 0; lane < WARP; ++lane) {
                    v[lane] = tpart[lane][i];
                    for (int l = 1; l < NW; ++l) v[lane] += tpart[WARP * l + lane][i];
                }
                for (int off = WARP / 2; off > 0; off >>= 1) {   // __shfl_xor swaps OLD values:
                    float nv[WARP];                               // stage the pair updates
                    for (int lane = 0; lane < WARP; ++lane) nv[lane] = v[lane] + v[lane ^ off];
                    std::memcpy(v, nv, sizeof(v));
                }
                y[size_t(j0) * n_out + row0 + i] = v[i];
            }
        }
    }
}

float rand_half_exact(std::mt19937& rng) {  // 1 + k/2048: exact in fp16 AND fp32
    const int k = int(rng() % 1024);
    return 1.0f + float(k) / 2048.0f;
}

int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL: %s\n", what); ++fails; }
}

}  // namespace

int main() {
    // ---- 1. NCOLS independence (and ROWS 1 vs 2 neutrality) of the multi-kernel structure
    for (int n_in : {1024, 2048}) {
        const int n_out = 40, ncols_max = 32;
        std::mt19937 rng(20261105u + unsigned(n_in));
        std::vector<Q80> w(size_t(n_out) * (n_in / 32));
        std::vector<Q81> x(size_t(ncols_max) * (n_in / 32));
        for (auto& b : w) { b.d = rand_half_exact(rng); for (auto& q : b.qs) q = int8_t(int(rng() % 255) - 127); }
        for (auto& b : x) { b.d = rand_half_exact(rng); for (auto& q : b.qs) q = int8_t(int(rng() % 255) - 127); }
        std::vector<float> yref;                        // the widest run is the comparison base;
        multi_emulate(w, x, n_in, n_out, ncols_max, 2, yref);  // every narrower run must match its
        for (int ncols : {2, 4, 8, 16, 32}) {                 // columns bit-for-bit
            std::vector<float> y;
            multi_emulate(w, x, n_in, n_out, ncols, 2, y);
            bool same = true;
            for (int j = 0; j < ncols && same; ++j)
                for (int r = 0; r < n_out; ++r)
                    if (std::memcmp(&y[size_t(j) * n_out + r], &yref[size_t(j) * n_out + r], 4) != 0) { same = false; break; }
            check(same, "column bits equal across NCOLS widths");
            std::printf("n_in=%4d ncols=%2d: %s\n", n_in, ncols, same ? "ok" : "DIFFER");
        }
        {   // ROWS = 1 vs ROWS = 2 at NCOLS = 8 (each row keeps its own partials - grid-only change)
            std::vector<float> y1, y8;
            multi_emulate(w, x, n_in, n_out, 8, 1, y1);
            multi_emulate(w, x, n_in, n_out, 8, 2, y8);
            check(y1 == y8, "ROWS=1 equals ROWS=2 bitwise");
        }
    }

    // ---- 2. tile plan coverage for the engine's real T values
    for (int rows : {1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 511, 512, 1024, 2048})
        for (int wide : {0, 16, 32}) {
            const auto tiles = qkv_tile_plan(rows, wide);
            std::vector<int> seen(rows, 0);
            int prev = 0;
            for (const auto& t : tiles) {
                check(t.ncols >= 1, "tile ncols >= 1");
                check(!t.wide || t.ncols == wide, "wide tiles run at their exact width");
                check(t.wide || t.ncols <= 8, "non-wide tiles keep ncols <= 8");
                check(t.row0 == prev, "tiles are ordered and contiguous");
                prev = t.row0 + t.ncols;
                for (int j = 0; j < t.ncols; ++j) seen[size_t(t.row0 + j)]++;
            }
            check(prev == rows, "plan covers exactly all rows");
            for (int r : seen) check(r == 1, "each row covered exactly once");
        }

    // ---- 3. eligibility mirrors
    check(qkv_bunch_eligible(4096, 1) == false, "ncols=1 (decode T=1) is not bunchable");
    for (int nc = 2; nc <= 8; ++nc) check(qkv_bunch_eligible(4096, nc), "bunch eligible at nc 2..8, n_in 4096");
    check(!qkv_bunch_eligible(4096, 9), "ncols=9 declined");
    check(!qkv_bunch_eligible(64, 4), "n_in/32 < BPI declined");
    check(!qkv_bunch_eligible(4096, 4, /*multi_exact=*/false), "upstream layout declined (falls back, bitwise)");
    check(!qkv_bunch_eligible(4096, 4, true, /*gfx906_wave_active=*/true), "gfx906 wave build declined");
    check(qkv_wide_eligible(4096, 16) && qkv_wide_eligible(4096, 32), "wide eligible at 16/32");
    check(!qkv_wide_eligible(4096, 8) && !qkv_wide_eligible(4096, 24), "wide only at exact 16/32");
    check(!qkv_wide_eligible(4096, 32, /*tsum_on=*/true), "wide declined under STRATA_TSUM=1");
    check(!qkv_wide_eligible(4096, 32, false, /*multi_exact=*/false), "wide declined under the upstream layout");
    check(!qkv_wide_eligible(4096, 32, false, true, /*gfx906_wave_active=*/true), "wide declined on gfx906 wave");
    check(q8_1_bytes(4096, 8) == 8 * 128 * 36, "q8_1 bytes: 36 per 32 elements per column");

    // ---- 4. per-projection stride math (MiMo geometry: q 12288 rows; k/v 768 full, 1536 swa)
    const auto tiles = qkv_tile_plan(512, 32);
    const QkvTile& first = tiles.front();
    check(first.wide && first.ncols == 32 && first.row0 == 0, "512 rows open with a 32-wide tile");
    for (int64_t n_out : {12288, 768, 1536}) {
        bool ok = true;
        for (int j = 0; j < first.ncols; ++j)
            for (int row = 0; row < 8; ++row)
                if (qkv_out_offset(first, j, row, n_out) != (int64_t(j)) * n_out + row) ok = false;
        check(ok, "output offsets hit the row-major [rows][n_out] buffers");
    }

    std::printf(fails == 0 ? "mimo_qkv_share_test: ALL OK\n" : "mimo_qkv_share_test: %d FAILURE(S)\n", fails);
    return fails == 0 ? 0 : 1;
}
