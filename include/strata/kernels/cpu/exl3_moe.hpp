// include/strata/kernels/cpu/exl3_moe.hpp - the EXL3 CPU expert tier (B1).
//
// A port of exllamav3's cpu/moe_mul1.cpp (turboderp/exllamav3): the mul1 (cb=2) trellis codebook
// is affine in a byte-sum, so dequantization and the activation product fuse into int8 dot
// products (AVX-512 VNNI/VBMI on Zen 4+, AVX2 elsewhere; the tiers are bit-exact with each
// other).  Strata uses it exactly where the GGUF engine uses its CPU expert pool: when the
// VRAM expert cache is FULL, a miss is COMPUTED here instead of evicting a hot expert - the
// overflow tier of the GGUF policy (src/core/expert_source.cpp "THE DECISION"), restored for
// EXL3.
//
// Layers are registered once with raw pointers into the mmap'd safetensors (which the caller
// must keep alive).  The vector tiers read the trellis from a REPACKED per-expert copy (band-8
// on AVX-512, band-2 + planar dwords on AVX2 - the layout those kernels stream sequentially);
// the copies live in a bounded LRU cache (STRATA_EXL3_CPU_CACHE_MIB, 0 uses native).
// In-flight jobs pin their resolved copies across eviction; those bytes may temporarily
// exceed the retained LRU budget. Every tier can also read the native GPU-compatible layout.
#pragma once

#include <cstdint>
#include <memory>

namespace strata::kernels::cpu {

/// The physical trellis layout of the bytes an Exl3CpuMatrix's `trellis` points at.
///
/// `Native` is the safetensors order: tile (kt, nt) at uint16 offset (kt * (n/16) + nt) * P.
/// `Band8` is the AVX-512 band kernels' canonical band-8 order (the layout the repack cache
/// copies into; notes/results/2026-11-06-cpu-banded-format has the contract): native tile
/// (kt, nt) is stored at uint16 offset (((nt/8) * Tk + kt) * 8 + nt%8) * P, with Tk = k/16
/// and P = 16*bits + 8*hb words per tile. Words inside a tile and the suh/svh scales are
/// byte-preserved - a pure tile permutation of unchanged size, never a requantization.
///
/// Registration with `Band8` requires an AVX-512BW-class CPU (the band kernels' tier) and one
/// of the verified rates K2/K2.5/K3/K4 (K8's dword kernel reads native only), and is REFUSED
/// at add_layer time otherwise - packed bytes are never silently decoded as native, on any
/// CPU. Dimensions must make n/16 a multiple of 8 (k and n multiples of 128, as ever).
enum class Exl3CpuTrellisLayout {
    Native = 0,
    Band8 = 1,
};

/// One quantized matrix of one expert projection.  `trellis` points at the projection's
/// trellis words in the physical layout `layout` names - by default the native safetensors
/// order [k/16, n/16, 16*K] int16 (16*K+8 at the half-integer rates); suh/svh are fp16 bit
/// patterns of k and n entries.  k and n must be multiples of 128, K in [1, 8] (+0.5).
///
/// BINARY ABI: adding `layout` grew the struct (40 -> 48 bytes with padding).  SOURCE
/// compatibility is kept - every existing 7-field aggregate initializer still registers
/// native bytes - but binary compatibility is NOT: all consumers must rebuild when linking a
/// CPU library built with this header.  Engine binaries compiled against the previous header
/// keep working with the previously compiled library until such a rebuild.
struct Exl3CpuMatrix {
    const uint16_t* trellis = nullptr;
    const uint16_t* suh = nullptr;
    const uint16_t* svh = nullptr;
    int k = 0;
    int n = 0;
    int bits = 0;   // integer part of the rate
    int hb = 0;     // 1 = the half-integer rate bits + 0.5
    /// Physical layout of `trellis` (see Exl3CpuTrellisLayout).  LAST member with a default,
    /// so existing 7-field aggregate initializers keep registering native bytes.  The caller
    /// owns the bytes in either layout and must keep them alive, exactly like the pointers.
    Exl3CpuTrellisLayout layout = Exl3CpuTrellisLayout::Native;
};

class Exl3CpuMoe {
public:
    /// True where the kernels exist (x86-64) and the CPU carries at least the tier the build
    /// can dispatch safely.  False on other ISAs, and on MSVC builds (whose single TU carries
    /// AVX-512 code) when the machine is below AVX2.
    static bool supported();

    /// True when THIS process's resolved CPU tier (STRATA_EXL3_CPU_MAX_ISA honored) can
    /// register caller-packed Band8 trellises: the AVX-512BW-class band kernels.  Callers
    /// that are about to spend minutes building a packed cache use this to refuse a machine
    /// that could only fail later, at add_layer time.
    static bool band8_supported();

    Exl3CpuMoe();
    ~Exl3CpuMoe();
    Exl3CpuMoe(const Exl3CpuMoe&) = delete;
    Exl3CpuMoe& operator=(const Exl3CpuMoe&) = delete;

    /// Register one MoE layer's routed experts (gate/up/down per expert, index e in the
    /// vectors).  All pointers must outlive this object.  Throws std::runtime_error on a
    /// geometry the kernels cannot run (see Exl3CpuMatrix), on ANY expert whose gate/up/down
    /// do not chain H->I->H with the layer's expert-0 dimensions (checked per expert, from
    /// metadata only), on an unknown layout value, and
    /// on a Band8 matrix whose CPU tier, rate (K2/K2.5/K3/K4 only) or per-expert projection
    /// agreement fails - the three projections of ONE expert must share a layout (a layer may
    /// mix canonical and native experts); a mixed-projection expert is refused outright, never
    /// partially reinterpreted.  Validation reads metadata only: no trellis byte is touched
    /// at registration.  A registered Band8 matrix is consumed directly from the caller's
    /// bytes regardless of STRATA_EXL3_CPU_SWIZZLE / STRATA_EXL3_CPU_CACHE_MIB (those gate
    /// optional conversions of NATIVE bytes, not a declared physical format; no LRU entry,
    /// repack or second copy is ever made for it).
    void add_layer(int layer, const Exl3CpuMatrix* gates, const Exl3CpuMatrix* ups,
                   const Exl3CpuMatrix* downs, int num_experts);

    bool ready() const;

    /// Force the worker count (0 = the default: one per physical core, the calling thread
    /// joining as worker 0).  Optional; the pool otherwise starts on first use.
    void set_threads(int threads);

    /// `count` routed selections of ONE layer.  Token t[i] reads its fp32 activation at
    /// x + t[i] * x_stride; selection i's expert ids[i] has its UNWEIGHTED fp32 output row
    /// written to out + r[i] * out_stride.  Router weights are NOT applied - the caller's
    /// combine applies them, exactly like the GGUF CPU pool's rows.  Selections of one expert
    /// are grouped and computed together.  Blocking; internally parallel over the pool.
    void run_rows(int layer, const float* x, int64_t x_stride, const int* ids, const int* t,
                  const int* r, int count, float* out, int64_t out_stride);

    /// The overlapping variant for the engine's decode/verify callbacks: prepare the same
    /// computation (classify, repack, arena) and publish it on the pool's workers WITHOUT the
    /// calling thread joining, then return at once.  The caller does its GPU work and calls
    /// run_rows_join() to barrier.  The input and output buffers must stay untouched until
    /// the join.  One outstanding job per calling thread; with a single-worker pool the join
    /// runs the job inline (the degenerate blocking shape).
    void run_rows_bg(int layer, const float* x, int64_t x_stride, const int* ids, const int* t,
                     const int* r, int count, float* out, int64_t out_stride);
    void run_rows_join();

    struct Stats {
        int64_t calls = 0;            // run_rows invocations
        int64_t rows = 0;             // selections computed
        int64_t repack_fills = 0;     // experts repacked into the cache
        int64_t repack_evictions = 0;
        int64_t native_reads = 0;     // expert chunks served from native (uncached) bytes
        double ms = 0.0;              // blocking service plus background residual join wait (not compute time)
        double prepare_ms = 0.0;      // host preparation/repack before background issue
        double join_ms = 0.0;         // residual background wait after GPU enqueue
        double trellis_gb = 0.0;      // trellis bytes the GEMV phases streamed
        int threads = 0;
        int isa = 0;                  // 0 scalar, 1 avx2, 2 bw, 3 vnni, 4 vbmi
        int bg_workers = 0;           // participants of the most recent run_rows_bg job (the
                                      // pool's workers WITHOUT the calling thread - the number
                                      // the decode setup and phase barriers actually assumed)
        int64_t cache_bytes = 0;      // current repack-cache size
    };
    Stats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::kernels::cpu
