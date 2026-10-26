// src/kernels/cuda/exl3_experts.cu - one layer's routed EXL3 experts on the GPU (see the header).
#include "strata/kernels/exl3_experts.hpp"
#include "strata/artifact/exl3_model.hpp"
#include "strata/core/exl3_flags.hpp"
#include "strata/kernels/exl3_route_plan.hpp"
#include "strata/kernels/exl3_arena_layout.hpp"
#include "strata/platform/aux_cpus.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iterator>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define STRATA_EXL3_PAUSE() _mm_pause()
#else
#define STRATA_EXL3_PAUSE() ((void) 0)
#endif

namespace strata::kernels {

struct Exl3ExpertStore::Impl {
    std::vector<Exl3Mat> gate, up, down;
    std::vector<void*> allocs;
    std::unique_ptr<Exl3MoeWorkspace> workspace;
    int device = -1;
    Impl() {
        if (cudaGetDevice(&device) != cudaSuccess) throw std::runtime_error("exl3: no CUDA device");
        workspace = std::make_unique<Exl3MoeWorkspace>();
    }
    ~Impl() {
        int previous = -1;
        (void) cudaGetDevice(&previous); (void) cudaSetDevice(device);
        for (void* p : allocs) (void) cudaFree(p);
        workspace.reset();
        if (previous >= 0) (void) cudaSetDevice(previous);
    }
};

namespace {

Exl3Mat upload(const Exl3Linear& L, std::vector<void*>& allocs, cudaStream_t st, size_t& bytes) {
    auto alloc = [&](size_t n, const void* src) -> void* {
        void* p = nullptr;
        if (cudaMalloc(&p, n) != cudaSuccess) { std::fprintf(stderr, "exl3 store: cudaMalloc failed\n"); std::abort(); }
        (void) st;
        if (cudaMemcpy(p, src, n, cudaMemcpyHostToDevice) != cudaSuccess) { std::fprintf(stderr, "exl3 store: memcpy failed\n"); std::abort(); }
        allocs.push_back(p);
        bytes += n;
        return p;
    };
    int words = 256 * L.bits / 16;
    size_t tr_bytes = (size_t) L.ki * L.nj * words * 2;
    Exl3Mat M;
    M.trellis = (const uint16_t*) alloc(tr_bytes, L.trellis);
    M.suh = (const uint16_t*) alloc((size_t) L.in() * 2, L.suh);
    M.svh = (const uint16_t*) alloc((size_t) L.out() * 2, L.svh);
    M.ki = L.ki; M.nj = L.nj; M.bits = L.bits; M.cb = L.cb;
    return M;
}

}  // namespace

Exl3ExpertStore::Exl3ExpertStore(const std::string& model_dir, int layer, int n_experts, void* stream) {
    if (layer < 0 || layer >= 48 || n_experts <= 0 || n_experts > 512)
        throw std::invalid_argument("exl3: invalid expert store geometry");
    auto pending = std::make_unique<Impl>();
    auto* state = pending.get();
    n_experts_ = n_experts;
    cudaStream_t st = (cudaStream_t) stream;
    Exl3Model m(model_dir);
    const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".mlp.experts.";
    for (int e = 0; e < n_experts; ++e) {
        const std::string b = prefix + std::to_string(e);
        state->gate.push_back(upload(m.linear(b + ".gate_proj"), state->allocs, st, bytes_));
        state->up.push_back(upload(m.linear(b + ".up_proj"), state->allocs, st, bytes_));
        state->down.push_back(upload(m.linear(b + ".down_proj"), state->allocs, st, bytes_));
    }
    state->workspace->reserve(n_experts, state->gate[0].ki * 16, state->gate[0].nj * 16,
                              state->down[0].nj * 16);
    impl_ = pending.release();
}

Exl3ExpertStore::~Exl3ExpertStore() { delete impl_; }

void Exl3ExpertStore::run(const uint16_t* x, const int* ids, const float* weights, int k, uint16_t* out,
                          void* stream) const {
    if (k == 0) return;
    int device = -1;
    if (k < 0 || k > n_experts_ || !x || !ids || !weights || !out ||
        cudaGetDevice(&device) != cudaSuccess || device != impl_->device)
        throw std::invalid_argument("exl3: expert store arguments/device mismatch");
    std::vector<Exl3Mat> g(k), u(k), d(k);
    for (int i = 0; i < k; ++i) {
        int e = ids[i];
        if (e < 0 || e >= n_experts_) throw std::invalid_argument("exl3: expert index out of range");
        g[i] = impl_->gate[e];
        u[i] = impl_->up[e];
        d[i] = impl_->down[e];
    }
    exl3_moe_ffn(g.data(), u.data(), d.data(), weights, k, x, out, stream, impl_->workspace.get());
}


// ================================ streaming experts ================================

namespace {

template <typename T> struct StreamBuffer {
    T* p = nullptr;
    size_t capacity = 0;
    void reserve(size_t count) {
        if (count <= capacity) return;
        if (count > SIZE_MAX / sizeof(T)) throw std::overflow_error("exl3: stream buffer overflow");
        T* next = nullptr;
        if (cudaMalloc((void**) &next, count * sizeof(T)) != cudaSuccess)
            throw std::runtime_error("exl3: stream buffer allocation failed");
        (void) cudaFree(p); p = next; capacity = count;
    }
    ~StreamBuffer() { if (p) (void) cudaFree(p); }
};

struct MultiBuffers {
    StreamBuffer<float> input, output;
    StreamBuffer<uint16_t> x, rows;
    StreamBuffer<int> input_rows, destinations;
    float* host_output = nullptr;
    size_t host_capacity = 0;
    void reserve(int tokens, int routes, int n, bool host_input = true) {
        if (host_input) input.reserve((size_t) tokens * n);
        x.reserve((size_t) tokens * n);
        output.reserve((size_t) routes * n); rows.reserve((size_t) routes * n);
        input_rows.reserve((size_t) routes);
        destinations.reserve((size_t) routes);
        const size_t count = (size_t) routes * n;
        if (count > host_capacity) {
            float* next = nullptr;
            if (cudaMallocHost((void**) &next, count * sizeof(float)) != cudaSuccess)
                throw std::runtime_error("exl3: pinned route output allocation failed");
            if (host_output) (void) cudaFreeHost(host_output);
            host_output = next; host_capacity = count;
        }
    }
    ~MultiBuffers() { if (host_output) (void) cudaFreeHost(host_output); }
};

template <typename T> struct MappedBuffer {
    T* host = nullptr;
    T* device = nullptr;
    void allocate(size_t count) {
        if (cudaHostAlloc((void**) &host, count * sizeof(T), cudaHostAllocMapped) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &device, host, 0) != cudaSuccess)
            throw std::runtime_error("exl3: verifier mapped allocation failed");
        std::fill_n(host, count, T{});
    }
    ~MappedBuffer() { if (host) (void) cudaFreeHost(host); }
};

constexpr int verify_routes = 8 * 64;
struct VerifyRecord {
    int routes = 0, misses = 0;
    int ids[verify_routes] = {};
    uint8_t hit[verify_routes] = {};
};

struct VerifyGraphBuffers {
    int max_tokens = 0, k = 0, max_ff = 0;
    bool host_active = false;
    MappedBuffer<Exl3Mat> table; // [layer, projection, expert], published only between readers
    StreamBuffer<Exl3Mat> device_table;
    MappedBuffer<VerifyRecord> record;
    MappedBuffer<uint32_t> completed;
    bool check = false;
    float* graph_out = nullptr;
    MappedBuffer<float> reference, actual;
    long long checked_rows = 0;
    StreamBuffer<Exl3Mat> gate, up, down;
    StreamBuffer<int> input_rows;
    StreamBuffer<uint32_t> skip;
    StreamBuffer<uint16_t> x;
    Exl3MoeWorkspace workspace; // never shared with callback/prefill workspaces
    long long hit_routes = 0, miss_routes = 0, all_hit_layers = 0, layers = 0;
    ~VerifyGraphBuffers() {
        std::fprintf(stderr, "strata EXL3 verify graph: %lld hit routes stayed on GPU, %lld host routes, "
                     "%lld/%lld all-hit layers bypassed the GPU host wait\n",
                     hit_routes, miss_routes, all_hit_layers, layers);
        if (check) std::fprintf(stderr, "strata EXL3 verify graph: %lld hit rows checked byte-exact against host FFN\n", checked_rows);
    }
};

__global__ void verify_gather_kernel(const Exl3Mat* table, const int32_t* ids, int routes, int k,
                                      Exl3Mat* gate, Exl3Mat* up, Exl3Mat* down, int* input_rows,
                                      VerifyRecord* record, uint32_t* skip, uint32_t ring,
                                      const float* x, int elements, float* mapped_x,
                                      int32_t* mapped_ids, uint32_t* seq) {
    __shared__ int misses;
    if (threadIdx.x == 0) misses = 0;
    __syncthreads();
    for (int r = threadIdx.x; r < routes; r += blockDim.x) {
        const int e = ids[r];
        const bool valid = e >= 0 && e < 512;
        gate[r] = valid ? table[e] : Exl3Mat{};
        up[r] = valid ? table[512 + e] : Exl3Mat{};
        down[r] = valid ? table[1024 + e] : Exl3Mat{};
        input_rows[r] = r / k;
        record->ids[r] = e;
        record->hit[r] = gate[r].trellis != nullptr;
        if (!gate[r].trellis) atomicAdd(&misses, 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        record->routes = routes;
        record->misses = misses;
        *skip = misses == 0 ? ring : 0;
    }
    if (misses) {
        for (int i = threadIdx.x; i < elements; i += blockDim.x) mapped_x[i] = x[i];
        for (int i = threadIdx.x; i < routes; i += blockDim.x) mapped_ids[i] = ids[i];
    }
    // Every lane's mapped stores precede the sequence. Hits run behind this kernel while
    // the callback protects their slots and stages only the missing experts elsewhere.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile uint32_t*) seq = ring;
}

template <bool Misses>
__global__ void verify_output_kernel(const Exl3Mat* gate, const uint16_t* rows, const float* mapped,
                                      float* out, int routes, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= routes * n) return;
    const bool hit = gate[i / n].trellis != nullptr;
    if constexpr (Misses) { if (!hit) out[i] = mapped[i]; }
    else { if (hit) out[i] = __half2float(__ushort_as_half(rows[i])); }
}

__global__ void verify_complete_kernel(uint32_t* completed) {
    __threadfence_system();
    *completed = 1;
}

__global__ void f16_to_f32_rows_kernel(const uint16_t* __restrict__ h, float* __restrict__ y, long n) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(*(const __half*) &h[i]);
}

__global__ void f32_to_f16_rows_kernel(const float* __restrict__ x, uint16_t* __restrict__ h, long n) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = __half_as_ushort(__float2half_rn(x[i]));
}

__global__ void scatter_route_output(const float* in, const int* destinations, float* out, int rows, int n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t) rows * n) out[(int64_t) destinations[i / n] * n + i % n] = in[i];
}

}  // namespace

struct Exl3ExpertStream::Impl {
    std::string dir;
    std::unique_ptr<Exl3Model> model;
    int device = -1;
    cudaStream_t compute_stream = nullptr;
    std::unique_ptr<Exl3MoeWorkspace> workspace;
    std::unique_ptr<MultiBuffers> multi;
    std::unique_ptr<VerifyGraphBuffers> verify_graph;
    Exl3RoutePlan route_plan;
    std::mutex run_mu;
    int n_embd = 2560;
    int n_layers = 48;
    int layer_lo = 0, layer_hi = 48;
    int max_k = 64;
    int cache = 0;                     // slots per layer; 0 = the per-token scratch
    uint16_t* d_x = nullptr;
    float* d_input = nullptr;
    uint16_t* d_rows = nullptr;
    float* d_out = nullptr;

    struct Slot { uint16_t* tr = nullptr; uint16_t* suh = nullptr; uint16_t* svh = nullptr;
                  size_t trb = 0, suhb = 0, svhb = 0; };
    struct Blocks {
        std::vector<Slot> g, u, d; std::vector<Exl3Mat> gm, um, dm;
        // Per slot, the "last reader" event of the native staging ring's algorithm (prefill.cpp's
        // `used`): recorded on the compute stream after the slot's last queued reader, so the
        // staging stream may overwrite the slot (a new expert's upload) without a wave-end sync.
        // STRATA_EXL3_PIPE only; a stale/never-recorded event waits as already complete.
        std::vector<cudaEvent_t> used;
    };

    Blocks scratch;                    // cache == 0: `max_k` slots, every use overwrites them
    struct Layer : Blocks {
        std::vector<void*> allocs;         // one block per (projection, plane), `cache` slots in it
        std::unordered_map<int, int> at;   // expert -> slot
        std::vector<uint32_t> tick;        // per slot: the last use, for the LRU
        int filled = 0;                    // slots handed out so far; the map's size does not track it after an eviction
        uint32_t clock = 0;
        std::vector<int> evicted;          // STRATA_EXL3_TIME: ids of the last 16 evictions, for churn accounting
        size_t evicted_at = 0;
    };
    std::vector<Layer> layers;         // cache > 0: `cache` slots per layer, resident across tokens
    bool arena_requested = false;      // PREFILL_BORROW currently selects backing only, no loan
    uint8_t* arena = nullptr;          // single owner; Layer::allocs stays empty in this arm
    size_t arena_bytes = 0, arena_rank_bytes = 0;
    // ---- the stream loan (contract in the header).  `active` is the routable slot prefix: without
    // a loan it is `cache`.  INVARIANT, every owned layer, between calls: the layer's mapped slots
    // are EXACTLY [0, filled) - `at` is a bijection onto that prefix and filled <= active.  The run
    // paths' dense fill/evict preserve it; lend clamps it; restore extends it into the tail.
    int active = 0;
    struct TailRecord { int layer; int expert; int slot; uint32_t tick; };   // what a lent rank held
    struct {
        int ranks = 0;                 // 0: no active loan (an empty saved list is still a loan)
        uint64_t epoch = 0;            // bumps per successful lend/restore; never on a rejection
        std::vector<TailRecord> saved; // appended at lend; an (layer, expert) may repeat across growths
        std::vector<std::vector<int>> restored;   // per owned layer: the LAST restore's placements
    } loan;

    // ---- the host-side cache (see the header).  Per layer, like the VRAM one: the routing is layer-specific,
    // and a slot holds that layer's nine pieces at fixed offsets, so a hit is a memcpy into the staging buffer.
    std::vector<uint8_t> ram;                     // pageable: pinning this much needs a memlock limit
    // STATIC ownership exclusion (peer tier): byte per (layer, expert); never admitted here
    const uint8_t* own_table = nullptr;
    const uint8_t* pending_table = nullptr;
    Exl3ExpertStream::RouteObserver route_observer = nullptr;
    void* route_observer_user = nullptr;
    int64_t own_layers = 0, own_expert = 0;
    std::atomic<int64_t> own_excl{0};
    bool owned_elsewhere(int64_t layer, int32_t expert) const {
        return own_table != nullptr && layer >= 0 && layer < own_layers &&
               expert >= 0 && expert < own_expert &&
               own_table[(size_t) layer * (size_t) own_expert + (size_t) expert] != 0;
    }
    bool admission_excluded(int64_t layer, int32_t expert) const {
        return owned_elsewhere(layer, expert) ||
            (pending_table && layer >= 0 && layer < own_layers && expert >= 0 && expert < own_expert &&
             pending_table[(size_t) layer * own_expert + expert] != 0);
    }
    // grow the scratch blocks to `count` slots (owned-elsewhere fallback compute needs a whole
    // wave's worth; setup time only - never in a callback).  On failure the stream requires
    // DESTRUCTION (partial successful allocations are retained until then; there is no
    // rollback), and an incomplete earlier growth is reported rather than skipped.
    bool grow_scratch(int count, std::string* err) {
        // COMPLETE-or-nothing: an earlier partial growth (g longer than u/d, or missing slot
        // events) leaves the stream unusable rather than silently skipping those components
        if (count <= (int) scratch.g.size()) {
            const bool complete = (int) scratch.u.size() >= count && (int) scratch.d.size() >= count &&
                                  (!pipe_stage || (int) scratch.used.size() >= count);
            if (!complete && err) *err = "exl3: scratch growth incomplete (destroy the stream)";
            return complete;
        }
        const std::string b0 = "model.language_model.layers." + std::to_string(layer_lo) + ".mlp.experts.0.";
        auto grow = [&](std::vector<Slot>& slots, std::vector<Exl3Mat>& mats, const char* proj) -> bool {
            Exl3Linear L = model->linear(b0 + proj);
            size_t trb = 0, suhb = 0, svhb = 0;
            for (int l = layer_lo; l < layer_hi; ++l) {
                const auto Sz = model->linear("model.language_model.layers." + std::to_string(l) +
                                              ".mlp.experts.0." + proj);
                trb = std::max(trb, (size_t) Sz.ki * Sz.nj * 32 * Sz.bits);
                suhb = std::max(suhb, (size_t) Sz.in() * 2);
                svhb = std::max(svhb, (size_t) Sz.out() * 2);
            }
            for (int i = (int) slots.size(); i < count; ++i) {
                // one slot's planes are owned locally until BOTH vectors publish them; any
                // failure - allocation or a throwing push_back - frees exactly what it holds
                Slot sl{};
                Exl3Mat m{};
                auto release = [&]() {
                    for (void* q : {(void*) sl.tr, (void*) sl.suh, (void*) sl.svh}) if (q) (void) cudaFree(q);
                };
                if (cudaMalloc((void**) &sl.tr, trb) != cudaSuccess ||
                    cudaMalloc((void**) &sl.suh, suhb) != cudaSuccess ||
                    cudaMalloc((void**) &sl.svh, svhb) != cudaSuccess) {
                    release();
                    if (err) *err = "exl3: scratch growth failed";
                    return false;
                }
                sl.trb = trb; sl.suhb = suhb; sl.svhb = svhb;
                m.trellis = sl.tr; m.suh = sl.suh; m.svh = sl.svh;
                m.ki = L.ki; m.nj = L.nj; m.bits = L.bits; m.cb = L.cb;
                try {
                    slots.push_back(sl);
                } catch (...) { release(); return false; }
                try {
                    mats.push_back(m);
                } catch (...) {
                    release();
                    slots.pop_back();
                    return false;
                }
            }
            return true;
        };
        if (!grow(scratch.g, scratch.gm, "gate_proj") || !grow(scratch.u, scratch.um, "up_proj") ||
            !grow(scratch.d, scratch.dm, "down_proj")) return false;
        if (pipe_stage) {
            for (size_t i = scratch.used.size(); i < (size_t) count; ++i) {
                cudaEvent_t e = nullptr;
                if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
                    if (err) *err = "exl3: scratch slot event failed";
                    return false;
                }
                scratch.used.push_back(e);
            }
        }
        return true;
    }
    std::vector<size_t> ram_base;                 // per layer: where its slots start in `ram`
    std::vector<size_t> ram_slot_bytes;           // per layer: one expert's nine pieces
    std::vector<std::array<size_t, 9>> ram_off;   // per layer: the nine piece offsets inside a slot
    int ram_slots = 0;                            // slots per layer
    struct RamLayer {
        std::unordered_map<int, int> at;          // expert -> slot
        std::vector<uint32_t> tick;               // per slot: the last use, for the LRU
        std::vector<int> key;                     // slot -> expert
        int filled = 0;
        uint32_t clock = 0;
    };
    std::vector<RamLayer> ram_layers;
    std::vector<uint8_t> ram_pinned;
    std::mutex ram_mu;
    long long ram_hits = 0, ram_misses = 0;
    int cur_layer = 0;                            // the layer `run` is on (the queue is host-side, single-threaded)

    // STRATA_EXL3_TIME=1: per-token upload/ffn/d2h milliseconds and the cache hit rate
    long long hits = 0, misses = 0;
    long long cpu_refused = 0;   // routes handed to the caller's CPU tier (STRATA_EXL3_CPU_TIER)
    bool cpu_tier = false;       // the env switch; a null cpu_rows buffer keeps today's behavior
    double up_ms = 0, ffn_ms = 0, d2h_ms = 0, call_ms = 0;
    uint64_t upload_bytes = 0;
    cudaEvent_t profile[4] = {nullptr, nullptr, nullptr, nullptr};
    bool wmma_group = false;   // STRATA_EXL3_WMMA=1: the gfx11 matrix-core group FFN (opt-in, A/B)
    bool wave_sync = false;    // STRATA_EXL3_WAVESYNC=1: one device-FFN stream sync per wave, not per tile
    long long sync_count = 0;  // device-FFN stream syncs and the host time spent waiting in them
    double sync_ms = 0.0;
    long long evictions = 0, churn_misses = 0;   // cache diagnostics under STRATA_EXL3_TIME

    // STRATA_EXL3_PROF=1 (opt-in, verification-profile milestone): window-service profile.
    // HOST COUNTERS ONLY - no events, no GPU buffers, no allocation anywhere: the fine
    // per-(layer,expert)-group route-row histogram, dispatch-arm route counts, and host-elapsed
    // phase spans (route planning / admission+preparation / staging-worker join).  One line per
    // verify window next to the STRATA_EXL3_TIME line (counters accumulate over the window's
    // per-layer calls and print after the range's last layer, like that line).  Unset = the
    // default path does no extra work.
    bool prof = false;
    long long prof_routes = 0, prof_groups = 0;
    long long prof_row_hist[8] = {};   // per-group route rows: 1,2,3,4,5,6-8,9-15,>=16
    long long prof_arm_routes[4] = {}; // routes dispatched: 0 = indexed GEMV, 1 = group (wmma|gemm),
                                       // 2 = expert-batch, 3 = grouped small-M (STRATA_EXL3_SMALL_M)
    double prof_plan_ms = 0, prof_prep_ms = 0, prof_join_ms = 0;

    // STRATA_EXL3_PIPE (default on, "0" off): prefill wave pipelining (A/B).  While wave W's FFN
    // runs on the compute stream, wave W+1's expert reads and H2D uploads run on the staging stream
    // into slots wave W is done with (per-slot `used` events order the reuse - the native prefill
    // ring's algorithm on these slots), so PCIe transfers overlap compute instead of queueing
    // behind a wave-end sync.  Implies the per-wave barrier.
    bool pipe_stage = false;
    bool expert_batch = false;  // temporary STRATA_EXL3_EXPERT_BATCH: bounded multi-expert launches
    long long batch_groups = 0, batch_calls = 0, group_hist[5] = {};
    long long ffn_launches = 0, standalone_groups = 0, batch_rejects = 0;
    long long split_stops[4] = {}; // small next group, route cap, row-tile boundary, eight-group cap

    // STRATA_EXL3_OVERLAP=1: the HOST-IO arm's waves (run_multi - the verifier decode windows and
    // run()'s duplicate-id redirect; never run_device/prefill) issue the resident routes' FFN
    // BEFORE the miss staging joins, the same split the single-token run() path has always had.
    // The wave's routes are partitioned into a hit pass and a miss pass that SHARE the route
    // buffers by offset (hit rows [0, nh), miss rows [nh, total)), so no mid-call drain and no
    // extra device memory are needed; the miss pass is ordered after the staging by the existing
    // stage_done event.  STRATA_EXL3_OVERLAP=0 keeps the original all-after-staging order.
    bool overlap_stage = false;
    long long ov_splits = 0, ov_fallbacks = 0;   // lifetime split/fallback wave counts (overlap_debug)
    cudaEvent_t ov_ev[2] = {nullptr, nullptr};   // STRATA_EXL3_TIME + OVERLAP: the hit pass' span on st
    double ov_hit_ms = 0, ov_stage_ms = 0, ov_overlap_ms = 0;   // per-window, reset with the others

    // STRATA_EXL3_SMALL_M=1 (opt-in, primary-only milestone): the prototype grouped small-M FFN
    // (`exl3_moe_rows_grouped`, docs/EXL3-HIP-BENCH.md) replaces the indexed GEMV on the HOST-IO
    // arms' decode windows whose shape measured a per-call win: window tokens >= 3, every group of
    // the pass at most 6 rows (the prototype's compiled width), and the pass's route reuse >= 1.4
    // (measured on gfx1100: mix-M3 reuse 1.43 won -9.4% e2e; mix-M2 reuse 1.25 LOST +8.9%, mix-M1
    // 1.00 lost +4..7% - so M1/M2-shaped passes stay on the indexed GEMV).  A simple transparent
    // threshold, not a heuristic grid; every stay is counted and printed.  The device/prefill arms
    // and all default paths are untouched.  An active ownership-elsewhere tier (peer or remote
    // helper -> own_table set) bypasses the new path for the whole primary service, preserving
    // those baselines; whether the peer tier adopts it is a later milestone's decision.
    bool small_m = false;
    int primary_multirow = 0;           // opt-in fused groups, retaining three projection launches
    bool skip_empty_stage = false, pinned_plan = false;
    bool primary_pack = false;
    long long primary_packed_calls = 0, primary_packed_routes = 0;
    struct HostPlan {
        void* allocation = nullptr;
        uint8_t* device = nullptr;  // freed under Impl's owning device, after stream drain
        Exl3Mat *gate = nullptr, *up = nullptr, *down = nullptr;
        int *first = nullptr, *rows = nullptr;
        ~HostPlan(){if(allocation) (void)cudaFreeHost(allocation);}
    } host_plan[2];
    int pinned_plan_capacity = 0;
    long long small_m_calls = 0, small_m_routes = 0;
    long long small_m_stays[3] = {};     // passes kept on the indexed GEMV: window<2 / group>6 rows / reuse<1.2
    int small_m_envelope = 0;            // routes pre-reserved at SETUP for the flag's whole device budget
    bool small_m_trace = false;          // STRATA_EXL3_SMALL_M_TRACE=1: unbuffered stderr breadcrumbs,
                                         // a stall-debug aid (finds the exact blocking call); no default output
    // STRATA_EXL3_SMALL_M's per-pass plan sources, FLAG-OWNED and PERSISTENT for the stream's
    // life, preallocated once at construction: the host callback must not allocate, and the
    // flag-off path must not even construct them.  [0] = the hit pass of a split wave / the
    // default arm's tile, [1] = the miss pass - separate sets so a miss-pass rebuild never
    // overwrites a hit pass's tables while its pageable uploads are being consumed.  Host-IO
    // flag calls are never pipelined (pipeline is device_io) and every flag-on wave/tile ends
    // with a full stream barrier, so rebuilding between waves/tiles is safe under the same
    // pageable-source contract the route arrays use.
    std::vector<int> sm_gfirst[2], sm_grows[2];
    bool append_small_group(int pass, int& prefix, int rows, const Exl3Mat& g,
                            const Exl3Mat& u, const Exl3Mat& d, int& widest) {
        if (primary_multirow && (g.cb != 2 || u.cb != 2 || d.cb != 2)) return false;
        const int width = primary_multirow ? primary_multirow : rows;
        while (rows > 0) {
            const int count = std::min(rows, width);
            sm_gfirst[pass].push_back(prefix); sm_grows[pass].push_back(count);
            sm_gg[pass].push_back(g); sm_gu[pass].push_back(u); sm_gd[pass].push_back(d);
            prefix += count; rows -= count; widest = std::max(widest, count);
        }
        return true;
    }
    std::vector<Exl3Mat> sm_gg[2], sm_gu[2], sm_gd[2];
    // The per-wave host sources (route/matrix arrays the FFN entry points upload from pageable
    // memory).  Two banks, alternating per wave: the pipeline reuses bank (W-2)'s vectors only
    // after wave W-2's `wave_done` event (recorded after its last enqueue on the compute stream)
    // has fired, so no source is overwritten while a queued upload may still read it.
    struct WaveHost {
        std::vector<Exl3Mat> gates, ups, downs;    // one entry per group of the wave
        std::vector<Exl3Mat> route_g, route_u, route_d;
        std::vector<int> input_rows;
        std::vector<Exl3Group> groups;  // wave-long descriptor sources, never mutate queued slices
        std::vector<int> ov_dest;       // STRATA_EXL3_OVERLAP: destinations in the permuted (hits-first) order
    } wave_host[2];
    cudaEvent_t wave_done[2] = {nullptr, nullptr};
    // STRATA_EXL3_TIME + STRATA_EXL3_PIPE: four timestamp events per wave (staging begin/end on the
    // staging stream, FFN begin/end on the compute stream) - measured at the call-end barrier to
    // show the staging-vs-FFN overlap directly, not inferred from stream counts.
    std::vector<cudaEvent_t> pipe_ev;
    double pipe_stage_ms = 0, pipe_ffn_ms = 0, pipe_overlap_ms = 0;
    long long pipe_waves = 0;

    // ---- the miss reads
    //
    // **A MISS IS READ WITH `pread`, NOT OUT OF THE MMAP.**  Copying a tensor out of the mmap faults every
    // 4 KiB page on the way in: measured on 60 real expert tensors, 0.3 GB/s through the mmap against
    // 10.4 GB/s with `preadv` into a reused buffer, both from the page cache.  The read is the whole of the
    // doorbell: the token graph spins while the host fetches the expert.
    //
    // The destination is pinned staging, one buffer per worker, so the kernel writes straight into memory
    // the H2D can DMA from: no pageable staging copy, and the buffer's own pages never fault again.
    struct Job { uint16_t* dst; const uint16_t* src; int fd; size_t off; size_t bytes;
                 uint8_t* ram_dst = nullptr;       // a miss: fill this slot as a side effect of the read
                 const uint8_t* ram_src = nullptr; // a hit: copy this out of RAM through the pinned staging
                 cudaEvent_t wait = nullptr;       // STRATA_EXL3_PIPE: the destination slot's used event
               };
    struct PackedExpert {
        std::array<const uint16_t*, 9> pieces{};
        std::array<size_t, 9> bytes{};
    };
    // Only allocated by PACK_STAGE. Job fd=-2/src points here; the default Job layout is unchanged.
    std::unique_ptr<std::array<PackedExpert, 512>> packed_sources;
    size_t packed_count = 0;
    long long packed_jobs = 0;        // TIME evidence: experts actually coalesced
    std::vector<Job> jobs;
    std::vector<std::thread> workers;
    // **TWO BUFFERS PER WORKER, WITH AN EVENT ON EACH.**  The DMA out of a staging buffer is asynchronous,
    // so the next pread into the same buffer would overwrite data the DMA has not read yet (that is how the
    // first version produced NaNs).  Two buffers let a worker read into one while the other drains, and the
    // event orders the reuse.
    struct Stage { uint8_t* buf[2] = {nullptr, nullptr}; cudaEvent_t done[2] = {nullptr, nullptr}; int turn = 0; };
    std::vector<Stage> stage;          // one per worker (one for the sequential fallback)
    size_t stage_bytes = 0;
    bool pinned_mmap = false;          // opt-in: mmap misses use the existing double-buffered stage
    bool pack_stage = false;           // STRATA_EXL3_PACK_STAGE (default on, "0" off): gather an
                                       // arena expert's nine planes into one DMA
    std::map<std::string, int> fds;    // shard path -> fd, opened once
    std::mutex api_mu;                 // concurrent enqueues to one stream are not safe on the null stream
    // the resident/miss split: two output blocks (the FFN of each part writes its own) and the row maps
    uint16_t* d_hit = nullptr;
    uint16_t* d_miss = nullptr;
    int* d_hit_rows = nullptr;
    int* d_miss_rows = nullptr;
    std::atomic<size_t> job_at{0};
    std::atomic<uint64_t> work_done{0};
    std::atomic<int> stage_error{0};
    std::mutex work_mu;
    std::condition_variable work_cv;
    std::vector<std::unique_ptr<std::condition_variable>> worker_cv;
    uint64_t work_gen = 0;             // guarded by work_mu
    int work_target = 0;              // generation participants, guarded by work_mu
    bool work_quit = false;            // guarded by work_mu
    int n_workers = 0;
    int stage_inline = 0;            // STRATA_EXL3_STAGE_INLINE (default 2, "0" off): small host-decode
                                      // waves avoid waking the full pool
    bool stage_wake_limit = false;
    long long limited_waves = 0, skipped_acks = 0;
    long long inline_waves = 0, inline_jobs = 0;
    long long prof_stage_jobs[7] = {}; // 0,1,2,3,4,5..8,>8 jobs per host wave
    cudaStream_t work_stream = nullptr;   // the staging copies' own stream (see the ctor)
    cudaEvent_t stage_done = nullptr;     // recorded on it, waited on the FFN's stream

    bool staged(cudaError_t e) {
        if (e != cudaSuccess) stage_error.store((int) e, std::memory_order_relaxed);
        return e == cudaSuccess;
    }

    void stage_one(const Job& j, Stage& stg) {
        if (j.bytes == 0) return;
        const uint16_t* src = j.src;
        if ((j.ram_src != nullptr || (pinned_mmap && j.fd < 0) || j.fd == -2) && stg.buf[0] != nullptr) {
            // **A HIT GOES THROUGH THE PINNED BUFFER TOO.**  `cudaMemcpyAsync` from pageable memory is
            // synchronous: it would stall each worker for the length of the copy, which is exactly what the
            // double buffer and the events exist to avoid.
            const int b = stg.turn;
            stg.turn ^= 1;
            if (stg.done[b] != nullptr && !staged(cudaEventSynchronize(stg.done[b]))) return;
            // NO_PREAD used to fall straight through to a pageable hipMemcpyAsync under
            // api_mu, serializing workers and potentially waiting behind a slot event.
            // The same pinned pair used by RAM hits/pread also works for mmap sources.
            const void* host_src = j.ram_src != nullptr ? (const void*) j.ram_src : (const void*) j.src;
            if (j.fd == -2) {
                const auto& packed = *(const PackedExpert*) j.src;
                size_t off = 0;
                for (size_t p = 0; p < packed.pieces.size(); ++p) {
                    std::memcpy(stg.buf[b] + off, packed.pieces[p], packed.bytes[p]);
                    off += packed.bytes[p];
                }
            } else std::memcpy(stg.buf[b], host_src, j.bytes);
            if (j.ram_dst != nullptr) std::memcpy(j.ram_dst, stg.buf[b], j.bytes);
            const std::lock_guard<std::mutex> lk(api_mu);
            if (j.wait != nullptr && !staged(cudaStreamWaitEvent(work_stream, j.wait, 0))) return;
            staged(cudaMemcpyAsync(j.dst, stg.buf[b], j.bytes, cudaMemcpyHostToDevice, work_stream));
            if (stg.done[b] != nullptr) staged(cudaEventRecord(stg.done[b], work_stream));
            return;
        }
#if !defined(_WIN32)
        if (j.fd >= 0 && stg.buf[0] != nullptr) {
            const int b = stg.turn;
            stg.turn ^= 1;
            if (stg.done[b] != nullptr && !staged(cudaEventSynchronize(stg.done[b]))) return;
            const ssize_t got = ::pread(j.fd, stg.buf[b], j.bytes, (off_t) j.off);
            if (got == (ssize_t) j.bytes) {
                src = (const uint16_t*) stg.buf[b];
                // the host cache is filled as a side effect of the read that had to happen anyway; the copy is
                // synchronous, so it is complete before this buffer is read into again (the double buffer's
                // event only guards the DMA)
                if (j.ram_dst != nullptr) {
                    static std::atomic<int> fdbg{0};
                    if (std::getenv("STRATA_EXL3_RAM_DEBUG") != nullptr && fdbg.fetch_add(1) < 10)
                        std::fprintf(stderr, "ram fill: %zu bytes\n", j.bytes);
                    std::memcpy(j.ram_dst, stg.buf[b], j.bytes);
                }
                {
                    // **ONE ENQUEUE AT A TIME.**  The workers read in parallel, but two threads enqueueing
                    // `cudaMemcpyAsync` on the same (legacy default) stream raced: the same run produced
                    // different tokens with 4 workers and identical ones with 1.  The copies are asynchronous,
                    // so serialising only the call costs nothing.
                    const std::lock_guard<std::mutex> lk(api_mu);
                    if (j.wait != nullptr && !staged(cudaStreamWaitEvent(work_stream, j.wait, 0))) return;
                    staged(cudaMemcpyAsync(j.dst, src, j.bytes, cudaMemcpyHostToDevice, work_stream));
                    if (stg.done[b] != nullptr) staged(cudaEventRecord(stg.done[b], work_stream));
                }
                return;
            }
            std::fprintf(stderr, "strata generate: EXL3 expert read: pread %zu B at %zu: %zd\n", j.bytes, j.off, got);
        }
#endif
        {
            if (j.ram_dst) std::memcpy(j.ram_dst, src, j.bytes);
            const std::lock_guard<std::mutex> lk(api_mu);
            if (j.wait != nullptr && !staged(cudaStreamWaitEvent(work_stream, j.wait, 0))) return;
            staged(cudaMemcpyAsync(j.dst, src, j.bytes, cudaMemcpyHostToDevice, work_stream));
        }
    }

    // Queue one expert's nine pieces: the FILE location (for the pread), the mmap pointer (the fallback), and
    // the host cache: on a hit the pieces come out of RAM, on a miss the read fills the slot as a side effect.
    void queue_expert(int expert, const Slot* gs, const Slot* us, const Slot* ds,
                      const Exl3Linear& Lg, const Exl3Linear& Lu, const Exl3Linear& Ld) {
        const std::string b = "model.language_model.layers." + std::to_string(cur_layer) + ".mlp.experts." +
                              std::to_string(expert) + ".";
        const uint8_t* ram_src = nullptr;
        uint8_t* ram_dst = nullptr;
        const Slot* slots[] = {gs, us, ds};
        const Exl3Linear* linears[] = {&Lg, &Lu, &Ld};
        std::array<size_t, 9> sizes{};
        bool ram_layout = true;
        size_t plane_at = 0;
        for (int i = 0; i < 3; ++i) {
            const auto& L = *linears[i];
            sizes[(size_t) i * 3] = (size_t) L.ki * L.nj * 32 * L.bits;
            sizes[(size_t) i * 3 + 1] = (size_t) L.in() * 2;
            sizes[(size_t) i * 3 + 2] = (size_t) L.out() * 2;
            if (sizes[(size_t) i * 3] > slots[i]->trb || sizes[(size_t) i * 3 + 1] > slots[i]->suhb ||
                sizes[(size_t) i * 3 + 2] > slots[i]->svhb)
                throw std::runtime_error("exl3: expert exceeds allocated weight slot");
            for (int p = 0; p < 3; ++p) {
                const size_t piece = (size_t) i * 3 + p;
                if (ram_off[(size_t) cur_layer][piece] != plane_at) ram_layout = false;
                plane_at += sizes[piece];
            }
        }
        ram_layout = ram_layout && plane_at == ram_slot_bytes[(size_t) cur_layer];
        if (ram_slots > 0 && ram_layout) {
            const std::lock_guard<std::mutex> lk(ram_mu);
            RamLayer& rl = ram_layers[(size_t) cur_layer];
            const size_t base = ram_base[(size_t) cur_layer];
            const size_t slot_bytes = ram_slot_bytes[(size_t) cur_layer];
            auto it = rl.at.find(expert);
            int s = -1;
            if (it != rl.at.end()) {
                s = it->second;
                ram_src = ram.data() + base + (size_t) s * slot_bytes;
                ++ram_hits;
            } else {
                ++ram_misses;
                if (rl.filled < ram_slots) {
                    s = rl.filled++;
                } else {
                    for (int i = 0; i < ram_slots; ++i)
                        if (!ram_pinned[(size_t) i] && (s < 0 || rl.tick[(size_t) i] < rl.tick[(size_t) s])) s = i;
                    if (s >= 0) rl.at.erase(rl.key[(size_t) s]);
                }
                if (s >= 0) {
                    rl.at[expert] = s;
                    rl.key[(size_t) s] = expert;
                    ram_dst = ram.data() + base + (size_t) s * slot_bytes;
                }
            }
            if (s >= 0) { rl.tick[(size_t) s] = ++rl.clock; ram_pinned[(size_t) s] = 1; }
            static std::atomic<int> dbg{0};
            const int dn = dbg.fetch_add(1);
            if (std::getenv("STRATA_EXL3_RAM_DEBUG") != nullptr && dn % 500 == 0)
                std::fprintf(stderr, "ram: lookup %d L%d e%d %s slot %d map=%zu filled=%d\n", dn, cur_layer, expert,
                             ram_src != nullptr ? "HIT" : "miss", s, rl.at.size(), rl.filled);
        }
        const std::array<size_t, 9>& ro = ram_off[(size_t) cur_layer];
        const size_t job0 = jobs.size();
        std::string path; size_t off = 0, bytes = 0;
        int piece = 0;
        auto push = [&](uint16_t* dst, const std::string& base, const char* plane, const uint16_t* src, size_t n) {
            // STRATA_EXL3_NO_PREAD=0 keeps the pread path; default and 1 read through the mmap
            static const bool no_pread = strata::core::exl3_flag("STRATA_EXL3_NO_PREAD", nullptr, true);
            if (ram_src != nullptr) {   // a hit: no file read at all
                jobs.push_back({dst, src, -1, 0, n, nullptr, ram_src + ro[(size_t) piece]});
                ++piece;
                return;
            }
            const int fd = (!no_pread && model->locate(base + plane, path, off, bytes) && bytes == n) ? fd_of(path) : -1;
            jobs.push_back({dst, src, fd, off, n, ram_dst != nullptr ? ram_dst + ro[(size_t) piece] : nullptr});
            ++piece;
        };
        const std::string bg = b + "gate_proj", bu = b + "up_proj", bd = b + "down_proj";
        push(gs->tr, bg, ".trellis", Lg.trellis, sizes[0]);
        push(gs->suh, bg, ".suh", Lg.suh, sizes[1]);
        push(gs->svh, bg, ".svh", Lg.svh, sizes[2]);
        push(us->tr, bu, ".trellis", Lu.trellis, sizes[3]);
        push(us->suh, bu, ".suh", Lu.suh, sizes[4]);
        push(us->svh, bu, ".svh", Lu.svh, sizes[5]);
        push(ds->tr, bd, ".trellis", Ld.trellis, sizes[6]);
        push(ds->suh, bd, ".suh", Ld.suh, sizes[7]);
        push(ds->svh, bd, ".svh", Ld.svh, sizes[8]);
        // Arena-backed slots place the nine planes consecutively. Gather the mmap pieces
        // into one pinned slab, then issue ONE upload/event instead of nine small ones.
        // Other layouts/read modes retain the ordinary jobs, as does a pinned allocation failure.
        if (pack_stage && packed_count < packed_sources->size() &&
            ram_src == nullptr && ram_dst == nullptr && jobs.size() == job0 + 9 &&
            std::all_of(stage.begin(), stage.end(), [](const Stage& s) { return s.buf[0] && s.buf[1]; })) {
            Job packed = jobs[job0];
            PackedExpert sources;
            size_t off = 0;
            bool contiguous = true;
            for (size_t p = 0; p < 9; ++p) {
                const Job& j = jobs[job0 + p];
                contiguous = contiguous && j.fd < 0 &&
                    (uintptr_t) j.dst == (uintptr_t) packed.dst + off;
                sources.pieces[p] = j.src;
                sources.bytes[p] = j.bytes;
                off += j.bytes;
            }
            if (contiguous && off <= stage_bytes) {
                auto* stored = &(*packed_sources)[packed_count++];
                *stored = sources;
                packed.bytes = off;
                packed.fd = -2;
                packed.src = (const uint16_t*) stored;
                jobs.resize(job0);
                jobs.push_back(packed);
                if (profile[0] != nullptr) ++packed_jobs;
            }
        }
    }

    void prepare_expert(Blocks& blocks, int slot, int expert, bool slot_wait = true) {
        const std::string base = "model.language_model.layers." + std::to_string(cur_layer) + ".mlp.experts." +
                                  std::to_string(expert) + ".";
        const auto g = model->linear(base + "gate_proj");
        const auto u = model->linear(base + "up_proj");
        const auto d = model->linear(base + "down_proj");
        if (g.in() != n_embd || u.in() != n_embd || g.out() != u.out() ||
            d.in() != g.out() || d.out() != n_embd)
            throw std::runtime_error("exl3: incompatible expert FFN dimensions");
        const size_t job0 = jobs.size();
        queue_expert(expert, &blocks.g[(size_t) slot], &blocks.u[(size_t) slot], &blocks.d[(size_t) slot], g, u, d);
        if (pipe_stage && slot_wait) {
            // STRATA_EXL3_PIPE: this wave's upload may target a slot an earlier wave's queued kernels
            // are still reading - the staging stream waits the slot's last-reader event before the copy.
            const cudaEvent_t ev = blocks.used[(size_t) slot];
            for (size_t i = job0; i < jobs.size(); ++i) jobs[i].wait = ev;
        }
        auto set = [](Exl3Mat& m, const Exl3Linear& L) {
            m.ki = L.ki; m.nj = L.nj; m.bits = L.bits; m.cb = L.cb;
        };
        set(blocks.gm[(size_t) slot], g); set(blocks.um[(size_t) slot], u); set(blocks.dm[(size_t) slot], d);
    }

    int fd_of(const std::string& path) {
#if defined(_WIN32)
        (void) path;
        return -1;
#else
        auto it = fds.find(path);
        if (it != fds.end()) return it->second;
        const int fd = ::open(path.c_str(), O_RDONLY);
        fds[path] = fd;
        return fd;
#endif
    }

    void stage_parallel(bool host_decode = false) {
        if (prof && host_decode) {
            const size_t count = jobs.size();
            ++prof_stage_jobs[count <= 4 ? count : count <= 8 ? 5 : 6];
        }
        if (jobs.empty()) return;
        stage_error.store(0, std::memory_order_relaxed);
        const bool small_inline = host_decode && stage_inline > 0 && jobs.size() <= (size_t)stage_inline;
        if (workers.empty() || small_inline) {
            // Every preceding wave joined all participating workers. No new generation is
            // published here, so stage[0] is exclusively ours. Its existing
            // per-buffer DMA events still guard reuse and work_stream ordering.
            for (const Job& j : jobs) stage_one(j, stage[0]);
            if (small_inline) { ++inline_waves; inline_jobs += (long long)jobs.size(); }
        } else {
        const bool limited = host_decode && stage_wake_limit;
        const int target = limited ? std::min(n_workers, (int)jobs.size()) : n_workers;
        job_at.store(0, std::memory_order_relaxed);
        work_done.store(0, std::memory_order_relaxed);
        {
            const std::lock_guard<std::mutex> lk(work_mu);
            work_target = target;
            ++work_gen;
        }
        if (stage_wake_limit) {
            // A separate CV per worker prevents a fast participant from taking
            // a second notification meant for a still-sleeping participant.
            for (int i = 0; i < target; ++i) worker_cv[(size_t)i]->notify_one();
            if (limited && target < n_workers) { ++limited_waves; skipped_acks += n_workers - target; }
        } else work_cv.notify_all();
        while (work_done.load(std::memory_order_acquire) < (uint64_t) target) STRATA_EXL3_PAUSE();
        }
        const int error = stage_error.load(std::memory_order_relaxed);
        packed_count = 0;   // workers have consumed every source pointer; DMA reads pinned slabs
        if (error != 0) throw std::runtime_error(std::string("exl3: expert staging failed: ") +
                                                cudaGetErrorString((cudaError_t) error));
    }

    void worker_loop(int wid) {
        strata::aux_cpus::pin_current_thread();
        // CUDA's current device is thread-local; a stage on GPU1 cannot use a worker's default GPU0.
        if (cudaSetDevice(device) != cudaSuccess) {
            std::fprintf(stderr, "exl3: staging worker cannot select device %d\n", device);
            std::abort();
        }
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(work_mu);
                auto& cv = stage_wake_limit ? *worker_cv[(size_t)wid] : work_cv;
                cv.wait(lk, [&] { return work_quit || work_gen != seen; });
                if (work_quit) return;
                seen = work_gen;
                // A spurious/late wake can see a generation without its own
                // notification. Only its fixed participant prefix may read jobs.
                if (stage_wake_limit && wid >= work_target) continue;
            }
            for (;;) {
                const size_t i = job_at.fetch_add(1, std::memory_order_relaxed);
                if (i >= jobs.size()) break;
                stage_one(jobs[i], stage[(size_t) wid]);
            }
            work_done.fetch_add(1, std::memory_order_release);
        }
    }

    ~Impl() {
        if (inline_waves)
            std::fprintf(stderr, "exl3 inline stage: cap=%d, %lld host-decode waves / %lld jobs bypassed worker wake/join\n",
                         stage_inline, inline_waves, inline_jobs);
        if (limited_waves)
            std::fprintf(stderr, "exl3 stage wake limit: %lld waves, %lld idle-worker acknowledgements avoided\n",
                         limited_waves, skipped_acks);
        if (primary_packed_calls)
            std::fprintf(stderr, "exl3 primary packed plan: %lld passes / %lld routes, one plan H2D and direct FFN output per pass\n",
                         primary_packed_calls, primary_packed_routes);
        int previous = -1;
        (void) cudaGetDevice(&previous); (void) cudaSetDevice(device);
        {
            const std::lock_guard<std::mutex> lk(work_mu);
            work_quit = true;
        }
        work_cv.notify_all();
        for (auto& cv : worker_cv) cv->notify_all();
        for (std::thread& t : workers) if (t.joinable()) t.join();
        if (arena || pinned_plan) {
            // Also covers constructor unwinding after arena allocation. Pinned sources and arena
            // backing outlive all readers; do not rely on cudaFree's implicit synchronization.
            if (compute_stream) (void) cudaStreamSynchronize(compute_stream);
            if (work_stream) (void) cudaStreamSynchronize(work_stream);
        }
        for (Stage& st : stage) {
            for (uint8_t* p : st.buf) if (p != nullptr) (void) cudaFreeHost(p);
            for (cudaEvent_t e : st.done) if (e != nullptr) (void) cudaEventDestroy(e);
        }
#if !defined(_WIN32)
        for (auto& kv : fds) if (kv.second >= 0) (void) ::close(kv.second);
#endif
        for (void* p : { (void*) d_x, (void*) d_input, (void*) d_rows, (void*) d_out, (void*) d_hit, (void*) d_miss,
                         (void*) d_hit_rows, (void*) d_miss_rows }) (void) cudaFree(p);
        if (stage_done != nullptr) (void) cudaEventDestroy(stage_done);
        for (auto e : profile) if (e) (void) cudaEventDestroy(e);
        for (cudaEvent_t e : wave_done) if (e != nullptr) (void) cudaEventDestroy(e);
        for (cudaEvent_t e : pipe_ev) if (e != nullptr) (void) cudaEventDestroy(e);
        for (cudaEvent_t e : ov_ev) if (e != nullptr) (void) cudaEventDestroy(e);
        for (cudaEvent_t e : scratch.used) if (e != nullptr) (void) cudaEventDestroy(e);
        for (Layer& l : layers)
            for (cudaEvent_t e : l.used) if (e != nullptr) (void) cudaEventDestroy(e);
        if (work_stream != nullptr) (void) cudaStreamDestroy(work_stream);
        for (auto& v : { &scratch.g, &scratch.u, &scratch.d })
            for (Slot& s : *v) { (void) cudaFree(s.tr); (void) cudaFree(s.suh); (void) cudaFree(s.svh); }
        for (Layer& l : layers)
            for (void* p : l.allocs) (void) cudaFree(p);
        if (arena) (void) cudaFree(arena);
        workspace.reset();
        for (auto& p : host_plan) if (p.device) { (void)cudaFree(p.device); p.device = nullptr; }
        multi.reset();
        verify_graph.reset();
        if (compute_stream) (void) cudaStreamDestroy(compute_stream);
        if (previous >= 0) (void) cudaSetDevice(previous);
    }
};

Exl3ExpertStream::Exl3ExpertStream(const std::string& model_dir, int n_embd, int n_layers, int slots_per_layer,
                                   void* stream, size_t ram_cache_bytes, size_t reserve_device_bytes,
                                   int layer_lo, int layer_hi, int prefill_tokens) {
    if (layer_hi < 0) layer_hi = n_layers;
    if (n_embd <= 0 || n_layers <= 0 || n_layers > 48 || slots_per_layer < -1 || slots_per_layer > 512 ||
        layer_lo < 0 || layer_lo >= layer_hi || layer_hi > n_layers)
        throw std::invalid_argument("exl3: invalid expert stream geometry");
    auto pending = std::make_unique<Impl>();
    impl_ = pending.get();
    if (cudaGetDevice(&impl_->device) != cudaSuccess ||
        cudaStreamCreateWithFlags(&impl_->compute_stream, cudaStreamNonBlocking) != cudaSuccess)
        throw std::runtime_error("exl3: cannot create expert execution context");
    impl_->workspace = std::make_unique<Exl3MoeWorkspace>();
    impl_->multi = std::make_unique<MultiBuffers>();
    if (const char* timing = std::getenv("STRATA_EXL3_TIME"); timing && timing[0] && timing[0] != '0')
        for (auto& event : impl_->profile)
            if (cudaEventCreate(&event) != cudaSuccess) throw std::runtime_error("exl3: profiling event creation failed");
    if (const char* wmma = std::getenv("STRATA_EXL3_WMMA"); wmma && wmma[0] && wmma[0] != '0')
        impl_->wmma_group = true;   // opt-in (RDNA/WMMA group FFN, launcher-set): the reconstruct+BLAS
                                     // path stays the default without it
    // The RDNA-independent measured switches are ON by default (explicit "0" off) - see
    // strata/core/exl3_flags.hpp and notes/05-measured-numbers.md for each one's A/B.
    impl_->wave_sync = strata::core::exl3_flag("STRATA_EXL3_WAVESYNC", nullptr, true);
    impl_->pipe_stage = strata::core::exl3_flag("STRATA_EXL3_PIPE", nullptr, true);
    if (const char* v = std::getenv("STRATA_EXL3_EXPERT_BATCH"); v && v[0]) {
        impl_->expert_batch = v[0] != '0';   // opt-in (RDNA/WMMA batched group FFN, launcher-set)
    }
    impl_->overlap_stage = strata::core::exl3_flag("STRATA_EXL3_OVERLAP", nullptr, true);
    if (const char* v = std::getenv("STRATA_EXL3_SMALL_M"); v && v[0] && v[0] != '0')
        impl_->small_m = true;   // opt-in: grouped small-M FFN on the HOST-IO decode windows whose
                                  // shape measured a win (see Impl); device arms and defaults untouched
    if (const char* v = std::getenv("STRATA_EXL3_PRIMARY_MULTIROW"); v && std::atoi(v) > 0) {
        if (exl3_fused_multirow_supported()) {
            impl_->primary_multirow = std::max(2, std::min(4, std::atoi(v)));
            impl_->small_m = true;
        } else std::fprintf(stderr, "exl3 primary multirow: needs gfx11 RDNA=1/FUSE=3; using established dispatch\n");
    }
    if(const char* v=std::getenv("STRATA_EXL3_SKIP_EMPTY_STAGE");v && v[0] && v[0]!='0') impl_->skip_empty_stage=true;
    if(const char* v=std::getenv("STRATA_EXL3_PINNED_PLAN");v && v[0] && v[0]!='0') impl_->pinned_plan=true;
    if (strata::core::exl3_flag("STRATA_EXL3_PRIMARY_PACK_PLAN", nullptr, true)) {
        impl_->primary_pack = true;   // default on: data transport only (pinned plan slabs); the
        impl_->pinned_plan = true;    // RDNA fused kernels stay gated by PRIMARY_MULTIROW
    }
    {   // numeric parse kept: unset -> the measured default 2, "0" -> off, N -> clamp [0,8]
        const char* v = strata::core::exl3_env("STRATA_EXL3_STAGE_INLINE");
        impl_->stage_inline = v != nullptr ? std::max(0, std::min(8, std::atoi(v))) : 2;
    }
    if (const char* v = std::getenv("STRATA_EXL3_STAGE_WAKE_LIMIT"); v && v[0] && v[0] != '0')
        impl_->stage_wake_limit = true;
    if (const char* v = std::getenv("STRATA_EXL3_SMALL_M_TRACE"); v && v[0] && v[0] != '0')
        impl_->small_m_trace = true;
    if (const char* v = std::getenv("STRATA_EXL3_PROF"); v && v[0] && v[0] != '0')
        impl_->prof = true;   // opt-in window-service profile (host counters only; see Impl)
    if (const char* v = std::getenv("STRATA_EXL3_PINNED_MMAP"); v && v[0] && v[0] != '0')
        impl_->pinned_mmap = true;
    impl_->pack_stage = strata::core::exl3_flag("STRATA_EXL3_PACK_STAGE", nullptr, true);
    if (impl_->pack_stage) impl_->packed_sources = std::make_unique<std::array<Impl::PackedExpert, 512>>();
    if (impl_->overlap_stage && impl_->profile[0] != nullptr)   // the overlap-evidence events, TIME only
        for (auto& event : impl_->ov_ev)
            if (cudaEventCreate(&event) != cudaSuccess) throw std::runtime_error("exl3: profiling event creation failed");
    impl_->arena_requested = strata::core::exl3_flag("STRATA_EXL3_PREFILL_BORROW", nullptr, true);
    if (const char* v = std::getenv("STRATA_EXL3_CPU_TIER"); v && v[0] && v[0] != '0')
        impl_->cpu_tier = true;   // opt-in (B1): a caller's cpu_rows buffer switches the full-cache
                                  // policy from evict-and-admit to REFUSE (the GGUF pool's overflow)
    impl_->dir = model_dir;
    impl_->model = std::make_unique<Exl3Model>(model_dir);
    impl_->n_embd = n_embd;
    impl_->n_layers = n_layers;
    impl_->layer_lo = layer_lo; impl_->layer_hi = layer_hi;
    if(impl_->pinned_plan) {
        impl_->pinned_plan_capacity=8*impl_->max_k;
        const size_t count=impl_->pinned_plan_capacity;
        for(auto& plan:impl_->host_plan) {
            if(cudaHostAlloc(&plan.allocation,count*(3*sizeof(Exl3Mat)+4*sizeof(int)),cudaHostAllocPortable)!=cudaSuccess)
                throw std::runtime_error("exl3: pinned dispatch plan allocation failed");
            plan.gate=(Exl3Mat*)plan.allocation;plan.up=plan.gate+count;plan.down=plan.up+count;
            plan.first=(int*)(plan.down+count);plan.rows=plan.first+count;
            if (impl_->primary_pack && cudaMalloc((void**)&plan.device,
                    count*(3*sizeof(Exl3Mat)+4*sizeof(int))) != cudaSuccess)
                throw std::runtime_error("exl3: device dispatch plan allocation failed");
        }
    }
    // STRATA_EXL3_SMALL_M: the flag's ENTIRE device and host budget is pre-reserved HERE, at
    // setup, before any window or graph exists - a device allocation inside run_multi deadlocks
    // under the verify graph's host-flag spin (measured; docs/EXL3-HIP-BENCH.md).  The envelope
    // is the worst graph-coexisting window (M <= 8 x k <= max_k) at the widest layer FFN; calls
    // exceeding it simply stay on the indexed GEMV.
    if (impl_->small_m) {
        int ff = 0;
        for (int l = layer_lo; l < layer_hi; ++l)
            ff = std::max(ff, impl_->model->linear("model.language_model.layers." + std::to_string(l) +
                                                   ".mlp.experts.0.gate_proj").out());
        if (impl_->primary_multirow &&
            (size_t) std::max(n_embd, ff) * (impl_->primary_multirow <= 2 ? 2 : 4) * sizeof(float) > 40*1024) {
            std::fprintf(stderr, "exl3 primary multirow: geometry exceeds fused LDS bound; using indexed dispatch\n");
            impl_->primary_multirow = 0;
            const char* legacy = std::getenv("STRATA_EXL3_SMALL_M");
            impl_->small_m = legacy && legacy[0] && legacy[0] != '0';
        }
        impl_->small_m_envelope = 8 * impl_->max_k;
        impl_->workspace->reserve_small_m((size_t) impl_->small_m_envelope, (size_t) impl_->small_m_envelope,
                                           n_embd, ff, n_embd);
        if (impl_->primary_multirow)
            impl_->workspace->reserve_runtime_scratch(impl_->small_m_envelope, n_embd, ff, n_embd);
        for (int p = 0; p < 2; ++p) {   // the flag-owned persistent plan sources: allocated ONCE
            impl_->sm_gfirst[p].reserve((size_t) impl_->small_m_envelope);
            impl_->sm_grows[p].reserve((size_t) impl_->small_m_envelope);
            impl_->sm_gg[p].reserve((size_t) impl_->small_m_envelope);
            impl_->sm_gu[p].reserve((size_t) impl_->small_m_envelope);
            impl_->sm_gd[p].reserve((size_t) impl_->small_m_envelope);
        }
    }
    // **THE STAGING COPIES GET THEIR OWN STREAM.**  They used to share the caller's, which put them behind
    // the resident experts' FFN: the double-buffered workers then blocked on their own DMA events for the
    // length of that FFN (measured: the staging went from 24 to 37 ms/token).  With a stream of their own
    // the copies start at once, and the miss FFN waits on one event instead.
    if (cudaStreamCreateWithFlags(&impl_->work_stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&impl_->stage_done, cudaEventDisableTiming) != cudaSuccess) {
        std::fprintf(stderr, "strata generate: EXL3 expert stream: no staging stream\n");
        std::abort();
    }
    if (impl_->pipe_stage) for (cudaEvent_t& e : impl_->wave_done)   // the host bank's last enqueue
        if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: EXL3 expert stream: no wave event\n");
            std::abort();
        }
    (void) stream;
    // The staging workers: enough to saturate the host's memory bandwidth on the miss reads, not more.
    const char* nw = std::getenv("STRATA_EXL3_WORKERS");
    impl_->n_workers = nw != nullptr ? std::atoi(nw) : 4;
    if (impl_->n_workers < 0 || impl_->n_workers > 32)
        throw std::invalid_argument("exl3: STRATA_EXL3_WORKERS must be in [0,32]");
    // the largest piece any layer's expert has (the bitrate can differ per layer), and one expert's nine
    // pieces per layer for the host cache's slot layout
    impl_->ram_slot_bytes.assign((size_t) n_layers, 0);
    impl_->ram_off.assign((size_t) n_layers, {});
    for (int l = layer_lo; l < layer_hi; ++l) {
        const std::string b = "model.language_model.layers." + std::to_string(l) + ".mlp.experts.0.";
        size_t off = 0;
        int piece = 0;
        for (const char* p : {"gate_proj", "up_proj", "down_proj"}) {
            Exl3Linear L = impl_->model->linear(b + p);
            const size_t trb = (size_t) L.ki * L.nj * (256 * L.bits / 16) * 2;
            impl_->stage_bytes = std::max(impl_->stage_bytes, std::max(trb, (size_t) (L.in() + L.out()) * 2));
            for (const size_t n : {trb, (size_t) L.in() * 2, (size_t) L.out() * 2}) {
                impl_->ram_off[(size_t) l][(size_t) piece++] = off;
                off += n;
            }
        }
        impl_->ram_slot_bytes[(size_t) l] = off;
        if (impl_->pack_stage) impl_->stage_bytes = std::max(impl_->stage_bytes, off);
    }
    if (ram_cache_bytes > 0) {
        size_t per_layer = 0;
        for (size_t n : impl_->ram_slot_bytes) per_layer += n;
        if (per_layer > 0) impl_->ram_slots = (int) std::min<size_t>(ram_cache_bytes / per_layer, 1u << 20);
        if (impl_->ram_slots > 0) {
            impl_->ram_base.assign((size_t) n_layers, 0);
            size_t at = 0;
            for (int l = layer_lo; l < layer_hi; ++l) {
                impl_->ram_base[(size_t) l] = at;
                at += (size_t) impl_->ram_slots * impl_->ram_slot_bytes[(size_t) l];
            }
            try { impl_->ram.assign(at, 0); } catch (const std::exception&) { impl_->ram_slots = 0; }
            if (impl_->ram_slots > 0) {
                impl_->ram_layers.resize((size_t) n_layers);
                for (Impl::RamLayer& rl : impl_->ram_layers) {
                    rl.tick.assign((size_t) impl_->ram_slots, 0);
                    rl.key.assign((size_t) impl_->ram_slots, -1);
                }
            }
        }
    }
    impl_->stage.resize((size_t) (impl_->n_workers > 0 ? impl_->n_workers : 1));
    for (Impl::Stage& stg : impl_->stage) {
        for (int b = 0; b < 2; ++b) {
            if (cudaMallocHost((void**) &stg.buf[b], impl_->stage_bytes) != cudaSuccess) stg.buf[b] = nullptr;
            else if (cudaEventCreateWithFlags(&stg.done[b], cudaEventDisableTiming) != cudaSuccess) stg.done[b] = nullptr;
        }
        if (!stg.buf[0] || !stg.buf[1] || !stg.done[0] || !stg.done[1]) {
            // Without an event the DMA may still read a buffer when its next pread overwrites it.
            // Fall back as one pair, rather than running half of a double-buffered stage.
            for (int b = 0; b < 2; ++b) {
                if (stg.buf[b]) (void) cudaFreeHost(stg.buf[b]);
                if (stg.done[b]) (void) cudaEventDestroy(stg.done[b]);
                stg.buf[b] = nullptr; stg.done[b] = nullptr;
            }
        }
    }
    if (impl_->n_workers > 0) {
        impl_->workers.reserve((size_t) impl_->n_workers);
        if (impl_->stage_wake_limit)
            for (int i = 0; i < impl_->n_workers; ++i)
                impl_->worker_cv.push_back(std::make_unique<std::condition_variable>());
        for (int i = 0; i < impl_->n_workers; ++i) impl_->workers.emplace_back([im = impl_, i] { im->worker_loop(i); });
    }
    const int mk = impl_->max_k;
    auto alloc = [](uint16_t** p, size_t bytes) { if (cudaMalloc((void**) p, bytes ? bytes : 1) != cudaSuccess) std::abort(); };
    alloc(&impl_->d_x, (size_t) n_embd * 2);
    if (cudaMalloc((void**) &impl_->d_input, (size_t) n_embd * sizeof(float)) != cudaSuccess)
        throw std::runtime_error("exl3: input allocation failed");
    alloc(&impl_->d_rows, (size_t) mk * n_embd * 2);
    if (cudaMalloc((void**) &impl_->d_out, (size_t) mk * n_embd * 4) != cudaSuccess) std::abort();
    if (cudaMalloc((void**) &impl_->d_hit, (size_t) mk * n_embd * 2) != cudaSuccess ||
        cudaMalloc((void**) &impl_->d_miss, (size_t) mk * n_embd * 2) != cudaSuccess ||
        cudaMalloc((void**) &impl_->d_hit_rows, (size_t) mk * sizeof(int)) != cudaSuccess ||
        cudaMalloc((void**) &impl_->d_miss_rows, (size_t) mk * sizeof(int)) != cudaSuccess) std::abort();
    // size the per-slot weight buffers from expert 0 of layer 0 (all experts in a layer share a shape)
    const std::string b0 = "model.language_model.layers." + std::to_string(layer_lo) + ".mlp.experts.0.";
    auto setup_scratch = [&](std::vector<Impl::Slot>& slots, std::vector<Exl3Mat>& mats, const char* proj) {
        Exl3Linear L = impl_->model->linear(b0 + proj);
        size_t trb = 0, suhb = 0, svhb = 0;
        for (int l = layer_lo; l < layer_hi; ++l) {
            const auto S = impl_->model->linear("model.language_model.layers." + std::to_string(l) +
                                                ".mlp.experts.0." + proj);
            trb = std::max(trb, (size_t) S.ki * S.nj * 32 * S.bits);
            suhb = std::max(suhb, (size_t) S.in() * 2);
            svhb = std::max(svhb, (size_t) S.out() * 2);
        }
        slots.resize((size_t) mk); mats.resize((size_t) mk);
        for (int i = 0; i < mk; ++i) {
            alloc(&slots[(size_t) i].tr, trb);
            alloc(&slots[(size_t) i].suh, suhb);
            alloc(&slots[(size_t) i].svh, svhb);
            slots[(size_t) i].trb = trb; slots[(size_t) i].suhb = suhb; slots[(size_t) i].svhb = svhb;
            mats[(size_t) i].trellis = slots[(size_t) i].tr;
            mats[(size_t) i].suh = slots[(size_t) i].suh;
            mats[(size_t) i].svh = slots[(size_t) i].svh;
            mats[(size_t) i].ki = L.ki; mats[(size_t) i].nj = L.nj; mats[(size_t) i].bits = L.bits; mats[(size_t) i].cb = L.cb;
        }
    };
    setup_scratch(impl_->scratch.g, impl_->scratch.gm, "gate_proj");
    setup_scratch(impl_->scratch.u, impl_->scratch.um, "up_proj");
    setup_scratch(impl_->scratch.d, impl_->scratch.dm, "down_proj");
    // STRATA_EXL3_PIPE: the slots' last-reader events (the native prefill ring's `used`).  Created
    // only when the arm is on: the default path allocates nothing new and stays byte-identical.
    auto make_used = [](Impl::Blocks& blocks, size_t n) {
        blocks.used.assign(n, nullptr);
        for (cudaEvent_t& e : blocks.used)
            if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
                std::fprintf(stderr, "strata generate: EXL3 expert stream: no slot event\n");
                std::abort();
            }
    };
    if (impl_->pipe_stage) make_used(impl_->scratch, (size_t) mk);
    // Verify graphs wait for this pool. Allocating on that callback would synchronise the device
    // behind the waiting graph, so reserve the whole 8-token, max-k window before any launch.
    int max_ff = 0;
    for (int l = layer_lo; l < layer_hi; ++l)
        max_ff = std::max(max_ff, impl_->model->linear("model.language_model.layers." + std::to_string(l) +
                                                      ".mlp.experts.0.gate_proj").out());
    impl_->workspace->reserve(8 * mk, n_embd, max_ff, n_embd, true);
    const auto enabled = [](const char* name) {
        const char* v = std::getenv(name);
        return v && v[0] && v[0] != '0';
    };
    if (strata::core::exl3_flag("STRATA_EXL3_PEER_SPARSE", nullptr, true) ||
        enabled("STRATA_EXL3_RESERVE_FUSED") || impl_->primary_pack)
        impl_->workspace->reserve_runtime_scratch(8 * mk, n_embd, max_ff, n_embd);
    impl_->multi->reserve(8, 8 * mk, n_embd);
    if (prefill_tokens > 0) {
        // Price EXL3 prompt scratch before auto-sizing the cache; device inputs need no FP32 copy.
        impl_->multi->reserve(std::max(8, prefill_tokens), 4096, n_embd, false);
        impl_->workspace->reserve(4096, n_embd, max_ff, n_embd);
        if (impl_->expert_batch) // charge subgroup scratch before automatic cache sizing
            impl_->workspace->reserve_expert_batch(8, 4096, n_embd, max_ff, n_embd);
    }
    (void) stream;

    // ---- the resident cache.  `-1` sizes it from the free VRAM: one slot in every layer costs
    // `per_slot` bytes, and what is left after this instance's own buffers is the budget.  The GGUF
    // path's `--expert-cache` is a global count because its blobs are one pool; here the stream is per
    // layer, so the count is per layer.
    int slots = slots_per_layer;
    {
        size_t per_slot = 0;
        for (int l = layer_lo; l < layer_hi; ++l) {
            const std::string b = "model.language_model.layers." + std::to_string(l) + ".mlp.experts.0.";
            for (const char* p : {"gate_proj", "up_proj", "down_proj"}) {
                Exl3Linear L = impl_->model->linear(b + p);
                per_slot += (size_t) L.ki * L.nj * (256 * L.bits / 16) * 2 + (size_t) (L.in() + L.out()) * 2;
            }
        }
        if (slots < 0) {
            size_t free_b = 0, total_b = 0;
            (void) cudaMemGetInfo(&free_b, &total_b);
            const size_t reserve = reserve_device_bytes;
            slots = (per_slot > 0 && free_b > reserve) ? (int) std::min<size_t>((free_b - reserve) / per_slot, 512) : 0;
            std::fprintf(stderr, "strata generate: EXL3 expert cache auto: %.2f GiB free -> %d slots/layer\n",
                         (double) free_b / 1073741824.0, slots);
        }
        impl_->cache = slots > 0 ? slots : 0;
        if (impl_->pipe_stage && impl_->cache > 0) {
            // STRATA_EXL3_PIPE: create the slots' last-reader events now and, when the cache was
            // auto-sized, give their bytes back from the slot count (a re-probe of the free VRAM):
            // the pipelined arm runs under the SAME total VRAM budget as the default one, not on
            // top of it.  With an explicit --expert-cache the count is the operator's call and the
            // event bytes (~1 MiB total) are simply reported by the A/B's vram_free_mib.
            impl_->layers.resize((size_t) n_layers);
            size_t free_before = 0, free_after = 0, total_b = 0;
            (void) cudaMemGetInfo(&free_before, &total_b);
            for (int l = layer_lo; l < layer_hi; ++l) make_used(impl_->layers[(size_t) l], (size_t) impl_->cache);
            (void) cudaMemGetInfo(&free_after, &total_b);
            if (slots_per_layer < 0 && per_slot > 0 && free_before > free_after &&
                free_after > reserve_device_bytes) {
                const int fit = (int) std::min<size_t>((free_after - reserve_device_bytes) / per_slot,
                                                       (size_t) impl_->cache);
                if (fit < impl_->cache) {
                    for (int l = layer_lo; l < layer_hi; ++l) {
                        std::vector<cudaEvent_t>& used = impl_->layers[(size_t) l].used;
                        for (size_t s = (size_t) fit; s < used.size(); ++s) (void) cudaEventDestroy(used[s]);
                        used.resize((size_t) fit);
                    }
                    std::fprintf(stderr, "strata generate: EXL3 expert slot events cost %.1f MiB -> %d slots/layer\n",
                                 (double) (free_before - free_after) / 1048576.0, fit);
                    impl_->cache = fit;
                }
            }
        }
    }
    if (impl_->cache > 0) {
        std::unique_ptr<Exl3ArenaLayout> arena_layout;
        if (impl_->arena_requested) {
            std::vector<Exl3ArenaLayout::Planes> sizes;
            for (int l = layer_lo; l < layer_hi; ++l) {
                Exl3ArenaLayout::Planes planes{};
                int p = 0;
                for (const char* proj : {"gate_proj", "up_proj", "down_proj"}) {
                    const auto m = impl_->model->linear("model.language_model.layers." + std::to_string(l) +
                                                       ".mlp.experts.0." + proj);
                    if (m.ki <= 0 || m.nj <= 0 || m.ki > INT32_MAX / 16 || m.nj > INT32_MAX / 16 ||
                        m.bits < 1 || m.bits > 8 || m.ki % 8 || m.nj % 8)
                        throw std::invalid_argument("exl3: invalid arena matrix geometry");
                    const size_t tiles = (size_t) m.ki * (size_t) m.nj;
                    if (tiles > SIZE_MAX / (32u * (unsigned) m.bits))
                        throw std::overflow_error("exl3: arena plane size overflow");
                    planes[(size_t) p++] = tiles * 32u * (unsigned) m.bits;
                    planes[(size_t) p++] = (size_t) m.in() * 2;
                    planes[(size_t) p++] = (size_t) m.out() * 2;
                }
                const auto gate = impl_->model->linear("model.language_model.layers." + std::to_string(l) +
                                                       ".mlp.experts.0.gate_proj");
                const auto up = impl_->model->linear("model.language_model.layers." + std::to_string(l) +
                                                     ".mlp.experts.0.up_proj");
                const auto down = impl_->model->linear("model.language_model.layers." + std::to_string(l) +
                                                       ".mlp.experts.0.down_proj");
                if (gate.in() != n_embd || up.in() != n_embd || gate.out() != up.out() ||
                    down.in() != gate.out() || down.out() != n_embd)
                    throw std::invalid_argument("exl3: incompatible arena FFN geometry");
                sizes.push_back(planes);
            }
            arena_layout = std::make_unique<Exl3ArenaLayout>(sizes, impl_->cache);
            // Validated ki/nj multiples of8 make scales32*ki/32*nj and trellis32*bits*ki*nj
            // exact multiples of256. Therefore rank alignment adds ZERO bytes, including mixed K
            // and hidden widths: the existing raw per-rank auto/cache-event pricing is exact.
            size_t raw_rank = 0;
            for (const auto& layer : sizes)
                for (size_t bytes : layer) {
                    if (bytes % 256) throw std::logic_error("exl3: arena pricing alignment invariant");
                    raw_rank = Exl3ArenaLayout::add(raw_rank, bytes);
                }
            if (raw_rank != arena_layout->rank_bytes)
                throw std::logic_error("exl3: arena rank differs from auto-cache price");
            const cudaError_t status = cudaMalloc((void**) &impl_->arena, arena_layout->bytes);
            if (status != cudaSuccess) {
                impl_->arena = nullptr;
                (void) cudaGetLastError();
                std::fprintf(stderr, "exl3: contiguous arena allocation failed (%s); using original plane backing, no borrowing\n",
                             cudaGetErrorString(status));
                arena_layout.reset();
            } else {
                impl_->arena_bytes = arena_layout->bytes;
                impl_->arena_rank_bytes = arena_layout->rank_bytes;
                std::fprintf(stderr, "exl3: experimental arena-only backing: %zu bytes, rank %zu, %d slots/layer; lending NOT enabled\n",
                             impl_->arena_bytes, impl_->arena_rank_bytes, impl_->cache);
            }
        }
        impl_->layers.resize((size_t) n_layers);
        for (int l = layer_lo; l < layer_hi; ++l) {
            Impl::Layer& L = impl_->layers[(size_t) l];
            L.tick.assign((size_t) impl_->cache, 0);
            const std::string b = "model.language_model.layers." + std::to_string(l) + ".mlp.experts.0.";
            std::vector<Impl::Slot>* vs[3] = { &L.g, &L.u, &L.d };
            std::vector<Exl3Mat>* ms[3] = { &L.gm, &L.um, &L.dm };
            const char* projs[3] = { "gate_proj", "up_proj", "down_proj" };
            for (int i = 0; i < 3; ++i) {
                Exl3Linear S = impl_->model->linear(b + projs[i]);
                const size_t trb = (size_t) S.ki * S.nj * (256 * S.bits / 16) * 2;
                const size_t suhb = (size_t) S.in() * 2, svhb = (size_t) S.out() * 2;
                uint16_t *tr = nullptr, *suh = nullptr, *svh = nullptr;
                if (!arena_layout) {
                    alloc(&tr, trb * (size_t) impl_->cache);
                    alloc(&suh, suhb * (size_t) impl_->cache);
                    alloc(&svh, svhb * (size_t) impl_->cache);
                    L.allocs.push_back(tr); L.allocs.push_back(suh); L.allocs.push_back(svh);
                }
                vs[i]->resize((size_t) impl_->cache);
                ms[i]->resize((size_t) impl_->cache);
                for (int s = 0; s < impl_->cache; ++s) {
                    Impl::Slot& sl = (*vs[i])[(size_t) s];
                    if (arena_layout) {
                        const auto& off = arena_layout->offsets[(size_t) (l - layer_lo)];
                        uint8_t* rank = impl_->arena + (size_t) s * arena_layout->rank_bytes;
                        sl.tr = (uint16_t*) (rank + off[(size_t) i * 3]);
                        sl.suh = (uint16_t*) (rank + off[(size_t) i * 3 + 1]);
                        sl.svh = (uint16_t*) (rank + off[(size_t) i * 3 + 2]);
                    } else {
                        sl.tr = tr + (size_t) s * (trb / 2);
                        sl.suh = suh + (size_t) s * (suhb / 2);
                        sl.svh = svh + (size_t) s * (svhb / 2);
                    }
                    sl.trb = trb; sl.suhb = suhb; sl.svhb = svhb;
                    Exl3Mat& M = (*ms[i])[(size_t) s];
                    M.trellis = sl.tr; M.suh = sl.suh; M.svh = sl.svh;
                    M.ki = S.ki; M.nj = S.nj; M.bits = S.bits; M.cb = S.cb;
                }
            }
        }
    }
    impl_->active = impl_->cache;      // the whole cache is routable until a loan opens
    (void) pending.release();
}

Exl3ExpertStream::~Exl3ExpertStream() { delete impl_; }

int Exl3ExpertStream::slots_per_layer() const { return impl_->cache; }
int Exl3ExpertStream::ram_slots_per_layer() const { return impl_->ram_slots; }
size_t Exl3ExpertStream::ram_bytes() const { return impl_->ram.size(); }

// ================================ the stream loan ================================
// Contract in the header.  Both entry points hold `run_mu` (a loan never opens, grows or closes
// while a run is in flight) and drain the compute and staging streams first; a FAILED drain
// rejects the request with no state change - a failed sync means the context is unusable, and no
// loan is opened on top of it (nor any recovery claimed for it).

bool Exl3ExpertStream::loan_available() const { return impl_->arena != nullptr && impl_->cache > 0; }
size_t Exl3ExpertStream::loan_rank_bytes() const { return loan_available() ? impl_->arena_rank_bytes : 0; }
int Exl3ExpertStream::loan_max_ranks() const { return loan_available() ? impl_->cache - 1 : 0; }
bool Exl3ExpertStream::loan_active() const { return impl_->loan.ranks > 0; }
int Exl3ExpertStream::loan_active_ranks() const { return impl_->active; }
uint64_t Exl3ExpertStream::loan_epoch() const { return impl_->loan.epoch; }

bool Exl3ExpertStream::loan_dense_invariant() const {
    if (impl_->cache <= 0) return true;
    for (int l = impl_->layer_lo; l < impl_->layer_hi; ++l) {
        const Impl::Layer& L = impl_->layers[(size_t) l];
        if (L.filled < 0 || L.filled > impl_->active || L.at.size() != (size_t) L.filled) return false;
        std::vector<uint8_t> seen((size_t) L.filled, 0);
        for (const auto& kv : L.at) {
            if (kv.second < 0 || kv.second >= L.filled || seen[(size_t) kv.second]) return false;
            seen[(size_t) kv.second] = 1;
        }
    }
    return true;
}

// Validation seams (header): the ACTIVE loan's saved experts, and the LAST restore's placements,
// per owned layer, unique and ascending.  Call between run calls.
void Exl3ExpertStream::loan_debug_saved(int layer, std::vector<int>& experts) const {
    experts.clear();
    if (impl_->loan.ranks == 0 || layer < impl_->layer_lo || layer >= impl_->layer_hi) return;
    for (const Impl::TailRecord& r : impl_->loan.saved)
        if (r.layer == layer && std::find(experts.begin(), experts.end(), r.expert) == experts.end())
            experts.push_back(r.expert);
    std::sort(experts.begin(), experts.end());
}

void Exl3ExpertStream::loan_debug_restored(int layer, std::vector<int>& experts) const {
    experts.clear();
    if (impl_->loan.restored.empty() || layer < impl_->layer_lo || layer >= impl_->layer_hi) return;
    experts = impl_->loan.restored[(size_t) (layer - impl_->layer_lo)];
    std::sort(experts.begin(), experts.end());
}

void Exl3ExpertStream::overlap_debug(long long* splits, long long* fallbacks) const {
    if (splits) *splits = impl_->ov_splits;
    if (fallbacks) *fallbacks = impl_->ov_fallbacks;
}

void Exl3ExpertStream::small_m_debug(long long* calls, long long* routes, long long* stays,
                                     size_t* workspace_bytes) const {
    // Lifetime counters unless STRATA_EXL3_TIME is printing per-window summaries (its resets make
    // deltas, not absolutes, the stable read).  Test evidence for the opt-in's real dispatch
    // coverage and the pre-reserve contract, not a control.
    if (calls) *calls = impl_->small_m_calls;
    if (routes) *routes = impl_->small_m_routes;
    if (stays) std::copy(std::begin(impl_->small_m_stays), std::end(impl_->small_m_stays), stays);
    if (workspace_bytes) *workspace_bytes = impl_->workspace->allocated_bytes();
}

void Exl3ExpertStream::set_owned_elsewhere(const uint8_t* table, int64_t n_layers, int64_t n_expert) {
    // SETUP-ONLY and TRANSACTIONAL: call once at startup before any request (run paths read the
    // table lock-free afterwards; the table must outlive the stream - it is never read during
    // destruction).  On any failure nothing changes; on success any already-resident owned
    // expert is ERASED (run() consults residency before the exclusion, so a retained entry
    // would compute from cache and duplicate the owner's weights).
    const bool on = table != nullptr && n_layers > 0 && n_expert > 0;
    if (on && impl_->loan.ranks > 0)
        throw std::runtime_error("exl3: ownership exclusion cannot be enabled during an active loan");
    if (on) {
        const int want = std::max(impl_->max_k, impl_->active);   // scratch covers one full wave
        int prev = -1;
        (void) cudaGetDevice(&prev);
        const bool switched = cudaSetDevice(impl_->device) == cudaSuccess;
        const bool grown = switched && impl_->grow_scratch(want, nullptr);
        (void) cudaSetDevice(prev);
        if (!grown) throw std::runtime_error("exl3: ownership exclusion scratch growth failed");
    }
    // publish the table last, then drop any resident owned entries
    impl_->own_table = on ? table : nullptr;
    impl_->own_layers = n_layers;
    impl_->own_expert = n_expert;
    impl_->own_excl.store(0, std::memory_order_relaxed);
    if (on) {
        // A retained owned entry would compute from cache (run() checks residency first) and
        // duplicate the owner's weights.  Erasing one entry alone would leave the dense
        // [0, filled) slot layout with a hole (and confuse the loan machinery), so ANY owned
        // resident entry INVALIDATES its whole layer: map cleared, filled = 0.
        const std::lock_guard<std::mutex> lk(impl_->api_mu);
        for (int64_t l = 0; l < (int64_t) impl_->layers.size(); ++l) {
            auto& lay = impl_->layers[(size_t) l];
            bool touched = false;
            for (const auto& kv : lay.at)
                if (impl_->owned_elsewhere(l, kv.first)) { touched = true; break; }
            if (touched) {
                lay.at.clear();
                lay.filled = 0;
                ++impl_->own_excl;
            }
        }
    }
}

bool Exl3ExpertStream::holds(int64_t layer, int32_t expert) const {
    return layer >= 0 && layer < (int64_t) impl_->layers.size() &&
           impl_->layers[(size_t) layer].at.count(expert) != 0;
}

// `--expert-profile` on the EXL3 cache: the GGUF R4.2e profile fill (generate.cpp) with the same
// file - each owned layer takes its hottest ranked pairs into its free slots, dense [0, filled)
// order, through the ordinary staging path.  The initial state only: the LRU keeps adapting (the
// EXL3 A/B #6 froze the cache and measured it badly hurt).  One filled slot is read back and
// compared against the source before returning - a residency right about indices and wrong about
// bytes produces a plausible token, which is this project's most expensive failure mode.
bool Exl3ExpertStream::warmup_fill(const std::vector<std::pair<int32_t, int32_t>>& ranked,
                                   int64_t* n_filled, int64_t* n_bytes, std::string& err) {
    if (n_filled) *n_filled = 0;
    if (n_bytes) *n_bytes = 0;
    if (impl_->cache <= 0) { err = "exl3: the warmup fill needs a slot cache"; return false; }
    if (impl_->loan.ranks > 0) { err = "exl3: the warmup fill runs only with no loan active"; return false; }
    if (ranked.empty()) return true;
    const std::lock_guard<std::mutex> lock(impl_->run_mu);
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(impl_->device) != cudaSuccess) {
        err = "exl3: the warmup fill cannot select the stream device";
        return false;
    }
    struct RestoreDevice {
        int previous;
        ~RestoreDevice() { (void) cudaSetDevice(previous); }
    } restore{previous};
    // Per-layer buckets keep the ranking's order: each layer takes its hottest pairs first.
    std::vector<std::vector<int>> bucket(impl_->layers.size());
    for (const auto& pr : ranked) {
        const int32_t l = pr.first, e = pr.second;
        if (l < 0 || l >= (int32_t) impl_->layers.size() || e < 0 ||
            l < impl_->layer_lo || l >= impl_->layer_hi) continue;
        bucket[(size_t) l].push_back(e);
    }
    int64_t filled = 0, bytes = 0;
    std::pair<int32_t, int32_t> first{-1, -1};
    int first_slot = -1;
    try {
        for (int l = impl_->layer_lo; l < impl_->layer_hi; ++l) {
            Impl::Layer& lay = impl_->layers[(size_t) l];
            impl_->cur_layer = l;
            const int active = impl_->active;
            for (const int e : bucket[(size_t) l]) {
                if (lay.filled >= active) break;
                if (lay.at.count(e) != 0) continue;                  // a duplicate rank or a resident
                if (impl_->admission_excluded(l, e)) continue;       // another tier's pair
                const int s = lay.filled++;                          // dense fill, like a miss
                impl_->prepare_expert(lay, s, e, false);             // startup: no queued readers
                if (filled == 0 && std::getenv("STRATA_EXL3_FILL_DEBUG") != nullptr) {
                    std::fprintf(stderr, "fill debug: first expert %d jobs=%zu last_fd=%d packed_count=%zu "
                                 "stage_bytes=%zu\n", e, impl_->jobs.size(),
                                 impl_->jobs.empty() ? -99 : impl_->jobs.back().fd, impl_->packed_count,
                                 impl_->stage_bytes);
                    if (!impl_->jobs.empty() && impl_->jobs.back().fd == -2) {
                        const auto& pk = *(const Impl::PackedExpert*) impl_->jobs.back().src;
                        for (size_t p = 0; p < 9; ++p)
                            std::fprintf(stderr, "fill debug: piece %zu bytes=%zu src[0..3]=%02x %02x %02x %02x\n",
                                         p, pk.bytes[p], ((const uint8_t*) pk.pieces[p])[0],
                                         ((const uint8_t*) pk.pieces[p])[1], ((const uint8_t*) pk.pieces[p])[2],
                                         ((const uint8_t*) pk.pieces[p])[3]);
                    }
                }
                lay.at[e] = s;
                lay.tick[(size_t) s] = ++lay.clock;
                if (first.first < 0) { first = {l, e}; first_slot = s; }
                // nine planes per expert: trellis ki*nj*32*bits, suh ki*32, svh nj*32 bytes
                for (const Exl3Mat* m : {&lay.gm[(size_t) s], &lay.um[(size_t) s], &lay.dm[(size_t) s]})
                    bytes += (int64_t) m->ki * m->nj * 32 * m->bits + ((int64_t) m->ki + m->nj) * 32;
                ++filled;
            }
            if (!impl_->jobs.empty()) impl_->stage_parallel();       // this layer's planes land
            impl_->jobs.clear();   // run paths clear at entry; a multi-drain caller MUST clear here -
                                   // re-running a drained packed job re-reads its reused descriptor
            if (l == impl_->layer_lo && std::getenv("STRATA_EXL3_FILL_DEBUG") != nullptr) {
                if (impl_->work_stream) (void) cudaStreamSynchronize(impl_->work_stream);
                uint8_t probe[4] = {0, 0, 0, 0};
                (void) cudaMemcpy(probe, impl_->layers[(size_t) l].g[0].tr, 4, cudaMemcpyDeviceToHost);
                std::fprintf(stderr, "fill debug: after L%lld drain, slot0.tr[0..3]=%02x %02x %02x %02x\n",
                             (long long) l, probe[0], probe[1], probe[2], probe[3]);
            }
        }
    } catch (const std::exception& e) {
        err = std::string("exl3: the warmup fill failed: ") + e.what();
        return false;
    }
    // stage_parallel waits for the ENQUEUES, not the DMAs, and work_stream is non-blocking: sync
    // it here or the verify - and a first request's hit-path FFN - could read an unfinished upload.
    if (impl_->work_stream != nullptr && cudaStreamSynchronize(impl_->work_stream) != cudaSuccess) {
        err = "exl3: the warmup fill's uploads did not complete";
        return false;
    }
    // ---- one slot verified: every plane read back and compared against its source bytes
    if (first.first >= 0) {
        const int l = first.first, e = first.second;
        const std::string base = "model.language_model.layers." + std::to_string(l) + ".mlp.experts." +
                                 std::to_string(e) + ".";
        const Impl::Layer& lay = impl_->layers[(size_t) l];
        const Impl::Slot* dmm[3] = {&lay.g[(size_t) first_slot], &lay.u[(size_t) first_slot], &lay.d[(size_t) first_slot]};
        const char* proj[3] = {"gate_proj", "up_proj", "down_proj"};
        std::vector<uint8_t> tmp;
        for (int p = 0; p < 3; ++p) {
            const strata::Exl3Linear L = impl_->model->linear(base + proj[p]);
            const size_t sz[3] = {(size_t) L.ki * L.nj * 32 * L.bits, (size_t) L.in() * 2, (size_t) L.out() * 2};
            const void* src[3] = {L.trellis, L.suh, L.svh};
            const void* dst[3] = {dmm[p]->tr, dmm[p]->suh, dmm[p]->svh};
            for (int q = 0; q < 3; ++q) {
                tmp.resize(sz[q]);
                if (cudaMemcpy(tmp.data(), dst[q], sz[q], cudaMemcpyDeviceToHost) != cudaSuccess ||
                    std::memcmp(tmp.data(), src[q], sz[q]) != 0) {
                    static const char* plane[3] = {"trellis", "suh", "svh"};
                    size_t at = 0;
                    while (at < sz[q] && tmp[at] == ((const uint8_t*) src[q])[at]) ++at;
                    char got[80] = {0}, want[80] = {0};
                    for (int k = 0; k < 16 && at + k < sz[q]; ++k) {
                        std::snprintf(got + k * 5, 6, " %02x", tmp[at + k]);
                        std::snprintf(want + k * 5, 6, " %02x", ((const uint8_t*) src[q])[at + k]);
                    }
                    err = "exl3: the warmup fill's slot verify failed at layer " + std::to_string(l) +
                          " expert " + std::to_string(e) + " " + proj[p] + " " + plane[q] +
                          " size " + std::to_string(sz[q]) + " first diff at " + std::to_string(at) +
                          " got:" + got + " want:" + want;
                    return false;
                }
            }
        }
    }
    if (n_filled) *n_filled = filled;
    if (n_bytes) *n_bytes = bytes;
    return true;
}

void Exl3ExpertStream::set_pending_elsewhere(const uint8_t* table) { impl_->pending_table = table; }
void Exl3ExpertStream::set_route_observer(RouteObserver observer, void* user) {
    impl_->route_observer = observer; impl_->route_observer_user = user;
}
void Exl3ExpertStream::residency(int layers, int experts, std::vector<int32_t>& out) const {
    out.assign((size_t) layers * experts, -1);
    for (int l = impl_->layer_lo; l < std::min(layers, impl_->layer_hi); ++l)
        if (impl_->cache > 0)
            for (const auto& entry : impl_->layers[l].at)
                if (entry.first < experts) out[(size_t) l * experts + entry.first] = entry.second;
}

int64_t Exl3ExpertStream::owned_exclusions() const {
    return impl_->own_excl.load(std::memory_order_relaxed);
}

Exl3ExpertStream::Loan Exl3ExpertStream::lend_tail(int ranks) {
    Loan out;   // address == nullptr: the request changed nothing
    if (ranks <= 0 || !loan_available()) return out;
    const std::lock_guard<std::mutex> lock(impl_->run_mu);
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(impl_->device) != cudaSuccess) return out;
    struct RestoreDevice {
        int previous;
        ~RestoreDevice() { (void) cudaSetDevice(previous); }
    } restore{previous};
    if (ranks == impl_->loan.ranks) {          // a repeat of the current size: the same range again
        out.address = impl_->arena + (size_t) (impl_->cache - ranks) * impl_->arena_rank_bytes;
        out.bytes = (size_t) ranks * impl_->arena_rank_bytes;
        out.ranks = ranks;
        out.epoch = impl_->loan.epoch;
        return out;
    }
    // A shrink would have to hand ranks back selectively (restore first); one active rank stays
    // the floor (below it the cache degenerates into scratch while its bytes sit unreachable).
    if (ranks < impl_->loan.ranks || ranks > impl_->cache - 1) return out;
    const int new_active = impl_->cache - ranks;
    const int was_active = impl_->active;
    if (impl_->compute_stream && cudaStreamSynchronize(impl_->compute_stream) != cudaSuccess) return out;
    if (impl_->work_stream && cudaStreamSynchronize(impl_->work_stream) != cudaSuccess) return out;
    // Snapshot first, mutate last: everything that can throw runs while the loan is untouched.
    std::vector<Impl::TailRecord> add;
    add.reserve((size_t) (was_active - new_active) * (size_t) (impl_->layer_hi - impl_->layer_lo));
    for (int l = impl_->layer_lo; l < impl_->layer_hi; ++l)
        for (const auto& kv : impl_->layers[(size_t) l].at)
            if (kv.second >= new_active && kv.second < was_active)
                add.push_back({l, kv.first, kv.second, impl_->layers[(size_t) l].tick[(size_t) kv.second]});
    impl_->loan.saved.reserve(impl_->loan.saved.size() + add.size());   // may throw; still untouched
    impl_->loan.restored.assign((size_t) (impl_->layer_hi - impl_->layer_lo), {});   // ditto
    // Commit (noexcept): erase, append the records, clamp the dense watermark per layer.  The
    // mapped set stays exactly [0, filled) with filled <= new_active.
    for (const Impl::TailRecord& r : add) impl_->layers[(size_t) r.layer].at.erase(r.expert);
    impl_->loan.saved.insert(impl_->loan.saved.end(),
                             std::make_move_iterator(add.begin()), std::make_move_iterator(add.end()));
    for (int l = impl_->layer_lo; l < impl_->layer_hi; ++l)
        impl_->layers[(size_t) l].filled = std::min(impl_->layers[(size_t) l].filled, new_active);
    impl_->active = new_active;
    impl_->loan.ranks = ranks;
    ++impl_->loan.epoch;
    out.address = impl_->arena + (size_t) new_active * impl_->arena_rank_bytes;
    out.bytes = (size_t) ranks * impl_->arena_rank_bytes;
    out.ranks = ranks;
    out.epoch = impl_->loan.epoch;
    return out;
}

bool Exl3ExpertStream::restore_tail() {
    const std::lock_guard<std::mutex> lock(impl_->run_mu);
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(impl_->device) != cudaSuccess) return false;
    struct RestoreDevice {
        int previous;
        ~RestoreDevice() { (void) cudaSetDevice(previous); }
    } restore{previous};
    if (impl_->loan.ranks == 0) return false;   // no loan: an invalid request, not a no-op
    // A failed drain leaves the loan exactly as it was: the stream makes no "safe cold retry"
    // claim for a context whose syncs already fail.
    if (impl_->compute_stream && cudaStreamSynchronize(impl_->compute_stream) != cudaSuccess) return false;
    if (impl_->work_stream && cudaStreamSynchronize(impl_->work_stream) != cudaSuccess) return false;
    const int owned = impl_->layer_hi - impl_->layer_lo;
    struct Place { int layer; int expert; int slot; };
    std::vector<Place> places;
    std::vector<std::unordered_map<int, int>> next_at((size_t) owned);
    std::vector<int> next_filled((size_t) owned, -1);
    std::vector<std::vector<int>> next_restored((size_t) owned);   // the placements, precomputed here
    bool ok = true;
    impl_->jobs.clear();
    impl_->ram_pinned.assign((size_t) impl_->ram_slots, 0);
    try {
        // Plan: per layer (deterministic order = layer order, then the record's original slot),
        // survivors extend that layer's OWN occupied prefix into its own tail.  `next_at` starts as
        // a copy of the layer's map, so a record whose expert is resident - or already placed by an
        // earlier record of the same expert (growth can re-lend a re-admitted expert) - is skipped
        // exactly once; a hot prefix copy always wins over the overwritten tail bytes.
        for (int li = 0; li < owned; ++li) {
            const int l = impl_->layer_lo + li;
            next_at[(size_t) li] = impl_->layers[(size_t) l].at;    // copy; may throw, nothing mutated
            next_filled[(size_t) li] = impl_->layers[(size_t) l].filled;
        }
        for (int li = 0; li < owned; ++li) {
            const int l = impl_->layer_lo + li;
            Impl::Layer& L = impl_->layers[(size_t) l];
            std::vector<const Impl::TailRecord*> records;
            for (const Impl::TailRecord& r : impl_->loan.saved) if (r.layer == l) records.push_back(&r);
            std::sort(records.begin(), records.end(),
                      [](const Impl::TailRecord* a, const Impl::TailRecord* b) { return a->slot < b->slot; });
            int slot = next_filled[(size_t) li];    // the layer's occupied prefix: [filled, active) is
            for (const Impl::TailRecord* rp : records) {   // unused, the tail is [active, cache)
                if (next_at[(size_t) li].count(rp->expert) != 0) continue;
            if (impl_->admission_excluded((int64_t) l, rp->expert)) {   // peer-owned or reserved incoming
                ++impl_->own_excl;
                continue;
            }
                // Bounded admission, not an overflow error: one loan's records all come from the
                // DISJOINT rank increments of its lends, so their count (and unique experts) never
                // exceeds the lent ranks and slot < cache always holds under the dense invariant -
                // filled <= active = cache - ranks.  Should that bound ever be violated, the excess
                // records in this deterministic order are DROPPED (not resident; ordinary misses
                // re-admit them) rather than failing a valid loan's restore.
                if (slot >= impl_->cache) break;
                impl_->cur_layer = l;
                impl_->prepare_expert(L, slot, rp->expert);   // queue the re-read (pread/pinned staging)
                next_at[(size_t) li][rp->expert] = slot;      // may throw; still only the copy
                next_filled[(size_t) li] = slot + 1;
                places.push_back({l, rp->expert, slot});
                next_restored[(size_t) li].push_back(rp->expert);
                ++slot;
            }
        }
        impl_->stage_parallel();
        if (cudaStreamSynchronize(impl_->work_stream) != cudaSuccess)
            throw std::runtime_error("loan restore drain failed");
    } catch (const std::exception& error) {
        // Drain the partial uploads (their pinned/fd sources must be released), publish nothing.
        (void) cudaStreamSynchronize(impl_->work_stream);
        std::fprintf(stderr, "exl3 stream: loan restore failed: %s; every layer's residency invalidated\n",
                     error.what());
        ok = false;
    }
    if (ok) {
        // Publish only after every copy completed, and only with operations that cannot throw or
        // allocate: swap the prepared maps, placements and all in one noexcept sequence.
        for (int li = 0; li < owned; ++li) {
            Impl::Layer& L = impl_->layers[(size_t) (impl_->layer_lo + li)];
            L.at.swap(next_at[(size_t) li]);
            L.filled = next_filled[(size_t) li];
            if ((size_t) li < impl_->loan.restored.size())
                impl_->loan.restored[(size_t) li].swap(next_restored[(size_t) li]);
        }
        for (const Place& p : places)
            impl_->layers[(size_t) p.layer].tick[(size_t) p.slot] = ++impl_->layers[(size_t) p.layer].clock;
    } else {
        // The run error paths' rule: a slot that may hold half-uploaded bytes - device or host RAM
        // cache alike - must never be advertised.  Capacity is still returned; the next calls
        // rebuild residency with ordinary misses.  A restore that RAN and failed placed nothing.
        for (Impl::Layer& L : impl_->layers) { L.at.clear(); L.filled = 0; }
        for (Impl::RamLayer& rl : impl_->ram_layers) { rl.at.clear(); rl.filled = 0; }
        for (std::vector<int>& r : impl_->loan.restored) r.clear();
    }
    impl_->loan.saved.clear();
    impl_->loan.ranks = 0;
    impl_->active = impl_->cache;
    ++impl_->loan.epoch;
    return ok;
}

size_t Exl3ExpertStream::bytes() const {
    size_t n = (size_t) impl_->n_embd * 2 + (size_t) impl_->max_k * impl_->n_embd * (2 + 4);
    for (auto& s : impl_->scratch.g) n += s.trb + s.suhb + s.svhb;
    for (auto& s : impl_->scratch.u) n += s.trb + s.suhb + s.svhb;
    for (auto& s : impl_->scratch.d) n += s.trb + s.suhb + s.svhb;
    for (const Impl::Layer& l : impl_->layers)
        for (const auto& v : { &l.g, &l.u, &l.d })
            for (const Impl::Slot& s : *v) n += s.trb + s.suhb + s.svhb;
    if (impl_->verify_graph) {
        const auto& v = *impl_->verify_graph;
        n += v.workspace.allocated_bytes() + v.device_table.capacity * sizeof(Exl3Mat) +
             (v.gate.capacity + v.up.capacity + v.down.capacity) * sizeof(Exl3Mat) +
             v.input_rows.capacity * sizeof(int) + v.skip.capacity * sizeof(uint32_t) +
             v.x.capacity * sizeof(uint16_t);
    }
    return n;
}

void Exl3ExpertStream::enable_verify_graph(int max_tokens, int k) {
    if (impl_->verify_graph) return;
    if (max_tokens < 1 || max_tokens > 8 || k < 1 || k > impl_->max_k)
        throw std::invalid_argument("exl3: invalid verifier graph capacity");
    auto v = std::make_unique<VerifyGraphBuffers>();
    v->max_tokens = max_tokens; v->k = k;
    const int routes = max_tokens * k, n = impl_->n_embd;
    for (int l = impl_->layer_lo; l < impl_->layer_hi; ++l)
        v->max_ff = std::max(v->max_ff, impl_->model->linear("model.language_model.layers." +
            std::to_string(l) + ".mlp.experts.0.gate_proj").out());
    v->table.allocate((size_t) impl_->n_layers * 3 * 512);
    v->device_table.reserve((size_t) impl_->n_layers * 3 * 512);
    v->record.allocate(impl_->n_layers);
    const char* check = std::getenv("STRATA_EXL3_VERIFY_CHECK");
    v->check = check && check[0] && check[0] != '0';
    if (v->check) {
        v->completed.allocate(impl_->n_layers);
        v->reference.allocate((size_t) routes * n);
        v->actual.allocate((size_t) routes * n);
    }
    v->gate.reserve(routes); v->up.reserve(routes); v->down.reserve(routes);
    v->input_rows.reserve(routes); v->skip.reserve(1);
    v->x.reserve((size_t) max_tokens * n);
    v->workspace.reserve(routes, n, v->max_ff, n, true);
    v->workspace.reserve_runtime_scratch(routes, n, v->max_ff, n);
    // The callback's miss count can grow after the first captured window too.
    impl_->workspace->reserve_runtime_scratch(8 * impl_->max_k, n, v->max_ff, n);
    impl_->verify_graph = std::move(v);
    if (impl_->small_m) {
        impl_->small_m = false;
        std::fprintf(stderr, "strata EXL3: verifier graph uses indexed FFNs; SMALL_M is inactive\n");
    }
    refresh_verify_graph();
    std::fprintf(stderr, "strata EXL3: GPU-resident verifier hits enabled (%d rows, k=%d; one group)\n",
                 max_tokens, k);
}

void Exl3ExpertStream::refresh_verify_graph(void* stream) {
    auto& v = *impl_->verify_graph;
    if (v.check) std::fill_n(v.completed.host, impl_->n_layers, 0u);
    // Called with no verifier window in flight. Prefill, loans and other request paths may
    // have changed residency since the previous window; rebuild from the authoritative map.
    for (int l = impl_->layer_lo; l < impl_->layer_hi; ++l) {
        Exl3Mat* table = v.table.host + (size_t) l * 3 * 512;
        std::fill_n(table, 3 * 512, Exl3Mat{});
        // With a smaller active cache, preserve the existing bounded host waves. A hit
        // snapshot is used only when a whole window's distinct experts fits beside its hits.
        if (impl_->active < v.max_tokens * v.k) continue;
        const auto& c = impl_->layers[l];
        for (const auto& entry : c.at) {
            const int e = entry.first, s = entry.second;
            if (s < 0 || s >= impl_->active || impl_->owned_elsewhere(l, e)) continue;
            table[e] = c.gm[s]; table[512 + e] = c.um[s]; table[1024 + e] = c.dm[s];
        }
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (stream && cudaMemcpyAsync(v.device_table.p, v.table.host,
            (size_t) impl_->n_layers * 3 * 512 * sizeof(Exl3Mat),
            cudaMemcpyHostToDevice, (cudaStream_t) stream) != cudaSuccess)
        throw std::runtime_error("exl3: verifier table upload failed");
}

void Exl3ExpertStream::verify_graph_hits(int layer, const float* x, const int32_t*,
                                       int tokens, int k, float* out, uint32_t, void* stream) {
    auto& v = *impl_->verify_graph;
    const int n = impl_->n_embd, routes = tokens * k;
    const auto st = (cudaStream_t) stream;
    f32_to_f16_rows_kernel<<<(tokens * n + 255) / 256, 256, 0, st>>>(x, v.x.p, (long) tokens * n);
    const int ff = impl_->model->linear("model.language_model.layers." + std::to_string(layer) +
                                       ".mlp.experts.0.gate_proj").out();
    const uint16_t* rows = exl3_moe_rows_masked(v.gate.p, v.up.p, v.down.p, routes, n, ff, n, v.x.p,
                                               v.input_rows.p, stream, v.workspace);
    verify_output_kernel<false><<<(routes * n + 255) / 256, 256, 0, st>>>(
        v.gate.p, rows, nullptr, out, routes, n);
    if (v.check) {
        v.graph_out = out;
        verify_complete_kernel<<<1, 1, 0, st>>>(v.completed.device + layer);
    }
}

const uint32_t* Exl3ExpertStream::verify_graph_skip() const { return impl_->verify_graph->skip.p; }

void Exl3ExpertStream::verify_graph_publish(int layer, const float* x, const int32_t* ids, int tokens, int k,
                                          float* mapped_x, int32_t* mapped_ids, uint32_t* seq,
                                          uint32_t ring, void* stream) {
    auto& v = *impl_->verify_graph;
    if (tokens < 1 || tokens > v.max_tokens || k != v.k)
        throw std::runtime_error("exl3: verifier graph shape changed");
    verify_gather_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(v.device_table.p + (size_t) layer * 3 * 512,
        ids, tokens * k, k, v.gate.p, v.up.p, v.down.p, v.input_rows.p,
        v.record.device + layer, v.skip.p, ring, x, tokens * impl_->n_embd, mapped_x, mapped_ids, seq);
}

void Exl3ExpertStream::verify_graph_copy_misses(const float* mapped, float* out, int routes, void* stream) {
    auto& v = *impl_->verify_graph;
    const int n = impl_->n_embd;
    verify_output_kernel<true><<<(routes * n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(
        v.gate.p, nullptr, mapped, out, routes, n);
}

bool Exl3ExpertStream::verify_graph_begin_host(int layer) {
    auto& v = *impl_->verify_graph;
    const auto& rec = v.record.host[layer]; // doorbell's system fence published this record
    if (impl_->route_observer) impl_->route_observer(impl_->route_observer_user, layer, rec.ids, rec.routes);
    ++v.layers; v.hit_routes += rec.routes - rec.misses; v.miss_routes += rec.misses;
    v.all_hit_layers += rec.misses == 0;
    // Preserve hits' LRU use before any miss admissions, once per expert in ascending order,
    // matching the host route planner. Miss service pins these slots until staging is done.
    if (impl_->cache > 0) {
        auto& c = impl_->layers[layer];
        bool seen[512] = {};
        for (int r = 0; r < rec.routes; ++r) if (rec.hit[r]) seen[rec.ids[r]] = true;
        for (int e = 0; e < 512; ++e) {
            if (!seen[e]) continue;
            const auto it = c.at.find(e);
            if (it == c.at.end()) throw std::runtime_error("exl3: graph hit lost residency before host service");
            c.tick[it->second] = ++c.clock;
            ++impl_->hits;
        }
    }
    v.host_active = rec.misses != 0;
    return v.host_active;
}

void Exl3ExpertStream::verify_graph_end_host(int) { impl_->verify_graph->host_active = false; }
bool Exl3ExpertStream::verify_graph_host_active() const {
    return impl_->verify_graph && impl_->verify_graph->host_active;
}

void Exl3ExpertStream::verify_graph_check(int layer, const float* x, const int32_t* ids, int tokens, int k) {
    if (!impl_->verify_graph || !impl_->verify_graph->check) return;
    auto& v = *impl_->verify_graph;
    v.host_active = false;
    if (!run_multi(layer, x, ids, tokens, k, v.reference.host, nullptr))
        throw std::runtime_error("exl3: verify check oracle failed");
    v.host_active = true;
    const auto started = std::chrono::steady_clock::now();
    while (!*(volatile uint32_t*) (v.completed.host + layer)) {
        if (std::chrono::steady_clock::now() - started > std::chrono::seconds(10))
            throw std::runtime_error("exl3: verify check completion timeout");
        std::this_thread::yield();
    }
    const size_t elements = (size_t) tokens * k * impl_->n_embd;
    if (cudaMemcpyAsync(v.actual.host, v.graph_out, elements * sizeof(float), cudaMemcpyDeviceToHost,
                        impl_->compute_stream) != cudaSuccess || cudaStreamSynchronize(impl_->compute_stream) != cudaSuccess)
        throw std::runtime_error("exl3: verify check readback failed");
    const auto& rec = v.record.host[layer];
    for (int r = 0; r < tokens * k; ++r) if (rec.hit[r]) {
        const size_t base = (size_t) r * impl_->n_embd;
        if (std::memcmp(v.reference.host + base, v.actual.host + base, impl_->n_embd * sizeof(float))) {
            int different = 0; float worst = 0;
            for (int i = 0; i < impl_->n_embd; ++i) {
                different += v.reference.host[base+i] != v.actual.host[base+i];
                worst = std::max(worst, std::abs(v.reference.host[base+i] - v.actual.host[base+i]));
            }
            std::fprintf(stderr, "exl3 verify check: L%d T%d row%d expert%d differs in %d lanes maxabs=%g\n",
                         layer, tokens, r, ids[r], different, worst);
            throw std::runtime_error("exl3: graph hit differs from host oracle");
        }
        ++v.checked_rows;
    }
}

bool Exl3ExpertStream::run(int layer, const float* x_f32, const int* ids, int k, float* out, void* stream,
                           uint8_t* cpu_rows) {
    if (k == 0) return true;
    if (k < 0 || k > impl_->max_k || layer < impl_->layer_lo || layer >= impl_->layer_hi ||
        x_f32 == nullptr || ids == nullptr || out == nullptr) return false;
    // STRATA_EXL3_CPU_TIER (B1): without the env switch a caller's buffer changes nothing
    if (cpu_rows != nullptr && !impl_->cpu_tier) cpu_rows = nullptr;
    if (cpu_rows != nullptr) std::memset(cpu_rows, 0, (size_t) k);
    for (int i = 0; i < k; ++i) if (ids[i] < 0 || ids[i] >= 512) return false;
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < i; ++j)
            if (ids[i] == ids[j]) return run_multi(layer, x_f32, ids, 1, k, out, stream, cpu_rows);
    const std::lock_guard<std::mutex> run_lock(impl_->run_mu);
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(impl_->device) != cudaSuccess) return false;
    struct RestoreDevice {
        int previous;
        ~RestoreDevice() { (void) cudaSetDevice(previous); }
    } restore{previous};
    try {
    if (stream == nullptr) stream = (void*) impl_->compute_stream;
    cudaStream_t st = (cudaStream_t) stream;
    const long n = impl_->n_embd;
    static const bool timing = std::getenv("STRATA_EXL3_TIME") != nullptr;
    const auto t_call0 = std::chrono::steady_clock::now();
    const auto t_up0 = std::chrono::steady_clock::now();
    if (cudaMemcpyAsync(impl_->d_input, x_f32, (size_t) n * sizeof(float), cudaMemcpyHostToDevice, st) != cudaSuccess)
        return false;
    f32_to_f16_rows_kernel<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(impl_->d_input, impl_->d_x, n);

    impl_->jobs.clear();
    impl_->ram_pinned.assign((size_t) impl_->ram_slots, 0);
    // ---- which slots this token's k experts live in.  A cache hit costs nothing; a miss reads the
    // expert out of the mmap and uploads it into the least recently used slot of this layer.
    // `active` (not `cache`): while a loan is open only the prefix is routable/evictable; k above
    // it takes the scratch, never a lent rank.
    const int active = impl_->active;
    std::vector<Exl3Mat> gm, um, dm;                 // the no-split path (no cache): one list of k
    std::vector<Exl3Mat> hg, hu, hd, mg, mu, md;     // the split path: resident rows and miss rows
    std::vector<int> hrow, mrow;
    Impl::Layer* cache = (impl_->cache > 0 && layer >= 0 && layer < (int) impl_->layers.size() &&
                          k <= active) ? &impl_->layers[(size_t) layer] : nullptr;
    impl_->cur_layer = layer;
    if (cache != nullptr) {
        hg.reserve((size_t) k); hu.reserve((size_t) k); hd.reserve((size_t) k);
        mg.reserve((size_t) k); mu.reserve((size_t) k); md.reserve((size_t) k);
        hrow.reserve((size_t) k); mrow.reserve((size_t) k);
        for (int i = 0; i < k; ++i) {
            const int e = ids[i];
            auto it = cache->at.find(e);
            int s = -1;
            const bool resident = it != cache->at.end();
            if (resident) {
                s = it->second;
                ++impl_->hits;
            } else if (impl_->admission_excluded(layer, e)) {
                // owned by another tier (peer): COMPUTE on the fallback path without admitting -
                // the bounded scratch, never a cache slot and never a lent rank
                ++impl_->misses;
                ++impl_->own_excl;
                mrow.push_back(i);
                impl_->prepare_expert(impl_->scratch, i, e);
                mg.push_back(impl_->scratch.gm[(size_t) i]);
                mu.push_back(impl_->scratch.um[(size_t) i]);
                md.push_back(impl_->scratch.dm[(size_t) i]);
                continue;
            } else {
                ++impl_->misses;
                // STRATA_EXL3_CPU_TIER: with the cache FULL, REFUSE instead of evicting (the GGUF
                // pool's overflow policy): no admission, no upload, the slot set stays as it is.
                // The caller computes this row on the CPU; `out` gets a zeroed row below.
                // STRATA_EXL3_CPU_MAX_PER_CALL bounds the per-call refusals (see the pre-pass).
                if (cpu_rows != nullptr && cache->filled >= active) {
                    static const int cpu_cap = [] {
                        const char* v = std::getenv("STRATA_EXL3_CPU_MAX_PER_CALL");
                        return (v != nullptr && std::atoi(v) > 0) ? std::atoi(v) : 0;
                    }();
                    int refused = 0;
                    for (int q = 0; q < i; ++q) refused += cpu_rows[q] != 0;
                    if (cpu_cap == 0 || refused < cpu_cap) {
                        cpu_rows[i] = 1;
                        ++impl_->cpu_refused;
                        continue;
                    }
                }
                if (cache->filled < active) {
                    s = cache->filled++;                 // still filling: the slots are handed out in order
                } else {
                    s = (int) (std::min_element(cache->tick.begin(), cache->tick.begin() + active) -
                               cache->tick.begin());
                    for (auto victim = cache->at.begin(); victim != cache->at.end(); ++victim)
                        if (victim->second == s) { cache->at.erase(victim); break; }
                }
                impl_->prepare_expert(*cache, s, e);
                cache->at[e] = s;
            }
            cache->tick[(size_t) s] = ++cache->clock;
            if (resident) {
                hrow.push_back(i);
                hg.push_back(cache->gm[(size_t) s]); hu.push_back(cache->um[(size_t) s]); hd.push_back(cache->dm[(size_t) s]);
            } else {
                mrow.push_back(i);
                mg.push_back(cache->gm[(size_t) s]); mu.push_back(cache->um[(size_t) s]); md.push_back(cache->dm[(size_t) s]);
            }
        }
    } else {
        gm.resize((size_t) k); um.resize((size_t) k); dm.resize((size_t) k);
        for (int i = 0; i < k; ++i) {
            impl_->prepare_expert(impl_->scratch, i, ids[i]);
            gm[(size_t) i] = impl_->scratch.gm[(size_t) i];
            um[(size_t) i] = impl_->scratch.um[(size_t) i];
            dm[(size_t) i] = impl_->scratch.dm[(size_t) i];
        }
        // no cache: nothing is resident, so the whole FFN follows the staging (the split has nothing to run early)
        impl_->stage_parallel();
        // the FFN reads the staged slots, so it waits the staging event HERE: the shared
        // `stage_done` wait below runs after these kernels are already queued, too late for them
        if (!impl_->jobs.empty() &&
            (cudaEventRecord(impl_->stage_done, impl_->work_stream) != cudaSuccess ||
             cudaStreamWaitEvent(st, impl_->stage_done, 0) != cudaSuccess))
            throw std::runtime_error("exl3: expert upload event failed");
        exl3_moe_rows(gm.data(), um.data(), dm.data(), ids, k, impl_->d_x, impl_->d_rows, stream, impl_->workspace.get());
        // consumed: the shared tail below must not re-stage them (a duplicate copy would race
        // these kernels)
        impl_->jobs.clear();
    }
    // **THE RESIDENT HALF'S FFN RUNS WHILE THE MISSES ARE READ.**  Their weights are already in VRAM, so
    // nothing has to wait for the host: the kernel is enqueued before the workers start, and the miss FFN
    // follows the staging copies on the same stream.  Without the split the whole FFN waited for the last
    // miss, which is most of the doorbell.
    if (!hrow.empty())
        exl3_moe_rows_part(hg.data(), hu.data(), hd.data(), (int) hrow.size(), impl_->d_x, impl_->d_hit, stream, impl_->workspace.get());
    impl_->stage_parallel();
    const auto t_up1 = std::chrono::steady_clock::now();
    // the miss FFN (and the scatter) read what the staging wrote: order them after it
    if (!impl_->jobs.empty()) {
        (void) cudaEventRecord(impl_->stage_done, impl_->work_stream);
        (void) cudaStreamWaitEvent(st, impl_->stage_done, 0);
    }
    if (!mrow.empty()) {
        (void) cudaMemcpyAsync(impl_->d_hit_rows, hrow.data(), hrow.size() * sizeof(int), cudaMemcpyHostToDevice, st);
        (void) cudaMemcpyAsync(impl_->d_miss_rows, mrow.data(), mrow.size() * sizeof(int), cudaMemcpyHostToDevice, st);
        exl3_moe_rows_part(mg.data(), mu.data(), md.data(), (int) mrow.size(), impl_->d_x, impl_->d_miss, stream, impl_->workspace.get());
        exl3_rows_scatter(impl_->d_hit, impl_->d_hit_rows, (int) hrow.size(), impl_->d_miss, impl_->d_miss_rows,
                          (int) mrow.size(), (int) n, impl_->d_rows, stream);
    } else if (!hrow.empty()) {
        (void) cudaMemcpyAsync(impl_->d_hit_rows, hrow.data(), hrow.size() * sizeof(int), cudaMemcpyHostToDevice, st);
        exl3_rows_scatter(impl_->d_hit, impl_->d_hit_rows, (int) hrow.size(), nullptr, nullptr, 0, (int) n,
                          impl_->d_rows, stream);
    }
    const auto t_ffn1 = std::chrono::steady_clock::now();
    f16_to_f32_rows_kernel<<<(unsigned)((n * k + 255) / 256), 256, 0, st>>>(impl_->d_rows, impl_->d_out, n * k);
    if (cudaMemcpyAsync(out, impl_->d_out, (size_t) n * k * 4, cudaMemcpyDeviceToHost, st) != cudaSuccess) return false;
    const cudaError_t se = cudaStreamSynchronize(st);
    if (impl_->work_stream != nullptr) (void) cudaStreamSynchronize(impl_->work_stream);   // its copies feed this layer's FFN
    if (se != cudaSuccess) std::fprintf(stderr, "exl3 stream run(layer=%d): %s\n", layer, cudaGetErrorString(se));
    // The scatter above copied every k row from the device, refused ones included (garbage
    // there); the caller overwrites them, but zero them so a skipped overwrite can never leak
    // plausible-looking values from a previous token
    if (cpu_rows != nullptr)
        for (int i = 0; i < k; ++i)
            if (cpu_rows[i] != 0) std::memset(out + (size_t) i * n, 0, (size_t) n * sizeof(float));
    if (timing) {
        const auto t_d2h1 = std::chrono::steady_clock::now();
        impl_->up_ms += std::chrono::duration<double, std::milli>(t_up1 - t_up0).count();
        impl_->ffn_ms += std::chrono::duration<double, std::milli>(t_ffn1 - t_up1).count();
        impl_->d2h_ms += std::chrono::duration<double, std::milli>(t_d2h1 - t_ffn1).count();
        impl_->call_ms += std::chrono::duration<double, std::milli>(t_d2h1 - t_call0).count();
        if (layer == impl_->n_layers - 1) {
            const long long uses = impl_->hits + impl_->misses;
            std::fprintf(stderr, "exl3 stream: %lld experts, %.1f%% cached, cpu %lld, host %.1f%% (%lld reads) | "
                                 "upload %.1f ms, ffn %.1f ms, d2h %.1f ms, pool %.1f ms per token\n", uses,
                         100.0 * (double) impl_->hits / (double) std::max<long long>(uses, 1),
                         impl_->cpu_refused,
                         100.0 * (double) impl_->ram_hits / (double) std::max<long long>(impl_->ram_hits + impl_->ram_misses, 1),
                         impl_->ram_misses, impl_->up_ms, impl_->ffn_ms, impl_->d2h_ms, impl_->call_ms);
            impl_->hits = impl_->misses = 0;
            impl_->cpu_refused = 0;
            impl_->ram_hits = impl_->ram_misses = 0;
            impl_->up_ms = impl_->ffn_ms = impl_->d2h_ms = impl_->call_ms = 0;
        }
    }
    return se == cudaSuccess;
    } catch (const std::exception& error) {
        (void) cudaStreamSynchronize(stream ? (cudaStream_t) stream : impl_->compute_stream);
        (void) cudaStreamSynchronize(impl_->work_stream);
        for (auto& layer_cache : impl_->layers) { layer_cache.at.clear(); layer_cache.filled = 0; }
        for (auto& layer_cache : impl_->ram_layers) { layer_cache.at.clear(); layer_cache.filled = 0; }
        std::fprintf(stderr, "exl3 stream(layer=%d): %s\n", layer, error.what());
        return false;
    }
}

bool Exl3ExpertStream::run_multi(int layer, const float* x_f32, const int32_t* ids, int tokens, int k,
                                float* out, void* stream, uint8_t* cpu_rows) {
    return run_multi_impl(layer, x_f32, ids, tokens, k, out, stream, false, cpu_rows);
}

void Exl3ExpertStream::classify(int layer, const int32_t* ids, int tokens, int k, uint8_t* flags) const
{
    // STRATA_EXL3_CPU_TIER's overlap helper: the refusal decision run()/run_multi would make
    // RIGHT NOW, without side effects.  flags[t*k+j] = 1 marks a route the call will refuse
    // (cache full, nothing admitted or evicted for it); the caller starts its CPU tier on
    // those rows, then calls run/run_multi with the SAME buffer, whose decision is recomputed
    // identically (the cache state cannot change between the two on one calling thread).
    // The per-path consumption order is mirrored exactly: the single-token loop walks routes
    // in order, the multi plan groups misses by ascending expert id - a divergence would
    // refuse a different expert than the CPU computed, and the row would be nobody's.
    if (flags == nullptr || tokens <= 0 || k <= 0) return;
    const int64_t routes = (int64_t) tokens * k;
    if (routes <= 0 || routes > (1 << 20)) return;
    std::memset(flags, 0, (size_t) routes);
    if (!impl_->cpu_tier) return;
    const int active = impl_->active;
    const bool single = tokens == 1;
    const bool layer_ok = layer >= impl_->layer_lo && layer < impl_->layer_hi &&
                          layer >= 0 && layer < (int) impl_->layers.size();
    const Impl::Layer* cache = (impl_->cache > 0 && layer_ok && (!single || k <= active))
                                   ? &impl_->layers[(size_t) layer] : nullptr;
    if (cache == nullptr) return;
    static const int cpu_cap = [] {
        const char* v = std::getenv("STRATA_EXL3_CPU_MAX_PER_CALL");
        return (v != nullptr && std::atoi(v) > 0) ? std::atoi(v) : 0;
    }();
    if (single)
    {
        int fill_room = active - cache->filled;
        int refused = 0;
        std::vector<int> admitted;
        admitted.reserve((size_t) k);
        for (int i = 0; i < k; ++i)
        {
            const int e = ids[i];
            const bool resident = cache->at.count(e) != 0 ||
                                  std::find(admitted.begin(), admitted.end(), e) != admitted.end();
            if (resident || impl_->admission_excluded(layer, e)) continue;
            if (fill_room > 0) { --fill_room; admitted.push_back(e); continue; }
            if (cpu_cap > 0 && refused + 1 > cpu_cap) continue;
            flags[i] = 1;
            ++refused;
        }
        return;
    }
    // multi: the plan's groups (post-resident-prioritization) keep the misses in ascending
    // expert order, and only misses consume fill room
    std::vector<int> present;
    present.reserve((size_t) routes);
    for (int64_t i = 0; i < routes; ++i)
        if (std::find(present.begin(), present.end(), ids[i]) == present.end())
            present.push_back(ids[i]);
    std::sort(present.begin(), present.end());
    int fill_room = active - cache->filled;
    int refused_routes = 0;
    for (int e : present)
    {
        if (cache->at.count(e) != 0 || impl_->admission_excluded(layer, e)) continue;
        if (fill_room > 0) { --fill_room; continue; }
        int rows = 0;
        for (int64_t i = 0; i < routes; ++i) rows += ids[i] == e;
        if (cpu_cap > 0 && refused_routes + rows > cpu_cap) continue;
        refused_routes += rows;
        for (int64_t i = 0; i < routes; ++i)
            if (ids[i] == e) flags[i] = 1;
    }
}

bool Exl3ExpertStream::run_device(int layer, const float* x_f32, const int32_t* ids, int tokens, int k,
                                 float* out, void* stream, const uint8_t* skip) {
    return run_multi_impl(layer, x_f32, ids, tokens, k, out, stream, true, nullptr, false, skip);
}

bool Exl3ExpertStream::run_multi_complement(int layer, const float* x_f32, const int32_t* ids,
                                          int tokens, int k, float* out, void* stream, const uint8_t* cpu_skip) {
    return run_multi_impl(layer, x_f32, ids, tokens, k, out, stream, false, nullptr, true, cpu_skip);
}

bool Exl3ExpertStream::run_multi_impl(int layer, const float* x_f32, const int32_t* ids, int tokens, int k,
                                     float* out, void* stream, bool device_io, uint8_t* cpu_rows, bool skip_owned,
                                     const uint8_t* cpu_skip) {
    if (tokens == 0) return true;
    if (tokens < 0 || k <= 0 || k > impl_->max_k || layer < impl_->layer_lo || layer >= impl_->layer_hi ||
        !x_f32 || !ids || !out || tokens > INT32_MAX / k) return false;
    // STRATA_EXL3_CPU_TIER (B1): host-IO calls only (the prefill arm never passes a buffer),
    // and the env switch must be on
    if (cpu_rows != nullptr && (!impl_->cpu_tier || device_io)) cpu_rows = nullptr;
    if (cpu_rows != nullptr) std::memset(cpu_rows, 0, (size_t) tokens * (size_t) k);
    const std::lock_guard<std::mutex> lock(impl_->run_mu);
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(impl_->device) != cudaSuccess) return false;
    struct RestoreDevice {
        int previous;
        ~RestoreDevice() { (void) cudaSetDevice(previous); }
    } restore{previous};
    const auto st = stream ? (cudaStream_t) stream : impl_->compute_stream;
    const bool timing = impl_->profile[0] != nullptr;
    const auto call_started = std::chrono::steady_clock::now();
    try {
        auto& plan = impl_->route_plan;
        const bool prof = impl_->prof;
        const auto prof_t0 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        plan.build(ids, tokens, k, 512);
        if (cpu_skip) {
            // A fixed mask, not a second admission decision: the CPU job already
            // owns these destinations. Validate entire groups before any upload.
            std::vector<uint8_t> drop(plan.experts.size());
            for (size_t g = 0; g < plan.experts.size(); ++g) {
                const bool skip = cpu_skip[plan.destinations[plan.offsets[g]]] != 0;
                for (int r = plan.offsets[g]; r < plan.offsets[g + 1]; ++r)
                    if ((cpu_skip[plan.destinations[r]] != 0) != skip)
                        throw std::invalid_argument("exl3: CPU skip mask splits an expert group");
                drop[g] = skip;
            }
            plan.drop_groups(drop);
            if (plan.experts.empty()) return true;
        }
        if (!device_io && verify_graph_host_active()) {
            // The GPU's mask is the pre-admission snapshot, not a map that miss service may
            // mutate. Drop whole expert groups while retaining original destinations/tokens.
            const auto& rec = impl_->verify_graph->record.host[layer];
            std::vector<uint8_t> drop(plan.experts.size(), 0);
            for (size_t g = 0; g < plan.experts.size(); ++g)
                for (int r = 0; r < rec.routes; ++r)
                    if (rec.ids[r] == plan.experts[g] && rec.hit[r]) { drop[g] = 1; break; }
            plan.drop_groups(drop);
            if (plan.experts.empty()) return true;
        }
        if (skip_owned && impl_->own_table != nullptr) {
            std::vector<uint8_t> drop(plan.experts.size());
            for (size_t g = 0; g < plan.experts.size(); ++g)
                drop[g] = impl_->owned_elsewhere(layer, plan.experts[g]);
            plan.drop_groups(drop);   // destinations retain the original token*k+j indices
            if (plan.experts.empty()) return true;   // all peer-owned: no input upload or GPU work
        }
        if (prof) impl_->prof_plan_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - prof_t0).count();
        constexpr int tile_limit = 4096;
        const int tile = std::min(tokens * k, tile_limit);
        const int n = impl_->n_embd;
        auto& batch = *impl_->multi;
        batch.reserve(tokens, tile, n, !device_io);
        if (!device_io && cudaMemcpyAsync(batch.input.p, x_f32, (size_t) tokens * n * sizeof(float),
                                          cudaMemcpyHostToDevice, st) != cudaSuccess)
            throw std::runtime_error("exl3: multi input upload failed");
        const long input_elements = (long) tokens * n;
        f32_to_f16_rows_kernel<<<(unsigned) ((input_elements + 255) / 256), 256, 0, st>>>(
            device_io ? x_f32 : batch.input.p, batch.x.p, input_elements);
        // STRATA_EXL3_SMALL_M: this call's enablement.  HOST-IO arms only (device_io is the
        // prefill path), and ownership-elsewhere bypasses it unless this is the explicit peer
        // complement (owned groups have already been removed). NOTHING is allocated here:
        // the flag's device and host
        // budget is the setup-time envelope (see the constructor), and a call exceeding the
        // envelope - or one where anything would have to grow - simply stays on the indexed
        // GEMV, because a device allocation under the verify graph's flag-spin deadlocks and
        // the host callback must not allocate at all.
        const bool small_m_on = impl_->small_m && !device_io && (impl_->own_table == nullptr || skip_owned) &&
                                std::min(tile, tokens * k) <= impl_->small_m_envelope;
        if (impl_->small_m_trace)
            std::fprintf(stderr, "smtrace call layer=%d tokens=%d k=%d groups=%zu small_m_on=%d\n",
                         layer, tokens, k, plan.experts.size(), (int) small_m_on);
        impl_->cur_layer = layer;
        auto* cache = impl_->cache > 0 ? &impl_->layers[(size_t) layer] : nullptr;
        // A loan's lent tail is outside every scan below: waves, fills and evictions only ever see
        // the active prefix, so staging never targets a rank the borrower owns.
        const int active = impl_->active;
        if (cache && !cache->at.empty()) {
            const auto prof_t1 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            std::vector<uint8_t> resident(plan.experts.size());
            for (size_t g = 0; g < plan.experts.size(); ++g)
                resident[g] = cache->at.count(plan.experts[g]) != 0;
            plan.prioritize_resident(resident);
            if (prof) impl_->prof_plan_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - prof_t1).count();
        }
        // ---- STRATA_EXL3_CPU_TIER (B1): the full-cache refusal pre-pass.  A group the cache
        // would have to EVICT for is refused outright - nothing admitted, evicted or uploaded -
        // its routes flagged in cpu_rows for the caller to compute on the CPU, and the group
        // dropped from the plan so every arm below (waves, overlap split, batching, tiles)
        // sees GPU work only and never writes the refused rows.  `fill_room` mirrors the
        // admission loop's accounting exactly: only a miss with room consumes a fill slot, so
        // the split here and the admissions below agree on which experts this call uploads.
        // STRATA_EXL3_CPU_MAX_PER_CALL caps the refusal count per call (in ROUTES): beyond the
        // cap the miss keeps today's evict-and-upload, bounding the CPU tier's share of a
        // window when the tail is heavier than the CPU can hide (measured: 47% of routes on
        // the CPU cost PROSE 21% vs control; a bounded share keeps the win without the loss).
        if (cpu_rows != nullptr && cache != nullptr) {
            int fill_room = active - cache->filled;
            static const int cpu_cap = [] {
                const char* v = std::getenv("STRATA_EXL3_CPU_MAX_PER_CALL");
                return (v != nullptr && std::atoi(v) > 0) ? std::atoi(v) : 0;
            }();
            int refused_routes = 0;
            std::vector<uint8_t> drop(plan.experts.size(), 0);
            bool any = false;
            for (size_t g = 0; g < plan.experts.size(); ++g) {
                const int e = plan.experts[g];
                if (cache->at.count(e) != 0) continue;             // resident: a GPU hit
                if (impl_->admission_excluded(layer, e)) continue;    // peer/reserved incoming scratch path
                if (fill_room > 0) { --fill_room; continue; }      // the loop below admits this
                const int rows = plan.offsets[g + 1] - plan.offsets[g];
                if (cpu_cap > 0 && refused_routes + rows > cpu_cap) continue;   // over the cap: evict as today
                drop[g] = 1;
                refused_routes += rows;
                for (int r = plan.offsets[g]; r < plan.offsets[g + 1]; ++r)
                    cpu_rows[plan.destinations[(size_t) r]] = 1;
                impl_->cpu_refused += rows;
                any = true;
            }
            if (any) plan.drop_groups(drop);
        }
        const int wave_size = cache ? active : impl_->max_k;
        // The per-wave host sources (the route/matrix arrays the FFN entry points upload from
        // pageable memory) live in a two-bank ring: with STRATA_EXL3_PIPE a wave's FFN may still be
        // in flight while the next wave rebuilds its own bank, so bank (W-2)'s vectors are reused
        // only after wave W-2's wave_done event fires (see the wave loop).
        Impl::WaveHost default_host;   // preserve the baseline's call-local host arrays
        const bool pipeline = device_io && impl_->pipe_stage;
        auto reserve_host = [&](Impl::WaveHost& bank) {
            bank.input_rows.reserve((size_t) tile);
            bank.route_g.reserve((size_t) tile); bank.route_u.reserve((size_t) tile); bank.route_d.reserve((size_t) tile);
        };
        if (pipeline) for (Impl::WaveHost& bank : impl_->wave_host) reserve_host(bank);
        else reserve_host(default_host);
        // STRATA_EXL3_SMALL_M's dispatch test for ONE pass (the hit pass, the miss pass, or the
        // default arm's tile): the measured win set, kept as one transparent threshold rather
        // than a heuristic grid.  Every stay is counted, so the STRATA_EXL3_TIME summary shows
        // the real dispatch coverage, not intent.  Thresholds re-measured for the RDNA grouped
        // twin (strata-exl3-smallm, 2026-10-22): grouped wins EVERY M >= 2 shape down to reuse
        // 1.25 (mix-M2 -8%), and only a pass of pure 1-row groups loses (mix-M1 +8.5%) - so the
        // old tokens >= 3 / reuse >= 1.4 gates (written for the scalar prototype, which LOST
        // the mixed shapes by +21..50%) widen to tokens >= 2 / reuse >= 1.2.
        auto small_m_take = [&](long long routes_pass, int groups_pass, int widest) {
            if (!small_m_on) return false;
            if (tokens < 2) { ++impl_->small_m_stays[0]; return false; }
            if (widest > (impl_->primary_multirow ? impl_->primary_multirow : 6)) { ++impl_->small_m_stays[1]; return false; }
            if (groups_pass < 1 || (double) routes_pass / (double) groups_pass < 1.2) {
                ++impl_->small_m_stays[2]; return false;
            }
            return true;
        };
        const bool pin_plan=impl_->pinned_plan && !device_io && tokens*k<=impl_->pinned_plan_capacity;
        const bool pack_primary = impl_->primary_pack && pin_plan;
        auto indexed_entry=[&](int pass,const Exl3Mat* g,const Exl3Mat* u,const Exl3Mat* d,int rows,
                               const int* row_index,const int* host_rows,uint16_t* output) {
            if (pack_primary) {
                auto& p = impl_->host_plan[pass];
                const size_t bytes = (size_t)rows * sizeof(Exl3Mat);
                auto* base = (uint8_t*)p.allocation;
                std::memcpy(base, g, bytes); std::memcpy(base + bytes, u, bytes);
                std::memcpy(base + 2*bytes, d, bytes);
                std::memcpy(base + 3*bytes, host_rows, (size_t)rows * sizeof(int));
                if (cudaMemcpyAsync(p.device, base, 3*bytes + (size_t)rows*sizeof(int),
                                    cudaMemcpyHostToDevice, st) != cudaSuccess)
                    throw std::runtime_error("exl3: packed primary plan upload failed");
                const Exl3MoeDevicePlan plan{(const Exl3Mat*)p.device,
                    (const Exl3Mat*)(p.device + bytes), (const Exl3Mat*)(p.device + 2*bytes), nullptr, nullptr};
                exl3_moe_rows_indexed_device_plan(g,u,d,rows,batch.x.p,
                    (const int*)(p.device + 3*bytes),output,plan,(void*)st,*impl_->workspace);
                ++impl_->primary_packed_calls; impl_->primary_packed_routes += rows;
                return;
            }
            if(pin_plan) {
                auto& p=impl_->host_plan[pass];const size_t bytes=(size_t)rows*sizeof(Exl3Mat);
                std::memcpy(p.gate,g,bytes);std::memcpy(p.up,u,bytes);std::memcpy(p.down,d,bytes);
                g=p.gate;u=p.up;d=p.down;
            }
            exl3_moe_rows_indexed(g,u,d,rows,batch.x.p,row_index,output,(void*)st,*impl_->workspace);
        };
        auto grouped_entry=[&](int pass,int groups,int rows,const int* row_index,const int* host_rows,uint16_t* output) {
            const Exl3Mat *g=impl_->sm_gg[pass].data(),*u=impl_->sm_gu[pass].data(),*d=impl_->sm_gd[pass].data();
            const int *first=impl_->sm_gfirst[pass].data(),*counts=impl_->sm_grows[pass].data();
            if (pack_primary) {
                auto& p = impl_->host_plan[pass];
                const size_t bytes = (size_t)groups * sizeof(Exl3Mat), table = (size_t)groups * sizeof(int);
                auto* base = (uint8_t*)p.allocation;
                std::memcpy(base,g,bytes); std::memcpy(base+bytes,u,bytes); std::memcpy(base+2*bytes,d,bytes);
                std::memcpy(base+3*bytes,first,table); std::memcpy(base+3*bytes+table,counts,table);
                std::memcpy(base+3*bytes+2*table,host_rows,(size_t)rows*sizeof(int));
                if (cudaMemcpyAsync(p.device,base,3*bytes+2*table+(size_t)rows*sizeof(int),
                                    cudaMemcpyHostToDevice,st) != cudaSuccess)
                    throw std::runtime_error("exl3: packed primary group plan upload failed");
                const Exl3MoeDevicePlan plan{(const Exl3Mat*)p.device,
                    (const Exl3Mat*)(p.device+bytes),(const Exl3Mat*)(p.device+2*bytes),
                    (const int*)(p.device+3*bytes),(const int*)(p.device+3*bytes+table)};
                (void)exl3_moe_rows_grouped_device_plan(g,u,d,first,counts,groups,rows,batch.x.p,
                    (const int*)(p.device+3*bytes+2*table),plan,(void*)st,*impl_->workspace,
                    impl_->primary_multirow!=0,output);
                ++impl_->primary_packed_calls; impl_->primary_packed_routes += rows;
                return;
            }
            if(pin_plan) {
                auto& p=impl_->host_plan[pass];const size_t bytes=(size_t)groups*sizeof(Exl3Mat);
                std::memcpy(p.gate,g,bytes);std::memcpy(p.up,u,bytes);std::memcpy(p.down,d,bytes);
                std::memcpy(p.first,first,(size_t)groups*sizeof(int));std::memcpy(p.rows,counts,(size_t)groups*sizeof(int));
                g=p.gate;u=p.up;d=p.down;first=p.first;counts=p.rows;
            }
            exl3_moe_rows_grouped(g,u,d,first,counts,groups,rows,batch.x.p,row_index,output,
                                   (void*)st,*impl_->workspace,impl_->primary_multirow!=0);
        };
        // wave_arm: the wave's route tables are uploaded once, so the device row/destination buffers
        // must hold a whole wave's routes (a tile holds at most `tile`); reserve the maximum up front
        // so no buffer grows - and frees - while earlier waves' work is still queued.  The FFN
        // scratch gets the same one-shot treatment, via the group-scratch seam so nothing but this
        // arm allocates the group buffers: the indexed GEMV only ever sees counts < 16, and the
        // group entry points reserve their BASE arrays at the full group count (a later, larger
        // tile of a wave must not grow them under earlier queued tiles).  Below 16 no group tile
        // can run at all, so neither group allocation is made.  These blocking allocations run
        // before any tile of this call is queued.
        // wave_arm: one barrier per wave (not per tile).  STRATA_EXL3_PIPE pipelines the waves on
        // top of that granularity - it needs the wave batching, so it implies wave_arm.
        const bool wave_arm = device_io && (impl_->wave_sync || impl_->pipe_stage);
        if (wave_arm) {
            size_t max_wave_routes = 0;
            int max_group_rows = 0;
            for (size_t first = 0; first < plan.experts.size(); first += (size_t) wave_size) {
                const size_t end = std::min(first + (size_t) wave_size, plan.experts.size());
                max_wave_routes = std::max(max_wave_routes, (size_t) (plan.offsets[end] - plan.offsets[first]));
            }
            for (size_t g = 0; g < plan.experts.size(); ++g)
                max_group_rows = std::max(max_group_rows, std::min(tile, plan.offsets[g + 1] - plan.offsets[g]));
            const int ff = impl_->model->linear("model.language_model.layers." + std::to_string(layer) +
                                                ".mlp.experts.0.gate_proj").out();
            impl_->workspace->reserve(std::max(1, std::min(15, max_group_rows)), n, ff, n, true);
            if (max_group_rows >= 16) {
                impl_->workspace->reserve(max_group_rows, n, ff, n, false);
                impl_->workspace->reserve_group_scratch(max_group_rows, n, ff, n, impl_->wmma_group);
            }
            batch.input_rows.reserve(max_wave_routes);
            batch.destinations.reserve(max_wave_routes);
            if (impl_->expert_batch && impl_->wmma_group) {
                impl_->workspace->reserve_expert_batch(8, tile, n, ff, n);
                // At most eight descriptors appended per attempted starting group, including
                // rejected attempts. This upper bound prevents reallocation under queued copies.
                for (auto& b : impl_->wave_host) b.groups.reserve(8 * plan.experts.size());
                default_host.groups.reserve(8 * plan.experts.size());
            }
        }
        // On an exception the local staging vectors above are destroyed during unwinding, before the
        // catch body runs; this drains the compute stream while they are still alive.  A null
        // cudaStream_t is the legacy default stream, not "no stream", so the drain syncs it too.
        // Both runtimes also document pageable H2D copies as consumed before cudaMemcpyAsync returns
        // (CUDA stages pageable sources; ROCm 7.14's hipMemcpyAsync note: "If host or dst are not
        // pinned, the memory copy will be performed synchronously"), so this is defense in depth.
        // The normal path disarms it after the final barrier and adds no sync.
        struct StreamDrain {
            cudaStream_t stream;
            bool armed = true;
            ~StreamDrain() { if (armed) (void) cudaStreamSynchronize(stream); }
        } drain{st};
        // The device-FFN barrier, shared by both arms; the sync itself is unconditional, the wait
        // bookkeeping only runs under STRATA_EXL3_TIME so an uninstrumented build does no extra work.
        const auto sync_compute = [&]() {
            const auto started = timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            const cudaError_t se = cudaStreamSynchronize(st);
            if (timing) {
                ++impl_->sync_count;
                impl_->sync_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
            }
            return se;
        };
        if (timing && pipeline) {
            // Four timestamp events per wave, all created before anything is queued (no allocation
            // under enqueued work); measured after the call-end barrier.
            const size_t want = (plan.experts.size() + (size_t) wave_size - 1) / (size_t) wave_size * 4;
            while (impl_->pipe_ev.size() < want) {
                cudaEvent_t e = nullptr;
                if (cudaEventCreate(&e) != cudaSuccess) throw std::runtime_error("exl3: pipe profiling event failed");
                impl_->pipe_ev.push_back(e);
            }
        }
        for (size_t first = 0, wave = 0; first < plan.experts.size(); first += (size_t) wave_size, ++wave) {
            const size_t end = std::min(first + (size_t) wave_size, plan.experts.size());
            if (impl_->small_m_trace)
                std::fprintf(stderr, "smtrace wave first=%zu end=%zu routes=%d\n",
                             first, end, plan.offsets[end] - plan.offsets[first]);
            // STRATA_EXL3_PIPE: this wave's staging overwrites slots and host banks wave (W-2) last
            // used; its wave_done event (recorded after that wave's final enqueue) is the release.
            // Waves 0 and 1 have no predecessor in flight (every call enters drained - the previous
            // call's barrier, or this arm's own call-end one), and a never-recorded event waits as
            // already complete.
            if (pipeline && cudaEventSynchronize(impl_->wave_done[wave & 1]) != cudaSuccess)
                throw std::runtime_error("exl3: wave bank release failed");
            Impl::WaveHost& bank = pipeline ? impl_->wave_host[wave & 1] : default_host;
            auto& gates = bank.gates; auto& ups = bank.ups; auto& downs = bank.downs;
            auto& route_g = bank.route_g; auto& route_u = bank.route_u; auto& route_d = bank.route_d;
            auto& input_rows = bank.input_rows;
            gates.clear(); ups.clear(); downs.clear(); bank.groups.clear(); impl_->jobs.clear();
            // stage_parallel() joined the previous wave's workers before we get here: every RAM
            // hit has already been memcpy'd into a worker's pinned buffer and every RAM fill has
            // completed. DMA reads those buffers, not the RAM slots; Stage::done guards their reuse.
            // Thus RAM slots may be unpinned now even while the preceding GPU wave is still live.
            // Without pinned staging, the runtime consumes the pageable RAM/mmap source during
            // cudaMemcpyAsync (see the source-lifetime contract below); no RAM source escapes here.
            impl_->ram_pinned.assign((size_t) impl_->ram_slots, 0);
            // Slots referenced by this wave cannot be evicted until its final tile has completed.
            std::vector<uint8_t> pinned(cache ? (size_t) active : 0, 0);
            if (cache && verify_graph_host_active()) {
                // The graph runs these hits concurrently with miss upload/compute. Keep their
                // slots out of the replacement set until this callback has finished staging.
                const auto& rec = impl_->verify_graph->record.host[layer];
                for (int r = 0; r < rec.routes; ++r) if (rec.hit[r]) {
                    const auto it = cache->at.find(rec.ids[r]);
                    if (it == cache->at.end()) throw std::runtime_error("exl3: graph hit slot was evicted");
                    pinned[it->second] = 1;
                }
            }
            int scratch_slot = 0;   // owned-elsewhere groups compute into scratch, never the cache
            // The wave's per-group slot map: each group's slot is released (used event) right after
            // its last tile is queued, so later waves' staging may reuse it early.
            std::vector<int> wave_slots(pipeline ? end - first : 0);
            std::vector<uint8_t> wave_scratch;   // only with an ownership table (default: none)
            if (impl_->own_table != nullptr && pipeline) wave_scratch.assign(end - first, 0);
            // STRATA_EXL3_OVERLAP wants this host-IO wave's groups classified hit/miss.  The
            // classification only READS what the admission loop below already decides; the
            // default arm never allocates the flag vector.
            // Graph hits already run on a separate workspace/stream. Its host complement
            // contains misses only; a too-small/loaned cache's fallback uses the ordinary
            // single pass rather than nesting the older host hit/miss overlap scheduler.
            const bool want_split = impl_->overlap_stage && !device_io && cache != nullptr &&
                                    !verify_graph_host_active();
            std::vector<uint8_t> group_hit(want_split ? end - first : 0, 0);
            Impl::Blocks* wave_blocks = cache != nullptr ? static_cast<Impl::Blocks*>(cache) : &impl_->scratch;
            const auto prof_t2 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            for (size_t group = first; group < end; ++group) {
                const int expert = plan.experts[group];
                if (prof) {
                    const int gr = plan.offsets[group + 1] - plan.offsets[group];
                    ++impl_->prof_row_hist[gr <= 5 ? gr - 1 : gr <= 8 ? 5 : gr <= 15 ? 6 : 7];
                }
                int slot = (int) (group - first);
                Impl::Blocks* blocks = &impl_->scratch;
                const bool own = cache != nullptr && impl_->admission_excluded(layer, expert);
                if (cache && !own) {
                    blocks = cache;
                    const auto hit = cache->at.find(expert);
                    if (hit != cache->at.end()) {
                        slot = hit->second; ++impl_->hits;
                        if (want_split) group_hit[group - first] = 1;
                    }
                    else {
                        ++impl_->misses;
                        if (timing && !cache->evicted.empty())
                            for (const int e : cache->evicted) if (e == expert) { ++impl_->churn_misses; break; }
                        if (cache->filled < active) slot = cache->filled++;
                        else {
                            slot = -1;
                            for (int s = 0; s < active; ++s)
                                if (!pinned[(size_t) s] && (slot < 0 || cache->tick[(size_t) s] < cache->tick[(size_t) slot]))
                                    slot = s;
                            if (slot < 0) throw std::runtime_error("exl3: all expert slots pinned");
                            for (auto it = cache->at.begin(); it != cache->at.end(); ++it)
                                if (it->second == slot) {
                                    if (timing) {
                                        if (cache->evicted.empty()) cache->evicted.assign(16, -1);
                                        cache->evicted[cache->evicted_at++ % cache->evicted.size()] = it->first;
                                        ++impl_->evictions;
                                    }
                                    cache->at.erase(it); break;
                                }
                        }
                        // The sparse HOST complement drains every wave before returning/reusing
                        // slots. It has no in-flight previous-wave reader to wait on. The PIPE
                        // events belong to device-IO prefill; replaying those stale records can
                        // otherwise wait behind the verifier graph that is waiting for us.
                        impl_->prepare_expert(*cache, slot, expert, !skip_owned && !verify_graph_host_active());
                        cache->at[expert] = slot;
                    }
                    pinned[(size_t) slot] = 1;
                    cache->tick[(size_t) slot] = ++cache->clock;
                } else {
                    ++impl_->misses;
                    if (own) {
                        ++impl_->own_excl;
                        slot = scratch_slot++;   // bounded: grow_scratch sized the scratch for a wave
                    }
                    impl_->prepare_expert(impl_->scratch, slot, expert, !skip_owned && !verify_graph_host_active());
                }
                gates.push_back(blocks->gm[(size_t) slot]); ups.push_back(blocks->um[(size_t) slot]);
                downs.push_back(blocks->dm[(size_t) slot]);
                if (pipeline) {
                    wave_slots[group - first] = slot;
                    if (!wave_scratch.empty()) wave_scratch[group - first] = own ? 1 : 0;
                }
            }
            if (prof) {
                impl_->prof_prep_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - prof_t2).count();
                impl_->prof_groups += (long long) (end - first);
                impl_->prof_routes += (long long) (plan.offsets[end] - plan.offsets[first]);
            }
            // STRATA_EXL3_OVERLAP (host-IO waves only): classify this wave into a hit pass and a
            // miss pass.  A wave splits only when it fits one tile (always true for the engine's
            // decode windows) AND both halves are nonempty - an all-hit or all-miss wave, a wave
            // wider than one tile or a cache-less stream keeps the ORIGINAL single-pass order
            // below, byte for byte.  The permuted route table puts the hit routes first, so the
            // two passes partition the wave's existing buffers by offset ([0, nh) and [nh, total))
            // instead of needing their own: no extra device memory, no mid-call drain.
            const int route_base = plan.offsets[first];
            const int route_end = plan.offsets[end];
            const int total_routes = route_end - route_base;
            bool overlap_arm = want_split && total_routes > 0 && total_routes <= tile;
            int nh = 0;
            if (overlap_arm) {
                size_t wg = first;
                for (int r = route_base; r < route_end; ++r) {
                    while (r >= plan.offsets[wg + 1]) ++wg;
                    if (group_hit[wg - first]) ++nh;
                }
                overlap_arm = nh > 0 && nh < total_routes;
            }
            if (want_split && !overlap_arm) ++impl_->ov_fallbacks;
            // STRATA_EXL3_SMALL_M: per-pass group tables for the grouped entry, for the split wave
            // below.  The split is per GROUP (a group's rows all belong to one pass), so each
            // pass's routes are a run of whole groups - exactly the contiguous-slice contract the
            // grouped entry validates.  Tables are 0-based per pass; the two passes use two
            // SEPARATE host sets so the miss pass's tables never overwrite the hit pass's while
            // its pageable uploads are still being consumed.  Declared before both pass blocks.
            int sm_groups[2] = {0, 0}, sm_widest[2] = {0, 0};
            const bool sm_candidate = small_m_on && tokens >= 2;
            if (overlap_arm) {
                input_rows.clear(); route_g.clear(); route_u.clear(); route_d.clear(); bank.ov_dest.clear();
                for (int pass = 0; pass < 2; ++pass) {   // pass 0: the hit routes, pass 1: the misses
                    size_t wg = first;
                    for (int r = route_base; r < route_end; ++r) {
                        while (r >= plan.offsets[wg + 1]) ++wg;
                        if ((group_hit[wg - first] ? 0 : 1) != pass) continue;
                        input_rows.push_back(plan.destinations[(size_t) r] / k);
                        route_g.push_back(gates[wg - first]);
                        route_u.push_back(ups[wg - first]);
                        route_d.push_back(downs[wg - first]);
                        bank.ov_dest.push_back(plan.destinations[(size_t) r]);
                    }
                }
                if (sm_candidate)
                    for (int p = 0; p < 2; ++p) {
                        auto& gf = impl_->sm_gfirst[p]; auto& gr = impl_->sm_grows[p];
                        auto& gg = impl_->sm_gg[p]; auto& gu = impl_->sm_gu[p]; auto& gd = impl_->sm_gd[p];
                        gf.clear(); gr.clear(); gg.clear(); gu.clear(); gd.clear();
                        int prefix = 0, widest = 0;
                        bool valid = true;
                        for (size_t wg = first; wg < end; ++wg) {
                            if ((group_hit[wg - first] ? 0 : 1) != p) continue;
                            const int rows = plan.offsets[wg + 1] - plan.offsets[wg];
                            if (!impl_->append_small_group(p, prefix, rows, gates[wg-first], ups[wg-first],
                                                           downs[wg-first], widest)) { valid = false; break; }
                        }
                        if (!valid || prefix != (p == 0 ? nh : total_routes - nh)) {   // defensive: tables must
                            gf.clear(); gr.clear(); gg.clear(); gu.clear(); gd.clear();   // cover the pass exactly
                        } else { sm_groups[p] = (int) gf.size(); sm_widest[p] = widest; }
                    }
                if (timing)   // the tile loop below would have counted each group once; count here
                    for (size_t wg = first; wg < end; ++wg) {
                        const int rows = plan.offsets[wg + 1] - plan.offsets[wg];
                        ++impl_->group_hist[rows < 16 ? 0 : rows < 24 ? 1 : rows < 64 ? 2 : rows < 256 ? 3 : 4];
                    }
                // Pre-reserve BOTH passes' workspace geometry before anything is queued: the entry
                // points' reserves are grow-only, and a grow FREES buffers the already-queued hit
                // kernels still read.  max(nh, total - nh) covers each pass; it never exceeds the
                // single-pass count the original order reserves, so no extra memory either.
                const int n_in = (int) gates[0].ki * 16, n_ff = (int) gates[0].nj * 16;
                impl_->workspace->reserve(std::max(nh, total_routes - nh), n_in, n_ff, n, true);
            }
            if (timing) {
                for (const auto& job : impl_->jobs) impl_->upload_bytes += job.bytes;
                (void) cudaEventRecord(impl_->profile[0], impl_->work_stream);
                if (pipeline) (void) cudaEventRecord(impl_->pipe_ev[wave * 4 + 0], impl_->work_stream);
            }
            if (overlap_arm) {
                // The hit pass is enqueued BEFORE the staging workers are joined: its GEMV runs on
                // the compute stream while the workers pread/H2D this wave's misses on the staging
                // stream.  It reads only already-resident slots (the admission loop pinned every
                // hit slot before any miss could evict it) and its own [0, nh) buffer partition.
                if (!pack_primary && cudaMemcpyAsync(batch.input_rows.p, input_rows.data(), (size_t) total_routes * sizeof(int),
                                    cudaMemcpyHostToDevice, st) != cudaSuccess)
                    throw std::runtime_error("exl3: overlap route upload failed");
                if (timing) {
                    (void) cudaEventRecord(impl_->profile[2], st);   // spans BOTH passes incl. the
                    if (impl_->ov_ev[0] != nullptr) (void) cudaEventRecord(impl_->ov_ev[0], st);   // staging wait
                }
                const bool sm_hit = small_m_take(nh, sm_groups[0], sm_widest[0]);
                if (impl_->small_m_trace)
                    std::fprintf(stderr, "smtrace overlap-hit nh=%d sm=%d groups=%d widest=%d\n",
                                 nh, (int) sm_hit, sm_groups[0], sm_widest[0]);
                if (sm_hit) {   // STRATA_EXL3_SMALL_M hit pass
                    if (prof) impl_->prof_arm_routes[3] += nh;
                    ++impl_->small_m_calls; impl_->small_m_routes += nh;
                    grouped_entry(0,sm_groups[0],nh,batch.input_rows.p,input_rows.data(),batch.rows.p);
                } else {
                    if (prof) impl_->prof_arm_routes[0] += nh;
                    indexed_entry(0,route_g.data(),route_u.data(),route_d.data(),nh,batch.input_rows.p,input_rows.data(),batch.rows.p);
                }
                const long hit_elements = (long) nh * n;
                f16_to_f32_rows_kernel<<<(unsigned) ((hit_elements + 255) / 256), 256, 0, st>>>(
                    batch.rows.p, batch.output.p, hit_elements);
                if (timing && impl_->ov_ev[1] != nullptr) (void) cudaEventRecord(impl_->ov_ev[1], st);
            }
            {
                const auto prof_t3 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                if (impl_->small_m_trace) std::fprintf(stderr, "smtrace stage-join begin\n");
                impl_->stage_parallel(!device_io);
                if (impl_->small_m_trace) std::fprintf(stderr, "smtrace stage-join done\n");
                if (prof) impl_->prof_join_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - prof_t3).count();
            }
            if (timing) {
                (void) cudaEventRecord(impl_->profile[1], impl_->work_stream);
                if (pipeline) (void) cudaEventRecord(impl_->pipe_ev[wave * 4 + 1], impl_->work_stream);
            }
            if ((!impl_->skip_empty_stage || device_io || !impl_->jobs.empty()) &&
                (cudaEventRecord(impl_->stage_done, impl_->work_stream) != cudaSuccess ||
                 cudaStreamWaitEvent(st, impl_->stage_done, 0) != cudaSuccess))
                throw std::runtime_error("exl3: expert upload event failed");
            // wave_arm: build and upload the wave's whole route table once.  The host vectors and the
            // pageable uploads that read them (this one and the FFN entry points' own matrix-array
            // uploads) must stay unmodified until this wave is done with them: the wave-end sync in
            // the default arm, or the bank's wave_done event under STRATA_EXL3_PIPE (the per-tile
            // loop only takes pointers into them).
            if (wave_arm) {
                input_rows.clear(); route_g.clear(); route_u.clear(); route_d.clear();
                size_t wg = first;
                for (int r = route_base; r < route_end; ++r) {
                    while (r >= plan.offsets[wg + 1]) ++wg;
                    input_rows.push_back(plan.destinations[(size_t) r] / k);
                    route_g.push_back(gates[wg - first]); route_u.push_back(ups[wg - first]); route_d.push_back(downs[wg - first]);
                }
                if (route_end > route_base &&
                    (cudaMemcpyAsync(batch.input_rows.p, input_rows.data(), (size_t) (route_end - route_base) * sizeof(int),
                                     cudaMemcpyHostToDevice, st) != cudaSuccess ||
                     cudaMemcpyAsync(batch.destinations.p, plan.destinations.data() + route_base,
                                     (size_t) (route_end - route_base) * sizeof(int), cudaMemcpyHostToDevice, st) != cudaSuccess))
                    throw std::runtime_error("exl3: route upload failed");
            }
            if (overlap_arm) {
                // The miss pass: ordered after this wave's staging by the stage_done wait above and
                // after the hit kernels by stream order; it writes the [nh, total) partition while
                // the hit pass owns [0, nh) - the shared buffers never see both passes at once.
                const int nm = total_routes - nh;
                const bool sm_miss = small_m_take(nm, sm_groups[1], sm_widest[1]);
                if (impl_->small_m_trace)
                    std::fprintf(stderr, "smtrace overlap-miss nh=%d nm=%d sm=%d groups=%d widest=%d\n",
                                 nh, nm, (int) sm_miss, sm_groups[1], sm_widest[1]);
                if (sm_miss) {   // STRATA_EXL3_SMALL_M miss pass
                    if (prof) impl_->prof_arm_routes[3] += nm;
                    ++impl_->small_m_calls; impl_->small_m_routes += nm;
                    grouped_entry(1,sm_groups[1],nm,batch.input_rows.p+nh,input_rows.data()+nh,batch.rows.p+(size_t)nh*n);
                } else {
                    if (prof) impl_->prof_arm_routes[0] += nm;
                    indexed_entry(1,route_g.data()+nh,route_u.data()+nh,route_d.data()+nh,nm,
                                  batch.input_rows.p+nh,input_rows.data()+nh,batch.rows.p+(size_t)nh*n);
                }
                const long miss_elements = (long) nm * n;
                f16_to_f32_rows_kernel<<<(unsigned) ((miss_elements + 255) / 256), 256, 0, st>>>(
                    batch.rows.p + (size_t) nh * n, batch.output.p + (size_t) nh * n, miss_elements);
                if (timing) (void) cudaEventRecord(impl_->profile[3], st);
                // One D2H of the whole permuted output and one barrier: the original per-tile
                // cadence.  The scatter walks the permutation, so each destination is written
                // exactly once (the hit and miss passes hold disjoint route sets).
                const long total_elements = (long) total_routes * n;
                if (impl_->small_m_trace) std::fprintf(stderr, "smtrace overlap-flush begin\n");
                if (cudaMemcpyAsync(batch.host_output, batch.output.p, (size_t) total_elements * sizeof(float),
                                    cudaMemcpyDeviceToHost, st) != cudaSuccess || cudaStreamSynchronize(st) != cudaSuccess)
                    throw std::runtime_error("exl3: multi FFN execution failed");
                if (impl_->small_m_trace) std::fprintf(stderr, "smtrace overlap-flush done\n");
                if (timing) {   // after the barrier, like the original arm: the events are complete
                    float ms = 0;
                    if (cudaEventElapsedTime(&ms, impl_->profile[2], impl_->profile[3]) == cudaSuccess)
                        impl_->ffn_ms += ms;
                }
                for (int i = 0; i < total_routes; ++i)
                    std::memcpy(out + (size_t) bank.ov_dest[(size_t) i] * n,
                                batch.host_output + (size_t) i * n, (size_t) n * sizeof(float));
                ++impl_->ov_splits;
                // Event-measured overlap evidence (STRATA_EXL3_TIME only; never inferred from the
                // env): the hit pass' span on the compute stream against this wave's staging span
                // on the work stream, both elapsed from profile[0], recorded just before the
                // staging workers were joined.  Cross-stream elapsed times share the device clock.
                if (timing && impl_->ov_ev[0] != nullptr && impl_->ov_ev[1] != nullptr) {
                    auto since0 = [&](cudaEvent_t e) {
                        float ms = -1;
                        return cudaEventElapsedTime(&ms, impl_->profile[0], e) == cudaSuccess ? (double) ms : -1.0;
                    };
                    const double a0 = since0(impl_->ov_ev[0]), a1 = since0(impl_->ov_ev[1]),
                                 b1 = since0(impl_->profile[1]);
                    if (a0 >= 0 && a1 >= 0 && b1 >= 0) {
                        impl_->ov_hit_ms += a1 - a0;
                        impl_->ov_stage_ms += b1;
                        impl_->ov_overlap_ms += std::max(0.0, std::min(a1, b1) - std::max(a0, 0.0));
                    }
                }
            } else {
            // (the original single-pass tile loop; overlap_arm took the split path above)
            size_t group = first;
            if (timing && wave_arm) {
                (void) cudaEventRecord(impl_->profile[2], st);
                if (pipeline) (void) cudaEventRecord(impl_->pipe_ev[wave * 4 + 2], st);
            }
            for (int route = route_base; route < route_end;) {
                while (route >= plan.offsets[group + 1]) ++group;
                if (timing && route == plan.offsets[group]) {
                    const int rows = plan.offsets[group+1]-route;
                    ++impl_->group_hist[rows<16?0:rows<24?1:rows<64?2:rows<256?3:4];
                }
                if (wave_arm && impl_->expert_batch && impl_->wmma_group && route == plan.offsets[group]) {
                    const size_t begin = bank.groups.size();
                    int total = 0;
                    for (size_t g = group; g < end && g < group+8; ++g) {
                        const int rows = plan.offsets[g + 1] - plan.offsets[g];
                        if (rows < 16) {
                            if (timing) ++impl_->split_stops[0];
                            break;
                        }
                        if (rows > tile - total) {
                            if (timing) ++impl_->split_stops[1];
                            break;
                        }
                        if ((rows >= 24) != (plan.offsets[group + 1] - route >= 24)) {
                            if (timing) ++impl_->split_stops[2];
                            break;
                        }
                        bank.groups.push_back({gates[g - first], ups[g - first], downs[g - first], total, rows});
                        total += rows;
                    }
                    const int groups = (int) (bank.groups.size() - begin);
                    if (timing && groups == 8 && group + 8 < end) ++impl_->split_stops[3];
                    if (groups >= 2 && exl3_moe_expert_batch(bank.groups.data() + begin, groups, batch.x.p,
                            batch.input_rows.p + (route - route_base), batch.destinations.p + (route - route_base),
                            out, (void*) st, *impl_->workspace)) {
                        if (prof) impl_->prof_arm_routes[2] += total;
                        if (timing) {
                            impl_->batch_groups += groups;
                            ++impl_->batch_calls;
                            impl_->ffn_launches += 10;   // known dispatched chain, not profiler API tracing
                            for (int g = 1; g < groups; ++g) {
                                const int rows = bank.groups[begin + g].rows;
                                ++impl_->group_hist[rows < 24 ? 1 : rows < 64 ? 2 : rows < 256 ? 3 : 4];
                            }
                        }
                        // Bounded subgroup releases all readers here, rather than delaying the entire wave.
                        for (int g = 0; pipeline && g < groups; ++g)
                            if (cudaEventRecord((!wave_scratch.empty() && wave_scratch[group + g - first]
                                                     ? &impl_->scratch : wave_blocks)
                                                    ->used[(size_t) wave_slots[group + g - first]], st) != cudaSuccess)
                                throw std::runtime_error("exl3: batched slot release failed");
                        route += total;
                        group += (size_t) groups - 1;
                        continue;
                    }
                    if (timing) {
                        if (groups >= 2) ++impl_->batch_rejects;
                        else ++impl_->standalone_groups;
                    }
                }
                const int count = std::min(tile, (device_io ? plan.offsets[group + 1] : route_end) - route);
                const size_t tile_off = wave_arm ? (size_t) (route - route_base) : 0;
                // The group containing this tile's FIRST route.  The input_rows build below
                // ADVANCES `group` to the tile's last group, so capture the start here.
                const size_t tile_group0 = group;
                if (!wave_arm) {
                    input_rows.clear(); route_g.clear(); route_u.clear(); route_d.clear();
                    for (int r = route; r < route + count; ++r) {
                        while (r >= plan.offsets[group + 1]) ++group;
                        const size_t local = group - first;
                        input_rows.push_back(plan.destinations[(size_t) r] / k);
                        route_g.push_back(gates[local]); route_u.push_back(ups[local]); route_d.push_back(downs[local]);
                    }
                    if (!pack_primary && cudaMemcpyAsync(batch.input_rows.p, input_rows.data(), (size_t) count * sizeof(int),
                                        cudaMemcpyHostToDevice, st) != cudaSuccess)
                        throw std::runtime_error("exl3: route upload failed");
                }
                const Exl3Mat* tile_g = route_g.data() + tile_off;
                const Exl3Mat* tile_u = route_u.data() + tile_off;
                const Exl3Mat* tile_d = route_d.data() + tile_off;
                const int* tile_rows = batch.input_rows.p + tile_off;
                const long elements = (long) count * n;
                if (timing && !wave_arm) (void) cudaEventRecord(impl_->profile[2], st);
                if (device_io && count >= 16) {
                    if (prof) impl_->prof_arm_routes[1] += count;   // group arm (wmma|gemm)
                    // STRATA_EXL3_WMMA=1: the gfx11 matrix-core group FFN (trellis decoded into the
                    // WMMA fragments, no FP16 weight materialized) against the reconstruct+hipBLAS path.
                    // Counts below 16 stay on the indexed GEMV (docs/EXL3-HIP-BENCH.md, A/B #4).
                    if (wave_arm && !impl_->workspace->group_tile_fits(count, tile_g[0].ki * 16, tile_g[0].nj * 16,
                                                                      n, impl_->wmma_group))
                        // Allocation invariant of this arm: the pre-wave reserves cover every buffer
                        // the group entry points touch, so none may grow - and free - under the tiles
                        // already queued.  A violation is a scheduling bug, never a workload condition.
                        throw std::runtime_error("exl3: wave-arm workspace would grow under queued tiles");
                    if (!(impl_->wmma_group &&
                          exl3_moe_group_wmma(tile_g[0], tile_u[0], tile_d[0], count, batch.x.p,
                                              tile_rows, batch.output.p, (void*) st, *impl_->workspace)))
                        exl3_moe_group_gemm(tile_g[0], tile_u[0], tile_d[0], count, batch.x.p,
                                            tile_rows, batch.output.p, (void*) st, *impl_->workspace);
                } else {
                    // STRATA_EXL3_SMALL_M (host-IO decode windows only; the device arms above are
                    // untouched): the grouped arm needs the tile to be a run of WHOLE groups - it
                    // must start on a group boundary and end on one - so a tile that cuts a group
                    // in two keeps the indexed GEMV.  The host-IO arm syncs per tile, so rebuilding
                    // the bank's pass-0 tables here is never under queued work from an earlier tile.
                    bool sm_tile = false;
                    if (impl_->small_m_trace)
                        std::fprintf(stderr, "smtrace tile route=%d count=%d group0=%d\n",
                                     route, count, (int) (route == plan.offsets[tile_group0]));
                    if (small_m_on && !device_io && route == plan.offsets[tile_group0]) {
                        auto& gf = impl_->sm_gfirst[0]; auto& gr = impl_->sm_grows[0];
                        auto& gg = impl_->sm_gg[0]; auto& gu = impl_->sm_gu[0]; auto& gd = impl_->sm_gd[0];
                        gf.clear(); gr.clear(); gg.clear(); gu.clear(); gd.clear();
                        int prefix = 0, widest = 0;
                        bool valid = true;
                        size_t g = tile_group0;
                        while (g < end && prefix + (plan.offsets[g + 1] - plan.offsets[g]) <= count) {
                            const int rows = plan.offsets[g + 1] - plan.offsets[g];
                            if (!impl_->append_small_group(0, prefix, rows, gates[g-first], ups[g-first],
                                                           downs[g-first], widest)) { valid = false; break; }
                            ++g;
                        }
                        if (valid && prefix == count)
                            sm_tile = small_m_take(count, (int) gf.size(), widest);
                        else {   // the tile ends inside a group: not a whole-group run
                            gf.clear(); gr.clear(); gg.clear(); gu.clear(); gd.clear();
                            ++impl_->small_m_stays[1];
                        }
                    }
                    if (sm_tile) {
                        if (impl_->small_m_trace)
                            std::fprintf(stderr, "smtrace tile-grouped count=%d groups=%zu\n",
                                         count, impl_->sm_grows[0].size());
                        if (prof) impl_->prof_arm_routes[3] += count;
                        ++impl_->small_m_calls; impl_->small_m_routes += count;
                        grouped_entry(0,(int)impl_->sm_grows[0].size(),count,tile_rows,input_rows.data()+tile_off,batch.rows.p);
                    } else {
                        if (prof) impl_->prof_arm_routes[0] += count;
                        indexed_entry(0,tile_g,tile_u,tile_d,count,tile_rows,input_rows.data()+tile_off,batch.rows.p);
                    }
                    f16_to_f32_rows_kernel<<<(unsigned) ((elements + 255) / 256), 256, 0, st>>>(batch.rows.p, batch.output.p, elements);
                }
                if (timing && !wave_arm) (void) cudaEventRecord(impl_->profile[3], st);
                if (device_io) {
                    if (!wave_arm &&
                        cudaMemcpyAsync(batch.destinations.p, plan.destinations.data() + route, (size_t) count * sizeof(int),
                                        cudaMemcpyHostToDevice, st) != cudaSuccess)
                        throw std::runtime_error("exl3: destination upload failed");
                    scatter_route_output<<<(unsigned) ((elements + 255) / 256), 256, 0, st>>>(
                        batch.output.p, batch.destinations.p + tile_off, out, count, n);
                    if (!wave_arm && sync_compute() != cudaSuccess)
                        throw std::runtime_error("exl3: device FFN failed");
                } else {
                if (impl_->small_m_trace) std::fprintf(stderr, "smtrace tile-flush begin\n");
                if (cudaMemcpyAsync(batch.host_output, batch.output.p, (size_t) elements * sizeof(float),
                                    cudaMemcpyDeviceToHost, st) != cudaSuccess || cudaStreamSynchronize(st) != cudaSuccess)
                    throw std::runtime_error("exl3: multi FFN execution failed");
                if (impl_->small_m_trace) std::fprintf(stderr, "smtrace tile-flush done\n");
                for (int r = 0; r < count; ++r)
                    std::memcpy(out + (size_t) plan.destinations[(size_t) route + r] * n,
                                batch.host_output + (size_t) r * n, (size_t) n * sizeof(float));
                }
                if (timing && !wave_arm) {
                    float ms = 0;
                    if (cudaEventElapsedTime(&ms, impl_->profile[2], impl_->profile[3]) == cudaSuccess) impl_->ffn_ms += ms;
                }
                route += count;
                // STRATA_EXL3_PIPE: the group's last reader is now queued - record the slot's `used`
                // event (the native prefill ring's release) so the next wave's staging may overwrite
                // it while the rest of this wave's tiles still run.  Later records of the same slot
                // supersede this one for later waits; a wait enqueued before this record captured
                // the previous one (the same snapshot rule the native ring relies on).
                if (pipeline && route >= plan.offsets[group + 1] &&
                    cudaEventRecord((!wave_scratch.empty() && wave_scratch[group - first] ? &impl_->scratch : wave_blocks)
                                        ->used[(size_t) wave_slots[group - first]], st) != cudaSuccess)
                    throw std::runtime_error("exl3: slot last-reader event failed");
            }
            }   // end of the original single-pass tile loop (the overlap_arm else)
            if (wave_arm) {
                if (timing) {
                    (void) cudaEventRecord(impl_->profile[3], st);
                    if (pipeline) (void) cudaEventRecord(impl_->pipe_ev[wave * 4 + 3], st);
                }
                if (pipeline) {
                    // STRATA_EXL3_PIPE: NO wave-end barrier - wave W+1's reads/uploads run on the
                    // staging stream behind this wave's FFN, into slots released above.  wave_done
                    // closes this wave's host bank (its pageable uploads) for reuse at W+2.
                    if (cudaEventRecord(impl_->wave_done[wave & 1], st) != cudaSuccess)
                        throw std::runtime_error("exl3: wave event failed");
                } else {
                    // One barrier per wave, the same granularity the slot pinning already assumes:
                    // the next wave's prepare/staging may overwrite expert slots and route tables
                    // this wave's queued kernels and pageable uploads are still using.
                    const cudaError_t se = sync_compute();
                    if (se != cudaSuccess) throw std::runtime_error("exl3: device FFN failed");
                }
                if (timing && !pipeline) {
                    // This interval spans the first tile's dispatch through the last tile's scatter,
                    // host enqueue gaps included - NOT comparable to the default arm's per-tile FFN
                    // kernel sums above.  The pipeline arm accumulates the same interval at the
                    // call-end barrier instead (mid-call the events are not yet complete).
                    float ms = 0;
                    if (cudaEventElapsedTime(&ms, impl_->profile[2], impl_->profile[3]) == cudaSuccess) impl_->ffn_ms += ms;
                }
            }
            if (timing && !pipeline) {
                float ms = 0;
                if (cudaEventElapsedTime(&ms, impl_->profile[0], impl_->profile[1]) == cudaSuccess) impl_->up_ms += ms;
            }
        }
        if (pipeline) {
            // The pipelined arm's one barrier per CALL (the default arm barriers per wave): after
            // it every upload of the call has been consumed by its tiles, so the host banks and the
            // slots are free for the next call (which enters drained, exactly as before).
            if (sync_compute() != cudaSuccess) throw std::runtime_error("exl3: device FFN failed");
            if (cudaStreamSynchronize(impl_->work_stream) != cudaSuccess)
                throw std::runtime_error("exl3: staging stream drain failed");
        }
        if (timing && pipeline && !plan.experts.empty()) {
            // Event-measured copy/compute overlap of this call, not inferred: per wave, the staging
            // interval [0,1] on the staging stream against the PREVIOUS wave's FFN interval [2,3] on
            // the compute stream (the pairing the pipeline creates).
            const size_t waves = (plan.experts.size() + (size_t) wave_size - 1) / (size_t) wave_size;
            auto at = [&](size_t idx) {
                float ms = 0;
                return cudaEventElapsedTime(&ms, impl_->pipe_ev[0], impl_->pipe_ev[idx]) == cudaSuccess ? (double) ms
                                                                                                        : -1.0;
            };
            double stagems = 0, ffnms = 0, ovms = 0;
            for (size_t w = 0; w < waves; ++w) {
                const double s0 = at(w * 4 + 0), s1 = at(w * 4 + 1);
                const double c0 = at(w * 4 + 2), c1 = at(w * 4 + 3);
                if (s0 < 0 || s1 < 0 || c0 < 0 || c1 < 0) continue;
                stagems += s1 - s0;
                ffnms += c1 - c0;
                if (w > 0) {
                    const double p0 = at((w - 1) * 4 + 2), p1 = at((w - 1) * 4 + 3);
                    if (p0 >= 0 && p1 >= 0) ovms += std::max(0.0, std::min(s1, p1) - std::max(s0, p0));
                }
            }
            impl_->pipe_stage_ms += stagems;
            impl_->pipe_ffn_ms += ffnms;
            impl_->pipe_overlap_ms += ovms;
            impl_->pipe_waves += (long long) waves;
            // Same interval semantics as the default arm's upload/FFN counters (staging span on the
            // staging stream; wave dispatch-through-last-scatter on the compute stream), so the two
            // arms' summary lines stay comparable.
            impl_->up_ms += stagems;
            impl_->ffn_ms += ffnms;
        }
        if (timing) {
            impl_->call_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - call_started).count();
            if (layer + 1 == impl_->layer_hi) {
                std::fprintf(stderr, "exl3 CUDA%d [%d,%d) rows=%d: pool %.2f ms, upload %.2f ms (%.2f MiB), FFN %.2f ms, "
                                 "syncs %lld (%.2f ms), cache %lld/%lld, evict %lld (churn %lld), cpu-refused %lld\n",
                    impl_->device, impl_->layer_lo, impl_->layer_hi, tokens, impl_->call_ms, impl_->up_ms,
                    (double) impl_->upload_bytes / (1 << 20), impl_->ffn_ms, impl_->sync_count, impl_->sync_ms,
                    impl_->hits, impl_->hits + impl_->misses, impl_->evictions, impl_->churn_misses,
                    impl_->cpu_refused);
                if (impl_->pipe_waves > 0)
                    std::fprintf(stderr, "exl3 pipe [%d,%d) rows=%d: %lld waves, staging %.2f ms, FFN %.2f ms, "
                                         "staging-overlapping-prior-FFN %.2f ms\n",
                        impl_->layer_lo, impl_->layer_hi, tokens, impl_->pipe_waves, impl_->pipe_stage_ms,
                        impl_->pipe_ffn_ms, impl_->pipe_overlap_ms);
                if (impl_->pack_stage) {
                    std::fprintf(stderr, "exl3 packed stage: %lld experts uploaded as one DMA each\n", impl_->packed_jobs);
                    impl_->packed_jobs = 0;
                }
                if (impl_->ov_splits > 0 || impl_->ov_fallbacks > 0)
                    std::fprintf(stderr, "exl3 overlap [%d,%d) rows=%d: %lld split waves, %lld fallbacks; "
                                         "hit pass %.2f ms, staging %.2f ms, hit-during-staging %.2f ms "
                                         "(the FFN counter above spans both passes incl. the staging wait)\n",
                        impl_->layer_lo, impl_->layer_hi, tokens, impl_->ov_splits, impl_->ov_fallbacks,
                        impl_->ov_hit_ms, impl_->ov_stage_ms, impl_->ov_overlap_ms);
                if (impl_->small_m_calls > 0 || impl_->small_m_stays[0] != 0 || impl_->small_m_stays[1] != 0 ||
                    impl_->small_m_stays[2] != 0)
                    std::fprintf(stderr, "exl3 smallm [%d,%d) rows=%d: grouped calls=%lld routes=%lld; "
                                         "stays on the indexed GEMV window<2:%lld group>6-or-cut:%lld reuse<1.2:%lld\n",
                        impl_->layer_lo, impl_->layer_hi, tokens, impl_->small_m_calls, impl_->small_m_routes,
                        impl_->small_m_stays[0], impl_->small_m_stays[1], impl_->small_m_stays[2]);
                impl_->small_m_calls = impl_->small_m_routes = 0;
                std::fill(std::begin(impl_->small_m_stays), std::end(impl_->small_m_stays), 0);
                impl_->call_ms = impl_->up_ms = impl_->ffn_ms = 0;
                impl_->ov_hit_ms = impl_->ov_stage_ms = impl_->ov_overlap_ms = 0;
                impl_->sync_count = 0; impl_->sync_ms = 0;
                impl_->hits = impl_->misses = 0; impl_->upload_bytes = 0;
                impl_->evictions = impl_->churn_misses = 0;
                impl_->cpu_refused = 0;
                std::fprintf(stderr,"exl3 groups: <16=%lld 16..23=%lld 24..63=%lld 64..255=%lld >=256=%lld; batched=%lld subgroups=%lld (10 dispatches/subgroup, grid z<=8)\n",
                    impl_->group_hist[0],impl_->group_hist[1],impl_->group_hist[2],impl_->group_hist[3],impl_->group_hist[4],impl_->batch_groups,impl_->batch_calls);
                std::fill(std::begin(impl_->group_hist),std::end(impl_->group_hist),0);
                if (device_io) std::fprintf(stderr,"exl3 workspace requested bytes=%zu (cache bytes via expert stream separately)\n",impl_->workspace->allocated_bytes());
                if (device_io) std::fprintf(stderr,"exl3 batch dispatch: gather=%lld wmma=%lld postscale=%lld silu=%lld (total=%lld); standalone=%lld rejected=%lld; stops small=%lld routes=%lld rowtiles=%lld cap8=%lld\n",
                    3*impl_->batch_calls,3*impl_->batch_calls,3*impl_->batch_calls,impl_->batch_calls,impl_->ffn_launches,
                    impl_->standalone_groups,impl_->batch_rejects,impl_->split_stops[0],impl_->split_stops[1],impl_->split_stops[2],impl_->split_stops[3]);
                impl_->batch_groups=impl_->batch_calls=impl_->ffn_launches=impl_->standalone_groups=impl_->batch_rejects=0;
                std::fill(std::begin(impl_->split_stops),std::end(impl_->split_stops),0);
                impl_->pipe_stage_ms = impl_->pipe_ffn_ms = impl_->pipe_overlap_ms = 0;
                impl_->pipe_waves = 0;
            }
        }
        if (prof && layer + 1 == impl_->layer_hi) {   // the window's totals over its per-layer calls
            std::fprintf(stderr,
                "exl3 prof [%d,%d) M=%d routes=%lld groups=%lld (groups==distinct experts, expert-major plan; "
                "route reuse %.2fx): per-expert route rows 1:%lld 2:%lld 3:%lld 4:%lld 5:%lld 6-8:%lld 9-15:%lld "
                ">=16:%lld; arm routes indexed:%lld group:%lld batch:%lld grouped-smallm:%lld; host elapsed "
                "plan %.3f prep %.3f join %.3f ms\n",
                impl_->layer_lo, impl_->layer_hi, tokens,
                (long long) impl_->prof_routes, (long long) impl_->prof_groups,
                impl_->prof_groups > 0 ? (double) impl_->prof_routes / (double) impl_->prof_groups : 0.0,
                (long long) impl_->prof_row_hist[0], (long long) impl_->prof_row_hist[1],
                (long long) impl_->prof_row_hist[2], (long long) impl_->prof_row_hist[3],
                (long long) impl_->prof_row_hist[4], (long long) impl_->prof_row_hist[5],
                (long long) impl_->prof_row_hist[6], (long long) impl_->prof_row_hist[7],
                (long long) impl_->prof_arm_routes[0], (long long) impl_->prof_arm_routes[1],
                (long long) impl_->prof_arm_routes[2], (long long) impl_->prof_arm_routes[3],
                impl_->prof_plan_ms, impl_->prof_prep_ms, impl_->prof_join_ms);
            std::fprintf(stderr, "exl3 prof staging: jobs0:%lld 1:%lld 2:%lld 3:%lld 4:%lld 5..8:%lld >8:%lld\n",
                impl_->prof_stage_jobs[0], impl_->prof_stage_jobs[1], impl_->prof_stage_jobs[2],
                impl_->prof_stage_jobs[3], impl_->prof_stage_jobs[4], impl_->prof_stage_jobs[5], impl_->prof_stage_jobs[6]);
            std::fill(std::begin(impl_->prof_stage_jobs), std::end(impl_->prof_stage_jobs), 0);
            impl_->prof_routes = impl_->prof_groups = 0;
            std::fill(std::begin(impl_->prof_row_hist), std::end(impl_->prof_row_hist), 0);
            std::fill(std::begin(impl_->prof_arm_routes), std::end(impl_->prof_arm_routes), 0);
            impl_->prof_plan_ms = impl_->prof_prep_ms = impl_->prof_join_ms = 0;
        }
        drain.armed = false;   // the last barrier (per wave, or the pipeline's per call) has drained the stream
        // CPU-tier rows were never scattered (their groups left the plan) and are NOT written
        // here: the caller owns them (it typically started its CPU tier on them BEFORE this
        // call - zeroing would race its writes)
        return true;
    } catch (const std::exception& e) {
        // A failed window is not reusable: uploads may have replaced only part of a cache slot.
        (void) cudaStreamSynchronize(st);
        (void) cudaStreamSynchronize(impl_->work_stream);
        for (auto& cache : impl_->layers) { cache.at.clear(); cache.filled = 0; }
        for (auto& cache : impl_->ram_layers) { cache.at.clear(); cache.filled = 0; }
        std::fprintf(stderr, "exl3 multi(layer=%d): %s\n", layer, e.what());
        return false;
    }
}

}  // namespace strata::kernels
