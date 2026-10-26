// src/kernels/exl3_transport_bench.cpp - NON-PRODUCTION diagnostic harness (decode-isolation).
//
// Isolates, on the REAL 3bpw model's miss geometry, two costs the engine profile lumps together:
//
//  A. The staging span (the ~18.5 ms "upload" interval of a warm decode window) decomposed into
//     its parts, matched arms on the same experts/bytes:
//       (a-current) the engine worker pattern: pread(page cache) -> per-worker 2 pinned buffers
//                   with done-events -> ONE H2D enqueue per plane under the api mutex (15 workers
//                   by default; a 1-worker variant exposes scheduling);
//       (b-pinned)  the same 9-plane enqueue pattern from a preloaded PINNED source (no pread,
//                   no workers) - isolates DMA+enqueue from the read side;
//       (b-one-big) one contiguous H2D of the same bytes - the bandwidth UPPER BOUND, not an
//                   engine prediction;
//       (c-packed)  per-expert packing into one contiguous pinned buffer (pack CPU cost timed
//                   SEPARATELY) + one H2D per expert - the fragmentation delta with the pack
//                   bill accounted.  No new checkpoint format, no RAM shadow: MiBs only.
//  B. The hit/miss split's pure ISSUE cost with NO miss reads: device-resident weights, the same
//     math, one all-routes indexed call vs two offset-partitioned calls (T=1: 9+1, T=4: 35+5),
//     plus the existing test-only slot-table primitive as a READONLY probe of a prebuilt-device-
//     descriptor path.  Outputs are compared byte-for-byte (the f16 rows both arms write); the
//     workspace is pre-reserved outside every timed loop; no allocation inside any timed region.
//     The engine's f16->f32 conversion launch is NOT timed here (common to both arms up to one
//     extra elementwise launch in the two-pass arm) - the single-vs-two delta is the measurement.
//
// Cost definitions (printed with every line): "wall" is host steady_clock around the measured
// region; "cpu_*_ms" are SUMS across threads (CPU work, never wall); "dma_span_ms"/"gpu_span_ms"
// are GPU event intervals, elapsed only after the stream synced.  Events are created and
// destroyed; source buffer lifetimes span all timed regions.  Microbench evidence, NOT engine
// gains.  Default path of the engine has no new timers - this file is a standalone tool.
//
// Usage: exl3_transport_bench <model_dir> [device=0] [experts=54] [layer=0] [reps=20]
//   experts defaults to 54: one warm tuned-config window's distinct misses (95.9 MiB / 1.77 MiB).
#include "strata/artifact/exl3_model.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/exl3_experts.hpp"

#include <cuda_runtime.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using strata::Exl3Linear;
using strata::Exl3Model;
using strata::kernels::Exl3Mat;
using strata::kernels::Exl3MoeWorkspace;
using Clock = std::chrono::steady_clock;

namespace {

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Plane { std::string path; size_t off = 0, bytes = 0; };
struct ExpertPlanes { Plane p[9]; size_t bytes = 0; };

void locate_expert(const Exl3Model& m, int layer, int expert, ExpertPlanes& out) {
    const std::string base = "model.language_model.layers." + std::to_string(layer) + ".mlp.experts." +
                             std::to_string(expert) + ".";
    const char* proj[3] = {"gate_proj", "up_proj", "down_proj"};
    const char* plane[3] = {".trellis", ".suh", ".svh"};
    out.bytes = 0;
    for (int i = 0; i < 3; ++i)
        for (int p = 0; p < 3; ++p) {
            Plane& pl = out.p[(size_t) i * 3 + p];
            if (!m.locate(base + proj[i] + plane[p], pl.path, pl.off, pl.bytes))
                throw std::runtime_error("exl3 transport bench: cannot locate " + base + proj[i] + plane[p]);
            out.bytes += pl.bytes;
        }
}

// Arm (a): faithful to the engine's Impl::stage_one file-read path and worker_loop: pread into
// one of two pinned buffers (a done-event guards reuse), then ONE H2D enqueue per plane under
// the api mutex.  No RAM-tier side effect (the engine's decode path has no RAM tier here).
struct WorkerPool {
    struct Shared {
        const std::vector<ExpertPlanes>* experts = nullptr;
        uint8_t* dst = nullptr;
        std::vector<std::pair<size_t, size_t>> jobs;   // (expert, plane), engine job-list order
        std::atomic<size_t> job_at{0};
        std::atomic<uint64_t> done{0};
        std::atomic<uint64_t> gen{0};
        std::atomic<bool> quit{false};
        std::mutex api_mu;
        std::mutex mu;
        std::condition_variable cv;
        std::atomic<double> cpu_pread_ms{0.0};      // SUM across threads: CPU work, not wall
        std::atomic<double> cpu_enqueue_ms{0.0};    // SUM: in-lock enqueue work
        std::atomic<double> cpu_lock_wait_ms{0.0};  // SUM: waiting to enter the api mutex
        cudaStream_t stream = nullptr;
    } sh;
    struct Buf {
        uint8_t* b[2] = {nullptr, nullptr};
        cudaEvent_t done[2] = {nullptr, nullptr};
        int turn = 0;
    };
    std::vector<Buf> bufs;
    std::vector<std::thread> threads;
    std::map<std::string, int> fd_cache;

    int open_fd(const std::string& path) {
        auto it = fd_cache.find(path);
        if (it != fd_cache.end()) return it->second;
        const int fd = ::open(path.c_str(), O_RDONLY);
        fd_cache[path] = fd;
        return fd;
    }

    void start(int workers, size_t max_plane_bytes, cudaStream_t stream,
               const std::vector<ExpertPlanes>* experts, uint8_t* dst) {
        sh.stream = stream;
        sh.experts = experts;
        sh.dst = dst;
        bufs.resize((size_t) workers);
        for (Buf& b : bufs) {
            if (cudaMallocHost((void**) &b.b[0], max_plane_bytes) != cudaSuccess ||
                cudaMallocHost((void**) &b.b[1], max_plane_bytes) != cudaSuccess)
                throw std::runtime_error("transport bench: pinned worker buffer failed");
            for (int i = 0; i < 2; ++i)
                if (cudaEventCreateWithFlags(&b.done[i], cudaEventDisableTiming) != cudaSuccess)
                    throw std::runtime_error("transport bench: worker event failed");
        }
        for (int w = 0; w < workers; ++w) threads.emplace_back([this, w] { loop((size_t) w); });
    }

    void loop(size_t wid) {
        Buf& b = bufs[wid];
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(sh.mu);
                sh.cv.wait(lk, [&] { return sh.quit.load() || sh.gen.load() != seen; });
                if (sh.quit.load()) return;
                seen = sh.gen.load();
            }
            for (;;) {
                const size_t i = sh.job_at.fetch_add(1, std::memory_order_relaxed);
                if (i >= sh.jobs.size()) break;
                const auto [e, p] = sh.jobs[i];
                const ExpertPlanes& ep = sh.experts->at(e);
                const Plane& pl = ep.p[p];
                size_t at = 0;
                for (int q = 0; q < (int) p; ++q) at += ep.p[(size_t) q].bytes;
                uint8_t* d = sh.dst + e * ep.bytes + at;
                const int bu = b.turn;
                b.turn ^= 1;
                if (cudaEventSynchronize(b.done[bu]) != cudaSuccess) return;
                const Clock::time_point t0 = Clock::now();
                if (::pread(open_fd(pl.path), b.b[bu], pl.bytes, (off_t) pl.off) != (ssize_t) pl.bytes) return;
                sh.cpu_pread_ms.fetch_add(ms_between(t0, Clock::now()), std::memory_order_relaxed);
                const Clock::time_point l0 = Clock::now();
                const std::lock_guard<std::mutex> lk(sh.api_mu);   // ONE enqueue at a time (engine rule)
                sh.cpu_lock_wait_ms.fetch_add(ms_between(l0, Clock::now()), std::memory_order_relaxed);
                if (cudaMemcpyAsync(d, b.b[bu], pl.bytes, cudaMemcpyHostToDevice, sh.stream) != cudaSuccess)
                    return;
                if (cudaEventRecord(b.done[bu], sh.stream) != cudaSuccess) return;
                sh.cpu_enqueue_ms.fetch_add(ms_between(l0, Clock::now()), std::memory_order_relaxed);
            }
            sh.done.fetch_add(1, std::memory_order_release);
        }
    }

    void run(const std::vector<std::pair<size_t, size_t>>& jobs) {
        sh.jobs = jobs;
        sh.job_at.store(0, std::memory_order_relaxed);
        sh.done.store(0, std::memory_order_relaxed);
        sh.gen.fetch_add(1, std::memory_order_release);
        sh.cv.notify_all();
        while (sh.done.load(std::memory_order_acquire) < (uint64_t) threads.size())
            std::this_thread::yield();
    }

    ~WorkerPool() {
        { const std::lock_guard<std::mutex> lk(sh.mu); sh.quit.store(true); }
        sh.cv.notify_all();
        for (std::thread& t : threads) if (t.joinable()) t.join();
        for (Buf& b : bufs)
            for (int i = 0; i < 2; ++i) {
                if (b.b[i]) (void) cudaFreeHost(b.b[i]);
                if (b.done[i]) (void) cudaEventDestroy(b.done[i]);
            }
        for (auto& kv : fd_cache) if (kv.second >= 0) ::close(kv.second);
    }
};

double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

}  // namespace

// The doorbell-style PCIe poll lives in exl3_transport_spin.cu (kernel launches need the GPU
// language path; this .cpp compiles as host C++).
void exl3_transport_spin_poll(void* device_flag, unsigned want, void* stream);

// ---------------- the read-path fixture: warm vs (mincore-verified) cold, root-free ----------------
//
// A self-created scratch file ONLY (deterministic non-zero bytes, fsync'd, in the caller's
// artifact dir): posix_fadvise(DONTNEED) drops THAT file's pages - never the model's, never the
// global cache.  mincore verifies residency before every arm/trial; if eviction is incomplete the
// result is printed UNKNOWN, not cold.  Both arms read the exact same ranges (every plane of
// every expert); the mmap-copy arm keeps the mapping alive until the copies drain (the pageable
// semantics the engine's NO_PREAD arm rides).  /proc/self/io read_bytes and getrusage major-fault
// deltas evidence whether reads actually hit disk.  EXPLICIT LIMIT: this controlled fixture does
// NOT establish real-model cold TG - it measures the two read arms under controlled residency.
size_t read_io_bytes() {
    std::FILE* f = std::fopen("/proc/self/io", "r");
    if (f == nullptr) return (size_t) -1;
    size_t v = 0; char line[128];
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "read_bytes: %zu", &v) == 1) break;
    std::fclose(f);
    return v;
}
long majflt() { struct rusage ru; getrusage(RUSAGE_SELF, &ru); return ru.ru_majflt; }

size_t resident_pages(const void* addr, size_t bytes) {
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const size_t n = (bytes + page - 1) / page;
    std::vector<unsigned char> vec(n, 0);
    if (mincore(const_cast<void*>(addr), bytes, vec.data()) != 0) return (size_t) -1;
    size_t r = 0;
    for (unsigned char c : vec) r += (c & 1) != 0;
    return r;
}

int run_fixture(const std::string& model_dir, const std::string& scratch_dir, int device, int K, int trials) {
    if (cudaSetDevice(device) != cudaSuccess) { std::fprintf(stderr, "SKIP: no CUDA device %d\n", device); return 77; }
    strata::Exl3Model model(model_dir);
    std::vector<ExpertPlanes> experts((size_t) K);
    size_t max_plane = 0;
    for (int e = 0; e < K; ++e) {
        locate_expert(model, 0, e, experts[(size_t) e]);   // geometry only; model data is never read
        for (int p = 0; p < 9; ++p) max_plane = std::max(max_plane, experts[(size_t) e].p[(size_t) p].bytes);
    }
    const size_t total = (size_t) K * experts[0].bytes;
    const std::string path = scratch_dir + "/readpath-fixture.bin";
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) { std::fprintf(stderr, "fixture: cannot create %s\n", path.c_str()); return 1; }
    {   // deterministic non-zero bytes (an LCG block pattern, never sparse)
        std::vector<unsigned char> chunk(1 << 16);
        uint32_t state = 0x5eed1234u;
        for (size_t written = 0; written < total;) {
            for (size_t i = 0; i < chunk.size(); i += 4) {
                state = 1103515245u * state + 12345u;
                chunk[i] = (unsigned char) (state >> 24); chunk[i + 1] = (unsigned char) (state >> 16);
                chunk[i + 2] = (unsigned char) (state >> 8); chunk[i + 3] = (unsigned char) state;
            }
            const size_t n = std::min(chunk.size(), total - written);
            if (std::fwrite(chunk.data(), 1, n, f) != n) { std::fclose(f); return 1; }
            written += n;
        }
        std::fflush(f);
        if (::fsync(::fileno(f)) != 0) { std::fclose(f); return 1; }
    }
    std::fclose(f);
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { std::fprintf(stderr, "fixture: cannot open\n"); return 1; }
    void* map = ::mmap(nullptr, total, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { ::close(fd); return 1; }
    constexpr int kBuf = 3;   // the pread arm's pinned ping-pong depth: a buffer is only overwritten
                              // after its own last H2D copy drained (source valid until drain, like the mapping)
    uint8_t* pinned = nullptr; uint8_t* dst = nullptr;
    cudaStream_t dma = nullptr;
    cudaEvent_t ev[3] = {nullptr, nullptr, nullptr};
    if (cudaMallocHost((void**) &pinned, max_plane * kBuf) != cudaSuccess ||
        cudaMalloc((void**) &dst, total) != cudaSuccess ||
        cudaStreamCreateWithFlags(&dma, cudaStreamNonBlocking) != cudaSuccess) {
        (void) cudaFreeHost(pinned); (void) cudaFree(dst); (void) munmap(map, total); ::close(fd); return 1;
    }
    for (int b = 0; b < kBuf; ++b)
        if (cudaEventCreateWithFlags(&ev[b], cudaEventDisableTiming) != cudaSuccess) {
            for (int c = 0; c < b; ++c) (void) cudaEventDestroy(ev[c]);
            (void) cudaStreamDestroy(dma); (void) cudaFreeHost(pinned); (void) cudaFree(dst);
            (void) munmap(map, total); ::close(fd); return 1;
        }
    struct Clean { ~Clean() { for (cudaEvent_t e : ev_g) (void) cudaEventDestroy(e);
                              (void) cudaStreamDestroy(dma_g); (void) cudaFree(dst_g); (void) cudaFreeHost(pinned_g);
                              (void) munmap(map_g, total_g); ::close(fd_g); ::unlink(path_g); }
        cudaStream_t dma_g; uint8_t *dst_g, *pinned_g; void* map_g; size_t total_g; int fd_g; const char* path_g;
        cudaEvent_t ev_g[3]; }
        clean{dma, dst, pinned, map, total, fd, path.c_str(), {ev[0], ev[1], ev[2]}};

    // one "expert" = its 9 plane ranges inside the fixture; both arms read EVERY range
    auto ranges = [&](int e) {
        std::vector<std::pair<size_t, size_t>> out;
        size_t at = (size_t) e * experts[0].bytes;
        for (int p = 0; p < 9; ++p) { out.push_back({at, experts[(size_t) e].p[(size_t) p].bytes}); at += out.back().second; }
        return out;
    };
    std::vector<std::pair<size_t, size_t>> dst_ranges;   // device destinations, plane-aligned
    for (int e = 0; e < K; ++e) {
        size_t at = (size_t) e * experts[0].bytes;
        for (int p = 0; p < 9; ++p) { dst_ranges.push_back({at, experts[(size_t) e].p[(size_t) p].bytes}); at += dst_ranges.back().second; }
    }
    auto run_pread = [&]() {   // one reader thread; per plane: pread into a free pinned buffer, async H2D
        size_t i = 0;
        for (int e = 0; e < K; ++e)
            for (const auto& [off, bytes] : ranges(e)) {
                const int b = (int) (i % (size_t) kBuf);
                if (i >= (size_t) kBuf) (void) cudaEventSynchronize(ev[b]);   // this buffer's last copy drained
                if (::pread(fd, pinned + (size_t) b * max_plane, bytes, (off_t) off) != (ssize_t) bytes)
                    throw std::runtime_error("fixture pread");
                if (cudaMemcpyAsync(dst + dst_ranges[i].first, pinned + (size_t) b * max_plane, bytes,
                                    cudaMemcpyHostToDevice, dma) != cudaSuccess)
                    throw std::runtime_error("fixture H2D");
                (void) cudaEventRecord(ev[b], dma);
                ++i;
            }
        (void) cudaStreamSynchronize(dma);
    };
    auto run_mmap = [&]() {   // pageable source, mapping alive through the drain (NO_PREAD's path)
        size_t i = 0;
        for (int e = 0; e < K; ++e)
            for (const auto& [off, bytes] : ranges(e)) {
                if (cudaMemcpyAsync(dst + dst_ranges[i].first, (const uint8_t*) map + off, bytes,
                                    cudaMemcpyHostToDevice, dma) != cudaSuccess)
                    throw std::runtime_error("fixture mmap H2D");
                ++i;
            }
        (void) cudaStreamSynchronize(dma);
    };
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const size_t npages = (total + page - 1) / page;

    std::printf("fixture: file=%s bytes=%zu (%.2f MiB) experts=%d planes=9 trials=%d\n",
                path.c_str(), total, (double) total / (1 << 20), K, trials);
    std::printf("fixture wall: host clock around the arm's read+H2D issue through its final "
                "cudaStreamSynchronize (DMA drained) - both boundaries on the host clock, unlike the "
                "transport bench's worker-report wall and event dma_span, which are never subtracted\n");
    {   // WARM: cache the file, verify residency before EVERY arm, then alternate arms trial by trial
        run_mmap();   // one full pass faults every page in (pageable copy reads the mapping)
        size_t r = resident_pages(map, total);
        std::printf("fixture warm: resident %zu/%zu pages (%.1f%%)\n", r, npages, 100.0 * (double) r / npages);
        std::vector<double> wp, wm; size_t io0 = read_io_bytes(); int checks = 0, partial = 0;
        for (int t = 0; t < trials; ++t) {
            for (int arm = 0; arm < 2; ++arm) {
                ++checks;
                r = resident_pages(map, total);
                if (r != npages) {
                    ++partial;
                    std::printf("fixture warm trial %d arm %d: only %zu/%zu pages resident\n", t, arm, r, npages);
                }
                const Clock::time_point t0 = Clock::now();
                if (arm == 0) run_pread(); else run_mmap();
                (arm == 0 ? wp : wm).push_back(ms_between(t0, Clock::now()));
            }
        }
        const size_t d_io = read_io_bytes() - io0;
        std::printf("fixture WARM: pread+pinned+H2D %.3f ms | mmap-copy+H2D %.3f ms | read_bytes delta %zu "
                    "(0 expected warm) | mincore checks %d (%d partial)\n", median(wp), median(wm), d_io,
                    checks, partial);
        // correctness: the device bytes each arm moved must equal the source, outside the timed trials
        std::vector<uint8_t> back(total);
        for (int arm = 0; arm < 2; ++arm) {
            if (arm == 0) run_pread(); else run_mmap();
            if (cudaMemcpy(back.data(), dst, total, cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("fixture D2H");
            const bool same = std::memcmp(back.data(), map, total) == 0;
            std::printf("fixture verify: %s moved the source's %zu bytes %s\n",
                        arm == 0 ? "pread+pinned+H2D" : "mmap-copy+H2D", total, same ? "PASS" : "FAIL");
            if (!same) return 1;
        }
    }
    {   // COLD: fadvise(DONTNEED) THIS FILE ONLY, munmap/remap, mincore-verify before every trial
        bool verified = true;
        std::vector<double> cp, cm; size_t io_sum = 0; long flt_sum = 0;
        for (int t = 0; t < trials; ++t) {
            if (::munmap(map, total) != 0) { verified = false; break; }
            if (::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) != 0) verified = false;
            map = ::mmap(nullptr, total, PROT_READ, MAP_SHARED, fd, 0);
            if (map == MAP_FAILED) { std::fprintf(stderr, "fixture: remap failed\n"); return 1; }
            clean.map_g = map;   // the cleanup guard tracks the live mapping (the old one is gone)
            const size_t r = resident_pages(map, total);
            if (r != 0) {
                std::printf("fixture trial %d: EVICTION-INCOMPLETE (%zu/%zu pages resident) - result UNKNOWN, "
                            "not cold\n", t, r, npages);
                verified = false;
                break;
            }
            const size_t io0 = read_io_bytes(); const long f0 = majflt();
            const Clock::time_point t0 = Clock::now();
            if ((t & 1) == 0) run_pread(); else run_mmap();
            const double ms = ms_between(t0, Clock::now());
            if ((t & 1) == 0) cp.push_back(ms); else cm.push_back(ms);
            io_sum += read_io_bytes() - io0; flt_sum += majflt() - f0;
        }
        if (!verified) {
            std::printf("fixture COLD: NOT VERIFIED - no cold numbers claimed\n");
        } else {
            std::printf("fixture COLD(%s): pread+pinned+H2D %.3f ms (n=%zu) | mmap-copy+H2D %.3f ms (n=%zu) | "
                        "read_bytes sum %zu (expect ~%zu x n) | major faults sum %ld\n",
                        "mincore-verified", cp.empty() ? -1.0 : median(cp), cp.size(),
                        cm.empty() ? -1.0 : median(cm), cm.size(), io_sum, total, flt_sum);
        }
    }
    std::printf("fixture limit: a controlled scratch-file measurement of the two read arms; it does NOT "
                "establish real-model cold TG\n");
    return 0;
}

// -------- part C (--m5): the M5 bounded experiment - kernel/protocol cost of the resident hit arm --------
//
// NON-PRODUCTION microbench, CLEARLY NOT AN ENGINE TG MEASUREMENT.  It quantifies what the
// test-only `exl3_moe_rows_slot_table` primitive would buy the verifier's resident hit arm over
// the current host-array `exl3_moe_rows_indexed` path: the GPU kernel difference and the HOST
// PROTOCOL saving are reported SEPARATELY and never summed into an "ideal tok/s".
//
// Contracts exercised (the primitive's header): stable pointers, workspace pre-reserved before
// the capture (no allocation under replay), table content immutable and resident for every
// replay (pinned slot subset), invalid-id rejection OUTSIDE capture only (the entry validates on
// the host and throws before anything launches; a captured graph is built from an already
// validated warm call), x/input_rows/slot ids alive through each replay.  The readback-validation
// the PUBLIC entry performs per call is measured as its own probe, so "outside capture" and
// "replay" costs are distinguishable.  Route counts are T=1 and T=4 with k=10 (the 3bpw model's
// experts per token); the geometry is the real 2560->640 K3 experts, n_dev of them resident.
// Commit/rollback and MTP acceptance gates are ENGINE-side; nothing here models them.
int run_m5(const std::string& model_dir, int device, int n_dev, int reps) {
    if (cudaSetDevice(device) != cudaSuccess) { std::fprintf(stderr, "SKIP: no CUDA device %d\n", device); return 77; }
    strata::Exl3Model model(model_dir);
    const int width = 2560, hidden = 640;
    std::vector<uint8_t*> dev((size_t) n_dev * 9, nullptr);
    std::vector<Exl3Mat> hg((size_t) n_dev), hu((size_t) n_dev), hd((size_t) n_dev);
    for (int e = 0; e < n_dev; ++e) {
        const std::string b = "model.language_model.layers.0.mlp.experts." + std::to_string(e + 100) + ".";
        const strata::Exl3Linear lin[3] = {model.linear(b + "gate_proj"), model.linear(b + "up_proj"),
                                           model.linear(b + "down_proj")};
        Exl3Mat* mats[3] = {&hg[(size_t) e], &hu[(size_t) e], &hd[(size_t) e]};
        for (int i = 0; i < 3; ++i) {
            const size_t tr = (size_t) lin[i].ki * lin[i].nj * 32 * lin[i].bits;
            for (int p = 0; p < 3; ++p) {
                const size_t bytes = p == 0 ? tr : (size_t) (p == 1 ? lin[i].in() : lin[i].out()) * 2;
                if (cudaMalloc((void**) &dev[(size_t) e * 9 + (size_t) i * 3 + p], bytes) != cudaSuccess)
                    throw std::runtime_error("m5 alloc failed");
                if (p == 0 && cudaMemcpy(dev[(size_t) e * 9 + (size_t) i * 3], lin[i].trellis, tr,
                                         cudaMemcpyHostToDevice) != cudaSuccess)
                    throw std::runtime_error("m5 upload failed");
                if (p > 0 && cudaMemset(dev[(size_t) e * 9 + (size_t) i * 3 + p], 0x38, bytes) != cudaSuccess)
                    throw std::runtime_error("m5 scale init failed");
            }
            mats[i]->trellis = (const uint16_t*) dev[(size_t) e * 9 + (size_t) i * 3];
            mats[i]->suh = (const uint16_t*) dev[(size_t) e * 9 + (size_t) i * 3 + 1];
            mats[i]->svh = (const uint16_t*) dev[(size_t) e * 9 + (size_t) i * 3 + 2];
            mats[i]->ki = lin[i].ki; mats[i]->nj = lin[i].nj;
            mats[i]->bits = lin[i].bits; mats[i]->cb = lin[i].cb;
        }
    }
    cudaStream_t cs = nullptr; cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    uint16_t *dx = nullptr, *drows = nullptr; int *drows_idx = nullptr, *dslots = nullptr;
    Exl3Mat *dtg = nullptr, *dtu = nullptr, *dtd = nullptr;
    if (cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreate(&ev0) != cudaSuccess ||   // timing-enabled: cudaEventElapsedTime needs it
        cudaEventCreate(&ev1) != cudaSuccess ||
        cudaMalloc((void**) &dx, (size_t) 4 * width * 2) != cudaSuccess ||
        cudaMalloc((void**) &drows, (size_t) 40 * width * 2) != cudaSuccess ||
        cudaMalloc((void**) &drows_idx, 40 * sizeof(int)) != cudaSuccess ||
        cudaMalloc((void**) &dslots, 40 * sizeof(int)) != cudaSuccess ||
        cudaMalloc((void**) &dtg, (size_t) n_dev * sizeof(Exl3Mat)) != cudaSuccess ||
        cudaMalloc((void**) &dtu, (size_t) n_dev * sizeof(Exl3Mat)) != cudaSuccess ||
        cudaMalloc((void**) &dtd, (size_t) n_dev * sizeof(Exl3Mat)) != cudaSuccess)
        throw std::runtime_error("m5 stream/buffers failed");
    if (cudaMemset(dx, 0x38, (size_t) 4 * width * 2) != cudaSuccess ||
        cudaMemcpy(dtg, hg.data(), (size_t) n_dev * sizeof(Exl3Mat), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dtu, hu.data(), (size_t) n_dev * sizeof(Exl3Mat), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dtd, hd.data(), (size_t) n_dev * sizeof(Exl3Mat), cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("m5 init failed");
    struct MGuard { ~MGuard() { (void) cudaEventDestroy(ev0_g); (void) cudaEventDestroy(ev1_g);
                                (void) cudaStreamDestroy(cs_g); (void) cudaFree(dx_g); (void) cudaFree(drows_g);
                                (void) cudaFree(drows_idx_g); (void) cudaFree(dslots_g); (void) cudaFree(dtg_g);
                                (void) cudaFree(dtu_g); (void) cudaFree(dtd_g);
                                for (uint8_t* p : dev_g) (void) cudaFree(p); }
        cudaEvent_t ev0_g, ev1_g; cudaStream_t cs_g; uint16_t *dx_g, *drows_g; int *drows_idx_g, *dslots_g;
        Exl3Mat *dtg_g, *dtu_g, *dtd_g; std::vector<uint8_t*>& dev_g; }
        guard{ev0, ev1, cs, dx, drows, drows_idx, dslots, dtg, dtu, dtd, dev};
    Exl3MoeWorkspace ws;
    ws.reserve(40, width, hidden, width, true);   // pre-reserved OUTSIDE every timed loop and the capture

    std::printf("m5: model=%s device=%d resident experts=%d (2560->640 K3, immutable tables, pinned subset), "
                "k=10 routes/token, T=1 and T=4, reps=%d\n", model_dir.c_str(), device, n_dev, reps);
    std::printf("m5 limit: a kernel/protocol microbench - NOT an engine TG measurement; commit/rollback and "
                "MTP acceptance gates are engine-side and untouched\n");
    using strata::kernels::exl3_moe_rows_indexed;
    using strata::kernels::exl3_moe_rows_slot_table;

    auto time_arm = [&](const char* what, int R, auto&& fn) {
        for (int i = 0; i < 5; ++i) fn();
        (void) cudaStreamSynchronize(cs);
        std::vector<double> issue; double span_sum = 0;
        for (int r = 0; r < reps; ++r) {
            const Clock::time_point t0 = Clock::now();
            (void) cudaEventRecord(ev0, cs);
            fn();
            (void) cudaEventRecord(ev1, cs);
            const Clock::time_point t1 = Clock::now();   // host-issue ends when the CALL returns
            (void) cudaStreamSynchronize(cs);
            float span = -1;
            (void) cudaEventElapsedTime(&span, ev0, ev1);
            issue.push_back(ms_between(t0, t1));
            span_sum += span;
        }
        std::printf("m5 T-k10 R=%d %s: host-issue %.3f ms | gpu_span %.3f ms\n", R, what, median(issue),
                    span_sum / reps);
    };

    for (const int T : {1, 4}) {
        const int R = T * 10;
        // three route sets: two distinct mixes and one with DUPLICATE slots (scatter order matters)
        std::vector<std::vector<int>> sets(3, std::vector<int>((size_t) R));
        std::vector<std::vector<int>> rows(3, std::vector<int>((size_t) R));
        for (int s = 0; s < 3; ++s)
            for (int r = 0; r < R; ++r) {
                sets[(size_t) s][(size_t) r] = s == 2 ? (r / 2) % n_dev : (r * (s == 0 ? 3 : 7) + s) % n_dev;
                rows[(size_t) s][(size_t) r] = r % T;
            }
        auto host_route = [&](int s) {
            std::vector<Exl3Mat> g((size_t) R), u((size_t) R), d((size_t) R);
            for (int r = 0; r < R; ++r) {
                const int e = sets[(size_t) s][(size_t) r];
                g[(size_t) r] = hg[(size_t) e]; u[(size_t) r] = hu[(size_t) e]; d[(size_t) r] = hd[(size_t) e];
            }
            return std::make_tuple(g, u, d);
        };
        auto upload = [&](int s) {
            if (cudaMemcpyAsync(drows_idx, rows[(size_t) s].data(), (size_t) R * sizeof(int),
                                cudaMemcpyHostToDevice, cs) != cudaSuccess ||
                cudaMemcpyAsync(dslots, sets[(size_t) s].data(), (size_t) R * sizeof(int),
                                cudaMemcpyHostToDevice, cs) != cudaSuccess)
                throw std::runtime_error("m5 upload failed");
        };
        // reference outputs per set (host-array path), then every arm must match them
        std::vector<std::vector<uint16_t>> ref(3, std::vector<uint16_t>((size_t) R * width));
        for (int s = 0; s < 3; ++s) {
            auto [g, u, d] = host_route(s);
            upload(s);
            exl3_moe_rows_indexed(g.data(), u.data(), d.data(), R, dx, drows_idx, drows, cs, ws);
            (void) cudaStreamSynchronize(cs);
            if (cudaMemcpy(ref[(size_t) s].data(), drows, (size_t) R * width * 2, cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("m5 ref readback failed");
        }
        // slot-table public path must match each set (and reject an invalid id OUTSIDE capture)
        bool st_ok = true;
        for (int s = 0; s < 3; ++s) {
            upload(s);
            exl3_moe_rows_slot_table(dtg, dtu, dtd, n_dev, dslots, R, width, hidden, width, dx, drows_idx,
                                     drows, cs, ws);
            (void) cudaStreamSynchronize(cs);
            std::vector<uint16_t> y((size_t) R * width);
            if (cudaMemcpy(y.data(), drows, (size_t) R * width * 2, cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("m5 readback failed");
            st_ok = st_ok && std::memcmp(y.data(), ref[(size_t) s].data(), y.size() * 2) == 0;
        }
        std::printf("m5 T-k10 R=%d slot-table vs host-array over 3 route sets (one with duplicate slots): %s\n",
                    R, st_ok ? "byte-identical PASS" : "FAIL");
        {
            std::vector<int> bad = sets[0]; bad[(size_t) R - 1] = n_dev + 3;
            if (cudaMemcpyAsync(dslots, bad.data(), (size_t) R * sizeof(int), cudaMemcpyHostToDevice, cs) != cudaSuccess)
                throw std::runtime_error("m5 upload failed");
            (void) cudaStreamSynchronize(cs);
            bool threw = false;
            try {
                exl3_moe_rows_slot_table(dtg, dtu, dtd, n_dev, dslots, R, width, hidden, width, dx, drows_idx,
                                         drows, cs, ws);
            } catch (const std::exception&) { threw = true; }
            std::printf("m5 T-k10 R=%d invalid slot id OUTSIDE capture: %s\n", R,
                        threw ? "throws before launch PASS" : "silent FAIL");
        }
        // capture the primitive's device work (validation is skipped inside capture by contract)
        upload(0);
        exl3_moe_rows_slot_table(dtg, dtu, dtd, n_dev, dslots, R, width, hidden, width, dx, drows_idx,
                                 drows, cs, ws);   // warm call: the validation, workspace and statics
        (void) cudaStreamSynchronize(cs);
        cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
        if (cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal) != cudaSuccess) { std::printf("m5 begin-capture FAILED\n"); return 1; }
        exl3_moe_rows_slot_table(dtg, dtu, dtd, n_dev, dslots, R, width, hidden, width, dx, drows_idx,
                                 drows, cs, ws);
        if (cudaStreamEndCapture(cs, &graph) != cudaSuccess || graph == nullptr) { std::printf("m5 end-capture FAILED\n"); return 1; }
        if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess) { std::printf("m5 instantiate FAILED\n"); return 1; }
        // replay with MULTIPLE CHANGED route inputs (sets 0,1,2,0...) - same addresses/types as the host arms
        bool rp_ok = true;
        for (int rep = 0; rep < 6; ++rep) {
            const int s = rep % 3;
            upload(s);
            if (cudaGraphLaunch(exec, cs) != cudaSuccess) { rp_ok = false; break; }
            (void) cudaStreamSynchronize(cs);
            std::vector<uint16_t> y((size_t) R * width);
            if (cudaMemcpy(y.data(), drows, (size_t) R * width * 2, cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("m5 readback failed");
            rp_ok = rp_ok && std::memcmp(y.data(), ref[(size_t) s].data(), y.size() * 2) == 0;
        }
        std::printf("m5 T-k10 R=%d graph replay x6 over changed route inputs (3 sets, one duplicated): %s\n",
                    R, rp_ok ? "byte-identical PASS" : "FAIL");
        // ---- timing arms: the GPU kernel diff and the host protocol cost stay separate ----
        {
            const auto routed = host_route(0);
            const std::vector<Exl3Mat> &g = std::get<0>(routed), &u = std::get<1>(routed), &d = std::get<2>(routed);
            time_arm("host-array indexed (current hit arm)", R, [&] {
                upload(0);
                exl3_moe_rows_indexed(g.data(), u.data(), d.data(), R, dx, drows_idx, drows, cs, ws);
            });
        }
        time_arm("slot-table PUBLIC per-call (validation inside)", R, [&] {
            upload(0);
            exl3_moe_rows_slot_table(dtg, dtu, dtd, n_dev, dslots, R, width, hidden, width, dx, drows_idx,
                                     drows, cs, ws);
        });
        time_arm("slot-table graph replay + input upload", R, [&] {
            upload(0);
            (void) cudaGraphLaunch(exec, cs);
        });
        time_arm("slot-table graph replay only (inputs staged)", R, [&] {
            (void) cudaGraphLaunch(exec, cs);
        });
        {   // the public entry's readback validation alone (what a replay drops)
            std::vector<int> ids((size_t) R);
            for (int i = 0; i < 5; ++i) {
                (void) cudaMemcpyAsync(ids.data(), dslots, (size_t) R * sizeof(int), cudaMemcpyDeviceToHost, cs);
                (void) cudaStreamSynchronize(cs);
            }
            std::vector<double> issue;
            for (int r = 0; r < reps; ++r) {
                const Clock::time_point t0 = Clock::now();
                (void) cudaMemcpyAsync(ids.data(), dslots, (size_t) R * sizeof(int), cudaMemcpyDeviceToHost, cs);
                (void) cudaStreamSynchronize(cs);
                int lo = 0;
                for (int i = 0; i < R; ++i) lo += ids[(size_t) i] < 0 || ids[(size_t) i] >= n_dev;
                if (lo != 0) throw std::runtime_error("m5 validation probe saw an invalid id");
                issue.push_back(ms_between(t0, Clock::now()));
            }
            std::printf("m5 T-k10 R=%d slot-id readback validation (public entry, per call): host %.3f ms\n",
                        R, median(issue));
        }
        (void) cudaGraphExecDestroy(exec);
        (void) cudaGraphDestroy(graph);
    }
    std::printf("m5 done (kernel/protocol microbench, NOT engine TG)\n");
    return 0;
}

// -------- part D (--peer): the PEER tier feasibility fixture (EXL3 adapter prototype) --------
//
// NON-PRODUCTION, a fixture - NOT an engine TG measurement.  It mirrors PeerExperts'
// launch/finish contract with the EXL3 backing (Exl3ExpertStream::run_multi - host float X in,
// host float rows out, per-expert FFN) on a SECOND device: the primary (dev0) computes the SAME
// expert selection through the same call (the engine's host-IO baseline), the peer (dev1)
// computes it THERE and the rows come back to the host - the portable-host-buffer path, no P2P
// assumed (P2P is probed and reported only).  The tier's disjointness/adapt policy is engine-side
// and only stubbed here as an explicit residency pre-check that REFUSES non-resident ids (the
// "no quiet wrong-slot fallback" rule).  All sources and workspaces are allocated in setup, never
// inside a call; every device-sensitive operation runs under a device guard and the caller's
// device is verified restored afterwards.
int run_peer(const std::string& model_dir, int dev0, int dev1, int layer_lo, int layer_hi,
             int slots, int reps) {
    // ---- argument validation BEFORE any device work; bad arguments are a usage error (2)
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess) { std::fprintf(stderr, "peer: no CUDA devices\n"); return 1; }
    if (dev0 < 0 || dev0 >= ndev || dev1 < 0 || dev1 >= ndev || dev0 == dev1 ||
        layer_lo < 0 || layer_hi <= layer_lo || layer_hi > 48 || slots < 1 || slots > 500 ||
        100 + slots > 512 || reps < 1 || reps > 10000) {
        std::fprintf(stderr, "peer: invalid arguments - dev0/dev1 distinct in [0,%d), 0<=layer_lo<layer_hi<=48, "
                     "1<=slots<=500 with 100+slots<=512 (expert ids are in [100,100+slots)), 1<=reps<=10000\n", ndev);
        return 2;
    }
    if (ndev < 2) { std::fprintf(stderr, "SKIP: peer fixture needs two visible devices (have %d)\n", ndev); return 77; }
    int can01 = 0, can10 = 0;
    (void) cudaDeviceCanAccessPeer(&can01, dev0, dev1);
    (void) cudaDeviceCanAccessPeer(&can10, dev1, dev0);
    std::printf("peer: devices %d (primary) and %d (peer); P2P %d/%d probed, NOT relied on\n",
                dev0, dev1, can01, can10);
    const int n_embd = 2560, n_layers = 48, k = 10;
    // every device-sensitive operation runs under this guard; it restores the CALLER's device
    struct On {
        int prev;
        explicit On(int d) { cudaGetDevice(&prev); if (prev != d) cudaSetDevice(d); }
        ~On() { int c = 0; cudaGetDevice(&c); if (c != prev) cudaSetDevice(prev); }
    };
    int caller = -1;
    (void) cudaGetDevice(&caller);
    bool ok = true;
    // the streams own ONLY [layer_lo, layer_hi) - the owned range is passed explicitly
    std::unique_ptr<strata::kernels::Exl3ExpertStream> base, peer;
    size_t free_before = 0, free_after = 0, total_b = 0;
    { On on(dev0); base = std::make_unique<strata::kernels::Exl3ExpertStream>(model_dir, n_embd, n_layers,
                                                                             slots, nullptr, 0,
                                                                             (size_t) 512 << 20,
                                                                             layer_lo, layer_hi); }
    {
        On on(dev1);
        (void) cudaMemGetInfo(&free_before, &total_b);
        peer = std::make_unique<strata::kernels::Exl3ExpertStream>(model_dir, n_embd, n_layers,
                                                                   slots, nullptr, 0,
                                                                   (size_t) 512 << 20, layer_lo, layer_hi);
        (void) cudaMemGetInfo(&free_after, &total_b);
    }
    {   // price the exact bytes: cache slots are exact; bytes() also covers scratch (labeled)
        const size_t cache_bytes = (size_t) slots * (size_t) (layer_hi - layer_lo) * 1862400u;
        std::printf("peer: %d layers [%d,%d), %d slots/layer = %zu cache bytes (%.1f MiB) EXACT; "
                    "stream bytes() %.1f MiB (INCLUDES scratch) per stream; measured permanent VRAM on the "
                    "peer device %.1f MiB (cudaMemGetInfo delta)\n",
                    layer_hi - layer_lo, layer_lo, layer_hi, slots, cache_bytes,
                    (double) cache_bytes / 1048576.0, (double) peer->bytes() / 1048576.0,
                    (double) (free_before - free_after) / 1048576.0);
    }
    // X and out live on the host for the whole run (the launch/finish contract's source lifetime)
    std::vector<float> x((size_t) 4 * n_embd), out0((size_t) 4 * k * n_embd), out1((size_t) 4 * k * n_embd);
    uint32_t s = 12345;
    for (auto& v : x) { s = 1103515245u * s + 12345u; v = (float) ((int) (s >> 16) % 2000 - 1000) * 0.001f; }
    std::memset(out0.data(), 0, out0.size() * sizeof(float));
    std::memset(out1.data(), 0, out1.size() * sizeof(float));
    // finite + non-constant: byte-parity of two NaN/constant buffers must never PASS
    auto sane = [](const float* p, size_t n) {
        if (n == 0) return false;
        size_t finite = 0;
        bool varied = false;
        for (size_t i = 0; i < n; ++i) { if (std::isfinite(p[i])) ++finite; if (p[i] != p[0]) varied = true; }
        return finite == n && varied;
    };
    // the fixture's launch wrapper: a REAL pre-check before ANY stream call (the adapter's
    // "refuse, never a quiet fallback" rule, fixture-local - the production guard is NOT built)
    int64_t launches = 0;
    auto resident = [&](int32_t e) { return e >= 100 && e < 100 + slots; };
    auto launch = [&](int layer, const std::vector<int32_t>& ids, int T, std::vector<float>& out,
                      int dev, strata::kernels::Exl3ExpertStream& strm, bool* refused) -> bool {
        *refused = false;
        if (layer < layer_lo || layer >= layer_hi) { *refused = true; return false; }   // bounds-checked lookup
        for (size_t r = 0; r < ids.size(); ++r)
            if (!resident(ids[r])) { *refused = true; return false; }   // refused BEFORE the stream is called
        ++launches;
        On on(dev);
        return strm.run_multi(layer, x.data(), ids.data(), T, k, out.data(), nullptr);
    };
    std::printf("peer limit: fixture transport+compute feasibility, NOT engine TG; run_multi is "
                "SYNCHRONOUS, so this measures serial roundtrip latency only - no async launch/finish\n");
    // ---- the FRESH first call on the cold peer stream, BEFORE any correctness/warmup call:
    // it fills this route set's experts (page cache -> link -> GPU1) AND computes - the number
    // below includes compute and is not a pure fill time
    std::vector<int32_t> ids0((size_t) k);
    for (int r = 0; r < k; ++r) ids0[(size_t) r] = 100 + (r / 2) % slots;
    double first_call_ms = -1;
    {
        const Clock::time_point f0 = Clock::now();
        bool refused = false;
        const bool ran = launch(layer_lo, ids0, 1, out1, dev1, *peer, &refused);
        first_call_ms = ms_between(f0, Clock::now());
        ok = ok && ran && !refused;
        std::printf("peer: cold first call (layer %d, fills %d experts x 1,862,400 B + COMPUTE): %.1f ms "
                    "(fill+compute, NOT a pure fill time)\n", layer_lo, slots, first_call_ms);
    }

    for (int T : {1, 4}) {
        const int R = T * k;
        // route sets: one with duplicates, one distinct mix; experts = the resident subset
        std::vector<std::vector<int32_t>> ids(2, std::vector<int32_t>((size_t) R));
        for (int set = 0; set < 2; ++set)
            for (int r = 0; r < R; ++r)
                ids[(size_t) set][(size_t) r] = 100 + (set == 0 ? (r / 2) : (r * 3)) % slots;
        for (int layer = layer_lo; layer < layer_hi; ++layer) {
            // ---- correctness: peer rows == primary rows for the same selection (duplicates included)
            for (int set = 0; set < 2; ++set) {
                std::memset(out0.data(), 0, out0.size() * sizeof(float));
                std::memset(out1.data(), 0, out1.size() * sizeof(float));
                bool ref = false;
                const bool a = launch(layer, ids[(size_t) set], T, out0, dev0, *base, &ref);
                const bool b = launch(layer, ids[(size_t) set], T, out1, dev1, *peer, &ref);
                int cur = -1;
                (void) cudaGetDevice(&cur);
                const size_t bytes = (size_t) R * n_embd * sizeof(float);
                const bool same = a && b && sane(out0.data(), bytes / 4) && sane(out1.data(), bytes / 4) &&
                                  std::memcmp(out0.data(), out1.data(), bytes) == 0;
                std::printf("peer T=%d layer=%d rows=%d set=%d: %s (finite+non-constant checked)\n", T, layer,
                            R, set, same ? "byte-identical PASS" : "FAIL");
                ok = ok && same;
                if (cur != caller) {
                    std::printf("peer: DEVICE CONTEXT NOT RESTORED (now %d, caller %d)\n", cur, caller);
                    ok = false;
                }
            }
            // ---- invalid residency via the REAL wrapper: refused AND the stream was never called
            {
                std::vector<int32_t> bad = ids[0]; bad[(size_t) R - 1] = 100 + slots + 7;
                const int64_t before = launches;
                bool refused = false;
                const bool ran = launch(layer, bad, T, out1, dev1, *peer, &refused);
                const bool pass = refused && !ran && launches == before;
                std::printf("peer T=%d layer=%d invalid residency: %s (stream calls %lld -> %lld)\n", T, layer,
                            pass ? "refused before launch PASS" : "SILENT FALLBACK FAIL",
                            (long long) before, (long long) launches);
                ok = ok && pass;
            }
            if (layer != layer_lo) continue;   // timing on one layer (the per-layer cost is layer-uniform)
            // ---- timing: primary baseline vs peer roundtrip vs link-only floor
            auto wall = [&](const char* what, auto&& fn) {
                for (int i = 0; i < 5; ++i) fn();
                std::vector<double> ms;
                for (int r = 0; r < reps; ++r) {
                    const Clock::time_point t0 = Clock::now();
                    if (!fn()) ok = false;
                    ms.push_back(ms_between(t0, Clock::now()));
                }
                std::printf("peer T=%d R=%d %s: wall %.3f ms | per-layer traffic X %d B + rows %d B = %d B\n",
                            T, R, what, median(ms), T * n_embd * 4, R * n_embd * 4,
                            (T + R) * n_embd * 4);
            };
            wall("primary run_multi on GPU0 (host-IO baseline)", [&] {
                On on(dev0);
                return base->run_multi(layer, x.data(), ids[0].data(), T, k, out0.data(), nullptr);
            });
            wall("peer run_multi on GPU1 (serial roundtrip, rows to host)", [&] {
                On on(dev1);
                return peer->run_multi(layer, x.data(), ids[0].data(), T, k, out1.data(), nullptr);
            });
            {   // link-only floor: the same bytes with no compute (X up, rows down) on GPU1
                struct Dev2 {
                    int dev;
                    float *a = nullptr, *b = nullptr;
                    ~Dev2() { if (a || b) { int p = 0; cudaGetDevice(&p); cudaSetDevice(dev); if (a) cudaFree(a);
                                            if (b) cudaFree(b); cudaSetDevice(p); } }
                } d2{dev1};
                bool alloc_ok = true;
                {
                    On on(dev1);
                    alloc_ok = cudaMalloc((void**) &d2.a, (size_t) T * n_embd * 4) == cudaSuccess &&
                               cudaMalloc((void**) &d2.b, (size_t) R * n_embd * 4) == cudaSuccess;
                }
                ok = ok && alloc_ok;
                if (alloc_ok)
                    wall("link-only floor (X H2D + rows D2H, no compute)", [&] {
                        On on(dev1);
                        return cudaSuccess == cudaMemcpyAsync(d2.a, x.data(), (size_t) T * n_embd * 4,
                                                              cudaMemcpyHostToDevice, nullptr) &&
                               cudaSuccess == cudaMemcpyAsync(out1.data(), d2.b, (size_t) R * n_embd * 4,
                                                              cudaMemcpyDeviceToHost, nullptr) &&
                               cudaSuccess == cudaDeviceSynchronize();
                    });
            }
        }
    }
    { On on(dev1); peer.reset(); }
    { On on(dev0); base.reset(); }
    int cur = -1;
    (void) cudaGetDevice(&cur);
    if (cur != caller) { std::printf("peer: DEVICE CONTEXT NOT RESTORED AT EXIT (%d vs %d)\n", cur, caller); ok = false; }
    std::printf("peer done (fixture, NOT engine TG) - %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model_dir> [device=0] [experts=54] [layer=0] [reps=20]\n"
                             "       %s --fixture <model_dir> <scratch_dir> [device=0] [experts=54] [trials=15]\n"
                             "       %s --m5 <model_dir> [device=0] [experts=16] [reps=200]\n"
                             "       %s --peer <model_dir> [dev0=0] [dev1=1] [layer_lo=0] [layer_hi=2] "
                             "[slots=4] [reps=100]\n",
                     argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    if (std::strcmp(argv[1], "--fixture") == 0) {
        if (argc < 4) { std::fprintf(stderr, "fixture mode needs <model_dir> <scratch_dir>\n"); return 2; }
        return run_fixture(argv[2], argv[3], argc > 4 ? std::atoi(argv[4]) : 0,
                           argc > 5 ? std::atoi(argv[5]) : 54, argc > 6 ? std::atoi(argv[6]) : 15);
    }
    if (std::strcmp(argv[1], "--m5") == 0) {
        if (argc < 3) { std::fprintf(stderr, "m5 mode needs <model_dir>\n"); return 2; }
        return run_m5(argv[2], argc > 3 ? std::atoi(argv[3]) : 0,
                      argc > 4 ? std::atoi(argv[4]) : 16, argc > 5 ? std::atoi(argv[5]) : 200);
    }
    if (std::strcmp(argv[1], "--peer") == 0) {
        if (argc < 3) { std::fprintf(stderr, "peer mode needs <model_dir>\n"); return 2; }
        return run_peer(argv[2], argc > 3 ? std::atoi(argv[3]) : 0, argc > 4 ? std::atoi(argv[4]) : 1,
                        argc > 5 ? std::atoi(argv[5]) : 0, argc > 6 ? std::atoi(argv[6]) : 2,
                        argc > 7 ? std::atoi(argv[7]) : 4, argc > 8 ? std::atoi(argv[8]) : 100);
    }
    const std::string dir = argv[1];
    const int device = argc > 2 ? std::atoi(argv[2]) : 0;
    const int K = argc > 3 ? std::atoi(argv[3]) : 54;
    const int layer = argc > 4 ? std::atoi(argv[4]) : 0;
    const int reps = argc > 5 ? std::atoi(argv[5]) : 20;
    if (K < 1 || K > 512 || reps < 1 || reps > 1000) return 2;
    if (cudaSetDevice(device) != cudaSuccess) { std::fprintf(stderr, "SKIP: no CUDA device %d\n", device); return 77; }

    strata::Exl3Model model(dir);
    std::vector<ExpertPlanes> experts((size_t) K);
    size_t max_plane = 0; int large = 0, small = 0;
    for (int e = 0; e < K; ++e) {
        locate_expert(model, layer, e, experts[(size_t) e]);
        if (e > 0 && experts[(size_t) e].bytes != experts[0].bytes)
            throw std::runtime_error("transport bench: non-uniform expert size (unexpected geometry)");
        for (int p = 0; p < 9; ++p) {
            max_plane = std::max(max_plane, experts[(size_t) e].p[(size_t) p].bytes);
            if (experts[(size_t) e].p[(size_t) p].bytes >= (1 << 16)) ++large; else ++small;
        }
    }
    const size_t total = (size_t) K * experts[0].bytes;
    std::printf("transport-bench CUDA%d model=%s layer=%d experts=%d expert_bytes=%zu total_MiB=%.2f "
                "planes/expert=9 (>=64KiB: %d, small: %d) reps=%d\n",
                device, dir.c_str(), layer, K, experts[0].bytes, (double) total / (1 << 20), large, small, reps);

    uint8_t* dst = nullptr;
    cudaStream_t dma;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    if (cudaMalloc((void**) &dst, total) != cudaSuccess ||
        cudaStreamCreateWithFlags(&dma, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreate(&ev0) != cudaSuccess || cudaEventCreate(&ev1) != cudaSuccess)
        throw std::runtime_error("transport bench: dst/stream/event failed");
    struct Guard { ~Guard() { (void) cudaEventDestroy(ev0_g); (void) cudaEventDestroy(ev1_g);
                              (void) cudaStreamDestroy(dma_g); (void) cudaFree(dst_g);
                              if (pinned_g) (void) cudaFreeHost(pinned_g); }
        cudaEvent_t ev0_g, ev1_g; cudaStream_t dma_g; uint8_t* dst_g; uint8_t* pinned_g; } guard{ev0, ev1, dma, dst, nullptr};

    std::vector<std::pair<size_t, size_t>> jobs;
    jobs.reserve((size_t) K * 9);
    for (int e = 0; e < K; ++e)
        for (int p = 0; p < 9; ++p) jobs.push_back({(size_t) e, (size_t) p});

    // -------- arm (a): the engine worker pattern, 15 workers then 1 --------
    for (int workers : {15, 1}) {
        WorkerPool pool;
        pool.start(workers, max_plane, dma, &experts, dst);
        pool.run(jobs);
        (void) cudaStreamSynchronize(dma);   // warmup: page cache + driver steady state
        std::vector<double> walls;
        double pread_sum = 0, enq_sum = 0, wait_sum = 0, span_sum = 0;
        for (int r = 0; r < reps; ++r) {
            pool.sh.cpu_pread_ms.store(0); pool.sh.cpu_enqueue_ms.store(0); pool.sh.cpu_lock_wait_ms.store(0);
            const Clock::time_point t0 = Clock::now();
            (void) cudaEventRecord(ev0, dma);
            pool.run(jobs);
            const Clock::time_point t1 = Clock::now();   // wall ends when the last worker reports done
            (void) cudaEventRecord(ev1, dma);
            (void) cudaStreamSynchronize(dma);           // events complete BEFORE elapsed
            float span = -1;
            (void) cudaEventElapsedTime(&span, ev0, ev1);
            walls.push_back(ms_between(t0, t1));
            pread_sum += pool.sh.cpu_pread_ms.load(); enq_sum += pool.sh.cpu_enqueue_ms.load();
            wait_sum += pool.sh.cpu_lock_wait_ms.load(); span_sum += span;
        }
        std::printf("arm-a-current workers=%d: wall %.3f ms | cpu_pread_sum %.3f ms | cpu_enqueue(in-lock) %.3f ms | "
                    "cpu_lock_wait_sum %.3f ms | dma_span %.3f ms | %.2f MiB -> %.2f GB/s by span, %.2f GB/s by wall "
                    "(cpu sums are CPU work across threads, NOT wall)\n",
                    workers, median(walls), pread_sum / reps, enq_sum / reps, wait_sum / reps, span_sum / reps,
                    (double) total / (1 << 20), (double) total / 1e6 / std::max(span_sum / reps, 1e-9),
                    (double) total / 1e6 / std::max(median(walls), 1e-9));
    }

    // -------- arm (a+spin): the engine worker pattern WITH a concurrent doorbell-style poll --------
    // One block spins on a mapped host flag on a separate stream for the whole batch, mirroring
    // the verifier graph's wait kernel; the host releases it after the batch joins.  Isolates the
    // PCIe-poll-vs-staging-DMA contention from every other engine-context difference.
    {
        // Exception safety: the spinner must ALWAYS be released (a spinning kernel would hold
        // the GPU forever), so every release path funnels through the catch below.
        uint32_t* hflag = nullptr; uint32_t* dflag = nullptr;
        cudaStream_t spin = nullptr;
        try {
        if (cudaHostAlloc((void**) &hflag, 4, cudaHostAllocMapped) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &dflag, hflag, 0) != cudaSuccess ||
            cudaStreamCreateWithFlags(&spin, cudaStreamNonBlocking) != cudaSuccess)
            throw std::runtime_error("spin arm alloc failed");
        WorkerPool pool;
        pool.start(15, max_plane, dma, &experts, dst);
        *hflag = 0;
        exl3_transport_spin_poll(dflag, 1u << 30, spin);
        pool.run(jobs);
        (void) cudaStreamSynchronize(dma);
        *hflag = 1u << 30;   // release
        (void) cudaStreamSynchronize(spin);
        std::vector<double> walls; double span_sum = 0;
        for (int r = 0; r < reps; ++r) {
            const Clock::time_point t0 = Clock::now();
            (void) cudaEventRecord(ev0, dma);
            exl3_transport_spin_poll(dflag, 1u << 30, spin);
            pool.run(jobs);
            const Clock::time_point t1 = Clock::now();
            (void) cudaEventRecord(ev1, dma);
            (void) cudaStreamSynchronize(dma);
            *hflag = 1u << 30;
            (void) cudaStreamSynchronize(spin);
            float span = -1;
            (void) cudaEventElapsedTime(&span, ev0, ev1);
            walls.push_back(ms_between(t0, t1));
            span_sum += span;
            *hflag = 0;   // re-arm for the next rep (the spinner of the next rep is launched above)
        }
        std::printf("arm-a-spin workers=15: wall %.3f ms | dma_span %.3f ms | %.2f MiB -> %.2f GB/s by span "
                    "(doorbell-style PCIe poll concurrent with the staging DMA)\n",
                    median(walls), span_sum / reps, (double) total / (1 << 20),
                    (double) total / 1e6 / std::max(span_sum / reps, 1e-9));
        } catch (...) {
            if (hflag != nullptr) *hflag = 1u << 30;   // never leave the GPU spinning
            if (spin != nullptr) (void) cudaStreamSynchronize(spin);
            throw;
        }
        (void) cudaStreamDestroy(spin);
        (void) cudaFreeHost(hflag);
    }

    // -------- arm (b): pinned source, the SAME 9-plane enqueue pattern (no pread, no workers) --------
    uint8_t* pinned = nullptr;
    if (cudaMallocHost((void**) &pinned, total) != cudaSuccess) throw std::runtime_error("pinned alloc failed");
    guard.pinned_g = pinned;
    for (int e = 0; e < K; ++e) {   // one-time preload (setup, untimed)
        size_t at = 0;
        for (int p = 0; p < 9; ++p) {
            const Plane& pl = experts[(size_t) e].p[(size_t) p];
            const int fd = ::open(pl.path.c_str(), O_RDONLY);
            if (fd < 0 || ::pread(fd, pinned + (size_t) e * experts[0].bytes + at, pl.bytes, (off_t) pl.off) != (ssize_t) pl.bytes)
                throw std::runtime_error("preload pread failed");
            ::close(fd);
            at += pl.bytes;
        }
    }
    {
        (void) cudaStreamSynchronize(dma);
        std::vector<double> walls; double span_sum = 0;
        for (int r = 0; r < reps; ++r) {
            const Clock::time_point t0 = Clock::now();
            (void) cudaEventRecord(ev0, dma);
            for (const auto& [e, p] : jobs) {   // identical transfer sizes/order, one enqueuer
                size_t at = 0;
                for (int q = 0; q < (int) p; ++q) at += experts[e].p[(size_t) q].bytes;
                if (cudaMemcpyAsync(dst + e * experts[0].bytes + at, pinned + e * experts[0].bytes + at,
                                    experts[e].p[(size_t) p].bytes, cudaMemcpyHostToDevice, dma) != cudaSuccess)
                    throw std::runtime_error("b copy failed");
            }
            (void) cudaEventRecord(ev1, dma);
            (void) cudaStreamSynchronize(dma);
            float span = -1;
            (void) cudaEventElapsedTime(&span, ev0, ev1);
            walls.push_back(ms_between(t0, Clock::now()));
            span_sum += span;
        }
        std::printf("arm-b-pinned-9plane: wall %.3f ms | dma_span %.3f ms | %.2f MiB -> %.2f GB/s by span "
                    "(isolates DMA+enqueue from pread/workers)\n",
                    median(walls), span_sum / reps, (double) total / (1 << 20),
                    (double) total / 1e6 / std::max(span_sum / reps, 1e-9));
        std::vector<double> spans;   // (b-one-big): single contiguous H2D - bandwidth UPPER BOUND
        for (int r = 0; r < reps; ++r) {
            (void) cudaEventRecord(ev0, dma);
            if (cudaMemcpyAsync(dst, pinned, total, cudaMemcpyHostToDevice, dma) != cudaSuccess)
                throw std::runtime_error("big copy failed");
            (void) cudaEventRecord(ev1, dma);
            (void) cudaStreamSynchronize(dma);
            float span = -1;
            (void) cudaEventElapsedTime(&span, ev0, ev1);
            spans.push_back(span);
        }
        std::printf("arm-b-one-big: dma_span median %.3f ms | %.2f MiB -> %.2f GB/s - bandwidth UPPER BOUND, "
                    "not an engine prediction\n",
                    median(spans), (double) total / (1 << 20), (double) total / 1e6 / std::max(median(spans), 1e-9));
    }

    // -------- arm (c): pack cost (CPU, timed separately) + one H2D per expert --------
    {
        uint8_t* packed = nullptr;
        if (cudaMallocHost((void**) &packed, experts[0].bytes) != cudaSuccess)
            throw std::runtime_error("pack buffer failed");
        double pack_ms = 0;
        for (int r = 0; r < reps; ++r) {   // one expert's 9 planes -> contiguous, memcpy only
            const Clock::time_point t0 = Clock::now();
            size_t at = 0;
            for (int p = 0; p < 9; ++p) {
                std::memcpy(packed + at, pinned + at, experts[0].p[(size_t) p].bytes);
                at += experts[0].p[(size_t) p].bytes;
            }
            pack_ms += ms_between(t0, Clock::now());
        }
        (void) cudaFreeHost(packed);
        std::vector<double> walls; double span_sum = 0;
        for (int r = 0; r < reps; ++r) {
            const Clock::time_point t0 = Clock::now();
            (void) cudaEventRecord(ev0, dma);
            for (int e = 0; e < K; ++e)   // one contiguous H2D per expert
                if (cudaMemcpyAsync(dst + (size_t) e * experts[0].bytes, pinned + (size_t) e * experts[0].bytes,
                                    experts[0].bytes, cudaMemcpyHostToDevice, dma) != cudaSuccess)
                    throw std::runtime_error("c copy failed");
            (void) cudaEventRecord(ev1, dma);
            (void) cudaStreamSynchronize(dma);
            float span = -1;
            (void) cudaEventElapsedTime(&span, ev0, ev1);
            walls.push_back(ms_between(t0, Clock::now()));
            span_sum += span;
        }
        std::printf("arm-c-packed: pack CPU %.3f ms/expert (memcpy only, timed separately) | K-H2D wall %.3f ms | "
                    "dma_span %.3f ms | %.2f MiB -> %.2f GB/s by span\n",
                    pack_ms / reps, median(walls), span_sum / reps, (double) total / (1 << 20),
                    (double) total / 1e6 / std::max(span_sum / reps, 1e-9));
    }

    // ================= part B: the split's pure issue cost (device-resident, no reads) =================
    {
        const int n_dev = 10, width = 2560, hidden = 640;
        std::vector<uint8_t*> dev((size_t) n_dev * 9, nullptr);
        std::vector<Exl3Mat> hg((size_t) n_dev), hu((size_t) n_dev), hd((size_t) n_dev);
        for (int e = 0; e < n_dev; ++e) {
            const std::string b = "model.language_model.layers." + std::to_string(layer) + ".mlp.experts." +
                                  std::to_string(e + 100) + ".";
            const strata::Exl3Linear lin[3] = {model.linear(b + "gate_proj"), model.linear(b + "up_proj"),
                                               model.linear(b + "down_proj")};
            Exl3Mat* mats[3] = {&hg[(size_t) e], &hu[(size_t) e], &hd[(size_t) e]};
            for (int i = 0; i < 3; ++i) {
                const size_t tr = (size_t) lin[i].ki * lin[i].nj * 32 * lin[i].bits;
                for (int p = 0; p < 3; ++p) {
                    const size_t bytes = p == 0 ? tr : (size_t) (p == 1 ? lin[i].in() : lin[i].out()) * 2;
                    if (cudaMalloc((void**) &dev[(size_t) e * 9 + (size_t) i * 3 + p], bytes) != cudaSuccess)
                        throw std::runtime_error("part-B alloc failed");
                    if (p == 0 && cudaMemcpy(dev[(size_t) e * 9 + (size_t) i * 3], lin[i].trellis, tr,
                                             cudaMemcpyHostToDevice) != cudaSuccess)
                        throw std::runtime_error("part-B upload failed");
                    if (p > 0 && cudaMemset(dev[(size_t) e * 9 + (size_t) i * 3 + p], 0x38, bytes) != cudaSuccess)
                        throw std::runtime_error("part-B scale init failed");
                }
                mats[i]->trellis = (const uint16_t*) dev[(size_t) e * 9 + (size_t) i * 3];
                mats[i]->suh = (const uint16_t*) dev[(size_t) e * 9 + (size_t) i * 3 + 1];
                mats[i]->svh = (const uint16_t*) dev[(size_t) e * 9 + (size_t) i * 3 + 2];
                mats[i]->ki = lin[i].ki; mats[i]->nj = lin[i].nj;
                mats[i]->bits = lin[i].bits; mats[i]->cb = lin[i].cb;
            }
        }
        cudaStream_t cs;
        uint16_t* dx = nullptr; uint16_t* drows = nullptr; int* drows_idx = nullptr; int* dslots = nullptr;
        Exl3Mat *dtg = nullptr, *dtu = nullptr, *dtd = nullptr;
        if (cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) != cudaSuccess ||
            cudaMalloc((void**) &dx, (size_t) 4 * width * 2) != cudaSuccess ||
            cudaMalloc((void**) &drows, (size_t) 40 * width * 2) != cudaSuccess ||
            cudaMalloc((void**) &drows_idx, 40 * sizeof(int)) != cudaSuccess ||
            cudaMalloc((void**) &dslots, 40 * sizeof(int)) != cudaSuccess ||
            cudaMalloc((void**) &dtg, (size_t) n_dev * sizeof(Exl3Mat)) != cudaSuccess ||
            cudaMalloc((void**) &dtu, (size_t) n_dev * sizeof(Exl3Mat)) != cudaSuccess ||
            cudaMalloc((void**) &dtd, (size_t) n_dev * sizeof(Exl3Mat)) != cudaSuccess)
            throw std::runtime_error("part-B stream/buffers failed");
        if (cudaMemset(dx, 0x38, (size_t) 4 * width * 2) != cudaSuccess ||
            cudaMemcpy(dtg, hg.data(), (size_t) n_dev * sizeof(Exl3Mat), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(dtu, hu.data(), (size_t) n_dev * sizeof(Exl3Mat), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(dtd, hd.data(), (size_t) n_dev * sizeof(Exl3Mat), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("part-B init failed");
        struct BGuard { ~BGuard() { (void) cudaStreamDestroy(cs_g); (void) cudaFree(dx_g); (void) cudaFree(drows_g);
                                    (void) cudaFree(drows_idx_g); (void) cudaFree(dslots_g); (void) cudaFree(dtg_g);
                                    (void) cudaFree(dtu_g); (void) cudaFree(dtd_g);
                                    for (uint8_t* p : dev_g) (void) cudaFree(p); }
            cudaStream_t cs_g; uint16_t *dx_g, *drows_g; int *drows_idx_g, *dslots_g;
            Exl3Mat *dtg_g, *dtu_g, *dtd_g; std::vector<uint8_t*>& dev_g; } bguard{cs, dx, drows, drows_idx, dslots,
                                                                                    dtg, dtu, dtd, dev};
        Exl3MoeWorkspace ws;
        ws.reserve(40, width, hidden, width, true);   // PRE-RESERVED outside every timed loop

        struct Ffn { int routes = 0; std::vector<int> rows; std::vector<Exl3Mat> g, u, d; std::vector<int> slot; };
        auto build = [&](int tokens, int k) {
            Ffn f; f.routes = tokens * k;
            for (int t = 0; t < tokens; ++t)
                for (int j = 0; j < k; ++j) {
                    const int e = (j * 3 + t) % n_dev;
                    f.rows.push_back(t); f.slot.push_back(e);
                    f.g.push_back(hg[(size_t) e]); f.u.push_back(hu[(size_t) e]); f.d.push_back(hd[(size_t) e]);
                }
            return f;
        };
        auto upload_rows = [&](const int* rows, int n) {
            if (cudaMemcpyAsync(drows_idx, rows, (size_t) n * sizeof(int), cudaMemcpyHostToDevice, cs) != cudaSuccess)
                throw std::runtime_error("rows upload failed");
        };
        using strata::kernels::exl3_moe_rows_indexed;

        for (const auto& [tokens, k, split] : {std::make_tuple(1, 10, 9), std::make_tuple(4, 10, 35)}) {
            const Ffn f = build(tokens, k);
            const int R = f.routes, S = split;
            const size_t rows_bytes = (size_t) R * width * 2;

            // reference: one single all-routes call
            upload_rows(f.rows.data(), R);
            exl3_moe_rows_indexed(f.g.data(), f.u.data(), f.d.data(), R, dx, drows_idx, drows, cs, ws);
            (void) cudaStreamSynchronize(cs);
            std::vector<uint16_t> ref(rows_bytes / 2);
            if (cudaMemcpy(ref.data(), drows, rows_bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("ref readback failed");
            (void) cudaMemset(drows, 0, rows_bytes);

            {   // two offset passes, same math - must be byte-identical
                upload_rows(f.rows.data(), S);
                exl3_moe_rows_indexed(f.g.data(), f.u.data(), f.d.data(), S, dx, drows_idx, drows, cs, ws);
                upload_rows(f.rows.data() + S, R - S);
                exl3_moe_rows_indexed(f.g.data() + S, f.u.data() + S, f.d.data() + S, R - S, dx, drows_idx,
                                      drows + (size_t) S * width, cs, ws);
                (void) cudaStreamSynchronize(cs);
                std::vector<uint16_t> two(rows_bytes / 2);
                if (cudaMemcpy(two.data(), drows, rows_bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
                    throw std::runtime_error("two readback failed");
                std::printf("part-B T=%d routes=%d split=%d+%d: two-pass rows %s\n", tokens, R, S, R - S,
                            std::memcmp(ref.data(), two.data(), rows_bytes) == 0 ? "byte-identical PASS" : "FAIL");
                (void) cudaMemset(drows, 0, rows_bytes);
            }
            {   // READONLY probe: the prebuilt slot-table primitive, byte-identity + cost
                if (cudaMemcpyAsync(dslots, f.slot.data(), (size_t) R * sizeof(int), cudaMemcpyHostToDevice, cs) != cudaSuccess)
                    throw std::runtime_error("slots upload failed");
                strata::kernels::exl3_moe_rows_slot_table(dtg, dtu, dtd, n_dev, dslots, R, width, hidden, width,
                                                          dx, drows_idx, drows, cs, ws);
                upload_rows(f.rows.data(), R);
                (void) cudaStreamSynchronize(cs);
                std::vector<uint16_t> stl(rows_bytes / 2);
                if (cudaMemcpy(stl.data(), drows, rows_bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
                    throw std::runtime_error("slot readback failed");
                std::printf("part-B T=%d slot-table-prebuilt: rows %s (cost includes the primitive's per-call "
                            "slot-id readback validation)\n", tokens,
                            std::memcmp(ref.data(), stl.data(), rows_bytes) == 0 ? "byte-identical PASS" : "FAIL");
                (void) cudaMemset(drows, 0, rows_bytes);
            }

            // timing: host issue wall + GPU span (no reads; workspace pre-reserved; events synced first)
            auto time_arm = [&](const char* name, auto&& fn) {
                for (int i = 0; i < 5; ++i) fn();
                (void) cudaStreamSynchronize(cs);
                std::vector<double> issue; double span_sum = 0;
                for (int r = 0; r < reps; ++r) {
                    const Clock::time_point t0 = Clock::now();
                    (void) cudaEventRecord(ev0, cs);
                    fn();
                    (void) cudaEventRecord(ev1, cs);
                    const Clock::time_point t1 = Clock::now();
                    (void) cudaStreamSynchronize(cs);
                    float span = -1;
                    (void) cudaEventElapsedTime(&span, ev0, ev1);
                    issue.push_back(ms_between(t0, t1));
                    span_sum += span;
                }
                std::printf("part-B T=%d %s: host-issue %.3f ms | gpu_span %.3f ms\n",
                            tokens, name, median(issue), span_sum / reps);
            };
            time_arm("single-all-routes", [&] {
                upload_rows(f.rows.data(), R);
                exl3_moe_rows_indexed(f.g.data(), f.u.data(), f.d.data(), R, dx, drows_idx, drows, cs, ws);
            });
            time_arm("two-offset-passes", [&] {
                upload_rows(f.rows.data(), S);
                exl3_moe_rows_indexed(f.g.data(), f.u.data(), f.d.data(), S, dx, drows_idx, drows, cs, ws);
                upload_rows(f.rows.data() + S, R - S);
                exl3_moe_rows_indexed(f.g.data() + S, f.u.data() + S, f.d.data() + S, R - S, dx, drows_idx,
                                      drows + (size_t) S * width, cs, ws);
            });
            time_arm("slot-table-prebuilt", [&] {
                (void) cudaMemcpyAsync(dslots, f.slot.data(), (size_t) R * sizeof(int), cudaMemcpyHostToDevice, cs);
                upload_rows(f.rows.data(), R);
                strata::kernels::exl3_moe_rows_slot_table(dtg, dtu, dtd, n_dev, dslots, R, width, hidden, width,
                                                          dx, drows_idx, drows, cs, ws);
            });
        }
    }
    std::printf("transport-bench done\n");
    return 0;
}
