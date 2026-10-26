// Real-model graph seam: dynamic residency, replay, all-hit no-host progress and miss-slot safety.
#include "strata/kernels/exl3_experts.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using strata::kernels::Exl3ExpertStream;
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template <class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t count) { check(cudaMalloc((void**) &p, count * sizeof(T))); }
    ~Device() { if (p) (void) cudaFree(p); }
};
template <class T> struct Mapped {
    T* h = nullptr; T* d = nullptr;
    explicit Mapped(size_t count) {
        check(cudaHostAlloc((void**) &h, count * sizeof(T), cudaHostAllocMapped));
        check(cudaHostGetDevicePointer((void**) &d, h, 0));
        std::memset(h, 0, count * sizeof(T));
    }
    ~Mapped() { if (h) (void) cudaFreeHost(h); }
};

bool exercise(const char* model, int slots) {
    constexpr int n = 2560, k = 10, max_t = 8, max_routes = max_t * k;
    // Reference never evicts, isolating graph slot reuse from the oracle's cache policy.
    Exl3ExpertStream oracle(model, n, 1, 512, nullptr, 0, 0, 0, 1);
    Exl3ExpertStream graph_stream(model, n, 1, slots, nullptr, 0, 0, 0, 1);
    std::vector<float> x(max_t * n), expected(max_routes * n), warm(max_routes * n);
    std::vector<int32_t> ids(max_routes);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin((float) i * .017f) * .15f;
    for (int r = 0; r < max_routes; ++r) ids[r] = r;
    if (!oracle.run_multi(0, x.data(), ids.data(), max_t, k, warm.data(), nullptr) ||
        !graph_stream.run_multi(0, x.data(), ids.data(), max_t, k, warm.data(), nullptr))
        throw std::runtime_error("warmup failed");
    graph_stream.enable_verify_graph(max_t, k);
    cudaStream_t cs = nullptr;
    check(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
    Device<float> dx(x.size()), out(expected.size() + 2);
    Device<int32_t> di(ids.size());
    Mapped<float> mx(x.size()), my(expected.size());
    Mapped<int32_t> mi(ids.size());
    Mapped<uint32_t> seq(1), flag(1);
    bool ok = true;
    // Capture each shape once, then change expert IDs/residency and inputs under the SAME graph.
    for (int t : {1, 4, 8}) {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        check(cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal));
        graph_stream.verify_graph_publish(0, dx.p, di.p, t, k, mx.d, mi.d, seq.d, 1, cs);
        graph_stream.verify_graph_hits(0, dx.p, di.p, t, k, out.p + 1, 1, cs);
        strata::kernels::wait_flag_ge_or(flag.d, 1, graph_stream.verify_graph_skip(), cs);
        graph_stream.verify_graph_copy_misses(my.d, out.p + 1, t * k, cs);
        check(cudaStreamEndCapture(cs, &graph));
        check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int mode : {0, 1, 2, 0, 0}) {
            // 0: repeat warm routes; 1: mixed hits/misses with duplicates and scrambled tokens;
            // 2: entirely new experts; returning to 0 also tests changed slot membership.
            for (int r = 0; r < t * k; ++r)
                ids[r] = mode == 0 ? r : mode == 1 ? (r % 2 ? 90 + r % 9 : r % 8) : 200 + r;
            for (size_t i = 0; i < x.size(); ++i) x[i] += .0001f;
            if (!oracle.run_multi(0, x.data(), ids.data(), t, k, expected.data(), nullptr))
                throw std::runtime_error("oracle failed");
            *seq.h = *flag.h = 0;
            std::fill_n(mx.h, x.size(), -123.f);
            std::fill_n(my.h, expected.size(), -456.f);
            std::vector<float> actual(expected.size() + 2, 12345.f);
            check(cudaMemcpyAsync(dx.p, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, cs));
            check(cudaMemcpyAsync(di.p, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice, cs));
            check(cudaMemcpyAsync(out.p, actual.data(), actual.size() * sizeof(float), cudaMemcpyHostToDevice, cs));
            graph_stream.refresh_verify_graph(cs);
            check(cudaGraphLaunch(exec, cs));
            const auto start = std::chrono::steady_clock::now();
            while (*(volatile uint32_t*) seq.h != 1) {
                if (std::chrono::steady_clock::now() - start > std::chrono::seconds(10))
                    throw std::runtime_error("graph never rang");
                std::this_thread::yield();
            }
            const bool host = graph_stream.verify_graph_begin_host(0);
            bool progress = true;
            if (host) {
                if (!graph_stream.run_multi(0, mx.h, mi.h, t, k, my.h, nullptr))
                    throw std::runtime_error("miss callback failed");
            } else {
                // An all-hit graph must finish with flag still ZERO and leave host input untouched.
                cudaError_t q;
                do {
                    q = cudaStreamQuery(cs);
                    if (std::chrono::steady_clock::now() - start > std::chrono::seconds(10)) {
                        progress = false; break;
                    }
                } while (q == cudaErrorNotReady);
                progress = progress && q == cudaSuccess && mx.h[0] == -123.f;
            }
            graph_stream.verify_graph_end_host(0);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) flag.h = 1;
            check(cudaStreamSynchronize(cs));
            check(cudaMemcpy(actual.data(), out.p, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
            const bool identical = std::memcmp(actual.data() + 1, expected.data(), (size_t) t * k * n * sizeof(float)) == 0;
            if (!identical) {
                int different = 0;
                float worst = 0;
                for (int i = 0; i < t * k * n; ++i) {
                    different += actual[i + 1] != expected[i];
                    worst = std::max(worst, std::abs(actual[i + 1] - expected[i]));
                }
                std::printf("  different=%d max_abs=%g\n", different, worst);
                std::vector<float> repeat(expected.size());
                if (!oracle.run_multi(0, x.data(), ids.data(), t, k, repeat.data(), nullptr))
                    throw std::runtime_error("repeat oracle failed");
                std::printf("  repeat_oracle_identical=%d repeat_matches_graph=%d\n",
                    std::memcmp(repeat.data(), expected.data(), (size_t) t*k*n*sizeof(float)) == 0,
                    std::memcmp(repeat.data(), actual.data()+1, (size_t) t*k*n*sizeof(float)) == 0);
            }
            const bool sentinels = actual.front() == 12345.f && actual[(size_t) t * k * n + 1] == 12345.f;
            const bool finite = std::all_of(actual.begin() + 1, actual.begin() + 1 + t * k * n,
                                           [](float f) { return std::isfinite(f); });
            const bool pass = identical && sentinels && progress && finite;
            ok &= pass;
            std::printf("slots=%d T=%d mode=%d host=%d identical=%d sentinels=%d progress=%d %s\n",
                slots, t, mode, (int) host, (int) identical, (int) sentinels, (int) progress, pass ? "PASS" : "FAIL");
        }
        check(cudaGraphExecDestroy(exec));
        check(cudaGraphDestroy(graph));
    }
    check(cudaStreamDestroy(cs));
    return ok;
}
} // namespace

int main(int argc, char** argv) {
    const char* model = argc > 1 ? argv[1] : std::getenv("STRATA_EXL3_TEST_MODEL");
    int devices = 0;
    if (!model || cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) return 77;
    try {
        const bool normal = exercise(model, 80);
        const bool small = exercise(model, 2); // too-small active cache: host waves, no graph hits
        return normal && small ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exl3_verify_graph_test: %s\n", e.what());
        return 1;
    }
}
