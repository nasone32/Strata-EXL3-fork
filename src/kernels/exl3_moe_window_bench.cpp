// Resident EXL3 window compute benchmark at the model's actual 2560/640 geometry.
// No weight uploads in the timed loop. This is not an end-to-end engine benchmark.
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/exl3_experts.hpp"
#include "strata/kernels/elementwise.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace strata::kernels;
static void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template <class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t size) { check(cudaMalloc((void**) &p, size * sizeof(T))); }
    ~Buffer() { (void) cudaFree(p); }
    Buffer(const Buffer&) = delete;
};
struct Matrix {
    Buffer<uint16_t> tr, su, sv;
    Exl3Mat gpu;
    Matrix(int in, int out, std::mt19937& rng, int bits = 6, int frac = 0)
        : tr((size_t) (in/16)*(out/16)*(16*bits + 8*frac)), su(in), sv(out) {
        std::vector<uint16_t> packed((size_t) (in/16)*(out/16)*(16*bits + 8*frac)), scales((size_t) std::max(in,out));
        for (auto& v : packed) v = (uint16_t) rng();
        for (auto& v : scales) v = __half_as_ushort(__float2half(0.025f));
        check(cudaMemcpy(tr.p,packed.data(),packed.size()*2,cudaMemcpyHostToDevice));
        check(cudaMemcpy(su.p,scales.data(),in*2,cudaMemcpyHostToDevice));
        check(cudaMemcpy(sv.p,scales.data(),out*2,cudaMemcpyHostToDevice));
        gpu = {tr.p,su.p,sv.p,in/16,out/16,bits,2,frac};
    }
};
int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--copy-costs") == 0) {
        try {
            int count=0; check(cudaGetDeviceCount(&count));
            constexpr size_t expert_bytes=1843200, capacity=10*expert_bytes;
            for (int device=0;device<std::min(count,2);++device) {
                check(cudaSetDevice(device));
                void* host=nullptr; check(cudaHostAlloc(&host,capacity,cudaHostAllocPortable));
                struct FreePinned { void* p; ~FreePinned(){(void)cudaFreeHost(p);} } cleanup{host};
                std::memset(host,17,capacity);
                Buffer<uint8_t> dst(capacity);
                cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
                cudaEvent_t begin,end; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
                for(int groups:{1,2,4,8,10}) {
                    const size_t bytes=expert_bytes*groups;
                    for(int i=0;i<10;++i) check(cudaMemcpyAsync(dst.p,host,bytes,cudaMemcpyHostToDevice,stream));
                    check(cudaEventRecord(begin,stream));
                    for(int i=0;i<100;++i) check(cudaMemcpyAsync(dst.p,host,bytes,cudaMemcpyHostToDevice,stream));
                    check(cudaEventRecord(end,stream)); check(cudaEventSynchronize(end));
                    float ms=0; check(cudaEventElapsedTime(&ms,begin,end));
                    std::printf("GPU_COPY_COST device=%d groups=%d bytes=%zu us=%.3f GB_s=%.3f pinned=1\n",
                                device,groups,bytes,ms*10,bytes/(ms*10*1000));
                    std::fflush(stdout);
                }
                check(cudaEventDestroy(begin)); check(cudaEventDestroy(end)); check(cudaStreamDestroy(stream));
            }
            return 0;
        } catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
    }
    if (argc > 2 && std::strcmp(argv[1], "--stream-costs") == 0) {
        try {
            check(cudaSetDevice(0));
            Exl3ExpertStream experts(argv[2],2560,48,0,nullptr,0,(size_t)512<<20,5,6,8);
            std::mt19937 rng(1234);
            std::normal_distribution<float> dist(0.0f,0.5f);
            std::vector<float> x(4*2560), out(40*2560);
            for (auto& v : x) v=dist(rng);
            for (int tokens : {1,4}) for (int groups : {1,2,4,8,10}) {
                std::vector<int32_t> ids(tokens*groups);
                auto run = [&](int base) {
                    for (int r=0;r<tokens*groups;++r) ids[r]=(base+r%groups)%256;
                    if (!experts.run_multi(5,x.data(),ids.data(),tokens,groups,out.data(),nullptr))
                        throw std::runtime_error("stream cost: expert service failed");
                };
                for (int i=0;i<(256+groups-1)/groups;++i) run(i*groups);
                auto start=std::chrono::steady_clock::now();
                for (int i=0;i<128;++i) run(i*groups);
                const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/128;
                for (int r=0;r<tokens*groups*2560;++r)
                    if (!std::isfinite(out[r])) throw std::runtime_error("stream cost: nonfinite output");
                std::printf("GPU_STREAM_COST tokens=%d groups=%d rows=%d rotate_us=%.3f cache_slots=0\n",
                            tokens,groups,tokens*groups,us);
                std::fflush(stdout);
            }
            return 0;
        } catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
    }
    const int routes = argc > 1 ? std::atoi(argv[1]) : 40;
    const int device = argc > 2 ? std::atoi(argv[2]) : 0;
    const bool group_mode = argc > 3 && (std::strcmp(argv[3], "group") == 0 || std::strcmp(argv[3], "mimo-group") == 0);
    if (routes < 1 || routes > 512) return 2;
    try {
        check(cudaSetDevice(device));
        if (argc > 3 && std::strcmp(argv[3], "mimo") == 0) {
            // Actual MiMo decode geometry, distinct resident experts, one shared token. Plans
            // are device-resident, so the graph timing isolates the kernels from host dispatch.
            if (routes > 8) throw std::runtime_error("MiMo decode bench supports 1..8 routes");
            const int frac = argc > 4 && std::strchr(argv[4], '.') ? 1 : 0;
            std::mt19937 rng(1234);
            std::vector<Matrix*> weights;
            struct Cleanup { std::vector<Matrix*>& v; ~Cleanup(){for(auto* p:v) delete p;} } cleanup{weights};
            std::vector<Exl3Mat> g, u, d;
            for (int e = 0; e < routes; ++e) {
                weights.push_back(new Matrix(4096, 2048, rng, 2, frac)); g.push_back(weights.back()->gpu);
                weights.push_back(new Matrix(4096, 2048, rng, 2, frac)); u.push_back(weights.back()->gpu);
                weights.push_back(new Matrix(2048, 4096, rng, 2, frac)); d.push_back(weights.back()->gpu);
            }
            Buffer<Exl3Mat> dg(routes), du(routes), dd(routes);
            check(cudaMemcpy(dg.p,g.data(),routes*sizeof(Exl3Mat),cudaMemcpyHostToDevice));
            check(cudaMemcpy(du.p,u.data(),routes*sizeof(Exl3Mat),cudaMemcpyHostToDevice));
            check(cudaMemcpy(dd.p,d.data(),routes*sizeof(Exl3Mat),cudaMemcpyHostToDevice));
            Buffer<int> row(routes); check(cudaMemset(row.p,0,routes*sizeof(int)));
            Buffer<uint16_t> x(4096), y((size_t) routes*4096);
            std::vector<uint16_t> input(4096);
            for (auto& v:input) v=__half_as_ushort(__float2half(((int)(rng()%2000)-1000)*0.001f));
            check(cudaMemcpy(x.p,input.data(),input.size()*2,cudaMemcpyHostToDevice));
            cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            Exl3MoeWorkspace ws; ws.reserve_runtime_scratch(routes,4096,2048,4096);
            const Exl3MoeDevicePlan plan{dg.p,du.p,dd.p,nullptr,nullptr};
            auto run=[&]{exl3_moe_rows_indexed_device_plan(g.data(),u.data(),d.data(),routes,x.p,row.p,y.p,plan,stream,ws);};
            for(int i=0;i<10;++i) run();
            check(cudaStreamSynchronize(stream));
            cudaGraph_t graph; cudaGraphExec_t exec;
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)); run();
            check(cudaStreamEndCapture(stream,&graph)); check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
            cudaEvent_t a,b; check(cudaEventCreate(&a));check(cudaEventCreate(&b));
            for(int mode=0;mode<2;++mode) {
                check(cudaEventRecord(a,stream));
                for(int i=0;i<200;++i) { if(mode)check(cudaGraphLaunch(exec,stream));else run(); }
                check(cudaEventRecord(b,stream));check(cudaEventSynchronize(b));
                float ms=0;check(cudaEventElapsedTime(&ms,a,b));
                std::printf("MIMO_FFN device=%d K=%s routes=%d native=%s mode=%s us=%.3f\n",device,frac?"2.5":"2",routes,
                            std::getenv("STRATA_EXL3_RDNA_NATIVE") ? std::getenv("STRATA_EXL3_RDNA_NATIVE") : "0",
                            mode?"graph":"eager",ms*5);
            }
            std::vector<uint16_t> output((size_t) routes*4096);
            check(cudaMemcpy(output.data(),y.p,output.size()*2,cudaMemcpyDeviceToHost));
            for(auto v:output) if(!std::isfinite(__half2float(__ushort_as_half(v)))) throw std::runtime_error("nonfinite MiMo FFN output");
            check(cudaEventDestroy(a));check(cudaEventDestroy(b));check(cudaGraphExecDestroy(exec));
            check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
            return 0;
        }
        if (argc > 3 && std::strcmp(argv[3], "verify") == 0) {
            // DFlash VERIFICATION-window shape at MiMo geometry (4096 -> 2048 -> 4096): T = M+1
            // target rows, k = 8 routed experts per row, only the PEER-OWNED share of the routes
            // crosses (the engine's kind==2 mask), experts repeat ACROSS rows (never within one
            // row's top-k), and the pool mixes K2/K2.5 exactly like the model's trellis.  Two arms
            // replay the peer's ACTUAL enqueue chains on identical routes:
            //   indexed  (today): f32 x H2D + convert + idx H2D + 3 descriptor H2D (inside
            //             exl3_moe_rows_indexed) + the native/RDNA kernel chain + rows D2H
            //   grouped  (candidate, PEER_MULTIROW=2 + frac): ONE pinned plan H2D (x, idx,
            //             per-group descriptors, group tables) + convert + 3 fused kernels
            //             (direct output, no D2D) + rows D2H
            // Parity: an INDEPENDENT per-route scalar oracle (exl3_moe_rows, one route per call -
            // the exl3_peer_test oracle) is the byte-level ground truth; byte identity between the
            // arms and against it is REPORTED, never assumed (the indexed-native arm's fdot2
            // arithmetic and the fused arm's fp32 FMA order are different kernels).
            const char* ms = argc > 4 ? argv[4] : "3,4,5,8";
            const char* shares = argc > 5 ? argv[5] : "10,25,40";
            const char* kmodes = argc > 6 ? argv[6] : "mixed,k2,k25";
            auto list = [](const char* s) {
                std::vector<int> out;
                std::string t;
                std::stringstream ss(s);
                while (std::getline(ss, t, ',')) if (!t.empty()) out.push_back(std::atoi(t.c_str()));
                return out;
            };
            for (const char* env : {"STRATA_EXL3_RDNA", "STRATA_EXL3_FUSE", "STRATA_EXL3_RDNA_NATIVE",
                                    "STRATA_EXL3_NATIVE_GROUPED", "STRATA_EXL3_GEMV_SPLIT"})
                std::printf("VERIFY_MIMO env %s=%s\n", env, std::getenv(env) ? std::getenv(env) : "unset");
            std::fflush(stdout);
            constexpr int hidden = 4096, intermediate = 2048, topk = 8, pool = 48, shared_pool = 6;
            constexpr double owner_p = 0.4;   // measured peer share of routed rows (profiling 2026-11-01: ~40%)
            for (const std::string& kmode_s : std::vector<std::string>{"mixed", "k2", "k25"}) {
                if (!std::strstr(kmodes, kmode_s.c_str())) continue;
                std::mt19937 rng(kmode_s == "mixed" ? 20261103u : (kmode_s == "k2" ? 20261104u : 20261105u));
                // resident weight pool: K2/K2.5 interleaved (the model mixes them per expert)
                std::vector<Matrix*> weights;
                struct Wcleanup { std::vector<Matrix*>& v; ~Wcleanup() { for (auto* p : v) delete p; } } wc{weights};
                std::vector<Exl3Mat> pg, pu, pd;   // pool descriptors, expert id -> index
                for (int e = 0; e < pool; ++e) {
                    const int frac = kmode_s == "k25" ? 1 : (kmode_s == "k2" ? 0 : (e & 1));
                    weights.push_back(new Matrix(hidden, intermediate, rng, 2, frac));
                    pg.push_back(weights.back()->gpu);
                    weights.push_back(new Matrix(hidden, intermediate, rng, 2, frac));
                    pu.push_back(weights.back()->gpu);
                    weights.push_back(new Matrix(intermediate, hidden, rng, 2, frac));
                    pd.push_back(weights.back()->gpu);
                }
                for (int M : list(ms)) {
                    if (M < 1 || M > 8) continue;   // window rows = M + 1 <= 9 (DFlash block 8 + anchor)
                    const int T = M + 1;
                    for (int share : list(shares)) {
                        if (share < 0 || share > 100) continue;
                        // ---- route generation (deterministic): the window's rows share `share`% of
                        // their experts through a 6-expert hot pool, the rest are drawn apart
                        std::mt19937 rr((uint32_t) (1000003u * M + 1009 * (uint32_t) share +
                                                    (kmode_s == "mixed" ? 3u : kmode_s == "k2" ? 5u : 7u)));
                        const int n_shared = share * topk / 100;
                        std::vector<int32_t> ids((size_t) T * topk, -1);
                        std::vector<uint8_t> owner((size_t) T * topk, 0);
                        int hot[shared_pool];
                        for (int i = 0; i < shared_pool; ++i) hot[i] = (int) (rr() % pool);
                        for (int t = 0; t < T; ++t) {
                            int picked = 0;
                            std::vector<int> order(shared_pool);
                            for (int i = 0; i < shared_pool; ++i) order[i] = i;
                            std::shuffle(order.begin(), order.end(), rr);
                            for (int i = 0; i < n_shared; ++i) ids[(size_t) t * topk + picked++] = hot[order[i]];
                            for (int guard = 0; picked < topk && guard < 1000; ++guard) {
                                const int cand = (int) (rr() % pool);
                                bool dup = false;
                                for (int i = 0; i < picked; ++i) dup = dup || ids[(size_t) t * topk + i] == cand;
                                if (!dup) ids[(size_t) t * topk + picked++] = cand;
                            }
                            if (picked < topk) throw std::runtime_error("verify bench: route generation failed");
                        }
                        int routes = 0;
                        for (int r = 0; r < T * topk; ++r) {
                            owner[(size_t) r] = (rr() % 1000) < (int) (owner_p * 1000) ? 1 : 0;
                            if (owner[(size_t) r]) ++routes;
                        }
                        if (routes < 2) continue;
                        // ---- the peer's own table build: keep owned routes in route order, then
                        // sort by (expert, route) and chunk groups at the multirow width (this is
                        // Exl3Peer::launch's exact sequence)
                        constexpr int width = 2;
                        std::vector<int> row_of;
                        for (int r = 0; r < T * topk; ++r) if (owner[(size_t) r]) row_of.push_back(r);
                        std::sort(row_of.begin(), row_of.end(), [&](int a, int b) {
                            return ids[a] != ids[b] ? ids[a] < ids[b] : a < b;
                        });
                        std::vector<int> first, grows;
                        std::vector<Exl3Mat> gg, gu, gd;
                        std::vector<int> idx((size_t) routes);
                        for (int q = 0; q < routes; ++q) {
                            const int r = row_of[(size_t) q];
                            idx[(size_t) q] = r / topk;
                            const int e = ids[r];
                            if (q == 0 || ids[row_of[(size_t) q - 1]] != e || grows.back() == width) {
                                first.push_back(q); grows.push_back(0);
                                gg.push_back(pg[(size_t) e]); gu.push_back(pu[(size_t) e]); gd.push_back(pd[(size_t) e]);
                            }
                            ++grows.back();
                        }
                        const int groups = (int) first.size();
                        int two_row = 0;
                        for (int g : grows) two_row += g == 2;
                        int k25 = 0;
                        for (int g = 0; g < groups; ++g) k25 += gg[(size_t) g].frac * grows[(size_t) g];
                        // ---- buffers: everything the peer uploads/downloads is PINNED (the peer's
                        // own transport is pinned end to end; pageable copies would dominate timing)
                        std::vector<float> hx((size_t) T * hidden);
                        for (auto& v : hx) v = ((int) (rng() % 2000) - 1000) * 0.001f;
                        Buffer<float> dx32((size_t) T * hidden);
                        Buffer<uint16_t> dx16((size_t) T * hidden);
                        Buffer<int> didx((size_t) routes);
                        Buffer<uint16_t> dy((size_t) routes * hidden);
                        check(cudaMemcpy(dx32.p, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
                        check(cudaMemcpy(didx.p, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice));
                        f32_to_f16_bulk(dx32.p, dx16.p, (int64_t) T * hidden, nullptr);
                        cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                        Exl3MoeWorkspace ws;   // one workspace both arms share, like the peer tier
                        ws.reserve(routes, hidden, intermediate, hidden, true);
                        ws.reserve_small_m(routes, groups, hidden, intermediate, hidden);
                        ws.reserve_runtime_scratch(routes, hidden, intermediate, hidden);
                        // pinned host transport: x f32, idx, per-route descriptors (indexed), rows f16 back
                        struct Pinned {
                            void* p = nullptr;
                            size_t n = 0;
                            void alloc(size_t bytes, const void* src, const char* what) {
                                n = bytes;
                                if (cudaHostAlloc(&p, bytes, cudaHostAllocPortable) != cudaSuccess)
                                    throw std::runtime_error(std::string("verify bench: pin ") + what);
                                if (src) std::memcpy(p, src, bytes);
                            }
                            ~Pinned() { if (p) (void) cudaFreeHost(p); }
                        };
                        Pinned pin_x, pin_idx, pin_desc, pin_plan, pin_rows;
                        pin_x.alloc(hx.size() * sizeof(float), hx.data(), "x");
                        pin_idx.alloc(idx.size() * sizeof(int), idx.data(), "idx");
                        pin_rows.alloc((size_t) routes * hidden * sizeof(uint16_t), nullptr, "rows");
                        // indexed arm: the peer's exact indexed chain (pinned x H2D + convert + idx
                        // H2D + 3 pinned descriptor H2D inside the entry + kernel chain + rows D2H)
                        std::vector<Exl3Mat> hg((size_t) routes), hu((size_t) routes), hd((size_t) routes);
                        for (int q = 0; q < routes; ++q) {
                            const int e = ids[row_of[(size_t) q]];   // route-major order == sorted order: the peer
                            hg[(size_t) q] = pg[(size_t) e];         // sorts BEFORE building tables either arm
                            hu[(size_t) q] = pu[(size_t) e];
                            hd[(size_t) q] = pd[(size_t) e];
                        }
                        pin_desc.alloc(3 * (size_t) routes * sizeof(Exl3Mat), nullptr, "desc");
                        std::memcpy((char*) pin_desc.p + 0 * routes * sizeof(Exl3Mat), hg.data(), routes * sizeof(Exl3Mat));
                        std::memcpy((char*) pin_desc.p + 1 * routes * sizeof(Exl3Mat), hu.data(), routes * sizeof(Exl3Mat));
                        std::memcpy((char*) pin_desc.p + 2 * routes * sizeof(Exl3Mat), hd.data(), routes * sizeof(Exl3Mat));
                        auto indexed = [&] {
                            check(cudaMemcpyAsync(dx32.p, pin_x.p, hx.size() * 4, cudaMemcpyHostToDevice, stream));
                            f32_to_f16_bulk(dx32.p, dx16.p, (int64_t) T * hidden, stream);
                            check(cudaGetLastError());
                            check(cudaMemcpyAsync(didx.p, pin_idx.p, idx.size() * 4, cudaMemcpyHostToDevice, stream));
                            exl3_moe_rows_indexed((const Exl3Mat*) pin_desc.p,
                                                  (const Exl3Mat*) pin_desc.p + 1 * routes,
                                                  (const Exl3Mat*) pin_desc.p + 2 * routes,
                                                  routes, dx16.p, didx.p, dy.p, stream, ws);
                            check(cudaMemcpyAsync(pin_rows.p, dy.p, (size_t) routes * hidden * 2,
                                                  cudaMemcpyDeviceToHost, stream));
                        };
                        // grouped arm: the peer's packed-plan chain (one pinned slab H2D, convert,
                        // 3 fused kernels with direct output, rows D2H)
                        struct Slab { size_t x, indices, gate, up, down, first, rows, bytes; } slab{};
                        {
                            size_t at = 0;
                            auto take = [&](size_t bytes) { const size_t o = at; at = (at + bytes + 255) & ~(size_t)255; return o; };
                            slab.x = take((size_t) T * hidden * sizeof(float));
                            slab.indices = take((size_t) routes * sizeof(int));
                            slab.gate = take((size_t) groups * sizeof(Exl3Mat));
                            slab.up = take((size_t) groups * sizeof(Exl3Mat));
                            slab.down = take((size_t) groups * sizeof(Exl3Mat));
                            slab.first = take((size_t) groups * sizeof(int));
                            slab.rows = take((size_t) groups * sizeof(int));
                            slab.bytes = at;
                        }
                        {
                            std::vector<uint8_t> tmp(slab.bytes);
                            auto put = [&](size_t o, const void* src, size_t bytes) { std::memcpy(tmp.data() + o, src, bytes); };
                            put(slab.x, hx.data(), (size_t) T * hidden * sizeof(float));
                            put(slab.indices, idx.data(), (size_t) routes * sizeof(int));
                            put(slab.gate, gg.data(), (size_t) groups * sizeof(Exl3Mat));
                            put(slab.up, gu.data(), (size_t) groups * sizeof(Exl3Mat));
                            put(slab.down, gd.data(), (size_t) groups * sizeof(Exl3Mat));
                            put(slab.first, first.data(), (size_t) groups * sizeof(int));
                            put(slab.rows, grows.data(), (size_t) groups * sizeof(int));
                            pin_plan.alloc(slab.bytes, tmp.data(), "plan");
                        }
                        Buffer<uint8_t> dplan(slab.bytes / sizeof(uint8_t));
                        check(cudaMemcpy(dplan.p, pin_plan.p, slab.bytes, cudaMemcpyHostToDevice));
                        auto grouped = [&] {
                            check(cudaMemcpyAsync(dplan.p, pin_plan.p, slab.bytes, cudaMemcpyHostToDevice, stream));
                            f32_to_f16_bulk((const float*) (dplan.p + slab.x), dx16.p, (int64_t) T * hidden, stream);
                            check(cudaGetLastError());
                            const Exl3MoeDevicePlan plan{(const Exl3Mat*) (dplan.p + slab.gate),
                                (const Exl3Mat*) (dplan.p + slab.up), (const Exl3Mat*) (dplan.p + slab.down),
                                (const int*) (dplan.p + slab.first), (const int*) (dplan.p + slab.rows)};
                            exl3_moe_rows_grouped_device_plan(gg.data(), gu.data(), gd.data(), first.data(),
                                grows.data(), groups, routes, dx16.p, (const int*) (dplan.p + slab.indices),
                                plan, stream, ws, true, dy.p);
                            check(cudaGetLastError());
                            check(cudaMemcpyAsync(pin_rows.p, dy.p, (size_t) routes * hidden * 2,
                                                  cudaMemcpyDeviceToHost, stream));
                        };
                        // ---- independent per-route scalar oracle (one route per call)
                        std::vector<uint16_t> oracle((size_t) routes * hidden), got((size_t) routes * hidden);
                        for (int q = 0; q < routes; ++q) {
                            const int e = ids[row_of[(size_t) q]];
                            exl3_moe_rows(&pg[(size_t) e], &pu[(size_t) e], &pd[(size_t) e],
                                          nullptr, 1, dx16.p + (int64_t) idx[(size_t) q] * hidden,
                                          dy.p + (int64_t) q * hidden, nullptr, &ws);
                            check(cudaGetLastError());
                        }
                        check(cudaStreamSynchronize(nullptr));
                        check(cudaMemcpy(oracle.data(), dy.p, oracle.size() * 2, cudaMemcpyDeviceToHost));
                        auto rel_err = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
                            double num = 0, den = 0;
                            for (size_t i = 0; i < a.size(); ++i) {
                                const double x = __half2float(__ushort_as_half(a[i])), y = __half2float(__ushort_as_half(b[i]));
                                num += (x - y) * (x - y); den += y * y;
                            }
                            return std::sqrt(num / std::max(den, 1e-30));
                        };
                        auto diff_frac = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
                            size_t n = 0;
                            for (size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
                            return (double) n / (double) a.size();
                        };
                        // outputs + byte identity (rows land in the pinned slab like the peer's finish)
                        indexed();
                        check(cudaStreamSynchronize(stream));
                        std::memcpy(got.data(), pin_rows.p, got.size() * 2);
                        const std::vector<uint16_t> idx1 = got;
                        grouped();
                        check(cudaStreamSynchronize(stream));
                        std::memcpy(got.data(), pin_rows.p, got.size() * 2);
                        const std::vector<uint16_t> grp = got;
                        const bool idx_eq_grp = idx1 == grp, idx_eq_or = idx1 == oracle, grp_eq_or = grp == oracle;
                        // ---- timing: eager (full enqueue chain, the peer's regime) + graph replay
                        auto time_arm = [&](auto&& fn) {
                            for (int i = 0; i < 20; ++i) fn();
                            check(cudaStreamSynchronize(stream));
                            cudaEvent_t a, b; check(cudaEventCreate(&a)); check(cudaEventCreate(&b));
                            check(cudaEventRecord(a, stream));
                            for (int i = 0; i < 300; ++i) fn();
                            check(cudaEventRecord(b, stream)); check(cudaEventSynchronize(b));
                            float ms = 0; check(cudaEventElapsedTime(&ms, a, b));
                            check(cudaEventDestroy(a)); check(cudaEventDestroy(b));
                            return ms * 1000.0f / 300;   // us per window
                        };
                        const float idx_us = time_arm(indexed);
                        const float grp_us = time_arm(grouped);
                        const float idx_us2 = time_arm(indexed);
                        auto graph_arm = [&](auto&& fn) {
                            check(cudaStreamSynchronize(stream));
                            cudaGraph_t graph; cudaGraphExec_t exec;
                            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                            fn();
                            check(cudaStreamEndCapture(stream, &graph));
                            check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
                            for (int i = 0; i < 10; ++i) check(cudaGraphLaunch(exec, stream));
                            check(cudaStreamSynchronize(stream));
                            cudaEvent_t a, b; check(cudaEventCreate(&a)); check(cudaEventCreate(&b));
                            check(cudaEventRecord(a, stream));
                            for (int i = 0; i < 300; ++i) check(cudaGraphLaunch(exec, stream));
                            check(cudaEventRecord(b, stream)); check(cudaEventSynchronize(b));
                            float ms = 0; check(cudaEventElapsedTime(&ms, a, b));
                            check(cudaEventDestroy(a)); check(cudaEventDestroy(b));
                            check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph));
                            return ms * 1000.0f / 300;
                        };
                        const float idx_g = graph_arm(indexed), grp_g = graph_arm(grouped);
                        const char* ng_env = std::getenv("STRATA_EXL3_NATIVE_GROUPED");
                        const int grp_native = ng_env && ng_env[0] != '\0' && ng_env[0] != '0';
                        std::printf("VERIFY_MIMO device=%d M=%d rows=%d share=%d kmode=%s routes=%d groups=%d "
                                    "reuse=%.3f two_row=%d k25_routes=%d grp_native=%d | eager_us idx=%.1f grp=%.1f idx2=%.1f "
                                    "| graph_us idx=%.1f grp=%.1f | bytes idx==grp=%d idx==oracle=%d grp==oracle=%d "
                                    "difffrac_idx_vs_grp=%.4f rel_grp_vs_oracle=%.3g rel_idx_vs_oracle=%.3g\n",
                                    device, M, T, share, kmode_s.c_str(), routes, groups,
                                    (double) routes / groups, two_row, k25, grp_native,
                                    idx_us, grp_us, idx_us2, idx_g, grp_g,
                                    idx_eq_grp ? 1 : 0, idx_eq_or ? 1 : 0, grp_eq_or ? 1 : 0,
                                    diff_frac(idx1, got), rel_err(grp, oracle), rel_err(idx1, oracle));
                        std::fflush(stdout);
                        if (!grp_eq_or)
                            std::fprintf(stderr, "VERIFY_MIMO NOTE: grouped bytes differ from the scalar "
                                                 "oracle at M=%d share=%d %s (rel %.3g, diff frac %.4f) - "
                                                 "byte identity is REPORTED, not assumed\n",
                                         M, share, kmode_s.c_str(), rel_err(grp, oracle), diff_frac(grp, oracle));
                        check(cudaStreamDestroy(stream));
                    }
                }
            }
            return 0;
        }
        if (group_mode) {
            const bool mimo = std::strcmp(argv[3], "mimo-group") == 0;
            const int hidden = mimo ? 4096 : 2560, intermediate = mimo ? 2048 : 640;
            // One expert, `routes` token rows: the prefill (>=16) group arms against each other.
            // argv[4] is K as a decimal ("4", "2", "2.5"): bits + frac for the fixture.
            std::mt19937 rng(7319);
            int bits = 6, frac = 0;
            if (argc > 4) {
                const std::string k = argv[4];
                const auto dot = k.find('.');
                bits = std::atoi(k.substr(0, dot == std::string::npos ? k.size() : dot).c_str());
                frac = dot == std::string::npos ? 0 : (std::atoi(k.c_str() + dot + 1) > 0 ? 1 : 0);
            }
            Matrix gate(hidden,intermediate,rng,bits,frac), up(hidden,intermediate,rng,bits,frac), down(intermediate,hidden,rng,bits,frac);
            const int tokens = argc > 5 ? std::atoi(argv[5]) : routes;   // engine-like: scattered rows into a big x
            Buffer<uint16_t> dx((size_t) tokens*hidden);
            std::vector<int> identity((size_t) routes);
            for (int r=0;r<routes;++r) identity[(size_t) r]=(r*113)%tokens;
            Buffer<int> dr(routes); Buffer<float> dy((size_t) routes*hidden);
            check(cudaMemcpy(dr.p,identity.data(),identity.size()*4,cudaMemcpyHostToDevice));
            std::vector<uint16_t> input((size_t) tokens * hidden);
            for (auto& v : input) v = __half_as_ushort(__float2half(((int)(rng()%2000)-1000)*.001f));
            check(cudaMemcpy(dx.p,input.data(),input.size()*2,cudaMemcpyHostToDevice));
            cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            Exl3MoeWorkspace workspace; workspace.reserve(routes,hidden,intermediate,hidden);
            auto time_arm = [&](const char* name, auto&& fn) {
                for (int i=0;i<10;++i) fn();
                check(cudaStreamSynchronize(stream));
                cudaEvent_t start,end; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
                check(cudaEventRecord(start,stream));
                for (int i=0;i<200;++i) fn();
                check(cudaEventRecord(end,stream)); check(cudaEventSynchronize(end));
                float ms=0; check(cudaEventElapsedTime(&ms,start,end));
                std::printf("CUDA%d K%s group rows=%d %s ms_per_layer_group=%.4f\n",device,frac? "2.5" : std::to_string(bits).c_str(),routes,name,ms/200);
                check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
            };
            time_arm("gemm(reconstruct+hgemm)", [&] {
                exl3_moe_group_gemm(gate.gpu,up.gpu,down.gpu,routes,dx.p,dr.p,dy.p,(void*) stream,workspace);
            });
            // STRATA_EXL3_GEMM_FP16_OUT paired A/B on the same fixture (in-process setter, the
            // stg_pad pattern): candidate after the baseline, then a baseline recheck so order
            // effects are visible in the table.  Values are NOT expected to match the baseline
            // bits (BLAS F16-C kernel selection) - the parity battery covers correctness.
            exl3_gemm_fp16_out_set(true);
            time_arm("gemm_fp16out(reconstruct+hgemm-f16c)", [&] {
                exl3_moe_group_gemm(gate.gpu,up.gpu,down.gpu,routes,dx.p,dr.p,dy.p,(void*) stream,workspace);
            });
            exl3_gemm_fp16_out_set(false);
            time_arm("gemm_recheck(reconstruct+hgemm)", [&] {
                exl3_moe_group_gemm(gate.gpu,up.gpu,down.gpu,routes,dx.p,dr.p,dy.p,(void*) stream,workspace);
            });
            // The STRATA_EXL3_RECON_BATCH path's own A/B at the same geometry: a 2-expert ragged
            // batch (dispatcher-eligible members are >= 16 rows, so this arm needs routes >= 32),
            // destination permutation, scratch pre-reserved.  Full chain: gather+reconstruct+3
            // strided GEMMs+silu+scatter.
            if (routes >= 32) {
                const int half = routes / 2;
                std::vector<int> dest((size_t) routes);
                for (int r = 0; r < routes; ++r) dest[(size_t) r] = routes - 1 - r;
                Buffer<int> dd((size_t) routes);
                check(cudaMemcpy(dd.p, dest.data(), routes * 4, cudaMemcpyHostToDevice));
                Exl3Group groups[] = {{gate.gpu,up.gpu,down.gpu,0,half},
                                      {gate.gpu,up.gpu,down.gpu,half,routes-half}};
                workspace.reserve_recon_batch(2, routes, hidden, intermediate, hidden);
                const auto recon = [&] {
                    (void) exl3_moe_recon_batch(groups, 2, dx.p, dr.p, dd.p, dy.p, (void*) stream, workspace);
                };
                time_arm("reconbatch(f32c)", recon);
                exl3_gemm_fp16_out_set(true);
                time_arm("reconbatch_fp16out(f16c)", recon);
                exl3_gemm_fp16_out_set(false);
                time_arm("reconbatch_recheck(f32c)", recon);
            }
            {   // the indexed GEMV path (the <16 dispatch) on the same one-expert shape
                std::vector<Exl3Mat> g((size_t) routes, gate.gpu), u((size_t) routes, up.gpu), d((size_t) routes, down.gpu);
                Buffer<uint16_t> dy16((size_t) routes*hidden);
                time_arm("indexed(gemv)", [&] {
                    exl3_moe_rows_indexed(g.data(),u.data(),d.data(),routes,dx.p,dr.p,dy16.p,(void*) stream,workspace);
                });
                if (mimo) {
                    std::vector<uint16_t> oracle((size_t) routes * hidden), actual(oracle.size());
                    check(cudaMemcpy(oracle.data(),dy16.p,oracle.size()*2,cudaMemcpyDeviceToHost));
                    for (int width : {0, 2, 4}) {
                        if (width == 0 && routes > 6) continue;
                        if (width && (size_t) hidden * width * 4 > 40*1024) {
                            std::printf("MIMO_GROUP rows=%d fused_width=%d unavailable: LDS exceeds 40KiB\n", routes, width);
                            continue;
                        }
                        std::vector<Exl3Mat> gg,gu,gd;
                        std::vector<int> first,counts;
                        for (int r = 0; r < routes;) {
                            first.push_back(r); const int n = std::min(width ? width : routes, routes-r);
                            counts.push_back(n); r += n;
                            gg.push_back(gate.gpu); gu.push_back(up.gpu); gd.push_back(down.gpu);
                        }
                        workspace.reserve_small_m(routes,gg.size(),hidden,intermediate,hidden);
                        workspace.reserve_runtime_scratch(routes,hidden,intermediate,hidden);
                        time_arm(width ? "grouped(fused-width2)" : "grouped(unfused)", [&] {
                            exl3_moe_rows_grouped(gg.data(),gu.data(),gd.data(),first.data(),counts.data(),
                                gg.size(),routes,dx.p,dr.p,dy16.p,stream,workspace,width!=0);
                        });
                        check(cudaMemcpy(actual.data(),dy16.p,actual.size()*2,cudaMemcpyDeviceToHost));
                        double diff=0,norm=0;
                        for(size_t i=0;i<actual.size();++i) {
                            const double a=__half2float(__ushort_as_half(actual[i])), b=__half2float(__ushort_as_half(oracle[i]));
                            diff+=(a-b)*(a-b);norm+=b*b;
                        }
                        const double rel=std::sqrt(diff/std::max(1e-30,norm));
                        std::printf("MIMO_GROUP parity rows=%d width=%d relative=%.6g\n",routes,width,rel);
                        if(!(rel<.02))throw std::runtime_error("grouped MiMo FFN parity failed");
                    }
                }
            }
            if (exl3_moe_group_wmma(gate.gpu,up.gpu,down.gpu,routes,dx.p,dr.p,dy.p,(void*) stream,workspace)) {
                time_arm("wmma(trellis-in-fragments)", [&] {
                    exl3_moe_group_wmma(gate.gpu,up.gpu,down.gpu,routes,dx.p,dr.p,dy.p,(void*) stream,workspace);
                });
                // STRATA_EXL3_WMMA_STG_PAD paired A/B on the same fixture: bytes must be equal
                // (layout-only), the timing difference is the measurement that matters.
                std::vector<float> y0((size_t) routes*hidden);
                check(cudaStreamSynchronize(stream));
                check(cudaMemcpy(y0.data(),dy.p,y0.size()*4,cudaMemcpyDeviceToHost));
                exl3_wmma_stg_pad_set(true);
                time_arm("wmma_stg_pad9", [&] {
                    exl3_moe_group_wmma(gate.gpu,up.gpu,down.gpu,routes,dx.p,dr.p,dy.p,(void*) stream,workspace);
                });
                exl3_wmma_stg_pad_set(false);
                std::vector<float> y1((size_t) routes*hidden);
                check(cudaStreamSynchronize(stream));
                check(cudaMemcpy(y1.data(),dy.p,y1.size()*4,cudaMemcpyDeviceToHost));
                std::printf("CUDA%d K%s group rows=%d stg_pad bytes_equal=%d\n",device,
                            frac? "2.5" : std::to_string(bits).c_str(),routes,(int)(y0==y1));
            }
            else
                std::printf("CUDA%d K%s group rows=%d wmma unavailable-on-this-device\n",device,frac? "2.5" : std::to_string(bits).c_str(),routes);
            check(cudaStreamDestroy(stream));
            return 0;
        }
        std::mt19937 rng(7319);
        const int wbits = argc > 3 ? std::atoi(argv[3]) : 6;   // the decode-window arm's bitrate (K3 on the 3bpw model)
        const bool multirow = argc > 4 && std::strcmp(argv[4], "multirow") == 0;
        const int width = argc > 5 ? std::atoi(argv[5]) : 4;
        if (multirow && (!exl3_fused_multirow_supported() || width < 1 || width > 4))
            throw std::runtime_error("multirow bench requires RDNA=1 FUSE=3 and width 1..4");
        // Ten experts with shared routes across four tokens. Distinct matrices avoid synthetic cache reuse.
        std::vector<Matrix*> weights;
        struct Cleanup { std::vector<Matrix*>& v; ~Cleanup() { for (auto p : v) delete p; } } cleanup{weights};
        for (int e=0;e<10;++e) {
            weights.push_back(new Matrix(2560,640,rng,wbits)); weights.push_back(new Matrix(2560,640,rng,wbits));
            weights.push_back(new Matrix(640,2560,rng,wbits));
        }
        std::vector<Exl3Mat> g(routes),u(routes),d(routes), gg, gu, gd;
        std::vector<int> first, count;
        std::vector<int> rows(routes);
        for (int r=0;r<routes;++r) {
            const int expert = multirow ? std::min(9, r * 10 / routes) : r%10;
            g[r]=weights[3*expert]->gpu; u[r]=weights[3*expert+1]->gpu; d[r]=weights[3*expert+2]->gpu;
            rows[r]=r%4;
            if (multirow) {
                if (r == 0 || g[r].trellis != g[r-1].trellis || count.back() == width) {
                    first.push_back(r); count.push_back(0);
                    gg.push_back(g[r]); gu.push_back(u[r]); gd.push_back(d[r]);
                }
                ++count.back();
            }
        }
        Buffer<int> dr(routes); Buffer<uint16_t> dx(4*2560),dy((size_t) routes*2560);
        check(cudaMemcpy(dr.p,rows.data(),rows.size()*4,cudaMemcpyHostToDevice));
        std::vector<uint16_t> input(4*2560);
        for (auto& v : input) v = __half_as_ushort(__float2half(((int)(rng()%2000)-1000)*0.001f));
        check(cudaMemcpy(dx.p,input.data(),input.size()*2,cudaMemcpyHostToDevice));
        cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        Exl3MoeWorkspace workspace; workspace.reserve(routes,2560,640,2560);
        auto run = [&] { exl3_moe_rows_indexed(g.data(),u.data(),d.data(),routes,dx.p,dr.p,dy.p,(void*) stream,workspace); };
        if (multirow) {
            workspace.reserve_small_m(routes,(int) first.size(),2560,640,2560);
            workspace.reserve_runtime_scratch(routes,2560,640,2560);
            auto grouped = [&] { exl3_moe_rows_grouped(gg.data(),gu.data(),gd.data(),first.data(),count.data(),
                (int) first.size(),routes,dx.p,dr.p,dy.p,stream,workspace,true); };
            // The native grouped arm (STRATA_EXL3_NATIVE_GROUPED=1 + STRATA_EXL3_RDNA_NATIVE=1,
            // admitted for the K2..K4 mul1 descriptors this bench builds) is a DIFFERENT kernel
            // with its own fdot2 accumulation order: byte identity with the indexed arm is NOT
            // expected under it.  Armed+admitted: each arm is held to its OWN repeat stability
            // instead; unarmed: the grouped/indexed bytes must match exactly as before.
            const char* ng_env = std::getenv("STRATA_EXL3_NATIVE_GROUPED");
            const char* nat_env = std::getenv("STRATA_EXL3_RDNA_NATIVE");
            const bool grp_native = ng_env && ng_env[0] != '\0' && ng_env[0] != '0' &&
                                    nat_env && nat_env[0] == '1';
            std::vector<uint16_t> oracle((size_t) routes*2560), actual(oracle.size());
            run(); check(cudaStreamSynchronize(stream));
            check(cudaMemcpy(oracle.data(),dy.p,oracle.size()*2,cudaMemcpyDeviceToHost));
            // the native grouped arm's OWN reference (unused when it is not armed): its
            // fdot2 order differs from the indexed arm's by design
            std::vector<uint16_t> grouped_ref;
            if (grp_native) {
                grouped(); check(cudaStreamSynchronize(stream));
                grouped_ref.resize(oracle.size());
                check(cudaMemcpy(grouped_ref.data(),dy.p,grouped_ref.size()*2,cudaMemcpyDeviceToHost));
            }
            const size_t bytes = workspace.allocated_bytes();
            for (int i=0;i<20;++i) {
                grouped(); check(cudaStreamSynchronize(stream));
                check(cudaMemcpy(actual.data(),dy.p,actual.size()*2,cudaMemcpyDeviceToHost));
                if (grp_native ? actual != grouped_ref : actual != oracle) {
                    size_t bad=0; while (actual[bad]==(grp_native?grouped_ref:oracle)[bad]) ++bad;
                    std::fprintf(stderr,"grouped parity failed (native=%d) row=%zu col=%zu actual=%04x oracle=%04x\n",
                        (int) grp_native,bad/2560,bad%2560,actual[bad],(grp_native?grouped_ref:oracle)[bad]); return 1;
                }
                run(); check(cudaStreamSynchronize(stream));
                check(cudaMemcpy(actual.data(),dy.p,actual.size()*2,cudaMemcpyDeviceToHost));
                if (actual != oracle) {
                    std::fprintf(stderr,"indexed arm bytes moved under the native flag\n"); return 1;
                }
            }
            check(cudaStreamSynchronize(stream));
            if (bytes != workspace.allocated_bytes()) throw std::runtime_error("workspace grew after reserve");
            auto time = [&](const char* name, auto&& fn) {
                for (int i=0;i<10;++i) fn();
                check(cudaStreamSynchronize(stream));
                cudaEvent_t begin,end; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
                check(cudaEventRecord(begin,stream));
                for (int i=0;i<300;++i) fn();
                check(cudaEventRecord(end,stream)); check(cudaEventSynchronize(end));
                float ms=0; check(cudaEventElapsedTime(&ms,begin,end));
                std::printf("CUDA%d K%d routes=%d groups=%zu width=%d split=%s grp_native=%d %s ms=%.6f "
                            "parity=%s nonzero=%zu\n",
                    device,wbits,routes,first.size(),width,std::getenv("STRATA_EXL3_GEMV_SPLIT") ?
                    std::getenv("STRATA_EXL3_GEMV_SPLIT") : "default",(int) grp_native,name,ms/300,
                    grp_native ? "repeat-stable" : "exact",
                    (size_t) std::count_if(oracle.begin(),oracle.end(),[](uint16_t v){return (v&0x7fff)!=0;}));
                check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
            };
            time("indexed-fuse3",run); time("grouped-fuse3",grouped); time("indexed-fuse3-repeat",run);
            auto graph_time = [&](const char* name, auto&& fn) {
                cudaGraph_t graph; cudaGraphExec_t exec;
                check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
                fn(); check(cudaStreamEndCapture(stream,&graph));
                check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
                size_t nodes=0; check(cudaGraphGetNodes(graph,nullptr,&nodes));
                // Three projection kernels plus the unchanged output copy; descriptor/plan
                // uploads are nodes too. Replay removes host enqueue gaps from this timing.
                for (int i=0;i<10;++i) check(cudaGraphLaunch(exec,stream));
                check(cudaStreamSynchronize(stream));
                cudaEvent_t begin,end; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
                check(cudaEventRecord(begin,stream));
                for (int i=0;i<300;++i) check(cudaGraphLaunch(exec,stream));
                check(cudaEventRecord(end,stream)); check(cudaEventSynchronize(end));
                float ms=0; check(cudaEventElapsedTime(&ms,begin,end));
                check(cudaMemcpy(actual.data(),dy.p,actual.size()*2,cudaMemcpyDeviceToHost));
                // the graph replays the dispatch it was CAPTURED with: the native grouped
                // arm must reproduce its own captured bytes, the indexed arm its own
                const std::vector<uint16_t>& want = grp_native ? grouped_ref : oracle;
                if (grp_native && std::strcmp(name, "indexed-fuse3") == 0) {
                    if (actual != oracle) throw std::runtime_error("indexed graph replay bytes moved");
                } else if (actual != want) throw std::runtime_error("graph replay parity failed");
                std::printf("CUDA%d K%d routes=%d groups=%zu width=%d grp_native=%d %s graph_nodes=%zu ms=%.6f "
                            "parity=%s\n",
                    device,wbits,routes,first.size(),width,(int) grp_native,name,nodes,ms/300,
                    grp_native ? "repeat-stable" : "exact");
                check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
                check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph));
            };
            graph_time("indexed-fuse3",run); graph_time("grouped-fuse3",grouped);
            check(cudaStreamDestroy(stream));
            return 0;
        }
        for (int i=0;i<10;++i) run();
        check(cudaStreamSynchronize(stream));
        cudaEvent_t start,end; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
        check(cudaEventRecord(start,stream));
        for (int i=0;i<200;++i) run();
        check(cudaEventRecord(end,stream)); check(cudaEventSynchronize(end));
        float ms=0; check(cudaEventElapsedTime(&ms,start,end));
        std::printf("CUDA%d K%d routes=%d split=%s resident_FFN_ms_per_layer_window=%.4f\n",device,wbits,routes,
                    std::getenv("STRATA_EXL3_GEMV_SPLIT") ? std::getenv("STRATA_EXL3_GEMV_SPLIT") : "default",ms/200);
        check(cudaEventDestroy(start)); check(cudaEventDestroy(end)); check(cudaStreamDestroy(stream));
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
