// Resident EXL3 window compute benchmark at the model's actual 2560/640 geometry.
// No weight uploads in the timed loop. This is not an end-to-end engine benchmark.
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/exl3_experts.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <random>
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
    Matrix(int in, int out, std::mt19937& rng, int bits = 6) : tr((size_t) in*out*bits/16), su(in), sv(out) {
        std::vector<uint16_t> packed((size_t) in*out*bits/16), scales((size_t) std::max(in,out));
        for (auto& v : packed) v = (uint16_t) rng();
        for (auto& v : scales) v = __half_as_ushort(__float2half(0.025f));
        check(cudaMemcpy(tr.p,packed.data(),packed.size()*2,cudaMemcpyHostToDevice));
        check(cudaMemcpy(su.p,scales.data(),in*2,cudaMemcpyHostToDevice));
        check(cudaMemcpy(sv.p,scales.data(),out*2,cudaMemcpyHostToDevice));
        gpu = {tr.p,su.p,sv.p,in/16,out/16,bits,2};
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
    const bool group_mode = argc > 3 && std::strcmp(argv[3], "group") == 0;   // one expert, `routes` rows: the prefill group arms
    if (routes < 1 || routes > 512) return 2;
    try {
        check(cudaSetDevice(device));
        if (group_mode) {
            // One expert, `routes` token rows: the prefill (>=16) group arms against each other.
            std::mt19937 rng(7319);
            const int bits = argc > 4 ? std::atoi(argv[4]) : 6;
            Matrix gate(2560,640,rng,bits), up(2560,640,rng,bits), down(640,2560,rng,bits);
            const int tokens = argc > 5 ? std::atoi(argv[5]) : routes;   // engine-like: scattered rows into a big x
            Buffer<uint16_t> dx((size_t) tokens*2560);
            std::vector<int> identity((size_t) routes);
            for (int r=0;r<routes;++r) identity[(size_t) r]=(r*113)%tokens;
            Buffer<int> dr(routes); Buffer<float> dy((size_t) routes*2560);
            check(cudaMemcpy(dr.p,identity.data(),identity.size()*4,cudaMemcpyHostToDevice));
            check(cudaMemset(dx.p,0,(size_t) tokens*2560*2));   // every row identity[] can reach: tokens, not routes
            cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            Exl3MoeWorkspace workspace; workspace.reserve(routes,2560,640,2560);
            auto time_arm = [&](const char* name, auto&& fn) {
                for (int i=0;i<10;++i) fn();
                check(cudaStreamSynchronize(stream));
                cudaEvent_t start,end; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
                check(cudaEventRecord(start,stream));
                for (int i=0;i<200;++i) fn();
                check(cudaEventRecord(end,stream)); check(cudaEventSynchronize(end));
                float ms=0; check(cudaEventElapsedTime(&ms,start,end));
                std::printf("CUDA%d K%d group rows=%d %s ms_per_layer_group=%.4f\n",device,bits,routes,name,ms/200);
                check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
            };
            time_arm("gemm(reconstruct+hgemm)", [&] {
                exl3_moe_group_gemm(gate.gpu,up.gpu,down.gpu,routes,dx.p,dr.p,dy.p,(void*) stream,workspace);
            });
            {   // the indexed GEMV path (the <16 dispatch) on the same one-expert shape
                std::vector<Exl3Mat> g((size_t) routes, gate.gpu), u((size_t) routes, up.gpu), d((size_t) routes, down.gpu);
                Buffer<uint16_t> dy16((size_t) routes*2560);
                time_arm("indexed(gemv)", [&] {
                    exl3_moe_rows_indexed(g.data(),u.data(),d.data(),routes,dx.p,dr.p,dy16.p,(void*) stream,workspace);
                });
            }
            if (exl3_moe_group_wmma(gate.gpu,up.gpu,down.gpu,routes,dx.p,dr.p,dy.p,(void*) stream,workspace))
                time_arm("wmma(trellis-in-fragments)", [&] {
                    exl3_moe_group_wmma(gate.gpu,up.gpu,down.gpu,routes,dx.p,dr.p,dy.p,(void*) stream,workspace);
                });
            else
                std::printf("CUDA%d K%d group rows=%d wmma unavailable-on-this-device\n",device,bits,routes);
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
            std::vector<uint16_t> oracle((size_t) routes*2560), actual(oracle.size());
            run(); check(cudaStreamSynchronize(stream));
            check(cudaMemcpy(oracle.data(),dy.p,oracle.size()*2,cudaMemcpyDeviceToHost));
            const size_t bytes = workspace.allocated_bytes();
            for (int i=0;i<20;++i) {
                grouped(); check(cudaStreamSynchronize(stream));
                check(cudaMemcpy(actual.data(),dy.p,actual.size()*2,cudaMemcpyDeviceToHost));
                if (actual != oracle) {
                    size_t bad=0; while (actual[bad]==oracle[bad]) ++bad;
                    std::fprintf(stderr,"parity failed row=%zu col=%zu actual=%04x oracle=%04x\n",
                        bad/2560,bad%2560,actual[bad],oracle[bad]); return 1;
                }
                run();
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
                std::printf("CUDA%d K%d routes=%d groups=%zu width=%d split=%s %s ms=%.6f parity=exact nonzero=%zu\n",
                    device,wbits,routes,first.size(),width,std::getenv("STRATA_EXL3_GEMV_SPLIT") ?
                    std::getenv("STRATA_EXL3_GEMV_SPLIT") : "default",name,ms/300,
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
                if (actual != oracle) throw std::runtime_error("graph replay parity failed");
                std::printf("CUDA%d K%d routes=%d groups=%zu width=%d %s graph_nodes=%zu ms=%.6f parity=exact\n",
                    device,wbits,routes,first.size(),width,name,nodes,ms/300);
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
