// CPU-oracle parity for route-count growth, indexed inputs and independent execution workspaces.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/cpu/exl3.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace strata::kernels;
constexpr int N = 128;
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
uint16_t half(float f) { const __half h = __float2half_rn(f); uint16_t v; std::memcpy(&v, &h, 2); return v; }
float fp32(uint16_t v) { __half h; std::memcpy(&h, &v, 2); return __half2float(h); }
template <class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) { check(cudaMalloc((void**) &p, n * sizeof(T))); }
    ~Buffer() { (void) cudaFree(p); }
    Buffer(const Buffer&) = delete;
};
struct Matrix {
    int inputs, outputs;
    std::vector<uint16_t> trellis, suh, svh, weight;
    Buffer<uint16_t> tr, su, sv;
    Exl3Mat gpu;
    Matrix(int bits, int cb, std::mt19937& rng, int in = N, int out = N) : inputs(in), outputs(out),
        trellis((size_t) (in / 16) * (out / 16) * 16 * bits), suh(in), svh(out), weight((size_t) in*out),
        tr(trellis.size()), su(in), sv(out) {
        for (auto& v : trellis) v = (uint16_t) rng();
        for (auto& v : suh) v = half(((int) (rng()%1000) - 500) * 0.0005f);
        for (auto& v : svh) v = half(((int) (rng()%1000) - 500) * 0.0005f);
        check(cudaMemcpy(tr.p, trellis.data(), trellis.size()*2, cudaMemcpyHostToDevice));
        check(cudaMemcpy(su.p, suh.data(), inputs*2, cudaMemcpyHostToDevice));
        check(cudaMemcpy(sv.p, svh.data(), outputs*2, cudaMemcpyHostToDevice));
        gpu = {tr.p, su.p, sv.p, inputs/16, outputs/16, bits, cb};
        cpu::exl3_reconstruct_weight(trellis.data(), inputs/16, outputs/16, bits, (cpu::Exl3Codebook) cb,
                                     suh.data(), svh.data(), weight.data());
    }
    std::vector<uint16_t> project(const uint16_t* x) const {
        std::vector<uint16_t> y(outputs);
        for (int j=0; j<outputs; ++j) {
            double s=0;
            for (int i=0; i<inputs; ++i) s += (double) fp32(x[i]) * fp32(weight[(size_t) i*outputs+j]);
            y[(size_t) j] = half((float) s);
        }
        return y;
    }
};

bool real_geometry(int device) {
    check(cudaSetDevice(device));
    std::mt19937 rng(104729);
    constexpr int width = 2560, hidden = 640;
    Matrix gate(6,2,rng,width,hidden), up(6,2,rng,width,hidden), down(6,2,rng,hidden,width);
    std::vector<uint16_t> x(4*width), ref(4*width), baseline;
    for (auto& v : x) v=half(((int)(rng()%1000)-500)*0.001f);
    for (int t=0;t<4;++t) {
        auto g=gate.project(x.data()+t*width); const auto u=up.project(x.data()+t*width);
        for (int i=0;i<hidden;++i) {const float v=fp32(g[i]);g[i]=half(v/(1+std::exp(-v))*fp32(u[i]));}
        const auto y=down.project(g.data()); std::copy(y.begin(),y.end(),ref.begin()+t*width);
    }
    Buffer<uint16_t> dx(x.size());check(cudaMemcpy(dx.p,x.data(),x.size()*2,cudaMemcpyHostToDevice));
    Exl3MoeWorkspace workspace;
    bool ok=true;
    for (int routes : {10,40,10}) {
        std::vector<Exl3Mat> g(routes,gate.gpu),u(routes,up.gpu),d(routes,down.gpu);
        std::vector<int> rows(routes);for(int r=0;r<routes;++r)rows[r]=r%4;
        Buffer<int> dr(routes);Buffer<uint16_t> dy((size_t) routes*width);
        check(cudaMemcpy(dr.p,rows.data(),rows.size()*sizeof(int),cudaMemcpyHostToDevice));
        exl3_moe_rows_indexed(g.data(),u.data(),d.data(),routes,dx.p,dr.p,dy.p,nullptr,workspace);
        std::vector<uint16_t> y((size_t) routes*width);
        check(cudaMemcpy(y.data(),dy.p,y.size()*2,cudaMemcpyDeviceToHost));
        double num=0,den=0;
        for(size_t i=0;i<y.size();++i) {
            const double expected=fp32(ref[(size_t) rows[i/width]*width+i%width]);
            const double delta=fp32(y[i])-expected;num+=delta*delta;den+=expected*expected;
        }
        const double rel=std::sqrt(num/(den+1e-30));
        const bool identical=baseline.empty() || std::equal(baseline.begin(),baseline.end(),y.begin());
        ok=ok && std::isfinite(rel) && rel<0.02 && identical;
        if(baseline.empty())baseline.assign(y.begin(),y.begin()+10*width);
        std::printf("device=%d geometry=2560/640 routes=%d relative_error=%.5g route_count_identical=%d %s\n",
                    device,routes,rel,identical,ok?"PASS":"FAIL");
    }
    // The gfx11 matrix-core group FFN on the real geometry, every bitrate the arm covers (K3..K6;
    // the 4.05bpw model's experts are K4), at group counts around the dispatch threshold and the
    // kernel's internal boundaries (row_tiles switches to 2 at 24; the 16-row A-tiles pad at every
    // multiple).  The routes scatter over MORE token rows than the group has (a prefill chunk's
    // shape): gate/up index the token buffer, the down projection reads the FFN's own route-major
    // slab, and the two must not be confused.  SKIP (not FAIL) where the device cannot take the path.
    for (int bits : {3, 4, 5, 6}) {
        Matrix wg(bits,2,rng,width,hidden), wu(bits,2,rng,width,hidden), wd(bits,2,rng,hidden,width);
        for (int rows : {7, 8, 15, 16, 23, 24, 31, 32, 40}) {
            const int tokens = 4 * rows + 13;
            std::vector<uint16_t> wx((size_t) tokens*width);
            std::vector<double> wref((size_t) rows*width);
            for (auto& v : wx) v=half(((int)(rng()%1000)-500)*0.001f);
            std::vector<int> token_of((size_t) rows);
            for (int r=0;r<rows;++r) token_of[(size_t) r]=(r*113)%tokens;
            for (int r=0;r<rows;++r) {
                const uint16_t* xr = wx.data()+(size_t) token_of[(size_t) r]*width;
                auto g=wg.project(xr); const auto u=wu.project(xr);
                for (int i=0;i<hidden;++i) {const float v=fp32(g[i]);g[i]=half(v/(1+std::exp(-v))*fp32(u[i]));}
                const auto y=wd.project(g.data());
                for (int i=0;i<width;++i) wref[(size_t) r*width+i]=fp32(y[i]);
            }
            Buffer<uint16_t> dwx(wx.size());
            check(cudaMemcpy(dwx.p,wx.data(),wx.size()*2,cudaMemcpyHostToDevice));
            Buffer<int> dr((size_t) rows);
            check(cudaMemcpy(dr.p,token_of.data(),token_of.size()*4,cudaMemcpyHostToDevice));
            Buffer<float> dout((size_t) rows*width);
            if (!exl3_moe_group_wmma(wg.gpu,wu.gpu,wd.gpu,rows,dwx.p,dr.p,dout.p,nullptr,workspace)) {
                std::printf("device=%d geometry=2560/640 K%d wmma_group rows=%d unavailable-on-this-device SKIP\n",
                            device,bits,rows);
                continue;
            }
            std::vector<float> wy((size_t) rows*width);
            check(cudaMemcpy(wy.data(),dout.p,wy.size()*4,cudaMemcpyDeviceToHost));
            double num=0,den=0;
            for (size_t i=0;i<wy.size();++i) { const double delta=wy[i]-wref[i]; num+=delta*delta; den+=wref[i]*wref[i]; }
            const double rel=std::sqrt(num/(den+1e-30));
            const bool pass=std::isfinite(rel)&&rel<0.02;
            ok=ok&&pass;
            std::printf("device=%d geometry=2560/640 K%d wmma_group rows=%d scattered relative_error=%.5g %s\n",
                        device,bits,rows,rel,pass?"PASS":"FAIL");
        }
    }
    // Multi-expert WMMA subgroup: independent signed weights/scales per projection and expert,
    // unequal groups, nonidentity input and output maps, K3..K6 against the CPU reconstruction.
    for (int bits : {3,4,5,6}) {
        Matrix g0(bits,2,rng),u0(bits,2,rng),d0(bits,2,rng);
        Matrix g1(bits,2,rng),u1(bits,2,rng),d1(bits,2,rng);
        for (int first_rows : {16,24}) {
            const int second_rows=first_rows+7, total=first_rows+second_rows, tokens=total+17;
            std::vector<uint16_t> x((size_t)tokens*N);
            for(auto& v:x) v=half(((int)(rng()%1000)-500)*0.001f);
            std::vector<int> rows(total),dest(total);
            for(int r=0;r<total;++r) {rows[r]=(r*7+11)%tokens; dest[r]=total-1-r;}
            Buffer<uint16_t> dx(x.size()); Buffer<int> dr(total),dd(total); Buffer<float> dy((size_t)total*N);
            check(cudaMemcpy(dx.p,x.data(),x.size()*2,cudaMemcpyHostToDevice));
            check(cudaMemcpy(dr.p,rows.data(),total*4,cudaMemcpyHostToDevice));
            check(cudaMemcpy(dd.p,dest.data(),total*4,cudaMemcpyHostToDevice));
            Exl3Group groups[]={{g0.gpu,u0.gpu,d0.gpu,0,first_rows},{g1.gpu,u1.gpu,d1.gpu,first_rows,second_rows}};
            workspace.reserve_expert_batch(2,total,N,N,N);
            if (!exl3_moe_expert_batch(groups, 2, dx.p, dr.p, dd.p, dy.p, nullptr, workspace)) {
                cudaDeviceProp prop{};
                check(cudaGetDeviceProperties(&prop, device));
#if defined(STRATA_WMMA_GFX11)
                if (std::string(prop.gcnArchName).starts_with("gfx11")) {
                    std::printf("device=%d expert_batch supported K%d unexpectedly rejected FAIL\n", device, bits);
                    ok = false;
                } else
#endif
                    std::printf("device=%d expert_batch unavailable-on-this-device SKIP\n", device);
                continue;
            }
            std::vector<float> y((size_t)total*N);
            check(cudaMemcpy(y.data(),dy.p,y.size()*4,cudaMemcpyDeviceToHost));
            double num=0,den=0;
            for(int r=0;r<total;++r) {
                auto& g=r<first_rows?g0:g1; auto& u=r<first_rows?u0:u1; auto& d=r<first_rows?d0:d1;
                auto gate=g.project(x.data()+(size_t)rows[r]*N); auto up=u.project(x.data()+(size_t)rows[r]*N);
                for(int j=0;j<N;++j) {float v=fp32(gate[j]);gate[j]=half(v/(1+std::exp(-v))*fp32(up[j]));}
                auto ref=d.project(gate.data());
                for(int j=0;j<N;++j) {double v=fp32(ref[j]),diff=y[(size_t)dest[r]*N+j]-v;num+=diff*diff;den+=v*v;}
            }
            const double rel=std::sqrt(num/(den+1e-30)); const bool pass=std::isfinite(rel)&&rel<0.02;
            ok=ok&&pass;std::printf("device=%d expert_batch K%d groups=%d/%d CPU relative_error=%.5g %s\n",device,bits,first_rows,second_rows,rel,pass?"PASS":"FAIL");
            auto bad=groups[1]; groups[1].gate.bits=bits==3?4:3;
            ok=ok&&!exl3_moe_expert_batch(groups,2,dx.p,dr.p,dd.p,dy.p,nullptr,workspace);
            groups[1]=bad;groups[1].gate.cb=1;
            ok=ok&&!exl3_moe_expert_batch(groups,2,dx.p,dr.p,dd.p,dy.p,nullptr,workspace);
            groups[1]=bad;groups[1].rows=7;
            ok=ok&&!exl3_moe_expert_batch(groups,2,dx.p,dr.p,dd.p,dy.p,nullptr,workspace);
        }
    }
    // The graph-side hit-arm primitive: descriptors gathered on the DEVICE from per-slot tables by
    // slot id, then the existing indexed kernels - no host interaction, capture-friendly.  Compared
    // BYTE FOR BYTE against exl3_moe_rows_indexed with the equivalent host arrays: distinct experts
    // per slot, routes that hit distinct slots, routes that SHARE a slot (duplicate expert), and a
    // repeated window.  SKIP (not FAIL) where the device cannot take the path.
    {
        const int slots = 6;
        for (int bits : {3, 4}) {
            std::vector<std::unique_ptr<Matrix>> wg, wu, wd;
            for (int e = 0; e < slots; ++e) {
                wg.emplace_back(new Matrix(bits,2,rng,width,hidden)); wu.emplace_back(new Matrix(bits,2,rng,width,hidden));
                wd.emplace_back(new Matrix(bits,2,rng,hidden,width));
            }
            std::vector<Exl3Mat> hg(slots), hu(slots), hd(slots);
            for (int e = 0; e < slots; ++e) { hg[(size_t)e]=wg[(size_t)e]->gpu; hu[(size_t)e]=wu[(size_t)e]->gpu; hd[(size_t)e]=wd[(size_t)e]->gpu; }
            Buffer<Exl3Mat> tg(hg.size()), tu(hu.size()), td(hd.size());
            check(cudaMemcpy(tg.p,hg.data(),hg.size()*sizeof(Exl3Mat),cudaMemcpyHostToDevice));
            check(cudaMemcpy(tu.p,hu.data(),hu.size()*sizeof(Exl3Mat),cudaMemcpyHostToDevice));
            check(cudaMemcpy(td.p,hd.data(),hd.size()*sizeof(Exl3Mat),cudaMemcpyHostToDevice));
            const int routes = 11, tokens = 4 * routes + 13;
            std::vector<uint16_t> wx((size_t) tokens*width);
            for (auto& v : wx) v=half(((int)(rng()%1000)-500)*0.001f);
            std::vector<int> token_of(routes), slot_of(routes);
            for (int r=0;r<routes;++r) token_of[(size_t) r]=(r*113)%tokens;
            for (int r=0;r<routes;++r) slot_of[(size_t) r]=(r*7)%slots;   // distinct + shared slots
            Buffer<uint16_t> dwx(wx.size());
            check(cudaMemcpy(dwx.p,wx.data(),wx.size()*2,cudaMemcpyHostToDevice));
            Buffer<int> dr(routes), dsl(routes);
            check(cudaMemcpy(dr.p,token_of.data(),token_of.size()*4,cudaMemcpyHostToDevice));
            check(cudaMemcpy(dsl.p,slot_of.data(),slot_of.size()*4,cudaMemcpyHostToDevice));
            Buffer<uint16_t> dhost((size_t) routes*width), dslot((size_t) routes*width);
            std::vector<Exl3Mat> rg(routes), ru(routes), rd(routes);
            for (int r=0;r<routes;++r) { rg[(size_t) r]=hg[(size_t) slot_of[(size_t) r]]; ru[(size_t) r]=hu[(size_t) slot_of[(size_t) r]]; rd[(size_t) r]=hd[(size_t) slot_of[(size_t) r]]; }
            Exl3MoeWorkspace ws;
            exl3_moe_rows_indexed(rg.data(),ru.data(),rd.data(),routes,dwx.p,dr.p,dhost.p,nullptr,ws);
            bool ran=false;
            try {
                exl3_moe_rows_slot_table(tg.p,tu.p,td.p,slots,dsl.p,routes,width,hidden,width,
                                         dwx.p,dr.p,dslot.p,nullptr,ws);
                ran=true;
            } catch (const std::exception& e) { std::printf("device=%d K%d slot_table error %s\n",device,bits,e.what()); }
            if (ran) {
                // repeat the window (same tables, same ids) - a decode window replay
                exl3_moe_rows_slot_table(tg.p,tu.p,td.p,slots,dsl.p,routes,width,hidden,width,
                                         dwx.p,dr.p,dslot.p,nullptr,ws);
                std::vector<uint16_t> yh((size_t) routes*width), ys((size_t) routes*width);
                check(cudaMemcpy(yh.data(),dhost.p,yh.size()*2,cudaMemcpyDeviceToHost));
                check(cudaMemcpy(ys.data(),dslot.p,ys.size()*2,cudaMemcpyDeviceToHost));
                const bool identical = std::memcmp(yh.data(),ys.data(),yh.size()*2)==0;
                ok = ok && identical;
                std::printf("device=%d geometry=2560/640 K%d slot_table routes=%d mixed-slots %s\n",
                            device,bits,routes,identical?"byte-identical PASS":"FAIL");
                // Negative probe: an INVALID slot id must fail visibly - the call throws outside
                // capture (route zeroed, wrong expert never computed) and the flag stays queryable.
                {
                    std::vector<int> bad(slot_of); bad[(size_t) routes - 1] = slots + 3;
                    Buffer<int> dbad((size_t) routes);
                    check(cudaMemcpy(dbad.p,bad.data(),bad.size()*4,cudaMemcpyHostToDevice));
                    bool threw = false;
                    try {
                        exl3_moe_rows_slot_table(tg.p,tu.p,td.p,slots,dbad.p,routes,width,hidden,width,
                                                 dwx.p,dr.p,dslot.p,nullptr,ws);
                    } catch (const std::exception&) { threw = true; }
                    ok = ok && threw;
                    std::printf("device=%d geometry=2560/640 K%d slot_table invalid-slot %s\n",
                                device,bits,threw?"throws-before-launch PASS":"silent FAIL");
                }
                // Graph capture/replay probe: the whole primitive (gather + kernels + D2D copy)
                // captured on a stream, instantiated and replayed - outputs must stay identical.
                cudaStream_t cs;
                if (cudaStreamCreateWithFlags(&cs,cudaStreamNonBlocking)==cudaSuccess) {
                    Buffer<uint16_t> dcapture((size_t) routes*width);
                    exl3_moe_rows_slot_table(tg.p,tu.p,td.p,slots,dsl.p,routes,width,hidden,width,
                                             dwx.p,dr.p,dcapture.p,(void*)cs,ws);   // warm workspace/statics outside capture
                    check(cudaStreamSynchronize(cs));
                    if (cudaStreamBeginCapture(cs,cudaStreamCaptureModeGlobal)==cudaSuccess) {
                        exl3_moe_rows_slot_table(tg.p,tu.p,td.p,slots,dsl.p,routes,width,hidden,width,
                                                 dwx.p,dr.p,dcapture.p,(void*)cs,ws);
                        cudaGraph_t graph=nullptr;
                        if (cudaStreamEndCapture(cs,&graph)==cudaSuccess && graph!=nullptr) {
                            cudaGraphExec_t exec=nullptr;
                            if (cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0)==cudaSuccess) {
                                for (int rep=0;rep<3;++rep) check(cudaGraphLaunch(exec,cs));
                                check(cudaStreamSynchronize(cs));
                                std::vector<uint16_t> yc((size_t) routes*width);
                                check(cudaMemcpy(yc.data(),dcapture.p,yc.size()*2,cudaMemcpyDeviceToHost));
                                const bool cap_ok = std::memcmp(yh.data(),yc.data(),yh.size()*2)==0;
                                ok = ok && cap_ok;
                                std::printf("device=%d geometry=2560/640 K%d slot_table graph_capture_replay %s\n",
                                            device,bits,cap_ok?"byte-identical PASS":"FAIL");
                                (void)cudaGraphExecDestroy(exec);
                            } else { std::printf("device=%d K%d slot_table instantiate FAILED\n",device,bits); ok=false; }
                        } else { std::printf("device=%d K%d slot_table end-capture FAILED\n",device,bits); ok=false; }
                        (void)cudaGraphDestroy(graph);
                    } else { std::printf("device=%d K%d slot_table begin-capture FAILED\n",device,bits); ok=false; }
                    (void)cudaStreamDestroy(cs);
                }
            }
        }
    }
    return ok;
}

bool run_device(int device) {
    check(cudaSetDevice(device));
    std::mt19937 rng(7919);
    Matrix g0(3,2,rng), u0(6,2,rng), d0(8,2,rng);
    Matrix g1(4,1,rng), u1(5,0,rng), d1(2,2,rng);
    Matrix* gate[] = {&g0,&g1}; Matrix* up[] = {&u0,&u1}; Matrix* down[] = {&d0,&d1};
    std::vector<uint16_t> x(3*N);
    for (auto& v : x) v = half(((int) (rng()%1000) - 500) * 0.001f);
    Buffer<uint16_t> dx(x.size()); check(cudaMemcpy(dx.p,x.data(),x.size()*2,cudaMemcpyHostToDevice));
    // Distinct streams run different input-row permutations. Separate workspaces must not overwrite metadata.
    auto task = [&](int seed) {
        check(cudaSetDevice(device));
        cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        Exl3MoeWorkspace ws;
        bool ok=true;
        for (int routes : {3,10,2,37,53}) {
            const bool gemm = routes == 37;
            std::vector<int> tokens((size_t) routes);
            std::vector<Exl3Mat> g((size_t) routes),u((size_t) routes),d((size_t) routes);
            std::vector<float> ref((size_t) routes*N);
            for (int r=0;r<routes;++r) {
                const int e=gemm ? seed%2 : (r+seed)%2; tokens[(size_t) r]=(r*2+seed)%3;
                g[(size_t) r]=gate[e]->gpu; u[(size_t) r]=up[e]->gpu; d[(size_t) r]=down[e]->gpu;
                auto gv=gate[e]->project(x.data()+(size_t) tokens[(size_t) r]*N);
                const auto uv=up[e]->project(x.data()+(size_t) tokens[(size_t) r]*N);
                for (int i=0;i<N;++i) { const float v=fp32(gv[(size_t) i]); gv[(size_t) i]=half(v/(1+std::exp(-v))*fp32(uv[(size_t) i])); }
                const auto y=down[e]->project(gv.data());
                for (int i=0;i<N;++i) ref[(size_t) r*N+i]=fp32(y[(size_t) i]);
            }
            Buffer<int> dt((size_t) routes); Buffer<uint16_t> dy((size_t) routes*N);
            Buffer<float> df((size_t) routes*N);
            check(cudaMemcpyAsync(dt.p,tokens.data(),tokens.size()*sizeof(int),cudaMemcpyHostToDevice,stream));
            if (gemm) exl3_moe_group_gemm(g[0],u[0],d[0],routes,dx.p,dt.p,df.p,(void*) stream,ws);
            else exl3_moe_rows_indexed(g.data(),u.data(),d.data(),routes,dx.p,dt.p,dy.p,(void*) stream,ws);
            std::vector<uint16_t> y((size_t) routes*N);
            std::vector<float> yf(y.size());
            if (gemm) check(cudaMemcpyAsync(yf.data(),df.p,yf.size()*4,cudaMemcpyDeviceToHost,stream));
            else check(cudaMemcpyAsync(y.data(),dy.p,y.size()*2,cudaMemcpyDeviceToHost,stream));
            check(cudaStreamSynchronize(stream));
            double num=0,den=0;
            for (size_t i=0;i<y.size();++i) { double delta=(gemm?yf[i]:fp32(y[i]))-ref[i]; num+=delta*delta; den+=(double) ref[i]*ref[i]; }
            const double rel=std::sqrt(num/(den+1e-30));
            // CPU reconstruct rounds each weight; folded GEMV rounds only projection outputs.
            if (rel>0.02 || !std::isfinite(rel)) ok=false;
            std::printf("device=%d stream=%d routes=%d relative_error=%.5g %s\n",device,seed,routes,rel,rel<=0.02?"PASS":"FAIL");
            if (gemm) {
                // The gfx11 matrix-core group FFN against the same reference (SKIP-arm when unavailable).
                std::vector<float> yw(yf.size());
                if (exl3_moe_group_wmma(g[0],u[0],d[0],routes,dx.p,dt.p,df.p,(void*) stream,ws)) {
                    check(cudaMemcpyAsync(yw.data(),df.p,yw.size()*4,cudaMemcpyDeviceToHost,stream));
                    check(cudaStreamSynchronize(stream));
                    double wn=0,wd=0;
                    for (size_t i=0;i<yw.size();++i) { double delta=yw[i]-ref[i]; wn+=delta*delta; wd+=(double) ref[i]*ref[i]; }
                    const double wrel=std::sqrt(wn/(wd+1e-30));
                    if (wrel>0.02 || !std::isfinite(wrel)) ok=false;
                    std::printf("device=%d stream=%d routes=%d wmma_group relative_error=%.5g %s\n",
                                device,seed,routes,wrel,wrel<=0.02?"PASS":"FAIL");
                } else {
                    std::printf("device=%d stream=%d routes=%d wmma_group unavailable-on-this-device SKIP\n",device,seed,routes);
                }
            }
        }
        check(cudaStreamDestroy(stream));
        return ok;
    };
    bool a=false,b=false; std::exception_ptr ea,eb;
    std::thread t0([&]{try {a=task(0);} catch (...) {ea=std::current_exception();}});
    std::thread t1([&]{try {b=task(1);} catch (...) {eb=std::current_exception();}});
    t0.join(); t1.join(); if(ea)std::rethrow_exception(ea); if(eb)std::rethrow_exception(eb);
    return real_geometry(device)&&a&&b;
}
}
int main() {
    int count=0;
    if (cudaGetDeviceCount(&count)!=cudaSuccess || count==0) { std::puts("SKIP: no CUDA device"); return 77; }
    try { bool ok=true; for(int d=0;d<count;++d) ok=run_device(d)&&ok; return ok?0:1; }
    catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
