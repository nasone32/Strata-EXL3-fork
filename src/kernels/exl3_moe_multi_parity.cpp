// CPU-oracle parity for route-count growth, indexed inputs and independent execution workspaces.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/cpu/exl3.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
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
    Matrix(int bits, int cb, std::mt19937& rng, int in = N, int out = N, int frac = 0) : inputs(in), outputs(out),
        trellis((size_t) (in / 16) * (out / 16) * (16 * bits + 8 * frac)), suh(in), svh(out), weight((size_t) in*out),
        tr(trellis.size()), su(in), sv(out) {
        for (auto& v : trellis) v = (uint16_t) rng();
        for (auto& v : suh) v = half(((int) (rng()%1000) - 500) * 0.0005f);
        for (auto& v : svh) v = half(((int) (rng()%1000) - 500) * 0.0005f);
        check(cudaMemcpy(tr.p, trellis.data(), trellis.size()*2, cudaMemcpyHostToDevice));
        check(cudaMemcpy(su.p, suh.data(), inputs*2, cudaMemcpyHostToDevice));
        check(cudaMemcpy(sv.p, svh.data(), outputs*2, cudaMemcpyHostToDevice));
        gpu = {tr.p, su.p, sv.p, inputs/16, outputs/16, bits, cb, frac};
        cpu::exl3_reconstruct_weight(trellis.data(), inputs/16, outputs/16, bits, (cpu::Exl3Codebook) cb,
                                     suh.data(), svh.data(), weight.data(), frac);
    }
    std::vector<uint16_t> project(const uint16_t* x) const {
        std::vector<uint16_t> y(outputs);
        for (int j=0;j<outputs;++j) {
            double s=0;
            for (int i=0;i<inputs;++i) s += (double) fp32(x[i]) * fp32(weight[(size_t) i*outputs+j]);
            y[(size_t) j] = half((float) s);
        }
        return y;
    }
};

// STRATA_EXL3_RECON_BATCH (C2): the batched weight reconstruction against the CPU oracle and
// against exl3_moe_group_gemm.  MiMo's own shape class: fractional/low-K mul1 experts, a batch
// that MIXES bitrates (K2 and K2.5 in one descriptor set - a MiMo layer routes both), unequal
// groups, scattered token rows and a nonidentity destination map, and geometry with more than
// one 128x128 Hadamard tile per axis.  The batched output must match the CPU reference tightly;
// against the per-group group_gemm it must be bit-identical or differ only by hipBLAS's internal
// reduction order (same F32 accumulation semantics, batched kernel selection) - the identical
// fraction is printed either way.
bool recon_batch_parity(int device) {
    check(cudaSetDevice(device));
    std::mt19937 rng(8191);
    constexpr int width = 256, hidden = 256;   // two HAD tiles per axis
    Matrix g0(2,2,rng,width,hidden),      u0(2,2,rng,width,hidden),      d0(2,2,rng,hidden,width);          // K2
    Matrix g1(2,2,rng,width,hidden,1),    u1(2,2,rng,width,hidden,1),    d1(2,2,rng,hidden,width,1);        // K2.5
    Matrix g2(2,2,rng,width,hidden),      u2(2,2,rng,width,hidden),      d2(2,2,rng,hidden,width);          // K2 again
    bool ok = true;
    struct Case { Matrix *g, *u, *d; int rows; };
    for (int variant = 0; variant < 2; ++variant) {
        // variant 0: a mixed-bitrate 2-expert batch (K2 + K2.5); variant 1: three K2 experts with
        // the widest in the middle (padding on both sides of the widest group)
        std::vector<Case> experts = variant == 0
            ? std::vector<Case>{{&g0,&u0,&d0,17},{&g1,&u1,&d1,23}}
            : std::vector<Case>{{&g2,&u2,&d2,16},{&g0,&u0,&d0,31},{&g1,&u1,&d1,16}};
        int total = 0, widest = 0; for (const Case& c : experts) { total += c.rows; widest = std::max(widest, c.rows); }
        const int tokens = 3 * total + 11;
        std::vector<uint16_t> x((size_t) tokens*width);
        for (auto& v : x) v=half(((int)(rng()%1000)-500)*0.001f);
        std::vector<int> rows(total), dest(total);
        for (int r=0;r<total;++r) { rows[r]=(r*7+5)%tokens; dest[r]=total-1-r; }
        std::vector<double> ref((size_t) total*width);
        int at = 0;
        for (const Case& c : experts) {
            for (int r=0;r<c.rows;++r,++at) {
                auto g=c.g->project(x.data()+(size_t) rows[at]*width); const auto u=c.u->project(x.data()+(size_t) rows[at]*width);
                for (int i=0;i<hidden;++i) {const float v=fp32(g[i]);g[i]=half(v/(1+std::exp(-v))*fp32(u[i]));}
                const auto y=c.d->project(g.data());
                for (int i=0;i<width;++i) ref[(size_t) at*width+i]=fp32(y[i]);
            }
        }
        Buffer<uint16_t> dx(x.size()); check(cudaMemcpy(dx.p,x.data(),x.size()*2,cudaMemcpyHostToDevice));
        Buffer<int> dr(total), dd(total);
        check(cudaMemcpy(dr.p,rows.data(),total*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(dd.p,dest.data(),total*4,cudaMemcpyHostToDevice));
        Buffer<float> dy((size_t) total*width);
        Exl3MoeWorkspace workspace;
        std::vector<Exl3Group> groups; int prefix = 0;
        for (const Case& c : experts) { groups.push_back({c.g->gpu,c.u->gpu,c.d->gpu,prefix,c.rows}); prefix += c.rows; }
        // rows_cap bounds the PADDED extent (count x widest), the engine's builder rule
        workspace.reserve_recon_batch((int) experts.size(), (int) experts.size() * widest + 16, width, hidden, width);
        cudaStream_t rs; check(cudaStreamCreateWithFlags(&rs, cudaStreamNonBlocking));
        if (!exl3_moe_recon_batch(groups.data(), (int) groups.size(), dx.p, dr.p, dd.p, dy.p, rs, workspace)) {
            std::printf("device=%d recon_batch variant=%d unexpectedly rejected FAIL\n", device, variant);
            return false;
        }
        { const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) std::printf("recon_batch launch error: %s\n", cudaGetErrorString(e)); }
        check(cudaStreamSynchronize(rs));
        check(cudaStreamDestroy(rs));
        std::vector<float> y((size_t) total*width);
        check(cudaMemcpy(y.data(),dy.p,y.size()*4,cudaMemcpyDeviceToHost));
        double num=0,den=0;
        for (int r=0;r<total;++r) for (int i=0;i<width;++i) {
            const double v = ref[(size_t) r*width+i], got = y[(size_t) dest[(size_t) r]*width+i];
            num += (got-v)*(got-v); den += v*v;
        }
        const double rel = std::sqrt(num/(den+1e-30));
        bool pass = std::isfinite(rel) && rel < 0.02;
        // the per-group reconstruct+BLAS chain on the same routes: same weights, same GEMM
        // semantics - identical bits unless hipBLAS picks a different reduction for the batch
        long long identical = 0; const long long elements = (long long) total*width;
        double gnum=0, gden=0;
        {
            Buffer<float> dg((size_t) widest*width);
            int base = 0;
            for (const Case& c : experts) {
                exl3_moe_group_gemm(c.g->gpu,c.u->gpu,c.d->gpu,c.rows,dx.p,dr.p+base,dg.p,nullptr,workspace);
                std::vector<float> gy((size_t) c.rows*width);
                check(cudaMemcpy(gy.data(),dg.p,gy.size()*4,cudaMemcpyDeviceToHost));
                for (int r=0;r<c.rows;++r) for (int i=0;i<width;++i) {
                    const float want = gy[(size_t) r*width+i], got = y[(size_t) dest[(size_t) (base+r)]*width+i];
                    if (want == got) ++identical;
                    gnum += (double) (got-want)*(got-want); gden += (double) want*want;
                }
                base += c.rows;
            }
        }
        const double grel = std::sqrt(gnum/(gden+1e-30));
        const bool gpass = std::isfinite(grel) && grel < 1e-3;
        ok = ok && pass && gpass;
        std::printf("device=%d recon_batch variant=%d experts=%zu rows=%d/%d CPU relative_error=%.5g %s; "
                    "vs group_gemm %.5g (%lld/%lld bits identical) %s\n",
                    device, variant, experts.size(), experts[0].rows, experts.back().rows, rel,
                    pass?"PASS":"FAIL", grel, identical, elements, gpass?"PASS":"FAIL");
    }
    // negative probes: a lone group, and a shape mismatch inside the batch, must both refuse
    {
        Exl3Group one{g0.gpu,u0.gpu,d0.gpu,0,20};
        Exl3MoeWorkspace workspace;
        workspace.reserve_recon_batch(2, 2 * 20 + 16, width, hidden, width);
        std::vector<uint16_t> hx(8 * width, 0);
        std::vector<int> hr(20, 0), hd(20);
        for (int r = 0; r < 20; ++r) hd[(size_t) r] = 19 - r;
        Buffer<uint16_t> dx(hx.size()); Buffer<int> dr(20), dd(20); Buffer<float> dy((size_t) 20*width);
        check(cudaMemcpy(dx.p, hx.data(), hx.size()*2, cudaMemcpyHostToDevice));
        check(cudaMemcpy(dr.p, hr.data(), hr.size()*4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(dd.p, hd.data(), hd.size()*4, cudaMemcpyHostToDevice));
        ok = ok && !exl3_moe_recon_batch(&one, 1, dx.p, dr.p, dd.p, dy.p, nullptr, workspace);
        // a REAL shape mismatch (384-wide vs the batch's 256) must refuse before any launch
        Matrix gbig(2,2,rng,3*N,3*N), ubig(2,2,rng,3*N,3*N), dbig(2,2,rng,3*N,3*N); (void) ubig; (void) dbig;
        Exl3Group two[]{Exl3Group{g0.gpu,u0.gpu,d0.gpu,0,20},Exl3Group{gbig.gpu,u0.gpu,d0.gpu,20,20}};
        ok = ok && !exl3_moe_recon_batch(two, 2, dx.p, dr.p, dd.p, dy.p, nullptr, workspace);
        std::printf("device=%d recon_batch negative_probes %s\n", device, ok?"PASS":"FAIL");
    }
    return ok;
}

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
    // MiMo's own expert mix through the INDEXED entry (the decode arm; STRATA_EXL3_RDNA's run
    // funnels take these K2/K2.5 tiles when the env is on - run this test at RDNA=0/1/2 and the
    // printed values must repeat, RDNA=1 bit-identical to RDNA=0 by construction): a mixed-K
    // route list, K2 and K2.5 interleaved per route, against the CPU reconstruction.
    {
        // Multiple H128 blocks and asymmetric FFN widths exercise the occupancy-aware
        // native RDNA split and epilogue, rather than only a single 128-wide transform.
        constexpr int N = 512, F = 256;
        Matrix fg(2,2,rng,N,F), fu(2,2,rng,N,F), fd(2,2,rng,F,N);            // K2
        Matrix hg(2,2,rng,N,F,1), hu(2,2,rng,N,F,1), hd(2,2,rng,F,N,1);      // K2.5
        std::vector<uint16_t> fx((size_t) 5*N);
        for (auto& v : fx) v=half(((int)(rng()%1000)-500)*0.001f);
        for (int routes : {3,17}) {
            std::vector<Exl3Mat> g(routes),u(routes),d(routes);
            std::vector<int> rows(routes);
            std::vector<float> fref((size_t) routes*N);
            for (int r=0;r<routes;++r) {
                const bool k25 = (r & 1) != 0;
                g[(size_t)r]=(k25?hg:fg).gpu; u[(size_t)r]=(k25?hu:fu).gpu; d[(size_t)r]=(k25?hd:fd).gpu;
                rows[(size_t)r]=(r*3)%5;
                auto gv=(k25?hg:fg).project(fx.data()+(size_t)rows[r]*N);
                const auto uv=(k25?hu:fu).project(fx.data()+(size_t)rows[r]*N);
                for (int i=0;i<F;++i) { const float v=fp32(gv[i]); gv[i]=half(v/(1+std::exp(-v))*fp32(uv[i])); }
                const auto y=(k25?hd:fd).project(gv.data());
                for (int i=0;i<N;++i) fref[(size_t)r*N+i]=fp32(y[i]);
            }
            Buffer<uint16_t> dfx(fx.size()); check(cudaMemcpy(dfx.p,fx.data(),fx.size()*2,cudaMemcpyHostToDevice));
            Buffer<int> dr(routes); check(cudaMemcpy(dr.p,rows.data(),routes*4,cudaMemcpyHostToDevice));
            Buffer<uint16_t> dy((size_t) routes*N);
            exl3_moe_rows_indexed(g.data(),u.data(),d.data(),routes,dfx.p,dr.p,dy.p,nullptr,workspace);
            std::vector<uint16_t> y((size_t) routes*N);
            check(cudaMemcpy(y.data(),dy.p,y.size()*2,cudaMemcpyDeviceToHost));
            double num=0,den=0;
            for(size_t i=0;i<y.size();++i) { const double delta=fp32(y[i])-fref[i]; num+=delta*delta; den+=fref[i]*fref[i]; }
            const double rel=std::sqrt(num/(den+1e-30));
            const bool pass=std::isfinite(rel)&&rel<0.02;
            ok=ok&&pass;
            std::printf("device=%d indexed_mixed_frac K2/K2.5 routes=%d relative_error=%.5g %s\n",
                        device,routes,rel,pass?"PASS":"FAIL");
        }
    }
    // The GROUPED entry (the SMALL_M / fused-multirow verify-window arm; its RDNA twin and the
    // FUSE=3 fused form take these K2/K2.5 tiles when the env is on) on mixed-K groups of 1..4
    // rows against the CPU reconstruction.  A batch of groups, K2 and K2.5 interleaved.
    {
        Matrix ag(2,2,rng,N,N), au(2,2,rng,N,N), ad(2,2,rng,N,N);            // K2
        Matrix bg(2,2,rng,N,N,1), bu(2,2,rng,N,N,1), bd(2,2,rng,N,N,1);      // K2.5
        const int sizes[] = {1, 2, 1, 3, 4, 2, 1, 4};
        std::vector<int> first, grows;
        int routes = 0;
        for (int s : sizes) { first.push_back(routes); grows.push_back(s); routes += s; }
        const int tokens = routes + 7;
        std::vector<uint16_t> gx((size_t) tokens*N);
        for (auto& v : gx) v=half(((int)(rng()%1000)-500)*0.001f);
        std::vector<int> rows(routes);
        std::vector<float> gref((size_t) routes*N);
        for (int r=0;r<routes;++r) rows[r]=(r*7+3)%tokens;
        for (int g=0;g<8;++g)
            for (int r=0;r<sizes[g];++r) {
                const int at = first[(size_t) g] + r;
                const bool k25 = (g & 1) != 0;
                auto gv=(k25?bg:ag).project(gx.data()+(size_t)rows[at]*N);
                const auto uv=(k25?bu:au).project(gx.data()+(size_t)rows[at]*N);
                for (int i=0;i<N;++i) { const float v=fp32(gv[i]); gv[i]=half(v/(1+std::exp(-v))*fp32(uv[i])); }
                const auto y=(k25?bd:ad).project(gv.data());
                for (int i=0;i<N;++i) gref[(size_t) at*N+i]=fp32(y[i]);
            }
        Buffer<uint16_t> dgx(gx.size()); check(cudaMemcpy(dgx.p,gx.data(),gx.size()*2,cudaMemcpyHostToDevice));
        Buffer<int> drows(routes); check(cudaMemcpy(drows.p,rows.data(),routes*4,cudaMemcpyHostToDevice));
        Buffer<uint16_t> dgy((size_t) routes*N);
        std::vector<Exl3Mat> gg(8), gu(8), gd(8);
        for (int g=0;g<8;++g) {
            const bool k25 = (g & 1) != 0;
            gg[(size_t)g]=(k25?bg:ag).gpu; gu[(size_t)g]=(k25?bu:au).gpu; gd[(size_t)g]=(k25?bd:ad).gpu;
        }
        exl3_moe_rows_grouped(gg.data(),gu.data(),gd.data(),first.data(),grows.data(),8,routes,
                              dgx.p,drows.p,dgy.p,nullptr,workspace);
        std::vector<uint16_t> gy((size_t) routes*N);
        check(cudaMemcpy(gy.data(),dgy.p,gy.size()*2,cudaMemcpyDeviceToHost));
        double num=0,den=0;
        for(size_t i=0;i<gy.size();++i) { const double delta=fp32(gy[i])-gref[i]; num+=delta*delta; den+=gref[i]*gref[i]; }
        const double rel=std::sqrt(num/(den+1e-30));
        const bool pass=std::isfinite(rel)&&rel<0.02;
        ok=ok&&pass;
        std::printf("device=%d grouped_mixed_frac K2/K2.5 groups=8 rows=%d relative_error=%.5g %s\n",
                    device,routes,rel,pass?"PASS":"FAIL");
        // The same mixed-K window through the FUSED grouped entry (the peer's PEER_MULTIROW=2 arm
        // shape; the opt-in frac admission was removed 2026-11-03 - no reliable end-to-end win).
        // On THESE fixtures: byte-identical to the unfused twin and to the INDEXED entry at the
        // same call shape and route list (a tested-fixture fact, not a claim over all frac
        // widths/geometries), plus the same CPU-reconstruction tolerance.  SKIP where the fused
        // build flags are absent.
        if (strata::kernels::exl3_fused_multirow_supported()) {
            workspace.reserve_small_m(routes, 8, N, N, N);
            workspace.reserve_runtime_scratch(routes, N, N, N);
            exl3_moe_rows_grouped(gg.data(),gu.data(),gd.data(),first.data(),grows.data(),8,routes,
                                  dgx.p,drows.p,dgy.p,nullptr,workspace,true);
            std::vector<uint16_t> fy((size_t) routes*N);
            check(cudaMemcpy(fy.data(),dgy.p,fy.size()*2,cudaMemcpyDeviceToHost));
            double fnum=0,fden=0;
            for(size_t i=0;i<fy.size();++i) { const double delta=fp32(fy[i])-gref[i]; fnum+=delta*delta; fden+=gref[i]*gref[i]; }
            const double frel=std::sqrt(fnum/(fden+1e-30));
            const bool fpass=std::isfinite(frel)&&frel<0.02 && fy==gy;
            ok=ok&&fpass;
            std::printf("device=%d grouped_fused_mixed_frac K2/K2.5 groups=8 rows=%d relative_error=%.5g bytes_equal_unfused=%d %s\n",
                        device,routes,frel,(int)(fy==gy),fpass?"PASS":"FAIL");
            // ... and byte-identical to the indexed entry on the same route list (the native
            // indexed arm - STRATA_EXL3_RDNA_NATIVE=1 - is a different kernel with its own fdot2
            // order, not byte-matched to the funnel family by construction; there the equality
            // is reported, not enforced)
            const char* native_env = std::getenv("STRATA_EXL3_RDNA_NATIVE");
            const bool native_on = native_env && native_env[0] == '1';
            std::vector<Exl3Mat> ig((size_t) routes), iu((size_t) routes), idd((size_t) routes);
            int at=0;
            for (int g=0;g<8;++g) for (int r=0;r<sizes[g];++r) {
                ig[(size_t) at]=gg[(size_t) g]; iu[(size_t) at]=gu[(size_t) g]; idd[(size_t) at]=gd[(size_t) g]; ++at;
            }
            exl3_moe_rows_indexed(ig.data(),iu.data(),idd.data(),routes,dgx.p,drows.p,dgy.p,nullptr,workspace);
            std::vector<uint16_t> iy((size_t) routes*N);
            check(cudaMemcpy(iy.data(),dgy.p,iy.size()*2,cudaMemcpyDeviceToHost));
            const bool iequal = iy==fy;
            const bool ipass = !native_on ? iequal : true;
            ok=ok&&ipass;
            std::printf("device=%d grouped_fused_mixed_frac_vs_indexed K2/K2.5 rows=%d native=%d bytes_equal=%d %s\n",
                        device,routes,(int)native_on,(int)iequal,ipass?"PASS":"FAIL");
        } else {
            std::printf("device=%d grouped_fused_mixed_frac K2/K2.5 SKIP (needs RDNA=1 FUSE=3)\n",device);
        }
    }
    // STRATA_EXL3_NATIVE_GROUPED (default off): the native grouped GEMV - the indexed native
    // kernel's packed mul1 decode + fdot2 core on the GROUP descriptors, one weight decode
    // per group - against the SAME CPU reconstruction, at the real MiMo geometry
    // (4096 -> 2048 -> 4096) and at a small fixture: group widths 1..4 mixed in one window,
    // K2 and K2.5 groups interleaved per GROUP (both rates in one launch), permuted token
    // rows, repeated replays across two streams/workspaces, a finite-value stress, a
    // poisoned-token nonfinite check, a captured-graph replay that must keep the dispatch it
    // was captured with, and REFUSAL cases (K3 tiles, 5-row groups) that must reproduce the
    // established arm's bytes exactly.  The dispatch reads both envs per call, so the A/B
    // arms run in one process here; byte identity between the native and established arms is
    // NOT expected (different accumulation order by design) and is only reported.
    {
        auto set_arm = [&](int native) {   // flip the grouped dispatch for the A/B arms
#if defined(_WIN32)
            _putenv_s("STRATA_EXL3_NATIVE_GROUPED", native ? "1" : "0");
            _putenv_s("STRATA_EXL3_RDNA_NATIVE", native ? "1" : "0");
#else
            setenv("STRATA_EXL3_NATIVE_GROUPED", native ? "1" : "0", 1);
            setenv("STRATA_EXL3_RDNA_NATIVE", native ? "1" : "0", 1);
#endif
        };
        struct ArmSave {
            std::string grp, nat;
            ArmSave() {
                const char* g = std::getenv("STRATA_EXL3_NATIVE_GROUPED");
                const char* n = std::getenv("STRATA_EXL3_RDNA_NATIVE");
                if (g) grp = g;
                if (n) nat = n;
            }
            ~ArmSave() {
#if defined(_WIN32)
                if (grp.empty()) _putenv_s("STRATA_EXL3_NATIVE_GROUPED", "0");
                else _putenv_s("STRATA_EXL3_NATIVE_GROUPED", grp.c_str());
                if (nat.empty()) _putenv_s("STRATA_EXL3_RDNA_NATIVE", "0");
                else _putenv_s("STRATA_EXL3_RDNA_NATIVE", nat.c_str());
#else
                if (grp.empty()) unsetenv("STRATA_EXL3_NATIVE_GROUPED");
                else setenv("STRATA_EXL3_NATIVE_GROUPED", grp.c_str(), 1);
                if (nat.empty()) unsetenv("STRATA_EXL3_RDNA_NATIVE");
                else setenv("STRATA_EXL3_RDNA_NATIVE", nat.c_str(), 1);
#endif
            }
        } arm_save;
        (void) arm_save;
        constexpr int H = 4096, FF = 2048;
        // four experts - two K2, two K2.5 - in an interleaved group plan.  Widths stay <= 2:
        // the peer's production chunk width (and the established FUSED arm's own LDS bound
        // at k=4096 rejects wider float slices); widths 3/4 run at the 512-wide fixture below.
        const int widths[] = {2, 1, 2, 2, 1, 2, 1, 2};
        const int ng = 8;
        std::vector<int> first, grows;
        int routes = 0;
        for (int s : widths) { first.push_back(routes); grows.push_back(s); routes += s; }
        Matrix e0g(2,2,rng,H,FF),   e0u(2,2,rng,H,FF),   e0d(2,2,rng,FF,H);      // K2
        Matrix e1g(2,2,rng,H,FF,1), e1u(2,2,rng,H,FF,1), e1d(2,2,rng,FF,H,1);    // K2.5
        Matrix e2g(2,2,rng,H,FF),   e2u(2,2,rng,H,FF),   e2d(2,2,rng,FF,H);      // K2
        Matrix e3g(2,2,rng,H,FF,1), e3u(2,2,rng,H,FF,1), e3d(2,2,rng,FF,H,1);    // K2.5
        Matrix* eg[4] = {&e0g, &e1g, &e2g, &e3g}, * eu[4] = {&e0u, &e1u, &e2u, &e3u},
               * ed[4] = {&e0d, &e1d, &e2d, &e3d};
        std::vector<Exl3Mat> gg(ng), gu(ng), gd(ng);
        for (int g = 0; g < ng; ++g) {
            gg[(size_t)g] = eg[g % 4]->gpu; gu[(size_t)g] = eu[g % 4]->gpu; gd[(size_t)g] = ed[g % 4]->gpu;
        }
        const int tokens = 27;
        std::vector<uint16_t> hx((size_t) tokens * H);
        for (auto& v : hx) v = half(((int) (rng() % 2000) - 1000) * 0.001f);
        std::vector<int> rows(routes);
        for (int r = 0; r < routes; ++r) rows[(size_t) r] = (r * 7 + 5) % tokens;   // permuted gather
        std::vector<double> ref((size_t) routes * H);
        for (int g = 0; g < ng; ++g)
            for (int r = 0; r < widths[g]; ++r) {
                const int at = first[(size_t) g] + r;
                auto gv = eg[g % 4]->project(hx.data() + (size_t) rows[(size_t) at] * H);
                const auto uv = eu[g % 4]->project(hx.data() + (size_t) rows[(size_t) at] * H);
                for (int i = 0; i < FF; ++i) {
                    const float v = fp32(gv[(size_t) i]);
                    gv[(size_t) i] = half(v / (1 + std::exp(-v)) * fp32(uv[(size_t) i]));
                }
                const auto y = ed[g % 4]->project(gv.data());
                for (int i = 0; i < H; ++i) ref[(size_t) at * H + i] = fp32(y[(size_t) i]);
            }
        Buffer<uint16_t> dx(hx.size());
        check(cudaMemcpy(dx.p, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        Buffer<int> drows(routes);
        check(cudaMemcpy(drows.p, rows.data(), routes * 4, cudaMemcpyHostToDevice));
        Buffer<uint16_t> dy((size_t) routes * H);
        Exl3MoeWorkspace ws_a, ws_b;
        ws_a.reserve(routes, H, FF, H, true);
        ws_a.reserve_small_m(routes, ng, H, FF, H);
        ws_a.reserve_runtime_scratch(routes, H, FF, H);
        cudaStream_t sa, sb;
        check(cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking));
        check(cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking));
        auto run = [&](bool native, bool fused, uint16_t* out) {
            set_arm(native);
            exl3_moe_rows_grouped(gg.data(), gu.data(), gd.data(), first.data(), grows.data(),
                                  ng, routes, dx.p, drows.p, out, sa, ws_a, fused);
            const cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
        };
        auto fetch = [&](uint16_t* out) {
            std::vector<uint16_t> y((size_t) routes * H);
            check(cudaStreamSynchronize(sa));
            check(cudaMemcpy(y.data(), out, y.size() * 2, cudaMemcpyDeviceToHost));
            return y;
        };
        auto rel_of = [&](const std::vector<uint16_t>& y) {
            double num = 0, den = 0;
            for (size_t i = 0; i < y.size(); ++i) {
                const double delta = fp32(y[i]) - ref[i];
                num += delta * delta; den += ref[i] * ref[i];
            }
            return std::sqrt(num / (den + 1e-30));
        };
        auto finite_all = [&](const std::vector<uint16_t>& y) {
            for (uint16_t v : y) if (!std::isfinite(__half2float(__ushort_as_half(v)))) return false;
            return true;
        };
        run(false, false, dy.p);
        const auto base_u = fetch(dy.p);
        run(true, false, dy.p);
        const auto nat_u = fetch(dy.p);
        const double rel_bu = rel_of(base_u), rel_nu = rel_of(nat_u);
        bool pass_u = std::isfinite(rel_nu) && rel_nu < 0.02 && finite_all(nat_u);
        size_t diff_u = 0;
        for (size_t i = 0; i < nat_u.size(); ++i) diff_u += nat_u[i] != base_u[i];
        ok = ok && pass_u;
        std::printf("device=%d native_grouped_unfused K2/K2.5 groups=8 rows=%d rel_base=%.5g rel_native=%.5g "
                    "bytes_diff_frac=%.4f %s\n", device, routes, rel_bu, rel_nu,
                    (double) diff_u / nat_u.size(), pass_u ? "PASS" : "FAIL");
        const bool fused_supported = strata::kernels::exl3_fused_multirow_supported();
        std::vector<uint16_t> base_f, nat_f;
        if (fused_supported) {
            run(false, true, dy.p);
            base_f = fetch(dy.p);
            run(true, true, dy.p);
            nat_f = fetch(dy.p);
            const double rel_bf = rel_of(base_f), rel_nf = rel_of(nat_f);
            // byte identity fused-vs-unfused is NOT asserted: the two arms pick different
            // parts counts (occupancy split), so the float partial sums group differently -
            // reported, and each arm is held to the CPU oracle independently
            bool pass_f = std::isfinite(rel_nf) && rel_nf < 0.02 && finite_all(nat_f);
            size_t diff_f = 0, diff_fu = 0;
            for (size_t i = 0; i < nat_f.size(); ++i) {
                diff_f += nat_f[i] != base_f[i];
                diff_fu += nat_f[i] != nat_u[i];
            }
            ok = ok && pass_f;
            std::printf("device=%d native_grouped_fused K2/K2.5 groups=8 rows=%d rel_base=%.5g rel_native=%.5g "
                        "bytes_diff_frac=%.4f bytes_diff_frac_vs_unfused=%.4f %s\n", device, routes, rel_bf, rel_nf,
                        (double) diff_f / nat_f.size(), (double) diff_fu / nat_f.size(), pass_f ? "PASS" : "FAIL");
            // captured-graph replay: captured with the flag ON, then the flag is turned OFF -
            // replays must reproduce the NATIVE bytes (an enqueued/captured launch keeps the
            // dispatch it was enqueued with)
            check(cudaStreamSynchronize(sa));
            cudaGraph_t graph; cudaGraphExec_t exec;
            check(cudaStreamBeginCapture(sa, cudaStreamCaptureModeThreadLocal));
            set_arm(true);
            exl3_moe_rows_grouped(gg.data(), gu.data(), gd.data(), first.data(), grows.data(),
                                  ng, routes, dx.p, drows.p, dy.p, sa, ws_a, true);
            check(cudaStreamEndCapture(sa, &graph));
            check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            set_arm(false);
            bool graph_ok = true;
            for (int rep = 0; rep < 3; ++rep) {
                check(cudaGraphLaunch(exec, sa));
                check(cudaStreamSynchronize(sa));
                const auto got = fetch(dy.p);
                graph_ok = graph_ok && got == nat_f;
            }
            check(cudaGraphExecDestroy(exec));
            check(cudaGraphDestroy(graph));
            ok = ok && graph_ok;
            std::printf("device=%d native_grouped_graph_replay flag_off_replays_stay_native=%d %s\n",
                        device, (int) graph_ok, graph_ok ? "PASS" : "FAIL");
            // repeat replays across a SECOND workspace/stream interleaved with this one: each
            // arm's every output must equal its own first (scratch reuse is byte-stable)
            ws_b.reserve(routes, H, FF, H, true);
            ws_b.reserve_small_m(routes, ng, H, FF, H);
            ws_b.reserve_runtime_scratch(routes, H, FF, H);
            Buffer<uint16_t> dy2((size_t) routes * H);
            bool stable = true;
            for (int rep = 0; rep < 4; ++rep) {
                set_arm(false);
                exl3_moe_rows_grouped(gg.data(), gu.data(), gd.data(), first.data(), grows.data(),
                                      ng, routes, dx.p, drows.p, dy.p, sa, ws_a, true);
                set_arm(true);
                exl3_moe_rows_grouped(gg.data(), gu.data(), gd.data(), first.data(), grows.data(),
                                      ng, routes, dx.p, drows.p, dy2.p, sb, ws_b, true);
                check(cudaStreamSynchronize(sa));
                check(cudaStreamSynchronize(sb));
                std::vector<uint16_t> a((size_t) routes * H), b((size_t) routes * H);
                check(cudaMemcpy(a.data(), dy.p, a.size() * 2, cudaMemcpyDeviceToHost));
                check(cudaMemcpy(b.data(), dy2.p, b.size() * 2, cudaMemcpyDeviceToHost));
                stable = stable && a == base_f && b == nat_f;
            }
            ok = ok && stable;
            std::printf("device=%d native_grouped_replays two_streams_byte_stable=%d %s\n",
                        device, (int) stable, stable ? "PASS" : "FAIL");
        } else {
            std::printf("device=%d native_grouped_fused K2/K2.5 SKIP (needs RDNA=1 FUSE=3)\n", device);
        }
        // refusal: K5 tiles and 5-row groups keep the established arm (byte-identical to the
        // flag-off run - same kernel, same order).  K3/K4 now ADMIT (the Qwen rates); their
        // positive battery follows below.
        {
            Matrix k5g(5,2,rng,H,FF), k5u(5,2,rng,H,FF), k5d(5,2,rng,FF,H);
            std::vector<Exl3Mat> r5g{ k5g.gpu, k5g.gpu }, r5u{ k5u.gpu, k5u.gpu }, r5d{ k5d.gpu, k5d.gpu };
            std::vector<int> f5_{ 0, 2 }, n5_{ 2, 2 };
            set_arm(false);
            exl3_moe_rows_grouped(r5g.data(), r5u.data(), r5d.data(), f5_.data(), n5_.data(),
                                  2, 4, dx.p, drows.p, dy.p, sa, ws_a);
            const auto off5k = fetch(dy.p);
            set_arm(true);
            exl3_moe_rows_grouped(r5g.data(), r5u.data(), r5d.data(), f5_.data(), n5_.data(),
                                  2, 4, dx.p, drows.p, dy.p, sa, ws_a);
            const auto on5k = fetch(dy.p);
            const bool refuse5k = on5k == off5k;
            ok = ok && refuse5k;
            std::printf("device=%d native_grouped_refusal K5 stays established=%d %s\n",
                        device, (int) refuse5k, refuse5k ? "PASS" : "FAIL");
            const int w5[] = { 1, 5, 2 };
            std::vector<int> f5{ 0, 1, 6 }, n5{ 1, 5, 2 };
            std::vector<Exl3Mat> g5(3), u5(3), d5(3);
            for (int g = 0; g < 3; ++g) {
                g5[(size_t)g] = (g & 1 ? e1g : e0g).gpu; u5[(size_t)g] = (g & 1 ? e1u : e0u).gpu;
                d5[(size_t)g] = (g & 1 ? e1d : e0d).gpu;
            }
            (void) w5;
            set_arm(false);
            exl3_moe_rows_grouped(g5.data(), u5.data(), d5.data(), f5.data(), n5.data(),
                                  3, 8, dx.p, drows.p, dy.p, sa, ws_a);
            const auto off5 = fetch(dy.p);
            set_arm(true);
            exl3_moe_rows_grouped(g5.data(), u5.data(), d5.data(), f5.data(), n5.data(),
                                  3, 8, dx.p, drows.p, dy.p, sa, ws_a);
            const auto on5 = fetch(dy.p);
            const bool refuse5 = on5 == off5;
            ok = ok && refuse5;
            std::printf("device=%d native_grouped_refusal 5-row groups stay established=%d %s\n",
                        device, (int) refuse5, refuse5 ? "PASS" : "FAIL");
        }
        // host-only admission unit (audit item: NO GPU work, no kernel launches, no pointer
        // dereference - the descriptors carry null pointers on purpose): the plan must refuse
        // every descriptor the native core cannot decode exactly (any frac outside {0,1}
        // included - the generic geometry validation does NOT reject those), refuse >4-row
        // windows and unarmed callers, admit the real MiMo shapes, and refuse LDS-infeasible
        // wide-k shapes at the split cap instead of enqueueing an ill-fitting launch
        {
            auto mat = [](int64_t ki, int64_t nj, int bits, int cb, int frac) {
                Exl3Mat m{};
                m.ki = (int) ki; m.nj = (int) nj; m.bits = bits; m.cb = cb; m.frac = frac;
                m.trellis = m.suh = m.svh = nullptr;   // never dereferenced by the plan
                return m;
            };
            int rates[3], parts[3];
            auto call = [&](const std::vector<Exl3Mat>& gg_, const std::vector<Exl3Mat>& uu_,
                            const std::vector<Exl3Mat>& dd_, int groups_, int max_rows_,
                            bool armed, bool fused_) {
                return exl3_native_grouped_plan(gg_.data(), uu_.data(), dd_.data(),
                                                groups_, max_rows_, armed, fused_, rates, parts);
            };
            auto admit = [&](const Exl3Mat& g, const Exl3Mat& u, const Exl3Mat& d,
                             int groups_, int max_rows_, bool armed, bool fused_) {
                // the plan reads one descriptor PER GROUP: hand it real arrays
                std::vector<Exl3Mat> gg_(groups_, g), uu_(groups_, u), dd_(groups_, d);
                return call(gg_, uu_, dd_, groups_, max_rows_, armed, fused_);
            };
            // refusal with POISONED outputs: false must leave every rate and part at 0 -
            // never the partial accumulation from groups/projections that were fine before
            // the refusal point (the public helper's contract; audit round 2)
            auto refuse_clean = [&](const std::vector<Exl3Mat>& gg_, const std::vector<Exl3Mat>& uu_,
                                    const std::vector<Exl3Mat>& dd_, int groups_, int max_rows_,
                                    bool armed, bool fused_) {
                for (int b = 0; b < 3; ++b) { rates[b] = 0x7f; parts[b] = 0x7f; }
                if (call(gg_, uu_, dd_, groups_, max_rows_, armed, fused_)) return false;
                for (int b = 0; b < 3; ++b)
                    if (rates[b] != 0 || parts[b] != 0) return false;
                return true;
            };
            const Exl3Mat g2 = mat(256, 128, 2, 2, 0), f25 = mat(256, 128, 2, 2, 1);
            const Exl3Mat d2 = mat(128, 256, 2, 2, 0), d25 = mat(128, 256, 2, 2, 1);
            bool host_ok = true;
            struct Case { const char* what; bool want; bool got; };
            std::vector<Case> cases;
            cases.push_back({"unarmed refuses", false, admit(g2, f25, d2, 4, 2, false, false)});
            cases.push_back({"frac=2 refuses", false, admit(g2, mat(256, 128, 2, 2, 2), d2, 4, 2, true, false)});
            cases.push_back({"frac=-1 refuses", false, admit(g2, mat(256, 128, 2, 2, -1), d2, 4, 2, true, false)});
            // K3 in the gate with K2 up/down: ADMITS now (mixed per-projection rates), and
            // the rate bits must be per projection: gate K3 (bit2), up/down K2 (bit0)
            cases.push_back({"gate K3 + K2 up/down admits (mixed projection rates)", true,
                              admit(mat(256, 128, 3, 2, 0), g2, d2, 4, 2, true, false)});
            const bool mix_ok = cases.back().got && rates[0] == 4 && rates[1] == 1 && rates[2] == 1;
            cases.push_back({"pure K3 admits (bit2 in every projection)", true,
                              admit(mat(256, 128, 3, 2, 0), mat(256, 128, 3, 2, 0),
                                    mat(128, 256, 3, 2, 0), 4, 2, true, false)});
            const bool k3_ok = cases.back().got && rates[0] == 4 && rates[1] == 4 && rates[2] == 4;
            cases.push_back({"pure K4 admits (bit3 in every projection)", true,
                              admit(mat(256, 128, 4, 2, 0), mat(256, 128, 4, 2, 0),
                                    mat(128, 256, 4, 2, 0), 4, 2, true, false)});
            const bool k4_ok = cases.back().got && rates[0] == 8 && rates[1] == 8 && rates[2] == 8;
            cases.push_back({"K3.5 (bits=3 frac=1) refuses", false,
                              admit(mat(256, 128, 3, 2, 1), g2, d2, 4, 2, true, false)});
            cases.push_back({"K4.5 (bits=4 frac=1) refuses", false,
                              admit(g2, mat(256, 128, 4, 2, 1), d2, 4, 2, true, false)});
            cases.push_back({"bits=5 refuses", false, admit(mat(256, 128, 5, 2, 0), g2, d2, 4, 2, true, false)});
            cases.push_back({"bits=1 refuses", false, admit(mat(256, 128, 1, 2, 0), g2, d2, 4, 2, true, false)});
            cases.push_back({"cb=1 refuses", false, admit(mat(256, 128, 2, 1, 0), g2, d2, 4, 2, true, false)});
            cases.push_back({"5-row window refuses", false, admit(g2, f25, d2, 4, 5, true, false)});
            // the real Qwen expert rates (whole-expert homogeneous, both models)
            cases.push_back({"Qwen 3bpw K3 2560/640 admits", true,
                              admit(mat(160, 40, 3, 2, 0), mat(160, 40, 3, 2, 0),
                                    mat(40, 160, 3, 2, 0), 8, 4, true, false)});
            const bool qw3_ok = cases.back().got && rates[0] == 4 && rates[2] == 4;
            cases.push_back({"Qwen 4bpw K4 2560/640 admits", true,
                              admit(mat(160, 40, 4, 2, 0), mat(160, 40, 4, 2, 0),
                                    mat(40, 160, 4, 2, 0), 8, 4, true, false)});
            const bool qw4_ok = cases.back().got && rates[0] == 8 && rates[2] == 8;
            // invalid LATER groups / mid-projection: group 0 (and more) is fine, the bad
            // descriptor arrives after the accumulator already holds real rates - the
            // refusal must still leave every output zero
            {
                const int ng = 4;
                std::vector<Exl3Mat> gg_(ng, g2), uu_(ng, f25), dd_(ng, d2);
                gg_[ng - 1].bits = 5;   // LAST group's gate: bits invalid after valid prefix
                cases.push_back({"later group bits=5 refuses clean", true,
                                 refuse_clean(gg_, uu_, dd_, ng, 2, true, false)});
                gg_[ng - 1] = g2; uu_[1].frac = 2;   // MIDDLE group's up: frac invalid
                cases.push_back({"mid group up frac=2 refuses clean", true,
                                 refuse_clean(gg_, uu_, dd_, ng, 2, true, false)});
                uu_[1] = f25; dd_[2].cb = 1;         // MIDDLE group's down: codebook invalid
                cases.push_back({"mid group down cb=1 refuses clean", true,
                                 refuse_clean(gg_, uu_, dd_, ng, 2, true, false)});
                dd_[2] = d25; uu_[1].bits = 4; uu_[1].frac = 1;   // K4.5: unsupported frac
                cases.push_back({"mid group up K4.5 refuses clean", true,
                                 refuse_clean(gg_, uu_, dd_, ng, 2, true, false)});
                uu_[1] = f25; gg_[0].frac = -1;      // FIRST group invalid again
                cases.push_back({"first group frac=-1 refuses clean", true,
                                 refuse_clean(gg_, uu_, dd_, ng, 2, true, false)});
            }
            cases.push_back({"MiMo width 2 admits", true, admit(g2, f25, d2, 8, 2, true, false)});
            const bool w2_ok = cases.back().got && parts[0] >= 1 && parts[2] >= 1;
            cases.push_back({"MiMo width 4 admits", true, admit(g2, f25, d25, 8, 4, true, false)});
            const bool w4_ok = cases.back().got && parts[0] >= 1 && parts[2] >= 1;
            // LDS feasibility at the split cap (setenv is safe: the plan reads it per call)
            setenv("STRATA_EXL3_GEMV_SPLIT", "1", 1);
            const Exl3Mat w5120 = mat(320, 128, 2, 2, 0), d5120 = mat(128, 320, 2, 2, 0);
            cases.push_back({"k=5120 width 4 split 1 unfused fits exactly", true,
                             admit(w5120, w5120, d5120, 4, 4, true, false)});
            const bool exact_ok = cases.back().got && parts[0] == 1 && parts[2] == 1;
            {   // LDS refusal AFTER the rates were computed and committed to the locals:
                // poisoned outputs must still come back all zero
                const int ng = 4;
                std::vector<Exl3Mat> gg_(ng, w5120), uu_(ng, w5120), dd_(ng, d5120);
                cases.push_back({"k=5120 width 4 split 1 FUSED (1 KiB static) refuses clean", true,
                                 refuse_clean(gg_, uu_, dd_, ng, 4, true, true)});
            }
            const Exl3Mat w8192 = mat(512, 128, 2, 2, 0), d8192 = mat(128, 512, 2, 2, 0);
            {   // the 64 KiB-slice refusal, same poisoned-output proof
                const int ng = 4;
                std::vector<Exl3Mat> gg_(ng, w8192), uu_(ng, w8192), dd_(ng, d8192);
                cases.push_back({"k=8192 width 4 split 1 refuses clean", true,
                                 refuse_clean(gg_, uu_, dd_, ng, 4, true, false)});
            }
            cases.push_back({"k=8192 width 1 split 1 admits", true,
                             admit(w8192, w8192, d8192, 4, 1, true, false)});
            unsetenv("STRATA_EXL3_GEMV_SPLIT");
            for (const Case& c : cases)
                if (c.want != c.got)
                    std::printf("device=%d native_grouped_plan unit: %s want=%d got=%d FAIL\n",
                                device, c.what, (int) c.want, (int) c.got);
            host_ok = host_ok && w2_ok && w4_ok && exact_ok && mix_ok && k3_ok && k4_ok && qw3_ok && qw4_ok;
            for (const Case& c : cases) host_ok = host_ok && (c.want == c.got);
            ok = ok && host_ok;
            std::printf("device=%d native_grouped_plan unit (host-only, null pointers, no launches): "
                        "%zu cases %s\n", device, cases.size(), host_ok ? "PASS" : "FAIL");
        }
        // GPU refusal proof for the two audit holes (frac outside {0,1}; LDS-infeasible
        // wide k at the split cap): with the flag ARMED the entry must run the established
        // arm and reproduce the unarmed bytes exactly - the malformed descriptors are the
        // caller's own trellis-sized fixtures, so the fallback reads only real memory
        {
            {   // frac=2 (words=48/tile): generic validation passes, native must refuse
                Matrix xg(2,2,rng,H,FF,2), xu(2,2,rng,H,FF,2), xd(2,2,rng,FF,H,2);
                std::vector<Exl3Mat> xgg{xg.gpu, xg.gpu}, xuu{xu.gpu, xu.gpu}, xdd{xd.gpu, xd.gpu};
                std::vector<int> xf{0, 2}, xn{2, 2};
                set_arm(false);
                exl3_moe_rows_grouped(xgg.data(), xuu.data(), xdd.data(), xf.data(), xn.data(),
                                      2, 4, dx.p, drows.p, dy.p, sa, ws_a);
                const auto offx = fetch(dy.p);
                set_arm(true);
                exl3_moe_rows_grouped(xgg.data(), xuu.data(), xdd.data(), xf.data(), xn.data(),
                                      2, 4, dx.p, drows.p, dy.p, sa, ws_a);
                const auto onx = fetch(dy.p);
                const bool refusex = onx == offx;
                ok = ok && refusex;
                std::printf("device=%d native_grouped_refusal frac=2 stays established=%d %s\n",
                            device, (int) refusex, refusex ? "PASS" : "FAIL");
            }
            {   // K3.5 (bits=3, frac=1; words=56/tile): generic validation passes (frac
                // tiles up to K4.5 are legal EXL3), the native core decodes no K>=3 frac
                // tile - the whole call must refuse to the established arm
                Matrix hg(3,2,rng,H,FF,1), hu(3,2,rng,H,FF,1), hd(3,2,rng,FF,H,1);
                std::vector<Exl3Mat> hgg{hg.gpu, hg.gpu}, huu{hu.gpu, hu.gpu}, hdd{hd.gpu, hd.gpu};
                std::vector<int> hf{0, 2}, hn{2, 2};
                set_arm(false);
                exl3_moe_rows_grouped(hgg.data(), huu.data(), hdd.data(), hf.data(), hn.data(),
                                      2, 4, dx.p, drows.p, dy.p, sa, ws_a);
                const auto offh = fetch(dy.p);
                set_arm(true);
                exl3_moe_rows_grouped(hgg.data(), huu.data(), hdd.data(), hf.data(), hn.data(),
                                      2, 4, dx.p, drows.p, dy.p, sa, ws_a);
                const auto onh = fetch(dy.p);
                const bool refuseh = onh == offh;
                ok = ok && refuseh;
                std::printf("device=%d native_grouped_refusal K3.5 stays established=%d %s\n",
                            device, (int) refuseh, refuseh ? "PASS" : "FAIL");
            }
            {   // wide k (8192) with 4-row groups at GEMV_SPLIT=1: the native plan refuses
                // (64 KiB slice > 40 KiB budget); both arms run the established unfused arm
                constexpr int WK = 8192, WF = 2048;
                Matrix wg(2,2,rng,WK,WF), wu(2,2,rng,WK,WF), wd(2,2,rng,WF,WK);
                std::vector<Exl3Mat> wgg{wg.gpu, wg.gpu}, wuu{wu.gpu, wu.gpu}, wdd{wd.gpu, wd.gpu};
                std::vector<int> wf_{0, 4}, wn{4, 4};
                const int wroutes = 8;
                std::vector<uint16_t> wx((size_t) 5 * WK);
                for (auto& v : wx) v = half(((int) (rng() % 2000) - 1000) * 0.001f);
                std::vector<int> wrows(wroutes);
                for (int r = 0; r < wroutes; ++r) wrows[(size_t) r] = (r * 3 + 1) % 5;
                Buffer<uint16_t> wdx(wx.size());
                check(cudaMemcpy(wdx.p, wx.data(), wx.size() * 2, cudaMemcpyHostToDevice));
                Buffer<int> wdrows(wroutes);
                check(cudaMemcpy(wdrows.p, wrows.data(), wroutes * 4, cudaMemcpyHostToDevice));
                Buffer<uint16_t> wdy((size_t) wroutes * WK);
                Exl3MoeWorkspace ws_w;
                ws_w.reserve(wroutes, WK, WF, WK, true);
                ws_w.reserve_small_m(wroutes, 2, WK, WF, WK);
                setenv("STRATA_EXL3_GEMV_SPLIT", "1", 1);
                set_arm(false);
                exl3_moe_rows_grouped(wgg.data(), wuu.data(), wdd.data(), wf_.data(), wn.data(),
                                      2, wroutes, wdx.p, wdrows.p, wdy.p, sa, ws_w);
                const auto offw = [&] { check(cudaStreamSynchronize(sa));
                    std::vector<uint16_t> y((size_t) wroutes * WK);
                    check(cudaMemcpy(y.data(), wdy.p, y.size() * 2, cudaMemcpyDeviceToHost)); return y; }();
                set_arm(true);
                exl3_moe_rows_grouped(wgg.data(), wuu.data(), wdd.data(), wf_.data(), wn.data(),
                                      2, wroutes, wdx.p, wdrows.p, wdy.p, sa, ws_w);
                const auto onw = [&] { check(cudaStreamSynchronize(sa));
                    std::vector<uint16_t> y((size_t) wroutes * WK);
                    check(cudaMemcpy(y.data(), wdy.p, y.size() * 2, cudaMemcpyDeviceToHost)); return y; }();
                unsetenv("STRATA_EXL3_GEMV_SPLIT");
                const bool refusalw = onw == offw;
                ok = ok && refusalw;
                std::printf("device=%d native_grouped_refusal k=8192 width=4 split=1 (64KiB slice) "
                            "stays established=%d %s\n", device, (int) refusalw, refusalw ? "PASS" : "FAIL");
            }
        }
        // finite stress (values near the fp16 comfort ceiling) and a poisoned-token nonfinite
        // check at the small fixture size: every route reading the poisoned token must be
        // nonfinite on BOTH arms, every other row finite
        {
            constexpr int S = 512, F = 256;
            Matrix pg(2,2,rng,S,F), pu(2,2,rng,S,F), pd(2,2,rng,F,S);
            Matrix qg(2,2,rng,S,F,1), qu(2,2,rng,S,F,1), qd(2,2,rng,F,S,1);
            const int sw[] = { 2, 1, 4, 3, 2 };
            std::vector<int> pf, pn;
            int proutes = 0;
            for (int s : sw) { pf.push_back(proutes); pn.push_back(s); proutes += s; }
            std::vector<Exl3Mat> sgg(5), squ(5), sqd(5);
            for (int g = 0; g < 5; ++g) {
                const bool k25 = (g & 1) != 0;
                sgg[(size_t)g] = (k25 ? qg : pg).gpu; squ[(size_t)g] = (k25 ? qu : pu).gpu;
                sqd[(size_t)g] = (k25 ? qd : pd).gpu;
            }
            const int stok = 19, poison = 7;
            std::vector<uint16_t> sx((size_t) stok * S);
            for (auto& v : sx) v = half(((int) (rng() % 2000) - 1000) * 0.0012f);   // 1.2x larger
            std::vector<int> srows(proutes);
            for (int r = 0; r < proutes; ++r) srows[(size_t) r] = (r * 5 + 2) % stok;
            Buffer<uint16_t> dsx(sx.size());
            check(cudaMemcpy(dsx.p, sx.data(), sx.size() * 2, cudaMemcpyHostToDevice));
            Buffer<int> dsrows(proutes);
            check(cudaMemcpy(dsrows.p, srows.data(), proutes * 4, cudaMemcpyHostToDevice));
            Buffer<uint16_t> dsy((size_t) proutes * S);
            Exl3MoeWorkspace ws_s;
            ws_s.reserve(proutes, S, F, S, true);
            ws_s.reserve_small_m(proutes, 5, S, F, S);
            ws_s.reserve_runtime_scratch(proutes, S, F, S);
            auto srun = [&](bool native, bool poison_on, bool fused = false) {
                auto host = sx;
                if (poison_on) {
                    const uint16_t inf16 = __half_as_ushort(__float2half(1.0f / 0.0f));
                    for (int i = 0; i < S; ++i) host[(size_t) poison * S + i] = inf16;
                    check(cudaMemcpy(dsx.p, host.data(), host.size() * 2, cudaMemcpyHostToDevice));
                }
                set_arm(native);
                exl3_moe_rows_grouped(sgg.data(), squ.data(), sqd.data(), pf.data(), pn.data(),
                                      5, proutes, dsx.p, dsrows.p, dsy.p, sa, ws_s, fused);
                const cudaError_t e = cudaGetLastError();
                if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
                check(cudaStreamSynchronize(sa));
                std::vector<uint16_t> y((size_t) proutes * S);
                check(cudaMemcpy(y.data(), dsy.p, y.size() * 2, cudaMemcpyDeviceToHost));
                return y;
            };
            // CPU reference for the 8x stress activations (also covers the 3/4-row groups
            // the real-geometry plan above cannot fuse)
            std::vector<double> sref((size_t) proutes * S);
            {
                const int swidths[5] = { 2, 1, 4, 3, 2 };
                for (int g = 0; g < 5; ++g)
                    for (int r = 0; r < swidths[g]; ++r) {
                        const int at = pf[(size_t) g] + r;
                        auto gv = (g & 1 ? qg : pg).project(sx.data() + (size_t) srows[(size_t) at] * S);
                        const auto uv = (g & 1 ? qu : pu).project(sx.data() + (size_t) srows[(size_t) at] * S);
                        for (int i = 0; i < F; ++i) {
                            const float v = fp32(gv[(size_t) i]);
                            gv[(size_t) i] = half(v / (1 + std::exp(-v)) * fp32(uv[(size_t) i]));
                        }
                        const auto yv = (g & 1 ? qd : pd).project(gv.data());
                        for (int i = 0; i < S; ++i) sref[(size_t) at * S + i] = fp32(yv[(size_t) i]);
                    }
            }
            auto srel = [&](const std::vector<uint16_t>& y) {
                double num = 0, den = 0;
                for (size_t i = 0; i < y.size(); ++i) {
                    const double delta = fp32(y[i]) - sref[i];
                    num += delta * delta; den += sref[i] * sref[i];
                }
                return std::sqrt(num / (den + 1e-30));
            };
            auto count_nonfinite = [](const std::vector<uint16_t>& y, std::vector<int>* rows_out = nullptr) {
                size_t n = 0;
                for (size_t i = 0; i < y.size(); ++i)
                    if (!std::isfinite(__half2float(__ushort_as_half(y[i])))) {
                        ++n;
                        if (rows_out != nullptr) {
                            const int r = (int) (i / 512);
                            if (std::find(rows_out->begin(), rows_out->end(), r) == rows_out->end())
                                rows_out->push_back(r);
                        }
                    }
                return n;
            };
            const auto stress = srun(false, false), stress_n = srun(true, false);
            // the fused arms only exist where the launcher armed them (RDNA=1 FUSE=3)
            const bool stress_fused = strata::kernels::exl3_fused_multirow_supported();
            const auto stress_f = stress_fused ? srun(false, false, true) : stress;
            const auto stress_nf = stress_fused ? srun(true, false, true) : stress_n;
            std::vector<int> nf_rows_f, nf_rows_nf;
            const size_t nf_u = count_nonfinite(stress), nf_n = count_nonfinite(stress_n);
            const size_t nf_f = count_nonfinite(stress_f, &nf_rows_f), nf_nf = count_nonfinite(stress_nf, &nf_rows_nf);
            const std::string rows_f = [&]{ std::string o; for (int r : nf_rows_f) o += std::to_string(r) + ","; return o; }();
            const std::string rows_nf = [&]{ std::string o; for (int r : nf_rows_nf) o += std::to_string(r) + ","; return o; }();
            std::printf("device=%d native_grouped_stress nonfinite counts est_u=%zu nat_u=%zu est_f=%zu(rows %s) nat_f=%zu(rows %s)\n",
                        device, nf_u, nf_n, nf_f, rows_f.c_str(), nf_nf, rows_nf.c_str());
            const double r0 = srel(stress), r1 = srel(stress_n), r2 = srel(stress_f), r3 = srel(stress_nf);
            bool finite_ok = finite_all(stress) && finite_all(stress_n) && finite_all(stress_f) && finite_all(stress_nf);
            finite_ok = finite_ok && std::isfinite(r1) && r1 < 0.02 && std::isfinite(r3) && r3 < 0.02;
            ok = ok && finite_ok;
            std::printf("device=%d native_grouped_finite_stress 1.2x activations widths=2,1,4,3,2 fused=%d "
                        "finite=%d rel est_u=%.5g nat_u=%.5g est_f=%.5g nat_f=%.5g %s\n",
                        device, (int) stress_fused, (int) finite_ok, r0, r1, r2, r3,
                        finite_ok ? "PASS" : "FAIL");
            const auto pois = srun(false, true), pois_n = srun(true, true);  // unfused
            auto pois_ok_fn = [&](const std::vector<uint16_t>& y, const char* what) {
                bool good = true;
                for (int r = 0; r < proutes; ++r) {
                    const bool poisoned = srows[(size_t) r] == poison;
                    for (int i = 0; i < S; ++i) {
                        const bool fin = std::isfinite(__half2float(__ushort_as_half(y[(size_t) r * S + i])));
                        if (poisoned && fin) { good = false; break; }
                        if (!poisoned && !fin) { good = false; break; }
                    }
                    if (!good) break;
                }
                std::printf("device=%d native_grouped_nonfinite %s poisoned_rows_all_nonfinite=%d %s\n",
                            device, what, (int) good, good ? "PASS" : "FAIL");
                return good;
            };
            ok = ok && pois_ok_fn(pois, "established") && pois_ok_fn(pois_n, "native");
        }
        check(cudaStreamDestroy(sa));
        check(cudaStreamDestroy(sb));
    }
    // The K3/K4 native grouped battery (2026-11-05): the Qwen3.8-Flash-Next EXL3 3bpw/4bpw
    // expert rates are HOMOGENEOUS integer K3 / K4 mul1 at 2560->640->2560 (verified in the
    // safetensors headers: gate/up [160,40,48|64], down [40,160,48|64], 25088 expert trios),
    // so this battery runs the native arm at exactly that geometry: per-rate batteries
    // (K3, K4), a MIXED-rate window (K2 + K2.5 + K3 + K4 groups and a mixed-projection trio
    // in ONE launch - every rate instantiation dispatched together, each returning before any
    // shared state for the groups that are not its rate), widths 1..4 including the ragged
    // 1-row tails, permuted token gathers, unfused + fused vs the CPU reconstruction, a
    // captured-graph replay that must keep its dispatch, two-stream/two-workspace repeat
    // stability, and a poisoned-token nonfinite classification.  H=2560 keeps even the
    // flag-OFF fused arm inside its own 40 KiB float-slice bound at width 4, so both arms
    // run the fused form here.
    {
        auto set_arm = [&](int native) {
#if defined(_WIN32)
            _putenv_s("STRATA_EXL3_NATIVE_GROUPED", native ? "1" : "0");
            _putenv_s("STRATA_EXL3_RDNA_NATIVE", native ? "1" : "0");
#else
            setenv("STRATA_EXL3_NATIVE_GROUPED", native ? "1" : "0", 1);
            setenv("STRATA_EXL3_RDNA_NATIVE", native ? "1" : "0", 1);
#endif
        };
        struct ArmSave {
            std::string grp, nat;
            ArmSave() {
                const char* g = std::getenv("STRATA_EXL3_NATIVE_GROUPED");
                const char* n = std::getenv("STRATA_EXL3_RDNA_NATIVE");
                if (g) grp = g;
                if (n) nat = n;
            }
            ~ArmSave() {
#if defined(_WIN32)
                if (grp.empty()) _putenv_s("STRATA_EXL3_NATIVE_GROUPED", "0");
                else _putenv_s("STRATA_EXL3_NATIVE_GROUPED", grp.c_str());
                if (nat.empty()) _putenv_s("STRATA_EXL3_RDNA_NATIVE", "0");
                else _putenv_s("STRATA_EXL3_RDNA_NATIVE", nat.c_str());
#else
                if (grp.empty()) unsetenv("STRATA_EXL3_NATIVE_GROUPED");
                else setenv("STRATA_EXL3_NATIVE_GROUPED", grp.c_str(), 1);
                if (nat.empty()) unsetenv("STRATA_EXL3_RDNA_NATIVE");
                else setenv("STRATA_EXL3_RDNA_NATIVE", nat.c_str(), 1);
#endif
            }
        } arm_save;
        (void) arm_save;
        constexpr int H = 2560, FF = 640;
        // one trio per rate, plus the mixed-projection trio (gate K3 / up K4 / down K3)
        Matrix k2g(2,2,rng,H,FF),   k2u(2,2,rng,H,FF),   k2d(2,2,rng,FF,H);
        Matrix f25g(2,2,rng,H,FF,1), f25u(2,2,rng,H,FF,1), f25d(2,2,rng,FF,H,1);
        Matrix k3g(3,2,rng,H,FF),   k3u(3,2,rng,H,FF),   k3d(3,2,rng,FF,H);
        Matrix k4g(4,2,rng,H,FF),   k4u(4,2,rng,H,FF),   k4d(4,2,rng,FF,H);
        const int tokens = 23;
        std::vector<uint16_t> hx((size_t) tokens * H);
        for (auto& v : hx) v = half(((int) (rng() % 2000) - 1000) * 0.001f);
        cudaStream_t sa, sb;
        check(cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking));
        check(cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking));
        // one battery over a group plan the caller supplies; returns the CPU reference
        auto battery = [&](const char* what, const std::vector<Matrix*>& tg,
                           const std::vector<Matrix*>& tu, const std::vector<Matrix*>& td,
                           const std::vector<int>& widths) {
            const int ng = (int) widths.size();
            std::vector<int> first, grows;
            int routes = 0;
            for (int s : widths) { first.push_back(routes); grows.push_back(s); routes += s; }
            std::vector<Exl3Mat> gg(ng), gu(ng), gd(ng);
            for (int g = 0; g < ng; ++g) {
                gg[(size_t)g] = tg[(size_t)g]->gpu; gu[(size_t)g] = tu[(size_t)g]->gpu;
                gd[(size_t)g] = td[(size_t)g]->gpu;
            }
            std::vector<int> rows(routes);
            for (int r = 0; r < routes; ++r) rows[(size_t) r] = (r * 7 + 5) % tokens;   // permuted gather
            std::vector<double> ref((size_t) routes * H);
            for (int g = 0; g < ng; ++g)
                for (int r = 0; r < widths[(size_t) g]; ++r) {
                    const int at = first[(size_t) g] + r;
                    auto gv = tg[(size_t)g]->project(hx.data() + (size_t) rows[(size_t) at] * H);
                    const auto uv = tu[(size_t)g]->project(hx.data() + (size_t) rows[(size_t) at] * H);
                    for (int i = 0; i < FF; ++i) {
                        const float v = fp32(gv[(size_t) i]);
                        gv[(size_t) i] = half(v / (1 + std::exp(-v)) * fp32(uv[(size_t) i]));
                    }
                    const auto y = td[(size_t)g]->project(gv.data());
                    for (int i = 0; i < H; ++i) ref[(size_t) at * H + i] = fp32(y[(size_t) i]);
                }
            Buffer<uint16_t> dx(hx.size());
            check(cudaMemcpy(dx.p, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
            Buffer<int> drows(routes);
            check(cudaMemcpy(drows.p, rows.data(), routes * 4, cudaMemcpyHostToDevice));
            Buffer<uint16_t> dy((size_t) routes * H);
            Exl3MoeWorkspace ws_a, ws_b;
            ws_a.reserve(routes, H, FF, H, true);
            ws_a.reserve_small_m(routes, ng, H, FF, H);
            ws_a.reserve_runtime_scratch(routes, H, FF, H);
            ws_b.reserve(routes, H, FF, H, true);
            ws_b.reserve_small_m(routes, ng, H, FF, H);
            ws_b.reserve_runtime_scratch(routes, H, FF, H);
            auto run = [&](bool native, bool fused, uint16_t* out, cudaStream_t s, Exl3MoeWorkspace& w) {
                set_arm(native);
                exl3_moe_rows_grouped(gg.data(), gu.data(), gd.data(), first.data(), grows.data(),
                                      ng, routes, dx.p, drows.p, out, s, w, fused);
                const cudaError_t e = cudaGetLastError();
                if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            };
            auto fetch = [&](uint16_t* out, cudaStream_t s) {
                std::vector<uint16_t> y((size_t) routes * H);
                check(cudaStreamSynchronize(s));
                check(cudaMemcpy(y.data(), out, y.size() * 2, cudaMemcpyDeviceToHost));
                return y;
            };
            auto rel_of = [&](const std::vector<uint16_t>& y) {
                double num = 0, den = 0;
                for (size_t i = 0; i < y.size(); ++i) {
                    const double delta = fp32(y[i]) - ref[i];
                    num += delta * delta; den += ref[i] * ref[i];
                }
                return std::sqrt(num / (den + 1e-30));
            };
            bool pass = true;
            std::vector<uint16_t> base_u, nat_u;
            {
                run(false, false, dy.p, sa, ws_a);
                base_u = fetch(dy.p, sa);
                run(true, false, dy.p, sa, ws_a);
                nat_u = fetch(dy.p, sa);
                const double rb = rel_of(base_u), rn = rel_of(nat_u);
                bool fin = true;
                for (uint16_t v : nat_u) if (!std::isfinite(__half2float(__ushort_as_half(v)))) fin = false;
                const bool p = std::isfinite(rn) && rn < 0.02 && fin;
                pass = pass && p;
                size_t diff = 0;
                for (size_t i = 0; i < nat_u.size(); ++i) diff += nat_u[i] != base_u[i];
                std::printf("device=%d native_grouped_%s unfused rows=%d rel_base=%.5g rel_native=%.5g "
                            "bytes_diff_frac=%.4f %s\n", device, what, routes, rb, rn,
                            (double) diff / nat_u.size(), p ? "PASS" : "FAIL");
            }
            const bool fused_supported = strata::kernels::exl3_fused_multirow_supported();
            std::vector<uint16_t> nat_f;
            if (fused_supported) {
                std::vector<uint16_t> base_f;
                run(false, true, dy.p, sa, ws_a);
                base_f = fetch(dy.p, sa);
                run(true, true, dy.p, sa, ws_a);
                nat_f = fetch(dy.p, sa);
                const double rb = rel_of(base_f), rn = rel_of(nat_f);
                bool fin = true;
                for (uint16_t v : nat_f) if (!std::isfinite(__half2float(__ushort_as_half(v)))) fin = false;
                const bool p = std::isfinite(rn) && rn < 0.02 && fin;
                pass = pass && p;
                std::printf("device=%d native_grouped_%s fused rows=%d rel_base=%.5g rel_native=%.5g %s\n",
                            device, what, routes, rb, rn, p ? "PASS" : "FAIL");
                // captured with the flag ON, replayed with it OFF: replays keep the dispatch
                check(cudaStreamSynchronize(sa));
                cudaGraph_t graph; cudaGraphExec_t exec;
                check(cudaStreamBeginCapture(sa, cudaStreamCaptureModeThreadLocal));
                set_arm(true);
                exl3_moe_rows_grouped(gg.data(), gu.data(), gd.data(), first.data(), grows.data(),
                                      ng, routes, dx.p, drows.p, dy.p, sa, ws_a, true);
                check(cudaStreamEndCapture(sa, &graph));
                check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
                set_arm(false);
                bool graph_ok = true;
                for (int rep = 0; rep < 3; ++rep) {
                    check(cudaGraphLaunch(exec, sa));
                    const auto got = fetch(dy.p, sa);
                    graph_ok = graph_ok && got == nat_f;
                }
                check(cudaGraphExecDestroy(exec));
                check(cudaGraphDestroy(graph));
                pass = pass && graph_ok;
                std::printf("device=%d native_grouped_%s graph_replay flag_off_replays_stay_native=%d %s\n",
                            device, what, (int) graph_ok, graph_ok ? "PASS" : "FAIL");
                // two streams/workspaces, interleaved, byte-stable per arm
                Buffer<uint16_t> dy2((size_t) routes * H);
                bool stable = true;
                for (int rep = 0; rep < 4; ++rep) {
                    run(false, true, dy.p, sa, ws_a);
                    run(true, true, dy2.p, sb, ws_b);
                    const auto a = fetch(dy.p, sa), b = fetch(dy2.p, sb);
                    stable = stable && a == base_f && b == nat_f;
                }
                pass = pass && stable;
                std::printf("device=%d native_grouped_%s two_streams_byte_stable=%d %s\n",
                            device, what, (int) stable, stable ? "PASS" : "FAIL");
            } else {
                std::printf("device=%d native_grouped_%s fused SKIP (needs RDNA=1 FUSE=3)\n", device, what);
            }
            return pass;
        };
        const std::vector<int> per_rate_widths = {2, 1, 2, 2, 1, 2, 1, 2};
        ok = ok && battery("K3", {&k3g, &k3g, &k3g, &k3g, &k3g, &k3g, &k3g, &k3g},
                           {&k3u, &k3u, &k3u, &k3u, &k3u, &k3u, &k3u, &k3u},
                           {&k3d, &k3d, &k3d, &k3d, &k3d, &k3d, &k3d, &k3d}, per_rate_widths);
        ok = ok && battery("K4", {&k4g, &k4g, &k4g, &k4g, &k4g, &k4g, &k4g, &k4g},
                           {&k4u, &k4u, &k4u, &k4u, &k4u, &k4u, &k4u, &k4u},
                           {&k4d, &k4d, &k4d, &k4d, &k4d, &k4d, &k4d, &k4d}, per_rate_widths);
        // the mixed-rate window: K2, K2.5, K3, mixed-projection (K3 gate / K4 up / K3 down)
        // and K4 groups in ONE launch, widths 2/3/4/1/2 (ragged 1-row tail included)
        ok = ok && battery("mixed_K2_K2.5_K3_K4",
                           {&k2g, &f25g, &k3g, &k3g, &k4g},
                           {&k2u, &f25u, &k3u, &k4u, &k4u},
                           {&k2d, &f25d, &k3d, &k3d, &k4d}, {2, 3, 4, 1, 2});
        // poisoned-token nonfinite classification on the mixed-rate window: every route
        // reading the poisoned token must be nonfinite on both arms, every other row finite
        {
            const int ng = 5;
            const int widths[ng] = {2, 3, 4, 1, 2};
            std::vector<int> first, grows;
            int routes = 0;
            for (int s : widths) { first.push_back(routes); grows.push_back(s); routes += s; }
            Matrix* tg[ng] = {&k2g, &f25g, &k3g, &k3g, &k4g};
            Matrix* tu[ng] = {&k2u, &f25u, &k3u, &k4u, &k4u};
            Matrix* td[ng] = {&k2d, &f25d, &k3d, &k3d, &k4d};
            std::vector<Exl3Mat> gg(ng), gu(ng), gd(ng);
            for (int g = 0; g < ng; ++g) {
                gg[(size_t)g] = tg[g]->gpu; gu[(size_t)g] = tu[g]->gpu; gd[(size_t)g] = td[g]->gpu;
            }
            const int poison = 4;
            auto host = hx;
            const uint16_t inf16 = __half_as_ushort(__float2half(1.0f / 0.0f));
            for (int i = 0; i < H; ++i) host[(size_t) poison * H + i] = inf16;
            std::vector<int> rows(routes);
            for (int r = 0; r < routes; ++r) rows[(size_t) r] = (r * 7 + 5) % tokens;
            Buffer<uint16_t> dx(host.size());
            check(cudaMemcpy(dx.p, host.data(), host.size() * 2, cudaMemcpyHostToDevice));
            Buffer<int> drows(routes);
            check(cudaMemcpy(drows.p, rows.data(), routes * 4, cudaMemcpyHostToDevice));
            Buffer<uint16_t> dy((size_t) routes * H);
            Exl3MoeWorkspace ws_p;
            ws_p.reserve(routes, H, FF, H, true);
            ws_p.reserve_small_m(routes, ng, H, FF, H);
            ws_p.reserve_runtime_scratch(routes, H, FF, H);
            auto classify = [&](bool native) {
                set_arm(native);
                exl3_moe_rows_grouped(gg.data(), gu.data(), gd.data(), first.data(), grows.data(),
                                      ng, routes, dx.p, drows.p, dy.p, sa, ws_p);
                const cudaError_t e = cudaGetLastError();
                if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
                check(cudaStreamSynchronize(sa));
                std::vector<uint16_t> y((size_t) routes * H);
                check(cudaMemcpy(y.data(), dy.p, y.size() * 2, cudaMemcpyDeviceToHost));
                bool good = true;
                for (int r = 0; r < routes; ++r) {
                    const bool poisoned = rows[(size_t) r] == poison;
                    for (int i = 0; i < H; ++i) {
                        const bool fin = std::isfinite(__half2float(__ushort_as_half(y[(size_t) r * H + i])));
                        if (poisoned && fin) { good = false; break; }
                        if (!poisoned && !fin) { good = false; break; }
                    }
                    if (!good) break;
                }
                return good;
            };
            const bool good_off = classify(false), good_on = classify(true);
            ok = ok && good_off && good_on;
            std::printf("device=%d native_grouped_mixed nonfinite established=%d native=%d %s\n",
                        device, (int) good_off, (int) good_on,
                        good_off && good_on ? "PASS" : "FAIL");
        }
        check(cudaStreamDestroy(sa));
        check(cudaStreamDestroy(sb));
    }
    // The gfx11 matrix-core group FFN on the real geometry, every bitrate the arm covers (K2..K6
    // plus the half-integer K2.5 - MiMo's expert mix is exactly K2/K2.5; the 4.05bpw model's
    // experts are K4), at group counts around the dispatch threshold and the kernel's internal
    // boundaries (row_tiles switches to 2 at 24; the 16-row A-tiles pad at every multiple).  The
    // routes scatter over MORE token rows than the group has (a prefill chunk's shape): gate/up
    // index the token buffer, the down projection reads the FFN's own route-major slab, and the
    // two must not be confused.  SKIP (not FAIL) where the device cannot take the path.
    for (auto [bits, frac] : std::initializer_list<std::pair<int,int>>{
             {2,0},{2,1},{3,0},{4,0},{5,0},{6,0}}) {
        Matrix wg(bits,2,rng,width,hidden,frac), wu(bits,2,rng,width,hidden,frac), wd(bits,2,rng,hidden,width,frac);
        const char* kn = frac ? "2.5" : nullptr;
        char kname[8]; if (!kn) std::snprintf(kname,sizeof kname,"%d",bits);
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
                std::printf("device=%d geometry=2560/640 K%s wmma_group rows=%d unavailable-on-this-device SKIP\n",
                            device,kn?kn:kname,rows);
                continue;
            }
            std::vector<float> wy((size_t) rows*width);
            check(cudaMemcpy(wy.data(),dout.p,wy.size()*4,cudaMemcpyDeviceToHost));
            double num=0,den=0;
            for (size_t i=0;i<wy.size();++i) { const double delta=wy[i]-wref[i]; num+=delta*delta; den+=wref[i]*wref[i]; }
            const double rel=std::sqrt(num/(den+1e-30));
            const bool pass=std::isfinite(rel)&&rel<0.02;
            ok=ok&&pass;
            std::printf("device=%d geometry=2560/640 K%s wmma_group rows=%d scattered relative_error=%.5g %s\n",
                        device,kn?kn:kname,rows,rel,pass?"PASS":"FAIL");
            // STRATA_EXL3_WMMA_STG_PAD A/B on the same fixture: the padded instantiation must
            // produce the SAME BYTES on every shape this arm covers (layout-only change; the
            // bank-hypothesis comment lives at the kernel).  In-process toggle via the setter,
            // both arms share weights/inputs/workspace.  wy already holds the unpadded bytes.
            exl3_wmma_stg_pad_set(true);
            if (exl3_moe_group_wmma(wg.gpu,wu.gpu,wd.gpu,rows,dwx.p,dr.p,dout.p,nullptr,workspace)) {
                std::vector<float> wy2((size_t) rows*width);
                check(cudaMemcpy(wy2.data(),dout.p,wy2.size()*4,cudaMemcpyDeviceToHost));
                const bool pad_equal = wy2 == wy;
                ok = ok && pad_equal;
                std::printf("device=%d geometry=2560/640 K%s wmma_group rows=%d stg_pad bytes_equal=%d %s\n",
                            device,kn?kn:kname,rows,(int)pad_equal,pad_equal?"PASS":"FAIL");
            }
            exl3_wmma_stg_pad_set(false);
        }
    }
    // Multi-expert WMMA subgroup: independent signed weights/scales per projection and expert,
    // unequal groups, nonidentity input and output maps, K2/K2.5/K3..K6 against the CPU
    // reconstruction - plus the mixed-K refusals (K2+K3 and K2+K2.5 must each split, not batch).
    for (auto [bits, frac] : std::initializer_list<std::pair<int,int>>{
             {2,0},{2,1},{3,0},{4,0},{5,0},{6,0}}) {
        Matrix g0(bits,2,rng,N,N,frac),u0(bits,2,rng,N,N,frac),d0(bits,2,rng,N,N,frac);
        Matrix g1(bits,2,rng,N,N,frac),u1(bits,2,rng,N,N,frac),d1(bits,2,rng,N,N,frac);
        const char* kn = frac ? "2.5" : nullptr;
        char kname[8]; if (!kn) std::snprintf(kname,sizeof kname,"%d",bits);
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
                    std::printf("device=%d expert_batch supported K%s unexpectedly rejected FAIL\n", device, kn?kn:kname);
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
            ok=ok&&pass;std::printf("device=%d expert_batch K%s groups=%d/%d CPU relative_error=%.5g %s\n",device,kn?kn:kname,first_rows,second_rows,rel,pass?"PASS":"FAIL");
            auto bad=groups[1]; groups[1].gate.bits=bits==3?4:3;
            ok=ok&&!exl3_moe_expert_batch(groups,2,dx.p,dr.p,dd.p,dy.p,nullptr,workspace);
            groups[1]=bad;groups[1].gate.cb=1;
            ok=ok&&!exl3_moe_expert_batch(groups,2,dx.p,dr.p,dd.p,dy.p,nullptr,workspace);
            groups[1]=bad;groups[1].rows=7;
            ok=ok&&!exl3_moe_expert_batch(groups,2,dx.p,dr.p,dd.p,dy.p,nullptr,workspace);
            // a (bits, frac) mix inside one batch must refuse: same KA, different half
            groups[1]=bad;
            if (frac == 0) groups[1].gate.frac = 1; else groups[1].gate.frac = 0;
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
    return recon_batch_parity(device)&&real_geometry(device)&&a&&b;
}
}
// STRATA_EXL3_WMMA_STG_PAD flag semantics.  The env applies exactly ONCE per process, at the
// first exl3_wmma_stg_pad_enabled()/exl3_wmma_stg_pad_set() call (both funnel through the same
// apply-once helper, so even a setter called first cannot be overridden later); afterwards the
// setter is authoritative in BOTH directions.  The env=1-at-first-read proof needs a fresh
// process (the once-flag is spent after the first call), so it is driven by EXTERNAL invocation
// - `--stg-pad-flag-test` (ctest runs it as exl3_moe_multi_parity_stgpad_env with the env set
// via cmake -E env; no process spawning happens inside this binary, and this mode touches no
// GPU API at all).  The in-process checks below use only std::getenv + the setter, which exist
// on every supported platform.
static int stg_pad_flag_checks(bool env_expected)
{
    bool ok = exl3_wmma_stg_pad_enabled() == env_expected;   // first read applies the env once
    exl3_wmma_stg_pad_set(!env_expected);
    ok = ok && exl3_wmma_stg_pad_enabled() == !env_expected; // setter overrides downwards
    exl3_wmma_stg_pad_set(env_expected);
    ok = ok && exl3_wmma_stg_pad_enabled() == env_expected;  // ...and back up
    std::printf("stg_pad flag: env %s at first read, then setter authoritative both ways: %s\n",
                env_expected ? "ON" : "unset", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    // No GPU API is touched before this point: the flag mode must stay usable with the GPU
    // runtime completely uninitialized (and the flag machinery exists on every backend).
    if (argc > 1 && std::strcmp(argv[1], "--stg-pad-flag-test") == 0) {
        const bool env_on = std::getenv("STRATA_EXL3_WMMA_STG_PAD") != nullptr &&
                            std::getenv("STRATA_EXL3_WMMA_STG_PAD")[0] != '0';
        return stg_pad_flag_checks(env_on);
    }
    {
        const char* preset = std::getenv("STRATA_EXL3_WMMA_STG_PAD");
        const bool env_was_set = preset != nullptr && preset[0] != '0';
        const int rc = stg_pad_flag_checks(env_was_set);   // default invocation: env normally unset
        if (rc != 0) return rc;
        // The setter-before-first-getter ordering (the old documented trap, now fixed): with the
        // env unset a leading set(true) must survive the env apply triggered by the first read.
        if (!env_was_set) {
            exl3_wmma_stg_pad_set(false);
            exl3_wmma_stg_pad_set(true);                   // BEFORE any getter call
            bool ok = exl3_wmma_stg_pad_enabled() == true;
            exl3_wmma_stg_pad_set(false);                  // leave off for the parity battery
            std::printf("stg_pad flag: setter before first read not overridden by env apply: %s\n",
                        ok ? "PASS" : "FAIL");
            if (!ok) return 1;
        }
    }
    int count=0;
    if (cudaGetDeviceCount(&count)!=cudaSuccess || count==0) { std::puts("SKIP: no CUDA device"); return 77; }
    try { bool ok=true; for(int d=0;d<count;++d) ok=run_device(d)&&ok; return ok?0:1; }
    catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
