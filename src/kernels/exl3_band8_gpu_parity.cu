// Isolated prototype: instantiate the actual production kernels, inspect float
// partials, and use the independent portable CPU codec as the arithmetic oracle.
#include "cuda/exl3.cu"
#include "strata/kernels/cpu/exl3.hpp"
#include <vector>
#include <random>
#include <cstring>
#include <cmath>
#include <string>
#if defined(STRATA_WMMA_GFX11)
#include <unistd.h>
#include <sys/wait.h>
#endif

using namespace strata::kernels;
using strata::Exl3TrellisLayout;
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool x, const char* s) { if (!x) throw std::runtime_error(s); }
uint16_t half(float x) { return __half_as_ushort(__float2half_rn(x)); }
float real(uint16_t x) { return __half2float(__ushort_as_half(x)); }
size_t live = 0, peak = 0, comparisons = 0;
template<class T> struct Buf {
    T* p = nullptr; size_t count;
    explicit Buf(size_t n) : count(n) { ck(cudaMalloc((void**)&p,n*sizeof(T))); live+=n*sizeof(T); peak=std::max(peak,live); }
    ~Buf() { if(p) { (void)cudaFree(p); live-=count*sizeof(T); } }
    void put(const std::vector<T>& v) { require(v.size()<=count,"upload capacity"); ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); }
    std::vector<T> get(size_t n=0) const { if(!n)n=count; std::vector<T> v(n); ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return v; }
    Buf(const Buf&)=delete;
};
// Exact CPU nonplanar helper mapping; reject before touching either span.
bool pack(const std::vector<uint16_t>& src, std::vector<uint16_t>& dst, int ki,int nj,int p) {
    if(ki<=0 || nj<=0 || ki%8 || nj%8 || src.size()!=(size_t)ki*nj*p) return false;
    dst.resize(src.size());
    for(int kt=0;kt<ki;++kt) for(int nt=0;nt<nj;++nt)
        std::memcpy(dst.data()+(((size_t)(nt/8)*ki+kt)*8+nt%8)*p,
                    src.data()+((size_t)kt*nj+nt)*p,p*2);
    return true;
}
void permutation() {
    size_t words=0;
    for(int p:{32,40,48,64}) for(int nj:{8,40,128,256}) {
        const int ki=8; std::vector<uint16_t> a((size_t)ki*nj*p),b,inv(a.size());
        for(size_t i=0;i<a.size();++i)a[i]=(uint16_t)i;
        require(pack(a,b,ki,nj,p),"pack");
        // Independent inverse: linear destination tile -> band,k,lane -> native.
        std::vector<unsigned char> seen((size_t)ki*nj);
        for(size_t d=0;d<b.size()/p;++d) {
            const size_t band=d/(ki*8), kt=(d/8)%ki, nt=band*8+d%8;
            const size_t s=kt*nj+nt;
            require(!seen[s]++,"inverse collision");
            for(int w=0;w<p;++w) { require(b[d*p+w]==a[s*p+w],"golden word"); inv[s*p+w]=b[d*p+w]; }
        }
        require(a==inv && a.size()==b.size(),"inverse bytes"); words+=a.size();
    }
    std::vector<uint16_t> empty,out{123};
    require(!pack(empty,out,8,9,32) && out==std::vector<uint16_t>{123},"reject before read");
    require(sizeof(Exl3Mat)==48,"Exl3Mat 64-bit ABI size changed");   // layout rides the old tail pad
    std::printf("host permutation words=%zu inverse/golden bitdiff=0 PASS\n",words);
}
struct Mat {
    int k,n,b,f; std::vector<uint16_t> native,packed,su,sv,weight;
    Buf<uint16_t> dn,db,dsu,dsv;
    Exl3Mat a,z;
    Mat(int k_,int n_,int b_,int f_,std::mt19937& rng):k(k_),n(n_),b(b_),f(f_),
        native((size_t)(k/16)*(n/16)*(16*b+8*f)),su(k),sv(n),
        dn(native.size()),db(native.size()),dsu(k),dsv(n) {
        for(auto& v:native)v=(uint16_t)rng();
        for(auto& v:su)v=half((0.10f+(rng()%32)*0.001f)*(rng()%2?1:-1));
        for(auto& v:sv)v=half((0.10f+(rng()%32)*0.001f)*(rng()%2?1:-1));
        require(pack(native,packed,k/16,n/16,16*b+8*f),"matrix pack");
        dn.put(native);db.put(packed);dsu.put(su);dsv.put(sv);
        // a: native bytes + Native tag (the default); z: the SAME scales/geometry over the packed
        // bytes, carrying the Band8 tag the metadata contract requires - the tag is what every
        // dispatch site reads, never the pointer.
        a={dn.p,dsu.p,dsv.p,k/16,n/16,b,2,f};z=a;z.trellis=db.p;z.layout=Exl3TrellisLayout::Band8;
    }
    void oracle() { if(weight.empty()) { weight.resize((size_t)k*n); cpu::exl3_reconstruct_weight(native.data(),k/16,n/16,b,cpu::Exl3Codebook::Mul1,su.data(),sv.data(),weight.data(),f); } }
    std::vector<uint16_t> project(const uint16_t* x) {
        oracle();std::vector<uint16_t> y(n);
        for(int j=0;j<n;++j) { double s=0;for(int i=0;i<k;++i)s+=(double)real(x[i])*real(weight[(size_t)i*n+j]);y[j]=half((float)s); }
        return y;
    }
};
template<class T> void equal(const std::vector<T>& a,const std::vector<T>& b,const char* name) {
    require(a.size()==b.size(),"compare sizes");size_t diff=0;
    for(size_t i=0;i<a.size();++i) {
        if constexpr(std::is_same_v<T,float>) require(std::isfinite(a[i]) && std::isfinite(b[i]),"nonfinite float");
        diff+=std::memcmp(&a[i],&b[i],sizeof(T))!=0;
    }
    comparisons+=a.size();
    if(diff) { std::printf("%s elements=%zu bitdiff=%zu FAIL\n",name,a.size(),diff);throw std::runtime_error("bitwise parity"); }
}
#if defined(STRATA_WMMA_GFX11)
template<int B,bool F,int R> void raw(Mat& m,int parts) {
    const int rows=R;Buf<float> x((size_t)rows*m.k),ya((size_t)rows*parts*m.n),yb((size_t)rows*parts*m.n);
    std::vector<float> hx(x.count);for(size_t i=0;i<hx.size();++i)hx[i]=((int)(i%61)-30)*0.0001f;x.put(hx);
    Buf<Exl3Mat> ma(1),mb(1);ma.put({m.a});mb.put({m.z});Buf<int> first(1),nr(1);first.put({0});nr.put({rows});
    const size_t smem=native_grouped_widest(m.k/16,parts)*R*2;
    rdna_native::gemv_grouped<F,R,false,false,B,false><<<dim3(m.n/128,parts,1),256,smem>>>(x.p,ma.p,first.p,nr.p,ya.p);
    rdna_native::gemv_grouped<F,R,false,false,B,true><<<dim3(m.n/128,parts,1),256,smem>>>(x.p,mb.p,first.p,nr.p,yb.p);
    ck(cudaGetLastError());equal(ya.get(),yb.get(),"grouped raw floats");
    if constexpr(B==2 && R==1) {
        const size_t shared=((m.k/128+parts-1)/parts)*128*4;
        rdna_native::gemv<F,false><<<dim3(m.n/128,parts,1),256,shared>>>(x.p,ma.p,ya.p,parts);
        require(exl3_experimental_band8_indexed_dot(&m.z,mb.p,1,x.p,x.count,yb.p,yb.count,parts,nullptr),"indexed API refused");
        ck(cudaGetLastError());equal(ya.get(),yb.get(),"indexed raw floats");
    }
    if(parts==1) {
        hipFuncAttributes a{},b{};
        ck(hipFuncGetAttributes(&a,(const void*)rdna_native::gemv_grouped<F,R,false,false,B,false>));
        ck(hipFuncGetAttributes(&b,(const void*)rdna_native::gemv_grouped<F,R,false,false,B,true>));
        std::printf("RESOURCES grouped K=%d frac=%d rows=%d native/band8 regs=%d/%d static_smem=%zu/%zu local=%zu/%zu\n",
            B,F,R,a.numRegs,b.numRegs,a.sharedSizeBytes,b.sharedSizeBytes,a.localSizeBytes,b.localSizeBytes);
        ck(hipFuncGetAttributes(&a,(const void*)rdna_native::gemv_grouped<F,R,true,false,B,false>));
        ck(hipFuncGetAttributes(&b,(const void*)rdna_native::gemv_grouped<F,R,true,false,B,true>));
        std::printf("RESOURCES fused K=%d frac=%d rows=%d native/band8 regs=%d/%d static_smem=%zu/%zu local=%zu/%zu\n",
            B,F,R,a.numRegs,b.numRegs,a.sharedSizeBytes,b.sharedSizeBytes,a.localSizeBytes,b.localSizeBytes);
    }
}
void raw_all(Mat& m,int parts) {
    auto launch=[&](auto b,auto f){constexpr int B=decltype(b)::value;constexpr bool F=decltype(f)::value;
        raw<B,F,1>(m,parts);raw<B,F,2>(m,parts);raw<B,F,4>(m,parts);};
    if(m.f)launch(std::integral_constant<int,2>{},std::true_type{});
    else if(m.b==2)launch(std::integral_constant<int,2>{},std::false_type{});
    else if(m.b==3)launch(std::integral_constant<int,3>{},std::false_type{});
    else launch(std::integral_constant<int,4>{},std::false_type{});
}
std::vector<float> fetch(float* p,size_t n) {std::vector<float> a(n);ck(cudaMemcpy(a.data(),p,n*4,cudaMemcpyDeviceToHost));return a;}
std::vector<uint16_t> fetch16(uint16_t* p,size_t n) {std::vector<uint16_t> a(n);ck(cudaMemcpy(a.data(),p,n*2,cudaMemcpyDeviceToHost));return a;}
void ffn(Mat& g,Mat& u,Mat& d,int width,bool fused,int parts,bool oracle,bool swap_groups=false) {
    const int k=g.k,ff=g.n,n=d.n;
    const std::vector<int> first{0,width},rows{width,std::max(1,width-1)};
    const int routes=rows[0]+rows[1];
    std::vector<Exl3Mat> ga(2,g.a),ua(2,u.a),da(2,d.a),gb(2,g.z),ub(2,u.z),db(2,d.z);
    if(swap_groups) {ga[1]=u.a;ua[1]=g.a;gb[1]=u.z;ub[1]=g.z;}
    std::vector<uint16_t> hx((size_t)7*k);for(size_t i=0;i<hx.size();++i)hx[i]=half(((int)(i%97)-48)*0.001f);
    std::vector<int> hmap(routes);for(int i=0;i<routes;++i)hmap[i]=(routes-i+2)%7;
    Buf<uint16_t> x(hx.size()),out((size_t)routes*n);x.put(hx);Buf<int> map(routes);map.put(hmap);
    Exl3MoeWorkspace w;w.reserve_small_m(routes,2,std::max(k,4096),std::max(ff,4096),n);
    const size_t before=w.allocated_bytes();
    auto& wi=w.implementation();
    auto upload=[&](const std::vector<Exl3Mat>& a,const std::vector<Exl3Mat>& b,const std::vector<Exl3Mat>& c){
        ck(cudaMemcpy(wi.smallm_gate.p,a.data(),2*sizeof(Exl3Mat),cudaMemcpyHostToDevice));
        ck(cudaMemcpy(wi.smallm_up.p,b.data(),2*sizeof(Exl3Mat),cudaMemcpyHostToDevice));
        ck(cudaMemcpy(wi.smallm_down.p,c.data(),2*sizeof(Exl3Mat),cudaMemcpyHostToDevice));
        ck(cudaMemcpy(wi.smallm_first.p,first.data(),8,cudaMemcpyHostToDevice));ck(cudaMemcpy(wi.smallm_rows.p,rows.data(),8,cudaMemcpyHostToDevice));};
    auto rate=[](const Mat& m){return m.f?2:m.b==2?1:m.b==3?4:8;};
    upload(ga,ua,da);
    moe_run_experts_grouped_dev<>(wi.smallm_gate.p,wi.smallm_up.p,wi.smallm_down.p,wi.smallm_first.p,wi.smallm_rows.p,
        2,routes,width,x.p,nullptr,w,k,map.p,k,ff,n,2,fused,out.p,
        rate(g)|(swap_groups?rate(u):0),rate(u)|(swap_groups?rate(g):0),rate(d),parts,parts,parts);
    ck(cudaGetLastError());auto ya=out.get();auto pa=fetch(wi.partials.p,(size_t)routes*parts*n);
    auto gate=fetch16(wi.g.p,(size_t)routes*ff),up=fetch16(wi.u.p,(size_t)routes*ff);
    // Only packed descriptors are supplied to the candidate entry.
    require(exl3_experimental_band8_rows({gb.data(),ub.data(),db.data()},first.data(),rows.data(),2,routes,
        x.p,map.p,out.p,nullptr,w,fused,parts),"BAND8 FFN refused");
    ck(cudaGetLastError());auto yb=out.get();equal(ya,yb,"FFN halves");equal(pa,fetch(wi.partials.p,pa.size()),"FFN down float partials");
    equal(gate,fetch16(wi.g.p,gate.size()),"FFN gate halves");equal(up,fetch16(wi.u.p,up.size()),"FFN up halves");
    require(before==w.allocated_bytes(),"candidate allocation");
    size_t nz=0;for(auto v:yb){require(std::isfinite(real(v)),"nonfinite");nz+=real(v)!=0;}require(nz>0,"allzero FFN");
    double rel=0;
    if(oracle) {
        double num=0,den=0;
        for(int r=0;r<routes;++r){Mat& gm=swap_groups && r>=width?u:g;Mat& um=swap_groups && r>=width?g:u;
            auto gv=gm.project(hx.data()+(size_t)hmap[r]*k);auto uv=um.project(hx.data()+(size_t)hmap[r]*k);
            for(int i=0;i<ff;++i){float v=real(gv[i]);gv[i]=half(v/(1+std::exp(-v))*real(uv[i]));}
            auto ref=d.project(gv.data());for(int j=0;j<n;++j){double v=real(ref[j]),delta=real(yb[(size_t)r*n+j])-v;num+=delta*delta;den+=v*v;}}
        rel=std::sqrt(num/(den+1e-30));require(std::isfinite(rel)&&rel<0.02,"CPU oracle norm");
    }
    peak=std::max(peak,live+before);
    std::printf("FFN k=%d ff=%d rate=%d+%d/%d+%d/%d+%d width=%d tail=%d fused=%d parts=%d half=%zu float=%zu bitdiff=0 nonzero=%zu cpu_norm=%g checked=%d workspace=%zu PASS\n",
        k,ff,g.b,g.f,u.b,u.f,d.b,d.f,width,rows[1],fused,parts,yb.size(),pa.size(),nz,rel,oracle,before);
}

// ---- stage-2 prefill: the WMMA group FFN on band8 views (bitwise vs the native WMMA entry) ----
// Fixed inputs, both arms on the SAME workspace/x/token mapper; the candidate must not touch
// allocations (caller-pre-reserved) and must match the native WMMA output bit for bit.
void wmma_group(Mat& g, Mat& u, Mat& d, int rows, bool stg) {
    const int k=g.k, ff=g.n, n=d.n, tokens=29;
    std::vector<uint16_t> hx((size_t)tokens*k);
    for(size_t i=0;i<hx.size();++i)hx[i]=half(((int)(i%97)-48)*0.001f);
    std::vector<int> hmap(rows);
    for(int r=0;r<rows;++r)hmap[r]=(r*17+5)%tokens;   // token-mapper mutations: boundaries+duplicate
    if(rows>=4){hmap[0]=0;hmap[1]=tokens-1;hmap[2]=hmap[3];}
    Buf<uint16_t> x(hx.size());x.put(hx);
    Buf<int> map(rows);map.put(hmap);
    Buf<float> ya((size_t)rows*n),yb((size_t)rows*n);
    Exl3MoeWorkspace w;
    w.reserve(rows,k,ff,n);w.reserve_group_scratch(rows,k,ff,n,true);
    const bool had=exl3_wmma_stg_pad_enabled();
    exl3_wmma_stg_pad_set(stg);   // matched STG stride for BOTH arms
    require(exl3_moe_group_wmma(g.a,u.a,d.a,rows,x.p,map.p,ya.p,nullptr,w),"native WMMA refused");
    const size_t before=w.allocated_bytes();
    require(exl3_experimental_band8_group_wmma(g.z,u.z,d.z,rows,x.p,map.p,yb.p,nullptr,w),
        "band8 WMMA refused");
    ck(cudaGetLastError());
    exl3_wmma_stg_pad_set(had);
    auto va=ya.get(),vb=yb.get();
    equal(va,vb,"WMMA group FFN floats");
    size_t nz=0;for(float v:vb){require(std::isfinite(v),"nonfinite WMMA");nz+=v!=0;}require(nz>0,"allzero WMMA");
    require(before==w.allocated_bytes(),"candidate allocation");
    peak=std::max(peak,live+before);
    std::printf("WMMA k=%d ff=%d n=%d rate=%d+%d/%d+%d/%d+%d rows=%d stg=%d float=%zu bitdiff=0 nonzero=%zu workspace=%zu PASS\n",
        k,ff,n,g.b,g.f,u.b,u.f,d.b,d.f,rows,stg?9:8,vb.size(),nz,before);
}

// The batched (expert_batch) fragment reader in isolation: one uploaded group, native vs packed
// trellis pointer, the raw batched kernel launched both ways. Proves the Band8 BATCHED READER
// only - the full exl3_moe_expert_batch host API on band8 descriptors stays UNTESTED (a future
// consumer would add an explicit experimental wrapper; not this stage).
template <int B, int RT, int STG, bool B8>
void wmma_batched_launch(const uint16_t* x, const Exl3Group* g, float* y, int rows, int in, int out, int frac) {
    const dim3 grid(std::min((rows + 16 * RT - 1) / (16 * RT), 32), (out + 255) / 256, 1);
    if constexpr (STG)
        routed_wmma_kernel<B, RT, true, 9, B8><<<grid, 512>>>(x, nullptr, y, 0, in, out, g, 0, frac);
    else
        routed_wmma_kernel<B, RT, true, 8, B8><<<grid, 512>>>(x, nullptr, y, 0, in, out, g, 0, frac);
    ck(cudaGetLastError());
}
template <int B>
void wmma_batched(Mat& m, int rows, bool stg) {
    const int in=m.k, out=m.n;
    std::vector<uint16_t> hx((size_t)rows*in);
    for(size_t i=0;i<hx.size();++i)hx[i]=half(((int)(i%89)-44)*0.001f);
    Buf<uint16_t> x(hx.size());x.put(hx);
    Buf<float> ya((size_t)rows*out),yb((size_t)rows*out);
    Exl3Group gn{},gb8{};
    gn.gate=gn.up=gn.down=m.a;gn.first=0;gn.rows=rows;
    gb8=gn;gb8.gate=gb8.up=gb8.down=m.z;
    Buf<Exl3Group> an(1),ab(1);an.put({gn});ab.put({gb8});
    const bool rt2=rows>=24;
    if(!rt2&&!stg){wmma_batched_launch<B,1,0,false>(x.p,an.p,ya.p,rows,in,out,m.f);
                   wmma_batched_launch<B,1,0,true >(x.p,ab.p,yb.p,rows,in,out,m.f);}
    if(!rt2&&stg) {wmma_batched_launch<B,1,1,false>(x.p,an.p,ya.p,rows,in,out,m.f);
                   wmma_batched_launch<B,1,1,true >(x.p,ab.p,yb.p,rows,in,out,m.f);}
    if(rt2&&!stg) {wmma_batched_launch<B,2,0,false>(x.p,an.p,ya.p,rows,in,out,m.f);
                   wmma_batched_launch<B,2,0,true >(x.p,ab.p,yb.p,rows,in,out,m.f);}
    if(rt2&&stg)  {wmma_batched_launch<B,2,1,false>(x.p,an.p,ya.p,rows,in,out,m.f);
                   wmma_batched_launch<B,2,1,true >(x.p,ab.p,yb.p,rows,in,out,m.f);}
    equal(ya.get(),yb.get(),"WMMA batched reader floats");
    auto vb=yb.get();size_t nz=0;
    for(float v:vb){require(std::isfinite(v),"nonfinite batched");nz+=v!=0;}
    require(nz>0,"allzero batched");
    std::printf("WMMA BATCHED K=%d frac=%d rows=%d stg=%d float=%zu bitdiff=0 nonzero=%zu PASS\n",
        m.b,m.f,rows,stg?9:8,vb.size(),nz);
}
void wmma_batched_all(Mat& m, int rows, bool stg) {
    if(m.f||m.b==2)wmma_batched<2>(m,rows,stg);   // K2/K2.5 share the bits=2 kernel (frac runtime)
    else if(m.b==3)wmma_batched<3>(m,rows,stg);
    else wmma_batched<4>(m,rows,stg);
}

// The batched weight reconstruction (STRATA_EXL3_RECON_BATCH flow) on band8 views: native
// exl3_moe_recon_batch vs exl3_experimental_band8_recon_batch, bitwise, FP16-out flag at its
// OFF (F32-C) baseline. rates may differ PER EXPERT (the K2 + K2.5 mix).
void recon_batch_parity(Mat& g1, Mat& u1, Mat& d1, Mat& g2, Mat& u2, Mat& d2, int r1, int r2) {
    const int k=g1.k, ff=g1.n, n=d1.n;
    require(g1.k==g2.k&&g1.n==g2.n&&d1.n==d2.n,"recon geo");
    const int routes=r1+r2, m_rows=std::max(r1,r2), tokens=routes+3;
    std::vector<uint16_t> hx((size_t)tokens*k);
    for(size_t i=0;i<hx.size();++i)hx[i]=half(((int)(i%83)-41)*0.001f);
    std::vector<int> hmap(routes),hdest(routes);
    for(int r=0;r<routes;++r){hmap[r]=(r*13+2)%tokens;hdest[r]=r;}
    hmap[0]=0;hmap[1]=tokens-1;if(routes>2)hmap[2]=hmap[1];   // token-mapper mutations
    Buf<uint16_t> x(hx.size());x.put(hx);
    Buf<int> map(routes),dest(routes);map.put(hmap);dest.put(hdest);
    Buf<float> ya((size_t)routes*n),yb((size_t)routes*n);
    Exl3Group gn[2]={{g1.a,u1.a,d1.a,0,r1},{g2.a,u2.a,d2.a,r1,r2}};
    Exl3Group gb[2]={{g1.z,u1.z,d1.z,0,r1},{g2.z,u2.z,d2.z,r1,r2}};
    Exl3MoeWorkspace w;
    w.reserve_recon_batch(2,2*m_rows,k,ff,n);
    require(exl3_moe_recon_batch(gn,2,x.p,map.p,dest.p,ya.p,nullptr,w),"native recon batch refused");
    const size_t before=w.allocated_bytes();
    require(exl3_experimental_band8_recon_batch(gb,2,x.p,map.p,dest.p,yb.p,nullptr,w),
        "band8 recon batch refused");
    ck(cudaGetLastError());
    auto va=ya.get(),vb=yb.get();
    equal(va,vb,"recon batch floats");
    size_t nz=0;for(float v:vb){require(std::isfinite(v),"nonfinite recon");nz+=v!=0;}require(nz>0,"allzero recon");
    require(before==w.allocated_bytes(),"recon candidate allocation");
    peak=std::max(peak,live+before);
    std::printf("RECONB k=%d ff=%d n=%d rate1=%d+%d rate2=%d+%d rows=%d+%d float=%zu bitdiff=0 nonzero=%zu workspace=%zu PASS\n",
        k,ff,n,g1.b,g1.f,g2.b,g2.f,r1,r2,vb.size(),nz,before);
}

// ---- the NORMAL entries' .layout routing: the same public APIs over Native bytes vs the SAME
// weights/scales in Band8 order + tag.  Same process, same env, same arm selection - the only
// delta is the tile address - so every comparison below must be BITWISE zero.  Also proves the
// refusal contracts: mixed tags throw/return-false BEFORE anything enqueues, an untagged packed
// view is refused by the experimental entries and provably mis-decodes through the native
// reader, and the workspace layout mode drives the device-only masked path (incl. graph
// capture replay).
void normal_ffn_indexed(Mat& g, Mat& u, Mat& d, int routes, int tokens) {
    const int k = g.k, ff = g.n, n = d.n;
    std::vector<uint16_t> hx((size_t) tokens * k);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = half(((int) (i % 71) - 35) * 0.001f);
    std::vector<int> hmap(routes);
    for (int r = 0; r < routes; ++r) hmap[r] = (r * 7 + 3) % tokens;   // duplicate tokens, tail cases
    hmap[0] = 0; if (routes > 1) hmap[routes - 1] = tokens - 1; if (routes > 2) hmap[1] = hmap[0];
    Buf<uint16_t> x(hx.size()); x.put(hx);
    Buf<int> map(routes); map.put(hmap);
    Buf<uint16_t> ya((size_t) routes * n), yb((size_t) routes * n);
    std::vector<Exl3Mat> ga(routes, g.a), ua(routes, u.a), da(routes, d.a);
    std::vector<Exl3Mat> gb(routes, g.z), ub(routes, u.z), db(routes, d.z);
    Exl3MoeWorkspace wa, wb;
    exl3_moe_rows_indexed(ga.data(), ua.data(), da.data(), routes, x.p, map.p, ya.p, nullptr, wa);
    exl3_moe_rows_indexed(gb.data(), ub.data(), db.data(), routes, x.p, map.p, yb.p, nullptr, wb);
    ck(cudaGetLastError());
    equal(ya.get(), yb.get(), "indexed NORMAL API halves");
    // mixed tags: gate Native + up Band8 in one call must THROW before anything launches
    {   bool threw = false;
        try { exl3_moe_rows_indexed(gb.data(), ub.data(), db.data(), routes, x.p, map.p, yb.p, nullptr, wb);
              auto keep = gb[0]; gb[0].layout = Exl3TrellisLayout::Native;   // mixed within the call
              try { exl3_moe_rows_indexed(gb.data(), ub.data(), db.data(), routes, x.p, map.p, yb.p, nullptr, wb); }
              catch (const std::invalid_argument&) { threw = true; }
              gb[0] = keep;
              if (!threw) throw std::runtime_error("mixed layout tags accepted by indexed entry");
        } catch (const std::runtime_error& e) {
            if (std::string(e.what()) != std::string("mixed layout tags accepted by indexed entry")) throw;
        }
    }
    // an UNTAGGED packed view rides the Native reader and computes garbage: this is the caller
    // contract the tag exists for, and the check below proves the dispatch really reads .layout
    // (if the two layouts ever alias, this comparison goes equal and the test fails loudly).
    {   std::vector<Exl3Mat> p0(routes, g.z), p1(routes, u.z), p2(routes, d.z);
        for (auto& m : p0) m.layout = Exl3TrellisLayout::Native;
        for (auto& m : p1) m.layout = Exl3TrellisLayout::Native;
        for (auto& m : p2) m.layout = Exl3TrellisLayout::Native;
        Buf<uint16_t> ybad((size_t) routes * n);
        exl3_moe_rows_indexed(p0.data(), p1.data(), p2.data(), routes, x.p, map.p, ybad.p, nullptr, wa);
        ck(cudaGetLastError());
        auto correct = ya.get(), wrong = ybad.get();
        size_t diff = 0;
        for (size_t i = 0; i < correct.size(); ++i) diff += correct[i] != wrong[i];
        require(diff > 0, "untagged packed bytes produced the native answer (tag ignored?)");
        comparisons += correct.size();
    }
    std::printf("NORMAL indexed k=%d ff=%d n=%d rate=%d+%d routes=%d half=%zu bitdiff=0 untagged_differs PASS\n",
                k, ff, n, g.b, g.f, routes, (size_t) routes * n);
}
void normal_ffn_broadcast(Mat& g, Mat& u, Mat& d, int experts) {
    const int k = g.k, n = d.n;
    std::vector<uint16_t> hx((size_t) experts * k);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = half(((int) (i % 53) - 26) * 0.001f);
    Buf<uint16_t> x(hx.size()); x.put(hx);
    Buf<uint16_t> ya((size_t) experts * n), yb((size_t) experts * n);
    std::vector<Exl3Mat> ga(experts, g.a), ua(experts, u.a), da(experts, d.a);
    std::vector<Exl3Mat> gb(experts, g.z), ub(experts, u.z), db(experts, d.z);
    Exl3MoeWorkspace wa, wb;
    exl3_moe_rows_part(ga.data(), ua.data(), da.data(), experts, x.p, ya.p, nullptr, &wa);
    exl3_moe_rows_part(gb.data(), ub.data(), db.data(), experts, x.p, yb.p, nullptr, &wb);
    ck(cudaGetLastError());
    equal(ya.get(), yb.get(), "broadcast NORMAL API halves");
    std::printf("NORMAL broadcast k=%d n=%d rate=%d+%d experts=%d half=%zu bitdiff=0 PASS\n",
                k, n, g.b, g.f, experts, (size_t) experts * n);
}
void normal_grouped(Mat& g, Mat& u, Mat& d, int width, bool fused) {
    // one window: 24 groups - 23 at `width` rows plus a 1-row tail - so group widths mix
    std::vector<int> first, rows;
    int prefix = 0;
    for (int i = 0; i < 24; ++i) {
        first.push_back(prefix);
        const int r = i == 23 ? 1 : width;
        rows.push_back(r);
        prefix += r;
    }
    const int routes = prefix, k = g.k, ff = g.n, n = d.n, tokens = 9;
    std::vector<uint16_t> hx((size_t) tokens * k);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = half(((int) (i % 61) - 30) * 0.001f);
    std::vector<int> hmap(routes);
    for (int r = 0; r < routes; ++r) hmap[r] = (r * 5 + 1) % tokens;
    Buf<uint16_t> x(hx.size()); x.put(hx);
    Buf<int> map(routes); map.put(hmap);
    Buf<uint16_t> ya((size_t) routes * n), yb((size_t) routes * n);
    std::vector<Exl3Mat> ga(24, g.a), ua(24, u.a), da(24, d.a), gb(24, g.z), ub(24, u.z), db(24, d.z);
    Exl3MoeWorkspace wa, wb;
    exl3_moe_rows_grouped(ga.data(), ua.data(), da.data(), first.data(), rows.data(), 24, routes,
                          x.p, map.p, ya.p, nullptr, wa, fused);
    exl3_moe_rows_grouped(gb.data(), ub.data(), db.data(), first.data(), rows.data(), 24, routes,
                          x.p, map.p, yb.p, nullptr, wb, fused);
    ck(cudaGetLastError());
    equal(ya.get(), yb.get(), "grouped NORMAL API halves");
    // mixed tags across the window's groups throw before launch
    {   auto keep = gb[5]; gb[5].layout = Exl3TrellisLayout::Native;
        bool threw = false;
        try { exl3_moe_rows_grouped(gb.data(), ub.data(), db.data(), first.data(), rows.data(), 24, routes,
                                    x.p, map.p, yb.p, nullptr, wb, fused); }
        catch (const std::invalid_argument&) { threw = true; }
        gb[5] = keep;
        require(threw, "mixed grouped tags accepted");
    }
    std::printf("NORMAL grouped k=%d ff=%d n=%d rate=%d+%d width=%d routes=%d fused=%d half=%zu bitdiff=0 PASS\n",
                k, ff, n, g.b, g.f, width, routes, fused ? 1 : 0, (size_t) routes * n);
}
// The grouped API's route-table contract is 1..SMALL_M_MAX_ROWS(=6) rows per group - the
// host plan arrays are sized for 6.  Width 8 must be REFUSED on the host by BOTH layouts
// before any enqueue, workspace growth or output write - never a silent "unsupported"
// fallback to another arm (a fallback computes wrong rows from a 6-row window).
void normal_grouped_refuses_width_8(Mat& g, Mat& u, Mat& d) {
    const int first[1] = {0}, rows8[1] = {8}, routes = 8, k = g.k, n = d.n, tokens = 9;
    std::vector<uint16_t> hx((size_t) tokens * k);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = half(((int) (i % 61) - 30) * 0.001f);
    std::vector<int> hmap(routes, 0);                       // prefix-consistent table, rows over the cap
    Buf<uint16_t> x(hx.size()); x.put(hx);
    Buf<int> map(routes); map.put(hmap);
    Buf<uint16_t> y((size_t) routes * n);
    ck(cudaMemset(y.p, 0xA5, y.count * 2));                 // sentinel: a refused call leaves it alone
    const std::vector<Exl3Mat> ga{g.a}, ua{u.a}, da{d.a}, gb{g.z}, ub{u.z}, db{d.z};
    for (int arm = 0; arm < 2; ++arm) {
        Exl3MoeWorkspace w;
        const size_t before = w.allocated_bytes();          // existing counter: no new instrumentation
        bool threw = false;
        try {
            exl3_moe_rows_grouped(arm ? gb.data() : ga.data(), arm ? ub.data() : ua.data(),
                                  arm ? db.data() : da.data(), first, rows8, 1, routes,
                                  x.p, map.p, y.p, nullptr, w, false);
        } catch (const std::invalid_argument&) { threw = true; }   // route-table validation type
        require(threw, arm ? "band8 accepted an 8-row group" : "native accepted an 8-row group");
        require(w.allocated_bytes() == before, "refused width-8 call changed workspace allocations");
    }
    ck(cudaGetLastError());
    const auto out = y.get();
    for (uint16_t v : out) require(v == 0xA5A5, "refused width-8 call enqueued output work");
    std::printf("NORMAL grouped width=8 refused-before-enqueue native/band8=2 routes=8 sentinel=%zu unchanged PASS\n",
                out.size());
}
void normal_group_gemm(Mat& g, Mat& u, Mat& d, int rows) {
    const int k = g.k, ff = g.n, n = d.n, tokens = 7;
    std::vector<uint16_t> hx((size_t) tokens * k);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = half(((int) (i % 59) - 29) * 0.001f);
    std::vector<int> hmap(rows);
    for (int r = 0; r < rows; ++r) hmap[r] = (r * 3 + 2) % tokens;
    Buf<uint16_t> x(hx.size()); x.put(hx);
    Buf<int> map(rows); map.put(hmap);
    Buf<float> ya((size_t) rows * n), yb((size_t) rows * n);
    Exl3MoeWorkspace wa, wb;
    for (int fp16 : {0, 1}) {
        exl3_gemm_fp16_out_set(fp16 != 0);
        exl3_moe_group_gemm(g.a, u.a, d.a, rows, x.p, map.p, ya.p, nullptr, wa);
        exl3_moe_group_gemm(g.z, u.z, d.z, rows, x.p, map.p, yb.p, nullptr, wb);
        ck(cudaGetLastError());
        equal(ya.get(), yb.get(), "group GEMM NORMAL API floats");
        std::printf("NORMAL group_gemm k=%d ff=%d n=%d rate=%d+%d rows=%d fp16out=%d float=%zu bitdiff=0 PASS\n",
                    k, ff, n, g.b, g.f, rows, fp16, (size_t) rows * n);
    }
    exl3_gemm_fp16_out_set(false);
}
void normal_wmma(Mat& g, Mat& u, Mat& d, int rows) {
    const int k = g.k, ff = g.n, n = d.n, tokens = 11;
    std::vector<uint16_t> hx((size_t) tokens * k);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = half(((int) (i % 67) - 33) * 0.001f);
    std::vector<int> hmap(rows);
    for (int r = 0; r < rows; ++r) hmap[r] = (r * 9 + 4) % tokens;
    if (rows >= 4) { hmap[0] = 0; hmap[1] = tokens - 1; hmap[2] = hmap[1]; }
    Buf<uint16_t> x(hx.size()); x.put(hx);
    Buf<int> map(rows); map.put(hmap);
    Buf<float> ya((size_t) rows * n), yb((size_t) rows * n);
    Exl3MoeWorkspace wa, wb;
    require(exl3_moe_group_wmma(g.a, u.a, d.a, rows, x.p, map.p, ya.p, nullptr, wa), "native WMMA refused");
    require(exl3_moe_group_wmma(g.z, u.z, d.z, rows, x.p, map.p, yb.p, nullptr, wb), "band8 WMMA NORMAL route refused");
    ck(cudaGetLastError());
    equal(ya.get(), yb.get(), "WMMA NORMAL API floats");
    std::printf("NORMAL wmma k=%d ff=%d n=%d rate=%d+%d rows=%d float=%zu bitdiff=0 PASS\n",
                k, ff, n, g.b, g.f, rows, (size_t) rows * n);
}
void normal_recon_batch(Mat& g1, Mat& u1, Mat& d1, Mat& g2, Mat& u2, Mat& d2,
                        Mat& gs, Mat& us, Mat& ds, int r1, int r2) {
    const int k = g1.k, ff = g1.n, n = d1.n, routes = r1 + r2, tokens = routes + 3;
    std::vector<uint16_t> hx((size_t) tokens * k);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = half(((int) (i % 79) - 39) * 0.001f);
    std::vector<int> hmap(routes), hdest(routes);
    for (int r = 0; r < routes; ++r) { hmap[r] = (r * 11 + 5) % tokens; hdest[r] = r; }
    for (int r = routes - 1; r > 0; --r) { const int j = (r * 13 + 7) % (r + 1); std::swap(hdest[r], hdest[j]); }
    Buf<uint16_t> x(hx.size()); x.put(hx);
    Buf<int> map(routes), dest(routes); map.put(hmap); dest.put(hdest);
    Buf<float> ya((size_t) routes * n), yb((size_t) routes * n);
    Exl3Group gn[2] = {{g1.a, u1.a, d1.a, 0, r1}, {g2.a, u2.a, d2.a, r1, r2}};
    // The two compared arms must decode IDENTICAL weights.  Band8 runs only the VERIFIED
    // rates (K2/K2.5/K3/K4): where the rate-mixed twin (K3.5/K4.5) is band8-legal both arms
    // mirror the mix (the real MiMo K2/K2.5 boundary); elsewhere both arms mirror the
    // same-rate twin gs, the native-only mixed pair stays an admission smoke below, and the
    // rate-mixed band8 call is asserted REFUSED (false = nothing enqueued).
    const bool twin_band8 = g2.b == 2 || ((g2.b == 3 || g2.b == 4) && g2.f == 0);
    Mat& tb = twin_band8 ? g2 : gs; Mat& tu = twin_band8 ? u2 : us; Mat& td = twin_band8 ? d2 : ds;
    Exl3Group gn2[2] = {{g1.a, u1.a, d1.a, 0, r1}, {tb.a, tu.a, td.a, r1, r2}};
    Exl3Group gb[2] = {{g1.z, u1.z, d1.z, 0, r1}, {tb.z, tu.z, td.z, r1, r2}};
    Exl3MoeWorkspace wa, wb;
    const int m_rows = std::max(r1, r2);
    wa.reserve_recon_batch(2, 2 * m_rows, k, ff, n);
    wb.reserve_recon_batch(2, 2 * m_rows, k, ff, n);
    for (int fp16 : {0, 1}) {   // the NORMAL route honors the flag for BOTH layouts
        exl3_gemm_fp16_out_set(fp16 != 0);
        require(exl3_moe_recon_batch(gn2, 2, x.p, map.p, dest.p, ya.p, nullptr, wa), "native recon refused");
        require(exl3_moe_recon_batch(gb, 2, x.p, map.p, dest.p, yb.p, nullptr, wb), "band8 recon NORMAL route refused");
        ck(cudaGetLastError());
        auto va = ya.get(), vb = yb.get();
        equal(va, vb, "recon batch NORMAL API floats");
        if (!twin_band8) {
            // native-only mixed pair (K3.5/K4.5 twin): dispatches, outputs finite/nonzero -
            // no band8 twin exists BY DESIGN, so this is an admission smoke, not parity
            require(exl3_moe_recon_batch(gn, 2, x.p, map.p, dest.p, ya.p, nullptr, wa),
                    "native mixed-rate recon refused");
            for (float v : ya.get()) require(std::isfinite(v), "nonfinite native mixed recon");
            // admission: an unverified rate never reaches a band8 kernel
            Exl3Group bad[2] = {gb[0], {g2.z, u2.z, d2.z, r1, r2}};
            require(!exl3_moe_recon_batch(bad, 2, x.p, map.p, dest.p, yb.p, nullptr, wb),
                    "band8 recon accepted an unverified rate");
        }
        // mixed tags: one Band8 group + one Native group is refused BEFORE enqueue
        {   Exl3Group mix[2] = {gn2[0], gb[1]};
            require(!exl3_moe_recon_batch(mix, 2, x.p, map.p, dest.p, yb.p, nullptr, wb), "mixed-tag recon accepted");
        }
        std::printf("NORMAL recon_batch k=%d ff=%d n=%d rate1=%d+%d rate2=%d+%d b8pair=%s rows=%d+%d fp16out=%d float=%zu bitdiff=0 PASS\n",
                    k, ff, n, g1.b, g1.f, g2.b, g2.f, twin_band8 ? "mixed" : "same-rate",
                    r1, r2, fp16, va.size());
    }
    exl3_gemm_fp16_out_set(false);
}
void normal_expert_batch(Mat& g, Mat& u, Mat& d, Mat& g2, Mat& u2, Mat& d2) {
    // MULTIGROUPS: four experts of differing widths (>=24 regime), permuted destinations,
    // per-route token map - the full batched host API, not a one-group kernel proof.
    const int rows[4] = {32, 40, 63, 100};
    const int routes = rows[0] + rows[1] + rows[2] + rows[3];
    const int k = g.k, ff = g.n, n = d.n, tokens = routes + 5;
    std::vector<uint16_t> hx((size_t) tokens * k);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = half(((int) (i % 73) - 36) * 0.001f);
    std::vector<int> hmap(routes), hdest(routes);
    for (int r = 0; r < routes; ++r) { hmap[r] = (r * 17 + 6) % tokens; hdest[r] = r; }
    for (int r = routes - 1; r > 0; --r) { const int j = (r * 19 + 5) % (r + 1); std::swap(hdest[r], hdest[j]); }
    Buf<uint16_t> x(hx.size()); x.put(hx);
    Buf<int> map(routes), dest(routes); map.put(hmap); dest.put(hdest);
    Buf<float> ya((size_t) routes * n), yb((size_t) routes * n);
    Exl3Group gn[4] = {{g.a, u.a, d.a, 0, rows[0]}, {g2.a, u2.a, d2.a, rows[0], rows[1]},
                       {g.a, u.a, d.a, rows[0] + rows[1], rows[2]}, {g2.a, u2.a, d2.a, routes - rows[3], rows[3]}};
    Exl3Group gb[4] = {{g.z, u.z, d.z, 0, rows[0]}, {g2.z, u2.z, d2.z, rows[0], rows[1]},
                       {g.z, u.z, d.z, rows[0] + rows[1], rows[2]}, {g2.z, u2.z, d2.z, routes - rows[3], rows[3]}};
    Exl3MoeWorkspace wa, wb;
    wa.reserve_expert_batch(4, routes, k, ff, n);
    wb.reserve_expert_batch(4, routes, k, ff, n);
    require(exl3_moe_expert_batch(gn, 4, x.p, map.p, dest.p, ya.p, nullptr, wa), "native expert batch refused");
    require(exl3_moe_expert_batch(gb, 4, x.p, map.p, dest.p, yb.p, nullptr, wb), "band8 expert batch refused");
    ck(cudaGetLastError());
    auto va = ya.get(), vb = yb.get();
    equal(va, vb, "expert batch NORMAL API floats");
    size_t nz = 0;
    for (float v : vb) { require(std::isfinite(v), "nonfinite expert batch"); nz += v != 0; }
    require(nz > 0, "allzero expert batch");
    // mixed tags across groups: refused before upload/launch
    {   Exl3Group mix[4]; for (int i = 0; i < 4; ++i) mix[i] = i == 1 ? gb[i] : gn[i];
        require(!exl3_moe_expert_batch(mix, 4, x.p, map.p, dest.p, yb.p, nullptr, wb), "mixed-tag expert batch accepted");
    }
    std::printf("NORMAL expert_batch k=%d ff=%d n=%d rate=%d+%d/%d+%d groups=4 rows=%d+%d+%d+%d float=%zu bitdiff=0 nonzero=%zu PASS\n",
                k, ff, n, g.b, g.f, g2.b, g2.f, rows[0], rows[1], rows[2], rows[3], va.size(), nz);
}
void normal_ws_mode(Mat& g, Mat& u, Mat& d, int routes) {
    // The device-only masked entry + the workspace layout mode: default Native, set Band8 once,
    // invalid enum throws; native masked vs band8 masked (device descriptors uploaded by hand)
    // and a graph CAPTURE + REPLAY of both arms.  No D2H descriptor read per call.
    const int k = g.k, ff = g.n, n = d.n, tokens = routes + 4;
    std::vector<uint16_t> hx((size_t) tokens * k);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = half(((int) (i % 89) - 44) * 0.001f);
    std::vector<int> hmap(routes);
    for (int r = 0; r < routes; ++r) hmap[r] = (r * 13 + 2) % tokens;
    Buf<uint16_t> x(hx.size()); x.put(hx);
    Buf<int> map(routes); map.put(hmap);
    std::vector<Exl3Mat> ga(routes, g.a), ua(routes, u.a), da(routes, d.a);
    std::vector<Exl3Mat> gb(routes, g.z), ub(routes, u.z), db(routes, d.z);
    Buf<Exl3Mat> dga(routes), dua(routes), dda(routes), dgb(routes), dub(routes), ddb(routes);
    dga.put(ga); dua.put(ua); dda.put(da); dgb.put(gb); dub.put(ub); ddb.put(db);
    Exl3MoeWorkspace wa, wb;
    wa.reserve(routes, k, ff, n, true);
    wb.reserve(routes, k, ff, n, true);
    require(wb.trellis_layout() == Exl3TrellisLayout::Native, "workspace default layout not Native");
    bool threw = false;
    try { wb.set_trellis_layout((Exl3TrellisLayout) 7); } catch (const std::invalid_argument&) { threw = true; }
    require(threw, "invalid layout enum accepted");
    wb.set_trellis_layout(Exl3TrellisLayout::Band8);
    require(wb.trellis_layout() == Exl3TrellisLayout::Band8, "layout mode not retained");
    Buf<uint16_t> ya((size_t) routes * n), yb((size_t) routes * n);
    // warm (non-captured) calls first - the capture contract validates on a warm call
    exl3_moe_rows_masked(dga.p, dua.p, dda.p, routes, k, ff, n, x.p, map.p, nullptr, wa);
    exl3_moe_rows_masked(dgb.p, dub.p, ddb.p, routes, k, ff, n, x.p, map.p, nullptr, wb);
    ck(cudaGetLastError());
    ck(cudaDeviceSynchronize());
    // capture + replay BOTH arms; the band8 arm replays through the WS mode only.  The entry
    // returns workspace-owned row pointers (valid until the next ordered FFN call on the WS).
    cudaStream_t sa, sb;
    ck(cudaStreamCreate(&sa)); ck(cudaStreamCreate(&sb));
    cudaGraph_t graph_a, graph_b;
    ck(cudaStreamBeginCapture(sa, cudaStreamCaptureModeGlobal));
    const uint16_t* rows_a =
        exl3_moe_rows_masked(dga.p, dua.p, dda.p, routes, k, ff, n, x.p, map.p, sa, wa);
    ck(cudaStreamEndCapture(sa, &graph_a));
    ck(cudaStreamBeginCapture(sb, cudaStreamCaptureModeGlobal));
    const uint16_t* rows_b =
        exl3_moe_rows_masked(dgb.p, dub.p, ddb.p, routes, k, ff, n, x.p, map.p, sb, wb);
    ck(cudaStreamEndCapture(sb, &graph_b));
    cudaGraphExec_t exec_a, exec_b;
    ck(cudaGraphInstantiate(&exec_a, graph_a, nullptr, nullptr, 0));
    ck(cudaGraphInstantiate(&exec_b, graph_b, nullptr, nullptr, 0));
    for (int rep = 0; rep < 3; ++rep) { ck(cudaGraphLaunch(exec_a, sa)); ck(cudaGraphLaunch(exec_b, sb)); }
    ck(cudaStreamSynchronize(sa)); ck(cudaStreamSynchronize(sb));
    ck(cudaMemcpy(ya.p, rows_a, ya.count * 2, cudaMemcpyDeviceToDevice));
    ck(cudaMemcpy(yb.p, rows_b, yb.count * 2, cudaMemcpyDeviceToDevice));
    equal(ya.get(), yb.get(), "masked WS-mode graph halves");
    // mutate the token map between replays: a replay reads the LIVE buffers
    for (int r = 0; r < routes; ++r) hmap[r] = (hmap[r] + 1 + (r & 1)) % tokens;
    map.put(hmap);
    ck(cudaGraphLaunch(exec_a, sa)); ck(cudaGraphLaunch(exec_b, sb));
    ck(cudaStreamSynchronize(sa)); ck(cudaStreamSynchronize(sb));
    ck(cudaMemcpy(ya.p, rows_a, ya.count * 2, cudaMemcpyDeviceToDevice));
    ck(cudaMemcpy(yb.p, rows_b, yb.count * 2, cudaMemcpyDeviceToDevice));
    equal(ya.get(), yb.get(), "masked WS-mode replay-after-mutation halves");
    ck(cudaGraphExecDestroy(exec_a)); ck(cudaGraphExecDestroy(exec_b));
    ck(cudaGraphDestroy(graph_a)); ck(cudaGraphDestroy(graph_b));
    ck(cudaStreamDestroy(sa)); ck(cudaStreamDestroy(sb));
    std::printf("NORMAL masked ws-mode k=%d ff=%d n=%d rate=%d+%d routes=%d replay=4 half=%zu bitdiff=0 PASS\n",
                k, ff, n, g.b, g.f, routes, (size_t) routes * n);
}
// The whole normal-API matrix at the four verified rates, one geometry each.  The env-driven
// arm selection (RDNA/FUSE/native flags) comes from the PARENT's child passes so the cached
// env reads stay deterministic per process: scalar (no env), rdna1, fuse3, native-grouped.
void normal_api_suite(const char* mode, bool fuse_forms) {
    ck(cudaSetDevice(0));
    hipDeviceProp_t prop{};
    ck(hipGetDeviceProperties(&prop, 0));
    if (!gfx_arch_is_gfx11_wmma(prop.gcnArchName)) { std::puts("NORMAL SUITE unsupported arch SKIP"); return; }
    std::mt19937 rng(20261107);
    for (int rate = 0; rate < 4; ++rate) {
        const int b = rate < 2 ? 2 : rate + 1, f = rate == 1;
        const int k = 1024, ff = 512, n = 1024;   // ki=64/nj=32: multiples of 8, permutation live
        Mat g(k, ff, b, f, rng), u(k, ff, b, f, rng), d(ff, n, b, f, rng);
        const size_t start = comparisons;
        normal_ffn_indexed(g, u, d, 13, 7);
        normal_ffn_broadcast(g, u, d, 4);
        // The public contract caps group width at 6 (fixed host plan arrays); 8/16 are NOT
        // API inputs - the negative arm proves both layouts refuse before enqueue.
        for (int width : {1, 2, 3, 4, 5, 6}) normal_grouped(g, u, d, width, false);
        normal_grouped_refuses_width_8(g, u, d);
        if (fuse_forms) { normal_grouped(g, u, d, 2, true); normal_grouped(g, u, d, 4, true); }
        normal_group_gemm(g, u, d, 48);
        normal_wmma(g, u, d, 8);      // below the experimental entry's 16-row floor: normal route
        normal_wmma(g, u, d, 100);
        {   // expert_batch takes ONE (bits,frac) per call (a K2/K2.5 boundary splits into two
            // calls upstream, header contract) - its second expert is a same-rate twin gs;
            // gs also mirrors the recon comparison at rates whose mixed twin (K3.5/K4.5) is
            // outside band8's verified set.  The rate-MIXED twin g2 below is the recon mix.
            Mat gs(k, ff, b, f, rng), us(k, ff, b, f, rng), ds(ff, n, b, f, rng);
            normal_expert_batch(g, u, d, gs, us, ds);
            Mat g2(k, ff, b, f ? 0 : 1, rng), u2(k, ff, b, f ? 0 : 1, rng), d2(ff, n, b, f ? 0 : 1, rng);
            normal_recon_batch(g, u, d, g2, u2, d2, gs, us, ds, 16, 24);
        }
        normal_ws_mode(g, u, d, 9);
        std::printf("NORMAL GEO rate=%d+%d mode=%s compared=%zu bitdiff=0 PASS\n", b, f, mode, comparisons - start);
    }
    ck(cudaDeviceSynchronize());
    std::printf("NORMAL SUITE mode=%s compared=%zu bitdiff=0 PASS\n", mode, comparisons);
}
#endif
}

// ===================== GPU TIMING BENCH (--bench) =====================
// Measures the SAME production grouped-GEMV arithmetic (rdna_native::gemv_grouped
// via moe_run_experts_grouped_dev / direct launches) on two trellis layouts:
// native (kt,nt) row-major tiles vs the CPU-repacked band8 permutation. Both
// arms run the identical host path with pre-uploaded DEVICE descriptors (the
// public band8 entry's per-call device queries + descriptor memcpys are measured
// separately as host enqueue overhead). One corpus buffer holds ONE layout at a
// time - never a native shadow during candidate timing; packing, H2D and
// allocation stay outside the timed regions. Every timed iteration streams the
// whole corpus (>96 MiB distinct weights) so the reads are bandwidth-honest.
// Rounds alternate arm order (AB/BA/AB) after a full warm revolution; the hot
// case repeats a small expert set instead. Per-iteration events time one full
// corpus revolution on one stream; graph mode replays the captured revolution.
#if defined(STRATA_WMMA_GFX11)
#include <chrono>
#include <algorithm>
#include <memory>
#include <functional>
static uint16_t bench_half(float x) { return __half_as_ushort(__float2half_rn(x)); }
struct BenchGeo {
    const char* name; int k, ff; int bg, fg, bu, fu, bd, fd; int experts;
};
static const BenchGeo kBenchGeos[] = {
    {"mimo-k2",  4096, 2048, 2, 0, 2, 0, 2, 0, 24},
    {"mimo-k25", 4096, 2048, 2, 1, 2, 1, 2, 1, 24},
    {"mimo-mix", 4096, 2048, 2, 0, 2, 1, 2, 0, 24},
    {"qwen-k3",  2560, 640,  3, 0, 3, 0, 3, 0, 64},
    {"qwen-k4",  2560, 640,  4, 0, 4, 0, 4, 0, 64},
};
struct WinTab {                       // one FFN group window over corpus experts
    int first_expert = 0, groups = 0, routes = 0, widest = 1;
    std::vector<int> rows, first, input_rows;   // host tables kept alive whole run
};
struct BenchCorpus {
    const BenchGeo& G;
    int ki, njg, njd;
    size_t words_g, words_u, words_d, per_expert, corpus_words;
    std::vector<uint16_t> hnative, hpacked, hscales;
    std::unique_ptr<Buf<uint16_t>> tre, scales;
    int resident = -1;                // -1 none, 0 native, 1 band8 (ONE layout resident)
    explicit BenchCorpus(const BenchGeo& g, std::mt19937& rng) : G(g) {
        ki = g.k / 16; njg = g.ff / 16; njd = g.k / 16;
        const int pg = 16 * g.bg + 8 * g.fg, pu = 16 * g.bu + 8 * g.fu, pd = 16 * g.bd + 8 * g.fd;
        words_g = (size_t) ki * njg * pg;
        words_u = (size_t) ki * njg * pu;
        words_d = (size_t) njg * njd * pd;      // down: ff x k (n = k)
        per_expert = words_g + words_u + words_d;
        corpus_words = per_expert * g.experts;
        hnative.resize(corpus_words);
        for (auto& v : hnative) v = (uint16_t) rng();
        // Band8 repack per plane (same bytes, tile (kt,nt) -> ((nt/8)*ki+kt)*8+nt%8);
        // each plane keeps its own geometry and word count (mixed rates differ).
        hpacked.resize(corpus_words);
        std::vector<uint16_t> src, dst;
        for (int e = 0; e < g.experts; ++e) {
            const uint16_t* np = hnative.data() + (size_t) e * per_expert;
            uint16_t* pp = hpacked.data() + (size_t) e * per_expert;
            const int pk[3] = {ki, ki, njg}, pn[3] = {njg, njg, njd}, pw[3] = {pg, pu, pd};
            for (int plane = 0; plane < 3; ++plane) {
                const size_t count = (size_t) pk[plane] * pn[plane] * pw[plane];
                src.assign(np, np + count);
                dst.resize(count);
                require(pack(src, dst, pk[plane], pn[plane], pw[plane]), "bench plane pack");
                std::memcpy(pp, dst.data(), count * 2);
                np += count; pp += count;
            }
        }
        hscales.resize((size_t) g.experts * (3 * (g.k + g.ff)));
        for (auto& v : hscales) v = bench_half((0.10f + (rng() % 32) * 0.001f) * (rng() % 2 ? 1 : -1));
        tre.reset(new Buf<uint16_t>(corpus_words));
        scales.reset(new Buf<uint16_t>(hscales.size()));
        scales->put(hscales);
    }
    void ensure(int arm, cudaStream_t st) {     // untimed layout swap, no reallocation
        if (resident == arm) return;
        const std::vector<uint16_t>& src = arm ? hpacked : hnative;
        ck(cudaMemcpyAsync(tre->p, src.data(), src.size() * 2, cudaMemcpyHostToDevice, st));
        ck(cudaStreamSynchronize(st));
        resident = arm;
    }
    Exl3Mat mat(int e, int plane) const {       // plane 0=gate 1=up 2=down (layout-independent)
        const uint16_t* tr = tre->p + (size_t) e * per_expert +
            (plane == 2 ? words_g + words_u : plane == 1 ? words_g : 0);
        const size_t so = ((size_t) e * 3 + plane) * ((size_t) G.k + G.ff);
        const int in = plane == 2 ? G.ff : G.k, out = plane == 2 ? G.k : G.ff;
        return {tr, scales->p + so, scales->p + so + in,
                in / 16, out / 16, plane == 0 ? G.bg : plane == 1 ? G.bu : G.bd, 2,
                plane == 0 ? G.fg : plane == 1 ? G.fu : G.fd};
    }
};
struct WinDev {                                // per-window DEVICE descriptors + tables
    Buf<Exl3Mat> dg, du, dd; Buf<int> first, rows, input_rows;
    int rates[3] = {0, 0, 0}, parts[3] = {0, 0, 0};
    int wgroups = 0, wroutes = 0, wwidest = 1;
    WinDev(int groups, int routes) : dg(groups), du(groups), dd(groups), first(groups),
        rows(groups), input_rows(routes) {}
};
// plan + upload one window; forced_parts mirrors the public entry's override check
static void win_plan_upload(const BenchCorpus& c, const WinTab& t, bool fused, int forced,
                            std::vector<Exl3Mat>& hg, std::vector<Exl3Mat>& hu,
                            std::vector<Exl3Mat>& hd, WinDev& d) {
    hg.clear(); hu.clear(); hd.clear();
    int prefix = 0;
    for (int g = 0; g < t.groups; ++g) {
        const int e = (t.first_expert + g) % c.G.experts;
        hg.push_back(c.mat(e, 0)); hu.push_back(c.mat(e, 1)); hd.push_back(c.mat(e, 2));
        require(t.first[g] == prefix, "bench window first table");
        prefix += t.rows[g];
    }
    require(prefix == t.routes, "bench window routes");
    require(exl3_native_grouped_plan(hg.data(), hu.data(), hd.data(), t.groups, t.widest,
                                     true, fused, d.rates, d.parts), "bench plan refused");
    const int width = t.widest <= 2 ? t.widest : 4;
    if (forced) {
        const int ki[3] = {hg[0].ki, hu[0].ki, hd[0].ki};
        for (int b = 0; b < 3; ++b) {
            require(forced <= ki[b] &&
                    native_grouped_widest(ki[b], forced) * width * 2 +
                        (fused ? kNativeGroupedStaticFused : 0) <= kNativeGroupedLdsBudget,
                    "bench forced parts LDS");
            d.parts[b] = forced;
        }
    }
    d.wgroups = t.groups; d.wroutes = t.routes; d.wwidest = t.widest;
    d.dg.put(hg); d.du.put(hu); d.dd.put(hd);
    d.first.put(t.first); d.rows.put(t.rows); d.input_rows.put(t.input_rows);
}
template<int R> void bench_raw_launch(bool frac, int kb, bool band8, dim3 grid, size_t smem,
                                      cudaStream_t st, const float* xh, const Exl3Mat* mats,
                                      const int* f, const int* r, float* z) {
    auto go = [&](auto b_tag) {
        constexpr bool B = decltype(b_tag)::value;
        if (frac) rdna_native::gemv_grouped<true, R, false, false, 2, B><<<grid, 256, smem, st>>>(xh, mats, f, r, z);
        else if (kb == 2) rdna_native::gemv_grouped<false, R, false, false, 2, B><<<grid, 256, smem, st>>>(xh, mats, f, r, z);
        else if (kb == 3) rdna_native::gemv_grouped<false, R, false, false, 3, B><<<grid, 256, smem, st>>>(xh, mats, f, r, z);
        else rdna_native::gemv_grouped<false, R, false, false, 4, B><<<grid, 256, smem, st>>>(xh, mats, f, r, z);
    };
    if (band8) go(std::true_type{}); else go(std::false_type{});
    ck(cudaGetLastError());
}
struct RawWin {                                 // all-experts raw window at one width
    int w, ps_g, ps_d;
    std::unique_ptr<Buf<Exl3Mat>> dg, dd;
    std::unique_ptr<Buf<int>> first, rows;
    RawWin(int w_, int experts) : w(w_), ps_g(0), ps_d(0),
        dg(new Buf<Exl3Mat>((size_t) experts)), dd(new Buf<Exl3Mat>((size_t) experts)),
        first(new Buf<int>((size_t) experts)), rows(new Buf<int>((size_t) experts)) {}
};
struct BenchCase {
    const char* name; int kind;   // 0 rawg 1 rawd 2 real 3 w4 4 hot 5 real-unfused
    int width; bool fused; bool graph; int forced_parts;
};
static const BenchCase kCases[] = {
    {"rawg1", 0, 1, false, false, 0}, {"rawg2", 0, 2, false, false, 0},
    {"rawg4", 0, 4, false, false, 0},
    {"rawd1", 1, 1, false, false, 0}, {"rawd2", 1, 2, false, false, 0},
    {"rawd4", 1, 4, false, false, 0},
    {"ffn_real", 2, 0, true, true, 0},
    {"ffn_real_unfused", 5, 0, false, false, 0},
    {"ffn_w4", 3, 4, true, false, 0},
    {"ffn_w4_p1", 3, 4, true, false, 1},
    {"ffn_hot", 4, 4, true, false, 0},
};
struct BenchSample { std::string cs, geo, mode; int arm, round, it; float ms; };
static double bench_median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
static int bench_main(int argc, char** argv) {
    int rounds = 3, iters = 4, hot_calls = 8, warmup_ms = 0, first_arm = 0;
    const char* only = nullptr;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
        else if (!std::strcmp(argv[i], "--hot-calls") && i + 1 < argc) hot_calls = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--warmup-ms") && i + 1 < argc) warmup_ms = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--first-arm") && i + 1 < argc) {
            const char* a = argv[++i];
            if (!std::strcmp(a, "native")) first_arm = 0;
            else if (!std::strcmp(a, "band8")) first_arm = 1;
            else require(false, "--first-arm must be native|band8");
        }
    }
    require(rounds >= 1 && iters >= 1 && hot_calls >= 1, "bench args must be positive");
    require(warmup_ms >= 0 && warmup_ms <= 1000, "warmup must be within [0,1000] ms");
    ck(cudaSetDevice(0));
    hipDeviceProp_t prop{}; ck(hipGetDeviceProperties(&prop, 0));
    require(gfx_arch_is_gfx11_wmma(prop.gcnArchName), "bench needs gfx11 wmma");
    std::printf("BENCH device=%s arch=%s wgps=%d (multiProcessorCount; RX 7900 XTX has 96 CUs / 48 WGPs) "
                "l2CacheSize=%u totalMem=%.1fGiB rounds=%d iters=%d warmup_ms=%d first_arm=%s\n",
                prop.name, prop.gcnArchName, prop.multiProcessorCount,
                (unsigned) prop.l2CacheSize, prop.totalGlobalMem / 1073741824.0, rounds, iters,
                warmup_ms, first_arm ? "band8" : "native");
    cudaStream_t st; ck(cudaStreamCreate(&st));
    std::vector<BenchSample> samples;
    for (int gi = 0; gi < (int) (sizeof(kBenchGeos) / sizeof(kBenchGeos[0])); ++gi) {
        const BenchGeo& G = kBenchGeos[gi];
        // (no geo-level skip: --only filters per case AND geo name below)
        const auto build0 = std::chrono::steady_clock::now();
        std::mt19937 rng(20261106 + gi), rrng(20261107 + gi);
        BenchCorpus corpus(G, rng);
        const double build_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - build0).count();
        // ---- windows: realistic (M=4 top-8: 32 routes over 24 experts), w4, hot ----
        auto make_real = [&](int first_expert) {
            WinTab t; t.first_expert = first_expert; t.groups = 24;
            t.rows.assign(t.groups, 1);
            std::vector<int> idx(t.groups);
            for (int i = 0; i < t.groups; ++i) idx[i] = i;
            std::shuffle(idx.begin(), idx.end(), rrng);
            for (int i = 0; i < 32 - t.groups; ++i) t.rows[idx[i]] = 2;   // 8 two-row groups
            t.routes = 32; t.widest = 2;
            int prefix = 0;
            for (int g = 0; g < t.groups; ++g) { t.first.push_back(prefix); prefix += t.rows[g]; }
            t.input_rows.resize(t.routes);
            for (int r = 0; r < t.routes; ++r) t.input_rows[r] = (int) (rrng() % 4);
            return t;
        };
        std::vector<WinTab> real = G.experts > 32
            ? std::vector<WinTab>{make_real(0), make_real(24), make_real(48)}
            : std::vector<WinTab>{make_real(0)};
        WinTab w4; w4.groups = G.experts / 4; w4.rows.assign(w4.groups, 4);
        w4.routes = G.experts; w4.widest = 4;
        for (int g = 0, p = 0; g < w4.groups; ++g) { w4.first.push_back(p); p += 4; }
        w4.input_rows.resize(w4.routes);
        for (int r = 0; r < w4.routes; ++r) w4.input_rows[r] = r % 4;
        const int hot_experts = G.experts > 32 ? 8 : 4;
        WinTab hot; hot.groups = hot_experts; hot.rows.assign(hot_experts, 4);
        hot.routes = hot_experts * 4; hot.widest = 4;
        for (int g = 0, p = 0; g < hot.groups; ++g) { hot.first.push_back(p); p += 4; }
        hot.input_rows.resize(hot.routes);
        for (int r = 0; r < hot.routes; ++r) hot.input_rows[r] = r % 4;
        // ---- raw windows (all experts, uniform rows=w; gate plane and down plane) ----
        std::vector<std::unique_ptr<RawWin>> raw;
        for (int w : {1, 2, 4}) {
            raw.emplace_back(new RawWin(w, G.experts));
            RawWin& rw = *raw.back();
            std::vector<Exl3Mat> hg, hu, hd;
            std::vector<int> first(G.experts), rowsv(G.experts, w);
            for (int e = 0; e < G.experts; ++e) first[e] = e * w;
            for (int e = 0; e < G.experts; ++e) { hg.push_back(corpus.mat(e, 0)); hd.push_back(corpus.mat(e, 2)); }
            hu = hg;                                              // plan needs a trio; gate/up share geometry
            int r3[3], p3[3];
            require(exl3_native_grouped_plan(hg.data(), hu.data(), hd.data(), G.experts, w,
                                             true, false, r3, p3), "raw plan refused");
            rw.ps_g = p3[0]; rw.ps_d = p3[2];
            rw.dg->put(hg); rw.dd->put(hd); rw.first->put(first); rw.rows->put(rowsv);
        }
        // ---- device IO ----
        Buf<float> xraw((size_t) G.experts * 4 * G.k);            // covers every raw width
        { std::vector<float> hx(xraw.count);
          for (size_t i = 0; i < hx.size(); ++i) hx[i] = ((int) (i % 61) - 30) * 0.0001f;
          xraw.put(hx); }
        Buf<uint16_t> x16((size_t) 4 * G.k);
        { std::vector<uint16_t> hx(x16.count);
          for (size_t i = 0; i < hx.size(); ++i) hx[i] = bench_half(((int) (i % 97) - 48) * 0.001f);
          x16.put(hx); }
        const int max_routes = std::max(32, std::max(G.experts, hot_experts * 4));
        Buf<uint16_t> out16((size_t) max_routes * G.k);
        Exl3MoeWorkspace ws;
        ws.reserve_small_m(max_routes, std::max(24, G.experts), G.k, G.ff, G.k);
        // ---- per-window device descriptors + plans (FFN cases) ----
        std::vector<std::unique_ptr<WinDev>> rdev;
        std::vector<std::vector<Exl3Mat>> hostg, hostu, hostd;   // host mirrors kept alive
        auto upload_window = [&](const WinTab& t, bool fused, int forced) {
            rdev.emplace_back(new WinDev(t.groups, t.routes));
            hostg.emplace_back(); hostu.emplace_back(); hostd.emplace_back();
            win_plan_upload(corpus, t, fused, forced, hostg.back(), hostu.back(),
                            hostd.back(), *rdev.back());
            return (int) rdev.size() - 1;
        };
        std::vector<int> real_fused, real_unfused, w4_auto, w4_p1, hot_id;
        for (const auto& t : real) real_fused.push_back(upload_window(t, true, 0));
        for (const auto& t : real) real_unfused.push_back(upload_window(t, false, 0));
        w4_auto.push_back(upload_window(w4, true, 0));
        w4_p1.push_back(upload_window(w4, true, 1));
        hot_id.push_back(upload_window(hot, true, 0));
        auto ffn_call = [&](const std::vector<int>& ids, bool band8, bool fused) {
            for (int id : ids) {
                const WinDev& d = *rdev[id];
                auto* fn = band8 ? &moe_run_experts_grouped_dev<true>
                                 : &moe_run_experts_grouped_dev<false>;
                (*fn)(d.dg.p, d.du.p, d.dd.p, d.first.p, d.rows.p, d.wgroups, d.wroutes,
                      d.wwidest, x16.p, st, ws, G.k, d.input_rows.p, G.k, G.ff, G.k, 2,
                      fused, out16.p, d.rates[0], d.rates[1], d.rates[2],
                      d.parts[0], d.parts[1], d.parts[2]);
            }
            ck(cudaGetLastError());
        };
        // ---- sanity: native vs band8 FFN outputs bitwise (bench wiring guard) ----
        corpus.ensure(0, st);
        ffn_call(real_fused, false, true); ck(cudaStreamSynchronize(st));
        auto out_native = out16.get(std::min<size_t>(out16.count, (size_t) 32 * G.k));
        corpus.ensure(1, st);
        ffn_call(real_fused, true, true); ck(cudaStreamSynchronize(st));
        auto out_band = out16.get(out_native.size());
        size_t bitdiff = 0;
        for (size_t i = 0; i < out_native.size(); ++i) bitdiff += out_native[i] != out_band[i];
        std::printf("BENCHSANITY geo=%s ffn_real native-vs-band8 bitdiff=%zu/%zu %s\n",
                    G.name, bitdiff, out_native.size(), bitdiff ? "FAIL" : "PASS");
        require(bitdiff == 0, "bench layout wiring");
        // ---- host enqueue overhead: public band8 entry vs matched direct dispatch ----
        // (each arm runs on ITS OWN layout: ensure() + resident assert before every loop -
        //  the native loop must never execute against band8-resident bytes)
        if (gi == 0) {
            const int K = 64, wid = real_fused[0];
            corpus.ensure(1, st);
            require(corpus.resident == 1, "host probe band8 residency");
            // Public-entry admission reads the TAG (an untagged packed view is refused):
            // a production caller stores Band8-tagged descriptors over the packed bytes.
            // mat() stays layout-independent for the raw/direct-template arms; tag the
            // probe trio once here, outside the timed loop, matching the resident arm.
            std::vector<Exl3Mat> pg = hostg[wid], pv = hostu[wid], pw = hostd[wid];
            for (std::vector<Exl3Mat>* arr : {&pg, &pv, &pw})
                for (Exl3Mat& m : *arr) m.layout = Exl3TrellisLayout::Band8;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < K; ++i)
                require(exl3_experimental_band8_rows({pg.data(), pv.data(),
                    pw.data()}, real[0].first.data(), real[0].rows.data(),
                    real[0].groups, real[0].routes, x16.p, rdev[wid]->input_rows.p,
                    out16.p, st, ws, true, 0), "public entry refused");
            const double pub_us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count() / K;
            ck(cudaStreamSynchronize(st));
            require(corpus.resident == 1, "host probe band8 residency");
            const auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < K; ++i) ffn_call(real_fused, true, true);
            const double dir_b_us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t1).count() / K;
            ck(cudaStreamSynchronize(st));
            corpus.ensure(0, st);                      // native bytes BEFORE the native loop
            require(corpus.resident == 0, "host probe native residency");
            const auto t2 = std::chrono::steady_clock::now();
            for (int i = 0; i < K; ++i) ffn_call(real_fused, false, true);
            const double dir_n_us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t2).count() / K;
            ck(cudaStreamSynchronize(st));
            std::printf("BENCHHOST us/call public-band8=%.1f direct-band8=%.1f direct-native=%.1f "
                        "(K=%d enqueue wall; syncs, layout swaps and GPU time excluded; "
                        "residency asserted per arm)\n",
                        pub_us, dir_b_us, dir_n_us, K);
        }
        // ---- timed cases ----
        for (const BenchCase& cs : kCases) {
            if (only && !std::strstr(cs.name, only) && !std::strstr(G.name, only)) continue;
            std::function<void(bool)> iter;
            double weight_bytes = 0;      // logical weight bytes the iteration DEMANDS (incl. re-reads)
            double active_bytes = 0;      // DISTINCT weight bytes touched per iteration
            int visits = 0;               // expert visits per iteration (>= distinct when repeating)
            const char* scope = "";       // honest cache-scope label; never a blanket HBM claim
            int groups0 = 0, routes0 = 0, width0 = 0, reps = 1;
            char parts_desc[64] = "";
            std::unique_ptr<Buf<float>> praw;
            const RawWin* rw = nullptr;
            if (cs.kind <= 1) {
                rw = raw[cs.width == 1 ? 0 : cs.width == 2 ? 1 : 2].get();
                const int ps = cs.kind == 0 ? rw->ps_g : rw->ps_d;
                const int outs = cs.kind == 0 ? G.ff : G.k;
                const int in_dim = cs.kind == 0 ? G.k : G.ff;
                praw.reset(new Buf<float>((size_t) G.experts * cs.width * ps * outs));
                const size_t plane_words = cs.kind == 0 ? corpus.words_g : corpus.words_d;
                weight_bytes = active_bytes = (double) plane_words * G.experts * 2;   // ONE plane only
                visits = G.experts;
                scope = "single-plane-LLC-scale";   // 32-60 MiB: below the 96 MiB Infinity Cache
                groups0 = G.experts; routes0 = G.experts * cs.width; width0 = cs.width;
                std::snprintf(parts_desc, sizeof parts_desc, "ps=%d in=%d out=%d", ps, in_dim, outs);
                float* z = praw->p;
                const int plane_down = cs.kind == 1;
                iter = [&G, &rw, plane_down, z, &st, &xraw](bool band8) {
                    const int w = rw->w;
                    const bool frac = plane_down ? (bool) G.fd : (bool) G.fg;
                    const int kb = plane_down ? G.bd : G.bg;
                    if (!plane_down) {
                        const size_t smem = native_grouped_widest(G.k / 16, rw->ps_g) * w * 2;
                        const dim3 grid(G.ff / 128, rw->ps_g, G.experts);
                        if (w == 1) bench_raw_launch<1>(frac, kb, band8, grid, smem, st, xraw.p, rw->dg->p, rw->first->p, rw->rows->p, z);
                        else if (w == 2) bench_raw_launch<2>(frac, kb, band8, grid, smem, st, xraw.p, rw->dg->p, rw->first->p, rw->rows->p, z);
                        else bench_raw_launch<4>(frac, kb, band8, grid, smem, st, xraw.p, rw->dg->p, rw->first->p, rw->rows->p, z);
                    } else {
                        const size_t smem = native_grouped_widest(G.ff / 16, rw->ps_d) * w * 2;
                        const dim3 grid(G.k / 128, rw->ps_d, G.experts);
                        if (w == 1) bench_raw_launch<1>(frac, kb, band8, grid, smem, st, xraw.p, rw->dd->p, rw->first->p, rw->rows->p, z);
                        else if (w == 2) bench_raw_launch<2>(frac, kb, band8, grid, smem, st, xraw.p, rw->dd->p, rw->first->p, rw->rows->p, z);
                        else bench_raw_launch<4>(frac, kb, band8, grid, smem, st, xraw.p, rw->dd->p, rw->first->p, rw->rows->p, z);
                    }
                };
            } else {
                const std::vector<int>* ids = &real_fused;
                bool fused = cs.fused;
                if (cs.kind == 5) { ids = &real_unfused; fused = false; }
                else if (cs.kind == 3) ids = cs.forced_parts ? &w4_p1 : &w4_auto;
                else if (cs.kind == 4) { ids = &hot_id; fused = true; reps = hot_calls; }
                const WinDev& d0 = *rdev[ids->at(0)];
                std::snprintf(parts_desc, sizeof parts_desc, "ps=%d/%d/%d", d0.parts[0], d0.parts[1], d0.parts[2]);
                if (cs.kind == 4) {
                    groups0 = hot.groups; routes0 = hot.routes; width0 = hot.widest;
                    weight_bytes = (double) corpus.per_expert * hot_experts * hot_calls * 2;
                    active_bytes = (double) corpus.per_expert * hot_experts * 2;
                    visits = hot_experts * hot_calls;
                    scope = "hot-subset-LLC-resident";
                } else if (cs.kind == 3) {
                    groups0 = w4.groups; routes0 = w4.routes; width0 = w4.widest;
                    // ONLY w4.groups = experts/4 DISTINCT experts are read (one per group),
                    // NOT the whole corpus: this is a subset/cache-scale case by construction.
                    weight_bytes = active_bytes = (double) corpus.per_expert * w4.groups * 2;
                    visits = w4.groups;
                    scope = "subset-quarter-LLC-scale";
                } else {
                    groups0 = (int) ids->size() * real[0].groups;
                    routes0 = (int) ids->size() * 32;
                    width0 = real[0].widest;
                    std::vector<int> used;
                    for (const WinTab& t : real)
                        for (int g = 0; g < t.groups; ++g) used.push_back((t.first_expert + g) % G.experts);
                    visits = (int) used.size();
                    std::sort(used.begin(), used.end());
                    const size_t unique = std::unique(used.begin(), used.end()) - used.begin();
                    weight_bytes = (double) unique * corpus.per_expert * 2;   // each expert read once
                    active_bytes = weight_bytes;
                    scope = unique == (size_t) G.experts && (int) ids->size() > 1
                        ? "whole-corpus-unique-with-repeat-visits"
                        : (int) ids->size() > 1 ? "whole-corpus-unique" : "whole-corpus";
                }
                iter = [&ffn_call, ids, fused, reps](bool band8) {
                    for (int i = 0; i < reps; ++i) ffn_call(*ids, band8, fused);
                };
            }
            cudaGraphExec_t gexec[2] = {nullptr, nullptr};
            std::printf("BENCHCASE geo=%s case=%s groups=%d routes=%d width=%d %s active_distinct_MiB=%.1f visits=%d logical_MiB=%.1f reps=%d scope=%s\n",
                        G.name, cs.name, groups0, routes0, width0, parts_desc,
                        active_bytes / 1048576.0, visits, weight_bytes / 1048576.0, reps, scope);
            for (int round = 0; round < rounds; ++round) {
                // Counterbalanced arm order: --first-arm band8 flips every block's
                // position, so first-use effects cannot alias onto one layout.
                const int lead = (round + first_arm) % 2;
                const int order[2] = {lead ? 1 : 0, lead ? 0 : 1};
                for (int oi = 0; oi < 2; ++oi) {
                    const int arm = order[oi];
                    corpus.ensure(arm, st);
                    std::function<void()> body = [&iter, arm] { iter(arm); };
                    if (cs.graph) {
                        if (!gexec[arm]) {
                            cudaGraph_t graph;
                            ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                            iter(arm);
                            ck(cudaStreamEndCapture(st, &graph));
                            ck(cudaGraphInstantiate(&gexec[arm], graph, 0));
                            ck(cudaGraphDestroy(graph));
                            // replay wiring guard: bitwise vs eager on the same arm
                            iter(arm); ck(cudaStreamSynchronize(st));
                            auto eager = out16.get(std::min<size_t>(out16.count, (size_t) 32 * G.k));
                            ck(cudaGraphLaunch(gexec[arm], st)); ck(cudaStreamSynchronize(st));
                            auto replay = out16.get(eager.size());
                            size_t bd = 0;
                            for (size_t i = 0; i < eager.size(); ++i) bd += eager[i] != replay[i];
                            std::printf("BENCHSANITY geo=%s case=%s arm=%d graph-replay bitdiff=%zu/%zu %s\n",
                                        G.name, cs.name, arm, bd, eager.size(), bd ? "FAIL" : "PASS");
                            require(bd == 0, "graph replay wiring");
                        }
                        cudaGraphExec_t& g = gexec[arm];
                        body = [&g, &st] { ck(cudaGraphLaunch(g, st)); };
                    }
                    body(); ck(cudaStreamSynchronize(st));           // warm full revolution
                    if (warmup_ms > 0) {
                        // Bounded STEADY-STATE warmup measured by GPU events: chunks of 8
                        // bodies, sync + event elapsed per chunk, until warmup_ms of GPU
                        // time. Never a host-enqueue-only spin (that would queue seconds
                        // of work). Clamped to 1000 ms by the CLI validation.
                        cudaEvent_t wa, wb;
                        ck(cudaEventCreate(&wa)); ck(cudaEventCreate(&wb));
                        float spent = 0;
                        while (spent < (float) warmup_ms) {
                            ck(cudaEventRecord(wa, st));
                            for (int i = 0; i < 8; ++i) body();
                            ck(cudaEventRecord(wb, st));
                            ck(cudaStreamSynchronize(st));
                            float chunk; ck(cudaEventElapsedTime(&chunk, wa, wb));
                            spent += chunk;
                        }
                        ck(cudaEventDestroy(wa)); ck(cudaEventDestroy(wb));
                        std::printf("BENCHWARM geo=%s case=%s arm=%d round=%d warmup_gpu_ms=%.1f\n",
                                    G.name, cs.name, arm, round, spent);
                    }
                    std::vector<cudaEvent_t> ev0(iters), ev1(iters);
                    for (int i = 0; i < iters; ++i) { ck(cudaEventCreate(&ev0[i])); ck(cudaEventCreate(&ev1[i])); }
                    for (int i = 0; i < iters; ++i) {
                        ck(cudaEventRecord(ev0[i], st)); body(); ck(cudaEventRecord(ev1[i], st));
                    }
                    ck(cudaStreamSynchronize(st));
                    for (int i = 0; i < iters; ++i) {
                        float ms; ck(cudaEventElapsedTime(&ms, ev0[i], ev1[i]));
                        samples.push_back({cs.name, G.name, cs.graph ? "graph" : "eager", arm, round, i, ms});
                        std::printf("BENCHCSV %s,%s,%s,%s,%d,%d,%.6f\n", cs.name, G.name,
                                    cs.graph ? "graph" : "eager", arm ? "band8" : "native",
                                    round, i, ms);
                        ck(cudaEventDestroy(ev0[i])); ck(cudaEventDestroy(ev1[i]));
                    }
                }
            }
            if (gexec[0]) ck(cudaGraphExecDestroy(gexec[0]));
            if (gexec[1]) ck(cudaGraphExecDestroy(gexec[1]));
            double mean[2] = {0, 0}; int cnt[2] = {0, 0};
            std::vector<double> all[2];
            for (const auto& s : samples)
                if (s.cs == cs.name && s.geo == G.name) { mean[s.arm] += s.ms; cnt[s.arm]++; all[s.arm].push_back(s.ms); }
            for (int arm = 0; arm < 2; ++arm)
                if (cnt[arm])
                    std::printf("BENCHAGG geo=%s case=%s arm=%s n=%d mean_ms=%.4f median_ms=%.4f min_ms=%.4f logical_GBs=%.1f (logical demand bytes/ms, not measured traffic)\n",
                                G.name, cs.name, arm ? "band8" : "native", cnt[arm],
                                mean[arm] / cnt[arm], bench_median(all[arm]),
                                *std::min_element(all[arm].begin(), all[arm].end()),
                                weight_bytes / (mean[arm] / cnt[arm]) / 1e6);
            if (cnt[0] && cnt[1])
                std::printf("BENCHRATIO geo=%s case=%s band8/native=%.4f (means n=%d/%d)\n", G.name, cs.name,
                            (mean[1] / cnt[1]) / (mean[0] / cnt[0]), cnt[0], cnt[1]);
        }
        std::printf("BENCHGEO geo=%s experts=%d corpus_MiB=%.1f host_build_ms=%.0f workspace_bytes=%zu bench_peak_MiB=%.1f\n",
                    G.name, G.experts, corpus.corpus_words * 2.0 / 1048576, build_ms,
                    ws.allocated_bytes(), (peak + ws.allocated_bytes()) / 1048576.0);
        ck(cudaGetLastError());
    }
    ck(cudaStreamDestroy(st));
    std::printf("BENCHDONE samples=%zu bench_peak_MiB=%.1f\n", samples.size(), peak / 1048576.0);
    return 0;
}

// ===================== GPU TIMING BENCH (--bench-wmma): stage-2 prefill =====================
// The SAME WMMA launch sequence (moe_group_wmma_impl, the shared template behind both public
// entries) on two trellis layouts: native vs CPU-repacked band8 - matched host calls, no
// asymmetric per-entry device queries inside the arms. One corpus buffer holds ONE layout at a
// time (ensure() outside every timed region); workspace, x, input_rows and out are preallocated
// per geometry. Full-FFN cases time the whole 3-projection FFN; proj128 times ONE
// routed_wmma_kernel launch on a pre-gathered slab (isolates the weight-read path); w128g
// replays a captured graph (both arms captured independently through the pre-reserved
// workspace). Arm order counterbalanced by --first-arm across runs; warmup is GPU-event-measured
// chunks of 8 bodies; EVERY sample is reported - primary is pooled all-blocks means.
static int bench_wmma_main(int argc, char** argv) {
    int rounds = 4, iters = 64, warmup_ms = 100, first_arm = 0;
    const char* only = nullptr;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
        else if (!std::strcmp(argv[i], "--warmup-ms") && i + 1 < argc) warmup_ms = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--first-arm") && i + 1 < argc) {
            const char* a = argv[++i];
            if (!std::strcmp(a, "native")) first_arm = 0;
            else if (!std::strcmp(a, "band8")) first_arm = 1;
            else require(false, "--first-arm must be native|band8");
        }
    }
    require(rounds >= 1 && iters >= 1 && warmup_ms >= 0 && warmup_ms <= 1000, "bench args must be positive");
    ck(cudaSetDevice(0));
    hipDeviceProp_t prop{};
    ck(hipGetDeviceProperties(&prop, 0));
    require(gfx_arch_is_gfx11_wmma(prop.gcnArchName), "bench needs gfx11 wmma");
    std::printf("BENCHW device=%s arch=%s wgps=%d l2CacheSize=%u rounds=%d iters=%d warmup_ms=%d first_arm=%s\n",
                prop.name, prop.gcnArchName, prop.multiProcessorCount, (unsigned) prop.l2CacheSize,
                rounds, iters, warmup_ms, first_arm ? "band8" : "native");
    cudaStream_t st; ck(cudaStreamCreate(&st));
    struct WCase { const char* name; int rows; bool graph, proj; };
    static const WCase kWCases[] = {
        {"w16", 16, false, false},   {"w64", 64, false, false},
        {"w128", 128, false, false}, {"w512", 512, false, false},
        {"w128g", 128, true, false}, {"proj128", 128, false, true},
    };
    const int max_rows = 512, tokens = 64;
    std::vector<BenchSample> samples;
    for (int gi = 0; gi < (int) (sizeof(kBenchGeos) / sizeof(kBenchGeos[0])); ++gi) {
        const BenchGeo& G = kBenchGeos[gi];
        std::mt19937 rng(20261108 + gi);
        BenchCorpus corpus(G, rng);
        const int k = G.k, ff = G.ff, n = G.k;   // down is ff x k at these geometries
        Buf<uint16_t> x16((size_t) tokens * k);
        { std::vector<uint16_t> hx(x16.count);
          for (size_t i = 0; i < hx.size(); ++i) hx[i] = bench_half(((int) (i % 97) - 48) * 0.001f);
          x16.put(hx); }
        Buf<int> map(max_rows);
        { std::vector<int> hm(max_rows);
          for (int r = 0; r < max_rows; ++r) hm[r] = (r * 17 + 5) % tokens;
          map.put(hm); }
        Buf<float> out((size_t) max_rows * n);
        Exl3MoeWorkspace ws;
        ws.reserve(max_rows, k, ff, n);
        ws.reserve_group_scratch(max_rows, k, ff, n, true);
        // layout-independent host descriptors into the ONE corpus buffer
        const Exl3Mat hg = corpus.mat(0, 0), hu = corpus.mat(0, 1), hd = corpus.mat(0, 2);
        (void) exl3_wmma_device();   // prime the device cache outside any capture
        // projection diagnostic: slab gathered ONCE (layout-independent), then one kernel/iter
        Buf<uint16_t> slab((size_t) 128 * k);
        Buf<float> py((size_t) 128 * ff);
        gather_h128_f16_kernel<<<dim3(k / 128, 128), 32, 0, st>>>(x16.p, map.p, hg.suh, slab.p, k);
        ck(cudaGetLastError());
        ck(cudaStreamSynchronize(st));
        auto proj_launch = [&](bool band8) {   // rows=128 -> row_tiles=2
            const dim3 grid(std::min((128 + 31) / 32, 32), (ff + 255) / 256);
            auto L = [&](auto tag) {
                constexpr int B = decltype(tag)::value;
                if (band8)
                    routed_wmma_kernel<B, 2, false, 8, true><<<grid, 512, 0, st>>>(slab.p, hg.trellis, py.p, 128, k, ff, nullptr, 0, G.fg);
                else
                    routed_wmma_kernel<B, 2, false, 8, false><<<grid, 512, 0, st>>>(slab.p, hg.trellis, py.p, 128, k, ff, nullptr, 0, G.fg);
            };
            if (G.bg == 2) L(std::integral_constant<int, 2>{});
            else if (G.bg == 3) L(std::integral_constant<int, 3>{});
            else L(std::integral_constant<int, 4>{});
            ck(cudaGetLastError());
        };
        // wiring guard: full FFN bitwise native vs band8, before any timing
        corpus.ensure(0, st);
        require(moe_group_wmma_impl<false>(hg, hu, hd, 64, x16.p, map.p, out.p, st, ws), "native arm refused");
        ck(cudaStreamSynchronize(st));
        auto on = out.get(std::min<size_t>(out.count, (size_t) 64 * n));
        corpus.ensure(1, st);
        require(moe_group_wmma_impl<true>(hg, hu, hd, 64, x16.p, map.p, out.p, st, ws), "band8 arm refused");
        ck(cudaStreamSynchronize(st));
        size_t bd = 0;
        { auto ob = out.get(on.size()); for (size_t i = 0; i < on.size(); ++i) bd += on[i] != ob[i]; }
        std::printf("BENCHWSANITY geo=%s wmma native-vs-band8 bitdiff=%zu/%zu %s\n",
                    G.name, bd, on.size(), bd ? "FAIL" : "PASS");
        require(bd == 0, "bench wmma wiring");
        for (const WCase& cs : kWCases) {
            if (only && !std::strstr(cs.name, only) && !std::strstr(G.name, only)) continue;
            std::function<void(bool)> iter;
            if (cs.proj) iter = [&proj_launch](bool band8) { proj_launch(band8); };
            else iter = [&hg, &hu, &hd, &cs, &x16, &map, &out, &st, &ws](bool band8) {
                if (band8) (void) moe_group_wmma_impl<true>(hg, hu, hd, cs.rows, x16.p, map.p, out.p, st, ws);
                else (void) moe_group_wmma_impl<false>(hg, hu, hd, cs.rows, x16.p, map.p, out.p, st, ws);
            };
            cudaGraphExec_t gexec[2] = {nullptr, nullptr};
            std::printf("BENCHWCASE geo=%s case=%s rows=%d mode=%s\n",
                        G.name, cs.name, cs.rows, cs.graph ? "graph" : "eager");
            for (int round = 0; round < rounds; ++round) {
                const int lead = (round + first_arm) % 2;   // counterbalanced arm order
                const int order[2] = {lead ? 1 : 0, lead ? 0 : 1};
                for (int oi = 0; oi < 2; ++oi) {
                    const int arm = order[oi];
                    corpus.ensure(arm, st);
                    std::function<void()> body = [&iter, arm] { iter(arm); };
                    if (cs.graph) {
                        if (!gexec[arm]) {
                            cudaGraph_t graph;
                            ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                            iter(arm);
                            ck(cudaStreamEndCapture(st, &graph));
                            ck(cudaGraphInstantiate(&gexec[arm], graph, 0));
                            ck(cudaGraphDestroy(graph));
                            iter(arm); ck(cudaStreamSynchronize(st));
                            auto eager = out.get();          // replay wiring guard, per arm
                            ck(cudaGraphLaunch(gexec[arm], st)); ck(cudaStreamSynchronize(st));
                            auto replay = out.get(eager.size());
                            size_t rb = 0;
                            for (size_t i = 0; i < eager.size(); ++i) rb += eager[i] != replay[i];
                            std::printf("BENCHWSANITY geo=%s case=%s arm=%d graph-replay bitdiff=%zu/%zu %s\n",
                                        G.name, cs.name, arm, rb, eager.size(), rb ? "FAIL" : "PASS");
                            require(rb == 0, "wmma graph replay wiring");
                        }
                        cudaGraphExec_t& g = gexec[arm];
                        body = [&g, &st] { ck(cudaGraphLaunch(g, st)); };
                    }
                    body(); ck(cudaStreamSynchronize(st));   // one warm body
                    if (warmup_ms > 0) {
                        cudaEvent_t wa, wb;
                        ck(cudaEventCreate(&wa)); ck(cudaEventCreate(&wb));
                        float spent = 0;
                        while (spent < (float) warmup_ms) {
                            ck(cudaEventRecord(wa, st));
                            for (int i = 0; i < 8; ++i) body();
                            ck(cudaEventRecord(wb, st));
                            ck(cudaStreamSynchronize(st));
                            float chunk; ck(cudaEventElapsedTime(&chunk, wa, wb));
                            spent += chunk;
                        }
                        ck(cudaEventDestroy(wa)); ck(cudaEventDestroy(wb));
                        std::printf("BENCHWWARM geo=%s case=%s arm=%d round=%d warmup_gpu_ms=%.1f\n",
                                    G.name, cs.name, arm, round, spent);
                    }
                    std::vector<cudaEvent_t> ev0(iters), ev1(iters);
                    for (int i = 0; i < iters; ++i) { ck(cudaEventCreate(&ev0[i])); ck(cudaEventCreate(&ev1[i])); }
                    for (int i = 0; i < iters; ++i) {
                        ck(cudaEventRecord(ev0[i], st)); body(); ck(cudaEventRecord(ev1[i], st));
                    }
                    ck(cudaStreamSynchronize(st));
                    for (int i = 0; i < iters; ++i) {
                        float ms; ck(cudaEventElapsedTime(&ms, ev0[i], ev1[i]));
                        samples.push_back({cs.name, G.name, cs.graph ? "graph" : "eager", arm, round, i, ms});
                        std::printf("BENCHWCSV %s,%s,%s,%s,%d,%d,%.6f\n", cs.name, G.name,
                                    cs.graph ? "graph" : "eager", arm ? "band8" : "native",
                                    round, i, ms);
                        ck(cudaEventDestroy(ev0[i])); ck(cudaEventDestroy(ev1[i]));
                    }
                    {   // per-round absolute means (the ratios' building blocks)
                        double sum = 0; int cnt = 0;
                        for (const auto& s : samples)
                            if (s.cs == cs.name && s.geo == G.name && s.arm == arm && s.round == round)
                                { sum += s.ms; ++cnt; }
                        std::printf("BENCHWROUND geo=%s case=%s arm=%s round=%d n=%d mean_ms=%.4f\n",
                                    G.name, cs.name, arm ? "band8" : "native", round, cnt,
                                    cnt ? sum / cnt : 0.0);
                    }
                }
            }
            if (gexec[0]) ck(cudaGraphExecDestroy(gexec[0]));
            if (gexec[1]) ck(cudaGraphExecDestroy(gexec[1]));
            double mean[2] = {0, 0}; int cnt[2] = {0, 0};
            std::vector<double> all[2];
            for (const auto& s : samples)
                if (s.cs == cs.name && s.geo == G.name) { mean[s.arm] += s.ms; cnt[s.arm]++; all[s.arm].push_back(s.ms); }
            for (int arm = 0; arm < 2; ++arm)
                if (cnt[arm])
                    std::printf("BENCHWAGG geo=%s case=%s arm=%s n=%d mean_ms=%.4f median_ms=%.4f min_ms=%.4f\n",
                                G.name, cs.name, arm ? "band8" : "native", cnt[arm],
                                mean[arm] / cnt[arm], bench_median(all[arm]),
                                *std::min_element(all[arm].begin(), all[arm].end()));
            if (cnt[0] && cnt[1])
                std::printf("BENCHWRATIO geo=%s case=%s band8/native=%.4f (means n=%d/%d)\n", G.name, cs.name,
                            (mean[1] / cnt[1]) / (mean[0] / cnt[0]), cnt[0], cnt[1]);
        }
        std::printf("BENCHWGEO geo=%s corpus_MiB=%.1f workspace_bytes=%zu bench_peak_MiB=%.1f\n",
                    G.name, corpus.corpus_words * 2.0 / 1048576, ws.allocated_bytes(),
                    (peak + ws.allocated_bytes()) / 1048576.0);
        ck(cudaGetLastError());
    }
    ck(cudaStreamDestroy(st));
    std::printf("BENCHWDONE samples=%zu bench_peak_MiB=%.1f\n", samples.size(), peak / 1048576.0);
    return 0;
}
#endif
int main(int argc, char** argv) {
    try {
        // The NORMAL-API suite runs as CHILD PROCESSES (one per dispatch-knob configuration)
        // because rdna_level()/the native flags are cached per process: a fresh exec makes each
        // pass's arm selection deterministic.  The parent spawns them before touching CUDA.
        if (argc > 1 && !std::strncmp(argv[1], "--normal-api-", 13)) {
#if defined(STRATA_WMMA_GFX11)
            const std::string mode = argv[1];
            if (mode == "--normal-api-rdna1") { ::setenv("STRATA_EXL3_RDNA", "1", 1); }
            if (mode == "--normal-api-fuse3") {
                ::setenv("STRATA_EXL3_RDNA", "1", 1); ::setenv("STRATA_EXL3_FUSE", "3", 1);
            }
            if (mode == "--normal-api-nativegrp") {
                ::setenv("STRATA_EXL3_RDNA", "1", 1);
                ::setenv("STRATA_EXL3_RDNA_NATIVE", "1", 1);
                ::setenv("STRATA_EXL3_NATIVE_GROUPED", "1", 1);
            }
            normal_api_suite(mode.c_str() + 13, mode == "--normal-api-fuse3");
            return 0;
#else
            (void) argv; return 77;
#endif
        }
        if (argc > 1 && !std::strcmp(argv[1], "--bench"))
#if defined(STRATA_WMMA_GFX11)
            return bench_main(argc, argv);
#else
            return 77;
#endif
        if (argc > 1 && !std::strcmp(argv[1], "--bench-wmma"))
#if defined(STRATA_WMMA_GFX11)
            return bench_wmma_main(argc, argv);
#else
            return 77;
#endif
        permutation();
#if !defined(STRATA_WMMA_GFX11)
        std::puts("native BAND8 GPU unsupported by this backend (host checks passed)");return 77;
#else
        // Spawn the four NORMAL-API dispatch passes (scalar / RDNA=1 / FUSE=3 / native-grouped)
        // as fresh processes - fork+exec keeps each child's cached env reads deterministic and
        // no CUDA state exists yet in the parent.  Each child must exit 0.
        for (const char* pass : {"--normal-api-scalar", "--normal-api-rdna1",
                                 "--normal-api-fuse3", "--normal-api-nativegrp"}) {
            pid_t pid = ::fork();
            if (pid == 0) { ::execl("/proc/self/exe", argv[0], pass, (char*) nullptr); _exit(127); }
            int status = 0;
            if (::waitpid(pid, &status, 0) != pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                std::fprintf(stderr, "FAIL: normal-API pass %s (status %d)\n", pass, status);
                return 1;
            }
        }
        std::puts("NORMAL API dispatch passes (scalar, rdna1, fuse3, native-grouped) PASS");
        ck(cudaSetDevice(0));hipDeviceProp_t prop{};ck(hipGetDeviceProperties(&prop,0));
        if(!gfx_arch_is_gfx11_wmma(prop.gcnArchName))return 77;
        std::printf("GPU0 %s %s\n",prop.name,prop.gcnArchName);
        std::mt19937 rng(20261106);
        for(int rate=0;rate<4;++rate) {
            const int b=rate<2?2:rate+1,f=rate==1,k=rate<2?4096:2560,ff=rate<2?2048:640;
            Mat g(k,ff,b,f,rng),u(k,ff,b,f,rng),d(ff,k,b,f,rng);
            size_t start=comparisons;
            for(int ps:{1,2,16}) {raw_all(g,ps);raw_all(d,ps);
                for(int width:{1,2,3,4})for(bool fused:{false,true})ffn(g,u,d,width,fused,ps,width==1 && !fused && ps==1);}
            Buf<uint16_t> wa((size_t)k*ff),wb((size_t)k*ff);
            exl3_reconstruct_weight(g.a.trellis,k/16,ff/16,b,2,g.a.suh,g.a.svh,wa.p,nullptr,f);
            require(exl3_experimental_band8_reconstruct(g.z,wb.p,nullptr),"reconstruct refused");
            equal(wa.get(),wb.get(),"whole reconstruction F16");
            std::printf("RATE %d+%d compared=%zu bitdiff=0 PASS\n",b,f,comparisons-start);
        }
        // ---- stage-2 prefill readers: WMMA group FFN, batched reader, recon batch ----
        {
            struct WGeo { const char* name; int k, ff, bg, fg, bu, fu, bd, fd; };
            static const WGeo wgeos[] = {
                {"mimo-k2",  4096, 2048, 2, 0, 2, 0, 2, 0},
                {"mimo-k25", 4096, 2048, 2, 1, 2, 1, 2, 1},
                {"mimo-mix", 4096, 2048, 2, 0, 2, 1, 2, 0},   // mixed projections 2/2.5
                {"qwen-k3",  2560, 640,  3, 0, 3, 0, 3, 0},   // nj=40: last 256-col band is a
                {"qwen-k4",  2560, 640,  4, 0, 4, 0, 4, 0},   // band-tail (band8 group 4)
            };
            for (const auto& wg : wgeos) {
                Mat g(wg.k,wg.ff,wg.bg,wg.fg,rng),u(wg.k,wg.ff,wg.bu,wg.fu,rng),d(wg.ff,wg.k,wg.bd,wg.fd,rng);
                const size_t start=comparisons;
                for(int rows:{16,24,32,64,128,512}) wmma_group(g,u,d,rows,false);
                wmma_group(g,u,d,64,true);   // matched STG-9 instantiation pair
                std::printf("WMMA GEO %s compared=%zu bitdiff=0 PASS\n",wg.name,comparisons-start);
            }
            {   // the batched (expert_batch) fragment reader in isolation; API stays untested
                Mat m2(4096,2048,2,0,rng),m25(4096,2048,2,1,rng),q3(2560,640,3,0,rng),q4(2560,640,4,0,rng);
                wmma_batched_all(m2,16,false);wmma_batched_all(m2,512,false);wmma_batched_all(m2,64,true);
                wmma_batched_all(m25,64,false);wmma_batched_all(q3,64,false);wmma_batched_all(q3,16,true);
                wmma_batched_all(q4,32,false);
                std::puts("WMMA batched reader (one expert) PASS - full expert_batch API on band8 UNTESTED");
            }
            {
                Mat g(4096,2048,2,0,rng),u(4096,2048,2,0,rng),d(2048,4096,2,0,rng);
                Mat g5(4096,2048,2,1,rng),u5(4096,2048,2,1,rng),d5(2048,4096,2,1,rng);
                Mat qg(2560,640,3,0,rng),qu(2560,640,3,0,rng),qd(640,2560,3,0,rng);
                recon_batch_parity(g,u,d,g,u,d,16,24);       // same-rate two-expert batch
                recon_batch_parity(g,u,d,g5,u5,d5,16,24);    // K2 + K2.5 mixed across experts
                recon_batch_parity(qg,qu,qd,qg,qu,qd,16,16); // qwen K3 band-tail geometry
            }
            {   // register/static-smem/local deltas of the Band8 instantiations
                hipFuncAttributes a{},b{};
                auto row=[&](const char* name,const void* n_,const void* b_){
                    ck(hipFuncGetAttributes(&a,n_));ck(hipFuncGetAttributes(&b,b_));
                    std::printf("RESOURCES %s native/band8 regs=%d/%d static_smem=%zu/%zu local=%zu/%zu\n",
                        name,a.numRegs,b.numRegs,a.sharedSizeBytes,b.sharedSizeBytes,a.localSizeBytes,b.localSizeBytes);};
                row("wmma K2 rt1",(const void*)routed_wmma_kernel<2,1,false,8,false>,(const void*)routed_wmma_kernel<2,1,false,8,true>);
                row("wmma K2 rt1 stg9",(const void*)routed_wmma_kernel<2,1,false,9,false>,(const void*)routed_wmma_kernel<2,1,false,9,true>);
                row("wmma K2 rt2",(const void*)routed_wmma_kernel<2,2,false,8,false>,(const void*)routed_wmma_kernel<2,2,false,8,true>);
                row("wmma K3 rt1",(const void*)routed_wmma_kernel<3,1,false,8,false>,(const void*)routed_wmma_kernel<3,1,false,8,true>);
                row("wmma K4 rt1",(const void*)routed_wmma_kernel<4,1,false,8,false>,(const void*)routed_wmma_kernel<4,1,false,8,true>);
                row("wmma K2 rt1 batched",(const void*)routed_wmma_kernel<2,1,true,8,false>,(const void*)routed_wmma_kernel<2,1,true,8,true>);
                row("wmma K2 rt2 batched",(const void*)routed_wmma_kernel<2,2,true,8,false>,(const void*)routed_wmma_kernel<2,2,true,8,true>);
                row("recon tile",(const void*)recon_tile_kernel<false>,(const void*)recon_tile_kernel<true>);
            }
        }
        {Mat g(256,640,2,0,rng),u(256,640,2,1,rng),d(640,256,4,0,rng);
            ffn(g,u,d,3,false,2,true,true);ffn(g,u,d,3,true,2,true,true);
            Exl3MoeWorkspace w;int first=0,rows=1;auto bad=g.z;bad.nj=9;
            require(!exl3_experimental_band8_reconstruct(bad,(uint16_t*)4,nullptr),"invalid reconstruction accepted");
            require(!exl3_experimental_band8_rows({&bad,&u.z,&d.z},&first,&rows,1,1,(uint16_t*)4,(int*)4,(uint16_t*)4,nullptr,w),"invalid nj accepted");
            require(!exl3_experimental_band8_rows({&g.z,&u.z,&d.z},&first,&rows,1,1,(uint16_t*)4,(int*)4,(uint16_t*)4,nullptr,w),"unreserved workspace accepted");
            w.reserve_small_m(1,1,4096,4096,256);
            for(int probe=0;probe<9;++probe) {
                auto m=g.z;int rf=0,rr=1,ps=0;
                if(probe==0)m.bits=5;if(probe==1){m.bits=3;m.frac=1;}
                if(probe==2)m.frac=2;if(probe==3)m.cb=1;if(probe==4)m.ki=0;
                if(probe==5)rf=1;if(probe==6)rr=5;if(probe==7)ps=17;if(probe==8)m.trellis=nullptr;
                require(!exl3_experimental_band8_rows({&m,&u.z,&d.z},&rf,&rr,1,1,(uint16_t*)4,(int*)4,(uint16_t*)4,nullptr,w,false,ps),"negative eligibility accepted");
            }
            ck(cudaGetLastError());std::puts("negative nj/unreserved/rate/frac/cb/dims/group/parts/null probes=12 PASS");}
        {   // stage-2 negative eligibility: everything refused BEFORE enqueue, no allocations
            Mat g(256,640,2,0,rng),u(256,640,2,0,rng),d(640,256,2,0,rng);
            const int k=256,ff=640,n=256,tokens=5;
            std::vector<uint16_t> hx((size_t)tokens*k);
            for(size_t i=0;i<hx.size();++i)hx[i]=bench_half(((int)(i%91)-45)*0.001f);
            Buf<uint16_t> x(hx.size());x.put(hx);
            std::vector<int> hm(16);for(int i=0;i<16;++i)hm[i]=i%tokens;
            Buf<int> map(16);map.put(hm);
            Buf<float> y((size_t)16*n);
            Exl3MoeWorkspace w;w.reserve(16,k,ff,n);w.reserve_group_scratch(16,k,ff,n,true);
            require(exl3_experimental_band8_group_wmma(g.z,u.z,d.z,16,x.p,map.p,y.p,nullptr,w),
                "band8 WMMA baseline refused");
            const size_t before=w.allocated_bytes();
            for(int rows:{1,8,15,513,1024})
                require(!exl3_experimental_band8_group_wmma(g.z,u.z,d.z,rows,x.p,map.p,y.p,nullptr,w),
                    "rows range accepted");
            {Exl3MoeWorkspace fresh;   // unreserved: refused, never grown, nothing enqueued
                require(!exl3_experimental_band8_group_wmma(g.z,u.z,d.z,16,x.p,map.p,y.p,nullptr,fresh),
                    "unreserved wmma workspace accepted");}
            auto bad=[&](const Exl3Mat& gg,const Exl3Mat& uu,const Exl3Mat& dd){
                require(!exl3_experimental_band8_group_wmma(gg,uu,dd,16,x.p,map.p,y.p,nullptr,w),
                    "ineligible wmma descriptor accepted");};
            {auto m=g.z;m.bits=5;bad(m,u.z,d.z);}
            {auto m=g.z;m.bits=3;m.frac=1;bad(m,u.z,d.z);}
            {auto m=g.z;m.frac=2;bad(m,u.z,d.z);}
            {auto m=g.z;m.cb=1;bad(m,u.z,d.z);}
            {auto m=g.z;m.ki=9;bad(m,u.z,d.z);}
            {auto m=g.z;m.nj=17;bad(m,u.z,d.z);}
            {auto m=g.z;m.trellis=nullptr;bad(m,u.z,d.z);}
            {auto m=d.z;m.ki=g.z.nj+1;bad(g.z,u.z,m);}
            require(before==w.allocated_bytes(),"wmma negative probes allocated");
            Exl3Group gb[2]={{g.z,u.z,d.z,0,8},{g.z,u.z,d.z,8,8}};
            {Exl3MoeWorkspace fresh;   // unreserved: the native flow's no-growth assert throws
                bool threw=false;
                try{exl3_experimental_band8_recon_batch(gb,2,x.p,map.p,map.p,y.p,nullptr,fresh);}
                catch(const std::runtime_error&){threw=true;}
                require(threw,"unreserved recon workspace accepted");}
            Exl3MoeWorkspace rw;rw.reserve_recon_batch(2,16,k,ff,n);
            require(!exl3_experimental_band8_recon_batch(gb,1,x.p,map.p,map.p,y.p,nullptr,rw),"recon count=1 accepted");
            {auto bb=gb[0];bb.gate.bits=6;Exl3Group gb2[2]={bb,gb[1]};
                require(!exl3_experimental_band8_recon_batch(gb2,2,x.p,map.p,map.p,y.p,nullptr,rw),"recon bad rate accepted");}
            ck(cudaGetLastError());
            // the OLD entry keeps its rows range and native-layout contract (rows=8 accepted)
            require(exl3_moe_group_wmma(g.a,u.a,d.a,8,x.p,map.p,y.p,nullptr,w),"old WMMA rows=8 refused");
            std::puts("negative wmma rows/workspace/rate/frac/cb/dims/relation + recon workspace/count/rate probes PASS"
                      " (old WMMA rows=8 still accepted)");}
        ck(cudaDeviceSynchronize());
        std::printf("TOTAL compared=%zu bitdiff=0 tracked_peak_device_bytes=%zu (reconstruct temporary adds 6*k*n bytes) PASS\n",comparisons,peak);
        return 0;
#endif
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
