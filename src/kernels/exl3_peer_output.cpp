#include "strata/kernels/exl3_peer.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <stdexcept>

namespace strata::kernels {
struct Exl3PeerOutput::Impl {
    int primary = -1, peer = -1, embd = 0, capacity = 0;
    Exl3PeerOutputRecord *host = nullptr, *device = nullptr, *peer_device = nullptr;
    uint16_t *rows = nullptr, *device_rows = nullptr;
    cudaEvent_t done = nullptr;
    bool queued = false;
    bool p2p = false;
    bool kernel_transfer = false;
    int* arrivals = nullptr;
    std::atomic<bool> released{false};
    ~Impl() {
        int previous = 0; (void) cudaGetDevice(&previous);
        if (peer >= 0 && cudaSetDevice(peer) == cudaSuccess && done) {
            if (queued) (void) cudaEventSynchronize(done);
            if(arrivals) (void)cudaFree(arrivals);
            (void) cudaEventDestroy(done);
        }
        if (primary >= 0 && cudaSetDevice(primary) == cudaSuccess) {
            if(p2p && device_rows) (void)cudaFree(device_rows);
            if (rows) (void) cudaFreeHost(rows);
            if (host) (void) cudaFreeHost(host);
        }
        (void) cudaSetDevice(previous);
    }
};
Exl3PeerOutput::Exl3PeerOutput() = default;
Exl3PeerOutput::~Exl3PeerOutput() = default;

bool Exl3PeerOutput::open(int primary, int peer, int embd, int capacity, std::string& err, bool p2p,bool kernel_transfer) {
    if (impl_ || primary < 0 || peer < 0 || primary == peer || embd <= 0 || embd%8 ||
        capacity < 1 || capacity > 512 || (kernel_transfer && !p2p)) {
        err = "exl3 peer output: invalid setup"; return false;
    }
    int previous = 0; (void) cudaGetDevice(&previous);
    struct Restore { int d; ~Restore(){(void)cudaSetDevice(d);} } restore{previous};
    auto m = std::make_unique<Impl>();
    m->primary=primary; m->peer=peer; m->embd=embd; m->capacity=capacity;
    m->p2p=p2p;
    m->kernel_transfer=kernel_transfer;
    auto ck = [&](cudaError_t e) { if (e == cudaSuccess) return true; err=cudaGetErrorString(e); return false; };
    const unsigned flags=cudaHostAllocPortable|cudaHostAllocMapped;
    if (!ck(cudaSetDevice(primary)) ||
        !ck(cudaHostAlloc((void**)&m->host,sizeof(Exl3PeerOutputRecord),flags)) ||
        !ck(cudaHostGetDevicePointer((void**)&m->device,m->host,0)) ||
        !ck(p2p ? cudaMalloc((void**)&m->device_rows,(size_t)capacity*embd*sizeof(uint16_t)) :
                  cudaHostAlloc((void**)&m->rows,(size_t)capacity*embd*sizeof(uint16_t),flags)) ||
        (!p2p && !ck(cudaHostGetDevicePointer((void**)&m->device_rows,m->rows,0))) ||
        !ck(cudaSetDevice(peer)) ||
        !ck(cudaHostGetDevicePointer((void**)&m->peer_device,m->host,0)) ||
        !ck(cudaEventCreateWithFlags(&m->done,cudaEventDisableTiming))) return false;
    if(p2p) {
        int access=0;
        if(!ck(cudaDeviceCanAccessPeer(&access,peer,primary)) || !access) {err="exl3 peer output: P2P unavailable";return false;}
        auto e=cudaDeviceEnablePeerAccess(primary,0);
        if(e==cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();
        else if(!ck(e)) return false;
    }
    if(kernel_transfer && (!ck(cudaMalloc((void**)&m->arrivals,sizeof(int))) ||
        !ck(cudaMemset(m->arrivals,0,sizeof(int))) || !ck(cudaStreamSynchronize(nullptr)))) return false;
    *m->host=Exl3PeerOutputRecord{};
    std::fill_n(m->host->rank,512,-1);
    impl_=std::move(m);
    return true;
}

bool Exl3PeerOutput::enqueue(const uint16_t* rows, const int* destinations, int count, void* stream, std::string& err) {
    if (!impl_ || !rows || !destinations || count<1 || count>impl_->capacity) {
        err="exl3 peer output: invalid frame"; return false;
    }
    auto& m=*impl_;
    auto* record=m.host;
    if (m.released.load() || *(volatile uint32_t*)&record->ready==UINT32_MAX || record->expected==UINT32_MAX-1 ||
        *(volatile uint32_t*)&record->ready<record->expected) {
        err="exl3 peer output: prior frame incomplete or released"; return false;
    }
    for (int i=0;i<count;++i) {
        if (destinations[i]<0 || destinations[i]>=m.capacity) {err="exl3 peer output: invalid destination";return false;}
        for(int j=0;j<i;++j) if(destinations[j]==destinations[i]) {err="exl3 peer output: duplicate destination";return false;}
    }
    std::fill_n(record->rank,512,-1);
    for(int i=0;i<count;++i) record->rank[destinations[i]]=i;
    ++record->expected;
    std::atomic_thread_fence(std::memory_order_release);
    const auto st=(cudaStream_t)stream;
    const size_t bytes=(size_t)count*m.embd*sizeof(uint16_t);
    cudaError_t e=cudaSuccess;
    if(m.kernel_transfer) {
        exl3_peer_output_transfer(rows,m.device_rows,(int64_t)count*m.embd,&m.peer_device->ready,
                                  record->expected,m.arrivals,stream);
        e=cudaGetLastError();
    } else e=m.p2p ? cudaMemcpyPeerAsync(m.device_rows,m.primary,rows,m.peer,bytes,st) :
                     cudaMemcpyAsync(m.rows,rows,bytes,cudaMemcpyDeviceToHost,st);
    if(e==cudaSuccess && !m.kernel_transfer) {
        exl3_peer_output_signal(&m.peer_device->ready,record->expected,stream);
        e=cudaGetLastError();
    }
    if(e==cudaSuccess) e=cudaEventRecord(m.done,st);
    if(e!=cudaSuccess) {
        // Failure-only drain: never free a frame under a partially queued DMA.
        (void)cudaStreamSynchronize(st);
        err=cudaGetErrorString(e); return false;
    }
    m.queued=true;
    return true;
}

void Exl3PeerOutput::empty() {
    if (!impl_) return;
    auto& r=*impl_->host;
    if(impl_->released.load() || r.expected>=UINT32_MAX-1) {release();return;}
    std::fill_n(r.rank,512,-1);
    if (*(volatile uint32_t*)&r.ready==UINT32_MAX) return;
    ++r.expected;
    std::atomic_thread_fence(std::memory_order_release);
    *(volatile uint32_t*)&r.ready=r.expected;
}
void Exl3PeerOutput::copy(float* parts,int routes,const uint32_t* skip,uint32_t ring,void* stream) {
    if(!impl_ || routes<1 || routes>impl_->capacity) throw std::runtime_error("exl3 peer output: invalid capture");
    exl3_peer_output_copy(impl_->device,impl_->device_rows,parts,routes,impl_->embd,skip,ring,stream);
}
void Exl3PeerOutput::release() {
    if(impl_ && !impl_->released.exchange(true)) *(volatile uint32_t*)&impl_->host->aborted=1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}
} // namespace strata::kernels
