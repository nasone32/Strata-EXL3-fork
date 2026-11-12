#pragma once
#include "strata/artifact/exl3_resident.hpp"
#include "strata/core/expert_source.hpp"
#include <cuda_runtime.h>

namespace strata::kernels {
// Use the SAME available-RAM/cgroup calculation as the GGUF arena. Paced
// registration commits one expert at a time; do not silently become pageable.
inline Exl3ResidentBacking::Options exl3_resident_host_options() {
    strata::core::detail::HostMemory memory;
    if (!strata::core::detail::host_available_memory(memory))
        throw std::runtime_error("resident EXL3: cannot determine available RAM/cgroup limit");
    Exl3ResidentBacking::Options opt;
    opt.available_bytes = memory.available;
    opt.lock = [](void* p,size_t n,std::string& error) {
        strata::core::detail::HostMemory hm;
        if (!strata::core::detail::host_available_memory(hm) || hm.available < n + (4ull<<30)) {
            error = "available RAM fell below 4 GiB headroom during paced registration"; return false;
        }
        const auto rc = cudaHostRegister(p,n,cudaHostRegisterPortable);
        if (rc == cudaSuccess) return true;
        error = cudaGetErrorString(rc); (void)cudaGetLastError(); return false;
    };
    opt.unlock = [](void* p,size_t) { (void)cudaHostUnregister(p); };
    // This registrar really GPU-registers every committed span, so the backing may answer
    // direct_dma_pinned() for its own bytes (the R1 prototype's opt-in transport reads the
    // arena without the gather copy). A test's fake registrar must NOT set this.
    opt.gpu_registered = true;
    return opt;
}
} // namespace strata::kernels
