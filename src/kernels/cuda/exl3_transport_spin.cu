// src/kernels/cuda/exl3_transport_spin.cu - the doorbell-style PCIe poll for the NON-PRODUCTION
// transport bench (src/kernels/exl3_transport_bench.cpp).  The engine's verifier graph spins on
// a mapped host flag (wait_flag_ge) while the staging DMA runs; this reproduces that poll on a
// chosen stream so the harness can isolate PCIe-poll-vs-DMA contention.  The .cu suffix keeps it
// on the GPU language path (HIP here, CUDA elsewhere); the .cpp harness cannot launch kernels.
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__global__ void spin_poll_kernel(volatile uint32_t* flag, uint32_t want) {
    while (*flag < want) {   // mapped host read in a tight loop, like the graph's wait kernel
    }
}

}  // namespace

void exl3_transport_spin_poll(void* device_flag, unsigned want, void* stream) {
    spin_poll_kernel<<<1, 1, 0, (cudaStream_t) stream>>>((volatile uint32_t*) device_flag, want);
}
