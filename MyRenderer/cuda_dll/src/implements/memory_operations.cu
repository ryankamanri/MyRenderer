#include "cuda_dll/src/memory_operations.cuh"
#include <cuda_runtime.h>

MemoryOperationsCode CUDAMalloc(void** out_p, size_t size)
{
    return (MemoryOperationsCode)cudaMalloc(out_p, size);
}

MemoryOperationsCode CUDAFree (void* p)
{
    return (MemoryOperationsCode)cudaFree(p);
}

MemoryOperationsCode TransmitToCUDA (void* host_p, void* cuda_p, size_t size)
{
    return (MemoryOperationsCode)cudaMemcpy(cuda_p, host_p, size, cudaMemcpyHostToDevice);
}

MemoryOperationsCode TransmitFromCUDA (void* host_p, void* cuda_p, size_t size)
{
    return (MemoryOperationsCode)cudaMemcpy(host_p, cuda_p, size, cudaMemcpyDeviceToHost);
}