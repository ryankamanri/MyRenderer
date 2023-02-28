#include "cuda_dll/src/memory_operations.cuh"
#include "cuda_dll/src/utils/cuda_error_check.cuh"
#include <cuda_runtime.h>

namespace __MemoryOperations
{
    constexpr const char* LOG_NAME = STR(MemoryOperations);
} // namespace __MemoryOperations


MemoryOperationsCode CUDAMalloc(void** out_p, size_t size)
{
    auto res = cudaMalloc(out_p, size);
    CUDA_ERROR_CHECK(res, __MemoryOperations::LOG_NAME);
    return (MemoryOperationsCode)res;
}

MemoryOperationsCode CUDAFree (void* p)
{
    auto res = cudaFree(p);
    CUDA_ERROR_CHECK(res, __MemoryOperations::LOG_NAME);
    return (MemoryOperationsCode)res;
}

MemoryOperationsCode TransmitToCUDA (void* host_p, void* cuda_p, size_t size)
{
    auto res = cudaMemcpy(cuda_p, host_p, size, cudaMemcpyHostToDevice);
    CUDA_ERROR_CHECK(res, __MemoryOperations::LOG_NAME);
    return (MemoryOperationsCode)res;
}

MemoryOperationsCode TransmitFromCUDA (void* host_p, void* cuda_p, size_t size)
{
    auto res = cudaMemcpy(host_p, cuda_p, size, cudaMemcpyDeviceToHost);
    CUDA_ERROR_CHECK(res, __MemoryOperations::LOG_NAME);
    return (MemoryOperationsCode)res;
}