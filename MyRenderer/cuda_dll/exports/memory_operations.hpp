#pragma once
#include "kamanri/utils/imexport.hpp"

typedef unsigned int MemoryOperationsCode;
namespace MemoryOperations$
{
    constexpr const MemoryOperationsCode CODE_NORM = 0;
} // namespace MemoryOperations$

// export functions types
typedef MemoryOperationsCode func_p(CUDAMalloc) (void** out_p, size_t size);
typedef MemoryOperationsCode func_p(CUDAFree) (void* p);
typedef MemoryOperationsCode func_p(TransmitToCUDA) (void* host_p, void* cuda_p, size_t size);
typedef MemoryOperationsCode func_p(TransmitFromCUDA) (void* host_p, void* cuda_p, size_t size);
