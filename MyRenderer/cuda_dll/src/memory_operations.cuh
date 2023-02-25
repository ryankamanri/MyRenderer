#pragma once
#include "cuda_dll/exports/memory_operations.hpp"

c_export MemoryOperationsCode CUDAMalloc (void** out_p, size_t size);
c_export MemoryOperationsCode CUDAFree (void* p);
c_export MemoryOperationsCode TransmitToCUDA (void* host_p, void* cuda_p, size_t size);
c_export MemoryOperationsCode TransmitFromCUDA (void* host_p, void* cuda_p, size_t size);