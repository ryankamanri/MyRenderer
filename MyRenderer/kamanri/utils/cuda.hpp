#pragma once
#include <cuda_runtime.h>

#define cuda #ifdef 

// cuda error check
/////////////////////////////////////////////////////////////////
#include "kamanri/utils/logs.hpp"

#define CUDA_ERROR_CHECK(res, log_name) \
{ \
	if (res != cudaSuccess) \
	{ \
		Log::Error(log_name, "CUDA ERROR:\n\t Code : %d\n\t Reason : %s\n", res, cudaGetErrorString(res)); \
		PRINT_LOCATION; \
		exit(-1); \
	} \
}

// cuda thread config
/////////////////////////////////////////////////////////////////////
namespace __CUDAThreadConfig
{
    unsigned int ThreadNumPerBlock(unsigned int num);

    unsigned int BlockNum(unsigned int num);

} // namespace __CUDAThreadConfig


#define thread_num(num) <<<__CUDAThreadConfig::BlockNum(num), __CUDAThreadConfig::ThreadNumPerBlock(num)>>>
#define thread_index (blockIdx.x * blockDim.x + threadIdx.x)

#define print_thread_info printf("threadIdx = (%u, %u, %u); blockIdx = (%u, %u, %u); blockDim = (%u, %u, %u); gridDim = (%u, %u, %u) \n", \
    threadIdx.x, threadIdx.y, threadIdx.z, \
    blockIdx.x, blockIdx.y, blockIdx.z, \
    blockDim.x, blockDim.y, blockDim.z, \
    gridDim.x, gridDim.y, gridDim.z);


