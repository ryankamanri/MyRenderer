#pragma once

namespace __CUDAThreadConfig
{
    unsigned int ThreadNumPerBlock(unsigned int num);

    unsigned int BlockNum(unsigned int num);

} // namespace __CUDAThreadConfig


#define thread_num(num) <<<__CUDAThreadConfig::BlockNum(num), __CUDAThreadConfig::ThreadNumPerBlock(num)>>>
#define thread_index (blockIdx.x * blockDim.x + threadIdx.x)

#define print_thread_info printf( \
    "threadIdx = (%u, %u, %u); blockIdx = (%u, %u, %u); blockDim = (%u, %u, %u); gridDim = (%u, %u, %u); thread_index = %u \n", \
    threadIdx.x, threadIdx.y, threadIdx.z, \
    blockIdx.x, blockIdx.y, blockIdx.z, \
    blockDim.x, blockDim.y, blockDim.z, \
    gridDim.x, gridDim.y, gridDim.z, \
    thread_index);