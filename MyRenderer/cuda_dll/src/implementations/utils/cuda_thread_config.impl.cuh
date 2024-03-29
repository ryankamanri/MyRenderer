#pragma once
#include "kamanri/utils/log.hpp"
#include "kamanri/utils/string.hpp"
#include "cuda_dll/src/utils/cuda_thread_config.cuh"

namespace __CUDAThreadConfig
{
	using namespace Kamanri::Utils;

	constexpr const char* LOG_NAME = STR(CUDAThreadConfig);
	constexpr unsigned int MAX_BLOCK_NUM = 0x7fffffff;
	constexpr unsigned int MAX_THREAD_NUM_PER_BLOCK = 0x400;
	constexpr unsigned int DEFAULT_THREAD_NUM_PER_BLOCK = 0x80;

	unsigned int ThreadNumPerBlock(unsigned int num)
	{
		if (num > MAX_BLOCK_NUM * MAX_THREAD_NUM_PER_BLOCK)
		{
			PrintLn("[%s]: num > MAX_BLOCK_NUM * MAX_THREAD_NUM_PER_BLOCK", LOG_NAME);
			PRINT_LOCATION;
			return 0;
		}
		if(num < DEFAULT_THREAD_NUM_PER_BLOCK) return num;
		unsigned int res = DEFAULT_THREAD_NUM_PER_BLOCK;
		while (res * MAX_BLOCK_NUM < num)
		{
			res *= 2;
		}
		return res;
		
	}

	unsigned int BlockNum(unsigned int num)
	{
		unsigned int thread_num_per_block = ThreadNumPerBlock(num);
		if(!thread_num_per_block) 
		{
			PrintLn("[%s]: Invalid thread_num_per_block value 0.", LOG_NAME);
			PRINT_LOCATION;
			return 0;
		}
		auto res = (num % thread_num_per_block == 0) ? 
		(num / thread_num_per_block) :
		((num / thread_num_per_block) + 1);
		return res;
	}
} // namespace __CUDAThreadConfig