#pragma once
#include "kamanri/utils/logs.hpp"

#define CUDA_ERROR_CHECK(res, log_name) \
	if (res != cudaSuccess) \
	{ \
		Kamanri::Utils::Log::Error(log_name, "CUDA ERROR:\n\t Code : %d\n\t Reason : %s\n", res, cudaGetErrorString(res)); \
		PRINT_LOCATION; \
	} \
