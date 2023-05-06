#pragma once
#include "kamanri/utils/log.hpp"

#define CUDA_ERROR_CHECK(res, log_name) \
	if (res != cudaSuccess) \
	{ \
		Kamanri::Utils::PrintLn("[%s]: CUDA ERROR:\n\t Code : %d\n\t Reason : %s\n", log_name, res, cudaGetErrorString(res)); \
		PRINT_LOCATION; \
	} \
