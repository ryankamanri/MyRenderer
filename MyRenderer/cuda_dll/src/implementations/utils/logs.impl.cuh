#pragma once

template <typename... Ts>
__device__ inline int DevicePrint(const char* formatStr, Ts... argv)
{
	int retCode = printf(formatStr, argv...);
	return retCode;
}