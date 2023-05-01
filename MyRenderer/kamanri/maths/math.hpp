#pragma once

#define DBL_MAX          1.7976931348623158e+308 // max value

namespace Kamanri
{
	namespace Maths
	{
		constexpr double PI = 3.14159265358979323846;

#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif
		inline double Min(double v1, double v2)
		{
			return (v1 < v2) ? v1 : v2;
		}

#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif
		inline double Max(double v1, double v2)
		{
			return (v1 > v2) ? v1 : v2;
		}
	}
}