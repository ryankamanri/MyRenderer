#pragma once
#include <memory>

namespace Kamanri
{
	namespace Utils
	{

		using byte = unsigned char;

		template <typename T>
		using P = std::unique_ptr<T>;

		template <typename T, typename... Ts>
		P<T> New(Ts &&...args)
		{
			return P<T>(new T(std::forward<Ts>(args)...));
		}

		template <typename T>
		P<T[]> NewArray(size_t size)
		{
			return P<T[]>(new T[size]);
		}


	}
}

#define CHECK_MEMORY_IS_ALLOCATED(p, log_name, return_value)                           \
	if (!p)                                                                            \
	{                                                                                  \
		Kamanri::Utils::Log::Error(log_name, "Try to visit the not-allocated memory."); \
		PRINT_LOCATION;                                                                 \
		return return_value;                                                           \
	}
