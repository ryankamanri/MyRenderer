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

		template <typename T>
		P<T> Copy(T* p)
		{
			T* new_p = (T*)malloc(sizeof(T));
			(*new_p) = (*p);
			return P<T>(new_p);
		}

		template <typename T>
		P<T[]> CopyArray(T* p, size_t size)
		{
			T* new_p = (T*)malloc(sizeof(T) * size);
			T* new_p_ = new_p;
			for(size_t i = 0; i < size; i++)
			{
				(*new_p++) = (*p++);
			}

			return P<T[]>(new_p_);
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
