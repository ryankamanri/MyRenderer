#pragma once
#include "kamanri/utils/logs.hpp"

//数组实现的栈，能存储任意类型的数据
namespace Kamanri
{
	namespace Utils
	{
		template<class T, size_t size = 10000>
		class ArrayStack
		{
			public:
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			void Push(T t)
			{
				if (IsFull()) 
				{
					Warn();
					return;
				}
				a[count++] = t;
			}
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			T GetTop() 
			{ 
				if (IsEmpty()) 
				{
					Warn();
					return a[count];
				}
				return a[count - 1]; 
			}
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			T Pop()
			{
				if (IsEmpty()) 
				{
					Warn();
					return a[count];
				}
				return a[--count];
			}
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			size_t Count() { return count; }
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			bool IsFull() { return count == size - 1; }
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			bool IsEmpty() { return count == 0; }
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			void Clean() { while(!IsEmpty()) Pop(); }
			private:
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			void Warn() { Kamanri::Utils::PrintLn("Invalid Stack Operation, size = %d, count = %d", size, count); }
			T a[size];//数组？
			size_t count = 0;
		};
	} // namespace Utils

} // namespace Kamanri


