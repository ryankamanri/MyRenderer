#ifndef FOO_CUH
#define FOO_CUH

#include <stdio.h>
#include "cuda_dll/foo.hpp"


// declared any function if other sorece code needs.
c_export void UseCUDA();
c_export void UseCUDA2(int block_num, int thread_num, TestStruct t);
c_export void MemoryReadTest(int* array, size_t size);



#endif