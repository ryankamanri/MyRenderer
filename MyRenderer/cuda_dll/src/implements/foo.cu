#include <cuda_runtime.h>
#include "cuda_dll/src/foo.cuh"
#include "cuda_dll/src/utils/cuda_thread_config.cuh"

// #include "kamanri/renderer/world/world3d.hpp"


#define CHECK(res) { if(res != cudaSuccess){printf("Error ：%s:%d , ", __FILE__,__LINE__);   \
printf("code : %d , reason : %s \n", res,cudaGetErrorString(res));exit(-1);}}

#ifdef __CUDA_RUNTIME_H__
__global__ 
#endif
void foo(int* a)
{
	printf("CUDA!\n");
	int index = thread_index;
	printf("a[%d] = %d at %p \n", index, a[index], a + index);
	a[index]++;
	printf("a[%d] = %d at %p \n", index, a[index], a + index);

	print_thread_info;
}

__global__ void Foo2(TestStruct t)
{
	printf("a = %d, b = %f at %p \n", t.a, t.b, &t);
	
	print_thread_info;
}


void UseCUDA()
{
	int* a;
	int b[10];
	
	cudaMalloc(&a, 10 * sizeof(int));
	foo thread_num(5) (a);
	printf("a at %p\n", a);
	cudaMemcpy(b, a, sizeof(int) * 10, cudaMemcpyDeviceToHost);
	
	cudaFree(a);
	for(size_t i = 0; i < 10; i++)
	{
		printf("b[%llu] = %d at %p \n", i, b[i], b + i);
	}
	CHECK(cudaDeviceSynchronize());
}


void UseCUDA2(int block_num, int thread_num, TestStruct t)
{
	Foo2<<<block_num, thread_num>>>(t);
	CHECK(cudaDeviceSynchronize());
}

void MemoryReadTest(int* array, size_t size)
{
	for(size_t i = 0; i < size; i++)
	{
		printf("a[%llu] = %d\n", i, array[i]);
	}
}
