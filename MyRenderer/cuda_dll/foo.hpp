#pragma once
#include "kamanri/utils/imexport.hpp"

// define struct/object as type of param or return value
class TestStruct
{
	public:
	int a;
	double b;
};


// define funcion pointer with format:
// typedef [return_type] func_p([func]) ([parameters...])
typedef void func_p(UseCUDA) ();
typedef void func_p(UseCUDA2) (int, int, TestStruct);
typedef void func_p(MemoryReadTest) (int*, size_t);