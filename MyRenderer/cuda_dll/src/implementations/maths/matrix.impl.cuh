#pragma once
#include "kamanri/maths/smatrix.hpp"
#include "cuda_dll/src/utils/log.cuh"


namespace __SMatrix
{

	/// @brief (-1)^RON
	/// @param v 
	/// @return 
	__device__ int Pow_NegativeOne_ReverseOrderNumber(size_t* list, size_t list_size)
	{
		int res = 1;
		for (size_t i = 1; i < list_size; i++)
		{
			for (size_t j = 0; j < i; j++)
			{
				if (list[j] > list[i]) res *= -1;
			}
		}
		return res;
	}


	__device__ inline Kamanri::Maths::SMatrixElemType SMGet(Kamanri::Maths::SMatrixElemType* sm, size_t n, size_t row, size_t col)
	{
		return sm[n * row + col];
	}

	// Square Matrix Determinant [dimension + 1]
	__device__ inline Kamanri::Maths::SMatrixElemType SMDet2(Kamanri::Maths::SMatrixElemType* sm, size_t n, size_t row_1, size_t row_2, size_t col_1, size_t col_2)
	{
		return (SMGet(sm, n, row_1, col_1) * SMGet(sm, n, row_2, col_2) - SMGet(sm, n, row_1, col_2) * SMGet(sm, n, row_2, col_1));
	}

	// Square Matrix Determinant [dimension + 1]
	__device__ inline Kamanri::Maths::SMatrixElemType SMDet3(Kamanri::Maths::SMatrixElemType* sm, size_t n, size_t row_1, size_t row_2, size_t row_3, size_t col_1, size_t col_2, size_t col_3)
	{
		return (SMGet(sm, n, row_1, col_1) * SMDet2(sm, n, row_2, row_3, col_2, col_3) - SMGet(sm, n, row_2, col_1) * SMDet2(sm, n, row_1, row_3, col_2, col_3) + SMGet(sm, n, row_3, col_1) * SMDet2(sm, n, row_1, row_2, col_2, col_3));
	}

	__device__ Kamanri::Maths::SMatrixElemType __Determinant(Kamanri::Maths::SMatrixElemType* psm, size_t* row_list, size_t* col_list, size_t row_count)
	{
		using namespace __SMatrix;
		Kamanri::Maths::SMatrixElemType result = 0;

		if (row_count < 4)
		{
			if (row_count == 1) result = SMGet(psm, row_count, row_list[0], col_list[0]);
			if (row_count == 2) result = SMDet2(psm, row_count, row_list[0], row_list[1], col_list[0], col_list[1]);
			if (row_count == 3) result = SMDet3(psm, row_count, row_list[0], row_list[1], row_list[2], col_list[0], col_list[1], col_list[2]);
			if (row_count == 4)
			{
				result = SMGet(psm, row_count, row_list[0], col_list[0]) * SMDet3(psm, row_count, 1, 2, 3, 1, 2, 3) -
					SMGet(psm, row_count, row_list[1], col_list[0]) * SMDet3(psm, row_count, row_list[0], row_list[2], row_list[3], col_list[1], col_list[2], col_list[3]) +
					SMGet(psm, row_count, row_list[2], col_list[0]) * SMDet3(psm, row_count, row_list[0], row_list[1], row_list[3], col_list[1], col_list[2], col_list[3]) -
					SMGet(psm, row_count, row_list[3], col_list[0]) * SMDet3(psm, row_count, row_list[0], row_list[1], row_list[2], col_list[1], col_list[2], col_list[3]);
			}
			// This is to avoid the bigger index is in front of the smaller
			result *= __SMatrix::Pow_NegativeOne_ReverseOrderNumber(row_list, row_count);
			result *= __SMatrix::Pow_NegativeOne_ReverseOrderNumber(col_list, row_count);

			return result;
		}
	}
}

__device__ Kamanri::Maths::SMatrix::SMatrix()
{
	this->_N = SMatrix$::MAX_SUPPORTED_DIMENSION;
}

__device__ Kamanri::Maths::SMatrix::SMatrix(size_t n)
{
	this->_N = SMatrix$::NOT_INITIALIZED_N;

	if (n != 2 && n != 3 && n != 4)
	{
		Kamanri::Utils::PrintLn("The size of initializer list is not valid: %d", (int)n);
		PRINT_LOCATION;
		return;
	}

	this->_N = n;
}

__device__ Kamanri::Maths::SMatrix::SMatrix(SMatrix const& sm)
{
	_N = sm._N;
	auto size = _N * _N;

	for(size_t i = 0; i < size; i++)
	{
		_SM[i] = sm._SM[i];
	}
}

__device__ Kamanri::Maths::SMatrix::SMatrix(std::initializer_list<Kamanri::Maths::SMatrixElemType> list)
{
	this->_N = SMatrix$::NOT_INITIALIZED_N;

	auto size = list.size();
	if (size != 4 && size != 9 && size != 16)
	{
		Kamanri::Utils::PrintLn("The size of initializer list is not valid: %d", (int) size);
		return;
	}

	this->_N = (size_t) sqrt((double) size);

	auto i = 0;
	for (auto list_elem : list)
	{
		_SM[i] = list_elem;
		i++;
	}
}

__device__ Kamanri::Maths::SMatrixCode Kamanri::Maths::SMatrix::operator=(std::initializer_list<SMatrixElemType> list)
{
	auto size = list.size();
	if(size != _N * _N)
	{
		Kamanri::Utils::PrintLn("The size of initializer list(%d) is not equal to SMatrix(%d)", size, _N * _N);
		PRINT_LOCATION;
		return SMatrix$::CODE_NOT_EQUEL_N;
	}

	auto i = 0;
	for(auto list_elem: list)
	{
		_SM[i] = list_elem;
		i++;
	}

	return SMatrix$::CODE_NORM;

}

__device__ Kamanri::Maths::SMatrixCode Kamanri::Maths::SMatrix::operator*=(Kamanri::Maths::SMatrixElemType value)
{
	for (size_t i = 0; i < _N * _N; i++)
	{
		_SM[i] *= value;
	}

	return SMatrix$::CODE_NORM;
}

__device__ Kamanri::Maths::SMatrixCode Kamanri::Maths::SMatrix::operator*(Vector& v) const
{
	if (_N != v.N())
	{
		Kamanri::Utils::PrintLn("Call of SMatrix::operator*: matrix and vector of unequal length: %d and %d", _N, v.N());
		return SMatrix$::CODE_NOT_EQUEL_N;
	}

	Vector v_temp = v;

	double value = 0;

	for (size_t row = 0; row < _N; row++)
	{
		value = 0;
		for (size_t col = 0; col < _N; col++)
		{
			value += _SM[row * _N + col] * v_temp[col];
		}
		v.Set(row, value);
	}

	return SMatrix$::CODE_NORM;
}

__device__ Kamanri::Maths::SMatrix Kamanri::Maths::SMatrix::operator-() const
{
	// use AA* == |A|E
	// A^(-1) == A* / |A|
	auto pm_asm = operator*();
	auto d = Determinant();
	if (d == SMatrix$::NOT_INITIALIZED_VALUE)
	{
		Kamanri::Utils::PrintLn("Invalid determinant %f.", d);
	}
	pm_asm *= (1 / d);

	return pm_asm;

}



// [rest] [value] [dimension] [value]

#define REST_V_2_0 1, 2
#define REST_V_2_1 0, 2
#define REST_V_2_2 0, 1

#define REST_V_3_0 1, 2, 3
#define REST_V_3_1 0, 2, 3
#define REST_V_3_2 0, 1, 3
#define REST_V_3_3 0, 1, 2

// [Square Matrix] [Complement] [N dimension]
// (pointer of square matrix, width, const complement dimension, const row, const column)
#define SM_C(p_sm, n, c_d, c_row, c_col) __SMatrix::SMDet##c_d##(p_sm, n, REST_V_##c_d##_##c_row, REST_V_##c_d##_##c_col)

// [Square Matrix] [Algebratic Complement] [N dimension]
// (pointer of square matrix, width, const complement dimension, const row, const column)
#define SM_AC(p_sm, n, c_d, c_row, c_col) (SM_C(p_sm, n, c_d, c_row, c_col) * (((c_row + c_col) % 2 == 0) ? 1.f : -1.f))

__device__ Kamanri::Maths::SMatrix Kamanri::Maths::SMatrix::operator*() const
{

	if (_N != 3 && _N != 4)
	{
		Kamanri::Utils::PrintLn("operator* not allowed when _N = %llu", _N);
		return SMatrix();
	}

	auto p_sm = (Kamanri::Maths::SMatrixElemType*) _SM;

	if (_N == 3)
	{
		SMatrix ret_sm =
		{
			SM_AC(p_sm, _N, 2, 0, 0), SM_AC(p_sm, _N, 2, 1, 0), SM_AC(p_sm, _N, 2, 2, 0),
			SM_AC(p_sm, _N, 2, 0, 1), SM_AC(p_sm, _N, 2, 1, 1), SM_AC(p_sm, _N, 2, 2, 1),
			SM_AC(p_sm, _N, 2, 0, 2), SM_AC(p_sm, _N, 2, 1, 2), SM_AC(p_sm, _N, 2, 2, 2),
		};
		return ret_sm;
	}

	SMatrix ret_sm =
	{
		SM_AC(p_sm, _N, 3, 0, 0), SM_AC(p_sm, _N, 3, 1, 0), SM_AC(p_sm, _N, 3, 2, 0), SM_AC(p_sm, _N, 3, 3, 0),
		SM_AC(p_sm, _N, 3, 0, 1), SM_AC(p_sm, _N, 3, 1, 1), SM_AC(p_sm, _N, 3, 2, 1), SM_AC(p_sm, _N, 3, 3, 1),
		SM_AC(p_sm, _N, 3, 0, 2), SM_AC(p_sm, _N, 3, 1, 2), SM_AC(p_sm, _N, 3, 2, 2), SM_AC(p_sm, _N, 3, 3, 2),
		SM_AC(p_sm, _N, 3, 0, 3), SM_AC(p_sm, _N, 3, 1, 3), SM_AC(p_sm, _N, 3, 2, 3), SM_AC(p_sm, _N, 3, 3, 3)
	};

	return ret_sm;

}

__device__ Kamanri::Maths::SMatrixElemType Kamanri::Maths::SMatrix::Determinant() const
{
	size_t row_list[4] = { 0, 1, 2, 3 };
	size_t col_list[4] = { 0, 1, 2, 3 };
	switch (_N)
	{
		case 2:
			return __SMatrix::__Determinant((Kamanri::Maths::SMatrixElemType*) _SM, row_list, col_list, 2);
		case 3:
			return __SMatrix::__Determinant((Kamanri::Maths::SMatrixElemType*) _SM, row_list, col_list, 3);
		case 4:
			return __SMatrix::__Determinant((Kamanri::Maths::SMatrixElemType*) _SM, row_list, col_list, 4);
		default:
			Kamanri::Utils::PrintLn("Invalid dimension %d", _N);
			break;
	}
}