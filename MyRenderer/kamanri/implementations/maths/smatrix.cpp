#include <cmath>
#include <algorithm>
#include <vector>
#include "kamanri/maths/smatrix.hpp"
#include "kamanri/utils/log.hpp"
#include "kamanri/utils/string.hpp"

using namespace Kamanri::Utils;
using namespace Kamanri::Maths;

namespace Kamanri
{
	namespace Maths
	{
		namespace __SMatrix
		{
			constexpr const char *LOG_NAME = STR(Kamanri::Maths::SMatrix);

			//////////////////////////////////////////////////////////////////////////////////////
			// Global helper functions

			/**
			 * @brief Calculate the count of value in given list
			 *
			 * @param list
			 * @param value
			 * @return int
			 */
			int Contains(std::vector<size_t> const &list, size_t value)
			{
				auto size = list.size();
				auto result = 0;
				for (size_t i = 0; i < size; i++)
				{
					if (list[i] == value)
						result++;
				}
				return result;
			}

			bool RemoveFromList(std::vector<size_t> &list, size_t item, size_t& out_index)
			{
				auto size = list.size();
				for (size_t i = 0; i < size; i++)
				{
					if (list[i] == item)
					{
						out_index = i;
						list.erase(list.begin() + i);
						return true;
					}
				}
				return false;
			}

			/**
			 * @brief Get the Sorted Index of object
			 *
			 * @param list
			 * @param value
			 * @return size_t
			 */
			size_t GetSortedIndex(std::vector<std::size_t> list, size_t value)
			{
				std::sort(list.begin(), list.end());
				auto result = 0;
				for (auto i = list.begin(); i != list.end(); i++)
				{
					if (*i == value)
					{
						return result;
					}
					result++;
				}
				return 0xFFFFFFFFFFFFFFFF;
			}

			/// @brief (-1)^RON
			/// @param v 
			/// @return 
			int Pow_NegativeOne_ReverseOrderNumber(std::vector<size_t>& v)
			{
				int res = 1;
				for (size_t i = 1; i < v.size(); i++)
				{
					for (size_t j = 0; j < i; j++)
					{
						if (v[j] > v[i]) res *= -1;
					}
				}
				return res;
			}


			inline SMatrixElemType SMGet(SMatrixElemType* sm, size_t n, size_t row, size_t col)
			{
				return sm[n * row + col];
			}

			// Square Matrix Determinant [dimension + 1]
			inline SMatrixElemType SMDet2(SMatrixElemType* sm, size_t n, size_t row_1, size_t row_2, size_t col_1, size_t col_2)
			{
				return (SMGet(sm, n, row_1, col_1) * SMGet(sm, n, row_2, col_2) - SMGet(sm, n, row_1, col_2) * SMGet(sm, n, row_2, col_1));
			}

			// Square Matrix Determinant [dimension + 1]
			inline SMatrixElemType SMDet3(SMatrixElemType* sm, size_t n, size_t row_1, size_t row_2, size_t row_3, size_t col_1, size_t col_2, size_t col_3)
			{
				return (SMGet(sm, n, row_1, col_1) * SMDet2(sm, n, row_2, row_3, col_2, col_3) - SMGet(sm, n, row_2, col_1) * SMDet2(sm, n, row_1, row_3, col_2, col_3) + SMGet(sm, n, row_3, col_1) * SMDet2(sm, n, row_1, row_2, col_2, col_3));
			}

			namespace operator_star_Vector
			{
				Vector v_temp;
			}

			namespace Determinant
			{
				std::vector<std ::size_t> row_list;
				std::vector<std ::size_t> col_list;
			} // namespace Determinant

			namespace AComplement
			{
				std::vector<std ::size_t> row_list;
				std::vector<std ::size_t> col_list;
			} // namespace AComplement
			
			
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// Class member functions

SMatrix::SMatrix()
{
	this->_N = SMatrix$::MAX_SUPPORTED_DIMENSION;
}

SMatrix::SMatrix(size_t n)
{
	this->_N = SMatrix$::NOT_INITIALIZED_N;

	if (n != 2 && n != 3 && n != 4)
	{
		Log::Error(__SMatrix::LOG_NAME, "The size of initializer list is not valid: %d", (int)n);
		PRINT_LOCATION;
		return;
	}

	this->_N = n;
}


SMatrix::SMatrix(SMatrix const& sm)
{
	_N = sm._N;
	auto size = _N * _N;

	for(size_t i = 0; i < size; i++)
	{
		_SM[i] = sm._SM[i];
	}
}

SMatrix::SMatrix(std::initializer_list<SMatrixElemType> list)
{
	this->_N = SMatrix$::NOT_INITIALIZED_N;

	auto size = list.size();
	if (size != 4 && size != 9 && size != 16)
	{
		Log::Error(__SMatrix::LOG_NAME, "The size of initializer list is not valid: %d", (int)size);
		PRINT_LOCATION;
		return;
	}

	this->_N = (size_t)sqrt((double)size);

	auto i = 0;
	for(auto list_elem: list)
	{
		_SM[i] = list_elem;
		i++;
	}
}

SMatrix::SMatrix(std::initializer_list<std::vector<SMatrixElemType>> v_list)
{
	this->_N = SMatrix$::NOT_INITIALIZED_N;
	auto v_count = v_list.size();

	auto begin = v_list.begin();


	for (size_t i = 0; begin + i != v_list.end(); i++)
	{
		auto& v = begin[i];
		auto v_size = v.size();
		if (v_size != v_count)
		{
			Log::Error(__SMatrix::LOG_NAME, "Invalid given square matrix: in vector %d, vector size = %d, vector size = %d", i + 1, v_size, v_count);
			PRINT_LOCATION;
			return;
		}
		for (size_t j = 0; j < v_size; j++)
		{
			_SM[j * v_count + i] = v[j];
		}
	}
	this->_N = v_count;
}


SMatrix &SMatrix::operator=(SMatrix const& sm)
{

	auto size = sm._N * sm._N;

	_N = sm._N;

	for(size_t i = 0; i < size; i++)
	{
		_SM[i] = sm._SM[i];
	}

	return *this;
}

SMatrixCode SMatrix::operator=(std::initializer_list<SMatrixElemType> list)
{
	auto size = list.size();
	if(size != _N * _N)
	{
		Log::Error(__SMatrix::LOG_NAME, "The size of initializer list(%d) is not equal to SMatrix(%d)", size, _N * _N);
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

SMatrixCode SMatrix::operator=(std::initializer_list<std::vector<SMatrixElemType>> v_list)
{

	auto v_count = v_list.size();
	if(v_count != _N)
	{
		Log::Error(__SMatrix::LOG_NAME, "The size of initializer vector list(%d) is not equal to SMatrix(%d)", v_count, _N);
		PRINT_LOCATION;
		return SMatrix$::CODE_NOT_EQUEL_N;
	}

	auto begin = v_list.begin();

	for (size_t i = 0; begin + i != v_list.end(); i++)
	{
		auto& v = begin[i];
		auto v_size = v.size();
		if (v_size != v_count)
		{
			Log::Error(__SMatrix::LOG_NAME, "Invalid given square matrix: in vector %d, vector size = %d, vector size = %d", i + 1, v_size, v_count);
			PRINT_LOCATION;
			return SMatrix$::CODE_NOT_EQUEL_N;
		}
		for (size_t j = 0; j < v_size; j++)
		{
			_SM[j * v_count + i] = v[j];
		}
	}

	return SMatrix$::CODE_NORM;

}



SMatrixElemType SMatrix::operator[](size_t index) const
{
	if (index < 0 || index > _N * _N)
	{
		Log::Error(__SMatrix::LOG_NAME, "Index %d out of bound %d", index, _N * _N);
		PRINT_LOCATION;
		return SMatrix$::CODE_INDEX_OUT_OF_BOUND;
	}

	return _SM[index];
}

// SMatrixElemType SMatrix::_Get(size_t row, size_t col) const
// {

//     if (row > _N || col > _N)
//     {
//         Log::Error(__SMatrix::LOG_NAME, "Index %d or %d out of bound %d", row, col, this->_N);
//         return SMatrix$::NOT_INITIALIZED_VALUE;
//     }

//     return _SM[row * _N + col];
// }


SMatrixElemType SMatrix::Get(size_t row, size_t col) const
{
	if (row > _N || col > _N)
	{
		Log::Error(__SMatrix::LOG_NAME, "Index %d or %d out of bound %d", row, col, this->_N);
		PRINT_LOCATION;
		return SMatrix$::CODE_INDEX_OUT_OF_BOUND;
	}

	return _SM[row * _N + col];
}

Vector SMatrix::_Get(size_t col) const
{

	if (col > _N)
	{
		Log::Error(__SMatrix::LOG_NAME, "Index %d out of bound %d", col, this->_N);
		PRINT_LOCATION;
		return {};
	}

	Vector v(_N);

	for (size_t i = 0; i < _N; i++)
	{
		v.Set(i, _SM[i * _N + col]);
	}

	return v;
}

Vector SMatrix::Get(size_t col) const
{
	if (col > _N)
	{
		Log::Error(__SMatrix::LOG_NAME, "Index %d out of bound %d", col, this->_N);
		PRINT_LOCATION;
		return Vector();
	}

	Vector v(_N);

	for (size_t i = 0; i < _N; i++)
	{
		v.Set(i, _SM[i * _N + col]);
	}

	return v;
}

SMatrixCode SMatrix::Set(size_t row, size_t col, SMatrixElemType value)
{
	if (row > _N || col > _N)
	{
		Log::Error(__SMatrix::LOG_NAME, "Index %d or %d out of bound %d", row, col, this->_N);
		PRINT_LOCATION;
		return SMatrix$::CODE_INDEX_OUT_OF_BOUND;
	}

	_SM[row * _N + col] = value;

	return SMatrix$::CODE_NORM;
}

SMatrixCode SMatrix::Set(size_t col, Vector const& v)
{
	if (col > _N)
	{
		Log::Error(__SMatrix::LOG_NAME, "Index %d out of bound %d", col, this->_N);
		PRINT_LOCATION;
		return SMatrix$::CODE_INDEX_OUT_OF_BOUND;
	}

	auto v_n = v.N();

	if(v_n != _N)
	{
		Log::Error(__SMatrix::LOG_NAME, "Call of SMatrix::Set: Unequal length of Smatrix and Vector: %d and %d", _N, v_n);
		PRINT_LOCATION;
		return SMatrix$::CODE_NOT_EQUEL_N;
	}

	for (size_t row = 0; row < _N; row++)
	{
		_SM[row * _N + col] = v[row];
	}

	return SMatrix$::CODE_NORM;
}

SMatrixCode SMatrix::operator+=(SMatrix const &sm)
{
	size_t n1 = _N;
	size_t n2 = sm._N;

	if (n1 != n2)
	{
		Log::Error(__SMatrix::LOG_NAME, "Call of SMatrix::operator+=: Two matrixes of unequal length: %d and %d", n1, n2);
		PRINT_LOCATION;
		return SMatrix$::CODE_NOT_EQUEL_N;
	}

	for (size_t i = 0; i < n1 * n1; i++)
	{
		_SM[i] += sm._SM[i];
	}

	return SMatrix$::CODE_NORM;
}

SMatrixCode SMatrix::operator+=(std::initializer_list<SMatrixElemType> list)
{
	SMatrix sm(list);
	return this->operator+=(sm);
}

SMatrixCode SMatrix::operator+=(std::initializer_list<std::vector<SMatrixElemType>> v_list)
{
	SMatrix sm(v_list);
	return this->operator+=(sm);
}

SMatrixCode SMatrix::operator-=(SMatrix const &sm)
{
	size_t n1 = _N;
	size_t n2 = sm._N;

	if (n1 != n2)
	{
		Log::Error(__SMatrix::LOG_NAME, "Call of SMatrix::operator+=: Two matrixes of unequal length: %d and %d", n1, n2);
		PRINT_LOCATION;
		return SMatrix$::CODE_NOT_EQUEL_N;
	}

	for (size_t i = 0; i < n1 * n1; i++)
	{
		_SM[i] -= sm._SM[i];
	}

	return SMatrix$::CODE_NORM;
}

SMatrixCode SMatrix::operator-=(std::initializer_list<SMatrixElemType> list)
{
	SMatrix sm(list);
	return this->operator-=(sm);
}

SMatrixCode SMatrix::operator-=(std::initializer_list<std::vector<SMatrixElemType>> v_list)
{
	SMatrix sm(v_list);
	return this->operator-=(sm);
}

SMatrixCode SMatrix::operator*=(SMatrix const& sm)
{
	if (_N != sm._N)
	{
		Log::Error(__SMatrix::LOG_NAME, "Call of SMatrix::operator*=: Two matrixes of unequal length: %d and %d", _N, sm._N);
		PRINT_LOCATION;
		return SMatrix$::CODE_NOT_EQUEL_N;
	}

	auto this_temp = *this;

	for(size_t col_sm = 0; col_sm < _N; col_sm++)
	{
		// for every column of sm as a vector v
		// carculate the transform this matrix * v
		// let the x-column, y-column multiply x-hat, y-hat... of v.
		auto v = sm._Get(col_sm);
		Vector out_v(_N);
		for(size_t col = 0; col < _N; col++)
		{
			auto hat = v[col];
			auto this_col = this_temp._Get(col);

			this_col *= hat;
			out_v += this_col;
		}

		Set(col_sm, out_v);
	}

	return SMatrix$::CODE_NORM;
}

SMatrixCode SMatrix::operator*=(std::initializer_list<SMatrixElemType> list)
{
	SMatrix sm(list);
	return this->operator*=(sm);
}

SMatrixCode SMatrix::operator*=(std::initializer_list<std::vector<SMatrixElemType>> v_list)
{
	SMatrix sm(v_list);
	return this->operator*=(sm);
}

SMatrixCode SMatrix::operator*=(SMatrixElemType value)
{
	for(size_t i = 0; i < _N * _N; i++)
	{
		_SM[i] *= value;
	}
	
	return SMatrix$::CODE_NORM;
}

SMatrixCode SMatrix::operator*(Vector& v) const
{
	using namespace __SMatrix::operator_star_Vector;
	if (_N != v.N())
	{
		Log::Error(__SMatrix::LOG_NAME, "Call of SMatrix::operator*: matrix and vector of unequal length: %d and %d", _N, v.N());
		PRINT_LOCATION;
		return SMatrix$::CODE_NOT_EQUEL_N;
	}
	
	v_temp = v;
	
	double value = 0;

	for(size_t row = 0; row < _N; row++)
	{
		value = 0;
		for(size_t col = 0; col < _N; col++)
		{
			value += _SM[row * _N + col] * v_temp[col];
		}
		v.Set(row, value);
	}

	return SMatrix$::CODE_NORM;
}

/**
 * @brief The transpose matrix. +A <==> A^T
 * 
 * @return PMyResult<SMatrix>> 
 */
SMatrix SMatrix::operator+() const
{
	SMatrix ret_sm(_N);

	for(size_t i = 0; i < _N; i ++)
	{
		for(size_t j = 0; j < _N; j++)
		{
			ret_sm._SM[j * _N + i] = _SM[i * _N + j];
		}
	}

	return ret_sm;

}

/**
 * @brief The inverse matrix. -A <==> A^(-1)
 * 
 * @return PMyResult<SMatrix>> 
 */
SMatrix SMatrix::operator-() const
{
	// use AA* == |A|E
	// A^(-1) == A* / |A|
	auto pm_asm = operator*();
	auto d = Determinant();
	if(d == SMatrix$::NOT_INITIALIZED_VALUE)
	{
		Log::Error(__SMatrix::LOG_NAME, "Invalid determinant %f.", d);
		PRINT_LOCATION;
	}
	pm_asm *= (1 / d);

	return pm_asm;

}





SMatrixCode SMatrix::PrintMatrix(LogLevel level, const char *decimal_count) const
{
	if(Log::Level() > level) return SMatrix$::CODE_NORM;

	std::string formatStr = "%.";
	formatStr.append(decimal_count);
	formatStr.append("f\t");

	if (this->_N == SMatrix$::NOT_INITIALIZED_N)
	{
		auto message = "The Matrix is NOT Initialized well";
		Log::Error(__SMatrix::LOG_NAME, message);
		PRINT_LOCATION;
		return SMatrix$::CODE_NOT_INITIALIZED_N;
	}

	for (size_t i = 0; i < this->_N; i++)
	{
		for (size_t j = 0; j < this->_N; j++)
		{
			auto value = _SM[_N * i + j];
			Print(formatStr.c_str(), value);
		}
		PrintLn();
	}
	return SMatrix$::CODE_NORM;
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


/**
 * @brief Calculate the adjoint matrix
 * 
 * @return SMatrix 
 */
SMatrix SMatrix::operator*() const
{

	if(_N != 3 && _N != 4)
	{
		Log::Error(__SMatrix::LOG_NAME, "operator* not allowed when _N = %llu", _N);
		PRINT_LOCATION;
		return SMatrix();
	}

	auto p_sm = (SMatrixElemType*)_SM;

	if(_N == 3)
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


/**
 * @brief Calculate the determinant of matrix recursively, already checked:
 * row_count == col_count
 * 
 * @param sm 
 * @param row_from 
 * @param row_end 
 * @param col_from 
 * @param col_end 
 * @return PMyResult<SMatrixElemType>> 
 */
SMatrixElemType SMatrix::_Determinant(SMatrixElemType* psm, std::vector<std::size_t>& row_list, std::vector<std::size_t>& col_list) const
{
	using namespace __SMatrix;
	auto row_count = row_list.size();
	SMatrixElemType result = 0;

	if (row_count < 4)
	{
		if (row_count == 1) result = SMGet(psm, _N, row_list[0], col_list[0]);
		if (row_count == 2) result = SMDet2(psm, _N, row_list[0], row_list[1], col_list[0], col_list[1]);
		if (row_count == 3) result = SMDet3(psm, _N, row_list[0], row_list[1], row_list[2], col_list[0], col_list[1], col_list[2]);
		if (row_count == 4)
		{
			result = SMGet(psm, _N, row_list[0], col_list[0]) * SMDet3(psm, _N, 1, 2, 3, 1, 2, 3) -
			SMGet(psm, _N, row_list[1], col_list[0]) * SMDet3(psm, _N, row_list[0], row_list[2], row_list[3], col_list[1], col_list[2], col_list[3]) +
			SMGet(psm, _N, row_list[2], col_list[0]) * SMDet3(psm, _N, row_list[0], row_list[1], row_list[3], col_list[1], col_list[2], col_list[3]) -
			SMGet(psm, _N, row_list[3], col_list[0]) * SMDet3(psm, _N, row_list[0], row_list[1], row_list[2], col_list[1], col_list[2], col_list[3]);
		}
		// This is to avoid the bigger index is in front of the smaller
		result *= __SMatrix::Pow_NegativeOne_ReverseOrderNumber(row_list);
		result *= __SMatrix::Pow_NegativeOne_ReverseOrderNumber(col_list);

		return result;
	}
	


	/// row_count > 4
	size_t col_first = col_list.front();
	size_t col_first_sorted = __SMatrix::GetSortedIndex(col_list, col_first);
	col_list.erase(col_list.begin());

	for(size_t i = 0; i < row_count; i++)
	{
		size_t row_first = row_list[0];
		size_t row_first_sorted = __SMatrix::GetSortedIndex(row_list, row_first);
		row_list.erase(row_list.begin());
		//////////////////////// calculate sub result
		auto value = psm[_N * row_first + col_first];

		// use -1^(a+b) * n * |A*_ab| to calculate
		auto result_sub = (
			_Determinant(psm, row_list, col_list) * 
			value * 
			(((row_first_sorted + col_first_sorted) % 2 == 0) ? 1.f : -1.f));

		result += result_sub;

		//////////////////////// calculate sub result
		row_list.push_back(row_first);
	}

	col_list.insert(col_list.begin(), col_first);

	return result;
}

// TODO: 
// make calc determinant returns vector as
// vector = |i, j, k...| avaliable.

SMatrixElemType SMatrix::Determinant(std::vector<std::size_t> row_list, std::vector<std::size_t> col_list) const
{
	auto row_count = row_list.size();
	auto col_count = col_list.size();

	if(row_count > _N || row_count == 0 || col_count > _N || col_count == 0)
	{
		Log::Error(__SMatrix::LOG_NAME, "Index size %d or %d invalid, or out of bound %d", row_count, col_count, this->_N);
		PRINT_LOCATION;
		return SMatrix$::NOT_INITIALIZED_VALUE;
	}
	if(row_count != col_count)
	{
		Log::Error(__SMatrix::LOG_NAME, "Unequal row_count %d and col_count %d", row_count, col_count);
		PRINT_LOCATION;
		return SMatrix$::NOT_INITIALIZED_VALUE;
	}
	for(size_t i = 0; i < row_count; i++)
	{
		if(__SMatrix::Contains(row_list, row_list[i]) > 1 || __SMatrix::Contains(col_list, col_list[i]) > 1)
		{
			Log::Error(__SMatrix::LOG_NAME, "Invalid duplicate value %d or %d in row_list or col_list", row_list[i], col_list[i]);
			PRINT_LOCATION;
			return SMatrix$::NOT_INITIALIZED_VALUE;
		}
	}
	

	SMatrixElemType res = _Determinant((SMatrixElemType*)_SM, row_list, col_list);

	return res;

}

SMatrixElemType SMatrix::Determinant() const
{
	using namespace __SMatrix::Determinant;
	switch (_N)
	{
	case 2:
		row_list = col_list = {0, 1};
		break;
	case 3:
		row_list = col_list = {0, 1, 2};
		break;
	case 4:
		row_list = col_list = {0, 1, 2, 3};
		break;
	default:
		Log::Error(__SMatrix::LOG_NAME, "Invalid dimension %d", _N);
		break;
	}
	return _Determinant((SMatrixElemType*)_SM, row_list, col_list);
}

SMatrixElemType SMatrix::_AComplement(SMatrixElemType* psm, std::vector<size_t>& row_list, std::vector<size_t>& col_list, size_t row, size_t col) const
{
	size_t row_index, col_index;
	if(!__SMatrix::RemoveFromList(row_list, row, row_index) || !__SMatrix::RemoveFromList(col_list, col, col_index))
	{   
		Log::Error(__SMatrix::LOG_NAME, "The row %d is not in row_list or the col %d is not in col_list", row, col);
		PRINT_LOCATION;
		return -1;
	}

	// return the determinant * -1^(a+b)
	auto d = _Determinant(psm, row_list, col_list) * (((row + col) % 2 == 0) ? 1.f : -1.f);
	row_list.insert(row_list.begin() + row_index, row);
	col_list.insert(col_list.begin() + col_index, col);
	return d;
}

/**
 * @brief Calculate the algebraic complement
 * 
 * @param row 
 * @param col 
 * @return PMyResult<SMatrixElemType>> 
 */
SMatrixElemType SMatrix::AComplement(size_t row, size_t col) const
{
	if(row >= _N || col >= _N)
	{
		Log::Error(__SMatrix::LOG_NAME, "Index size %d or %d out of bound %d", row, col, this->_N);
		PRINT_LOCATION;
		return SMatrix$::NOT_INITIALIZED_VALUE;
	}

	using namespace __SMatrix::AComplement;
	row_list.clear();
	col_list.clear();
	for (size_t i= 0; i < _N; i++)
	{
		row_list.push_back(i);
		col_list.push_back(i);
	}

	return _AComplement((SMatrixElemType*)_SM, row_list, col_list, row, col);
}