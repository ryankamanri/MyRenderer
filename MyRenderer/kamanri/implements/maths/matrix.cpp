#include <math.h>
#include <algorithm>
#include "../../maths/matrix.hpp"
#include "../../utils/logs.hpp"
#include "../../utils/string.hpp"

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
                for (auto i = 0; i < size; i++)
                {
                    if (list[i] == value)
                        result++;
                }
                return result;
            }

            bool RemoveFromList(std::vector<size_t> &list, size_t item)
            {
                auto begin = list.begin();
                auto end = list.end();
                for (auto i = begin; i < end; i++)
                {
                    if (*i.base() == item)
                    {
                        list.erase(i);
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
                auto begin = list.begin();
                auto end = list.end();
                auto result = 0;
                for (auto i = begin; i != end; i++)
                {
                    if (*i.base() == value)
                    {
                        return result;
                    }
                    result++;
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////
// Class member functions

SMatrix::SMatrix() = default;

SMatrix::SMatrix(size_t n)
{
    this->_N = SMatrix$::NOT_INITIALIZED_N;

    if (n != 2 && n != 3 && n != 4)
    {
        Log::Error(__SMatrix::LOG_NAME, "The size of initializer list is not valid: %d", (int)n);
        return;
    }

    this->_N = n;
    this->_SM = P<SMatrixElemType>(new SMatrixElemType[n * n]);
}

SMatrix::SMatrix(SMatrix &sm) : _SM(std::move(sm._SM)), _N(sm._N) {}

SMatrix::SMatrix(std::initializer_list<SMatrixElemType> list)
{
    this->_N = SMatrix$::NOT_INITIALIZED_N;

    auto n = list.size();
    if (n != 4 && n != 9 && n != 16)
    {
        Log::Error(__SMatrix::LOG_NAME, "The size of initializer list is not valid: %d", (int)n);
        return;
    }

    this->_N = (size_t)sqrt((double)n);
    this->_SM = P<SMatrixElemType>(new SMatrixElemType[n]);

    auto begin = list.begin();
    auto end = list.end();
    auto pSM = this->_SM.get();

    CHECK_MEMORY_IS_ALLOCATED(pSM, __SMatrix::LOG_NAME, )

    for (size_t i = 0; begin + i != end; i++)
    {
        *(pSM + i) = *(begin + i);
    }
}

SMatrix::SMatrix(std::initializer_list<std::vector<SMatrixElemType>> v_list)
{
    this->_N = SMatrix$::NOT_INITIALIZED_N;
    auto v_count = v_list.size();

    this->_SM = P<SMatrixElemType>(new SMatrixElemType[v_count * v_count]);

    auto begin = v_list.begin();
    auto end = v_list.end();
    auto pSM = this->_SM.get();

    CHECK_MEMORY_IS_ALLOCATED(pSM, __SMatrix::LOG_NAME, )

    for (size_t i = 0; begin + i != end; i++)
    {
        auto v = *(begin + i);
        auto v_size = v.size();
        if (v_size != v_count)
        {
            Log::Error(__SMatrix::LOG_NAME, "Invalid given square matrix: in vector %d, vector size = %d, vector size = %d", i + 1, v_size, v_count);
            return;
        }
        for (size_t j = 0; j < v_size; j++)
        {
            *(pSM + j * v_count + i) = v[j];
        }
    }
    this->_N = v_count;
}

P<SMatrix> SMatrix::Copy()
{
    auto new_sm = New<SMatrix>(_N);
    auto psm = _SM.get();
    auto psm_new = new_sm->_SM.get();

    CHECK_MEMORY_IS_ALLOCATED(psm, __SMatrix::LOG_NAME, new_sm)

    for (size_t i = 0; i < _N * _N; i++)
    {
        *(psm_new + i) = *(psm + i);
    }

    return new_sm;
}

SMatrix &SMatrix::operator=(SMatrix &sm)
{
    auto psm = sm._SM.get();

    CHECK_MEMORY_IS_ALLOCATED(psm, __SMatrix::LOG_NAME, *this)

    _N = sm._N;
    _SM = std::move(sm._SM);

    sm._N = Vector$::NOT_INITIALIZED_N;

    return *this;
}

/**
 * @brief return the n
 *
 * @return PMyResult<std::size_t>>
 */
PMyResult<std::size_t> SMatrix::N() const
{
    if (_N == Vector$::CODE_NOT_INITIALIZED_N)
    {
        auto message = "Call of SMatrix::N(): The vector is not initialized";
        Log::Error(__SMatrix::LOG_NAME, message);
        return RESULT_EXCEPTION(std::size_t, Vector$::CODE_INVALID_OPERATION, message);
    }
    return New<Result<std::size_t>>(_N);
}

PMyResult<SMatrixElemType> SMatrix::operator[](int index) const
{
    auto psm = _SM.get();
    CHECK_MEMORY_FOR_RESULT(SMatrixElemType, psm, __SMatrix::LOG_NAME, -1)

    if (index < 0 || index > _N * _N)
    {
        Log::Error(__SMatrix::LOG_NAME, "Index %d out of bound %d", index, _N * _N);
        return RESULT_EXCEPTION(SMatrixElemType, SMatrix$::CODE_INDEX_OUT_OF_BOUND, "Index out of bound");
    }

    return New<Result<SMatrixElemType>>(*(psm + index));
}

PMyResult<SMatrixElemType> SMatrix::Get(size_t row, size_t col) const
{
    auto psm = _SM.get();
    CHECK_MEMORY_FOR_RESULT(SMatrixElemType, psm, __SMatrix::LOG_NAME, -1)

    if (row > _N || col > _N)
    {
        Log::Error(__SMatrix::LOG_NAME, "Index %d or %d out of bound %d", row, col, this->_N);
        return RESULT_EXCEPTION(SMatrixElemType, SMatrix$::CODE_INDEX_OUT_OF_BOUND, "Index out of bound");
    }

    return New<Result<SMatrixElemType>>(*(psm + row * _N + col));
}

PMyResult<Vector> SMatrix::Get(size_t col) const
{
    auto psm = _SM.get();
    CHECK_MEMORY_FOR_RESULT(Vector, psm, __SMatrix::LOG_NAME, SMatrix$::CODE_NOT_INITIALIZED_MATRIX)

    if (col > _N)
    {
        Log::Error(__SMatrix::LOG_NAME, "Index %d out of bound %d", col, this->_N);
        return RESULT_EXCEPTION(Vector, SMatrix$::CODE_INDEX_OUT_OF_BOUND, "Index out of bound");
    }

    Vector v(_N);

    for (size_t i = 0; i < _N; i++)
    {
        v.Set(i, *(psm + i * _N + col));
    }

    return New<Result<Vector>>(v);
}

DefaultResult SMatrix::Set(size_t row, size_t col, SMatrixElemType value) const
{
    auto psm = _SM.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(psm, __SMatrix::LOG_NAME, SMatrix$::CODE_NOT_INITIALIZED_MATRIX)

    if (row > _N || col > _N)
    {
        Log::Error(__SMatrix::LOG_NAME, "Index %d or %d out of bound %d", row, col, this->_N);
        return DEFAULT_RESULT_EXCEPTION(SMatrix$::CODE_INDEX_OUT_OF_BOUND, "Index out of bound");
    }

    *(psm + row * _N + col) = value;

    return DEFAULT_RESULT;
}

DefaultResult SMatrix::Set(size_t col, Vector const& v) const
{
    auto psm = _SM.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(psm, __SMatrix::LOG_NAME, SMatrix$::CODE_NOT_INITIALIZED_MATRIX)

    if (col > _N)
    {
        Log::Error(__SMatrix::LOG_NAME, "Index %d out of bound %d", col, this->_N);
        return DEFAULT_RESULT_EXCEPTION(SMatrix$::CODE_INDEX_OUT_OF_BOUND, "Index out of bound");
    }

    auto v_n = **v.N();

    if(v_n != _N)
    {
        Log::Error(__SMatrix::LOG_NAME, "Call of SMatrix::Set: Unequal length of Smatrix and Vector: %d and %d", _N, v_n);
        return DEFAULT_RESULT_EXCEPTION(SMatrix$::CODE_NOT_EQUEL_N, "Unequal length of Smatrix and Vector");
    }

    for (size_t row = 0; row < _N; row++)
    {
        *(psm + row * _N + col) = v.GetFast((int)row);
    }

    return DEFAULT_RESULT;
}

DefaultResult SMatrix::operator+=(SMatrix const &sm)
{
    auto psm = _SM.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(psm, __SMatrix::LOG_NAME, SMatrix$::CODE_NOT_INITIALIZED_MATRIX)
    auto psm2 = sm._SM.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(psm2, __SMatrix::LOG_NAME, SMatrix$::CODE_NOT_INITIALIZED_MATRIX)

    size_t n1 = _N;
    size_t n2 = sm._N;

    if (n1 != n2)
    {
        Log::Error(__SMatrix::LOG_NAME, "Call of SMatrix::operator+=: Two matrixes of unequal length: %d and %d", n1, n2);
        return DEFAULT_RESULT_EXCEPTION(SMatrix$::CODE_NOT_EQUEL_N, "Two matrixes of unequal length");
    }

    for (size_t i = 0; i < n1 * n1; i++)
    {
        *(psm + i) += *(psm2 + i);
    }

    return DEFAULT_RESULT;
}

DefaultResult SMatrix::operator+=(std::initializer_list<SMatrixElemType> list)
{
    SMatrix sm(list);
    return this->operator+=(sm);
}

DefaultResult SMatrix::operator+=(std::initializer_list<std::vector<SMatrixElemType>> v_list)
{
    SMatrix sm(v_list);
    return this->operator+=(sm);
}

DefaultResult SMatrix::operator-=(SMatrix const &sm)
{
    auto psm = _SM.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(psm, __SMatrix::LOG_NAME, SMatrix$::CODE_NOT_INITIALIZED_MATRIX)
    auto psm2 = sm._SM.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(psm2, __SMatrix::LOG_NAME, SMatrix$::CODE_NOT_INITIALIZED_MATRIX)

    size_t n1 = _N;
    size_t n2 = sm._N;

    if (n1 != n2)
    {
        Log::Error(__SMatrix::LOG_NAME, "Call of SMatrix::operator+=: Two matrixes of unequal length: %d and %d", n1, n2);
        return DEFAULT_RESULT_EXCEPTION(SMatrix$::CODE_NOT_EQUEL_N, "Two matrixes of unequal length");
    }

    for (size_t i = 0; i < n1 * n1; i++)
    {
        *(psm + i) -= *(psm2 + i);
    }

    return DEFAULT_RESULT;
}

DefaultResult SMatrix::operator-=(std::initializer_list<SMatrixElemType> list)
{
    SMatrix sm(list);
    return this->operator-=(sm);
}

DefaultResult SMatrix::operator-=(std::initializer_list<std::vector<SMatrixElemType>> v_list)
{
    SMatrix sm(v_list);
    return this->operator-=(sm);
}

DefaultResult SMatrix::operator*=(SMatrix const& sm)
{
    auto psm = _SM.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(psm, __SMatrix::LOG_NAME, SMatrix$::CODE_NOT_INITIALIZED_MATRIX)
    auto psm2 = sm._SM.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(psm2, __SMatrix::LOG_NAME, SMatrix$::CODE_NOT_INITIALIZED_MATRIX)

    size_t n1 = _N;
    size_t n2 = sm._N;

    if (n1 != n2)
    {
        Log::Error(__SMatrix::LOG_NAME, "Call of SMatrix::operator*=: Two matrixes of unequal length: %d and %d", n1, n2);
        return DEFAULT_RESULT_EXCEPTION(SMatrix$::CODE_NOT_EQUEL_N, "Two matrixes of unequal length");
    }

    auto this_temp = this->Copy();

    for(size_t col_sm = 0; col_sm < _N; col_sm++)
    {
        // for every column of sm as a vector v
        // carculate the transform this matrix * v
        // let the x-column, y-column multiply x-hat, y-hat... of v.
        auto v = **sm.Get(col_sm);
        Vector out_v(_N);
        for(size_t col = 0; col < _N; col++)
        {
            auto hat = v.GetFast((int)col);
            auto this_col = **this_temp->Get(col);

            this_col *= hat;
            out_v += this_col;
        }

        Set(col_sm, out_v);
    }

    return DEFAULT_RESULT;
}

DefaultResult SMatrix::operator*=(std::initializer_list<SMatrixElemType> list)
{
    SMatrix sm(list);
    return this->operator*=(sm);
}

DefaultResult SMatrix::operator*=(std::initializer_list<std::vector<SMatrixElemType>> v_list)
{
    SMatrix sm(v_list);
    return this->operator*=(sm);
}

DefaultResult SMatrix::operator*=(SMatrixElemType value)
{
    auto psm = _SM.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(psm, __SMatrix::LOG_NAME, SMatrix$::CODE_NOT_INITIALIZED_MATRIX)

    for(size_t i = 0; i < _N * _N; i++)
    {
        *(psm + i) *= value;
    }
    
    return DEFAULT_RESULT;
}

DefaultResult SMatrix::operator*(Vector& v) const
{
    auto psm = _SM.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(psm, __SMatrix::LOG_NAME, SMatrix$::CODE_NOT_INITIALIZED_MATRIX)
    
    size_t n1 = _N;
    size_t n2 = **v.N();

    if (n1 != n2)
    {
        Log::Error(__SMatrix::LOG_NAME, "Call of SMatrix::operator*: matrix and vector of unequal length: %d and %d", n1, n2);
        return DEFAULT_RESULT_EXCEPTION(SMatrix$::CODE_NOT_EQUEL_N, "Matrix and vector of unequal length");
    }
    
    auto v_temp = *v.Copy();

    v.SetAll(0);

    for (size_t col = 0; col < _N; col++)
    {
        auto hat = v_temp.GetFast((int)col);
        auto this_col = **Get(col);

        this_col *= hat;
        v += this_col;
    }

    return DEFAULT_RESULT;
}

/**
 * @brief The transpose matrix. +A <==> A^T
 * 
 * @return PMyResult<SMatrix>> 
 */
PMyResult<SMatrix> SMatrix::operator+() const
{
    auto psm = _SM.get();
    CHECK_MEMORY_FOR_RESULT(SMatrix, psm, __SMatrix::LOG_NAME, SMatrix$::CODE_NOT_INITIALIZED_MATRIX)

    SMatrix ret_sm(_N);
    auto psm_ret = ret_sm._SM.get();

    for(size_t i = 0; i < _N; i ++)
    {
        for(size_t j = 0; j < _N; j++)
        {
            *(psm_ret + j * _N + i) = *(psm + i * _N + j);
        }
    }

    return New<Result<SMatrix>>(ret_sm);

}

/**
 * @brief The inverse matrix. -A <==> A^(-1)
 * 
 * @return PMyResult<SMatrix>> 
 */
PMyResult<SMatrix> SMatrix::operator-() const
{
    auto psm = _SM.get();
    CHECK_MEMORY_FOR_RESULT(SMatrix, psm, __SMatrix::LOG_NAME, SMatrix$::CODE_NOT_INITIALIZED_MATRIX)

    // use AA* == |A|E
    // A^(-1) == A* / |A|
    auto pm_asm = operator*();
    auto d = **Determinant();
    **pm_asm *= (1 / d);

    return pm_asm;

}

PMyResult<SMatrix> SMatrix::operator*() const
{
    auto psm = _SM.get();
    CHECK_MEMORY_FOR_RESULT(SMatrix, psm, __SMatrix::LOG_NAME, SMatrix$::CODE_NOT_INITIALIZED_MATRIX)

    SMatrix ret_sm(_N);
    auto psm_ret = ret_sm._SM.get();

    std::vector<std ::size_t> row_list;
    std::vector<std ::size_t> col_list;
    for (size_t i= 0; i < _N; i++)
    {
        row_list.push_back(i);
        col_list.push_back(i);
    }

    for(size_t i = 0; i < _N; i ++)
    {
        for(size_t j = 0; j < _N; j++)
        {
            *(psm_ret + i * _N + j) = _AComplement(psm, row_list, col_list, j, i);
        }
    }

    return New<Result<SMatrix>>(ret_sm);

} 



DefaultResult SMatrix::PrintMatrix(bool is_print, const char *decimal_count) const
{
    if(!is_print) return DEFAULT_RESULT;

    auto pSM = this->_SM.get();

    CHECK_MEMORY_FOR_DEFAULT_RESULT(pSM, __SMatrix::LOG_NAME, SMatrix$::CODE_NOT_INITIALIZED_MATRIX)

    std::string formatStr = "%.";
    formatStr.append(decimal_count);
    formatStr.append("f\t");

    if (this->_N == SMatrix$::NOT_INITIALIZED_N)
    {
        auto message = "The Matrix is NOT Initialized well";
        Log::Error(__SMatrix::LOG_NAME, message);

        return DEFAULT_RESULT_EXCEPTION(SMatrix$::CODE_NOT_INITIALIZED_N, message);
    }

    for (size_t i = 0; i < this->_N; i++)
    {
        for (size_t j = 0; j < this->_N; j++)
        {
            auto value = *(pSM + _N * i + j);
            Print(formatStr.c_str(), value);
        }
        PrintLn();
    }
    return DEFAULT_RESULT;
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
SMatrixElemType SMatrix::_Determinant(SMatrixElemType *psm, std::vector<std::size_t>& row_list, std::vector<std::size_t>& col_list) const
{
    auto row_count = row_list.size();
    auto col_count = col_list.size();
    
    if(row_count == 1)
    {
        return *(psm + _N * row_list[0] + col_list[0]);
    }
    if(row_count == 2)
    {
        auto result = *(psm + _N * row_list[0] + col_list[0]) * *(psm + _N * row_list[1] + col_list[1]) - *(psm + _N * row_list[0] + col_list[1]) * *(psm + _N * row_list[1] + col_list[0]);
        if(row_list[0] > row_list[1])
        {
            // This is to avoid the bigger index is front of the smaller
            result *= -1;
        }
        if(col_list[0] > col_list[1])
        {
            result *= -1;
        }

        return result;
    }

    SMatrixElemType result = 0;

    auto col_first = col_list.front();
    auto col_first_sorted = __SMatrix::GetSortedIndex(col_list, col_first);
    col_list.erase(col_list.begin());

    for(auto i = 0; i < row_count; i++)
    {
        auto row_first = row_list[0];
        auto row_first_sorted = __SMatrix::GetSortedIndex(row_list, row_first);
        row_list.erase(row_list.begin());
        //////////////////////// calculate sub result
        auto value = *(psm + _N * row_first + col_first);

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

PMyResult<SMatrixElemType> SMatrix::Determinant(std::vector<std::size_t> row_list, std::vector<std::size_t> col_list) const
{
    auto pSM = this->_SM.get();
    CHECK_MEMORY_FOR_RESULT(SMatrixElemType, pSM, __SMatrix::LOG_NAME, SMatrix$::CODE_NOT_INITIALIZED_MATRIX)

    auto row_count = row_list.size();
    auto col_count = col_list.size();

    if(row_count > _N || row_count == 0 || col_count > _N || col_count == 0)
    {
        Log::Error(__SMatrix::LOG_NAME, "Index size %d or %d invalid, or out of bound %d", row_count, col_count, this->_N);
        return RESULT_EXCEPTION(SMatrixElemType, SMatrix$::CODE_INDEX_OUT_OF_BOUND, "Index invalid, or out of bound");
    }
    if(row_count != col_count)
    {
        Log::Error(__SMatrix::LOG_NAME, "Unequal row_count %d and col_count %d", row_count, col_count);
        return RESULT_EXCEPTION(SMatrixElemType, SMatrix$::CODE_NOT_EQUEL_N, "Unequal row_count and col_count");
    }
    for(size_t i = 0; i < row_count; i++)
    {
        if(__SMatrix::Contains(row_list, row_list[i]) > 1 || __SMatrix::Contains(col_list, col_list[i]) > 1)
        {
            Log::Error(__SMatrix::LOG_NAME, "Invalid duplicate value %d or %d in row_list or col_list", row_list[i], col_list[i]);
            return RESULT_EXCEPTION(SMatrixElemType, SMatrix$::CODE_DUPLICATE_VALUE, "Invalid duplicate value or in row_list or col_list");
        }
    }
    

    SMatrixElemType res = _Determinant(pSM, row_list, col_list);

    return New<Result<SMatrixElemType>>(res);

}

PMyResult<SMatrixElemType> SMatrix::Determinant() const
{
    auto pSM = this->_SM.get();
    CHECK_MEMORY_FOR_RESULT(SMatrixElemType, pSM, __SMatrix::LOG_NAME, SMatrix$::CODE_NOT_INITIALIZED_MATRIX)

    std::vector<std ::size_t> row_list;
    std::vector<std ::size_t> col_list;
    for (size_t i= 0; i < _N; i++)
    {
        row_list.push_back(i);
        col_list.push_back(i);
    }

    SMatrixElemType res = _Determinant(pSM, row_list, col_list);

    return New<Result<SMatrixElemType>>(res);
}

SMatrixElemType SMatrix::_AComplement(SMatrixElemType *psm, std::vector<size_t> row_list, std::vector<size_t> col_list, size_t row, size_t col) const
{
    if(!__SMatrix::RemoveFromList(row_list, row) || !__SMatrix::RemoveFromList(col_list, col))
    {   
        Log::Error(__SMatrix::LOG_NAME, "The row %d is not in row_list or the col %d is not in col_list", row, col);
        return -1;
    }

    // return the determinant * -1^(a+b)
    return _Determinant(psm, row_list, col_list) * 
    (((row + col) % 2 == 0) ? 1.f : -1.f);
}

/**
 * @brief Calculate the algebraic complement
 * 
 * @param row 
 * @param col 
 * @return PMyResult<SMatrixElemType>> 
 */
PMyResult<SMatrixElemType> SMatrix::AComplement(size_t row, size_t col) const
{
    auto pSM = this->_SM.get();
    CHECK_MEMORY_FOR_RESULT(SMatrixElemType, pSM, __SMatrix::LOG_NAME, SMatrix$::CODE_NOT_INITIALIZED_MATRIX)

    if(row >= _N || col >= _N)
    {
        Log::Error(__SMatrix::LOG_NAME, "Index size %d or %d out of bound %d", row, col, this->_N);
        return RESULT_EXCEPTION(SMatrixElemType, SMatrix$::CODE_INDEX_OUT_OF_BOUND, "Index out of bound");
    }

    std::vector<std ::size_t> row_list;
    std::vector<std ::size_t> col_list;
    for (size_t i= 0; i < _N; i++)
    {
        row_list.push_back(i);
        col_list.push_back(i);
    }

    SMatrixElemType res = _AComplement(pSM, row_list, col_list, row, col);
    return New<Result<SMatrixElemType>>(res);
}