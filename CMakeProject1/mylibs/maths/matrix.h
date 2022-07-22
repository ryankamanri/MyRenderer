#pragma once
#include <initializer_list>
#include "../utils/memory.h"
#include "../utils/result.h"
#include "vector.h"

using SMatrixElemType = double;

constexpr int MYMATRIX_NOT_INITIALIZED_N = 0;

// Codes
constexpr int MYMATRIX_CODE_NOT_INITIALIZED_N = 101;
constexpr int MYMATRIX_CODE_NOT_INITIALIZED_MATRIX = 201;
constexpr int MYMATRIX_CODE_NOT_EQUEL_N = 301;
constexpr int MYMATRIX_CODE_INVALID_OPERATION = 401;
constexpr int MYMATRIX_CODE_INDEX_OUT_OF_BOUND = 501;
constexpr int MYMATRIX_CODE_DUPLICATE_VALUE = 601;
/**
 * @brief  n * n Square Matrix
 * 
 */
class SMatrix
{
    public:
        SMatrix();
        explicit SMatrix(size_t n);
        SMatrix(SMatrix& sm);
        SMatrix(std::initializer_list<SMatrixElemType> list);
        SMatrix(std::initializer_list<std::vector<SMatrixElemType>> v_list);
        P<SMatrix> Copy();
        SMatrix& operator=(SMatrix& sm);
        // Get the size of the Matrix.
        P<MyResult<std::size_t>> N() const;

        // Get the value of the Vector by index
        P<MyResult<SMatrixElemType>> operator [] (int n) const;
        P<MyResult<SMatrixElemType>> Get(size_t row, size_t col) const;
        P<MyResult<Vector>> Get(size_t col) const;
        // Setter
        DefaultResult Set(size_t row, size_t col, SMatrixElemType value) const;
        DefaultResult Set(size_t col, Vector const& v) const;

        DefaultResult operator += (SMatrix const& sm);
        DefaultResult operator += (std::initializer_list<SMatrixElemType> list);
        DefaultResult operator += (std::initializer_list<std::vector<SMatrixElemType>> v_list);

        DefaultResult operator -= (SMatrix const& sm);
        DefaultResult operator -= (std::initializer_list<SMatrixElemType> list);
        DefaultResult operator -= (std::initializer_list<std::vector<SMatrixElemType>> v_list);

        DefaultResult operator *= (SMatrix const& sm);
        DefaultResult operator *= (std::initializer_list<SMatrixElemType> list);
        DefaultResult operator *= (std::initializer_list<std::vector<SMatrixElemType>> v_list);
        DefaultResult operator *= (SMatrixElemType value);

        DefaultResult operator * (Vector& v) const;
        
        // Transpose matrix
        P<MyResult<SMatrix>> operator + () const;
        // Inverse matrix
        P<MyResult<SMatrix>> operator - () const;
        // Adjoint matrix
        P<MyResult<SMatrix>> operator * () const;

        DefaultResult PrintMatrix(bool is_print = true, const char* decimal_count = "2") const;

        // The determinant
        P<MyResult<SMatrixElemType>> Determinant(std::vector<std::size_t> row_list, std::vector<std::size_t> col_list) const;
        P<MyResult<SMatrixElemType>> Determinant() const;

        // algebraic complement
        P<MyResult<SMatrixElemType>> AComplement(size_t row, size_t col) const;

        

    private:
        // The pointer indicated to square matrix.
        P<SMatrixElemType> _SM;
        // The length of the square amtrix.
        std::size_t _N = MYMATRIX_NOT_INITIALIZED_N;
        // Calculate the determinant of matrix recursively
        SMatrixElemType _Determinant(SMatrixElemType* psm, std::vector<std::size_t>& row_list, std::vector<std::size_t>& col_list) const;

        // algebraic complement
        SMatrixElemType _AComplement(SMatrixElemType *psm, std::vector<size_t> row_list, std::vector<size_t> col_list, size_t row, size_t col) const;
};
