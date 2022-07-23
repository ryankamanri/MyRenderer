#pragma once
#include <initializer_list>
#include "../utils/memory.h"
#include "../utils/result.h"
#include "vectors.h"

namespace Kamanri
{
    namespace Maths
    {
        namespace Matrix
        {

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
                SMatrix(SMatrix &sm);
                SMatrix(std::initializer_list<SMatrixElemType> list);
                SMatrix(std::initializer_list<std::vector<SMatrixElemType>> v_list);
                Utils::Memory::P<SMatrix> Copy();
                SMatrix &operator=(SMatrix &sm);
                // Get the size of the Matrix.
                Utils::Memory::P<Utils::Result::MyResult<std::size_t>> N() const;

                // Get the value of the Vector by index
                Utils::Memory::P<Utils::Result::MyResult<SMatrixElemType>> operator[](int n) const;
                Utils::Memory::P<Utils::Result::MyResult<SMatrixElemType>> Get(size_t row, size_t col) const;
                Utils::Memory::P<Utils::Result::MyResult<Vectors::Vector>> Get(size_t col) const;
                // Setter
                Utils::Result::DefaultResult Set(size_t row, size_t col, SMatrixElemType value) const;
                Utils::Result::DefaultResult Set(size_t col, Vectors::Vector const &v) const;

                Utils::Result::DefaultResult operator+=(SMatrix const &sm);
                Utils::Result::DefaultResult operator+=(std::initializer_list<SMatrixElemType> list);
                Utils::Result::DefaultResult operator+=(std::initializer_list<std::vector<SMatrixElemType>> v_list);

                Utils::Result::DefaultResult operator-=(SMatrix const &sm);
                Utils::Result::DefaultResult operator-=(std::initializer_list<SMatrixElemType> list);
                Utils::Result::DefaultResult operator-=(std::initializer_list<std::vector<SMatrixElemType>> v_list);

                Utils::Result::DefaultResult operator*=(SMatrix const &sm);
                Utils::Result::DefaultResult operator*=(std::initializer_list<SMatrixElemType> list);
                Utils::Result::DefaultResult operator*=(std::initializer_list<std::vector<SMatrixElemType>> v_list);
                Utils::Result::DefaultResult operator*=(SMatrixElemType value);

                Utils::Result::DefaultResult operator*(Vectors::Vector &v) const;

                // Transpose matrix
                Utils::Memory::P<Utils::Result::MyResult<SMatrix>> operator+() const;
                // Inverse matrix
                Utils::Memory::P<Utils::Result::MyResult<SMatrix>> operator-() const;
                // Adjoint matrix
                Utils::Memory::P<Utils::Result::MyResult<SMatrix>> operator*() const;

                Utils::Result::DefaultResult PrintMatrix(bool is_print = true, const char *decimal_count = "2") const;

                // The determinant
                Utils::Memory::P<Utils::Result::MyResult<SMatrixElemType>> Determinant(std::vector<std::size_t> row_list, std::vector<std::size_t> col_list) const;
                Utils::Memory::P<Utils::Result::MyResult<SMatrixElemType>> Determinant() const;

                // algebraic complement
                Utils::Memory::P<Utils::Result::MyResult<SMatrixElemType>> AComplement(size_t row, size_t col) const;

            private:
                // The pointer indicated to square matrix.
                Utils::Memory::P<SMatrixElemType> _SM;
                // The length of the square amtrix.
                std::size_t _N = MYMATRIX_NOT_INITIALIZED_N;
                // Calculate the determinant of matrix recursively
                SMatrixElemType _Determinant(SMatrixElemType *psm, std::vector<std::size_t> &row_list, std::vector<std::size_t> &col_list) const;

                // algebraic complement
                SMatrixElemType _AComplement(SMatrixElemType *psm, std::vector<size_t> row_list, std::vector<size_t> col_list, size_t row, size_t col) const;
            };

        }
    }
}
