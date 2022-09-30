#pragma once
#include <initializer_list>
#include "../utils/memory.hpp"
#include "../utils/result.hpp"
#include "vector.hpp"

namespace Kamanri
{
    namespace Maths
    {

        using SMatrixElemType = double;

        namespace SMatrix$
        {
            // values
            constexpr int NOT_INITIALIZED_N = 0;
            constexpr SMatrixElemType NOT_INITIALIZED_VALUE = -1;

            // Codes
            constexpr int CODE_NOT_INITIALIZED_N = 101;
            constexpr int CODE_NOT_INITIALIZED_MATRIX = 201;
            constexpr int CODE_NOT_EQUEL_N = 301;
            constexpr int CODE_INVALID_OPERATION = 401;
            constexpr int CODE_INDEX_OUT_OF_BOUND = 501;
            constexpr int CODE_DUPLICATE_VALUE = 601;
        }

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
            Utils::DefaultResult Reset(std::initializer_list<SMatrixElemType> list);
            Utils::DefaultResult Reset(std::initializer_list<std::vector<SMatrixElemType>> v_list);
            Utils::P<SMatrix> Copy();
            SMatrix &operator=(SMatrix &sm);
            // Get the size of the Matrix.
            Utils::Result<std::size_t> N() const;

            // Get the value of the Vector by index
            Utils::Result<SMatrixElemType> operator[](int n) const;
            Utils::Result<SMatrixElemType> Get(size_t row, size_t col) const;
            Utils::Result<Vector> Get(size_t col) const;
            // Setter
            Utils::DefaultResult Set(size_t row, size_t col, SMatrixElemType value) const;
            Utils::DefaultResult Set(size_t col, Vector const &v) const;

            Utils::DefaultResult operator+=(SMatrix const &sm);
            Utils::DefaultResult operator+=(std::initializer_list<SMatrixElemType> list);
            Utils::DefaultResult operator+=(std::initializer_list<std::vector<SMatrixElemType>> v_list);

            Utils::DefaultResult operator-=(SMatrix const &sm);
            Utils::DefaultResult operator-=(std::initializer_list<SMatrixElemType> list);
            Utils::DefaultResult operator-=(std::initializer_list<std::vector<SMatrixElemType>> v_list);

            Utils::DefaultResult operator*=(SMatrix const &sm);
            Utils::DefaultResult operator*=(std::initializer_list<SMatrixElemType> list);
            Utils::DefaultResult operator*=(std::initializer_list<std::vector<SMatrixElemType>> v_list);
            Utils::DefaultResult operator*=(SMatrixElemType value);

            Utils::DefaultResult operator*(Vector &v) const;

            // Transpose matrix
            Utils::Result<SMatrix> operator+() const;
            // Inverse matrix
            Utils::Result<SMatrix> operator-() const;
            // Adjoint matrix
            Utils::Result<SMatrix> operator*() const;

            Utils::DefaultResult PrintMatrix(bool is_print = true, const char *decimal_count = "2") const;

            // The determinant
            Utils::Result<SMatrixElemType> Determinant(std::vector<std::size_t> row_list, std::vector<std::size_t> col_list) const;
            Utils::Result<SMatrixElemType> Determinant() const;

            // algebraic complement
            Utils::Result<SMatrixElemType> AComplement(size_t row, size_t col) const;

        private:
            // The pointer indicated to square matrix.
            Utils::P<SMatrixElemType> _SM;
            // The length of the square amtrix.
            std::size_t _N = SMatrix$::NOT_INITIALIZED_N;

            SMatrixElemType _Get(size_t row, size_t col) const;

            Vector _Get(size_t col) const;
            // Calculate the determinant of matrix recursively
            SMatrixElemType _Determinant(SMatrixElemType *psm, std::vector<std::size_t> &row_list, std::vector<std::size_t> &col_list) const;

            // algebraic complement
            SMatrixElemType _AComplement(SMatrixElemType *psm, std::vector<size_t> row_list, std::vector<size_t> col_list, size_t row, size_t col) const;
        };

    }
}
