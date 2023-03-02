#pragma once
#include <vector>
#include <initializer_list>
#include "vector.hpp"

namespace Kamanri
{
	namespace Maths
	{

		using SMatrixElemType = VectorElemType;
		using SMatrixCode = int;

		namespace SMatrix$
		{
			// values
			constexpr int NOT_INITIALIZED_N = Vector$::NOT_INITIALIZED_N;
			constexpr SMatrixElemType NOT_INITIALIZED_VALUE = Vector$::NOT_INITIALIZED_VALUE;
			constexpr int MAX_SUPPORTED_DIMENSION = Vector$::MAX_SUPPORTED_DIMENSION;

			// Codes
			constexpr SMatrixCode CODE_NORM = 0;
			constexpr SMatrixCode CODE_NOT_INITIALIZED_N = 101;
			constexpr SMatrixCode CODE_NOT_INITIALIZED_MATRIX = 201;
			constexpr SMatrixCode CODE_NOT_EQUEL_N = 301;
			constexpr SMatrixCode CODE_INVALID_OPERATION = 401;
			constexpr SMatrixCode CODE_INDEX_OUT_OF_BOUND = 501;
			constexpr SMatrixCode CODE_DUPLICATE_VALUE = 601;
		}

		/**
		 * @brief  n * n Square Matrix
		 *
		 */
		class SMatrix
		{
		public:
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			SMatrix();
			explicit SMatrix(size_t n);

			// SMatrix(SMatrix &&sm);
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			SMatrix(SMatrix const& sm);
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			SMatrix(std::initializer_list<SMatrixElemType> list);
			SMatrix(std::initializer_list<std::vector<SMatrixElemType>> v_list);

			SMatrix &operator=(SMatrix const& sm);
			SMatrixCode operator=(std::initializer_list<SMatrixElemType> list);
			SMatrixCode operator=(std::initializer_list<std::vector<SMatrixElemType>> v_list);
			// Get the size of the Matrix.
			inline std::size_t N() const { return _N; }

			// Get the value of the Vector by index
			SMatrixElemType operator[](size_t n) const;
			SMatrixElemType Get(size_t row, size_t col) const;
			Vector Get(size_t col) const;
			// Setter
			SMatrixCode Set(size_t row, size_t col, SMatrixElemType value);
			SMatrixCode Set(size_t col, Vector const &v);

			SMatrixCode operator+=(SMatrix const &sm);
			SMatrixCode operator+=(std::initializer_list<SMatrixElemType> list);
			SMatrixCode operator+=(std::initializer_list<std::vector<SMatrixElemType>> v_list);

			SMatrixCode operator-=(SMatrix const &sm);
			SMatrixCode operator-=(std::initializer_list<SMatrixElemType> list);
			SMatrixCode operator-=(std::initializer_list<std::vector<SMatrixElemType>> v_list);

			SMatrixCode operator*=(SMatrix const &sm);
			SMatrixCode operator*=(std::initializer_list<SMatrixElemType> list);
			SMatrixCode operator*=(std::initializer_list<std::vector<SMatrixElemType>> v_list);
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			SMatrixCode operator*=(SMatrixElemType value);
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			SMatrixCode operator*(Vector &v) const;

			// Transpose matrix
			SMatrix operator+() const;
			// Inverse matrix
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			SMatrix operator-() const;
			// Adjoint matrix
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			SMatrix operator*() const;

			SMatrixCode PrintMatrix(Utils::LogLevel level = Utils::Log$::INFO_LEVEL, const char *decimal_count = "2") const;

			// The determinant
			SMatrixElemType Determinant(std::vector<std::size_t> row_list, std::vector<std::size_t> col_list) const;
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			SMatrixElemType Determinant() const;

			// algebraic complement
			SMatrixElemType AComplement(size_t row, size_t col) const;

		private:
			// The pointer indicated to square matrix.
			SMatrixElemType _SM[SMatrix$::MAX_SUPPORTED_DIMENSION * SMatrix$::MAX_SUPPORTED_DIMENSION];
			// The length of the square amtrix.
			std::size_t _N = SMatrix$::NOT_INITIALIZED_N;

			// SMatrixElemType _Get(size_t row, size_t col) const;

			Vector _Get(size_t col) const;
			// Calculate the determinant of matrix recursively
			SMatrixElemType _Determinant(SMatrixElemType* psm, std::vector<std::size_t> &row_list, std::vector<std::size_t> &col_list) const;

			// algebraic complement
			SMatrixElemType _AComplement(SMatrixElemType* psm, std::vector<size_t>& row_list, std::vector<size_t>& col_list, size_t row, size_t col) const;
		};

	}
}
