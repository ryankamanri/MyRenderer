#pragma once

#ifndef SWIG

#include <vector>
#include "vector.hpp"

#endif

namespace Kamanri
{
	namespace Maths
	{
		using SMatrixElemType = VectorElemType;
		using SMatrixCode = int;

		namespace SMatrix$
		{
			// values
			constexpr std::size_t NOT_INITIALIZED_N = Vector$::NOT_INITIALIZED_N;
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

	} // namespace Maths
	
} // namespace Kamanri
