#pragma once

#ifndef SWIG
#include "kamanri/utils/log_declare.hpp"
#include <initializer_list>
#endif

namespace Kamanri
{
	namespace Maths
	{
		using VectorElemType = double;
		using VectorCode = int;
		
		namespace Vector$
		{
			constexpr VectorElemType NOT_INITIALIZED_VALUE = -1;
			constexpr std::size_t NOT_INITIALIZED_N = 0;
			constexpr int MAX_SUPPORTED_DIMENSION = 4;

			// Codes
			constexpr VectorCode CODE_NORM = 0;
			constexpr VectorCode CODE_NOT_INITIALIZED_N = 100;
			constexpr VectorCode CODE_NOT_INITIALIZED_VECTOR = 200;
			constexpr VectorCode CODE_NOT_EQUEL_N = 300;
			constexpr VectorCode CODE_INVALID_OPERATION = 400;
			constexpr VectorCode CODE_INDEX_OUT_OF_BOUND = 500;
		}

	} // namespace Maths
	
} // namespace Kamanri
