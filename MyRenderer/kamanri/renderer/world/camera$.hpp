#pragma once
#ifndef SWIG
#include "kamanri/utils/result_declare.hpp"
#include "kamanri/maths/vector.hpp"
#include "kamanri/utils/memory.hpp"
#include "kamanri/renderer/world/__/resources.hpp"
#include "blinn_phong_reflection_model.hpp"
#endif
namespace Kamanri
{
	namespace Renderer
	{
		namespace World
		{
			namespace Camera$
			{
				constexpr const int CODE_NULL_POINTER_PVERTICES = 100;
				constexpr const int CODE_INVALID_VECTOR_LENGTH = 200;
				constexpr const int CODE_UNEQUAL_NUM = 300;
			}
		}
	}
}