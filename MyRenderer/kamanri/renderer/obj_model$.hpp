#pragma once
#ifndef SWIG
#include <vector>
#include "kamanri/utils/result.hpp"
#include "tga_image.hpp"
#endif

namespace Kamanri
{
	namespace Renderer
	{
		namespace ObjModel$
		{
			constexpr int CODE_INVALID_TYPE = 0;
			constexpr int CODE_CANNOT_READ_FILE = 100;
			constexpr int CODE_READING_EXCEPTION = 200;
			constexpr int CODE_INDEX_OUT_OF_BOUND = 300;

			class Face
			{
			public:
				std::vector<int> vertex_indexes;
				std::vector<int> vertex_texture_indexes;
				std::vector<int> vertex_normal_indexes;
			};
		}
	}
}