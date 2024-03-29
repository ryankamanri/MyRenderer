#pragma once
#ifndef SWIG
#include "kamanri/maths/all.hpp"
#endif

namespace Kamanri
{
	namespace Renderer
	{
		namespace World
		{

			class FrameBuffer
			{
				public:
				/////////// update in triangle3D
				/// @brief the located triangle index
				size_t triangle_index;
				/// the point location
				Kamanri::Maths::Vector location;
				/// the vertex normal
				Kamanri::Maths::Vector vertex_normal;
				/// RGB color reflect
				unsigned int color;
				// /////// update in bpr model
				double power;
				unsigned int r;
				unsigned int g;
				unsigned int b;
				unsigned int specular_color;
				unsigned int diffuse_color;
				unsigned int ambient_color;
			};


		} // namespace World

	} // namespace Renderer

} // namespace Kamanri