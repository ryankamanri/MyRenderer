#pragma once
#include <vector>
#include "triangle3d.hpp"

namespace Kamanri
{
	namespace Renderer
	{
		namespace World
		{
			namespace __
			{
				class Environment
				{
				private:
					/* data */
				public:
					Environment() = default;
					Environment& operator=(Environment&& other);
					std::vector<Triangle3D> triangles;
					Triangle3D* cuda_triangles;
					/// @brief Store all objects.
					std::vector<Object> objects;
					Object* cuda_objects;
				};
			} // namespace __

		} // namespace World

	} // namespace Renderer

} // namespace Kamanri
