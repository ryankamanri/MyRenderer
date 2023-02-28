#pragma once
#include <vector>
#include "triangle3d.hpp"
#include "kamanri/renderer/world/bling_phong_reflection_model.hpp"

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
					Environment(BlingPhongReflectionModel&& model): bpr_model(std::move(model)) {}
					Environment& operator=(Environment&& other);
					BlingPhongReflectionModel bpr_model;
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
