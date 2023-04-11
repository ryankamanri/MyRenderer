#pragma once
#include <vector>
#include "kamanri/utils/memory.hpp"
#include "kamanri/utils/list.hpp"
#include "triangle3d.hpp"
#include "bounding_box.hpp"
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
					/// @brief Store all Triangles
					std::vector<Triangle3D> triangles;
					Utils::List<Triangle3D> cuda_triangles;

					/// @brief Store all objects.
					std::vector<Object> objects;
					Utils::List<Object> cuda_objects;

					Utils::P<BoundingBox[]> boxes;
					Utils::List<BoundingBox> cuda_boxes;
				};
			} // namespace __

		} // namespace World

	} // namespace Renderer

} // namespace Kamanri
