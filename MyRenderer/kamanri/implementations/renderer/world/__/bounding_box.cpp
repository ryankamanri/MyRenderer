#include "kamanri/renderer/world/__/bounding_box.hpp"
#include "kamanri/renderer/world/blinn_phong_reflection_model.hpp"
#include "kamanri/utils/list.hpp"
#include "kamanri/maths/math.hpp"

namespace Kamanri
{
	namespace Renderer
	{
		namespace World
		{
			namespace __
			{
				namespace __BoundingBox
				{
					namespace __IsThrough
					{
						using AxisType = size_t;
						constexpr AxisType X_AXIS = 0;
						constexpr AxisType Y_AXIS = 1;
						constexpr AxisType Z_AXIS = 2;
						inline bool IsThroughPlane(Maths::Vector const& location, Maths::Vector const& direction, BoundingBox const& box, AxisType axis_type, double value)
						{
							double x = 0, y = 0, z = 0, ratio = 0;

							switch (axis_type)
							{
							case X_AXIS:
								x = value;
								ratio = (x - location[X_AXIS]) / direction[X_AXIS];
								y = (direction[Y_AXIS] * ratio) + location[Y_AXIS];
								z = (direction[Z_AXIS] * ratio) + location[Z_AXIS];
								if ((y <= box.world_max[Y_AXIS] && y >= box.world_min[Y_AXIS]) || (z <= box.world_max[Z_AXIS] && z >= box.world_min[Z_AXIS]))
								{
									return true;
								}
								break;
							case Y_AXIS:
								y = value;
								ratio = (y - location[Y_AXIS]) / direction[Y_AXIS];
								x = (direction[X_AXIS] * ratio) + location[X_AXIS];
								z = (direction[Z_AXIS] * ratio) + location[Z_AXIS];
								if ((x <= box.world_max[X_AXIS] && x >= box.world_min[X_AXIS]) || (z <= box.world_max[Z_AXIS] && z >= box.world_min[Z_AXIS]))
								{
									return true;
								}
								break;
							case Z_AXIS:
								z = value;
								ratio = (z - location[Z_AXIS]) / direction[Z_AXIS];
								x = (direction[X_AXIS] * ratio) + location[X_AXIS];
								y = (direction[Y_AXIS] * ratio) + location[Y_AXIS];
								if ((x <= box.world_max[X_AXIS] && x >= box.world_min[X_AXIS]) || (y <= box.world_max[Y_AXIS] && y >= box.world_min[Y_AXIS]))
								{
									return true;
								}
							default:
								break;
							}

							return false;

						}
					}
				}
			}
		}
	}
}
using namespace Kamanri::Renderer::World::__;

void __BoundingBox::Merge(BoundingBox const& l_box, BoundingBox const& r_box, BoundingBox& out_box)
{
	out_box.world_min =
	{
		Maths::Min(l_box.world_min[0], r_box.world_min[0]),
		Maths::Min(l_box.world_min[1], r_box.world_min[1]),
		Maths::Min(l_box.world_min[2], r_box.world_min[2]),
		1
	};
	out_box.world_max =
	{
		Maths::Max(l_box.world_max[0], r_box.world_max[0]),
		Maths::Max(l_box.world_max[1], r_box.world_max[1]),
		Maths::Max(l_box.world_max[2], r_box.world_max[2]),
		1
	};
	out_box.screen_min =
	{
		Maths::Min(l_box.screen_min[0], r_box.screen_min[0]),
		Maths::Min(l_box.screen_min[1], r_box.screen_min[1]),
		Maths::Min(l_box.screen_min[2], r_box.screen_min[2]),
		1
	};
	out_box.screen_max =
	{
		Maths::Max(l_box.screen_max[0], r_box.screen_max[0]),
		Maths::Max(l_box.screen_max[1], r_box.screen_max[1]),
		Maths::Max(l_box.screen_max[2], r_box.screen_max[2]),
		1
	};

	out_box.triangle_index = l_box.triangle_index;
	out_box.triangle_count = l_box.triangle_count + r_box.triangle_count;
}

void BoundingBox$::Build(BoundingBox* boxes, std::vector<Triangle3D> const& triangles)
{
	size_t boxes_size = BoxSize(triangles.size());
	size_t b_i = LeftNodeIndex(triangles.size());
	size_t b_i_ = b_i;
	// init
	for (size_t t_i = 0; t_i < triangles.size(); t_i++, b_i++)
	{
		boxes[b_i].world_min = triangles[t_i].MinWorldBounding();
		boxes[b_i].world_max = triangles[t_i].MaxWorldBounding();
		boxes[b_i].screen_min = triangles[t_i].MinScreenBounding();
		boxes[b_i].screen_max = triangles[t_i].MaxScreenBounding();
		boxes[b_i].triangle_count = 1;
		boxes[b_i].triangle_index = t_i;
	}

	for(b_i; b_i < boxes_size; b_i++)
	{
		boxes[b_i].world_min = { DBL_MAX, DBL_MAX, DBL_MAX, 1 };
		boxes[b_i].world_max = { -DBL_MAX, -DBL_MAX, -DBL_MAX, 1 };
		boxes[b_i].screen_min = { DBL_MAX, DBL_MAX, DBL_MAX, 1 };
		boxes[b_i].screen_max = { -DBL_MAX, -DBL_MAX, -DBL_MAX, 1 };
		boxes[b_i].triangle_count = 0;
	}
	

	for (b_i = b_i_ - 1; b_i >= 0; b_i--)
	{
		__BoundingBox::Merge(boxes[LeftChildIndex(b_i)], boxes[RightChildIndex(b_i)], boxes[b_i]);
		if(b_i == 0) 
			break;
	}
}


bool __BoundingBox::IsThrough(BoundingBox const& box, Maths::Vector const& location, Maths::Vector const& direction)
{
	// TODO
	// solve
	// (x - x0) / dx = (y - y0) / dy = (z - z0) / dz
	// (for example) x = box.world_min[0]
	//

	using namespace __IsThrough;
	if (IsThroughPlane(location, direction, box, X_AXIS, box.world_min[X_AXIS])) return true;
	if (IsThroughPlane(location, direction, box, X_AXIS, box.world_max[X_AXIS])) return true;
	if (IsThroughPlane(location, direction, box, Y_AXIS, box.world_min[Y_AXIS])) return true;
	if (IsThroughPlane(location, direction, box, Y_AXIS, box.world_max[Y_AXIS])) return true;
	if (IsThroughPlane(location, direction, box, Z_AXIS, box.world_min[Z_AXIS])) return true;
	if (IsThroughPlane(location, direction, box, Z_AXIS, box.world_max[Z_AXIS])) return true;

	return false;
	
}

void BoundingBox$::MayThrough(
	BoundingBox* boxes, 
	size_t b_i, 
	Utils::List<Triangle3D> const& triangles, 
	Maths::Vector const& location, 
	Maths::Vector const& direction, 
	BlinnPhongReflectionModel$::PointLightBufferItem const& light_buffer_item, 
	void (*build_per_triangle_light_pixel)(
		BlinnPhongReflectionModel& bpr_model, 
		size_t x, 
		size_t y, 
		__::Triangle3D& triangle, 
		size_t point_light_index, 
		FrameBuffer& buffer), 
	BlinnPhongReflectionModel& bpr_model, 
	size_t x, 
	size_t y,
	size_t point_light_index, 
	FrameBuffer& buffer)
{
	Utils::ArrayStack<size_t> stack;
	stack.Push(b_i);
	while (!stack.IsEmpty())
	{
		b_i = stack.Pop();

		if (boxes[b_i].triangle_count == 0) continue;

		if (!light_buffer_item.is_exposed) return;

		if (!__BoundingBox::IsThrough(boxes[b_i], location, direction)) continue;

		if (boxes[b_i].triangle_count == 1)
		{
			build_per_triangle_light_pixel(bpr_model, x, y, triangles.data[boxes[b_i].triangle_index], point_light_index, buffer);
			if (!light_buffer_item.is_exposed) return;
			continue;
		}

		stack.Push(LeftChildIndex(b_i));
		stack.Push(RightChildIndex(b_i));
	}
	
}

void BoundingBox$::MayScreenCover(
	BoundingBox* boxes,
	size_t b_i,
	Utils::List<__::Triangle3D> const& triangles,
	size_t x,
	size_t y,
	void (*write_to_pixel_per_triangle)(
		__::Triangle3D& triangle, 
		size_t x,
		size_t y,
		FrameBuffer& buffer,
		double nearest_dist,
		Object* cuda_objects),
	FrameBuffer& buffer,
	double nearest_dist,
	Object* cuda_objects)
{
	Utils::ArrayStack<size_t> stack;
	stack.Push(b_i);
	while (!stack.IsEmpty())
	{
		b_i = stack.Pop();

		if (boxes[b_i].triangle_count == 0) continue;

		if (x < boxes[b_i].screen_min[0] ||
			x > boxes[b_i].screen_max[0] ||
			y < boxes[b_i].screen_min[1] ||
			y > boxes[b_i].screen_max[1]) // is not be covered
		{
			continue;
		}

		if (boxes[b_i].triangle_count == 1)
		{
			write_to_pixel_per_triangle(triangles.data[boxes[b_i].triangle_index], x, y, buffer, nearest_dist, cuda_objects);
			continue;
		}
		

		stack.Push(LeftChildIndex(b_i));
		stack.Push(RightChildIndex(b_i));
	}
}