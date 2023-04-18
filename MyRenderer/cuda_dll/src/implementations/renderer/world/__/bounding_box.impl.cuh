#pragma once
#include "kamanri/renderer/world/__/bounding_box.hpp"


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
						__device__ inline bool IsThroughPlane(Maths::Vector const& location, Maths::Vector const& direction, BoundingBox const& box, AxisType axis_type, double value)
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


__device__ bool Kamanri::Renderer::World::__::__BoundingBox::IsThrough(
	Kamanri::Renderer::World::__::BoundingBox const& box, 
	Kamanri::Maths::Vector const& location, 
	Kamanri::Maths::Vector const& direction)
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

__device__ void Kamanri::Renderer::World::__::BoundingBox$::MayThrough(
	Kamanri::Renderer::World::__::BoundingBox* boxes, 
	size_t b_i, 
	Kamanri::Utils::List<Triangle3D> const& triangles, 
	Kamanri::Maths::Vector const& location, 
	Kamanri::Maths::Vector const& direction, 
	Kamanri::Renderer::World::BlinnPhongReflectionModel$::PointLightBufferItem const& light_buffer_item, 
	void (*build_per_triangle_light_pixel)(
		Kamanri::Renderer::World::BlinnPhongReflectionModel& bpr_model, 
		size_t x, 
		size_t y, 
		Kamanri::Renderer::World::__::Triangle3D& triangle, 
		size_t point_light_index, 
		Kamanri::Renderer::World::FrameBuffer& buffer), 
	Kamanri::Renderer::World::BlinnPhongReflectionModel& bpr_model, 
	size_t x, 
	size_t y,
	size_t point_light_index, 
	Kamanri::Renderer::World::FrameBuffer& buffer)
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

__device__ void Kamanri::Renderer::World::__::BoundingBox$::MayScreenCover(
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