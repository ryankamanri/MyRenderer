#pragma once
#include "kamanri/renderer/world/__/bounding_box.hpp"

#define __IS_THROUGH__(a, b, c, d, judge_dimension_1, judge_dimension_2) 	\
{																			\
	conjunction = 															\
	{																		\
		location[0] * dydz - location[1] * dxdz,							\
		location[0] * dydz - location[2] * dxdy,							\
		d																	\
	};																		\
	sm = 																	\
	{																		\
		dydz, -dxdz, 0,														\
		dydz, 0, -dxdy,														\
		a, b, c																\
	};																		\
																			\
	(-sm) * conjunction;													\
																			\
	if(conjunction[judge_dimension_1] <= box.max[judge_dimension_1] 		\
	&& conjunction[judge_dimension_1] >= box.min[judge_dimension_1])		\
	{																		\
		return true;														\
	}																		\
																			\
	if(conjunction[judge_dimension_2] <= box.max[judge_dimension_2] 		\
	&& conjunction[judge_dimension_2] >= box.min[judge_dimension_2])		\
	{																		\
		return true;														\
	}																		\
}

__device__ bool Kamanri::Renderer::World::__::__BoundingBox::IsThrough(
	Kamanri::Renderer::World::__::BoundingBox const& box, 
	Kamanri::Maths::Vector const& location, 
	Kamanri::Maths::Vector const& direction)
{
	// TODO
	// solve
	// (x - x0) / dx = (y - y0) / dy = (z - z0) / dz
	// (for example) x = box.min[0]
	//
	auto dxdy = direction[0] * direction[1];
	auto dxdz = direction[0] * direction[2];
	auto dydz = direction[1] * direction[2];
	Maths::Vector conjunction(3);
	Maths::SMatrix sm(3);

	__IS_THROUGH__(1, 0, 0, box.min[0], 1, 2);
	__IS_THROUGH__(1, 0, 0, box.max[0], 1, 2);
	__IS_THROUGH__(0, 1, 0, box.min[1], 0, 2);
	__IS_THROUGH__(0, 1, 0, box.max[1], 0, 2);
	__IS_THROUGH__(0, 0, 1, box.min[2], 0, 1);
	__IS_THROUGH__(0, 0, 1, box.max[2], 0, 1);

	return false;
	
}

__device__ void Kamanri::Renderer::World::__::BoundingBox$::MayThrough(
	Kamanri::Renderer::World::__::BoundingBox* boxes, 
	size_t b_i, 
	Kamanri::Utils::List<Triangle3D> const& triangles, 
	Kamanri::Maths::Vector const& location, 
	Kamanri::Maths::Vector const& direction, 
	Kamanri::Renderer::World::BlingPhongReflectionModel$::PointLightBufferItem const& light_buffer_item, 
	void (*build_per_triangle_light_pixel)(
		Kamanri::Renderer::World::BlingPhongReflectionModel& bpr_model, 
		size_t x, 
		size_t y, 
		Kamanri::Renderer::World::__::Triangle3D& triangle, 
		size_t point_light_index, 
		Kamanri::Renderer::World::FrameBuffer& buffer), 
	Kamanri::Renderer::World::BlingPhongReflectionModel& bpr_model, 
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

		if (!light_buffer_item.is_exposed) continue;

		if (boxes[b_i].triangle_count <= 8)
		{
			for (size_t t_i = boxes[b_i].triangle_index; t_i < boxes[b_i].triangle_index + boxes[b_i].triangle_count; t_i++)
			{
				build_per_triangle_light_pixel(bpr_model, x, y, triangles.data[t_i], point_light_index, buffer);
			}
			continue;
		}

		if (!__BoundingBox::IsThrough(boxes[b_i], location, direction)) continue;

		stack.Push(LeftChildIndex(b_i));
		stack.Push(RightChildIndex(b_i));
	}
}