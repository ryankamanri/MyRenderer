#include "kamanri/renderer/world/__/bounding_box.hpp"
#include "kamanri/renderer/world/bling_phong_reflection_model.hpp"
#include "kamanri/utils/list.hpp"
#include "kamanri/maths/math.hpp"

using namespace Kamanri::Renderer::World::__;

void __BoundingBox::Merge(BoundingBox const& l_box, BoundingBox const& r_box, BoundingBox& out_box)
{
	out_box.min =
	{
		Maths::Min(l_box.min[0], r_box.min[0]),
		Maths::Min(l_box.min[1], r_box.min[1]),
		Maths::Min(l_box.min[2], r_box.min[2]),
		1
	};

	out_box.max =
	{
		Maths::Max(l_box.max[0], r_box.max[0]),
		Maths::Max(l_box.max[1], r_box.max[1]),
		Maths::Max(l_box.max[2], r_box.max[2]),
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
		boxes[b_i].min = triangles[t_i].MinWorldBounding();
		boxes[b_i].max = triangles[t_i].MaxWorldBounding();
		boxes[b_i].triangle_count = 1;
		boxes[b_i].triangle_index = t_i;
	}

	for(b_i; b_i < boxes_size; b_i++)
	{
		boxes[b_i].min = { DBL_MAX, DBL_MAX, DBL_MAX, 1 };
		boxes[b_i].max = { -DBL_MAX, -DBL_MAX, -DBL_MAX, 1 };
		boxes[b_i].triangle_count = 0;
	}
	

	for (b_i = b_i_ - 1; b_i >= 0; b_i--)
	{
		__BoundingBox::Merge(boxes[LeftChildIndex(b_i)], boxes[RightChildIndex(b_i)], boxes[b_i]);
		if(b_i == 0) 
			break;
	}
}

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

bool __BoundingBox::IsThrough(BoundingBox const& box, Maths::Vector const& location, Maths::Vector const& direction)
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

void BoundingBox$::MayThrough(
	BoundingBox* boxes, 
	size_t b_i, 
	Utils::List<Triangle3D> const& triangles, 
	Maths::Vector const& location, 
	Maths::Vector const& direction, 
	BlingPhongReflectionModel$::PointLightBufferItem const& light_buffer_item, 
	void (*build_per_triangle_light_pixel)(
		BlingPhongReflectionModel& bpr_model, 
		size_t x, 
		size_t y, 
		__::Triangle3D& triangle, 
		size_t point_light_index, 
		FrameBuffer& buffer), 
	BlingPhongReflectionModel& bpr_model, 
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