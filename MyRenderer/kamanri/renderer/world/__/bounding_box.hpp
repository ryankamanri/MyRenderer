#pragma once
#include <vector>
#include "kamanri/utils/array_stack.hpp"
#include "kamanri/utils/list.hpp"
#include "kamanri/maths/vector.hpp"
#include "triangle3d.hpp"
#include "kamanri/renderer/world/blinn_phong_reflection_model.hpp"

namespace Kamanri
{
	namespace Renderer
	{
		namespace World
		{
			namespace __
			{

				class BoundingBox
				{
					public:
					Maths::Vector world_min;
					Maths::Vector world_max;
					Maths::Vector screen_min;
					Maths::Vector screen_max;
					size_t triangle_count = 0;
					size_t triangle_index = 0;
				};

				namespace __BoundingBox
				{
					void Merge(BoundingBox const& box1, BoundingBox const& box2, BoundingBox& out_box);
#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif
						bool IsThrough(BoundingBox const& box, Maths::Vector const& location, Maths::Vector const& direction);
				}

				namespace BoundingBox$
				{
					inline size_t LeftNodeSize(size_t triangles_size)
					{
						size_t left_node_size = 1;
						while (left_node_size < triangles_size) left_node_size <<= 1;

						return left_node_size;
					}

					inline size_t LeftNodeIndex(size_t triangles_size)
					{
						return LeftNodeSize(triangles_size) - 1;
					}

					inline size_t BoxSize(size_t triangles_size)
					{
						return LeftNodeSize(triangles_size) * 2 - 1;
					}
#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif
					inline size_t LeftChildIndex(size_t b_index)
					{
						return 2 * b_index + 1;
					}
#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif
					inline size_t RightChildIndex(size_t b_index)
					{
						return 2 * b_index + 2;
					}

					void Build(BoundingBox* boxes, std::vector<Triangle3D> const& triangles);
#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif
						void MayThrough(
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
							FrameBuffer& buffer);

#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif
						void MayScreenCover(
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
							Object* cuda_objects = nullptr);

				} // namespace BoundingBox$


			} // namespace __

		} // namespace World

	} // namespace Renderer

} // namespace Kamanri
