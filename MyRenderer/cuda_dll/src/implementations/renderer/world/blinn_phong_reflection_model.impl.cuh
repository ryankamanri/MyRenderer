#pragma once
#include "kamanri/renderer/world/blinn_phong_reflection_model.hpp"

namespace Kamanri
{
	namespace Renderer
	{
		namespace World
		{
			namespace __BlinnPhongReflectionModel
			{
				__device__ inline size_t Scan_R270(size_t height, size_t x, size_t y)
				{
					return ((height - (y + 1)) * height + x);
				}

				__device__ inline size_t LightBufferLoc(size_t width, size_t height, size_t index, size_t x, size_t y)
				{
					return (width * height * index + Scan_R270(height, x, y));
				}

				__device__ inline double SpecularTransition(double min_theta,  double theta)
				{
					return pow((theta - min_theta) / (1 - min_theta), 3);
				}

				__device__ inline RGB GenerizeReflection(unsigned int r, unsigned int g, unsigned int b, double factor)
				{
					return BlinnPhongReflectionModel$::CombineRGB((unsigned int)(r * factor), (unsigned int)(g * factor), (unsigned int)(b * factor));
				}
			} // namespace __BlinnPhongReflectionModel
			
		} // namespace World
		
	} // namespace Renderer
	
} // namespace Kamanri



__device__ void Kamanri::Renderer::World::BlinnPhongReflectionModel::InitLightBufferPixel(size_t x, size_t y, FrameBuffer& buffer)
{
	using namespace __BlinnPhongReflectionModel;
	for (size_t i = 0; i < _cuda_point_lights.size; i++)
	{
		auto& this_item = _cuda_lights_buffer[LightBufferLoc(_screen_width, _screen_height, i, x, y)];
		this_item.distance = DBL_MAX;
		this_item.is_exposed = true;
		this_item.is_specular = false;
	}

}

__device__ void Kamanri::Renderer::World::BlinnPhongReflectionModel::__BuildPerTriangleLightPixel(size_t x, size_t y, __::Triangle3D& triangle, size_t point_light_index, FrameBuffer& buffer)
{
	using namespace __BlinnPhongReflectionModel;
	auto& light_buffer_item = _cuda_lights_buffer[LightBufferLoc(_screen_width, _screen_height, point_light_index, x, y)];
	auto& light_location = _cuda_point_lights.data[point_light_index].location_model_view_transformed;
	auto light_point_distance = light_location - buffer.location;
	if (triangle.Index() == buffer.triangle_index)
	{
		if (light_point_distance < light_buffer_item.distance) light_buffer_item.distance = light_point_distance;

		// judge whether is specular
		// camera is at (0, 0, 0, 1)
		auto point_camera_add_point_light_vector = light_location;
		point_camera_add_point_light_vector += { 0, 0, 0, 1 };
		point_camera_add_point_light_vector -= buffer.location;
		point_camera_add_point_light_vector -= buffer.location;

		point_camera_add_point_light_vector.Unitization();
		auto cos_theta = point_camera_add_point_light_vector * buffer.vertex_normal;
		if (cos_theta >= _specular_min_cos)
		{
			light_buffer_item.is_specular = true;
			light_buffer_item.specular_factor = SpecularTransition(_specular_min_cos, cos_theta);
		}
	}
	else
	{
		auto light_point_direction = buffer.location;
		light_point_direction -= light_location;

		double light_triangle_distance;
		if (triangle.IsThrough(light_location, light_point_direction, light_triangle_distance))
		{
			if (light_triangle_distance < light_point_distance)
			{
				light_buffer_item.distance = light_triangle_distance;
				light_buffer_item.is_exposed = false;
			}
		}
	}
}

__device__ void Kamanri::Renderer::World::BlinnPhongReflectionModel::__BuildPerTrianglePixel(size_t x, size_t y, __::Triangle3D& triangle, FrameBuffer& buffer)
{
	for (size_t i = 0; i < _cuda_point_lights.size; i++)
	{
		__BuildPerTriangleLightPixel(x, y, triangle, i, buffer);
	}

}

__device__ void Kamanri::Renderer::World::BlinnPhongReflectionModel::__BuildShadowPixel(size_t x, size_t y, Utils::List<__::Triangle3D> triangles, __::BoundingBox* boxes, FrameBuffer& buffer)
{
	// Utils::ArrayStack<size_t> triangle_index_stack;
	using namespace __BlinnPhongReflectionModel;
	for (size_t i = 0; i < _cuda_point_lights.size; i++)
	{
		auto& light_location = _cuda_point_lights.data[i].location_model_view_transformed;
		auto& light_buffer_item = _cuda_lights_buffer[LightBufferLoc(_screen_width, _screen_height, i, x, y)];
		auto light_point_direction = buffer.location;
		light_point_direction -= light_location;
		__::BoundingBox$::MayThrough(
			boxes, 
			0, 
			triangles, 
			light_location, 
			light_point_direction, 
			light_buffer_item, 
			[](BlinnPhongReflectionModel& bpr_model, 
			size_t x, 
			size_t y, 
			__::Triangle3D& triangle, 
			size_t point_light_index, 
			FrameBuffer& buffer){
				bpr_model.__BuildPerTriangleLightPixel(x, y, triangle, point_light_index, buffer);
			}, *this, x, y, i, buffer);
		
	}
	
}


/// @brief Require normal unitized.
/// @param location 
/// @param normal 
/// @param reflect_point 
__device__ void Kamanri::Renderer::World::BlinnPhongReflectionModel::WriteToPixel(size_t x, size_t y, FrameBuffer& buffer, DWORD& pixel)
{
	using namespace __BlinnPhongReflectionModel;
	buffer.r = buffer.g = buffer.b = 0;
	buffer.power = 0;
	buffer.specular_color = buffer.diffuse_color = buffer.ambient_color = 0;
	for (size_t i = 0; i < _cuda_point_lights.size; i++)
	{
		// Do
		auto& light_buffer_item = _cuda_lights_buffer[LightBufferLoc(_screen_width, _screen_height, i, x, y)];
		auto distance = _cuda_point_lights.data[i].location_model_view_transformed - buffer.location;
		auto direction = _cuda_point_lights.data[i].location_model_view_transformed;
		direction -= buffer.location;
		direction.Unitization();

		// power = theta / S * cos(theta)
		auto cos_theta = (buffer.vertex_normal * direction);

		if (cos_theta <= 0) continue;

		auto power = (_cuda_point_lights.data[i].power / (4 * Maths::PI * pow(distance, 2))) * cos_theta;
		buffer.power += power;

		auto receive_light_color = BlinnPhongReflectionModel$::RGBMul(_cuda_point_lights.data[i].color, power);
		BlinnPhongReflectionModel$::DivideRGB(
			BlinnPhongReflectionModel$::RGBReflect(receive_light_color, buffer.color),
			buffer.r, buffer.g, buffer.b,
			BlinnPhongReflectionModel$::AddHandle
		);

		buffer.specular_color += GenerizeReflection(buffer.r, buffer.g, buffer.b, power * light_buffer_item.specular_factor * light_buffer_item.is_specular * light_buffer_item.is_exposed);
		buffer.diffuse_color += GenerizeReflection(buffer.r, buffer.g, buffer.b, power * _diffuse_factor * light_buffer_item.is_exposed);

		// if(light_buffer_item.is_exposed)
		// {
		// 	Log::Debug(__BlinnPhongReflectionModel::LOG_NAME, "buffer(%llu, %llu) diffuse color: %6.X", x, y, buffer.diffuse_color);
		// }
		// if(light_buffer_item.is_specular && light_buffer_item.is_exposed)
		// {
		// 	Log::Debug(__BlinnPhongReflectionModel::LOG_NAME, "buffer(%llu, %llu) specular color: %6.X", x, y, buffer.specular_color);
		// }

	}

	buffer.ambient_color += BlinnPhongReflectionModel$::RGBMul(buffer.color, _ambient_factor);

	pixel = BlinnPhongReflectionModel$::RGBAdd(buffer.ambient_color, buffer.diffuse_color, buffer.specular_color);

	// DevicePrint("%X ", pixel);
}
