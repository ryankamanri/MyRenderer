#include "kamanri/renderer/world/bling_phong_reflection_model.hpp"
#include "kamanri/utils/string.hpp"
using namespace Kamanri::Renderer::World;
using namespace Kamanri::Utils;
using namespace Kamanri::Maths;
namespace Kamanri
{
    namespace Renderer
    {
        namespace World
        {
            namespace __BlingPhongReflectionModel
            {
                constexpr const char* LOG_NAME = STR(Kamanri::Renderer::World::BlingPhongReflectionModel);
            } // namespace __BlingPhongReflectionModel
            
        } // namespace World
        
    } // namespace Renderer
    
} // namespace Kamanri

#define Scan_R270(height, x, y) ((height - (y + 1)) * height + x)
#define LightBufferLoc(width, height, index, x, y) (width * height * index + Scan_R270(height, x, y))

using namespace Kamanri::Renderer::World::BlingPhongReflectionModel$;


BlingPhongReflectionModel::BlingPhongReflectionModel(std::vector<BlingPhongReflectionModel$::PointLight>&& point_lights, size_t screen_width, size_t screen_height, double specular_min_cos, double diffuse_factor, double ambient_factor)
{
	_point_lights = std::move(point_lights);
	_specular_min_cos = specular_min_cos;
	_diffuse_factor = diffuse_factor;
	_ambient_factor = ambient_factor;
	_screen_width = screen_width;
	_screen_height = screen_height;
    _lights_buffer = NewArray<PointLightBufferItem>(_point_lights.size() * screen_width * screen_height);
}

BlingPhongReflectionModel::BlingPhongReflectionModel(BlingPhongReflectionModel&& other)
{
    _point_lights = std::move(other._point_lights);
	_screen_width = other._screen_width;
	_screen_height = other._screen_height;
    _specular_min_cos = other._specular_min_cos;
	_diffuse_factor = other._diffuse_factor;
	_ambient_factor = other._ambient_factor;
	_lights_buffer = std::move(other._lights_buffer);

}

BlingPhongReflectionModel& BlingPhongReflectionModel::operator=(BlingPhongReflectionModel&& other)
{
    _point_lights = std::move(other._point_lights);
	_screen_width = other._screen_width;
	_screen_height = other._screen_height;
    _specular_min_cos = other._specular_min_cos;
	_diffuse_factor = other._diffuse_factor;
	_ambient_factor = other._ambient_factor;
	_lights_buffer = std::move(other._lights_buffer);
	return *this;
}

void BlingPhongReflectionModel::ModelViewTransform(Maths::SMatrix const& matrix)
{
	for(size_t i = 0; i < _point_lights.size(); i++)
	{
		_point_lights[i].location_model_view_transformed = _point_lights[i].location;
		matrix * _point_lights[i].location_model_view_transformed;
	}
}

void BlingPhongReflectionModel::InitLightBufferPixel(size_t x, size_t y, FrameBuffer& buffer)
{
	for(size_t i = 0; i < _point_lights.size(); i++)
	{
		auto& this_item = _lights_buffer[LightBufferLoc(_screen_width, _screen_height, i, x, y)];
		this_item.distance = DBL_MAX;
		this_item.is_exposed = true;
		this_item.is_specular = false;
	}
	
}

#define SpecularTransition(min_theta, theta) pow((theta - min_theta) / (1 - min_theta), 3)

void BlingPhongReflectionModel::__BuildPerTrianglePixel(size_t x, size_t y, __::Triangle3D& triangle, FrameBuffer& buffer)
{

	if (triangle.Index() == buffer.triangle_index)
	{
		for (size_t i = 0; i < _point_lights.size(); i++)
		{
			auto distance = _point_lights[i].location_model_view_transformed - buffer.location;
			auto& light_buffer_item = _lights_buffer[LightBufferLoc(_screen_width, _screen_height, i, x, y)];
			if (distance < light_buffer_item.distance) light_buffer_item.distance = distance;

			// judge whether is specular
			// camera is at (0, 0, 0, 1)
			auto point_camera_add_point_light_vector = _point_lights[i].location_model_view_transformed;
			point_camera_add_point_light_vector += { 0, 0, 0, 1 };
			point_camera_add_point_light_vector -= buffer.location;
			point_camera_add_point_light_vector -= buffer.location;

			point_camera_add_point_light_vector.Unitization();
			auto cos_theta = point_camera_add_point_light_vector * buffer.vertex_normal;
			if(cos_theta >= _specular_min_cos)
			{
				light_buffer_item.is_specular = true;
				light_buffer_item.specular_factor = SpecularTransition(_specular_min_cos, cos_theta);
			} 
		}
		return;
	}

	for (size_t i = 0; i < _point_lights.size(); i++)
	{
		auto& light_buffer_item = _lights_buffer[LightBufferLoc(_screen_width, _screen_height, i, x, y)];
		auto& light_location = _point_lights[i].location_model_view_transformed;
		auto light_point_direction = buffer.location;
		light_point_direction -= light_location;
		double light_point_distance = light_location - buffer.location;
		double distance;
		if(triangle.IsThrough(light_location, light_point_direction, distance))
		{
			if(distance < light_point_distance)
			{
				light_buffer_item.distance = distance;
				light_buffer_item.is_exposed = false;
			}
		}
	}

}

#define GenerizeReflection(r, g, b, factor) CombineRGB((unsigned int)(r * factor), (unsigned int)(g * factor), (unsigned int)(b * factor))


// inline RGB SpecularReflection()
// {
// 	for(size_t i = 0; i < _point_lights.size(); i++)
// 	{
// 		GenerizeReflection(_power * light_buffer_item.is_specular);
// 	}
	
// } 
// #define DiffuseReflection(light_buffer_item) GenerizeReflection(_power / _diffuse_factor * light_buffer_item.is_exposed)
// #define AmbientReflection(light_buffer_item) GenerizeReflection(_ambient_factor)

/// @brief Require normal unitized.
/// @param location 
/// @param normal 
/// @param reflect_point 
void BlingPhongReflectionModel::WriteToPixel(size_t x, size_t y, FrameBuffer& buffer, DWORD& pixel)
{
    buffer.r = buffer.g = buffer.b = 0;
	buffer.power = 0;
	buffer.specular_color = buffer.diffuse_color = buffer.ambient_color = 0;
    for(size_t i = 0; i < _point_lights.size(); i++)
    {
		// Do
		auto& light_buffer_item = _lights_buffer[LightBufferLoc(_screen_width, _screen_height, i, x, y)];
		auto distance = _point_lights[i].location_model_view_transformed - buffer.location;
		auto direction = _point_lights[i].location_model_view_transformed;
		direction -= buffer.location;
		direction.Unitization();

		// power = theta / S * cos(theta)
		auto cos_theta = (buffer.vertex_normal * direction);
				
		if (cos_theta <= 0) continue;
		
		auto power = (_point_lights[i].power / (4 * Maths::PI * pow(distance, 2))) * cos_theta;
		buffer.power += power;

		auto receive_light_color = RGBMul(_point_lights[i].color, power);
        DivideRGB(
			RGBReflect(receive_light_color, buffer.color), 
			buffer.r, buffer.g, buffer.b, 
			[](unsigned int& y, RGB x){ y += x; }
		);
		
		buffer.specular_color += GenerizeReflection(buffer.r, buffer.g, buffer.b, power * light_buffer_item.specular_factor * light_buffer_item.is_specular * light_buffer_item.is_exposed);
		buffer.diffuse_color += GenerizeReflection(buffer.r, buffer.g, buffer.b, power * _diffuse_factor * light_buffer_item.is_exposed);
		
		if(light_buffer_item.is_exposed)
		{
			Log::Debug(__BlingPhongReflectionModel::LOG_NAME, "buffer(%llu, %llu) diffuse color: %6.X", x, y, buffer.diffuse_color);
		}
		if(light_buffer_item.is_specular && light_buffer_item.is_exposed)
		{
			Log::Debug(__BlingPhongReflectionModel::LOG_NAME, "buffer(%llu, %llu) specular color: %6.X", x, y, buffer.specular_color);
		}

    }

	buffer.ambient_color += RGBMul(buffer.color, _ambient_factor);

	pixel = RGBAdd(buffer.ambient_color, buffer.diffuse_color, buffer.specular_color);

	// Log::Debug(__BlingPhongReflectionModel::LOG_NAME, 
	// "specular color: %X, diffuse color: %X, ambient color: %X, color: %X", 
	// buffer.specular_color, buffer.diffuse_color, buffer.ambient_color, pixel);

}


