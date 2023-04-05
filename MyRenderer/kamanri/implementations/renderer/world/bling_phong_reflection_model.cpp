#include "kamanri/renderer/world/bling_phong_reflection_model.hpp"
#include "kamanri/utils/string.hpp"
#include "cuda_dll/exports/memory_operations.hpp"
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
				
				dll cuda_dll;
				func_type(CUDAMalloc) cuda_malloc;
				func_type(CUDAFree) cuda_free;
				func_type(TransmitToCUDA) transmit_to_cuda;
				func_type(TransmitFromCUDA) transmit_from_cuda;

				void ImportFunctions()
				{
					load_dll(cuda_dll, cuda_dll, LOG_NAME);
					import_func(CUDAMalloc, cuda_dll, cuda_malloc, LOG_NAME);
					import_func(CUDAFree, cuda_dll, cuda_free, LOG_NAME);
					import_func(TransmitToCUDA, cuda_dll, transmit_to_cuda, LOG_NAME);
					import_func(TransmitFromCUDA, cuda_dll, transmit_from_cuda, LOG_NAME);
				}

				inline size_t Scan_R270(size_t height, size_t x, size_t y)
				{
					return ((height - (y + 1)) * height + x);
				}

				inline size_t LightBufferLoc(size_t width, size_t height, size_t index, size_t x, size_t y)
				{
					return (width * height * index + Scan_R270(height, x, y));
				}

				inline double SpecularTransition(double min_theta,  double theta)
				{
					return pow((theta - min_theta) / (1 - min_theta), 3);
				}

				inline RGB GenerizeReflection(unsigned int r, unsigned int g, unsigned int b, double factor)
				{
					return BlingPhongReflectionModel$::CombineRGB((unsigned int)(r * factor), (unsigned int)(g * factor), (unsigned int)(b * factor));
				}

            } // namespace __BlingPhongReflectionModel
            
        } // namespace World
        
    } // namespace Renderer
    
} // namespace Kamanri


using namespace Kamanri::Renderer::World::BlingPhongReflectionModel$;


BlingPhongReflectionModel::BlingPhongReflectionModel(std::vector<BlingPhongReflectionModel$::PointLight>&& point_lights, size_t screen_width, size_t screen_height, double specular_min_cos, double diffuse_factor, double ambient_factor, bool is_use_cuda)
{
	_point_lights = std::move(point_lights);
	_specular_min_cos = specular_min_cos;
	_diffuse_factor = diffuse_factor;
	_ambient_factor = ambient_factor;
	_screen_width = screen_width;
	_screen_height = screen_height;
    _lights_buffer = NewArray<PointLightBufferItem>(_point_lights.size() * screen_width * screen_height);
	_is_use_cuda = is_use_cuda;


	if(!is_use_cuda) return;
	
	__BlingPhongReflectionModel::ImportFunctions();

	auto point_lights_size = _point_lights.size();
	__BlingPhongReflectionModel::cuda_malloc(&(void*)_cuda_point_lights_size, sizeof(size_t));
	__BlingPhongReflectionModel::transmit_to_cuda(&point_lights_size, _cuda_point_lights_size, sizeof(size_t));
	__BlingPhongReflectionModel::cuda_malloc(&(void*)_cuda_point_lights, point_lights_size * sizeof(PointLight));
	__BlingPhongReflectionModel::transmit_to_cuda(&_point_lights[0], _cuda_point_lights, point_lights_size * sizeof(PointLight));

	auto lights_buffer_size = _point_lights.size() * _screen_width * _screen_height;
	__BlingPhongReflectionModel::cuda_malloc(&(void*)_cuda_lights_buffer, lights_buffer_size * sizeof(PointLightBufferItem));
}

BlingPhongReflectionModel::~BlingPhongReflectionModel()
{

}

void BlingPhongReflectionModel::DeleteCUDA()
{
	__BlingPhongReflectionModel::cuda_free(_cuda_lights_buffer);
	__BlingPhongReflectionModel::cuda_free(_cuda_point_lights);
	__BlingPhongReflectionModel::cuda_free(_cuda_point_lights_size);
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

	_cuda_lights_buffer = other._cuda_lights_buffer;
	_cuda_point_lights = other._cuda_point_lights;
	_cuda_point_lights_size = other._cuda_point_lights_size;

	_is_use_cuda = other._is_use_cuda;
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

	_cuda_lights_buffer = other._cuda_lights_buffer;
	_cuda_point_lights = other._cuda_point_lights;
	_cuda_point_lights_size = other._cuda_point_lights_size;

	_is_use_cuda = other._is_use_cuda;
	return *this;
}

void BlingPhongReflectionModel::ModelViewTransform(Maths::SMatrix const& matrix)
{
	for(size_t i = 0; i < _point_lights.size(); i++)
	{
		_point_lights[i].location_model_view_transformed = _point_lights[i].location;
		matrix * _point_lights[i].location_model_view_transformed;
	}

	if(!_is_use_cuda) return;
	__BlingPhongReflectionModel::transmit_to_cuda(&_point_lights[0], _cuda_point_lights, _point_lights.size() * sizeof(PointLight));
}

void BlingPhongReflectionModel::InitLightBufferPixel(size_t x, size_t y, FrameBuffer& buffer)
{
	using namespace __BlingPhongReflectionModel;
	for(size_t i = 0; i < _point_lights.size(); i++)
	{
		auto& this_item = _lights_buffer[LightBufferLoc(_screen_width, _screen_height, i, x, y)];
		this_item.distance = DBL_MAX;
		this_item.is_exposed = true;
		this_item.is_specular = false;
	}
	
}



void BlingPhongReflectionModel::__BuildPerTrianglePixel(size_t x, size_t y, __::Triangle3D& triangle, FrameBuffer& buffer)
{
	using namespace __BlingPhongReflectionModel;

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


/// @brief Require normal unitized.
/// @param location 
/// @param normal 
/// @param reflect_point 
void BlingPhongReflectionModel::WriteToPixel(size_t x, size_t y, FrameBuffer& buffer, DWORD& pixel)
{
	using namespace __BlingPhongReflectionModel;
	
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
		
		// if(light_buffer_item.is_exposed)
		// {
		// 	Log::Debug(__BlingPhongReflectionModel::LOG_NAME, "buffer(%llu, %llu) diffuse color: %6.X", x, y, buffer.diffuse_color);
		// }
		// if(light_buffer_item.is_specular && light_buffer_item.is_exposed)
		// {
		// 	Log::Debug(__BlingPhongReflectionModel::LOG_NAME, "buffer(%llu, %llu) specular color: %6.X", x, y, buffer.specular_color);
		// }

    }

	buffer.ambient_color += RGBMul(buffer.color, _ambient_factor);

	pixel = RGBAdd(buffer.ambient_color, buffer.diffuse_color, buffer.specular_color);

}


