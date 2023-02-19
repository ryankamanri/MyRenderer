#include <cmath>
#include "kamanri/maths/math.hpp"
#include "kamanri/utils/logs.hpp"
#include "kamanri/renderer/world/camera.hpp"
#include "kamanri/maths/matrix.hpp"
#include "kamanri/utils/string.hpp"

using namespace Kamanri::Utils;
using namespace Kamanri::Renderer::World;
using namespace Kamanri::Renderer::World::Camera$;
using namespace Kamanri::Maths;


namespace Kamanri
{
	namespace Renderer
	{
		namespace World
		{
			namespace __Camera
			{
				constexpr const char *LOG_NAME = STR(Kamanri::Renderer::World::Camera);

				/**
				 * @brief FIxed asin, filtered the situation of x > 1 and x < -1
				 *
				 * @param x
				 * @return double
				 */
				inline double Arcsin(double x)
				{
					return x > 1 ? asin(1) : (x < -1 ? asin(-1) : asin(x));
				}
				

				// Camera::Transform
				namespace Transform
				{
					SMatrix model_view_transform(4);
					SMatrix projection_screen_transform(4);
				} // namespace Transform
				

				// Camera::InverseUpperByDirection
				namespace InverseUpperByDirection
				{
					Vector upward_before = {0, 1, 0, 0};
					Vector upward_after = {0, 1, 0, 0};
				} // namespace InverseUpperByDirection
				

			} // namespace __Camera
			
		} // namespace World
		
	} // namespace Renderer
	
} // namespace Kamanri

using namespace Kamanri::Renderer::World::__Camera;

Camera::Camera()
{
	_location = Vector(4);
	_direction = Vector(4);
	_upward = Vector(4);
}

Camera::Camera(Vector location, Vector direction, Vector upper, double nearest_dist, double furthest_dist, unsigned int screen_width, unsigned int screen_height) : _nearest_dist(nearest_dist), _furthest_dist(furthest_dist), _screen_width(screen_width), _screen_height(screen_height)
{
	if (location.N() != 4 || direction.N() != 4 || upper.N() != 4)
	{
		Log::Error(__Camera::LOG_NAME, "Invalid vector length. location length: %d, direction length: %d, upper length: %d",
				   location.N(), direction.N(), upper.N());
				   PRINT_LOCATION;
		return;
	}

	if (location[3] != 1 || direction[3] != 0 || upper[3] != 0)
	{
		Log::Error(__Camera::LOG_NAME, "Invalid vector type. valid location/direction/upper type: 1/0/0 but given %.0f/%.0f/%.0f", location[3], direction[3], upper[3]);
		PRINT_LOCATION;
		return;
	}

	_location = location;
	auto dir_unit_res = direction.Unitization();
	if (dir_unit_res)
	{
		Log::Error(__Camera::LOG_NAME, "Failed to unitization direction");
		PRINT_LOCATION;
	}
	_direction = direction;

	auto upp_set_res = upper.Set(2, 0);
	auto upp_unit_res = upper.Unitization();
	if (upp_set_res || upp_unit_res)
	{
		Log::Error(__Camera::LOG_NAME, "Failed to set or unitization");
		PRINT_LOCATION;
	}
	_upward = upper;

	SetAngles();
}



void Camera::SetAngles()
{
	_beta = Arcsin(_direction[1]); // beta (-PI/2, PI/2)

	_alpha = Arcsin((_direction[0]) / cos(_beta)); // alpha (-PI, PI)

	if(_direction[2] > 0)
	{
		if(_alpha > 0) 
			_alpha = PI - _alpha;
		else
			_alpha = -PI - _alpha;
	}

	_gamma = Arcsin(_upward[0]); // gamma (-PI, PI)

	if(_upward[1] < 0)
	{
		if(_gamma > 0)
			_gamma = PI - _gamma;
		else
			_gamma = -PI - _gamma;
	}

	Log::Trace(__Camera::LOG_NAME, "The direction vector:");

	_direction.PrintVector(Log$::TRACE_LEVEL);

	Log::Trace(__Camera::LOG_NAME, "alpha = %.2f, beta = %.2f, gamma = %.2f",
		_alpha, _beta, _gamma);
}

Camera::Camera(Camera && camera) : _pvertices(camera._pvertices), _alpha(camera._alpha), _beta(camera._beta), _gamma(camera._gamma), _nearest_dist(camera._nearest_dist), _furthest_dist(camera._furthest_dist), _screen_width(camera._screen_width), _screen_height(camera._screen_height)
{
	_location = std::move(camera._location);
	_direction = std::move(camera._direction);
	_upward = std::move(camera._upward);
}

Camera& Camera::operator=(Camera&& other)
{
	_pvertices = other._pvertices;
	_pvertices_transformed = other._pvertices_transformed;
	_alpha = other._alpha;
	_beta = other._beta;
	_gamma = other._gamma;
	_nearest_dist = other._nearest_dist;
	_furthest_dist = other._furthest_dist;
	_screen_width = other._screen_width;
	_screen_height = other._screen_height;
	_location = std::move(other._location);
	_direction = std::move(other._direction);
	_upward = std::move(other._upward);
	return *this;
}

void Camera::SetVertices(std::vector<Maths::Vector> &vertices, std::vector<Maths::Vector> &vertices_transformed, std::vector<Maths::Vector> &vertices_model_view_transformed)
{
	_pvertices = &vertices;
	_pvertices_transformed = &vertices_transformed;
	_pvertices_model_view_transformed = &vertices_model_view_transformed;
}

#define min(a,b) (((a) < (b)) ? (a) : (b))

DefaultResult Camera::Transform()
{
	CHECK_MEMORY_FOR_DEFAULT_RESULT(_pvertices, __Camera::LOG_NAME, Camera$::CODE_NULL_POINTER_PVERTICES)

	SetAngles();

	Log::Trace(__Camera::LOG_NAME, "vertices count: %d", _pvertices->size());
	//
	auto sin_a = sin(_alpha);
	auto cos_a = cos(_alpha);
	auto sin_b = sin(_beta);
	auto cos_b = cos(_beta);
	auto sin_a_sin_b = sin_a * sin_b;
	auto sin_a_cos_b = sin_a * cos_b;
	auto cos_a_sin_b = cos_a * sin_b;
	auto cos_a_cos_b = cos_a * cos_b;
	auto sin_g = sin(_gamma);
	auto cos_g = cos(_gamma);
	auto lx = _location[0];
	auto ly = _location[1];
	auto lz = _location[2];


	Log::Trace(__Camera::LOG_NAME, "a5*4*3 * a2*1 * v:");

	// SMatrix model_location_transform = 
	// {
	//     1, 0, 0, -lx,
	//     0, 1, 0, -ly,
	//     0, 0, 1, -lz,
	//     0, 0, 0, 1  
	// };

	// SMatrix view_direction_transform = 
	// {
	//     cos_a, 0, sin_a, 0,
	//     -sin_a_sin_b, cos_b, cos_a_sin_b, 0,
	//     -sin_a_cos_b, -sin_b, cos_a_cos_b, 0,
	//     0, 0, 0, 1
	// };


	// SMatrix view_upper_transform = 
	// {
	//     cos_g, -sin_g, 0, 0,
	//     sin_g, cos_g, 0, 0,
	//     0, 0, 1, 0,
	//     0, 0, 0, 1
	// };


	using namespace __Camera::Transform;

	model_view_transform = 
	{
		cos_a*cos_g + sin_a_sin_b*sin_g, -cos_b*sin_g, sin_a*cos_g - cos_a_sin_b*sin_g, lx*(-cos_a*cos_g-sin_a_sin_b*sin_g) + ly*cos_b*sin_g + lz*(-sin_a*cos_g+cos_a_sin_b*sin_g),
		cos_a*sin_g - sin_a_sin_b*cos_g, cos_b*cos_g, sin_a*sin_g + cos_a_sin_b*cos_g, lx*(-cos_a*sin_g+sin_a_sin_b*cos_g) - ly*cos_b*cos_g + lz*(-sin_a*sin_g-cos_a_sin_b*cos_g),
		-sin_a_cos_b, -sin_b, cos_a_cos_b, lx*sin_a_cos_b + ly*sin_b - lz*cos_a_cos_b,
		0, 0, 0, 1
	};

	// SMatrix projection_transform = 
	// {
	//     _nearest_dist, 0, 0, 0,
	//     0, _nearest_dist, 0, 0,
	//     0, 0, _nearest_dist + _furthest_dist, -_nearest_dist * _furthest_dist,
	//     0, 0, 1, 0 
	// };

	// SMatrix screen_fit_transform = 
	// {
	//     _screen_width / 2, 0, 0, _screen_width / 2,
	//     0, -_screen_height / 2, 0, _screen_height / 2,
	//     0, 0, 1, 0, // not change z
	//     0, 0, 0, 1
	// };

	// SMatrix screen_fit_transform = 
	// {
	//     _screen_width / 2, 0, 0, _screen_width / 2,
	//     0, -_screen_height / 2, 0, _screen_height / 2,
	//     0, 0, min(_screen_width, _screen_height) / 2, 0,
	//     0, 0, 0, 1
	// };

	projection_screen_transform = 
	{
		(double)_screen_width * _nearest_dist / 2, 0, (double)_screen_width / 2, 0,
		0, -(double)_screen_height * _nearest_dist * cos_g / 2, (double)_screen_height / 2, 0,
		0, 0, (_nearest_dist + _furthest_dist), (-_nearest_dist * _furthest_dist),
		0, 0, 1, 0
	};

	// projection_screen_transform = 
	// {
	//     (double)_screen_width * _nearest_dist / 2, 0, (double)_screen_width / 2, 0,
	//     0, -(double)_screen_height * _nearest_dist * cos_g / 2, (double)_screen_height / 2, 0,
	//     0, 0, (_nearest_dist + _furthest_dist) * min(_screen_width, _screen_height) / 2, (-_nearest_dist * _furthest_dist) * min(_screen_width, _screen_height) / 2,
	//     0, 0, 1, 0
	// };

	// copy vertices_transformed from vertices(origin) and transform it.

	// TODO: CUDA parallelize "Transform Vertices"
	for(std::size_t i = 0; i != _pvertices_transformed->size(); i++)
	{
		// IMPORTANT!!
		_pvertices_transformed->at(i) = _pvertices->at(i); 
		_pvertices_model_view_transformed->at(i) = _pvertices->at(i);
		//

		Log::Trace(__Camera::LOG_NAME, "Start a vertex transform...");
		
		_pvertices_transformed->at(i).PrintVector(Log$::TRACE_LEVEL);
		model_view_transform * _pvertices_transformed->at(i);
		model_view_transform * _pvertices_model_view_transformed->at(i);
		_pvertices_transformed->at(i).PrintVector(Log$::TRACE_LEVEL);
		projection_screen_transform * _pvertices_transformed->at(i);
		_pvertices_transformed->at(i).PrintVector(Log$::TRACE_LEVEL);
		_pvertices_transformed->at(i) *= (1 / (_pvertices_transformed->at(i)[3])); // homogeneous coordinates unitization
		_pvertices_transformed->at(i).PrintVector(Log$::TRACE_LEVEL);
	}
	//
	return DEFAULT_RESULT;
}

DefaultResult Camera::InverseUpperByDirection(Maths::Vector const &last_direction)
{
	auto last_direction_n = last_direction.N();
	if (last_direction_n != 4)
	{
		Log::Error(__Camera::LOG_NAME, "Invalid last_direction length %d", last_direction_n);
		PRINT_LOCATION;
		return DEFAULT_RESULT_EXCEPTION(Camera$::CODE_INVALID_VECTOR_LENGTH, "Invalid last_direction length");
	}

	using namespace __Camera::InverseUpperByDirection;

	upward_before = {0, 1, 0, 0};
	upward_after = {0, 1, 0, 0};

	upward_before *= last_direction;
	upward_after *= _direction;

	if ((upward_before * upward_after) < 0)
	{
		_upward *= -1;
	}

	return DEFAULT_RESULT;
}