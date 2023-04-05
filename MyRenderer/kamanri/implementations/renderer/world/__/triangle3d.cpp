#include <cfloat>
#include "kamanri/renderer/world/__/triangle3d.hpp"
#include "kamanri/utils/logs.hpp"
#include "kamanri/maths/vector.hpp"
#include "kamanri/maths/matrix.hpp"
#include "kamanri/utils/string.hpp"

using namespace Kamanri::Utils;
using namespace Kamanri::Renderer::World::__;
using namespace Kamanri::Maths;
using namespace Kamanri::Maths;

namespace Kamanri
{
	namespace Renderer
	{
		namespace World
		{
			namespace __
			{
				namespace __Triangle3D
				{
					constexpr const char* LOG_NAME = STR(Kamanri::Renderer::World::__::Triangle3D);

					namespace IsScreenCover
					{
						double v1_v2_xy_determinant;
						double v2_v3_xy_determinant;
						double v3_v1_xy_determinant;
					} // namespace IsScreenCover
					
					namespace Build
					{
						SMatrix screen_vertices_matrix(3);
						Vector s_abc_vec(3);
						SMatrix world_vertices_matrix(3);
						Vector w_abc_vec(3);
						SMatrix areal_coordinates_build_matrix(3);
					} // namespace Build
					
					namespace WriteToPixel
					{
						

						// double areal_coordinates_1;
						// double areal_coordinates_2;
						// double areal_coordinates_3;

						// double world_z;

						// double img_u;
						// double img_v;
					} // namespace WriteToPixel

					namespace WriteTo
					{
						double nearest_dist;
					} // namespace WriteTo
					
					

					namespace ArealCoordinates
					{
						
						SMatrix a(3);
					} // namespace ArealCoordinates


					inline double Determinant(double a00, double a01, double a10, double a11)
					{
						return ((a00) * (a11) - (a10) * (a01));
					}

					inline double PerspectiveUndo(Vector const& areal_coordinates, double v1_factor, double v2_factor, double v3_factor)
					{
						return (1.0 / (areal_coordinates[0] / v1_factor + areal_coordinates[1] / v2_factor + areal_coordinates[2] / v3_factor));
					}

					inline double PerspectiveCorrect(Vector const& areal_coordinates, double v1_factor, double v2_factor, double v3_factor, double c_v1_factor, double c_v2_factor, double c_v3_factor, double c_this_factor)
					{
						return ((areal_coordinates[0] * v1_factor / c_v1_factor + areal_coordinates[1] * v2_factor / c_v2_factor + areal_coordinates[2] * v3_factor / c_v3_factor) * c_this_factor);
					}

					inline double Max(double x1, double x2, double x3)
					{
						return x1 > x2 ? (x1 > x3 ? x1 : x3) : (x2 > x3 ? x2 : x3);
					}

					inline double Min(double x1, double x2, double x3)
					{
						return x1 < x2 ? (x1 < x3 ? x1 : x3) : (x2 < x3 ? x2 : x3);
					}


				} // namespace __Triangle3D
				
			} // namespace __
			
		} // namespace World
		
	} // namespace Renderer
	
} // namespace Kamanri




Triangle3D::Triangle3D(std::vector<Object>& objects, size_t object_index, size_t index, size_t v1, size_t v2, size_t v3, size_t vt1, size_t vt2, size_t vt3, size_t vn1, size_t vn2, size_t vn3)
: _areal_coordinates_calculate_matrix(3)
{
	_p_objects = &objects;
	_object_index = object_index;
	_index = index;
	_v1 = v1;
	_v2 = v2;
	_v3 = v3;
	_vt1 = vt1;
	_vt2 = vt2;
	_vt3 = vt3;
	_vn1 = vn1;
	_vn2 = vn2;
	_vn3 = vn3;

}

void Triangle3D::PrintTriangle(LogLevel level) const
{
	if(Log::Level() > level) return;
	PrintLn("v1 | v2 | v3 : %d | %d | %d", _v1, _v2, _v3);
}


void Triangle3D::Build(Resources const& res)
{
	using namespace __Triangle3D::Build;

	// 1. Build the location of triangle
	_s_v1_x = res.vertices_transformed[_v1][0];
	_s_v1_y = res.vertices_transformed[_v1][1];
	_s_v1_z = res.vertices_transformed[_v1][2];
	_s_v2_x = res.vertices_transformed[_v2][0];
	_s_v2_y = res.vertices_transformed[_v2][1];
	_s_v2_z = res.vertices_transformed[_v2][2];
	_s_v3_x = res.vertices_transformed[_v3][0];
	_s_v3_y = res.vertices_transformed[_v3][1];
	_s_v3_z = res.vertices_transformed[_v3][2];

	// add world location
	_w_v1_x = res.vertices_model_view_transformed[_v1][0];
	_w_v2_x = res.vertices_model_view_transformed[_v2][0];
	_w_v3_x = res.vertices_model_view_transformed[_v3][0];
	_w_v1_y = res.vertices_model_view_transformed[_v1][1];
	_w_v2_y = res.vertices_model_view_transformed[_v2][1];
	_w_v3_y = res.vertices_model_view_transformed[_v3][1];
	_w_v1_z = res.vertices_model_view_transformed[_v1][2];
	_w_v2_z = res.vertices_model_view_transformed[_v2][2];
	_w_v3_z = res.vertices_model_view_transformed[_v3][2];


	// build world coordinates vector
	_w_v1_v2 = res.vertices_model_view_transformed[_v2];
	_w_v1_v2 -= res.vertices_model_view_transformed[_v1];
	_w_v2_v3 = res.vertices_model_view_transformed[_v3];
	_w_v2_v3 -= res.vertices_model_view_transformed[_v2];
	_w_v3_v1 = res.vertices_model_view_transformed[_v1];
	_w_v3_v1 -= res.vertices_model_view_transformed[_v3];

	// 2. build abc
	screen_vertices_matrix = 
	{
		_s_v1_x, _s_v1_y, _s_v1_z,
		_s_v2_x, _s_v2_y, _s_v2_z,
		_s_v3_x, _s_v3_y, _s_v3_z
	};
	world_vertices_matrix = 
	{
		_w_v1_x, _w_v1_y, _w_v1_z,
		_w_v2_x, _w_v2_y, _w_v2_z,
		_w_v3_x, _w_v3_y, _w_v3_z
	};
	
	s_abc_vec = {1, 1, 1};
	w_abc_vec = {1, 1, 1};

	(-screen_vertices_matrix) * s_abc_vec;
	(-world_vertices_matrix) * w_abc_vec;

	_s_a = s_abc_vec[0];
	_s_b = s_abc_vec[1];
	_s_c = s_abc_vec[2];
	_w_a = w_abc_vec[0];
	_w_b = w_abc_vec[1];
	_w_c = w_abc_vec[2];

	// 3. Build the color of every pixel in triangle
	_vt1_x = res.vertex_textures[_vt1][0];
	_vt1_y = res.vertex_textures[_vt1][1];
	_vt2_x = res.vertex_textures[_vt2][0];
	_vt2_y = res.vertex_textures[_vt2][1];
	_vt3_x = res.vertex_textures[_vt3][0];
	_vt3_y = res.vertex_textures[_vt3][1];

	// build vertex normals
	_vn1_x = res.vertex_normals_model_view_transformed[_vn1][0];
	_vn1_y = res.vertex_normals_model_view_transformed[_vn1][1];
	_vn1_z = res.vertex_normals_model_view_transformed[_vn1][2];
	_vn2_x = res.vertex_normals_model_view_transformed[_vn2][0];
	_vn2_y = res.vertex_normals_model_view_transformed[_vn2][1];
	_vn2_z = res.vertex_normals_model_view_transformed[_vn2][2];
	_vn3_x = res.vertex_normals_model_view_transformed[_vn3][0];
	_vn3_y = res.vertex_normals_model_view_transformed[_vn3][1];
	_vn3_z = res.vertex_normals_model_view_transformed[_vn3][2];

	// 4. build a and n_a of areal coordinates
	areal_coordinates_build_matrix = 
	{
		{ _s_v1_x, _s_v1_y, _s_v1_z },
		{ _s_v2_x, _s_v2_y, _s_v2_z },
		{ _s_v3_x, _s_v3_y, _s_v3_z },
	};

	_areal_coordinates_calculate_matrix = -areal_coordinates_build_matrix;

}


bool Triangle3D::IsScreenCover(double x, double y) const
{
	using namespace __Triangle3D;
	using namespace __Triangle3D::IsScreenCover;
	v1_v2_xy_determinant = Determinant
	(
		_s_v2_x - _s_v1_x, x - _s_v1_x,
		_s_v2_y - _s_v1_y, y - _s_v1_y
	);
	v2_v3_xy_determinant = Determinant
	(
		_s_v3_x - _s_v2_x, x - _s_v2_x,
		_s_v3_y - _s_v2_y, y - _s_v2_y
	);
	v3_v1_xy_determinant = Determinant
	(
		_s_v1_x - _s_v3_x, x - _s_v3_x,
		_s_v1_y - _s_v3_y, y - _s_v3_y
	);

	if (v1_v2_xy_determinant * v2_v3_xy_determinant >= 0 && v2_v3_xy_determinant * v3_v1_xy_determinant >= 0 && v3_v1_xy_determinant * v1_v2_xy_determinant >= 0)
	{
		return true;
	}

	return false;
}

/// @brief Judge the line(location, direction) is through the triangle
/// @param location 
/// @param direction 
/// @return 
bool Triangle3D::IsThrough(Maths::Vector& location, Maths::Vector& direction, double& output_distance)
{
	// solve
	// (x - x0) / dx = (y - y0) / dy = (z - z0) / dz
	// ax + by + cz - 1 = 0
	//
	auto dxdy = direction[0] * direction[1];
	auto dxdz = direction[0] * direction[2];
	auto dydz = direction[1] * direction[2];
	Vector conjunction = 
	{
		location[0] * dydz - location[1] * dxdz,
		location[0] * dydz - location[2] * dxdy,
		1
	};
	SMatrix a = 
	{
		dydz, -dxdz, 0,
		dydz, 0, -dxdy,
		_w_a, _w_b, _w_c
	};

	(-a) * conjunction;

	// judge the cross product

	Vector c_v1_v2 = 
	{
		conjunction[0] - _w_v1_x,
		conjunction[1] - _w_v1_y,
		conjunction[2] - _w_v1_z,
		1
	};

	Vector c_v2_v3 = 
	{
		conjunction[0] - _w_v2_x,
		conjunction[1] - _w_v2_y,
		conjunction[2] - _w_v2_z,
		1
	};
	Vector c_v3_v1 = 
	{
		conjunction[0] - _w_v3_x,
		conjunction[1] - _w_v3_y,
		conjunction[2] - _w_v3_z,
		1
	};

	c_v1_v2 *= _w_v1_v2;
	c_v2_v3 *= _w_v2_v3;
	c_v3_v1 *= _w_v3_v1;

	if((c_v1_v2 * c_v2_v3) >= 0 && (c_v2_v3 * c_v3_v1) >= 0 && (c_v3_v1 * c_v1_v2) >= 0)
	{
		Vector conjunction_4d = 
		{
			conjunction[0], conjunction[1], conjunction[2], 1
		};
		output_distance = conjunction_4d - location;
		return true;
	}

	return false;

}




void Triangle3D::WriteToPixel(size_t x, size_t y, FrameBuffer& frame_buffer, double nearest_dist, Object* cuda_objects) const
{
	using namespace __Triangle3D;
	// pruning
	if(x < Min(_s_v1_x, _s_v2_x, _s_v3_x) || x > Max(_s_v1_x, _s_v2_x, _s_v3_x)) return;
	if(y < Min(_s_v1_y, _s_v2_y, _s_v3_y) || y > Max(_s_v1_y, _s_v2_y, _s_v3_y)) return;
	if(!IsScreenCover((double)x, (double)y)) return;

	// get world location
	Vector screen_areal_coordinates(3);
	ScreenArealCoordinates((double)x, (double)y, screen_areal_coordinates);

	double world_z = PerspectiveUndo(screen_areal_coordinates, _w_v1_z, _w_v2_z, _w_v3_z);

	// z-buffer
	if(world_z < frame_buffer.location[2] || world_z > nearest_dist) return;


	double world_x = PerspectiveCorrect(screen_areal_coordinates, _w_v1_x, _w_v2_x, _w_v3_x, _w_v1_z, _w_v2_z, _w_v3_z, world_z);
	double world_y = PerspectiveCorrect(screen_areal_coordinates, _w_v1_y, _w_v2_y, _w_v3_y, _w_v1_z, _w_v2_z, _w_v3_z, world_z);

	// *****************************************
	// 透视矫正
	// *****************************************
	double img_u = PerspectiveCorrect(screen_areal_coordinates, _vt1_x, _vt2_x, _vt3_x, _w_v1_z, _w_v2_z, _w_v3_z, world_z);
	double img_v = PerspectiveCorrect(screen_areal_coordinates, _vt1_y, _vt2_y, _vt3_y, _w_v1_z, _w_v2_z, _w_v3_z, world_z);
	
	// write to buffers
	frame_buffer.triangle_index = _index;
	frame_buffer.location = 
	{
		world_x,
		world_y,
		world_z,
		1
	};

	frame_buffer.vertex_normal = 
	{
		PerspectiveCorrect(screen_areal_coordinates, _vn1_x, _vn2_x, _vn3_x, _w_v1_z, _w_v2_z, _w_v3_z, world_z),
		PerspectiveCorrect(screen_areal_coordinates, _vn1_y, _vn2_y, _vn3_y, _w_v1_z, _w_v2_z, _w_v3_z, world_z),
		PerspectiveCorrect(screen_areal_coordinates, _vn1_z, _vn2_z, _vn3_z, _w_v1_z, _w_v2_z, _w_v3_z, world_z),
		0
	};
	
	frame_buffer.vertex_normal.Unitization();

	frame_buffer.color = _p_objects->at(_object_index).GetImage().Get(img_u, img_v).rgb;

}


void Triangle3D::ScreenArealCoordinates(double x, double y, Maths::Vector& result) const
{
	using namespace __Triangle3D::ArealCoordinates;
	result = {x, y, ScreenZ(x, y)};
	
	// (v1, v2, v3, 0) (alpha, beta, gamma, 1)^T = target
	
	_areal_coordinates_calculate_matrix * result;

}