#pragma once
#include "kamanri/renderer/world/__/triangle3d.hpp"

#include "cuda_dll/src/implementations/maths/vector.impl.cuh"
#include "cuda_dll/src/implementations/maths/matrix.impl.cuh"
#include "maths/matrix.impl.cuh"
#include "renderer/tga_image.impl.cuh"

__device__ void Kamanri::Renderer::World::__::Triangle3D::ScreenArealCoordinates(double x, double y, Maths::Vector& result) const
{
	result = { x, y, ScreenZ(x, y) };

	// (v1, v2, v3, 0) (alpha, beta, gamma, 1)^T = target

	_areal_coordinates_calculate_matrix* result;

}

#define Determinant(a00, a01, a10, a11) ((a00) * (a11) - (a10) * (a01))

__device__ bool Kamanri::Renderer::World::__::Triangle3D::IsScreenCover(double x, double y) const
{
	double v1_v2_xy_determinant = Determinant
	(
		_s_v2_x - _s_v1_x, x - _s_v1_x,
		_s_v2_y - _s_v1_y, y - _s_v1_y
	);
	double v2_v3_xy_determinant = Determinant
	(
		_s_v3_x - _s_v2_x, x - _s_v2_x,
		_s_v3_y - _s_v2_y, y - _s_v2_y
	);
	double v3_v1_xy_determinant = Determinant
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

__device__ bool Kamanri::Renderer::World::__::Triangle3D::IsThrough(Maths::Vector& location, Maths::Vector& direction, double& output_distance)
{
	using namespace Maths;
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

	(-a)* conjunction;

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

	if ((c_v1_v2 * c_v2_v3) >= 0 && (c_v2_v3 * c_v3_v1) >= 0 && (c_v3_v1 * c_v1_v2) >= 0)
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

__device__ inline double Max(double x1, double x2, double x3)
{
	return x1 > x2 ? (x1 > x3 ? x1 : x3) : (x2 > x3 ? x2 : x3);
}

__device__ inline double Min(double x1, double x2, double x3)
{
	return x1 < x2 ? (x1 < x3 ? x1 : x3) : (x2 < x3 ? x2 : x3);
}

#define PerspectiveUndo(areal_coordinates, v1_factor, v2_factor, v3_factor) \
(1.0 / (areal_coordinates[0] / v1_factor + areal_coordinates[1] / v2_factor + areal_coordinates[2] / v3_factor))

#define PerspectiveCorrect(areal_coordinates, v1_factor, v2_factor, v3_factor, c_v1_factor, c_v2_factor, c_v3_factor, c_this_factor) \
((areal_coordinates[0] * v1_factor / c_v1_factor + areal_coordinates[1] * v2_factor / c_v2_factor + areal_coordinates[2] * v3_factor / c_v3_factor) * c_this_factor)

__device__ void Kamanri::Renderer::World::__::Triangle3D::WriteToPixel(size_t x, size_t y, FrameBuffer& frame_buffer, double nearest_dist, Object* cuda_objects) const
{
	// pruning
	if (x < Min(_s_v1_x, _s_v2_x, _s_v3_x) || x > Max(_s_v1_x, _s_v2_x, _s_v3_x)) return;
	if (y < Min(_s_v1_y, _s_v2_y, _s_v3_y) || y > Max(_s_v1_y, _s_v2_y, _s_v3_y)) return;
	if (!IsScreenCover(x, y)) return;

	// get world location
	Maths::Vector screen_areal_coordinates(3);
	ScreenArealCoordinates(x, y, screen_areal_coordinates);

	double world_z = PerspectiveUndo(screen_areal_coordinates, _w_v1_z, _w_v2_z, _w_v3_z);

	// z-buffer
	if (world_z < frame_buffer.location[2] || world_z > nearest_dist) return;


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

	frame_buffer.color = cuda_objects[_object_index].GetImage().Get(img_u, img_v).rgb;

}