#pragma once
#include <vector>
#include "kamanri/maths/all.hpp"
#include "kamanri/renderer/tga_image.hpp"
#include "kamanri/renderer/world/object.hpp"
#include "resources.hpp"
#include "buffers.hpp"
namespace Kamanri
{
	namespace Renderer
	{
		namespace World
		{
			namespace __
			{

				namespace Triangle3D$
				{
					constexpr size_t CODE_NOT_IN_TRIANGLE = 100;
					constexpr size_t INEXIST_INDEX = -1;
				} // namespace Triangle3D$

				
				/**
				 * @brief The Triangle3D class is designed to represent a triangle consist of 3 vertices,
				 * it will be the object of projection transformation and be rendered to screen.
				 *
				 */
				class Triangle3D
				{
				private:
					/// @brief The object triangle belongs to.
					std::vector<Object>* _p_objects;
					Object* _cuda_p_objects;

					size_t _object_index;
					size_t _index;

					// offset + index
					size_t _v1, _v2, _v3;
					size_t _vt1, _vt2, _vt3;
					size_t _vn1, _vn2, _vn3;
					
					/// values
					// on screen coordinates
					double _s_v1_x, _s_v1_y, _s_v1_z, _s_v2_x, _s_v2_y, _s_v2_z, _s_v3_x, _s_v3_y, _s_v3_z;
					// on world coordinates
					double _w_v1_x, _w_v2_x, _w_v3_x, _w_v1_y, _w_v2_y, _w_v3_y, _w_v1_z, _w_v2_z, _w_v3_z;

					double _vt1_x, _vt1_y, _vt2_x, _vt2_y, _vt3_x, _vt3_y;
					double _vn1_x, _vn1_y, _vn1_z, _vn2_x, _vn2_y, _vn2_z, _vn3_x, _vn3_y, _vn3_z;

					// factors of square, ax + by + cz - 1 = 0
					// on screen coordinates
					double _s_a, _s_b, _s_c;
					// on world coordinates
					double _w_a, _w_b, _w_c;

					// the world coordinates vector used in IsThrough
					Maths::Vector _w_v1_v2;
					Maths::Vector _w_v2_v3;
					Maths::Vector _w_v3_v1; 

					// areal coordinates calculate matrix
					Maths::SMatrix _areal_coordinates_calculate_matrix;
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
					inline double ScreenZ(double x, double y) const { return (1 - _s_a * x - _s_b * y) / _s_c; }
					
					
					friend void Object::__UpdateTriangleRef(std::vector<Triangle3D>& triangles, std::vector<Object>& objects, size_t index);
					
#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif
					void ScreenArealCoordinates(double x, double y, Maths::Vector& result) const;

				public:
					Triangle3D(std::vector<Object>& objects, size_t object_index, size_t index, size_t v1, size_t v2, size_t v3, size_t vt1, size_t vt2, size_t vt3, size_t vn1, size_t vn2, size_t vn3);
#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif			
					inline size_t Index() const { return _index; }
#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif
					bool IsScreenCover(double x, double y) const;
#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif
					bool IsThrough(Maths::Vector& location, Maths::Vector& direction, double& output_distance);
					void Build(Resources const& res);
					void PrintTriangle(Utils::LogLevel level = Utils::Log$::INFO_LEVEL) const;
#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif
					void WriteToPixel(size_t x, size_t y, FrameBuffer& frame_buffer, double nearest_dist, Object* cuda_objects = nullptr) const;

				};
			} // namespace __

		} // namespace Triangle3Ds

	} // namespace Renderer

} // namespace Kamanri