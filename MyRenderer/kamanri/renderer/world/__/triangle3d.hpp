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
					size_t _index;

					// offset + index
					size_t _v1, _v2, _v3;
					size_t _vt1, _vt2, _vt3;
					size_t _vn1, _vn2, _vn3;
					
					/// values
					// on screen coordinates
					double _v1_x, _v1_y, _v1_z, _v2_x, _v2_y, _v2_z, _v3_x, _v3_y, _v3_z;
					// on world coordinates
					double _w_v1_z, _w_v2_z, _w_v3_z;
					double _vt1_x, _vt1_y, _vt2_x, _vt2_y, _vt3_x, _vt3_y;
					double _vn1_x, _vn1_y, _vn1_z, _vn2_x, _vn2_y, _vn2_z, _vn3_x, _vn3_y, _vn3_z;

					// factors of square, ax + by + cz - 1 = 0
					double _a, _b, _c;

					// areal coordinates calculate matrix
					Maths::SMatrix _areal_coordinates_calculate_matrix;

					inline double ScreenZ(double x, double y) const { return (1 - _a * x - _b * y) / _c; }
					bool IsCover(double x, double y) const;
					void WritePixelTo(double x, double y, FrameBuffer& frame_buffer, DWORD& pixel) const;
					friend void Object::__UpdateTriangleRef(std::vector<Triangle3D>& triangles, std::vector<Object>& objects, size_t index);
					void ArealCoordinates(double x, double y, Maths::Vector& result) const;

				public:
					Triangle3D(std::vector<Object>& objects, size_t index, size_t v1, size_t v2, size_t v3, size_t vt1, size_t vt2, size_t vt3, size_t vn1, size_t vn2, size_t vn3);
					void Build(Resources const& res);
					void PrintTriangle(Utils::LogLevel level = Utils::Log$::INFO_LEVEL) const;
					void WriteTo(Buffers& buffers, double nearest_dist);
					
				};
			} // namespace __

		} // namespace Triangle3Ds

	} // namespace Renderer

} // namespace Kamanri