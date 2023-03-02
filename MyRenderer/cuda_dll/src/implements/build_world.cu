#include <cuda_runtime.h>
#include "cuda_dll/src/build_world.cuh"
#include "kamanri/utils/logs.hpp"
#include "kamanri/renderer/world/__/buffers.hpp"
#include "cuda_dll/src/utils/cuda_thread_config.cuh"
#include "cuda_dll/src/utils/cuda_error_check.cuh"


using namespace Kamanri::Utils;
using namespace Kamanri::Maths;
using namespace Kamanri::Renderer;
using namespace Kamanri::Renderer::World;
using namespace Kamanri::Renderer::World::__;

namespace __BuildWorld
{
	constexpr const char* LOG_NAME = STR(BuildWorld);

	
} // namespace BuildWorld$

__global__ void BuildPixelEntry(Kamanri::Renderer::World::World3D* p_world, unsigned int height)
{
	size_t x = thread_index / height;
	size_t y = thread_index - x * height;
	
	p_world->__BuildForPixel((size_t)x, (size_t)y);
}

BuildWorldCode BuildWorld(Kamanri::Renderer::World::World3D* p_world, unsigned int width, unsigned int height)
{
	BuildPixelEntry
		thread_num(width * height)
		(p_world, height);
	
	auto res = cudaDeviceSynchronize();

	CUDA_ERROR_CHECK(res, __BuildWorld::LOG_NAME);
	return BuildWorld$::CODE_NORM;
}

template <typename... Ts>
__device__ inline int DevicePrint(const char* formatStr, Ts... argv)
{
	int retCode = printf(formatStr, argv...);
	return retCode;
}

////////////////////////////////////////////////////////////////////////////
// Maths

		////////////////////////////////////////////
		// Vector
__device__ Kamanri::Maths::Vector::Vector(size_t n)
{
	this->_N = Vector$::NOT_INITIALIZED_N;

	if (n != 2 && n != 3 && n != 4)
	{
		DevicePrint("The size of initializer list is not valid: %d", (int) n);
		return;
	}

	this->_N = n;
}

__device__ Kamanri::Maths::Vector::Vector(Vector const& v)
{
	_N = v._N;

	for (size_t i = 0; i < _N; i++)
	{
		_V[i] = v._V[i];
	}

}

__device__ Kamanri::Maths::Vector::Vector(std::initializer_list<VectorElemType> list)
{
	this->_N = Vector$::NOT_INITIALIZED_N;

	auto n = list.size();
	if (n != 2 && n != 3 && n != 4)
	{
		DevicePrint("The size of initializer list is not valid: %d", (int)n);
		return;
	}

	this->_N = n;

	auto i = 0;
	for(auto list_elem: list)
	{
		_V[i] = list_elem;
		i++;
	}

}


__device__ VectorElemType Kamanri::Maths::Vector::operator*(Vector const& v) const
{
	size_t n1 = _N;
	size_t n2 = v._N;

	if (n1 != n2)
	{
		DevicePrint("Call of Vector::operator*=: Two vectors of unequal length: %d and %d", n1, n2);
		return Vector$::NOT_INITIALIZED_VALUE;
	}

	VectorElemType result = Vector$::NOT_INITIALIZED_N;

	for (size_t i = 0; i < n1; i++)
	{
		result += _V[i] * v._V[i];
	}

	return result;
}

__device__ VectorElemType Kamanri::Maths::Vector::operator-(Vector const& v)
{
	if (_N != 4)
	{
		DevicePrint("operator-: Invalid N");
		return Vector$::NOT_INITIALIZED_VALUE;
	}
	if (_V[3] != 1 || v._V[3] != 1)
	{
		DevicePrint("operator-: Not uniformed.");
		return Vector$::NOT_INITIALIZED_VALUE;
	}

	return sqrt(pow(_V[0] - v._V[0], 2) + pow(_V[1] - v._V[1], 2) + pow(_V[2] - v._V[2], 2));
}

__device__ VectorCode Kamanri::Maths::Vector::operator+=(Vector const &v)
{
	size_t n1 = _N;
	size_t n2 = v._N;

	if (n1 != n2)
	{
		DevicePrint("Call of Vector::operator+=: Two vectors of unequal length: %d and %d", n1, n2);
		return Vector$::CODE_NOT_EQUEL_N;
	}

	for (size_t i = 0; i < n1; i++)
	{
		_V[i] += v._V[i];
	}

	return Vector$::CODE_NORM;
}

__device__ VectorCode Kamanri::Maths::Vector::operator+=(std::initializer_list<VectorElemType> list)
{
	Vector v(list);
	return this->operator+=(v);
}

__device__ VectorCode Kamanri::Maths::Vector::operator-=(Vector const& v)
{
	size_t n1 = _N;
	size_t n2 = v._N;

	if (n1 != n2)
	{
		DevicePrint("Call of Vector::operator-=: It is impossible to add two vectors of unequal length: %d and %d", n1, n2);
		return Vector$::CODE_NOT_EQUEL_N;
	}


	for (size_t i = 0; i < n1; i++)
	{
		_V[i] -= v._V[i];
	}

	return Vector$::CODE_NORM;
}

__device__ VectorCode Kamanri::Maths::Vector::operator*=(Vector const& v)
{
	size_t n1 = _N;
	size_t n2 = v._N;

	if (n1 != n2)
	{
		DevicePrint("Call of Vector::operator*=: Two vectors of unequal length: %d and %d", n1, n2);
		return Vector$::CODE_NOT_EQUEL_N;
	}

	if(n1 != 3 && n1 != 4)
	{
		auto message = "Call of Vector::operator*=: Vector has not cross product when n != 3 or 4";
		DevicePrint(message);
		return Vector$::CODE_INVALID_OPERATION;
	}

	auto v0 = _V[1] * v._V[2] - _V[2] * v._V[1];
	auto v1 = _V[2] * v._V[0] - _V[0] * v._V[2];
	auto v2 = _V[0] * v._V[1] - _V[1] * v._V[0];

	_V[0] = v0;
	_V[1] = v1;
	_V[2] = v2;

	if(n1 == 4)
	{
		_V[3] = _V[3] * v._V[3];
	}

	return Vector$::CODE_NORM;
	
}

__device__ Vector& Kamanri::Maths::Vector::operator=(Vector const& v)
{
	_N = v._N;

	for (size_t i = 0; i < _N; i++)
	{
		_V[i] = v._V[i];
	}

	return *this;
}

__device__ VectorCode Kamanri::Maths::Vector::operator=(std::initializer_list<VectorElemType> list)
{
	auto n = list.size();
	if (n != _N)
	{
		DevicePrint("The size of initializer list(%d) is not equal to vector(%d)", (int) n, _N);
		return Vector$::CODE_NOT_EQUEL_N;
	}

	auto i = 0;
	for (auto list_elem : list)
	{
		_V[i] = list_elem;
		i++;
	}

	return Vector$::CODE_NORM;
}

__device__ VectorElemType Kamanri::Maths::Vector::operator[](size_t n) const
{
	if (n < 0 || n > this->_N)
	{
		DevicePrint("Kamanri::Maths::Vector::operator[]: Index %llu out of bound %llu\n", n, this->_N);
		return Vector$::NOT_INITIALIZED_VALUE;
	}

	return _V[n];
}

__device__ VectorCode Kamanri::Maths::Vector::Set(size_t index, VectorElemType value)
{
	if (index < 0 || index > this->_N)
	{
		DevicePrint("Kamanri::Maths::Vector::Set: Index %llu out of bound %llu\n", index, this->_N);
		return Vector$::NOT_INITIALIZED_VALUE;
	}
	_V[index] = value;
	return Vector$::CODE_NORM;
}

__device__ VectorCode Kamanri::Maths::Vector::Unitization()
{
	double length_square = 0;
	for (size_t i = 0; i < _N; i++)
	{
		length_square += pow(_V[i], 2);
	}

	if (length_square == 1)
	{
		// the length of vector is 1, need not to unitization.
		return Vector$::CODE_NORM;
	}

	auto length = sqrt(length_square);

	for (size_t i = 0; i < _N; i++)
	{
		_V[i] /= length;
	}

	return Vector$::CODE_NORM;
}
//////////////////////////////////////////////////
// Matrix
namespace __SMatrix
{

	/// @brief (-1)^RON
	/// @param v 
	/// @return 
	__device__ int Pow_NegativeOne_ReverseOrderNumber(size_t* list, size_t list_size)
	{
		int res = 1;
		for (size_t i = 1; i < list_size; i++)
		{
			for (size_t j = 0; j < i; j++)
			{
				if (list[j] > list[i]) res *= -1;
			}
		}
		return res;
	}


	__device__ inline SMatrixElemType SMGet(SMatrixElemType* sm, size_t n, size_t row, size_t col)
	{
		return sm[n * row + col];
	}

	// Square Matrix Determinant [dimension + 1]
	__device__ inline SMatrixElemType SMDet2(SMatrixElemType* sm, size_t n, size_t row_1, size_t row_2, size_t col_1, size_t col_2)
	{
		return (SMGet(sm, n, row_1, col_1) * SMGet(sm, n, row_2, col_2) - SMGet(sm, n, row_1, col_2) * SMGet(sm, n, row_2, col_1));
	}

	// Square Matrix Determinant [dimension + 1]
	__device__ inline SMatrixElemType SMDet3(SMatrixElemType* sm, size_t n, size_t row_1, size_t row_2, size_t row_3, size_t col_1, size_t col_2, size_t col_3)
	{
		return (SMGet(sm, n, row_1, col_1) * SMDet2(sm, n, row_2, row_3, col_2, col_3) - SMGet(sm, n, row_2, col_1) * SMDet2(sm, n, row_1, row_3, col_2, col_3) + SMGet(sm, n, row_3, col_1) * SMDet2(sm, n, row_1, row_2, col_2, col_3));
	}

	__device__ SMatrixElemType __Determinant(SMatrixElemType* psm, size_t* row_list, size_t* col_list, size_t row_count)
	{
		using namespace __SMatrix;
		SMatrixElemType result = 0;

		if (row_count < 4)
		{
			if (row_count == 1) result = SMGet(psm, row_count, row_list[0], col_list[0]);
			if (row_count == 2) result = SMDet2(psm, row_count, row_list[0], row_list[1], col_list[0], col_list[1]);
			if (row_count == 3) result = SMDet3(psm, row_count, row_list[0], row_list[1], row_list[2], col_list[0], col_list[1], col_list[2]);
			if (row_count == 4)
			{
				result = SMGet(psm, row_count, row_list[0], col_list[0]) * SMDet3(psm, row_count, 1, 2, 3, 1, 2, 3) -
					SMGet(psm, row_count, row_list[1], col_list[0]) * SMDet3(psm, row_count, row_list[0], row_list[2], row_list[3], col_list[1], col_list[2], col_list[3]) +
					SMGet(psm, row_count, row_list[2], col_list[0]) * SMDet3(psm, row_count, row_list[0], row_list[1], row_list[3], col_list[1], col_list[2], col_list[3]) -
					SMGet(psm, row_count, row_list[3], col_list[0]) * SMDet3(psm, row_count, row_list[0], row_list[1], row_list[2], col_list[1], col_list[2], col_list[3]);
			}
			// This is to avoid the bigger index is in front of the smaller
			result *= __SMatrix::Pow_NegativeOne_ReverseOrderNumber(row_list, row_count);
			result *= __SMatrix::Pow_NegativeOne_ReverseOrderNumber(col_list, row_count);

			return result;
		}



		/// row_count > 4
		// size_t col_first = col_list.front();
		// size_t col_first_sorted = __SMatrix::GetSortedIndex(col_list, col_first);
		// col_list.erase(col_list.begin());

		// for (size_t i = 0; i < row_count; i++)
		// {
		// 	size_t row_first = row_list[0];
		// 	size_t row_first_sorted = __SMatrix::GetSortedIndex(row_list, row_first);
		// 	row_list.erase(row_list.begin());
		// 	//////////////////////// calculate sub result
		// 	auto value = psm[_N * row_first + col_first];

		// 	// use -1^(a+b) * n * |A*_ab| to calculate
		// 	auto result_sub = (
		// 		_Determinant(psm, row_list, col_list) *
		// 		value *
		// 		(((row_first_sorted + col_first_sorted) % 2 == 0) ? 1.f : -1.f));

		// 	result += result_sub;

		// 	//////////////////////// calculate sub result
		// 	row_list.push_back(row_first);
		// }

		// col_list.insert(col_list.begin(), col_first);

		// return result;
	}
}

__device__ Kamanri::Maths::SMatrix::SMatrix()
{
	this->_N = SMatrix$::MAX_SUPPORTED_DIMENSION;
}

__device__ Kamanri::Maths::SMatrix::SMatrix(SMatrix const& sm)
{
	_N = sm._N;
	auto size = _N * _N;

	for(size_t i = 0; i < size; i++)
	{
		_SM[i] = sm._SM[i];
	}
}

__device__ Kamanri::Maths::SMatrix::SMatrix(std::initializer_list<SMatrixElemType> list)
{
	this->_N = SMatrix$::NOT_INITIALIZED_N;

	auto size = list.size();
	if (size != 4 && size != 9 && size != 16)
	{
		DevicePrint("The size of initializer list is not valid: %d", (int) size);
		return;
	}

	this->_N = (size_t) sqrt((double) size);

	auto i = 0;
	for (auto list_elem : list)
	{
		_SM[i] = list_elem;
		i++;
	}
}

__device__ SMatrixCode Kamanri::Maths::SMatrix::operator*=(SMatrixElemType value)
{
	for (size_t i = 0; i < _N * _N; i++)
	{
		_SM[i] *= value;
	}

	return SMatrix$::CODE_NORM;
}

__device__ SMatrixCode Kamanri::Maths::SMatrix::operator*(Vector& v) const
{
	if (_N != v.N())
	{
		DevicePrint("Call of SMatrix::operator*: matrix and vector of unequal length: %d and %d", _N, v.N());
		return SMatrix$::CODE_NOT_EQUEL_N;
	}

	Vector v_temp = v;

	double value = 0;

	for (size_t row = 0; row < _N; row++)
	{
		value = 0;
		for (size_t col = 0; col < _N; col++)
		{
			value += _SM[row * _N + col] * v_temp[col];
		}
		v.Set(row, value);
	}

	return SMatrix$::CODE_NORM;
}

__device__ SMatrix Kamanri::Maths::SMatrix::operator-() const
{
	// use AA* == |A|E
	// A^(-1) == A* / |A|
	auto pm_asm = operator*();
	auto d = Determinant();
	if (d == SMatrix$::NOT_INITIALIZED_VALUE)
	{
		DevicePrint("Invalid determinant %f.", d);
	}
	pm_asm *= (1 / d);

	return pm_asm;

}



// [rest] [value] [dimension] [value]

#define REST_V_2_0 1, 2
#define REST_V_2_1 0, 2
#define REST_V_2_2 0, 1

#define REST_V_3_0 1, 2, 3
#define REST_V_3_1 0, 2, 3
#define REST_V_3_2 0, 1, 3
#define REST_V_3_3 0, 1, 2

// [Square Matrix] [Complement] [N dimension]
// (pointer of square matrix, width, const complement dimension, const row, const column)
#define SM_C(p_sm, n, c_d, c_row, c_col) __SMatrix::SMDet##c_d##(p_sm, n, REST_V_##c_d##_##c_row, REST_V_##c_d##_##c_col)

// [Square Matrix] [Algebratic Complement] [N dimension]
// (pointer of square matrix, width, const complement dimension, const row, const column)
#define SM_AC(p_sm, n, c_d, c_row, c_col) (SM_C(p_sm, n, c_d, c_row, c_col) * (((c_row + c_col) % 2 == 0) ? 1.f : -1.f))

__device__ SMatrix Kamanri::Maths::SMatrix::operator*() const
{

	if (_N != 3 && _N != 4)
	{
		DevicePrint("operator* not allowed when _N = %llu", _N);
		return SMatrix();
	}

	auto p_sm = (SMatrixElemType*) _SM;

	if (_N == 3)
	{
		SMatrix ret_sm =
		{
			SM_AC(p_sm, _N, 2, 0, 0), SM_AC(p_sm, _N, 2, 1, 0), SM_AC(p_sm, _N, 2, 2, 0),
			SM_AC(p_sm, _N, 2, 0, 1), SM_AC(p_sm, _N, 2, 1, 1), SM_AC(p_sm, _N, 2, 2, 1),
			SM_AC(p_sm, _N, 2, 0, 2), SM_AC(p_sm, _N, 2, 1, 2), SM_AC(p_sm, _N, 2, 2, 2),
		};
		return ret_sm;
	}

	SMatrix ret_sm =
	{
		SM_AC(p_sm, _N, 3, 0, 0), SM_AC(p_sm, _N, 3, 1, 0), SM_AC(p_sm, _N, 3, 2, 0), SM_AC(p_sm, _N, 3, 3, 0),
		SM_AC(p_sm, _N, 3, 0, 1), SM_AC(p_sm, _N, 3, 1, 1), SM_AC(p_sm, _N, 3, 2, 1), SM_AC(p_sm, _N, 3, 3, 1),
		SM_AC(p_sm, _N, 3, 0, 2), SM_AC(p_sm, _N, 3, 1, 2), SM_AC(p_sm, _N, 3, 2, 2), SM_AC(p_sm, _N, 3, 3, 2),
		SM_AC(p_sm, _N, 3, 0, 3), SM_AC(p_sm, _N, 3, 1, 3), SM_AC(p_sm, _N, 3, 2, 3), SM_AC(p_sm, _N, 3, 3, 3)
	};

	return ret_sm;

}

__device__ SMatrixElemType Kamanri::Maths::SMatrix::Determinant() const
{
	size_t row_list[4] = { 0, 1, 2, 3 };
	size_t col_list[4] = { 0, 1, 2, 3 };
	switch (_N)
	{
		case 2:
			return __SMatrix::__Determinant((SMatrixElemType*) _SM, row_list, col_list, 2);
		case 3:
			return __SMatrix::__Determinant((SMatrixElemType*) _SM, row_list, col_list, 3);
		case 4:
			return __SMatrix::__Determinant((SMatrixElemType*) _SM, row_list, col_list, 4);
		default:
			DevicePrint("Invalid dimension %d", _N);
			break;
	}
}

///////////////////////////////////////////////////////////////
// Renderer

		///////////////////////////////////////////////
		// TGAImage
__device__ TGAColor Kamanri::Renderer::TGAImage::Get(const int x, const int y) const
{
	if (x < 0 || y < 0 || x >= _width || y >= _height)
		return {};
	return TGAColor(_cuda_data + (x + y * _width) * _bytes_per_pixel, _bytes_per_pixel);
}
__device__ TGAColor Kamanri::Renderer::TGAImage::Get(double u, double v) const
{
	int x = u * _width;
	int y = v * _height;
	return Get(x, _height - y); // y axis towards up, so use height - y
}

//////////////////////////////////////////////////////////////////
// Renderer::World
			//////////////////////////////////////////////////
			// World3D
__device__ void Kamanri::Renderer::World::World3D::__BuildForPixel(size_t x, size_t y)
{
	// set z = infinity
	_buffers.InitPixel(x, y);

	auto& buffer = _buffers.GetFrame(x, y);
	auto& bitmap_pixel = _buffers.GetBitmapBuffer(x, y);

	for (size_t i = 0; i < *_environment.cuda_triangles_size; i++)
	{
		_environment.cuda_triangles[i].WriteToPixel(x, y, buffer, _camera.NearestDist(), _environment.cuda_objects);
	}

	if (_buffers.GetFrame(x, y).location[2] == -DBL_MAX) return;

	// set distance = infinity, is exposed.
	_environment.bpr_model.InitLightBufferPixel(x, y, buffer);

	// while(1);

	for (size_t i = 0; i < *_environment.cuda_triangles_size; i++)
	{
		_environment.bpr_model.__BuildPerTrianglePixel(x, y, _environment.cuda_triangles[i], buffer);
	}

	_environment.bpr_model.WriteToPixel(x, y, buffer, bitmap_pixel);


}
////////////////////////////////////////////////////////////
// BlingPhongReflectionModel

// namespace __BlingPhongReflectionModel
// {
// 	namespace WriteToPixel
// 	{
// 		__device__ void Handle(unsigned int& y, RGB x) { y += x; }
// 	} // namespace WriteToPixel
	
// } // namespace __BlingPhongReflectionModel


#define Scan_R270(height, x, y) ((height - (y + 1)) * height + x)
#define LightBufferLoc(width, height, index, x, y) (width * height * index + Scan_R270(height, x, y))

__device__ void Kamanri::Renderer::World::BlingPhongReflectionModel::InitLightBufferPixel(size_t x, size_t y, FrameBuffer& buffer)
{
	for (size_t i = 0; i < *_cuda_point_lights_size; i++)
	{
		auto& this_item = _cuda_lights_buffer[LightBufferLoc(_screen_width, _screen_height, i, x, y)];
		this_item.distance = DBL_MAX;
		this_item.is_exposed = true;
		this_item.is_specular = false;
	}

}

#define SpecularTransition(min_theta, theta) pow((theta - min_theta) / (1 - min_theta), 3)

__device__ void Kamanri::Renderer::World::BlingPhongReflectionModel::__BuildPerTrianglePixel(size_t x, size_t y, __::Triangle3D& triangle, FrameBuffer& buffer)
{

	if (triangle.Index() == buffer.triangle_index)
	{
		for (size_t i = 0; i < *_cuda_point_lights_size; i++)
		{
			auto distance = _cuda_point_lights[i].location_model_view_transformed - buffer.location;
			auto& light_buffer_item = _cuda_lights_buffer[LightBufferLoc(_screen_width, _screen_height, i, x, y)];
			if (distance < light_buffer_item.distance) light_buffer_item.distance = distance;

			// judge whether is specular
			// camera is at (0, 0, 0, 1)
			auto point_camera_add_point_light_vector = _cuda_point_lights[i].location_model_view_transformed;
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
			// DevicePrint("triangle index: %llu, distance: %lf\n", triangle.Index(), distance);
		}
		return;
	}

	for (size_t i = 0; i < *_cuda_point_lights_size; i++)
	{
		auto& light_buffer_item = _cuda_lights_buffer[LightBufferLoc(_screen_width, _screen_height, i, x, y)];
		auto& light_location = _cuda_point_lights[i].location_model_view_transformed;
		auto light_point_direction = buffer.location;
		light_point_direction -= light_location;
		double light_point_distance = light_location - buffer.location;
		double distance;
		if (triangle.IsThrough(light_location, light_point_direction, distance))
		{
			if (distance < light_point_distance)
			{
				light_buffer_item.distance = distance;
				light_buffer_item.is_exposed = false;
			}
		}
	}

}

#define GenerizeReflection(r, g, b, factor) BlingPhongReflectionModel$::CombineRGB((unsigned int)(r * factor), (unsigned int)(g * factor), (unsigned int)(b * factor))


/// @brief Require normal unitized.
/// @param location 
/// @param normal 
/// @param reflect_point 
__device__ void Kamanri::Renderer::World::BlingPhongReflectionModel::WriteToPixel(size_t x, size_t y, FrameBuffer& buffer, DWORD& pixel)
{
	buffer.r = buffer.g = buffer.b = 0;
	buffer.power = 0;
	buffer.specular_color = buffer.diffuse_color = buffer.ambient_color = 0;
	for (size_t i = 0; i < *_cuda_point_lights_size; i++)
	{
		// Do
		auto& light_buffer_item = _cuda_lights_buffer[LightBufferLoc(_screen_width, _screen_height, i, x, y)];
		auto distance = _cuda_point_lights[i].location_model_view_transformed - buffer.location;
		auto direction = _cuda_point_lights[i].location_model_view_transformed;
		direction -= buffer.location;
		direction.Unitization();

		// power = theta / S * cos(theta)
		auto cos_theta = (buffer.vertex_normal * direction);

		if (cos_theta <= 0) continue;

		auto power = (_cuda_point_lights[i].power / (4 * Maths::PI * pow(distance, 2))) * cos_theta;
		buffer.power += power;

		auto receive_light_color = BlingPhongReflectionModel$::RGBMul(_cuda_point_lights[i].color, power);
		BlingPhongReflectionModel$::DivideRGB(
			BlingPhongReflectionModel$::RGBReflect(receive_light_color, buffer.color),
			buffer.r, buffer.g, buffer.b,
			BlingPhongReflectionModel$::AddHandle
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

	buffer.ambient_color += BlingPhongReflectionModel$::RGBMul(buffer.color, _ambient_factor);

	pixel = BlingPhongReflectionModel$::RGBAdd(buffer.ambient_color, buffer.diffuse_color, buffer.specular_color);

	// DevicePrint("%X ", pixel);
}


/////////////////////////////////////////////////////////////////
// Renderer::World::__
				////////////////////////////////////
				// Buffers

__device__ FrameBuffer& Kamanri::Renderer::World::__::Buffers::GetFrame(size_t x, size_t y)
{
	if (x < 0 || y < 0 || x >= _width || y >= _height)
	{
		DevicePrint("Invalid Index (%d, %d), return the 0 index content", y, x);
		return _cuda_buffers[0];
	}
	return _cuda_buffers[Scan_R270(_height, x, y)];

}

__device__ void Buffers::InitPixel(size_t x, size_t y)
{
	auto& frame = GetFrame(x, y);
	frame.location = Vector(4);
	frame.location.Set(2, -DBL_MAX);
	frame.vertex_normal = Vector(4);
	
	auto& bitmap = GetBitmapBuffer(x, y);
	bitmap = 0x0;
}

__device__ DWORD& Kamanri::Renderer::World::__::Buffers::GetBitmapBuffer(size_t x, size_t y)
{
	if (x < 0 || y < 0 || x >= _width || y >= _height)
	{
		DevicePrint("Invalid Index (%d, %d), return the 0 index content", x, y);
		return _cuda_bitmap_buffer[0];
	}
	return _cuda_bitmap_buffer[Scan_R270(_height, x, y)]; // (x, y) -> (x, _height - y)
}
////////////////////////////////////////
// Triangle3D
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

	// DevicePrint("%X ", frame_buffer.color);
}










