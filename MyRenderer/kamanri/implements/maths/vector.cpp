#include <cmath>
#include "kamanri/maths/vector.hpp"
#include "kamanri/utils/logs.hpp"
#include "kamanri/utils/memory.hpp"
#include "kamanri/utils/string.hpp"

using namespace Kamanri::Utils;
using namespace Kamanri::Maths;


namespace Kamanri
{
	namespace Maths
	{
		namespace __Vector
		{
			constexpr const char *LOG_NAME = STR(Kamanri::Maths::Vector);
		} // namespace __Vector
		
	} // namespace Maths
	
} // namespace Kamanri



Vector::Vector()
{
	this->_N = Vector$::MAX_SUPPORTED_DIMENSION;
}

Vector::Vector(size_t n)
{
	this->_N = Vector$::NOT_INITIALIZED_N;

	if (n != 2 && n != 3 && n != 4)
	{
		Log::Error(__Vector::LOG_NAME, "The size of initializer list is not valid: %d", (int)n);
		PRINT_LOCATION;
		return;
	}

	this->_N = n;
}

// Vector::Vector(Vector &&v) : _V(std::move(v._V)), _N(v._N)
// {
// 	v._N = Vector$::NOT_INITIALIZED_N;
// }

Vector::Vector(Vector const& v)
{
	_N = v._N;

	for(size_t i = 0; i < _N; i++)
	{
		_V[i] = v._V[i];
	}

}



Vector::Vector(std::initializer_list<VectorElemType> list)
{
	this->_N = Vector$::NOT_INITIALIZED_N;

	auto n = list.size();
	if (n != 2 && n != 3 && n != 4)
	{
		Log::Error(__Vector::LOG_NAME, "The size of initializer list is not valid: %d", (int)n);
		PRINT_LOCATION;
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

Vector& Vector::operator=(Vector const& v)
{
	_N = v._N;

	for(size_t i = 0; i < _N; i++)
	{
		_V[i] = v._V[i];
	}

	return *this;
}

VectorCode Vector::operator=(std::initializer_list<VectorElemType> list)
{
	auto n = list.size();
	if (n != _N)
	{
		Log::Error(__Vector::LOG_NAME, "The size of initializer list(%d) is not equal to vector(%d)", (int)n, _N);
		PRINT_LOCATION;
		return Vector$::CODE_NOT_EQUEL_N;
	}

	auto i = 0;
	for(auto list_elem: list)
	{
		_V[i] = list_elem;
		i++;
	}

	return Vector$::CODE_NORM;
}


VectorElemType Vector::operator[](size_t n) const
{
	if(n < 0 || n > this->_N)
	{
		Log::Error(__Vector::LOG_NAME, "Index %d out of bound %d", n, this->_N);
		PRINT_LOCATION;
		return Vector$::NOT_INITIALIZED_VALUE;
	}

	return _V[n];
}


VectorCode Vector::Set(size_t index, VectorElemType value)
{
	_V[index] = value;
	return Vector$::CODE_NORM;
}

VectorCode Vector::SetAll(VectorElemType value)
{
	for(size_t i = 0; i < _N;i ++)
	{
		_V[i] = value;
	}
	return Vector$::CODE_NORM;
}

VectorCode Vector::operator+=(Vector const &v)
{
	size_t n1 = _N;
	size_t n2 = v._N;

	if (n1 != n2)
	{
		Log::Error(__Vector::LOG_NAME, "Call of Vector::operator+=: Two vectors of unequal length: %d and %d", n1, n2);
		PRINT_LOCATION;
		return Vector$::CODE_NOT_EQUEL_N;
	}

	for (size_t i = 0; i < n1; i++)
	{
		_V[i] += v._V[i];
	}

	return Vector$::CODE_NORM;
}

VectorCode Vector::operator+=(std::initializer_list<VectorElemType> list)
{
	Vector v(list);
	return this->operator+=(v);
}

VectorCode Vector::operator-=(Vector const &v)
{
	size_t n1 = _N;
	size_t n2 = v._N;

	if (n1 != n2)
	{
		Log::Error(__Vector::LOG_NAME, "Call of Vector::operator-=: It is impossible to add two vectors of unequal length: %d and %d", n1, n2);
		PRINT_LOCATION;
		return Vector$::CODE_NOT_EQUEL_N;
	}

	
	for (size_t i = 0; i < n1; i++)
	{
		_V[i] += v._V[i];
	}

	return Vector$::CODE_NORM;
}

VectorCode Vector::operator-=(std::initializer_list<VectorElemType> list)
{
	Vector v(list);
	return this->operator-=(v);
}

VectorCode Vector::operator*=(Vector const& v)
{
	size_t n1 = _N;
	size_t n2 = v._N;

	if (n1 != n2)
	{
		Log::Error(__Vector::LOG_NAME, "Call of Vector::operator*=: Two vectors of unequal length: %d and %d", n1, n2);
		PRINT_LOCATION;
		return Vector$::CODE_NOT_EQUEL_N;
	}

	if(n1 != 3 && n1 != 4)
	{
		auto message = "Call of Vector::operator*=: Vector has not cross product when n != 3 or 4";
		Log::Error(__Vector::LOG_NAME, message);
		PRINT_LOCATION;
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

VectorCode Vector::operator*=(std::initializer_list<VectorElemType> list)
{
	Vector v(list);
	return this->operator*=(v);
}

VectorCode Vector::operator*=(VectorElemType value)
{
	for(size_t i = 0; i < _N; i++)
	{
		_V[i] *= value;
	}

	return Vector$::CODE_NORM;
}

VectorElemType Vector::operator*(Vector const& v) const
{
	size_t n1 = _N;
	size_t n2 = v._N;

	if (n1 != n2)
	{
		Log::Error(__Vector::LOG_NAME, "Call of Vector::operator*=: Two vectors of unequal length: %d and %d", n1, n2);
		PRINT_LOCATION;
		return Vector$::NOT_INITIALIZED_VALUE;
	}

	VectorElemType result = Vector$::NOT_INITIALIZED_N;

	for(size_t i = 0; i < n1; i++)
	{
		result += _V[i] * v._V[i];
	}

	return result;
}

VectorElemType Vector::operator*(std::initializer_list<VectorElemType> list) const
{
	Vector v(list);
	return this->operator*(v);
}

VectorCode Vector::PrintVector(LogLevel level, const char *decimal_count) const
{
	if(Log::Level() > level) return Vector$::CODE_NORM;

	std::string formatStr = "%.";
	formatStr.append(decimal_count);
	formatStr.append("f\t");

	if (this->_N == Vector$::NOT_INITIALIZED_N)
	{
		auto message = "Call of Vector::PrintVector(): The Vector is NOT Initialized well";
		Log::Error(__Vector::LOG_NAME, message);
		PRINT_LOCATION;
		return Vector$::CODE_NOT_INITIALIZED_N;
	}

	for (size_t i = 0; i < this->_N; i++)
	{
		auto value = _V[i];
		Print(formatStr.c_str(), value);
	}
	PrintLn();

	return Vector$::CODE_NORM;
}

VectorCode Vector::Unitization()
{
	double length_square = 0;
	for(size_t i = 0; i < _N; i++)
	{
		length_square += pow(_V[i], 2);
	}

	if(length_square == 0)
	{
		// the length of vector is 0, need not to unitization.
		return Vector$::CODE_NORM;
	}

	auto length = sqrt(length_square);

	for(size_t i = 0; i <_N; i++)
	{
		_V[i] /= length;
	}

	return Vector$::CODE_NORM;
}