#pragma once
#include "kamanri/maths/vector.hpp"
#include "cuda_dll/src/utils/log.cuh"

__device__ Kamanri::Maths::Vector::Vector(size_t n)
{
	this->_N = Vector$::NOT_INITIALIZED_N;

	if (n != 2 && n != 3 && n != 4)
	{
		Kamanri::Utils::PrintLn("The size of initializer list is not valid: %d", (int) n);
		return;
	}

	this->_N = n;
}

__device__ Kamanri::Maths::Vector::Vector(Kamanri::Maths::Vector const& v)
{
	_N = v._N;

	for (size_t i = 0; i < _N; i++)
	{
		_V[i] = v._V[i];
	}

}

__device__ Kamanri::Maths::Vector::Vector(std::initializer_list<Kamanri::Maths::VectorElemType> list)
{
	this->_N = Vector$::NOT_INITIALIZED_N;

	auto n = list.size();
	if (n != 2 && n != 3 && n != 4)
	{
		Kamanri::Utils::PrintLn("The size of initializer list is not valid: %d", (int)n);
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


__device__ Kamanri::Maths::VectorElemType Kamanri::Maths::Vector::operator*(Kamanri::Maths::Vector const& v) const
{
	size_t n1 = _N;
	size_t n2 = v._N;

	if (n1 != n2)
	{
		Kamanri::Utils::PrintLn("Call of Vector::operator*=: Two vectors of unequal length: %d and %d", n1, n2);
		return Vector$::NOT_INITIALIZED_VALUE;
	}

	Kamanri::Maths::VectorElemType result = Vector$::NOT_INITIALIZED_N;

	for (size_t i = 0; i < n1; i++)
	{
		result += _V[i] * v._V[i];
	}

	return result;
}

__device__ Kamanri::Maths::VectorElemType Kamanri::Maths::Vector::operator-(Kamanri::Maths::Vector const& v)
{
	if (_N != 4)
	{
		Kamanri::Utils::PrintLn("operator-: Invalid N");
		return Vector$::NOT_INITIALIZED_VALUE;
	}
	if (_V[3] != 1 || v._V[3] != 1)
	{
		Kamanri::Utils::PrintLn("operator-: Not uniformed.");
		return Vector$::NOT_INITIALIZED_VALUE;
	}

	return sqrt(pow(_V[0] - v._V[0], 2) + pow(_V[1] - v._V[1], 2) + pow(_V[2] - v._V[2], 2));
}

__device__ Kamanri::Maths::VectorCode Kamanri::Maths::Vector::operator+=(Kamanri::Maths::Vector const &v)
{
	size_t n1 = _N;
	size_t n2 = v._N;

	if (n1 != n2)
	{
		Kamanri::Utils::PrintLn("Call of Vector::operator+=: Two vectors of unequal length: %d and %d", n1, n2);
		return Vector$::CODE_NOT_EQUEL_N;
	}

	for (size_t i = 0; i < n1; i++)
	{
		_V[i] += v._V[i];
	}

	return Vector$::CODE_NORM;
}

__device__ Kamanri::Maths::VectorCode Kamanri::Maths::Vector::operator+=(std::initializer_list<Kamanri::Maths::VectorElemType> list)
{
	Kamanri::Maths::Vector v(list);
	return this->operator+=(v);
}

__device__ Kamanri::Maths::VectorCode Kamanri::Maths::Vector::operator-=(Kamanri::Maths::Vector const& v)
{
	size_t n1 = _N;
	size_t n2 = v._N;

	if (n1 != n2)
	{
		Kamanri::Utils::PrintLn("Call of Vector::operator-=: It is impossible to add two vectors of unequal length: %d and %d", n1, n2);
		return Vector$::CODE_NOT_EQUEL_N;
	}


	for (size_t i = 0; i < n1; i++)
	{
		_V[i] -= v._V[i];
	}

	return Vector$::CODE_NORM;
}

__device__ Kamanri::Maths::VectorCode Kamanri::Maths::Vector::operator*=(Kamanri::Maths::Vector const& v)
{
	size_t n1 = _N;
	size_t n2 = v._N;

	if (n1 != n2)
	{
		Kamanri::Utils::PrintLn("Call of Vector::operator*=: Two vectors of unequal length: %d and %d", n1, n2);
		return Vector$::CODE_NOT_EQUEL_N;
	}

	if(n1 != 3 && n1 != 4)
	{
		auto message = "Call of Vector::operator*=: Vector has not cross product when n != 3 or 4";
		Kamanri::Utils::PrintLn(message);
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

__device__ Kamanri::Maths::Vector& Kamanri::Maths::Vector::operator=(Kamanri::Maths::Vector const& v)
{
	_N = v._N;

	for (size_t i = 0; i < _N; i++)
	{
		_V[i] = v._V[i];
	}

	return *this;
}

__device__ Kamanri::Maths::VectorCode Kamanri::Maths::Vector::operator=(std::initializer_list<Kamanri::Maths::VectorElemType> list)
{
	auto n = list.size();
	if (n != _N)
	{
		Kamanri::Utils::PrintLn("The size of initializer list(%d) is not equal to vector(%d)", (int) n, _N);
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

__device__ Kamanri::Maths::VectorElemType Kamanri::Maths::Vector::operator[](size_t n) const
{
	if (n > this->_N)
	{
		Kamanri::Utils::PrintLn("Kamanri::Maths::Vector::operator[]: Index %llu out of bound %llu\n", n, this->_N);
		return Vector$::NOT_INITIALIZED_VALUE;
	}

	return _V[n];
}

__device__ Kamanri::Maths::VectorCode Kamanri::Maths::Vector::Set(size_t index, Kamanri::Maths::VectorElemType value)
{
	if (index > this->_N)
	{
		Kamanri::Utils::PrintLn("Kamanri::Maths::Vector::Set: Index %llu out of bound %llu\n", index, this->_N);
		return Vector$::NOT_INITIALIZED_VALUE;
	}
	_V[index] = value;
	return Vector$::CODE_NORM;
}

__device__ Kamanri::Maths::VectorCode Kamanri::Maths::Vector::Unitization()
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
