%module Maths
%feature("python:annotations", "c");

%{
#include "kamanri/maths/math.hpp"
#include "kamanri/maths/vector.hpp"
#include "kamanri/maths/smatrix.hpp"
%}

%include "kamanri/maths/math.hpp"
%include "kamanri/maths/vector.hpp"
%include <std_vector.i>


%template(ElementList) std::vector<double>; // need to declare template type

%extend Kamanri::Maths::Vector
{
	Vector(const std::vector<VectorElemType>& list)
	{
		auto n = list.size();
		if (n != 2 && n != 3 && n != 4)
		{
			printf("The size of initializer list is not valid: %llu", n);
			return nullptr;
		}

		auto vector = new Kamanri::Maths::Vector(n);

		auto i = 0;
		for (auto list_elem : list)
		{
			vector->Set(i, list_elem);
			i++;
		}

		return vector;
	}

	Vector& Copy(Vector const& v)
	{
		return $self->operator=(v);
	}

	VectorElemType Get(size_t n) const
	{
		return $self->operator[](n);
	}

}

%include "kamanri/maths/smatrix.hpp"

%extend Kamanri::Maths::SMatrix 
{
	SMatrix(const std::vector<SMatrixElemType>& list)
	{
		auto size = list.size();

		if (size != 4 && size != 9 && size != 16)
		{
			printf("The size of initializer list is not valid: %llu", size);
			return nullptr;
		}

		auto n = (size_t)sqrt((double) size);

		auto smatrix = new Kamanri::Maths::SMatrix(n);

		
		for (auto i = 0; i < n; i++)
		{
			for(auto j = 0; j < n; j++)
			{
				smatrix->Set(i, j, list[i * n + j]);
			}
		}

		return smatrix;
	}

}

