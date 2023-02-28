#pragma once
#include <initializer_list>
#include "kamanri/utils/logs.hpp"

namespace Kamanri
{
	namespace Maths
	{

		using VectorElemType = double;
		using VectorCode = int;

		namespace Vector$
		{
			constexpr VectorElemType NOT_INITIALIZED_VALUE = -1;
			constexpr int NOT_INITIALIZED_N = 0;
			constexpr int MAX_SUPPORTED_DIMENSION = 4;

			// Codes
			constexpr VectorCode CODE_NORM = 0;
			constexpr VectorCode CODE_NOT_INITIALIZED_N = 100;
			constexpr VectorCode CODE_NOT_INITIALIZED_VECTOR = 200;
			constexpr VectorCode CODE_NOT_EQUEL_N = 300;
			constexpr VectorCode CODE_INVALID_OPERATION = 400;
			constexpr VectorCode CODE_INDEX_OUT_OF_BOUND = 500;
		}

		class Vector
		{

		public:
			Vector();
			explicit Vector(size_t n);

			// Vector(Vector &&v);
			Vector(Vector const &v);
			Vector(std::initializer_list<VectorElemType> list);
			Vector& operator=(Vector const& v);
			VectorCode operator=(std::initializer_list<VectorElemType> list);
			// Get the size of the Vector.
			inline std::size_t N() const { return _N; }

			// Get the value of the Vector by index
			VectorElemType operator[](size_t n) const;

			// Get without result
			// VectorElemType GetFast(size_t n) const;

			// setter
			VectorCode Set(size_t index, VectorElemType value);
			VectorCode SetAll(VectorElemType value = 0);

			VectorElemType operator-(Vector const& v);

			VectorCode operator+=(Vector const &v);
			VectorCode operator+=(std::initializer_list<VectorElemType> list);

			VectorCode operator-=(Vector const &v);
			VectorCode operator-=(std::initializer_list<VectorElemType> list);

			// Cross product (Only n == 3 || 4 works )
			VectorCode operator*=(Vector const &v);
			VectorCode operator*=(std::initializer_list<VectorElemType> list);
			VectorCode operator*=(VectorElemType value);

			// Dot product
			VectorElemType operator*(Vector const &v) const;
			VectorElemType operator*(std::initializer_list<VectorElemType> list) const;

			VectorCode PrintVector(Utils::LogLevel level = Utils::Log$::INFO_LEVEL, const char *decimal_count = "2") const;

			VectorCode Unitization();

		private:
			// The pointer indicated to vector.
			VectorElemType _V[Vector$::MAX_SUPPORTED_DIMENSION];
			// The length of the vector.
			std::size_t _N = Vector$::NOT_INITIALIZED_N;
		};

	}
}
