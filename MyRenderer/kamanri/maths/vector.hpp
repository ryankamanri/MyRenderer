#pragma once
#include "kamanri/utils/logs_declare.hpp"

namespace std
{
	template<class T>
	class initializer_list;
} // namespace std


namespace Kamanri
{
	namespace Maths
	{

		using VectorElemType = double;
		using VectorCode = int;

		constexpr VectorElemType NOT_INITIALIZED_VALUE = -1;
		constexpr std::size_t NOT_INITIALIZED_N = 0;
		constexpr int MAX_SUPPORTED_DIMENSION = 4;
		
		class Vector$
		{
			public:
			static inline VectorElemType NOT_INITIALIZED_VALUE = Maths::NOT_INITIALIZED_N;
			static inline std::size_t NOT_INITIALIZED_N = Maths::NOT_INITIALIZED_N;
			static inline int MAX_SUPPORTED_DIMENSION = Maths::MAX_SUPPORTED_DIMENSION;

			// Codes
			static inline VectorCode CODE_NORM = 0;
			static inline VectorCode CODE_NOT_INITIALIZED_N = 100;
			static inline VectorCode CODE_NOT_INITIALIZED_VECTOR = 200;
			static inline VectorCode CODE_NOT_EQUEL_N = 300;
			static inline VectorCode CODE_INVALID_OPERATION = 400;
			static inline VectorCode CODE_INDEX_OUT_OF_BOUND = 500;
		};

		class Vector
		{

		public:
			Vector();
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			explicit Vector(size_t n);

			// Vector(Vector &&v);
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			Vector(Vector const &v);
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			Vector(std::initializer_list<VectorElemType> list);
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			Vector& operator=(Vector const& v);
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			VectorCode operator=(std::initializer_list<VectorElemType> list);
			// Get the size of the Vector.
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			inline std::size_t N() const { return _N; }

			// Get the value of the Vector by index
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			VectorElemType operator[](size_t n) const;

			// Get without result
			// VectorElemType GetFast(size_t n) const;

			// setter
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			VectorCode Set(size_t index, VectorElemType value);
			VectorCode SetAll(VectorElemType value = 0);

#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			VectorElemType operator-(Vector const& v);
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			VectorCode operator+=(Vector const &v);
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			VectorCode operator+=(std::initializer_list<VectorElemType> list);

#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			VectorCode operator-=(Vector const &v);
			VectorCode operator-=(std::initializer_list<VectorElemType> list);

			// Cross product (Only n == 3 || 4 works )
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			VectorCode operator*=(Vector const &v);
			VectorCode operator*=(std::initializer_list<VectorElemType> list);
			VectorCode operator*=(VectorElemType value);

			// Dot product
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			VectorElemType operator*(Vector const &v) const;
			VectorElemType operator*(std::initializer_list<VectorElemType> list) const;

			VectorCode PrintVector(Utils::LogLevel level = Utils::Log$::INFO_LEVEL, const char *decimal_count = "2") const;
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			VectorCode Unitization();

		private:
			// The pointer indicated to vector.
			VectorElemType _V[Maths::MAX_SUPPORTED_DIMENSION];
			// The length of the vector.
			std::size_t _N = Maths::NOT_INITIALIZED_N;
		};

	}
}
