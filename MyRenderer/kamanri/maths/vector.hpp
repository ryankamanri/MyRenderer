#pragma once
#include <initializer_list>
#include "kamanri/utils/memory.hpp"
#include "kamanri/utils/result.hpp"

namespace Kamanri
{
    namespace Maths
    {

        using VectorElemType = double;

        namespace Vector$
        {
            constexpr VectorElemType INVALID_VECTOR_ELEM_TYPE_VALUE = -1;

            constexpr int NOT_INITIALIZED_N = 0;

            // Codes
            constexpr int CODE_NOT_INITIALIZED_N = 100;
            constexpr int CODE_NOT_INITIALIZED_VECTOR = 200;
            constexpr int CODE_NOT_EQUEL_N = 300;
            constexpr int CODE_INVALID_OPERATION = 400;
            constexpr int CODE_INDEX_OUT_OF_BOUND = 500;
        }

        class Vector
        {

        public:
            Vector();
            explicit Vector(size_t n);

            Vector(Vector &&v);
            Vector(Vector const &v);
            Vector(std::initializer_list<VectorElemType> list);
            Vector& operator=(Vector const& v);
            Utils::DefaultResult operator=(std::initializer_list<VectorElemType> list);
            // Get the size of the Vector.
            Utils::Result<std::size_t> N() const;

            // Get the value of the Vector by index
            Utils::Result<VectorElemType> operator[](int n) const;

            // Get without result
            VectorElemType GetFast(int n) const;

            // setter
            Utils::DefaultResult Set(size_t index, VectorElemType value) const;
            Utils::DefaultResult SetAll(VectorElemType value = 0) const;

            Utils::DefaultResult operator+=(Vector const &v);
            Utils::DefaultResult operator+=(std::initializer_list<VectorElemType> list);

            Utils::DefaultResult operator-=(Vector const &v);
            Utils::DefaultResult operator-=(std::initializer_list<VectorElemType> list);

            // Cross product (Only n == 3 || 4 works )
            Utils::DefaultResult operator*=(Vector const &v);
            Utils::DefaultResult operator*=(std::initializer_list<VectorElemType> list);
            Utils::DefaultResult operator*=(VectorElemType value);

            // Dot product
            Utils::Result<VectorElemType> operator*(Vector const &v) const;
            Utils::Result<VectorElemType> operator*(std::initializer_list<VectorElemType> list) const;

            Utils::DefaultResult PrintVector(bool is_print = true, const char *decimal_count = "2") const;

            Utils::DefaultResult Unitization();

        private:
            // The pointer indicated to vector.
            Utils::P<VectorElemType[]> _V;
            // The length of the vector.
            std::size_t _N = Vector$::NOT_INITIALIZED_N;
        };

    }
}
