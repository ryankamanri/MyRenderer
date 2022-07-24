#pragma once
#include <initializer_list>
#include "../utils/memory.h"
#include "../utils/result.h"

namespace Kamanri
{
    namespace Maths
    {
        namespace Vectors
        {
            using VectorElemType = double;

            constexpr int MYVECTOR_NOT_INITIALIZED_N = 0;

            // Codes
            constexpr int MYVECTOR_CODE_NOT_INITIALIZED_N = 100;
            constexpr int MYVECTOR_CODE_NOT_INITIALIZED_VECTOR = 200;
            constexpr int MYVECTOR_CODE_NOT_EQUEL_N = 300;
            constexpr int MYVECTOR_CODE_INVALID_OPERATION = 400;
            constexpr int MYVECTOR_CODE_INDEX_OUT_OF_BOUND = 500;

            class Vector
            {

            public:
                Vector();
                explicit Vector(size_t n);
                Vector(Vector &v);
                Vector(std::initializer_list<VectorElemType> list);
                Utils::Memory::P<Vector> Copy() const;
                Vector &operator=(Vector &v);
                // Get the size of the Vector.
                Utils::Result::PMyResult<std::size_t> N() const;

                // Get the value of the Vector by index
                Utils::Result::PMyResult<VectorElemType> operator[](int n) const;
                // setter
                Utils::Result::DefaultResult Set(size_t index, VectorElemType value) const;
                Utils::Result::DefaultResult SetAll(VectorElemType value = 0) const;

                Utils::Result::DefaultResult operator+=(Vector const &v);
                Utils::Result::DefaultResult operator+=(std::initializer_list<VectorElemType> list);

                Utils::Result::DefaultResult operator-=(Vector const &v);
                Utils::Result::DefaultResult operator-=(std::initializer_list<VectorElemType> list);

                // Cross product (Only n == 3 works )
                Utils::Result::DefaultResult operator*=(Vector const &v);
                Utils::Result::DefaultResult operator*=(std::initializer_list<VectorElemType> list);
                Utils::Result::DefaultResult operator*=(VectorElemType value);

                // Dot product
                Utils::Result::PMyResult<VectorElemType> operator*(Vector const &v) const;
                Utils::Result::PMyResult<VectorElemType> operator*(std::initializer_list<VectorElemType> list) const;

                Utils::Result::DefaultResult PrintVector(bool is_print = true, const char *decimal_count = "2") const;

            private:
                // The pointer indicated to vector.
                Utils::Memory::P<VectorElemType> _V;
                // The length of the vector.
                std::size_t _N = MYVECTOR_NOT_INITIALIZED_N;
            };

        }
    }
}
