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



Vector::Vector() = default;

Vector::Vector(size_t n)
{
    this->_N = Vector$::NOT_INITIALIZED_N;

    if (n != 2 && n != 3 && n != 4)
    {
        Log::Error(__Vector::LOG_NAME, "The size of initializer list is not valid: %d", (int)n);
        return;
    }

    this->_N = n;
    this->_V = NewArray<VectorElemType>(n);
}

Vector::Vector(Vector &&v) : _V(std::move(v._V)), _N(v._N)
{
    v._N = Vector$::NOT_INITIALIZED_N;
}

Vector::Vector(Vector const& v)
{
    CHECK_MEMORY_IS_ALLOCATED(v._V, __Vector::LOG_NAME, )

    _N = v._N;
    _V = NewArray<VectorElemType>(_N);

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
        return;
    }

    this->_N = n;
    this->_V = NewArray<VectorElemType>(n);

    auto i = 0;
    for(auto list_elem: list)
    {
        _V[i] = list_elem;
        i++;
    }

}

Vector& Vector::operator=(Vector const& v)
{
    CHECK_MEMORY_IS_ALLOCATED(v._V, __Vector::LOG_NAME, *this)

    if(_N != v._N)
    {
        _N = v._N;
        _V = NewArray<VectorElemType>(_N);
    }

    for(size_t i = 0; i < _N; i++)
    {
        _V[i] = v._V[i];
    }

    return *this;
}

DefaultResult Vector::operator=(std::initializer_list<VectorElemType> list)
{
    auto n = list.size();
    if (n != _N)
    {
        Log::Error(__Vector::LOG_NAME, "The size of initializer list(%d) is not equal to vector(%d)", (int)n, _N);
        return DEFAULT_RESULT_EXCEPTION(Vector$::CODE_NOT_EQUEL_N, "The size of initializer list is not equal to vector");
    }

    CHECK_MEMORY_FOR_DEFAULT_RESULT(_V, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR);

    auto i = 0;
    for(auto list_elem: list)
    {
        _V[i] = list_elem;
        i++;
    }

    return DEFAULT_RESULT;
}

Result<std::size_t> Vector::N() const
{
    if (_N == Vector$::CODE_NOT_INITIALIZED_N)
    {
        auto message = "Call of Vector::N(): The vector is not initialized";
        Log::Error(__Vector::LOG_NAME, message);
        return RESULT_EXCEPTION(std::size_t, Vector$::CODE_INVALID_OPERATION, message);
    }
    return Result<std::size_t>(_N);
}

Result<VectorElemType> Vector::operator[](int n) const
{
    CHECK_MEMORY_FOR_RESULT(VectorElemType, _V, __Vector::LOG_NAME, -1)

    if(n < 0 || n > this->_N)
    {
        Log::Error(__Vector::LOG_NAME, "Index %d out of bound %d", n, this->_N);
        return RESULT_EXCEPTION(VectorElemType, Vector$::CODE_INDEX_OUT_OF_BOUND, "Index out of bound");
    }

    return Result<VectorElemType>(_V[n]);
}

VectorElemType Vector::GetFast(int n) const
{
    CHECK_MEMORY_IS_ALLOCATED(_V, __Vector::LOG_NAME, Vector$::INVALID_VECTOR_ELEM_TYPE_VALUE)

    if(n < 0 || n > this->_N)
    {
        Log::Error(__Vector::LOG_NAME, "Index %d out of bound %d", n, this->_N);
        return Vector$::INVALID_VECTOR_ELEM_TYPE_VALUE;
    }

    return _V[n];
}

DefaultResult Vector::Set(size_t index, VectorElemType value) const
{
    CHECK_MEMORY_FOR_DEFAULT_RESULT(_V, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)

    _V[index] = value;
    return DEFAULT_RESULT;
}

DefaultResult Vector::SetAll(VectorElemType value) const
{
    CHECK_MEMORY_FOR_DEFAULT_RESULT(_V, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)

    for(size_t i = 0; i < _N;i ++)
    {
        _V[i] = value;
    }
    return DEFAULT_RESULT;
}

DefaultResult Vector::operator+=(Vector const &v)
{

    CHECK_MEMORY_FOR_DEFAULT_RESULT(_V, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)
    CHECK_MEMORY_FOR_DEFAULT_RESULT(v._V, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)


    size_t n1 = _N;
    size_t n2 = v._N;

    if (n1 != n2)
    {
        Log::Error(__Vector::LOG_NAME, "Call of Vector::operator+=: Two vectors of unequal length: %d and %d", n1, n2);
        return DEFAULT_RESULT_EXCEPTION(Vector$::CODE_NOT_EQUEL_N, "Two vectors of unequal length");
    }

    for (size_t i = 0; i < n1; i++)
    {
        _V[i] += v._V[i];
    }

    return DEFAULT_RESULT;
}

DefaultResult Vector::operator+=(std::initializer_list<VectorElemType> list)
{
    Vector v(list);
    return this->operator+=(v);
}

DefaultResult Vector::operator-=(Vector const &v)
{

    CHECK_MEMORY_FOR_DEFAULT_RESULT(_V, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)
    CHECK_MEMORY_FOR_DEFAULT_RESULT(v._V, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)

    size_t n1 = _N;
    size_t n2 = v._N;

    if (n1 != n2)
    {
        Log::Error(__Vector::LOG_NAME, "Call of Vector::operator-=: It is impossible to add two vectors of unequal length: %d and %d", n1, n2);
        return DEFAULT_RESULT_EXCEPTION(Vector$::CODE_NOT_EQUEL_N, "It is impossible to add two vectors of unequal length");
    }

    
    for (size_t i = 0; i < n1; i++)
    {
        _V[i] += v._V[i];
    }

    return DEFAULT_RESULT;
}

DefaultResult Vector::operator-=(std::initializer_list<VectorElemType> list)
{
    Vector v(list);
    return this->operator-=(v);
}

DefaultResult Vector::operator*=(Vector const& v)
{

    CHECK_MEMORY_FOR_DEFAULT_RESULT(_V, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)
    CHECK_MEMORY_FOR_DEFAULT_RESULT(v._V, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)

    size_t n1 = _N;
    size_t n2 = v._N;

    if (n1 != n2)
    {
        Log::Error(__Vector::LOG_NAME, "Call of Vector::operator*=: Two vectors of unequal length: %d and %d", n1, n2);
        return DEFAULT_RESULT_EXCEPTION(Vector$::CODE_NOT_EQUEL_N, "Two vectors of unequal length");
    }

    if(n1 != 3 && n1 != 4)
    {
        auto message = "Call of Vector::operator*=: Vector has not cross product when n != 3 or 4";
        Log::Error(__Vector::LOG_NAME, message);
        return DEFAULT_RESULT_EXCEPTION(Vector$::CODE_INVALID_OPERATION, message);
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

    return DEFAULT_RESULT;
    
}

DefaultResult Vector::operator*=(std::initializer_list<VectorElemType> list)
{
    Vector v(list);
    return this->operator*=(v);
}

DefaultResult Vector::operator*=(VectorElemType value)
{
    CHECK_MEMORY_FOR_DEFAULT_RESULT(_V, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)
    
    for(size_t i = 0; i < _N; i++)
    {
        _V[i] *= value;
    }

    return DEFAULT_RESULT;
}

Result<VectorElemType> Vector::operator*(Vector const& v) const
{
    CHECK_MEMORY_FOR_RESULT(VectorElemType, _V, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)
    CHECK_MEMORY_FOR_RESULT(VectorElemType, v._V, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)

    size_t n1 = _N;
    size_t n2 = v._N;

    if (n1 != n2)
    {
        Log::Error(__Vector::LOG_NAME, "Call of Vector::operator*=: Two vectors of unequal length: %d and %d", n1, n2);
        return RESULT_EXCEPTION(VectorElemType, Vector$::CODE_NOT_EQUEL_N, "Two vectors of unequal length");
    }

    VectorElemType result = Vector$::NOT_INITIALIZED_N;

    for(size_t i = 0; i < n1; i++)
    {
        result += _V[i] * v._V[i];
    }

    return Result<VectorElemType>(result);
}

Result<VectorElemType> Vector::operator*(std::initializer_list<VectorElemType> list) const
{
    Vector v(list);
    return this->operator*(v);
}

DefaultResult Vector::PrintVector(bool is_print, const char *decimal_count) const
{
    if(!is_print) return DEFAULT_RESULT;

    CHECK_MEMORY_FOR_DEFAULT_RESULT(_V, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)

    std::string formatStr = "%.";
    formatStr.append(decimal_count);
    formatStr.append("f\t");

    if (this->_N == Vector$::NOT_INITIALIZED_N)
    {
        auto message = "Call of Vector::PrintVector(): The Vector is NOT Initialized well";
        Log::Error(__Vector::LOG_NAME, message);

        return DEFAULT_RESULT_EXCEPTION(Vector$::CODE_NOT_INITIALIZED_N, message);
    }

    for (size_t i = 0; i < this->_N; i++)
    {
        auto value = _V[i];
        Print(formatStr.c_str(), value);
    }
    PrintLn();

    return DEFAULT_RESULT;
}

DefaultResult Vector::Unitization()
{
    CHECK_MEMORY_FOR_DEFAULT_RESULT(_V, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)

    double length_square = 0;
    for(auto i = 0; i < _N; i++)
    {
        length_square += pow(_V[i], 2);
    }

    if(length_square == 0)
    {
        // the length of vector is 0, need not to unitization.
        return DEFAULT_RESULT;
    }

    auto length = sqrt(length_square);

    for(auto i = 0; i <_N; i++)
    {
        _V[i] /= length;
    }

    return DEFAULT_RESULT;
}