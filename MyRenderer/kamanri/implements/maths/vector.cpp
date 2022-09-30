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
    this->_V = P<VectorElemType>(new VectorElemType[n]);
}

Vector::Vector(Vector &v) : _V(std::move(v._V)), _N(v._N)
{
    v._N = Vector$::NOT_INITIALIZED_N;
}

Vector::Vector(Vector const& v)
{
    auto pv_old = v._V.get();
    CHECK_MEMORY_IS_ALLOCATED(pv_old, __Vector::LOG_NAME, )

    _N = v._N;
    _V = P<VectorElemType>(new VectorElemType[_N]);
    auto pv_new = _V.get();

    for(size_t i = 0; i < _N; i++)
    {
        *(pv_new + i) = *(pv_old + i);
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
    this->_V = P<VectorElemType>(new VectorElemType[n]);

    auto begin = list.begin();
    auto pV = this->_V.get();

    for (size_t i = 0; begin + i < list.end(); i++)
    {
        *(pV + i) = *(begin + i);
    }
}

DefaultResult Vector::Reset(std::initializer_list<VectorElemType> list)
{
    auto n = list.size();
    if (n != _N)
    {
        Log::Error(__Vector::LOG_NAME, "The size of initializer list(%d) is not equal to vector(%d)", (int)n, _N);
        return DEFAULT_RESULT_EXCEPTION(Vector$::CODE_NOT_EQUEL_N, "The size of initializer list is not equal to vector");
    }

    CHECK_MEMORY_FOR_DEFAULT_RESULT(_V.get(), __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR);

    for (size_t i = 0; list.begin() + i < list.end(); i++)
    {
        *(_V.get() + i) = *(list.begin() + i);
    }

    return DEFAULT_RESULT;
}

P<Vector> Vector::Copy() const
{
    auto new_v = New<Vector>(_N);
    auto pv = _V.get();
    auto pv_new = new_v->_V.get();

    CHECK_MEMORY_IS_ALLOCATED(pv, __Vector::LOG_NAME, new_v)

    for(size_t i = 0; i < _N; i++)
    {
        *(pv_new + i) = *(pv + i);
    }

    return new_v;
}

DefaultResult Vector::CopyFrom(Vector const& v)
{
    auto pv = _V.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(pv, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)
    auto pv2 = v._V.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(pv2, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)

    if(_N != v._N)
    {
        Log::Error(__Vector::LOG_NAME, "This vector n is %d but copy from the vector which n is %d", _N, v._N);
        return DEFAULT_RESULT_EXCEPTION(Vector$::CODE_NOT_EQUEL_N, "Not equal n");
    }

    for(size_t i = 0; i < _N; i++)
    {
        *(pv + i) = *(pv2 + i);
    }

    return DEFAULT_RESULT;

}

Vector& Vector::operator=(Vector& v)
{
    auto pv = v._V.get();

    CHECK_MEMORY_IS_ALLOCATED(pv, __Vector::LOG_NAME, *this)

    _N = v._N;
    _V = std::move(v._V);

    v._N = Vector$::NOT_INITIALIZED_N;

    return *this;
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
    auto pv = _V.get();
    CHECK_MEMORY_FOR_RESULT(VectorElemType, pv, __Vector::LOG_NAME, -1)

    if(n < 0 || n > this->_N)
    {
        Log::Error(__Vector::LOG_NAME, "Index %d out of bound %d", n, this->_N);
        return RESULT_EXCEPTION(VectorElemType, Vector$::CODE_INDEX_OUT_OF_BOUND, "Index out of bound");
    }

    return Result<VectorElemType>(*(pv + n));
}

VectorElemType Vector::GetFast(int n) const
{
    auto pv = _V.get();
    CHECK_MEMORY_IS_ALLOCATED(pv, __Vector::LOG_NAME, -1)

    if(n < 0 || n > this->_N)
    {
        Log::Error(__Vector::LOG_NAME, "Index %d out of bound %d", n, this->_N);
        return Vector$::INVALID_VECTOR_ELEM_TYPE_VALUE;
    }

    return *(pv + n);
}

DefaultResult Vector::Set(size_t index, VectorElemType value) const
{
    auto pv = _V.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(pv, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)

    *(pv + index) = value;
    return DEFAULT_RESULT;
}

DefaultResult Vector::SetAll(VectorElemType value) const
{
    auto pv = _V.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(pv, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)

    for(size_t i = 0; i < _N;i ++)
    {
        *(pv + i) = value;
    }
    return DEFAULT_RESULT;
}

DefaultResult Vector::operator+=(Vector const &v)
{

    auto pv = _V.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(pv, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)
    auto pv2 = v._V.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(pv2, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)


    size_t n1 = _N;
    size_t n2 = v._N;

    if (n1 != n2)
    {
        Log::Error(__Vector::LOG_NAME, "Call of Vector::operator+=: Two vectors of unequal length: %d and %d", n1, n2);
        return DEFAULT_RESULT_EXCEPTION(Vector$::CODE_NOT_EQUEL_N, "Two vectors of unequal length");
    }

    for (size_t i = 0; i < n1; i++)
    {
        *(pv + i) += *(pv2 + i);
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

    auto pv = _V.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(pv, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)
    auto pv2 = v._V.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(pv2, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)

    size_t n1 = _N;
    size_t n2 = v._N;

    if (n1 != n2)
    {
        Log::Error(__Vector::LOG_NAME, "Call of Vector::operator-=: It is impossible to add two vectors of unequal length: %d and %d", n1, n2);
        return DEFAULT_RESULT_EXCEPTION(Vector$::CODE_NOT_EQUEL_N, "It is impossible to add two vectors of unequal length");
    }

    
    for (size_t i = 0; i < n1; i++)
    {
        *(pv + i) -= *(pv2 + i);
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

    auto pv = _V.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(pv, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)
    auto pv2 = v._V.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(pv2, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)

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

    auto v0 = *(pv + 1) * *(pv2 + 2) - *(pv + 2) * *(pv2 + 1);
    auto v1 = *(pv + 2) * *(pv2 + 0) - *(pv + 0) * *(pv2 + 2);
    auto v2 = *(pv + 0) * *(pv2 + 1) - *(pv + 1) * *(pv2 + 0);

    *pv = v0;
    *(pv + 1) = v1;
    *(pv + 2) = v2;

    if(n1 == 4)
    {
        *(pv + 3) = *(pv + 3) * *(pv2 + 3);
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
    auto pv = _V.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(pv, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)
    
    for(size_t i = 0; i < _N; i++)
    {
        *(pv + i) *= value;
    }

    return DEFAULT_RESULT;
}

Result<VectorElemType> Vector::operator*(Vector const& v) const
{
    auto pv = _V.get();
    CHECK_MEMORY_FOR_RESULT(VectorElemType, pv, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)
    auto pv2 = v._V.get();
    CHECK_MEMORY_FOR_RESULT(VectorElemType, pv2, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)

    size_t n1 = _N;
    size_t n2 = v._N;

    if (n1 != n2)
    {
        Log::Error(__Vector::LOG_NAME, "Call of Vector::operator*=: Two vectors of unequal length: %d and %d", n1, n2);
        return RESULT_EXCEPTION(VectorElemType, Vector$::CODE_NOT_EQUEL_N, "Two vectors of unequal length");
    }

    VectorElemType result = 0;

    for(size_t i = 0; i < n1; i++)
    {
        result += *(pv + i) * *(pv2 + i);
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

    auto pV = this->_V.get();

    CHECK_MEMORY_FOR_DEFAULT_RESULT(pV, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)

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
        auto value = *(pV + i);
        Print(formatStr.c_str(), value);
    }
    PrintLn();

    return DEFAULT_RESULT;
}

DefaultResult Vector::Unitization()
{
    auto pv = _V.get();
    CHECK_MEMORY_FOR_DEFAULT_RESULT(pv, __Vector::LOG_NAME, Vector$::CODE_NOT_INITIALIZED_VECTOR)

    double length_square = 0;
    for(auto i = 0; i < _N; i++)
    {
        length_square += pow(*(pv + i), 2);
    }

    if(length_square == 0)
    {
        // the length of vector is 0, need not to unitization.
        return DEFAULT_RESULT;
    }

    auto length = sqrt(length_square);

    for(auto i = 0; i <_N; i++)
    {
        *(pv + i) /= length;
    }

    return DEFAULT_RESULT;
}