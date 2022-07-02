#pragma once
#include <initializer_list>
#include "mymemory.h"
#include "myresult.h"

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
        Vector(Vector& v);
        Vector(std::initializer_list<VectorElemType> list);
        P<Vector> Copy() const;
        Vector& operator=(Vector& v);
        // Get the size of the Vector.
        P<MyResult<std::size_t>> N() const;


        // Get the value of the Vector by index
        P<MyResult<VectorElemType>> operator [] (int n) const;
        // setter
        DefaultResult Set(size_t index, VectorElemType value) const;
        DefaultResult SetAll(VectorElemType value = 0) const;
    
        DefaultResult operator += (Vector const& v);
        DefaultResult operator += (std::initializer_list<VectorElemType> list);

        DefaultResult operator -= (Vector const& v);
        DefaultResult operator -= (std::initializer_list<VectorElemType> list);

        // Cross product (Only n == 3 works )
        DefaultResult operator *= (Vector const& v);
        DefaultResult operator *= (std::initializer_list<VectorElemType> list);
        DefaultResult operator *= (VectorElemType value);

        // Dot product
        P<MyResult<VectorElemType>> operator * (Vector const& v) const;
        P<MyResult<VectorElemType>> operator * (std::initializer_list<VectorElemType> list) const;
        

        DefaultResult PrintVector(bool is_print = true, const char* decimal_count = "2") const;
    
    private:
        // The pointer indicated to vector.
        P<VectorElemType> _V;
        // The length of the vector.
        std::size_t _N = MYVECTOR_NOT_INITIALIZED_N;

};


