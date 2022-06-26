#pragma once
#include <memory>

template<typename T>
using P = std::unique_ptr<T>;

template<typename T, typename... Ts>
P<T> New(Ts&&... args)
{
    return P<T>(new T(std::forward<Ts>(args)...));
}
