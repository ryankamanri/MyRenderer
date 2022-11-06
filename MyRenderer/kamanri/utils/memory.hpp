#pragma once
#include <memory>

namespace Kamanri
{
    namespace Utils
    {

        template <typename T>
        using P = std::unique_ptr<T>;

        template <typename T, typename... Ts>
        P<T> New(Ts &&...args)
        {
            return P<T>(new T(std::forward<Ts>(args)...));
        }

        // TODO: 
        // 1. a method NewArray to allocate an object type array
        // 2. a index method to find array location by index
        // 3. update CHECK_MEMORY_IS_ALLOCATED, receive wrapped P instead of bare pointer.
    }
}

#define CHECK_MEMORY_IS_ALLOCATED(p, log_name, return_value)                           \
    if (p == nullptr)                                                                  \
    {                                                                                  \
        Kamanri::Utils::Log::Error(log_name, "Try to visit the not-allocated memory"); \
        return return_value;                                                           \
    }


