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
    }
}

#define CHECK_MEMORY_IS_ALLOCATED(p, log_name, return_value)                           \
    if (p == nullptr)                                                                  \
    {                                                                                  \
        Kamanri::Utils::Log::Error(log_name, "Try to visit the not-allocated memory"); \
        return return_value;                                                           \
    }


