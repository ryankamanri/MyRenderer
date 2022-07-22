#pragma once
#include <memory>

template <typename T>
using P = std::unique_ptr<T>;

template <typename T, typename... Ts>
P<T> New(Ts &&...args)
{
    return P<T>(new T(std::forward<Ts>(args)...));
}

#define CHECK_MEMORY_IS_ALLOCATED(p, log_name, return_value)           \
    if (p == nullptr)                                                  \
    {                                                                  \
        Log::Error(log_name, "Try to visit the not-allocated memory"); \
        return return_value;                                           \
    }

#define CHECK_MEMORY_FOR_DEFAULT_RESULT(p, log_name, code) \
    CHECK_MEMORY_IS_ALLOCATED(p, log_name,                              \
                              New<MyResult<void *>>(                    \
                                  MyResult<void *>::EXCEPTION,          \
                                  code,                                 \
                                  "The memory is not initialized"))

#define CHECK_MEMORY_FOR_RESULT(T, p, log_name, code) \
    CHECK_MEMORY_IS_ALLOCATED(p, log_name,                         \
                              New<MyResult<T>>(                    \
                                  MyResult<T>::EXCEPTION,          \
                                  code,                            \
                                  "The memory is not initialized"))
