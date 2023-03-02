#pragma once
#include <Windows.h>
#include "string.hpp"
#include "logs.hpp"

#define c_export extern "C" __declspec(dllexport)


#define func_type(func) Type_##func
#define func_p(func) (*func_type(func))

#define dll HINSTANCE
#define load_dll(dll_src, mount, log_name) mount = LoadLibrary(STR(dll_src.dll)); \
if(!mount) { Kamanri::Utils::Log::Error(log_name, "Failed to load %s, error code: %d.\n", STR(dll_src), GetLastError()); PRINT_LOCATION; }


#define import_func(func, dll_src, mount, log_name) mount = (func_type(func))GetProcAddress(dll_src, STR(func)); \
if(!dll_src) { Kamanri::Utils::Log::Error(log_name, "Invalid dll source %s\n", STR(dll_src)); PRINT_LOCATION; } \
if(!mount) { Kamanri::Utils::Log::Error(log_name, "Failed to import %s from %s, error code: %d.\n", STR(func), STR(dll_src), GetLastError()); PRINT_LOCATION; }
