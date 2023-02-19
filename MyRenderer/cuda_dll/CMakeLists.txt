# CMakeList.txt: CMakeProject1 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.9)

project(CUDADll)
set(CMAKE_CXX_STANDARD 11)

# generate solarlint compilation database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(..)


# Debug Build
set(CMAKE_BUILD_TYPE Debug)
# Release Build, Ofast
# set(CMAKE_BUILD_TYPE Release)


if(CMAKE_BUILD_TYPE STREQUAL Release)
message(Open_the_Ofast_compile_option)
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall")
endif()


# 将源代码添加到此项目的可执行文件。
#################################################################### cuda_dll.dll
FIND_PACKAGE(CUDA REQUIRED)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

file(GLOB_RECURSE CUDA_DLL ./src/implements/*.cu ./src/implements/*.cpp)
message(CUDA_DLL: ${CUDA_DLL})
CUDA_ADD_LIBRARY(cuda_dll SHARED ${CUDA_DLL})
# cuda_add_library(cuda_dll ${CUDA_DLL})
#####################################################################


# Add dependencies
message(CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE})

