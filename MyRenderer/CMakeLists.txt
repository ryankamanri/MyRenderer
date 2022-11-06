# CMakeList.txt: CMakeProject1 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

project(MyRenderer)
set(CMAKE_CXX_STANDARD 11)

# generate solarlint compilation database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


include_directories(.)


# Debug Build
set(CMAKE_BUILD_TYPE "Debug")

# Release Build, Ofast
# set(CMAKE_BUILD_TYPE Release)


if(CMAKE_BUILD_TYPE STREQUAL Release)
message(Open_the_Ofast_compile_option)
set(CMAKE_CXX_FLAGS_RELEASE -Ofast)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall")
endif()


# 将源代码添加到此项目的可执行文件。
# aux_source_directory(./mylibs/implements MY_LIBS)
file(GLOB_RECURSE KAMANRI ./kamanri/implements/*.cpp)
add_library(kamanri ${KAMANRI})


# use in visual studio
# add_executable(MyRenderer WIN32 Main.cpp)
# use in general
add_executable(MyRenderer Main.cpp)

target_link_libraries(MyRenderer kamanri)

# Add dependencies

# TODO: 如有需要，请添加测试并安装目标。
message(CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE})

