# CMakeList.txt: CMakeProject1 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

project(CMakeTest)
set( CMAKE_BUILD_TYPE "Debug" )
set(CMAKE_CXX_STANDARD 11)




# 将源代码添加到此项目的可执行文件。
aux_source_directory(./mylibs/implements MY_LIBS)
add_library(my_libs ${MY_LIBS})

aux_source_directory(./renderers/implements RENDERERS)
add_library(renderers ${MY_LIBS} ${RENDERERS})

add_executable (CMakeProject1 "Main.cpp")
target_link_libraries(CMakeProject1 my_libs)
target_link_libraries(CMakeProject1 renderers)

# Add dependencies

# TODO: 如有需要，请添加测试并安装目标。
message(CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR})
