# CMakeList.txt: CMakeProject1 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

project(CMakeTest)
set( CMAKE_BUILD_TYPE "Debug" )
set(CMAKE_CXX_STANDARD 11)




# 将源代码添加到此项目的可执行文件。
# aux_source_directory(./mylibs/implements MY_LIBS)
file(GLOB_RECURSE KAMANRI ./kamanri/implements/*.cpp)
add_library(kamanri ${KAMANRI})


add_executable (CMakeProject1 Main.cpp)
target_link_libraries(CMakeProject1 kamanri)

# Add dependencies

# TODO: 如有需要，请添加测试并安装目标。
message(CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR})
message(MY_LIBS_SOURCE_FILES: ${KAMANRI})

