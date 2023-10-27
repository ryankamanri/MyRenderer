cmake_minimum_required (VERSION 3.15)
################################################################## cmake policy
# https://cmake.org/cmake/help/latest/policy/CMP0091.html
# MSVC runtime library flags are selected by an abstraction.
cmake_policy(SET CMP0091 NEW)
project(MyRenderer)

################################################################### set properties
set(CMAKE_CXX_STANDARD 17)

# Debug Build
# set(CMAKE_BUILD_TYPE Debug)

# Release Build, Ofast
set(CMAKE_BUILD_TYPE Release)


####################################### build switches
set(BUILD_CUDA_DLL ON)
set(BUILD_KAMANRI ON)
set(BUILD_EXECUTABLE ON)
set(BUILD_SWIG_PYTHON OFF) # DEPRECATED. Use sbin/build_swig_python.bat instead.

####################################### swig settings (DEPRECATED)
set(SWIG_SOURCE "kamanri/swig_kamanri.i")
set(SWIG_TARGET swig_kamanri)
set(PYTHON_DEBUG_LIBS_PATH "C:/ProgramData/Anaconda3/libs")

####################################### other settings
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # generate solarlint compilation database

set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL) # link to given library
set(NVCC_OPTIONS "-Xcompiler \"/wd 4819 /MD\" -std c++17") # need to set compiler options for nvcc individually

add_compile_options("/wd 4819") # disable warning 4819

################################################################## includes

include_directories(.)

################################################################### debug / release

if(CMAKE_BUILD_TYPE STREQUAL Release)
  message("Open the Release compile option!")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /MD")
endif()

#################################################################### cuda_dll.dll
if(${BUILD_CUDA_DLL})
  FIND_PACKAGE(CUDA)
  if(NOT CUDA_FOUND)
    message(STATUS "CUDA not found. Project will not be built.")
  else()
    message("Open cuda_dll build!")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${NVCC_OPTIONS}")
    file(GLOB_RECURSE CUDA_DLL cuda_dll/src/implementations/*.cu)
    message(CUDA_DLL: ${CUDA_DLL})
    cuda_add_library(cuda_dll SHARED ${CUDA_DLL})
  endif(NOT CUDA_FOUND)
endif()
##################################################################### kamanri
if(${BUILD_KAMANRI})
  message("Open kamanri build!")
  file(GLOB_RECURSE KAMANRI ./kamanri/implementations/*.cpp)
  message(KAMANRI: ${KAMANRI})
  add_library(kamanri ${KAMANRI})
endif()
##################################################################### kamanri_swig_python(DEPRECATED)

if(${BUILD_SWIG_PYTHON})
  message("Open swig python build!")
  cmake_policy(SET CMP0078 NEW) # make swig target() name not be changed.
  cmake_policy(SET CMP0086 NEW) # make swig output module name can be set.

  find_package(SWIG)
  find_package(PythonLibs)
  if(NOT SWIG_FOUND)
    message("Swig not found!")
  elseif(NOT PythonLibs_FOUND)
    message("Python libraries not found!")
  else()
    include(${SWIG_USE_FILE})
    include_directories(${PYTHON_INCLUDE_PATH})
    set(CMAKE_SWIG_FLAGS "")
    set_source_files_properties(
      ${SWIG_SOURCE} PROPERTIES 
      CPLUSPLUS ON
      SWIG_FLAGS "-includeall"
    )
    link_directories(${PYTHON_DEBUG_LIBS_PATH}) # link python debug binaries lib which is required by swig.
    swig_add_library(${SWIG_TARGET} TYPE MODULE LANGUAGE python SOURCES ${SWIG_SOURCE})
    target_link_libraries(${SWIG_TARGET} kamanri)
  endif()
endif()


######################################################################## executable
if(${BUILD_EXECUTABLE})
message("Open executable build!")
  add_executable(MyRenderer Main.cpp)
  target_link_libraries(MyRenderer kamanri)
endif()

message(CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE})

