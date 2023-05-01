cmake_minimum_required (VERSION 3.8)

project(MyRenderer)

################################################################### set properties
set(CMAKE_CXX_STANDARD 17)

# Debug Build
set(CMAKE_BUILD_TYPE Debug)

# Release Build, Ofast
# set(CMAKE_BUILD_TYPE Release)


####################################### build switches
# set(BUILD_CUDA_DLL ON)
set(BUILD_KAMANRI ON)
set(BUILD_EXECUTABLE ON)
set(BUILD_SWIG_PYTHON OFF) # DEPRECATED. Use sbin/build_swig_python.bat instead.

####################################### swig settings
set(SWIG_SOURCE "kamanri/swig_kamanri.i")
set(SWIG_TARGET swig_kamanri)
set(PYTHON_DEBUG_LIBS_PATH "C:/ProgramData/Anaconda3/libs")

# generate solarlint compilation database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

################################################################## includes

include_directories(.)

################################################################### debug / release

if(CMAKE_BUILD_TYPE STREQUAL Release)
  message("Open the Ofast compile option!")
  set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall")
endif()

#################################################################### cuda_dll.dll
if(${BUILD_CUDA_DLL})
  FIND_PACKAGE(CUDA)
  if(NOT CUDA_FOUND)
    message(STATUS "CUDA not found. Project will not be built.")
  else()
    message("Open cuda_dll build!")
    file(GLOB_RECURSE CUDA_DLL cuda_dll/src/implementations/*.cu)
    message(CUDA_DLL: ${CUDA_DLL})
    CUDA_ADD_LIBRARY(cuda_dll SHARED ${CUDA_DLL})
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

