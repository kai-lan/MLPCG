cmake_minimum_required(VERSION 3.10)

project(myamgcl CUDA CXX)

set(CMAKE_BUILD_TYPE Release)

set(USE_VEXCL OFF)
set(USE_CUDA ON)
set(MIXED_PRECISION ON)

if (USE_VEXCL)
  message("VexCL enabled")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_VEXCL")
elseif (USE_CUDA)
  message("CUDA enabled")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_CUDA")
else ()
  message("Using built-in (OpenMP)")
endif()

if (MIXED_PRECISION)
  message("Mixed precision enabled")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMIXED_PRECISION")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Add Boost (required for AMGCL)
find_package(Boost REQUIRED COMPONENTS
  program_options
  serialization
  unit_test_framework
)

include_directories(${CMAKE_SOURCE_DIR}/amgcl)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/lib/)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/../../external_libraries/eigen/)

include(lib/library_sources.cmake)

if (USE_VEXCL)
  add_subdirectory(${CMAKE_SOURCE_DIR}/vexcl)
  find_package(CUDA REQUIRED)
  enable_language(CUDA)
  cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_TARGET_ARCH})
elseif (USE_CUDA)
  find_package(CUDA REQUIRED)
  enable_language(CUDA)
  cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_TARGET_ARCH})
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    if (OPENMP_FOUND)
      list(APPEND CUDA_NVCC_FLAGS -Xcompiler ${OpenMP_CXX_FLAGS})
    endif()
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
      list(APPEND CUDA_NVCC_FLAGS
        -std=c++11
        -Xcompiler -Wno-vla
        -Xcompiler -fopenmp
        -Xcompiler -fPIC
      )
    endif()

    add_library(cuda_target INTERFACE)
    target_link_libraries(cuda_target INTERFACE ${CUDA_cusparse_LIBRARY})
  endif()
endif()


if (USE_CUDA)
  add_executable(myamgcl ${library_sources} amg.cu)
else()
  add_executable(myamgcl ${library_sources} amg.cpp)
endif()

if (USE_VEXCL)
  target_link_libraries(myamgcl VexCL::CUDA VexCL::OpenCL)
elseif (USE_CUDA)
  target_link_libraries(myamgcl cuda_target)
endif()

# if (NOT WIN32)
#   add_subdirectory(pybind11)
#   pybind11_add_module(pyamgcl_ext pyamgcl.cpp)
# endif()
