cmake_minimum_required(VERSION 3.10)

project(myamgcl CUDA CXX)

set(CMAKE_BUILD_TYPE Release)



set(COMPILE_VEXCL ON)
set(COMPILE_CUDA ON)
set(COMPILE_OPENMP ON)
set(MIXED_PRECISION ON)

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

add_library(amgcl INTERFACE)
add_library(amgcl::amgcl ALIAS amgcl)
target_compile_features(amgcl INTERFACE
    cxx_auto_type
    cxx_range_for
    cxx_rvalue_references
    cxx_right_angle_brackets
    cxx_static_assert
)

if (Boost_FOUND)
    target_include_directories(amgcl SYSTEM INTERFACE
        ${Boost_INCLUDE_DIRS}
        )
else()
    target_compile_definitions(amgcl INTERFACE AMGCL_NO_BOOST)
endif()
if (OPENMP_FOUND)
    target_compile_options(amgcl INTERFACE ${OpenMP_CXX_FLAGS})
    target_link_libraries(amgcl INTERFACE
        $<$<CXX_COMPILER_ID:GNU>:${OpenMP_CXX_FLAGS}>
        $<$<CXX_COMPILER_ID:Clang>:${OpenMP_CXX_FLAGS}>
        $<$<CXX_COMPILER_ID:Intel>:${OpenMP_CXX_FLAGS}>
        )
endif ()
target_compile_options(amgcl INTERFACE
    # Compiler is GNU (g++):
    $<$<CXX_COMPILER_ID:GNU>:$<BUILD_INTERFACE:-Wall;-Wextra;-Wpedantic>>
    # Compiler is Clang:
    $<$<CXX_COMPILER_ID:Clang>:$<BUILD_INTERFACE:-Wall;-Wextra;-Wpedantic;-Wno-c99-extensions>>
    # Compiler is MSVC:
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
    $<$<CXX_COMPILER_ID:MSVC>:/wd4715>
    )

target_compile_definitions(amgcl INTERFACE
    # Compiler is MSVC:
    $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX>
    $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>
    $<$<CXX_COMPILER_ID:MSVC>:_VARIADIC_MAX=10>
    $<$<CXX_COMPILER_ID:MSVC>:_SCL_SECURE_NO_WARNINGS>
    )

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(${CMAKE_SOURCE_DIR}/amgcl)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/lib/)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/../../external_libraries/eigen/)

include(lib/library_sources.cmake)

if (COMPILE_CUDA OR COMPILE_VEXCL)
  find_package(CUDA REQUIRED)
  set(CUDA_TARGET_ARCH "Ampere")
  cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_TARGET_ARCH})
endif()

if (COMPILE_VEXCL)
  message("Compiling VexCL")
  add_subdirectory(${CMAKE_SOURCE_DIR}/vexcl)
  add_executable(myamgcl_vexcl ${library_sources} amg.cpp)
  target_link_libraries(myamgcl_vexcl amgcl VexCL::CUDA VexCL::OpenCL Eigen3::Eigen)
  target_compile_definitions(myamgcl_vexcl PRIVATE -DUSE_VEXCL)
endif()

if (COMPILE_CUDA)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    find_package(CUDA)
    if (CUDA_FOUND)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")

        # set(CUDA_TARGET_ARCH "Kepler Maxwell Pascal" CACHE STRING "Target architecture(s) for CUDA")
        cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_TARGET_ARCH})

        if (OPENMP_FOUND)
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler ${OpenMP_CXX_FLAGS})
        endif()

        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
      list(APPEND CUDA_NVCC_FLAGS
                ${CUDA_ARCH_FLAGS} -Wno-deprecated-gpu-targets)

            list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fPIC -Xcompiler -Wno-vla)
        endif()

        add_library(cuda_target INTERFACE)
        target_link_libraries(cuda_target INTERFACE ${CUDA_cusparse_LIBRARY})
    endif()
  endif()
  add_executable(myamgcl_cuda ${library_sources} amg.cu)
  target_link_libraries(myamgcl_cuda amgcl cuda_target Eigen3::Eigen)
  target_compile_definitions(myamgcl_cuda PRIVATE -DUSE_CUDA)
endif()

if (COMPILE_OPENMP)
  message("Compiling OpenMP(builtin)")
  add_executable(myamgcl ${library_sources} amg.cpp)
  target_link_libraries(myamgcl amgcl Eigen3::Eigen)
endif()

# if (MIXED_PRECISION) # sadly, it seems mixed precision is not supported here
#   message("Mixed precision enabled")
#   if (COMPILE_OPENMP)
#     target_compile_definitions(myamgcl PRIVATE -DMIXED_PRECISION)
#   endif()
#   if (COMPILE_VEXCL)
#     target_compile_definitions(myamgcl_vexcl PRIVATE -DMIXED_PRECISION)
#   endif()
# endif()

# if (NOT WIN32)
#   add_subdirectory(pybind11)
#   pybind11_add_module(pysolverconfig ${library_sources} pysolverconfig.cpp)

  # if (COMPILE_VEXCL)
  #   pybind11_add_module(pyamgcl_vexcl ${library_sources} pyamgcl_vexcl.cpp)
  #   target_link_libraries(pyamgcl_vexcl PRIVATE VexCL::CUDA VexCL::OpenCL Eigen3::Eigen)
  #   target_compile_definitions(pyamgcl_vexcl PRIVATE -DUSE_VEXCL)
  #   if (MIXED_PRECISION)
  #     target_compile_definitions(pyamgcl_vexcl PRIVATE -DMIXED_PRECISION)
  #   endif()
  # endif()
  # if (COMPILE_CUDA)
  #   pybind11_add_module(pyamgcl_cuda ${library_sources} pyamgcl.cu)
  #   target_link_libraries(pyamgcl_cuda PRIVATE cuda_target Eigen3::Eigen)
  #   target_compile_definitions(pyamgcl_cuda PRIVATE -DUSE_CUDA)
  #   set_property(TARGET pyamgcl_cuda PROPERTY CUDA_ARCHITECTURES native)
  # endif()
  # if (COMPILE_OPENMP)
  #   pybind11_add_module(pyamgcl ${library_sources} pyamgcl.cpp)
  #   if (MIXED_PRECISION)
  #     target_compile_definitions(pyamgcl PRIVATE -DMIXED_PRECISION)
  #   endif()
  # endif()

# endif()
