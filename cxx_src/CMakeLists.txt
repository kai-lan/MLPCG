cmake_minimum_required(VERSION 3.10)

project(myamgcl)

set(CMAKE_BUILD_TYPE Release)

set(USE_CUDA ON)

if (USE_CUDA)
  message("CUDA enabled")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_CUDA")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O0 -g")

include_directories(${CMAKE_SOURCE_DIR}/amgcl)

# Add Boost (required for AMGCL)
find_package(Boost REQUIRED COMPONENTS
    program_options
    serialization
    unit_test_framework
)


include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/lib/)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/../../external_libraries/eigen/)
message("here, ${CMAKE_CURRENT_SOURCE_DIR}/../../")


# Now simply link against gtest or gtest_main as needed. Eg
include(sources.cmake)
include(lib/library_sources.cmake)

if (Boost_FOUND)
    # find_package(VexCL QUIET)
  add_subdirectory(${CMAKE_SOURCE_DIR}/vexcl)
endif()

if(USE_CUDA)
  find_package(CUDA REQUIRED)
  enable_language(CUDA)
  cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_TARGET_ARCH})
  # find_package(VexCL)
endif()

if (VexCL_FOUND)
  message("VexCL Found")
endif()

add_executable(myamgcl
	${library_sources}
	${sources}
)


IF(USE_CUDA)
    target_link_libraries(myamgcl VexCL::CUDA VexCL::OpenCL)
ENDIF()

if (NOT WIN32)
  add_subdirectory(pybind11)
  pybind11_add_module(pyamgcl_ext pyamgcl.cpp)
endif()
