cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch_extension_test)

# set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -m64 -c")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -m64 -c -Xptxas=-v")

add_executable(test sm_linear_3d_kernel.cu)
target_link_libraries(test "${TORCH_LIBRARIES}")
# set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

# add_library(test_ptx OBJECT sm_linear_3d_kernel.cu)
# set_property(TARGET test_ptx PROPERTY CUDA_PTX_COMPILATION ON)
# target_link_libraries(test_ptx "${TORCH_LIBRARIES}")