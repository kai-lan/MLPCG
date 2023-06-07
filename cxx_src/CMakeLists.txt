cmake_minimum_required(VERSION 3.10)

project(runUnitTests)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

set(USE_PARTIO ON)
set(USE_CUDA ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_PARTIO -DPARALLEL -DUSE_CUDA -fopenmp -O0 -g -DCGAL_DISABLE_ROUNDING_MATH_CHECK")

# Set compiler optimizations
if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /permissive- /openmp:llvm /MP /arch:AVX2")
else()
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# Add helpful compiler warnings
if(NOT WIN32)
    if (APPLE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-deprecated-declarations")
    endif()
endif()

# Set C++ version
if(NOT WIN32)
  set(CMAKE_CXX_STANDARD 20)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++20")
  add_definitions(-DNOMINMAX)
endif()

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
                 ${CMAKE_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()

# Add Boost (required for AMGCL)
find_package(Boost REQUIRED COMPONENTS
    program_options
    serialization
    unit_test_framework
)


include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
IF(APPLE)
    include_directories(SYSTEM /usr/local/opt/cgal/include/)
    find_package(ZLIB REQUIRED)
    #set(ZLIB_LIBS /usr/local/Cellar/zlib/1.2.11/lib/libz.dylib)
    #set(ZLIB_LIBS /usr/local/opt/zlib/lib/libz.1.dylib)
    include_directories("/usr/local/include/")
    link_directories("/usr/local/lib" "/opt/homebrew/lib")
ELSEIF(WIN32)
    find_package(CGAL CONFIG REQUIRED)
    find_package(ZLIB REQUIRED)
ELSE()  
  find_package(ZLIB REQUIRED)
ENDIF()
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/../../external_libraries/amgcl)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/../../external_libraries/eigen/)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/../../external_libraries/cxxopts/)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/../../external_libraries/partio/src/lib/)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/../../external_libraries/libigl/include/)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/lib/)
message("here, ${CMAKE_CURRENT_SOURCE_DIR}/../../")
include_directories("/usr/local/include")
link_directories("/usr/local/lib")


IF(USE_PARTIO)
  IF(WIN32)
    set(PARTIO_DIR "$<IF:$<CONFIG:Debug>,${CMAKE_CURRENT_SOURCE_DIR}/../../external_libraries/partio/build/src/lib/Debug,${CMAKE_CURRENT_SOURCE_DIR}/../../external_libraries/partio/build/src/lib/Release/>")
  ELSE()
    set(PARTIO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../external_libraries/partio/src/lib)
  ENDIF()
ENDIF()

include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")

# Now simply link against gtest or gtest_main as needed. Eg
include(sources.cmake)
include(lib/library_sources.cmake)

if (Boost_FOUND)
    find_package(VexCL QUIET)
endif()

if(USE_CUDA)
  find_package(CUDA REQUIRED)
  enable_language(CUDA)
  cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_TARGET_ARCH})
  find_package(VexCL)
endif()

if (VexCL_FOUND)
message("VexCL Found")
endif()

add_executable(runUnitTests 
	${library_sources}
	${sources}
)


IF(USE_PARTIO)
  IF(APPLE)
    target_link_libraries(runUnitTests gtest_main pthread TBB ${PARTIO_DIR}/libpartio.a)
  ELSEIF(WIN32)
    target_link_libraries(runUnitTests gtest_main ${PARTIO_DIR}/partio.lib)
  ELSE()
    target_link_libraries(runUnitTests gtest_main pthread tbb ${PARTIO_DIR}/libpartio.so)
  ENDIF()
ELSE()
  IF(APPLE)
    target_link_libraries(runUnitTests gtest_main pthread TBB)
  ELSEIF(WIN32)
    target_link_libraries(runUnitTests gtest_main)
  ELSE()
    target_link_libraries(runUnitTests gtest_main pthread tbb)
  ENDIF()
ENDIF()

#target_link_libraries(example amgcl::amgcl VexCL::OpenCL)
IF(USE_CUDA)
    target_link_libraries(runUnitTests gtest_main pthread tbb ${PARTIO_DIR}/libpartio.so VexCL::CUDA VexCL::OpenCL)
ELSE()
    target_link_libraries(runUnitTests gtest_main pthread tbb ${PARTIO_DIR}/libpartio.so)
ENDIF()

IF(WIN32)
ELSE()
    target_link_libraries(runUnitTests gmp mpfr)
ENDIF()

if(APPLE)
  link_directories(/usr/local/opt/libomp/lib)
endif()

IF(APPLE)
target_link_libraries(runUnitTests ${ZLIB_LIBS} z) 

ELSEIF(WIN32)
    target_link_libraries(runUnitTests CGAL::CGAL)
    target_link_libraries(runUnitTests ZLIB::ZLIB)
ELSE()
	target_link_libraries(runUnitTests stdc++fs ZLIB::ZLIB)
ENDIF()


