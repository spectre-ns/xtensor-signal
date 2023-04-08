############################################################################
# Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.1)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(xtensor-signal-benchmark)

    find_package(xsimd REQUIRED)
    set(XSIMD_INCLUDE_DIR ${xsimd_INCLUDE_DIRS})

    find_package(xtensor REQUIRED CONFIG)
    set(XTENSOR_INCLUDE_DIR ${xtensor_INCLUDE_DIRS})

    find_package(xtensor-signal REQUIRED CONFIG)
    set(XTENSOR_SIGNAL_INCLUDE_DIR ${xtensor-signal_INCLUDE_DIRS})

    find_package(xtensor-io REQUIRED)
    set(XTENSOR_IO_INCLUDE_DIR ${xtensor_io_INCLUDE_DIRS})
endif ()

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

set (CMAKE_CXX_STANDARD 17)

find_package(ZLIB REQUIRED)
find_package(benchmark REQUIRED)
find_package(Threads)

find_package(xsimd)
if (xsimd_FOUND)
    include_directories(${xsimd_INCLUDE_DIRS})
    add_definitions("-DXTENSOR_USE_XSIMD=1")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} -march=tigerlake")
set(XTENSOR_SIGNAL_BENCHMARKS
    main.cpp
    lfilter_benchmark.cpp
)

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} /MP /bigobj")
endif()

add_executable(benchmark_xtensor_signal ${XTENSOR_SIGNAL_BENCHMARKS} ${XTENSOR_SIGNAL_HEADERS})
target_link_libraries(benchmark_xtensor_signal PRIVATE ZLIB::ZLIB xtensor-signal benchmark::benchmark ${CMAKE_THREAD_LIBS_INIT})
add_custom_target(xbenchmark COMMAND ./benchmark_xtensor_signal DEPENDS benchmark_xtensor_signal)

