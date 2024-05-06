cmake_minimum_required(VERSION 3.10)
project(MyProject CXX CUDA)


set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})


find_package(CUDA REQUIRED)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler")

file(GLOB cudafile "*.cu")

add_executable(reduce ${cudafile})