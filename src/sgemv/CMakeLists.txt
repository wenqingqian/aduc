cmake_minimum_required(VERSION 3.10)
project(MyProject CXX CUDA)


set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})


find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -fopenmp")

file(GLOB cudafile "*.cu")

add_executable(sgemv ${cudafile})

target_link_libraries(sgemv OpenMP::OpenMP_CXX)