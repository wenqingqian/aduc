cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CUDA_STANDARD 17)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})


find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -fopenmp -w -Xcudafe --display_error_number --diag_suppress")

file(GLOB cudafile "*.cu")

cuda_add_executable(gemm ${cudafile})

target_link_libraries(gemm cublas OpenMP::OpenMP_CXX ${CUDA_LIBRARIES})


# cuda_add_executable 文件名不能有 : 号