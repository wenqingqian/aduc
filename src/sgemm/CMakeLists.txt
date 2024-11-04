cmake_minimum_required(VERSION 3.10)
project(MyProject CXX CUDA)


set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})


find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -fopenmp")

file(GLOB cudafile "*.cu")

add_executable(gemm ${cudafile})

target_link_libraries(gemm OpenMP::OpenMP_CXX ${CUDA_cublas_LIBRARY})

if(STORE_RESULT)
	target_compile_definitions(gemm PRIVATE -DSTORE_RESULT)
endif()
if(DISABLE_CPU)
	target_compile_definitions(gemm PRIVATE -DDISABLE_CPU)
endif()

# cuda_add_executable 文件名不能有 : 号