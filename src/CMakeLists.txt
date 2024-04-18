cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})


find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(gemm main.cpp)

cuda_add_library(my_cuda_lib test.cu)
target_link_libraries(my_cuda_lib ${CUDA_LIBRARIES})
target_link_libraries(my_cuda_lib cublas)

target_link_libraries(gemm my_cuda_lib)