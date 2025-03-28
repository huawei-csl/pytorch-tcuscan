cmake_minimum_required(VERSION 3.10)
project(eigen_spmv)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

FIND_PACKAGE (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(spmv_sparsesuite sparsesuite.cpp)
target_link_libraries (spmv_sparsesuite Eigen3::Eigen)
