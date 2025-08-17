cmake_minimum_required(VERSION 3.10)
project(eigen_spmv)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3 -Wall")
find_package(OpenMP REQUIRED)

find_package(OpenMP REQUIRED)
FIND_PACKAGE (Eigen3 3.3 REQUIRED NO_MODULE)

set(FMM_USE_FAST_FLOAT OFF)
set(FMM_USE_DRAGONBOX OFF)
set(FMM_USE_RYU OFF)

include(FetchContent)
FetchContent_Declare(
        fast_matrix_market
        GIT_REPOSITORY https://rnd-gitlab-eu.huawei.com/a00565472/fast_matrix_market.git
        GIT_TAG main
        GIT_SHALLOW TRUE
        TLS_VERIFY FALSE
)
FetchContent_MakeAvailable(fast_matrix_market)


add_executable(spmv_sparsesuite sparsesuite.cpp)
target_link_libraries (spmv_sparsesuite Eigen3::Eigen)
target_link_libraries(spmv_sparsesuite fast_matrix_market::fast_matrix_market)
target_link_libraries(spmv_sparsesuite pthread)
