cmake_minimum_required(VERSION 3.10)
project(kml_spmv)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Werror")
find_package(OpenMP REQUIRED)

# Set SPBLAS option
option(IS_KML_MULTI_ENABLED "Is KML multi enabled." ON)

# Set KML path
set(KML_PATH /usr/local/kml)


# Default build type is debug
set(CMAKE_BUILD_TYPE "Debug")

# Set conditional compilation
if(IS_KML_MULTI_ENABLED)
    message(STATUS "SPBLAS version used: multi.")
    # Add link library
    link_directories(${KML_PATH}/lib/kspblas/multi)
    # Find link library
    find_library(KSPBLAS kspblas HINTS ${KML_PATH}/lib/sve/kspblas/multi)
else()
    message(STATUS "SPBLAS version used: single.")
    # Add link library
    link_directories(${KML_PATH}/lib/kspblas/single)
    # Find link library
    find_library(KSPBLAS kspblas HINTS ${KML_PATH}/lib/sve/kspblas/single)
endif()

# Set debug compilation option
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    add_compile_options(-g)
endif()



set(FMM_USE_FAST_FLOAT OFF)
set(FMM_USE_DRAGONBOX OFF)
set(FMM_USE_RYU OFF)

include(FetchContent)
FetchContent_Declare(
        fast_matrix_market
        GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market.git
        GIT_TAG main
        GIT_SHALLOW TRUE
        TLS_VERIFY FALSE
)
FetchContent_MakeAvailable(fast_matrix_market)


add_executable(kml_spmv_sparsesuite kml_spmv_sparsesuite.cpp)
target_link_libraries(kml_spmv_sparsesuite fast_matrix_market::fast_matrix_market ${KSPBLAS} OpenMP::OpenMP_CXX)
target_include_directories(kml_spmv_sparsesuite PRIVATE ${KML_PATH}/include)
