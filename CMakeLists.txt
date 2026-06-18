cmake_minimum_required(VERSION 3.16.0)
project(tcuscan)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release") # "Debug"

option(PIP_INSTALL "True if building via pip" OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)

# user-defined configuration
set(SOC_VERSION
    "Ascend910B4"
    CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH
    "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE PATH "ASCEND CANN package installation directory")
set(RUN_MODE
    "npu"
    CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE
    "Debug"
    CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(CMAKE_INSTALL_PREFIX
    "${CMAKE_CURRENT_LIST_DIR}/out"
    CACHE STRING "path for install()" FORCE)

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
  set(ASCENDC_CMAKE_DIR
      ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
  set(ASCENDC_CMAKE_DIR
      ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
  set(ASCENDC_CMAKE_DIR
      ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
  message(
    FATAL_ERROR
      "ascendc_kernel_cmake does not exist, please check if CANN is installed.")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

include(FetchContent)

# certain operations need newer pto-isa header, not CANN default
# to use default CANN headers set GIT_TAG to the matching CANN version
FetchContent_Declare(
  libpto_isa_headers
  GIT_REPOSITORY https://gitcode.com/cann/pto-isa.git
  GIT_TAG v9.0.0)

FetchContent_Populate(libpto_isa_headers)

# TORCH_NPU_PATH is the location where PyTorch Ascend Adapter (torch_npu) is
# installed.
execute_process(
  COMMAND
    python3 -c
    "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_NPU_PATH)

string(REPLACE "\n" "" TORCH_NPU_PATH "${TORCH_NPU_PATH}")
string(REPLACE "path string is NULL" "" TORCH_NPU_PATH "${TORCH_NPU_PATH}")

message("***********************************************************")
message("* ASCEND_CANN_PACKAGE_PATH : ${ASCEND_CANN_PACKAGE_PATH}")
message("* TORCH_INSTALL_PREFIX     : ${TORCH_INSTALL_PREFIX}")
message("* TORCH_NPU_PATH           : ${TORCH_NPU_PATH}")
message("* TORCH_LIBRARIES          : ${TORCH_LIBRARIES}")
message("***********************************************************")

# ascendc_library use to add kernel file to generate AscendC library
ascendc_library(
  scan_kernels
  SHARED
  src/seg_scan_single_core.cpp
  src/scan_multi_core.cpp
  src/scan_batch.cpp
  src/matmul_cce.cpp
  src/scan_single_core.cpp
  src/seg_scan_vec_single_core.cpp
  src/seg_scan_mc_revert.cpp
  src/seg_sum_single_core.cpp
  src/seg_sum_multi_core.cpp
  src/seg_sum_single_cube.cpp
  src/row_scan.cpp
  src/block_scan.cpp
  src/triu_inv_rec_unroll.cpp
  src/scan_multi_cube.cpp
  src/simple_pad.cpp
  src/cube_reduce.cpp
  src/tri_inv_cube_col_sweep.cpp)

#ascendc_compile_options(scan_kernels PRIVATE -Wall -Wextra -Werror)

ascendc_compile_definitions(scan_kernels PRIVATE -DHAVE_WORKSPACE -DHAVE_TILING)

ascendc_include_directories(
  scan_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${libpto_isa_headers_SOURCE_DIR}/include
  ${libpto_isa_headers_SOURCE_DIR}/include/pto/common)

target_link_libraries(scan_kernels PRIVATE ascendcl c_sec tiling_api register
                                           platform ascendalog)

ascendc_library(
  vector_kernels
  SHARED
  src/vadd.cpp
  src/diff.cpp
  src/csr_gather.cpp
  src/copy.cpp
  src/gather_spmv.cpp
  src/mc_gather.cpp
  src/gen_lower.cpp
  src/reduce_tiles.cpp
  src/complete_rows.cpp
  src/complete_blocks.cpp
  src/tri_inv_col_sweep.cpp
  src/compare.cpp
  src/histogram.cpp)

#ascendc_compile_options(vector_kernels PRIVATE -Wall -Wextra -Werror)

ascendc_compile_definitions(vector_kernels PRIVATE -DHAVE_WORKSPACE
                            -DHAVE_TILING)

ascendc_include_directories(
  vector_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${libpto_isa_headers_SOURCE_DIR}/include
  ${libpto_isa_headers_SOURCE_DIR}/include/pto/common)

target_link_libraries(vector_kernels PRIVATE ascendcl c_sec tiling_api register
                                             platform ascendalog)

ascendc_library(split_kernels SHARED src/radix_sort.cpp src/compress.cpp
                src/split.cpp src/topk.cpp)

#ascendc_compile_options(split_kernels PRIVATE -Wall -Wextra -Werror)

ascendc_compile_definitions(split_kernels PRIVATE -DHAVE_WORKSPACE
                            -DHAVE_TILING)

ascendc_include_directories(
  split_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${libpto_isa_headers_SOURCE_DIR}/include
  ${libpto_isa_headers_SOURCE_DIR}/include/pto/common)

target_link_libraries(split_kernels PRIVATE ascendcl c_sec tiling_api register
                                            platform ascendalog)

pybind11_add_module(tcuscan_ops src/pybind11.cpp)

target_link_libraries(
  tcuscan_ops
  PUBLIC ${TORCH_LIBRARIES}
         c10
         torch_cpu
         torch_npu
         torch_python
         tiling_api
         scan_kernels
         vector_kernels
         split_kernels)

target_link_directories(tcuscan_ops PRIVATE ${TORCH_INSTALL_PREFIX}/lib
                        ${TORCH_NPU_PATH}/lib)
target_include_directories(tcuscan_ops PRIVATE ${TORCH_NPU_PATH}/include
                                               ${TORCH_INCLUDE_DIRS})

if(PIP_INSTALL)
  message(STATUS ">>> Running under scikit-build-core (pip install)")

  target_link_options(tcuscan_ops PRIVATE "LINKER:-no-as-needed")

  # install tcuscan_ops.so under site-packages/tcuscan
  install(TARGETS tcuscan_ops LIBRARY DESTINATION tcuscan)

  # Set RPATH to $ORIGIN/lib (See Python package layout)
  set_target_properties(
    tcuscan_ops
    PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
               INSTALL_RPATH_USE_LINK_PATH TRUE
               LINK_FLAGS "-Wl,-rpath,\${ORIGIN}/lib")

  # install dynamic libraries under site-packages/tcuscan/libs
  install(TARGETS scan_kernels vector_kernels split_kernels
          LIBRARY DESTINATION tcuscan/lib)
else()
  message(STATUS ">>>======================================================")
  message(STATUS ">>> Ignoring dynamic libraries COPY inside Python wheel.")
  message(STATUS ">>>======================================================")

  # Set RPATH to $(pwd)/build/lib
  set_target_properties(
    tcuscan_ops
    PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
               INSTALL_RPATH_USE_LINK_PATH TRUE
               LINK_FLAGS "-Wl,-rpath,\${ORIGIN}/build/lib"
               LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

endif()
