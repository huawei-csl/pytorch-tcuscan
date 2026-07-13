# --------------------------------------------------------------------------------
# Copyright (c) 2023-2026 Huawei Technologies Co., Ltd. All rights reserved. See
# LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pytorch-tcuscan/ for the full License text.
# --------------------------------------------------------------------------------#
# Builds scan_kernels, vector_kernels, split_kernels and the tcuscan_ops python
# extension with either kernel toolchain:
#
# -DUSE_ASC_LANGUAGE=ON (default)  ASC language (find_package(ASC)), as in
# cann/ops-sparse -DUSE_ASC_LANGUAGE=OFF           legacy ascendc.cmake
#
# The host wrappers in src/torch/*.h launch kernels with `<<<>>>`, so they no
# longer depend on the code-generated aclrtlaunch_<kernel>.h headers that only
# the legacy path emits. Both toolchains now build every target, and the legacy
# path can be dropped once it is no longer needed for comparison.

cmake_minimum_required(VERSION 3.16.0)

project(tcuscan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PIP_INSTALL "True if building via pip" OFF)
option(USE_ASC_LANGUAGE
       "Compile kernels with the ASC language toolchain (CANN >= 9.0)" ON)

# ========= user-defined configuration =========
set(SOC_VERSION
    "Ascend910B4"
    CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH
    "$ENV{ASCEND_HOME_PATH}"
    CACHE PATH "ASCEND CANN package installation directory")
set(RUN_MODE
    "npu"
    CACHE STRING "run mode: npu")
set(CMAKE_INSTALL_PREFIX
    "${CMAKE_CURRENT_LIST_DIR}/out"
    CACHE STRING "path for install()" FORCE)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Debug"
      CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()

if(NOT ASCEND_CANN_PACKAGE_PATH)
  set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

# ========= SOC_VERSION -> NPU_ARCH (ASC language only) =========
string(TOLOWER "${SOC_VERSION}" SOC_VERSION_LOWER)
if(SOC_VERSION_LOWER MATCHES "^ascend910b" OR SOC_VERSION_LOWER MATCHES
                                              "^ascend910_93")
  set(NPU_ARCH "dav-2201")
elseif(SOC_VERSION_LOWER MATCHES "^ascend950")
  set(NPU_ARCH "dav-3510")
elseif(SOC_VERSION_LOWER MATCHES "^ascend310p")
  set(NPU_ARCH "dav-2002")
else()
  message(
    FATAL_ERROR
      "Unsupported SOC_VERSION: ${SOC_VERSION}. Supported: ascend910b*, ascend910_93*, ascend950*, ascend310p*"
  )
endif()

# ========= kernel toolchain =========
if(USE_ASC_LANGUAGE)
  # find_package(ASC) provides the ASC language plus ASC-flavoured
  # ascendc_library / ascendc_compile_definitions / ascendc_include_directories.
  # enable_language(ASC) only resolves once find_package has put the ASC plugin
  # on CMAKE_MODULE_PATH.
  if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(
      FATAL_ERROR
        "USE_ASC_LANGUAGE requires ASCEND_HOME_PATH, source <cann>/set_env.sh first."
    )
  endif()
  find_package(ASC REQUIRED)
  enable_language(ASC)

  # CMakeASCInformation.cmake derives CMAKE_ASC_CREATE_SHARED_MODULE from
  # CMAKE_ASC_CREATE_SHARED_LIBRARY before that one is defined, so it ends up
  # empty and MODULE targets (the pybind11 extension) fail to generate.
  if(NOT CMAKE_ASC_CREATE_SHARED_MODULE)
    set(CMAKE_ASC_CREATE_SHARED_MODULE "${CMAKE_ASC_CREATE_SHARED_LIBRARY}")
  endif()
else()
  if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR
        ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
  elseif(EXISTS
         ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR
        ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
  elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR
        ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
  else()
    message(
      FATAL_ERROR
        "ascendc_kernel_cmake does not exist, please check if CANN is installed."
    )
  endif()
  include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)
endif()

# ========= host dependencies =========
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)

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
message("* USE_ASC_LANGUAGE         : ${USE_ASC_LANGUAGE}")
message("* SOC_VERSION / NPU_ARCH   : ${SOC_VERSION} / ${NPU_ARCH}")
message("* TORCH_INSTALL_PREFIX     : ${TORCH_INSTALL_PREFIX}")
message("* TORCH_NPU_PATH           : ${TORCH_NPU_PATH}")
message("* TORCH_LIBRARIES          : ${TORCH_LIBRARIES}")
message("***********************************************************")

# ========= kernel libraries =========
# Settings shared by every kernel library, whichever toolchain compiled it.
function(tcuscan_kernel_library_setup target_name)
  # tcuscan_ops is linked with RUNPATH $ORIGIN/build/lib, so the kernel
  # libraries have to land there whichever toolchain built them.
  set_target_properties(${target_name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                  ${CMAKE_BINARY_DIR}/lib)
  ascendc_compile_definitions(${target_name} PRIVATE -DHAVE_WORKSPACE
                              -DHAVE_TILING)
  ascendc_include_directories(${target_name} PRIVATE
                              ${CMAKE_CURRENT_SOURCE_DIR}/src)
  if(USE_ASC_LANGUAGE)
    # The ASC compiler takes the target architecture instead of SOC_VERSION.
    target_compile_options(
      ${target_name} PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>)
    target_link_libraries(${target_name} PRIVATE tiling_api platform ascendalog)
  else()
    target_link_libraries(${target_name} PRIVATE ascendcl c_sec tiling_api
                                                 register platform ascendalog)
  endif()
endfunction()

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
  src/seg_sum_multi_cube.cpp
  src/seg_sum_single_cube.cpp
  src/row_scan.cpp
  src/block_scan.cpp
  src/triu_inv_rec_unroll.cpp
  src/scan_multi_cube.cpp
  src/simple_pad.cpp
  src/cube_reduce.cpp
  src/spmv_v2.cpp
  src/spmv_v2_multi_cube.cpp
  src/tri_inv_cube_col_sweep.cpp)

tcuscan_kernel_library_setup(scan_kernels)

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
  src/histogram.cpp
  src/searchsorted.cpp)

tcuscan_kernel_library_setup(vector_kernels)

ascendc_library(split_kernels SHARED src/radix_sort.cpp src/compress.cpp
                src/split.cpp src/topk.cpp)

tcuscan_kernel_library_setup(split_kernels)

# ========= python extension =========
# torch_scan_cpu.cpp stays a plain C++ source: it is OpenMP host code and the
# ASC compiler ships no OpenMP runtime.
find_package(OpenMP REQUIRED COMPONENTS CXX)

if(USE_ASC_LANGUAGE)
  # The host wrappers in src/torch/*.h launch kernels with `<<<>>>`, which only
  # the ASC compiler understands, so the translation unit that includes them has
  # to be compiled as ASC rather than plain C++.
  set_source_files_properties(src/pybind11.cpp PROPERTIES LANGUAGE ASC)
endif()

pybind11_add_module(tcuscan_ops src/pybind11.cpp src/torch/torch_scan_cpu.cpp)

if(USE_ASC_LANGUAGE)
  target_compile_options(
    tcuscan_ops PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}
                        -std=c++17>)
  # Mixed ASC/CXX sources: link with the ASC driver so the kernel launch stubs
  # in pybind11.cpp resolve.
  set_target_properties(tcuscan_ops PROPERTIES LINKER_LANGUAGE ASC)
endif()

# Link the OpenMP runtime explicitly rather than through OpenMP::OpenMP_CXX,
# whose -fopenmp link flag the ASC driver does not accept.
target_link_libraries(tcuscan_ops PRIVATE ${OpenMP_CXX_LIBRARIES})

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
