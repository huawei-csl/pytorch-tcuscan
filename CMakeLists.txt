cmake_minimum_required(VERSION 3.16.0)
project(tcuscan)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")  # "Debug"


# user-defined configuration
set(SOC_VERSION "Ascend910B4" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "ASCEND CANN package installation directory")
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the CANN package is installed.")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# ascendc_library use to add kernel file to generate AscendC library
ascendc_library(kernels SHARED
    src/vadd.cpp
    src/diff.cpp
    src/seg_scan_single_core.cpp
    src/scan_multi_core.cpp
    src/csr_gather.cpp
    src/compress.cpp
    src/copy.cpp
    src/scan_single_core.cpp
    src/seg_scan_vec_single_core.cpp
)

ascendc_compile_definitions(kernels PRIVATE
-DHAVE_WORKSPACE
-DHAVE_TILING
)

ascendc_include_directories(kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(kernels PRIVATE
    ascendcl
    c_sec
    tiling_api
    register
    platform
    ascendalog
)


add_library(pybind11_lib SHARED src/pybind11.cpp)
target_link_libraries(pybind11_lib PRIVATE
  c10
  torch
  torch_cpu
  torch_npu
  torch_python
  tiling_api
  kernels
)

# Sets the PyTorch installation path on env variable 'TORCH_PATH'
execute_process(COMMAND python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_PATH
)


# Set the Ascend PyTorch (torch_npu) installation path on env variable 'TORCH_NPU_PATH'
set(ENV{ASCEND_HOME_PATH} ${ASCEND_CANN_PACKAGE_PATH})
execute_process(COMMAND python3 -c "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_NPU_PATH
)


# Set the pybind11 include dir on env variable 'PYBIND11_INC'
execute_process(COMMAND python3 -m pybind11 --includes
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYBIND11_INC
)
string(REPLACE " " ";" PYBIND11_INC ${PYBIND11_INC})

message("***********************************************************")
message("* ASCEND_CANN_PACKAGE_PATH : ${ASCEND_CANN_PACKAGE_PATH}")
message("* TORCH_PATH               : ${TORCH_PATH}")
message("* TORCH_NPU_PATH           : ${TORCH_NPU_PATH}")
message("* PYBIND11_INC             : ${PYBIND11_INC}")
message("***********************************************************")

target_link_directories(pybind11_lib PRIVATE
  ${TORCH_PATH}/lib
  ${TORCH_NPU_PATH}/lib
)
target_include_directories(pybind11_lib PRIVATE
  ${TORCH_NPU_PATH}/include
  ${TORCH_PATH}/include
  ${TORCH_PATH}/include/torch/csrc/api/include
)

target_compile_options(pybind11_lib PRIVATE
  ${PYBIND11_INC}
  -D_GLIBCXX_USE_CXX11_ABI=0
)

execute_process(COMMAND python3-config --extension-suffix
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYBIND11_SUFFIX
)
set_target_properties(pybind11_lib PROPERTIES
  OUTPUT_NAME tcuscan_ops${PYBIND11_SUFFIX}
  PREFIX "" SUFFIX ""
)

set_target_properties(pybind11_lib PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
