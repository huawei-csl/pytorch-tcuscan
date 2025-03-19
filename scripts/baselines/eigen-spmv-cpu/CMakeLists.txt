cmake_minimum_required(VERSION 3.10)
project(eigen_spmv)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

if(NOT DEFINED EIGEN_DIR)
  message(FATAL_ERROR "Specify EIGEN library path using -DEIGEN_DIR=<path_to_eigen>")
endif()

include_directories(${EIGEN_DIR})
if(NOT DEFINED SRC)
    message("A source should be defined with -DSRC=<name>. Using by default example")
    set(SRC example)
endif()
add_executable(spmv_${SRC} ${SRC}.cpp)

if(NOT DEFINED FILENAME)
  message("NEED INPUT MTX FILE PATH WITH -DFILENAME=<mtxpath>. SETTING ONE AS DEFAULT:{HOME}/.ssgetpy/MM/Williams/pdb1HYS/pdb1HYS.mtx ")
  set(FILENAME "$ENV{HOME}/.ssgetpy/MM/Williams/pdb1HYS/pdb1HYS.mtx")

endif()


add_custom_target(run
  COMMAND spmv_${SRC} ${FILENAME}
  DEPENDS spmv_${SRC}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
