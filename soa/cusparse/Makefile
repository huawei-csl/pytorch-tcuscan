EXEC_NAME:=spmv_sparsesuite_app
SRC_NAME:=spmv_sparsesuite.cu
BASE_SPARSE_MATRIX_PATH:=${HOME}/.ssgetpy/MM/
LOCAL_SPARSE_MATRIX_NAME:='Williams/mc2depi/mc2depi.mtx'
FULL_SPARSE_MATRIX_PATH=${BASE_SPARSE_MATRIX_PATH}${LOCAL_SPARSE_MATRIX_NAME}

compile:
        nvcc ${SRC_NAME} -lcusparse_static -o ${EXEC_NAME}

run:
        ./${EXEC_NAME} ${FULL_SPARSE_MATRIX_PATH}

compile_and_run: compile run
