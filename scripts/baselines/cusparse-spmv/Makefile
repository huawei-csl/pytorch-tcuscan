.phony: clean run run_csr_gather

BASE_SPARSE_MATRIX_PATH:=${HOME}/.ssgetpy/MM/
LOCAL_SPARSE_MATRIX_NAME:='Williams/mc2depi/mc2depi.mtx'
FULL_SPARSE_MATRIX_PATH=${BASE_SPARSE_MATRIX_PATH}${LOCAL_SPARSE_MATRIX_NAME}


clean:
        rm -rf spmv_sparsesuite_app spmv_csr_gather

spmv_sparsesuite_app:
        nvcc -O3 spmv_sparsesuite.cu -lcusparse_static -o spmv_sparsesuite_app

run: spmv_sparsesuite_app
        ./spmv_sparsesuite_app ${FULL_SPARSE_MATRIX_PATH}

spmv_csr_gather:
        nvcc -O3 spmv_csr_gather.cu -o spmv_csr_gather

run_csr_gather: spmv_csr_gather
        ./spmv_csr_gather ${FULL_SPARSE_MATRIX_PATH}
