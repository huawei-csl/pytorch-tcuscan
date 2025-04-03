#!/bin/bash

# You must download the sparse suite matrices first using `sparse-suite-downloader`.
BASE=/scratch/TCUSCAN/sparse-suite-matrices/ssgetpy-downloaded-matrices
#BASE=${HOME}/.ssgetpy/MM

# ID: 2373
./build/spmv_sparsesuite ${BASE}/pdb1HYS/pdb1HYS.mtx
# ID: 2375
./build/spmv_sparsesuite ${BASE}/cant/cant.mtx
# ID: 374
./build/spmv_sparsesuite ${BASE}/rma10/rma10.mtx
# # ID: 1598 (removed for now, because they are complex)
# ./build/spmv_sparsesuite ${BASE}/conf5_4-8x8-05/conf5_4-8x8-05.mtx
# # ID: 1599 (removed for now, because they are complex)
# ./build/spmv_sparsesuite ${BASE}/conf5_4-8x8-10/conf5_4-8x8-10.mtx
# ID: 1385
./build/spmv_sparsesuite ${BASE}/mip1/mip1.mtx
