#!/bin/bash

# You must download the sparse suite matrices first using `sparse-suite-downloader`.
BASE=/scratch/TCUSCAN/sparse-suite-matrices/ssgetpy-downloaded-matrices
#BASE=${HOME}/.ssgetpy/MM

# ID: 2373
./build/spmv_sparsesuite ${BASE}/Williams/pdb1HYS/pdb1HYS.mtx
# ID: 2375
./build/spmv_sparsesuite ${BASE}/Williams/cant/cant.mtx
# ID: 374
./build/spmv_sparsesuite ${BASE}/Bova/rma10/rma10.mtx
# ID: 1598
./build/spmv_sparsesuite ${BASE}/QCD/conf5_4-8x8-05/conf5_4-8x8-05.mtx
# ID: 1599
./build/spmv_sparsesuite ${BASE}/QCD/conf5_4-8x8-10/conf5_4-8x8-10.mtx
# ID: 1385
./build/spmv_sparsesuite ${BASE}/Andrianov/mip1/mip1.mtx
