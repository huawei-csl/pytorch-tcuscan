#!/bin/bash
export OMP_NUM_THREADS=32
# You must download the sparse suite matrices first using `sparse-suite-downloader`.
BASE=/scratch/TCUSCAN/sparse-suite-matrices/ssgetpy-downloaded-matrices
EXECUTABLE=run.py
CONDA_ENV=mkl_env

# ID: 2373
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} ${BASE}/pdb1HYS/pdb1HYS.mtx
# ID: 2375s
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} ${BASE}/cant/cant.mtx
# ID: 374
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} ${BASE}/rma10/rma10.mtx
# ID: 1385
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} ${BASE}/mip1/mip1.mtx
