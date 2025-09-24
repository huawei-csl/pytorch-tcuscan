#!/bin/bash

# You must download the sparse suite matrices first using `sparse-suite-downloader`.
BASE="/scratch/asobczyk/datasets/suite-sparse-alenex26"
EXECUTABLE=run.py
CONDA_ENV=baseline-mkl

export MKL_NUM_THREADS=1
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/vsp_bcsstk30_500sep_10in_1Kout/vsp_bcsstk30_500sep_10in_1Kout.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/kron_g500-logn16/kron_g500-logn16.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/enron/enron.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/water_tank/water_tank.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/mip1/mip1.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/gupta2/gupta2.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/bcircuit/bcircuit.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/TSOPF_FS_b300_c2/TSOPF_FS_b300_c2.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/nasasrb/nasasrb.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/qa8fm/qa8fm.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/g7jac200/g7jac200.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/pct20stif/pct20stif.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/c-67b/c-67b.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/H2O/H2O.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/Ga3As3H12/Ga3As3H12.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/me2010/me2010.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/k1_san/k1_san.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/crankseg_2/crankseg_2.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/laminar_duct3D/laminar_duct3D.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/pdb1HYS/pdb1HYS.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/pkustk04/pkustk04.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/crankseg_1/crankseg_1.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/struct3/struct3.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/c-70/c-70.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/Chebyshev4/Chebyshev4.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/GaAsH6/GaAsH6.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/srb1/srb1.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/cant/cant.mtx

export MKL_NUM_THREADS=32
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/vsp_bcsstk30_500sep_10in_1Kout/vsp_bcsstk30_500sep_10in_1Kout.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/kron_g500-logn16/kron_g500-logn16.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/enron/enron.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/water_tank/water_tank.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/mip1/mip1.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/gupta2/gupta2.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/bcircuit/bcircuit.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/TSOPF_FS_b300_c2/TSOPF_FS_b300_c2.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/nasasrb/nasasrb.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/qa8fm/qa8fm.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/g7jac200/g7jac200.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/pct20stif/pct20stif.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/c-67b/c-67b.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/H2O/H2O.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/Ga3As3H12/Ga3As3H12.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/me2010/me2010.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/k1_san/k1_san.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/crankseg_2/crankseg_2.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/laminar_duct3D/laminar_duct3D.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/pdb1HYS/pdb1HYS.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/pkustk04/pkustk04.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/crankseg_1/crankseg_1.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/struct3/struct3.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/c-70/c-70.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/Chebyshev4/Chebyshev4.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/GaAsH6/GaAsH6.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/srb1/srb1.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE}"/cant/cant.mtx
