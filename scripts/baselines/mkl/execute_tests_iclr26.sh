#!/bin/bash

# You must download the sparse suite matrices first using `sparse-suite-downloader`.
BASE_SS="/scratch/asobczyk/datasets/suite-sparse-alenex26"
BASE_LLM="/scratch/asobczyk/datasets/big-bird-artificial-matrices"
EXECUTABLE=run.py
CONDA_ENV=baseline-mkl

export MKL_NUM_THREADS=1
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/vsp_bcsstk30_500sep_10in_1Kout/vsp_bcsstk30_500sep_10in_1Kout.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/kron_g500-logn16/kron_g500-logn16.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/enron/enron.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/water_tank/water_tank.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/mip1/mip1.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/gupta2/gupta2.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/bcircuit/bcircuit.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/TSOPF_FS_b300_c2/TSOPF_FS_b300_c2.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/nasasrb/nasasrb.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/qa8fm/qa8fm.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/g7jac200/g7jac200.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/pct20stif/pct20stif.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/c-67b/c-67b.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/H2O/H2O.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/Ga3As3H12/Ga3As3H12.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/me2010/me2010.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/k1_san/k1_san.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/crankseg_2/crankseg_2.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/laminar_duct3D/laminar_duct3D.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/pdb1HYS/pdb1HYS.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/pkustk04/pkustk04.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/crankseg_1/crankseg_1.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/struct3/struct3.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/c-70/c-70.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/Chebyshev4/Chebyshev4.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/GaAsH6/GaAsH6.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/srb1/srb1.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/cant/cant.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-2_r-4_w-1_g-2_n-65536/b-2_r-4_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-4_r-2_w-1_g-2_n-65536/b-4_r-2_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-16_r-4_w-1_g-2_n-65536/b-16_r-4_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-16_r-2_w-1_g-2_n-65536/b-16_r-2_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-2_r-2_w-1_g-2_n-65536/b-2_r-2_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-64_r-4_w-1_g-2_n-65536/b-64_r-4_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-4_r-4_w-1_g-2_n-65536/b-4_r-4_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-4_r-8_w-1_g-2_n-65536/b-4_r-8_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-16_r-8_w-1_g-2_n-65536/b-16_r-8_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-4_r-6_w-1_g-2_n-65536/b-4_r-6_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-16_r-6_w-1_g-2_n-65536/b-16_r-6_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-64_r-2_w-1_g-2_n-65536/b-64_r-2_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-2_r-8_w-1_g-2_n-65536/b-2_r-8_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-64_r-8_w-1_g-2_n-65536/b-64_r-8_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-64_r-6_w-1_g-2_n-65536/b-64_r-6_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-2_r-6_w-1_g-2_n-65536/b-2_r-6_w-1_g-2_n-65536.mtx

export MKL_NUM_THREADS=32
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/vsp_bcsstk30_500sep_10in_1Kout/vsp_bcsstk30_500sep_10in_1Kout.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/kron_g500-logn16/kron_g500-logn16.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/enron/enron.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/water_tank/water_tank.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/mip1/mip1.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/gupta2/gupta2.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/bcircuit/bcircuit.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/TSOPF_FS_b300_c2/TSOPF_FS_b300_c2.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/nasasrb/nasasrb.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/qa8fm/qa8fm.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/g7jac200/g7jac200.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/pct20stif/pct20stif.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/c-67b/c-67b.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/H2O/H2O.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/Ga3As3H12/Ga3As3H12.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/me2010/me2010.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/k1_san/k1_san.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/crankseg_2/crankseg_2.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/laminar_duct3D/laminar_duct3D.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/pdb1HYS/pdb1HYS.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/pkustk04/pkustk04.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/crankseg_1/crankseg_1.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/struct3/struct3.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/c-70/c-70.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/Chebyshev4/Chebyshev4.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/GaAsH6/GaAsH6.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/srb1/srb1.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_SS}"/cant/cant.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-2_r-4_w-1_g-2_n-65536/b-2_r-4_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-4_r-2_w-1_g-2_n-65536/b-4_r-2_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-16_r-4_w-1_g-2_n-65536/b-16_r-4_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-16_r-2_w-1_g-2_n-65536/b-16_r-2_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-2_r-2_w-1_g-2_n-65536/b-2_r-2_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-64_r-4_w-1_g-2_n-65536/b-64_r-4_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-4_r-4_w-1_g-2_n-65536/b-4_r-4_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-4_r-8_w-1_g-2_n-65536/b-4_r-8_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-16_r-8_w-1_g-2_n-65536/b-16_r-8_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-4_r-6_w-1_g-2_n-65536/b-4_r-6_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-16_r-6_w-1_g-2_n-65536/b-16_r-6_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-64_r-2_w-1_g-2_n-65536/b-64_r-2_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-2_r-8_w-1_g-2_n-65536/b-2_r-8_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-64_r-8_w-1_g-2_n-65536/b-64_r-8_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-64_r-6_w-1_g-2_n-65536/b-64_r-6_w-1_g-2_n-65536.mtx
conda run -n ${CONDA_ENV} python3 ${EXECUTABLE} "${BASE_LLM}"/b-2_r-6_w-1_g-2_n-65536/b-2_r-6_w-1_g-2_n-65536.mtx
