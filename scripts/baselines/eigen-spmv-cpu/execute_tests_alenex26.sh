#!/bin/bash

# You must download the sparse suite matrices first using `sparse-suite-downloader`.
BASE="/scratch/TCUSCAN/sparse-suite-matrices/ssgetpy-downloaded-matrices"
export OMP_PLACES=cores

export OMP_NUM_THREADS=1
./build/spmv_sparsesuite "${BASE}"/vsp_bcsstk30_500sep_10in_1Kout/vsp_bcsstk30_500sep_10in_1Kout.mtx
./build/spmv_sparsesuite "${BASE}"/kron_g500-logn16/kron_g500-logn16.mtx
./build/spmv_sparsesuite "${BASE}"/enron/enron.mtx
./build/spmv_sparsesuite "${BASE}"/water_tank/water_tank.mtx
./build/spmv_sparsesuite "${BASE}"/mip1/mip1.mtx
./build/spmv_sparsesuite "${BASE}"/gupta2/gupta2.mtx
./build/spmv_sparsesuite "${BASE}"/bcircuit/bcircuit.mtx
./build/spmv_sparsesuite "${BASE}"/TSOPF_FS_b300_c2/TSOPF_FS_b300_c2.mtx
./build/spmv_sparsesuite "${BASE}"/nasasrb/nasasrb.mtx
./build/spmv_sparsesuite "${BASE}"/qa8fm/qa8fm.mtx
./build/spmv_sparsesuite "${BASE}"/g7jac200/g7jac200.mtx
./build/spmv_sparsesuite "${BASE}"/pct20stif/pct20stif.mtx
./build/spmv_sparsesuite "${BASE}"/c-67b/c-67b.mtx
./build/spmv_sparsesuite "${BASE}"/H2O/H2O.mtx
./build/spmv_sparsesuite "${BASE}"/Ga3As3H12/Ga3As3H12.mtx
./build/spmv_sparsesuite "${BASE}"/me2010/me2010.mtx
./build/spmv_sparsesuite "${BASE}"/k1_san/k1_san.mtx
./build/spmv_sparsesuite "${BASE}"/crankseg_2/crankseg_2.mtx
./build/spmv_sparsesuite "${BASE}"/laminar_duct3D/laminar_duct3D.mtx
./build/spmv_sparsesuite "${BASE}"/pdb1HYS/pdb1HYS.mtx
./build/spmv_sparsesuite "${BASE}"/pkustk04/pkustk04.mtx
./build/spmv_sparsesuite "${BASE}"/crankseg_1/crankseg_1.mtx
./build/spmv_sparsesuite "${BASE}"/struct3/struct3.mtx
./build/spmv_sparsesuite "${BASE}"/c-70/c-70.mtx
./build/spmv_sparsesuite "${BASE}"/Chebyshev4/Chebyshev4.mtx
./build/spmv_sparsesuite "${BASE}"/GaAsH6/GaAsH6.mtx
./build/spmv_sparsesuite "${BASE}"/srb1/srb1.mtx
./build/spmv_sparsesuite "${BASE}"/cant/cant.mtx

export OMP_NUM_THREADS=32
./build/spmv_sparsesuite "${BASE}"/vsp_bcsstk30_500sep_10in_1Kout/vsp_bcsstk30_500sep_10in_1Kout.mtx
./build/spmv_sparsesuite "${BASE}"/kron_g500-logn16/kron_g500-logn16.mtx
./build/spmv_sparsesuite "${BASE}"/enron/enron.mtx
./build/spmv_sparsesuite "${BASE}"/water_tank/water_tank.mtx
./build/spmv_sparsesuite "${BASE}"/mip1/mip1.mtx
./build/spmv_sparsesuite "${BASE}"/gupta2/gupta2.mtx
./build/spmv_sparsesuite "${BASE}"/bcircuit/bcircuit.mtx
./build/spmv_sparsesuite "${BASE}"/TSOPF_FS_b300_c2/TSOPF_FS_b300_c2.mtx
./build/spmv_sparsesuite "${BASE}"/nasasrb/nasasrb.mtx
./build/spmv_sparsesuite "${BASE}"/qa8fm/qa8fm.mtx
./build/spmv_sparsesuite "${BASE}"/g7jac200/g7jac200.mtx
./build/spmv_sparsesuite "${BASE}"/pct20stif/pct20stif.mtx
./build/spmv_sparsesuite "${BASE}"/c-67b/c-67b.mtx
./build/spmv_sparsesuite "${BASE}"/H2O/H2O.mtx
./build/spmv_sparsesuite "${BASE}"/Ga3As3H12/Ga3As3H12.mtx
./build/spmv_sparsesuite "${BASE}"/me2010/me2010.mtx
./build/spmv_sparsesuite "${BASE}"/k1_san/k1_san.mtx
./build/spmv_sparsesuite "${BASE}"/crankseg_2/crankseg_2.mtx
./build/spmv_sparsesuite "${BASE}"/laminar_duct3D/laminar_duct3D.mtx
./build/spmv_sparsesuite "${BASE}"/pdb1HYS/pdb1HYS.mtx
./build/spmv_sparsesuite "${BASE}"/pkustk04/pkustk04.mtx
./build/spmv_sparsesuite "${BASE}"/crankseg_1/crankseg_1.mtx
./build/spmv_sparsesuite "${BASE}"/struct3/struct3.mtx
./build/spmv_sparsesuite "${BASE}"/c-70/c-70.mtx
./build/spmv_sparsesuite "${BASE}"/Chebyshev4/Chebyshev4.mtx
./build/spmv_sparsesuite "${BASE}"/GaAsH6/GaAsH6.mtx
./build/spmv_sparsesuite "${BASE}"/srb1/srb1.mtx
./build/spmv_sparsesuite "${BASE}"/cant/cant.mtx
