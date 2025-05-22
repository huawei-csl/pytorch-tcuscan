#!/bin/bash

# You must download the sparse suite matrices first using `sparse-suite-downloader`.
BASE=/scratch/TCUSCAN/sparse-suite-matrices
#BASE=${HOME}/.ssgetpy/MM

# Sparse suite matrices from DASP paper.
./build/kml_spmv_sparsesuite ${BASE}/pwtk/pwtk.mtx
./build/kml_spmv_sparsesuite ${BASE}/FullChip/FullChip.mtx
./build/kml_spmv_sparsesuite ${BASE}/mip1/mip1.mtx
./build/kml_spmv_sparsesuite ${BASE}/mc2depi/mc2depi.mtx
./build/kml_spmv_sparsesuite ${BASE}/webbase-1M/webbase-1M.mtx
./build/kml_spmv_sparsesuite ${BASE}/circuit5M/circuit5M.mtx
./build/kml_spmv_sparsesuite ${BASE}/Si41Ge41H72/Si41Ge41H72.mtx
./build/kml_spmv_sparsesuite ${BASE}/Ga41As41H72/Ga41As41H72.mtx
./build/kml_spmv_sparsesuite ${BASE}/in-2004/in-2004.mtx
./build/kml_spmv_sparsesuite ${BASE}/eu-2005/eu-2005.mtx
./build/kml_spmv_sparsesuite ${BASE}/shipsec1/shipsec1.mtx
./build/kml_spmv_sparsesuite ${BASE}/mac_econ_fwd500/mac_econ_fwd500.mtx
./build/kml_spmv_sparsesuite ${BASE}/scircuit/scircuit.mtx
./build/kml_spmv_sparsesuite ${BASE}/pdb1HYS/pdb1HYS.mtx
./build/kml_spmv_sparsesuite ${BASE}/consph/consph.mtx
./build/kml_spmv_sparsesuite ${BASE}/cant/cant.mtx
./build/kml_spmv_sparsesuite ${BASE}/cop20k_A/cop20k_A.mtx
./build/kml_spmv_sparsesuite ${BASE}/dc2/dc2.mtx
./build/kml_spmv_sparsesuite ${BASE}/rma10/rma10.mtx
./build/kml_spmv_sparsesuite ${BASE}/ASIC_680k/ASIC_680k.mtx
