#!/bin/bash

BASE=/scratch/gsorrentino/.ssgetpy/MM

./build/spmv_sparsesuite ${BASE}/Williams/pdb1HYS/pdb1HYS.mtx
./build/spmv_sparsesuite ${BASE}/Williams/cant/cant.mtx
./build/spmv_sparsesuite ${BASE}/Bova/rma10/rma10.mtx
./build/spmv_sparsesuite ${BASE}/QCD/conf5_4-8x8-05/conf5_4-8x8-05.mtx
./build/spmv_sparsesuite ${BASE}/QCD/conf5_4-8x8-10/conf5_4-8x8-10.mtx
./build/spmv_sparsesuite ${BASE}/Andrianov/mip1/mip1.mtx
