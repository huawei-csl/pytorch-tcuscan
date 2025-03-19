#!/bin/bash
BASE=/scratch/gsorrentino/.ssgetpy/MM/
cd build || exit
make run FILENAME=${BASE}Williams/pdb1HYS/pdb1HYS.mtx
make run FILENAME=${BASE}Williams/mc2depi/mc2depi.mtx
make run FILENAME=${BASE}Williams/mac_econ_fwd500/mac_econ_fwd500.mtx
make run FILENAME=${BASE}Sandia/ASIC_680k/ASIC_680k.mtx
make run FILENAME=${BASE}Freescale/circuit5M/circuit5M.mtx
