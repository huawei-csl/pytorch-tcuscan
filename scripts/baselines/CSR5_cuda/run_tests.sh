#!/bin/bash
echo path,nrows,ncols,nnz,conversionTime,spmvTime,GFlops > csr5_record.csv

./spmv /scratch/TCUSCAN/sparse-suite-matrices/ssgetpy-downloaded-matrices/mip1/mip1.mtx
./spmv /scratch/TCUSCAN/sparse-suite-matrices/ssgetpy-downloaded-matrices/pdb1HYS/pdb1HYS.mtx
./spmv /scratch/TCUSCAN/sparse-suite-matrices/ssgetpy-downloaded-matrices/rma10/rma10.mtx
./spmv /scratch/TCUSCAN/sparse-suite-matrices/ssgetpy-downloaded-matrices/cant/cant.mtx
