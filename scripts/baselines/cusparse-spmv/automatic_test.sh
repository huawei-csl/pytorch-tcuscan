#!/bin/bash
echo "Compiling the program..."
nvcc -O3 spmv_sparsesuite.cu -lcusparse_static -o spmv_sparsesuite_app
BASE_SPARSE_MATRIX_PATH="${HOME}/.ssgetpy/MM/"
find "$BASE_SPARSE_MATRIX_PATH" -type f -name "*.mtx" | while read -r mtx_file; do
    echo "Running test for: $mtx_file"
    ./spmv_sparsesuite_app "$mtx_file"
done
