import csv
import os
import sys
import time

import numpy as np
from scipy.io import mmread
from sparse_dot_mkl import dot_product_mkl

if __name__ == "__main__":
    matrix_file = sys.argv[1]
    matrix_name = os.path.split(matrix_file)[1].split(".")[0]
    A = mmread(matrix_file).astype(np.float32).tocsr()
    n = A.shape[0]
    nnz = A.nnz
    b = np.random.rand(
        n,
    ).astype(np.float32)
    n_execs = 50
    num_threads = os.getenv("MKL_NUM_THREADS", "1")
    timings = []
    print(f"Matrix name: {matrix_name}")
    x_mkl = dot_product_mkl(A, b)

    for i in range(n_execs):
        # print(f"Running iteration {i} out or {n_execs}")
        t = time.time()
        x_mkl = dot_product_mkl(A, b)
        t_mkl = time.time() - t
        timings.append(t_mkl * 1000000)

    csv_filename = f"bench_results_mkl_spmv_{num_threads}T_{matrix_name}.csv"
    header = ["benchname", "size", "time_us"]
    with open(csv_filename, "w", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for timing in timings:
            row = [matrix_name, nnz, timing]
            writer.writerow(row)
