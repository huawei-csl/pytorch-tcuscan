import csv
import os
import sys
import time
from pathlib import Path

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
    n_execs = 1
    num_threads = os.getenv("MKL_NUM_THREADS", "1")
    timings = []
    print(f"Matrix name: {matrix_name}")
    x_mkl = dot_product_mkl(A, b)

    for i in range(n_execs):
        # print(f"Running iteration {i} out or {n_execs}")
        tic = time.perf_counter_ns()
        x_mkl = dot_product_mkl(A, b)
        toc = time.perf_counter_ns()
        timings.append((toc - tic) * 0.001)  # nanoseconds to microseconds

    csv_filename = f"bench_results_mkl_spmv_{num_threads}T.csv"
    header = ["benchname", "size", "mkl_num_threads", "time_us"]

    if not Path(csv_filename).is_file():
        with open(csv_filename, "w", encoding="UTF-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    with open(csv_filename, "a", encoding="UTF-8") as f:
        writer = csv.writer(f)
        for timing in timings:
            row = [matrix_name, nnz, num_threads, timing]
            writer.writerow(row)
