import argparse
import time

import numpy as np
import torch
from scipy.io import mmread
from scipy.sparse import csr_matrix, random
from torch_scatter import segment_sum_csr


def uniform_rvs(shape):
    return np.random.uniform(0, 1, size=shape)


def convert_to_segments(fullpath):
    "Converts a sparse matrix from ssget into a segmented sum/scan input (x, f)"

    filename = fullpath
    A = mmread(filename)

    B = csr_matrix(A)

    # Data vector
    x = B.data

    # Flags vector
    f = np.zeros(B.nnz + 1)
    # The last value of the row_ptr should not be in f.
    f[B.indptr] = 1
    return x, f[:-1], B.indptr


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="torch_scatter", description="Pytorch baseline for scatter operations"
    )
    parser.add_argument("--matrixpath", type=str, default="Random")
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument(
        "--distr", type=str, choices=["PowerLaw", "Uniform"], default="Uniform"
    )

    args = parser.parse_args()
    fullpath = args.matrixpath

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"going to use: {device}")

    if fullpath != "Random":
        with open(
            "gpu_torch.csv",
            "a",
            encoding="UTF-8",
        ) as fd:
            fd.write("benchname,nnz,nrows,inputSize,outputSize,time_s\n")
            my_x, my_f, indptr = convert_to_segments(fullpath)
            my_x = torch.Tensor(my_x).to(device)
            indptr = torch.Tensor(indptr).to(torch.long).to(device)
            bench_name = fullpath.split("/")[-1]
            elapsed_array = np.zeros(100)
            for i in range(1, 50, 1):
                start_time = time.time()
                val = segment_sum_csr(src=my_x, indptr=indptr)
                end_time = time.time()
                elapsed = end_time - start_time
                elapsed_array[i] = elapsed
            fd.write(
                f"{bench_name},{len(my_x)},{len(indptr)-1},{len(my_x)},{len(indptr)-1},{np.mean(elapsed_array)}\n"
            )
    else:
        with open("gpu_torch_random.csv", "a", encoding="UTF-8") as fd:
            fd.write("operator,benchname,nnz,nrow,inputSize,outputSize,time_s\n")
            prob = args.distr
            density = args.density
            for nnr in range(10000, 50000, 1000):
                B = random(
                    nnr,
                    nnr,
                    density=density,
                    format="csr",
                    dtype=np.float32,
                    data_rvs=uniform_rvs,
                )
                my_x = torch.Tensor(B.data).to(device)
                indptr = torch.Tensor(B.indptr).to(device).to(torch.long)
                bench_name = f"Random_{prob}"
                elapsed_array = np.zeros(100)
                for i in range(1, 50, 1):
                    start_time = time.time()
                    val = segment_sum_csr(src=my_x, indptr=indptr)
                    end_time = time.time()
                    elapsed = end_time - start_time
                    elapsed_array[i] = elapsed
                print(f"Completed nrow: {nnr} with density {density}")
                fd.write(
                    f"A5000_torch_scatter,{bench_name},{len(my_x)},{nnr},{len(my_x)},{nnr},{np.mean(elapsed_array)}\n"
                )
    fd.close()
