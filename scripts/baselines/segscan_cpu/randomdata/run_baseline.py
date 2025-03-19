import argparse
import os
from math import ceil

import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix


def convert_to_segments(fullpath):
    "Converts a sparse matrix from ssget into a segmented sum/scan input (x, f)"

    A = mmread(fullpath)

    B = csr_matrix(A)

    # Data vector
    x = B.data

    # Flags vector
    f = np.zeros(B.nnz + 1)
    # The last value of the row_ptr should not be in f.
    f[B.indptr] = 1
    nnz = B.nnz
    return x, f[:-1], nnz


parser = argparse.ArgumentParser(
    prog="torch_profile", description="Profiler for torch_npu operators"
)
parser.add_argument("--matrixpath", type=str, default="randomvals")
parser.add_argument("--baselinepath", type=str)
parser.add_argument("--max_size", type=int, default=1e8, required=False)
parser.add_argument("--num_cores", type=int, default=1, required=False)
parser.add_argument("--density", type=float, default=0.001, required=False)

args = parser.parse_args()
matrixpath = args.matrixpath
baselinepath = args.baselinepath
max_size = args.max_size
num_cores = args.num_cores
sp_density = args.density

if matrixpath != "randomvals":
    my_x, my_f, nnz = convert_to_segments(matrixpath)
    density = nnz / len(my_f)
    size = len(my_x)
    os.system(f"{os. getcwd()}/test_baseline {size} {density}")
else:
    s = 128
    max_size = args.max_size
    max_iters = ceil(max_size / (num_cores * s * s))
    for i in range(1, max_iters, 16 * 128 // s):
        size = i * num_cores * s * s
        os.system(f"{os. getcwd()}/test_baseline {size} {sp_density}")
