import argparse
import time
import types
from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix, random
from torch_scatter import segment_sum_csr

import torch


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


@dataclass
class Device:
    module: types.ModuleType
    str: str

    def sync(self) -> None:
        self.module.synchronize()

    def event(self) -> "typing.Self.module.Event":  # noqa
        return self.module.Event(enable_timing=True)


parser = argparse.ArgumentParser(
    prog="torch_scatter", description="Pytorch baseline for scatter operations"
)
parser.add_argument("--matrixpath", type=str, default="")
parser.add_argument(
    "--mode",
    type=str,
    choices=["RandomMatrix", "Random", "Real"],
    default="RandomMatrix",
)
parser.add_argument("--density", type=float, default=0.05)
parser.add_argument(
    "--distr", type=str, choices=["PowerLaw", "Uniform"], default="Uniform"
)
parser.add_argument("--max_size", type=int, default=1e8, required=False)

args = parser.parse_args()
fullpath = args.matrixpath
mode = args.mode
if torch.cuda.is_available():
    device = Device(torch.cuda, "cuda:0")
else:
    device = Device(torch, "cpu")

if mode == "Real":
    with open(
        "gpu_torch.csv",
        "a",
        encoding="UTF-8",
    ) as fd:
        fd.write("benchname,nnz,nrows,size,outputsize,time_ms\n")
        my_x, my_f, indptr = convert_to_segments(fullpath)
        my_x = torch.Tensor(my_x).to(device)
        indptr = torch.Tensor(indptr).to(torch.long).to(device)
        bench_name = fullpath.split("/")[-1]
        elapsed_array = np.zeros(10)
        cache_size = 256 * 1024 * 1024
        cache = torch.ones(cache_size, dtype=torch.int8, device=device.str)
        start_events = [device.event() for _ in range(10)]
        end_events = [device.event() for _ in range(10)]
        for i in range(1, 50, 1):
            if device.str != "cpu":
                cache.zero_()
                device.sync()
                start_events[i].record()
            else:
                start_time = time.time()
            val = segment_sum_csr(src=my_x, indptr=indptr)
            if device.str != "cpu":
                end_events[i].record()
                device.sync()
                elapsed_array[i] = start_events[i].elapsed_time(end_events[i])
            else:
                end_time = time.time()
                elapsed_array[i] = end_time - start_time
            fd.write(
                f"{bench_name},{len(my_x)},{len(indptr)-1},{len(my_x)},{len(indptr)-1},{np.mean(elapsed_array)}\n"
            )

elif mode == "RandomMatrix":
    with open(
        "gpu_torch_random_matrix.csv",
        "a",
        encoding="UTF-8",
    ) as fd:
        fd.write("operator,benchname,nnz,nrow,size,outputsize,time_ms\n")
        prob = args.distr
        density = args.density
        min_range = 10000
        max_range = 50000
        step = 1000
        n = (max_range - min_range) / step
        for nnr in range(min_range, max_range, step):
            B = random(
                nnr,
                nnr,
                density=density,
                format="csr",
                dtype=np.float32,
                data_rvs=uniform_rvs,
            )
            my_x = torch.Tensor(B.data).to(device.str)
            indptr = torch.Tensor(B.indptr).to(device.str).to(torch.long)
            bench_name = f"Random_{prob}"
            elapsed_array = np.zeros(10)
            cache_size = 256 * 1024 * 1024
            cache = torch.ones(cache_size, dtype=torch.int8, device=device.str)
            start_events = [device.event() for _ in range(10)]
            end_events = [device.event() for _ in range(10)]
            for i in range(1, 10, 1):
                if device.str != "cpu":
                    cache.zero_()
                    device.sync()
                    start_events[i].record()
                else:
                    start_time = time.time()
                val = segment_sum_csr(src=my_x, indptr=indptr)
                if device.str != "cpu":
                    end_events[i].record()
                    device.sync()
                    elapsed_array[i] = start_events[i].elapsed_time(end_events[i])
                else:
                    end_time = time.time()
                    elapsed_array[i] = end_time - start_time
            print(f"Completed nrow: {nnr} with density {density}")
            fd.write(
                f"A5000_torch_scatter,{bench_name},{len(my_x)},{nnr},{len(my_x)},{nnr},{np.mean(elapsed_array)}\n"
            )
else:
    with open(
        "gpu_torch_random_fdensity.csv",
        "a",
        encoding="UTF-8",
    ) as fd:
        fd.write("operator,benchname,size,outputsize,time_ms\n")
        prob = args.distr
        density = args.density
        max_size = args.max_size
        s = 128
        num_cores = 20
        max_iters = ceil(max_size / (num_cores * s * s))
        for j in range(1, max_iters, 16 * 128 // s):
            size = j * num_cores * s * s
            my_x = torch.randn(size).to(torch.float32).to(device.str)
            my_f = torch.empty(size).uniform_(0, 1) < density
            my_f = my_f.to(torch.long)
            non_zero_indices = torch.nonzero(my_f)
            row_counts = torch.bincount(non_zero_indices[:, 0])
            # Compute indptr (cumulative sum of non-zero elements per row)
            indptr = torch.cat([torch.tensor([0]), torch.cumsum(row_counts, dim=0)]).to(
                device.str
            )
            bench_name = f"Random_{prob}"
            elapsed_array = np.zeros(10)
            cache_size = 256 * 1024 * 1024
            cache = torch.ones(cache_size, dtype=torch.int8, device=device.str)
            start_events = [device.event() for _ in range(10)]
            end_events = [device.event() for _ in range(10)]
            for i in range(1, 10, 1):
                if device.str != "cpu":
                    cache.zero_()
                    device.sync()
                    start_events[i].record()
                else:
                    start_time = time.time()
                val = segment_sum_csr(src=my_x, indptr=indptr)
                if device.str != "cpu":
                    end_events[i].record()
                    device.sync()
                    elapsed_array[i] = start_events[i].elapsed_time(end_events[i])
                else:
                    end_time = time.time()
                    elapsed_array[i] = end_time - start_time
            print(f"Completed size: {size} with density {density}")
            fd.write(
                f"A5000_torch_scatter,{bench_name},{len(my_x)},{len(my_f)},{np.mean(elapsed_array)}\n"
            )
fd.close()
