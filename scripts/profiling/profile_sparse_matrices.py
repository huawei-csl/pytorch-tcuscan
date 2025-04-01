#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# PyTorch profiling code is part of TCUSCAN-CH CSTT project.
#
# Copyright 2024 Huawei Technologies Co., Ltd

import argparse
import logging
import os
import sys
import types
import typing
from dataclasses import dataclass
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa
from scipy.io import mmread
from scipy.sparse import csr_matrix

import tcuscan_ops


def pad_to_multiple(x: torch.Tensor, s: int):
    N = x.shape[-1]
    target_size = ((N + s * s - 1) // (s * s)) * (s * s)
    pad_amount = target_size - N
    padded_x = F.pad(x, (0, pad_amount), mode="constant", value=0)
    return padded_x


def convert_to_segments(fullpath):
    "Converts a sparse matrix from ssget into a segmented sum/scan input (x, f)"

    filename = fullpath + ".mtx"
    A = mmread(filename)

    B = csr_matrix(A)

    # Data vector
    x = B.data

    # Flags vector
    f = np.zeros(B.nnz + 1)
    # The last value of the row_ptr should not be in f.
    f[B.indptr] = 1
    return x, f[:-1], B


file_handler = logging.FileHandler(filename="torch_profiler.log")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

# _MULTIPLIER = [1, 2, 3, 5, 8, 9, 12, 16, 24, 32]
_MULTIPLIER = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    24,
    32,
    44,
    56,
    68,
    80,
    100,
    120,
    140,
    160,
    180,
    200,
    500,
    1000,
]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=handlers,
)

logger = logging.getLogger(__name__)

STR_TO_DTYPE = {"fp16": torch.float16, "int16": torch.int16, "int8": torch.int8}


DEVICE = os.environ.get("DEVICE_TYPE", "npu")

NPU_DEVICE = "npu:0"
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)
assert torch.npu.is_available()

WARMUP_ITERS = 10
BENCH_ITERS = 100

STR_TO_DTYPE = {
    "fp16": torch.float16,
    "int16": torch.int16,
    "int8": torch.int8,
    "fp32": torch.float32,
}


@dataclass
class Device:
    module: types.ModuleType
    str: str

    def sync(self) -> None:
        self.module.synchronize()

    def event(self) -> "typing.Self.module.Event":
        return self.module.Event(enable_timing=True)


def _run_benchmark(
    device: Device,
    fn: typing.Callable,
    warmup_iters: int = WARMUP_ITERS,
    benchmark_iters: int = BENCH_ITERS,
) -> float:
    """
    Benchmark a given function with warmup.

    Args:
        device: Device to run benchmark on.
        fn: Function to benchmark.
        warmup_iters: Number of warmup runs.
        benchmark_iters: Number of benchmark runs.

    Returns:
        Average time in microseconds.
    """
    start_events = [device.event() for _ in range(benchmark_iters)]
    end_events = [device.event() for _ in range(benchmark_iters)]

    device.sync()
    for _ in range(warmup_iters):
        fn()

    device.sync()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2 cache
    # doesn't contain any input data before the run
    # Copied from https://github.com/triton-lang/triton/blob/v2.1.0/python/triton/testing.py#L110

    # cache_size = 256 * 1024 * 1024
    # cache = torch.ones(cache_size, dtype=torch.int8, device=device.str)

    for i in range(benchmark_iters):
        # cache.zero_()
        device.sync()
        start_events[i].record()
        fn()
        end_events[i].record()
        device.sync()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time_ms = sum(times_ms) / len(times_ms)
    avg_time_us = avg_time_ms * 1000
    return avg_time_us


def segmented_scan_single_core_benchmark(
    device: Device, x: torch.Tensor, f: torch.Tensor, s: int
) -> float:
    """
    Benchmark Segmented Scan Single Core kernel.

    Args:
        device: Device to run benchmark on.
        x: Input value tensor
        f: Input mask tensor
        s: Matrix size tiling parameter.
    """

    x_npu = x.npu()
    f_npu = f.npu()

    def run_seg_scan_single_core() -> None:
        _ = tcuscan_ops.run_seg_scan(x_npu, f_npu, s)

    return _run_benchmark(device, run_seg_scan_single_core)


def vec_segmented_scan_single_core_benchmark(
    device: Device, x: torch.Tensor, f: torch.Tensor, s: int
) -> float:
    """
    Benchmark Vectorized Segmented Scan Single Core kernel.

    Args:
        device: Device to run benchmark on.
        s: block size [32,64,128]
        vec_len: Input vector length.
        segm_density: Float value corresponding to the density"""

    x_npu = x.npu()
    f_npu = f.npu()

    def run_vec_seg_scan_single_core() -> None:
        _ = tcuscan_ops.run_seg_scan_vec(x_npu, f_npu, s)

    return _run_benchmark(device, run_vec_seg_scan_single_core)


def compress_benchmark(device: Device, x: torch.Tensor, f: torch.Tensor, s: int):
    """
    Benchmark TCUSCAN compress kernel.

    Args:
        device: Device to run benchmark on.
        x: Input value tensor
        f: Input mask tensor
        s: Matrix size tiling parameter.

    Returns:
        Average time in microseconds.
    """
    x_npu = x.npu()
    f_npu = f.npu()

    def run_compress() -> None:
        _ = tcuscan_ops.run_compress(x_npu, f_npu, s)

    return _run_benchmark(device, run_compress)


def mcscan_benchmark(device: Device, x: torch.Tensor, s: int) -> float:
    """
    Benchmark TCUSCAN multi-core scan kernel.

    Args:
        device: Device to run benchmark on.
        x: Input value tensor
        s: Matrix size tiling parameter.

    Returns:
        Average time in microseconds.
    """
    x_npu = x.npu()

    def run_scan() -> None:
        _ = tcuscan_ops.run_scan_multi_core(x_npu, s)

    return _run_benchmark(device, run_scan)


def baseline_diff_benchmark(device: Device, x: torch.Tensor) -> float:
    """
    Benchmark vector diff kernel with torch.

    Args:
        device: Device to run benchmark on.
        x: Input value tensor

    Returns:
        Average time in microseconds.
    """

    def run_diff() -> None:
        _ = torch.diff(x)

    return _run_benchmark(device, run_diff)


def baseline_spmv(device: Device, B: csr_matrix, s: int):
    rng = np.random.default_rng(seed=42)
    vals = torch.from_numpy((B.data).astype(np.float16))
    idx = torch.from_numpy((B.indptr).astype(np.uint32))
    cols = torch.from_numpy((B.indices).astype(np.uint32))
    vector = torch.from_numpy(rng.uniform(1, 9, len(idx) - 1).astype(np.float16))
    vals_npu = vals.npu()
    idx_npu = idx.npu()
    col_npu = cols.npu()
    vec_npu = vector.npu()
    torch.npu.synchronize()

    def run_spmv():
        _ = tcuscan_ops.run_spmv(vals_npu, idx_npu, col_npu, vec_npu, s)

    return _run_benchmark(device, run_spmv)


def benchmark(
    device: Device,
    op_name: str,
    dtype: str,
    fn: typing.Callable,
    benchname: str,
) -> None:
    """
    Benchmark a given function.

    Args:
        device: Device to run benchmark on.
        op_name: Operator name.
        dtype: Input data type.
        fn: Function to benchmark.
        benchname: name of the benchmark used
    """
    filename = f"bench_results_{op_name}_{dtype}_{benchname}.csv"
    with open(filename, "w", encoding="UTF-8") as fd:
        fd.write("benchname,operator,dtype,time_us\n")

        logger.info(
            f"Benchmark: {benchname}, OP:{op_name}, dtype: {dtype}, device: {device.str}"
        )
        time = fn(device)
        fd.write(f"{benchname},{op_name},{dtype},{time:.2f}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="torch_profile", description="Profiler for torch_npu operators"
    )

    parser.add_argument(
        "--bench",
        choices=[
            "seg_scan_sc",
            "compress",
            "mcscan",
            "diff",
            "vec_seg_scan_sc",
            "spmv",
        ],
    )
    parser.add_argument("--dtype", choices=["int8", "fp16", "int16", "fp32"])
    parser.add_argument("--s", type=int, default=64, required=False)
    parser.add_argument("--max_size", type=int, default=1e8, required=False)
    parser.add_argument("--num_cores", type=int, default=20, required=False)
    parser.add_argument("--matrixpath", type=str)
    args = parser.parse_args()

    fullpath = args.matrixpath
    my_x, my_f, B = convert_to_segments(fullpath)
    my_x = torch.Tensor(my_x)
    my_f = torch.Tensor(my_f)

    dtype = args.dtype
    bench = args.bench
    max_size = args.max_size
    num_cores = args.num_cores
    s = args.s
    my_x = pad_to_multiple(my_x, s)
    my_f = pad_to_multiple(my_f, s)
    if DEVICE == "npu":
        device = Device(torch.npu, NPU_DEVICE)
    elif DEVICE == "cpu":
        device = Device(torch, "cpu")
    else:
        device = Device(torch.cuda, "cuda:0")

    if bench == "seg_scan_sc" and dtype in ["fp16"]:
        bench_name = fullpath.split("/")[-1]
        my_x = pad_to_multiple(my_x, s)
        my_f = pad_to_multiple(my_f, s)
        benchmark(
            device,
            f"seg_scan_sc_{s}",
            dtype,
            partial(segmented_scan_single_core_benchmark, x=my_x, f=my_f, s=s),
            bench_name,
        )
    elif bench == "vec_seg_scan_sc" and dtype in ["fp16"]:
        bench_name = fullpath.split("/")[-1]
        benchmark(
            device,
            f"vec_seg_scan_sc_{s}",
            dtype,
            partial(vec_segmented_scan_single_core_benchmark, x=my_x, f=my_f, s=s),
            bench_name,
        )
    elif bench == "compress" and dtype in ["fp16", "fp32"]:
        bench_name = fullpath.split("/")[-1]
        benchmark(
            device,
            f"compress_{s}",
            dtype,
            partial(compress_benchmark, x=my_x, f=my_f, s=s),
            bench_name,
        )
    elif bench == "mcscan" and dtype in ["fp16"]:
        bench_name = fullpath.split("/")[-1]
        benchmark(
            device,
            f"mcscan_{s}",
            dtype,
            partial(mcscan_benchmark, x=my_x, s=s),
            bench_name,
        )
    elif bench == "diff" and dtype in ["fp16", "fp32"]:
        bench_name = fullpath.split("/")[-1]
        my_x = pad_to_multiple(my_x, s)
        my_f = pad_to_multiple(my_f, s)
        benchmark(
            device, "diff", dtype, partial(baseline_diff_benchmark, x=my_x), bench_name
        )
    elif bench == "spmv" and dtype in ["fp16"]:
        bench_name = fullpath.split("/")[-1]
        benchmark(
            device,
            f"spmv_{s}",
            dtype,
            partial(
                baseline_spmv,
                B=B,
                s=s,
            ),
            bench_name,
        )
    else:
        raise RuntimeError(
            f"Unsupported benchmark setup: bench:{bench}, dtype:{dtype}, s:{s}"
        )
