#!/usr/bin/env python3
# --------------------------------------------------------------------------------
# Copyright (c) 2023-2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pytorch-tcuscan/
# for the full License text.
# --------------------------------------------------------------------------------

import argparse
import logging
import os
import sys
import types
import typing
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import scipy
import torch.nn.functional as F
import torch_npu  # noqa
from scipy.io import mmread
from scipy.sparse import csr_matrix

import tcuscan_ops
import torch


def pad_to_multiple(x: torch.Tensor, s: int):
    N = x.shape[-1]
    target_size = ((N + s * s - 1) // (s * s)) * (s * s)
    pad_amount = target_size - N
    padded_x = F.pad(x, (0, pad_amount), mode="constant", value=0)
    return padded_x


def convert_to_segments(B: scipy.sparse.csr_matrix):
    "Converts a CSR sparse matrix from ssget into a segmented sum/scan input (x, f)"
    # Sparse matrix nnzs
    x = B.data

    # Flags vector
    f = np.zeros(B.nnz + 1)
    # The last value of the row_ptr should not be in f.
    f[B.indptr] = 1
    return x, f[:-1]


file_handler = logging.FileHandler(filename="torch_profiler.log")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=handlers,
)

logger = logging.getLogger(__name__)

STR_TO_DTYPE = {"fp16": torch.float16, "int16": torch.int16, "int8": torch.int8}


DEVICE = os.environ.get("DEVICE_TYPE", "npu")

NPU_DEVICE = "npu:1"
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


def _dtype_bytes(dtype: torch.dtype) -> int:
    if dtype.is_floating_point:
        return torch.finfo(dtype).bits // 8
    return torch.iinfo(dtype).bits // 8


def _spmv_io_bytes(nrow: int, nnz: int, dtype: torch.dtype) -> int:
    "Returns the number of bytes read/written by a SpMV operation on a CSR matrix with given dimensions and data type."
    elem = _dtype_bytes(dtype)
    return (
        nnz * elem  # values
        + nnz * 4  # col indices (int32)
        + (nrow + 1) * 4  # row pointers (uint32)
        + nrow * elem  # input vector x (A @ x)
        + nrow * 4  # output vector (fp32 or int32)
    )


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

    cache_size = 256 * 1024 * 1024
    cache = torch.ones(cache_size, dtype=torch.int8, device=device.str)

    for i in range(benchmark_iters):
        cache.zero_()
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

    return (
        _run_benchmark(device, run_seg_scan_single_core),
        len(x_npu),
        int(f_npu.sum().item()),
    )


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

    return _run_benchmark(device, run_vec_seg_scan_single_core), len(x_npu), sum(f_npu)


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

    return _run_benchmark(device, run_compress), len(x_npu), int(f_npu.sum().item())


def mcscan_benchmark(device: Device, B: csr_matrix, s: int) -> float:
    """
    Benchmark TCUSCAN multi-core scan kernel.

    Args:
        device: Device to run benchmark on.
        B: Input CSR Matrix
        s: Matrix size tiling parameter.

    Returns:
        Average time in microseconds.
    """

    vals = torch.from_numpy((B.data).astype(np.float16))
    x_npu = vals.npu()

    def run_scan() -> None:
        _ = tcuscan_ops.run_scan_multi_core(x_npu, s)

    return _run_benchmark(device, run_scan), B.nnz, B.shape[0]


def baseline_diff_benchmark(device: Device, B: csr_matrix) -> float:
    """
    Benchmark vector diff kernel with torch.

    Args:
        device: Device to run benchmark on.
        x: Input value tensor

    Returns:
        Average time in microseconds.
    """
    vals = torch.from_numpy((B.data).astype(np.float32))
    vals_npu = vals.npu()

    def run_diff() -> None:
        _ = torch.diff(vals_npu)

    return _run_benchmark(device, run_diff), B.nnz, B.shape[0]


def spmv_multi_cube_benchmark(device: Device, B: csr_matrix, s: int):
    """
    Benchmark TCUSCAN SpMV kernel.

    Args:
        device: Device to run benchmark on.
        B: Input CSR Matrix
        s: Matrix size tiling parameter.

    Returns:
        Average time in microseconds.
    """
    rng = np.random.default_rng(seed=42)
    vals = torch.from_numpy((B.data).astype(np.float16))
    idx = torch.from_numpy((B.indptr).astype(np.uint32))
    cols = torch.from_numpy((B.indices).astype(np.int32))
    vector = torch.from_numpy(rng.uniform(1, 9, len(idx) - 1).astype(np.float16))
    vals_npu = vals.npu()
    idx_npu = idx.npu()
    col_npu = cols.npu()
    vec_npu = vector.npu()
    ones = torch.ones((s, s), dtype=torch.float16, device=NPU_DEVICE)
    upper = torch.triu(ones)
    lower_strict = torch.tril(ones, -1)
    torch.npu.synchronize()

    def run_spmv_multi_cube():
        _ = tcuscan_ops.run_spmv_multi_cube(
            vals_npu, idx_npu, col_npu, vec_npu, upper, lower_strict
        )

    return _run_benchmark(device, run_spmv_multi_cube), B.nnz, B.shape[0]


def spmv_vec_only_benchmark(device: Device, B: csr_matrix, s: int):
    """
    Benchmark TCUSCAN SpMV kernel.

    Args:
        device: Device to run benchmark on.
        B: Input CSR Matrix
        s: Matrix size tiling parameter.

    Returns:
        Average time in microseconds.
    """
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
        _ = tcuscan_ops.run_spmv_vec_only(vals_npu, idx_npu, col_npu, vec_npu, s)

    return _run_benchmark(device, run_spmv), B.nnz, B.shape[0]


def spmv_benchmark(device: Device, B: csr_matrix, s: int):
    """
    Benchmark TCUSCAN SpMV kernel.

    Args:
        device: Device to run benchmark on.
        B: Input CSR Matrix
        s: Matrix size tiling parameter.

    Returns:
        Average time in microseconds.
    """
    rng = np.random.default_rng(seed=42)
    vals = torch.from_numpy((B.data).astype(np.float16))
    idx = torch.from_numpy((B.indptr).astype(np.uint32))
    cols = torch.from_numpy((B.indices).astype(np.int32))
    vector = torch.from_numpy(rng.uniform(1, 9, len(idx) - 1).astype(np.float16))
    vals_npu = vals.npu()
    idx_npu = idx.npu()
    col_npu = cols.npu()
    vec_npu = vector.npu()

    torch.npu.synchronize()

    def run_spmv():
        _ = tcuscan_ops.run_spmv(vals_npu, idx_npu, col_npu, vec_npu, s)

    return _run_benchmark(device, run_spmv), B.nnz, B.shape[0]


def spmv_v2_benchmark(device: Device, B: csr_matrix, s: int):
    """
    Benchmark TCUSCAN SpMV v2 kernel (segmented-sum based).

    Args:
        device: Device to run benchmark on.
        B: Input CSR Matrix
        s: Matrix size tiling parameter.

    Returns:
        Average time in microseconds.
    """
    rng = np.random.default_rng(seed=42)
    vals = torch.from_numpy((B.data).astype(np.float16))
    idx = torch.from_numpy((B.indptr).astype(np.uint32))
    cols = torch.from_numpy((B.indices).astype(np.int32))
    vector = torch.from_numpy(rng.uniform(1, 9, len(idx) - 1).astype(np.float16))
    vals_npu = vals.npu()
    idx_npu = idx.npu()
    col_npu = cols.npu()
    vec_npu = vector.npu()

    torch.npu.synchronize()

    def run_spmv_v2():
        _ = tcuscan_ops.run_spmv_v2(vals_npu, idx_npu, col_npu, vec_npu, s)

    return _run_benchmark(device, run_spmv_v2), B.nnz, B.shape[0]


def csr_gather_benchmark(device: Device, B: csr_matrix):
    """
    Benchmark TCUSCAN multi-core csr_gather kernel.

    Args:
        device: Device to run benchmark on.
        B: Input CSR Matrix

    Returns:
        Average time in microseconds.
    """

    rng = np.random.default_rng(seed=42)
    vals = torch.from_numpy((B.data).astype(np.float16))
    idx = torch.from_numpy((B.indptr).astype(np.uint32))
    cols = torch.from_numpy((B.indices).astype(np.int32))
    vector = torch.from_numpy(rng.uniform(1, 9, len(idx) - 1).astype(np.float16))
    vals_npu = vals.npu()
    col_npu = cols.npu()
    vec_npu = vector.npu()

    def run_csr_gather():
        _ = tcuscan_ops.run_csr_gather(vals_npu, col_npu, vec_npu)

    return _run_benchmark(device, run_csr_gather), B.nnz, B.shape[0]


def gather_spmv_benchmark(device: Device, B: csr_matrix, s: int):
    """
    Benchmark TCUSCAN multi-core gather_spmv kernel.

    Args:
        device: Device to run benchmark on.
        B: Input CSR Matrix
        s: tile size for the vector gather spmv

    Returns:
        Average time in microseconds.
    """
    vals = torch.from_numpy((B.data).astype(np.float32))
    idx = torch.from_numpy((B.indptr).astype(np.uint32))
    vals_npu = vals.npu()
    idx_npu = idx.npu()

    def run_gather_spmv():
        _ = tcuscan_ops.run_gather_spmv(vals_npu, idx_npu, s)

    return _run_benchmark(device, run_gather_spmv), B.nnz, B.shape[0]


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
    report_path = os.getenv("TCUSCAN_BENCHMARK_REPORT_PATH", "./")
    if not os.path.exists(report_path):
        os.makedirs(report_path)
    logger.info(f"Directory that reports are stored: {report_path}")
    filename = os.path.join(report_path, f"bench_results_{op_name}_{dtype}.csv")
    file = Path(filename)
    if not file.is_file():
        with open(filename, "w", encoding="UTF-8") as fd:
            fd.write("benchname,operator,dtype,nnz,nrows,time_us,bw_gbps\n")

    with open(filename, "a", encoding="UTF-8") as fd:
        logger.info(
            f"Benchmark: {benchname}, OP:{op_name}, dtype: {dtype}, device: {device.str}"
        )
        time, nnz, nrows = fn(device)
        bw_gbps = _spmv_io_bytes(nrows, nnz, STR_TO_DTYPE[dtype]) / (time * 1e-6) / 1e9
        fd.write(
            f"{benchname},{op_name},{dtype},{nnz},{nrows},{time:.2f},{bw_gbps:.2f}\n"
        )


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
            "spmv_v2",
            "spmv_multi_cube",
            "csr_gather",
            "gather_spmv",
        ],
    )
    parser.add_argument("--dtype", choices=["int8", "fp16", "int16", "fp32"])
    parser.add_argument("--s", type=int, default=128, required=False)
    parser.add_argument("--max_size", type=int, default=1e8, required=False)
    parser.add_argument("--num_cores", type=int, default=20, required=False)
    parser.add_argument("--matrixpath", type=str)
    args = parser.parse_args()

    fullpath = args.matrixpath
    A = mmread(f"{fullpath}.mtx")
    B = csr_matrix(A)
    my_x, my_f = convert_to_segments(B)
    my_x = torch.Tensor(my_x)
    my_f = torch.Tensor(my_f)

    dtype = args.dtype
    bench = args.bench
    max_size = args.max_size
    num_cores = args.num_cores
    s = args.s
    if DEVICE == "npu":
        device = Device(torch.npu, NPU_DEVICE)
    elif DEVICE == "cpu":
        device = Device(torch, "cpu")
    else:
        device = Device(torch.cuda, "cuda:0")

    bench_name = fullpath.split("/")[-1]

    if bench == "seg_scan_sc" and dtype in ["fp16"]:
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
        benchmark(
            device,
            f"vec_seg_scan_sc_{s}",
            dtype,
            partial(vec_segmented_scan_single_core_benchmark, x=my_x, f=my_f, s=s),
            bench_name,
        )
    elif bench == "compress" and dtype in ["fp16", "fp32"]:
        benchmark(
            device,
            f"compress_{s}",
            dtype,
            partial(compress_benchmark, x=my_x, f=my_f, s=s),
            bench_name,
        )
    elif bench == "mcscan" and dtype in ["fp16"]:
        benchmark(
            device,
            f"mcscan_{s}",
            dtype,
            partial(mcscan_benchmark, B=B, s=s),
            bench_name,
        )
    elif bench == "diff" and dtype in ["fp16", "fp32"]:
        benchmark(
            device, "diff", dtype, partial(baseline_diff_benchmark, B=B), bench_name
        )
    elif bench == "spmv_multi_cube" and dtype in ["fp16"]:
        benchmark(
            device,
            f"spmv_multi_cube_{s}",
            dtype,
            partial(
                spmv_multi_cube_benchmark,
                B=B,
                s=s,
            ),
            bench_name,
        )
    elif bench == "spmv" and dtype in ["fp16"]:
        benchmark(
            device,
            f"spmv_{s}",
            dtype,
            partial(
                spmv_benchmark,
                B=B,
                s=s,
            ),
            bench_name,
        )
    elif bench == "spmv_v2" and dtype in ["fp16"]:
        benchmark(
            device,
            f"spmv_v2_{s}",
            dtype,
            partial(
                spmv_v2_benchmark,
                B=B,
                s=s,
            ),
            bench_name,
        )
    elif bench == "csr_gather" and dtype in ["fp16"]:
        benchmark(
            device,
            "csr_gather",
            dtype,
            partial(
                csr_gather_benchmark,
                B=B,
            ),
            bench_name,
        )
    elif bench == "gather_spmv" and dtype in ["fp32"]:
        benchmark(
            device,
            f"gather_spmv_{s}",
            dtype,
            partial(
                gather_spmv_benchmark,
                B=B,
                s=s,
            ),
            bench_name,
        )
    else:
        raise RuntimeError(
            f"Unsupported benchmark setup: bench:{bench}, dtype:{dtype}, s:{s}"
        )
