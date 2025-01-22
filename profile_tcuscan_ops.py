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
import numpy as np
from dataclasses import dataclass
from functools import partial
from math import ceil

import torch

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
if DEVICE == "npu":
    import torch_npu

    try:
        import tcuscan_ops
    except:
        RuntimeError("Please run 'make build' first.")
        exit(-1)
    torch.npu.config.allow_internal_format = False

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


def vadd_benchmark(device: Device, vec_len: int) -> float:
    """
    Benchmark vector addition kernel.

    Args:
        device: Device to run benchmark on.
        vec_len: Input vector length.

    Returns:
        Average time in microseconds.
    """
    x = torch.rand(vec_len, device=device.str, dtype=torch.float16)
    y = torch.rand(vec_len, device=device.str, dtype=torch.float16)

    def run_vadd() -> None:
        z = tcuscan_ops.run_add_custom(x, y)

    return _run_benchmark(device, run_vadd)


def clone_benchmark(device: Device, size: int, dtype: torch.dtype) -> float:
    """
    Benchmark torch.clone kernel.

    Args:
        device: Device to run benchmark on.
        size: Size of the arrays to use.
        dtype: Data type of the input array.

    Returns:
        Average time in microseconds.
    """
    if dtype == torch.float16:
        x = torch.rand(size, device=device.str, dtype=dtype)
    elif dtype == torch.int16:
        x = torch.randint(0, 2**7 - 1, (size,), device=device.str, dtype=dtype)
    else:
        raise ValueError("Incorrect copy data type")

    def run_clone() -> None:
        z = torch.clone(x)

    return _run_benchmark(device, run_clone)


def diff_benchmark(device: Device, vec_len: int, dtype=torch.dtype) -> float:
    """
    Benchmark vector diff kernel.

    Args:
        device: Device to run benchmark on.
        vec_len: Input vector length.

    Returns:
        Average time in microseconds.
    """
    x = torch.rand(vec_len, device=device.str, dtype=dtype)

    def run_diff() -> None:
        z = tcuscan_ops.run_diff(x)

    return _run_benchmark(device, run_diff)


def baseline_diff_benchmark(device: Device, vec_len: int, dtype=torch.dtype) -> float:
    """
    Benchmark baseline `torch.diff(x)` kernel.

    Args:
        device: Device to run benchmark on.
        vec_len: Input vector length.

    Returns:
        Average time in microseconds.
    """
    x = torch.rand(vec_len, device=device.str, dtype=torch.float16)
    if dtype in [torch.float16, torch.float32]:
        x = torch.rand(vec_len, device=device.str, dtype=dtype)
    else:
        raise ValueError("Invalid diff_cann input data type")

    def run_diff() -> None:
        z = torch.diff(x)

    return _run_benchmark(device, run_diff)


def baseline_diffp_benchmark(device: Device, vec_len: int, dtype=torch.dtype) -> float:
    """
    Benchmark baseline `torch.diff(x, prepend=)` kernel.

    Args:
        device: Device to run benchmark on.
        vec_len: Input vector length.

    Returns:
        Average time in microseconds.
    """
    x = torch.rand(vec_len, device=device.str, dtype=torch.float16)
    if dtype in [torch.float16, torch.float32]:
        x = torch.rand(vec_len, device=device.str, dtype=dtype)
    else:
        raise ValueError("Invalid diff_cann input data type")

    def run_diff() -> None:
        z = torch.diff(x, prepend=torch.zeros(1, device=device.str))

    return _run_benchmark(device, run_diff)


def csr_gather_benchmark(device: Device, vec_len: int) -> float:
    """
    Benchmark CSR gather kernel.

    Args:
        device: Device to run benchmark on.
        vec_len: Input vector length.

    Returns:
        Average time in microseconds.
    """

    # Maximum value of x cannot exceed 20K (UB shared memory size)
    max_x_len = 2 * 1024

    input_x = torch.rand(max_x_len, device=device.str).half()

    input_values = torch.randn(vec_len).half().npu()
    input_cols = torch.randint(
        low=0, high=max_x_len, size=(vec_len,), dtype=torch.int32
    ).npu()

    def run_csr_gather() -> None:
        z = tcuscan_ops.run_csr_gather(input_values, input_cols, input_x)

    return _run_benchmark(device, run_csr_gather)


def segmented_scan_single_core_benchmark(
    device: Device, s: int, vec_len: int, segm_density: float
) -> float:
    """
    Benchmark Segmented Scan Single Core kernel.

    Args:
        device: Device to run benchmark on.
        s: block size [32,64,128]
        vec_len: Input vector length.
        segm_density: Float value corresponding to the density"""

    x = torch.randn(vec_len).half().npu()
    f = torch.empty(vec_len).uniform_(0, 1) < segm_density
    f = f.to(torch.int8).npu()

    def run_seg_scan_single_core() -> None:
        result = tcuscan_ops.run_seg_scan(x, f, s)

    return _run_benchmark(device, run_seg_scan_single_core)


def compress_benchmark(device: Device, size: int, dtype: torch.dtype, s: int):

    x = torch.randn(size).half().npu()
    mask = (torch.rand(size=(size,)) < 0.05).to(torch.int8).npu()
    if (dtype == torch.float16) or (dtype == torch.float32):
        x = torch.rand(size, device=device.str, dtype=dtype)
    else:
        raise RuntimeError(f"dtype {dtype} is not supported in TCUSCAN scan operator")

    def run_compress() -> None:
        z = tcuscan_ops.run_compress(x, mask, s)

    return _run_benchmark(device, run_compress)


def mcscan_benchmark(device: Device, size: int, dtype: torch.dtype, s: int) -> float:
    """
    Benchmark TCUSCAN multi-core scan kernel.

    Args:
        device: Device to run benchmark on.
        size: Size of the arrays to use.
        dtype: Data type of the input/output arrays.
        s: Matrix size tiling parameter.

    Returns:
        Average time in microseconds.
    """
    if dtype == torch.float16:
        x = torch.rand(size, device=device.str, dtype=dtype)
    elif dtype == torch.int8:
        x = torch.randint(
            0, torch.iinfo(dtype).max, (size,), device=device.str, dtype=dtype
        )
    else:
        raise RuntimeError(f"dtype {dtype} is not supported in TCUSCAN scan operator")

    def run_scan() -> None:
        out = tcuscan_ops.run_scan_multi_core(x, s)

    return _run_benchmark(device, run_scan)


def benchmark(
    device: Device,
    op_name: str,
    dtype: str,
    fn: typing.Callable,
    sizes: typing.List[int],
) -> None:
    """
    Benchmark a given function.

    Args:
        device: Device to run benchmark on.
        op_name: Operator name.
        dtype: Input data type.
        fn: Function to benchmark.
        sizes: Sizes of the arrays to use.
    """
    with open(f"bench_results_{op_name}_{dtype}.csv", "w") as fd:

        once = True
        for size in sizes:
            if "seg_scan_sc" in op_name:
                names = op_name.split("_")
                sparsity = names[-1]
                op = "_".join(names[:-1])
                if once:
                    fd.write("operator,dtype,size,s,density,time_us\n")
                    once = False
                logger.info(
                    f"OP:{op}, dtype: {dtype}, size: {size[0]:,}, device: {device.str}"
                )
                time = fn(device, size[1], size[0], float(sparsity))
                fd.write(f"{op},{dtype},{size[0]},{size[1]},{sparsity},{time:.2f}\n")
            else:
                if once:
                    fd.write("operator,dtype,size,time_us\n")
                    once = False
                logger.info(
                    f"OP:{op_name}, dtype: {dtype}, size: {size:,}, device: {device.str}"
                )
                time = fn(device, size)
                fd.write(f"{op_name},{dtype},{size},{time:.2f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="torch_profile", description="Profiler for torch_npu operators"
    )

    parser.add_argument(
        "--bench",
        choices=[
            "vadd",
            "copy",
            "diff",
            "diff_cann",
            "diffp_cann",
            "csr_gather",
            "seg_scan_sc",
        ],
    )
    parser.add_argument("--dtype", choices=["int8", "fp16", "int16", "fp32"])
    parser.add_argument("--s", type=int, default=64, required=False)
    parser.add_argument("--max_size", type=int, default=1e8, required=False)
    parser.add_argument("--num_cores", type=int, default=20, required=False)
    parser.add_argument("--density", type=float, default=0.1, required=False)
    args = parser.parse_args()

    bench = args.bench
    dtype = args.dtype
    max_size = args.max_size
    num_cores = args.num_cores
    s = args.s
    sp_density = args.density

    if DEVICE == "npu":
        device = Device(torch.npu, NPU_DEVICE)
    elif DEVICE == "cpu":
        device = Device(torch, "cpu")
    else:
        device = Device(torch.cuda, "cuda:0")

    # Maximum number of iterations
    max_iters = ceil(max_size / (num_cores * s * s))

    # Input sizes to benchmark
    sizes = [i * num_cores * s * s for i in range(1, max_iters, 16 * 128 // s)]

    if bench == "vadd":
        benchmark(device, "vadd", "fp16", vadd_benchmark, sizes)
    elif bench == "copy" and dtype in ["int16", "fp16"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            "copy",
            dtype,
            partial(clone_benchmark, dtype=tdtype),
            sizes,
        )
    elif bench == "diff_cann" and dtype in ["fp16", "fp32"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            "diff_cann",
            dtype,
            partial(baseline_diff_benchmark, dtype=tdtype),
            sizes,
        )
    elif bench == "diffp_cann" and dtype in ["fp16", "fp32"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            "diffp_cann",
            dtype,
            partial(baseline_diffp_benchmark, dtype=tdtype),
            sizes,
        )
    elif bench == "diff" and dtype in ["fp16", "fp32"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            "diff",
            dtype,
            partial(diff_benchmark, dtype=tdtype),
            sizes,
        )
    elif bench == "csr_gather":
        benchmark(device, "csr_gather", "fp16", csr_gather_benchmark, sizes)
    elif bench == "seg_scan_sc" and dtype in ["fp16"]:
        sizes = []
        possible_sizes = [32, 64, 128]
        for mul in _MULTIPLIER:
            for s in possible_sizes:
                val = mul * s * s
                touple = (val, s)
                sizes.append(touple)
        benchmark(
            device,
            f"seg_scan_sc_{sp_density}",
            "fp16",
            segmented_scan_single_core_benchmark,
            sizes,
        )
    else:
        raise RuntimeError(
            f"Unsupported benchmark setup: bench:{bench}, dtype:{dtype}, s:{s}"
        )
