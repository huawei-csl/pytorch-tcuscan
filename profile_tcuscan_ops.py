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
from math import ceil
from typing import Optional

import torch

DEVICE = os.environ.get("DEVICE_TYPE", "npu")
if DEVICE == "npu":
    import torch_npu  # noqa

    NPU_DEVICE = "npu:1"
    torch.npu.config.allow_internal_format = False
    torch.npu.set_device(NPU_DEVICE)
    assert torch.npu.is_available()

import tcuscan_ops  # noqa

file_handler = logging.FileHandler(filename="torch_profiler.log")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

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
        _ = tcuscan_ops.run_add_custom(x, y)

    return _run_benchmark(device, run_vadd)


def copy_benchmark(device: Device, size: int, s: int, dtype: torch.dtype) -> float:
    """
    Benchmark torch.clone kernel.

    Args:
        device: Device to run benchmark on.
        size: Size of the arrays to use.
        dtype: Data type of the input array.

    Returns:
        Average time in microseconds.
    """
    x = torch.rand(size, device=device.str, dtype=dtype)

    def run_copy() -> None:
        _ = tcuscan_ops.run_copy(x, s)

    return _run_benchmark(device, run_copy)


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
        _ = torch.clone(x)

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
        _ = tcuscan_ops.run_diff(x)

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
        _ = torch.diff(x)

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
        _ = torch.diff(x, prepend=torch.zeros(1, device=device.str))

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
        _ = tcuscan_ops.run_csr_gather(input_values, input_cols, input_x)

    return _run_benchmark(device, run_csr_gather)


def segmented_scan_single_core_benchmark(
    device: Device, vec_len: int, s: int, segm_density: float
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
        _ = tcuscan_ops.run_seg_scan(x, f, s)

    return _run_benchmark(device, run_seg_scan_single_core)


def vec_segmented_scan_single_core_benchmark(
    device: Device, vec_len: int, s: int, segm_density: float
) -> float:
    """
    Benchmark Vectorized Segmented Scan Single Core kernel.

    Args:
        device: Device to run benchmark on.
        s: block size [32,64,128]
        vec_len: Input vector length.
        segm_density: Float value corresponding to the density"""

    x = torch.randn(vec_len).half().npu()
    f = torch.empty(vec_len).uniform_(0, 1) < segm_density
    f = f.to(torch.int8).npu()

    def run_vec_seg_scan_single_core() -> None:
        _ = tcuscan_ops.run_seg_scan_vec(x, f, s)

    return _run_benchmark(device, run_vec_seg_scan_single_core)


def compress_benchmark(
    device: Device, size: int, dtype: torch.dtype, s: int, segm_density: float
):

    x = torch.randn(size).half().npu()
    mask = (torch.rand(size=(size,)) < segm_density).to(torch.int8).npu()
    if dtype in {torch.float16, torch.float32}:
        x = torch.rand(size, device=device.str, dtype=dtype)
    else:
        raise RuntimeError(f"dtype {dtype} is not supported in TCUSCAN scan operator")

    def run_compress() -> None:
        _ = tcuscan_ops.run_compress(x, mask, s)

    return _run_benchmark(device, run_compress)


def segmented_sum_benchmark(
    device: Device, size: int, segm_density: float, dtype: torch.dtype, s: int
):

    x = torch.randn(size, dtype=dtype)
    f = (torch.randn(size) < segm_density).to(torch.int8)
    f[0] = 0
    x_npu = x.npu()
    f_npu = torch.concat([f[1:], torch.ones(1, dtype=torch.int8)]).contiguous().npu()

    def run_seg_sum() -> None:
        _ = tcuscan_ops.run_seg_sum(x_npu, f_npu, s)

    return _run_benchmark(device, run_seg_sum)


def scscan_benchmark(device: Device, size: int, dtype: torch.dtype, s: int) -> float:
    x = torch.rand(size, device=device.str, dtype=dtype)

    def run_scan() -> None:
        _ = tcuscan_ops.run_scan_single_core(x, s)

    return _run_benchmark(device, run_scan)


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
        _ = tcuscan_ops.run_scan_multi_core(x, s)

    return _run_benchmark(device, run_scan)


def benchmark(
    device: Device,
    op_name: str,
    dtype: str,
    fn: typing.Callable,
    sizes: typing.List[int],
    density: Optional[float] = None,
):
    """
    Benchmark a given function.

    Args:
        device: Device to run benchmark on.
        op_name: Operator name.
        dtype: Input data type.
        fn: Function to benchmark.
        sizes: Sizes of the arrays to use.
        density: percentage of non-zero elements in a sparse matrix
    """

    density_str = ""
    if density is not None:
        density_str = "density,"

    with open(
        f"bench_results_{op_name}_{dtype}_{ None if (density_str is None) else density}.csv",
        "w",
        encoding="UTF-8",
    ) as fd:
        fd.write("operator,dtype,size,density,time_us\n")

        for size in sizes:
            logger.info(
                f"OP:{op_name}, dtype: {dtype}, size: {size:}, density: {None if (density_str is None) else density }, device: {device.str}"
            )
            time = fn(device, size)
            fd.write(f"{op_name},{dtype},{size},{density},{time:.2f}\n")


if __name__ == "__main__":  # noqa
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
            "mcscan",
            "compress",
            "segmented_sum",
            "custom_copy",
            "vec_seg_scan_sc",
            "scscan",
        ],
    )
    parser.add_argument("--dtype", choices=["int8", "fp16", "int16", "fp32"])
    parser.add_argument("--s", type=int, default=64, required=False)
    parser.add_argument("--max_size", type=int, default=1e8, required=False)
    parser.add_argument("--num_cores", type=int, default=20, required=False)
    parser.add_argument("--density", type=float, default=None, required=False)
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
            sp_density,
        )
    elif bench == "custom_copy" and dtype in ["fp32", "fp16"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            f"custom_copy_{s}",
            dtype,
            partial(copy_benchmark, s=s, dtype=tdtype),
            sizes,
            sp_density,
        )
    elif bench == "diff_cann" and dtype in ["fp16", "fp32"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            "diff_cann",
            dtype,
            partial(baseline_diff_benchmark, dtype=tdtype),
            sizes,
            sp_density,
        )
    elif bench == "diffp_cann" and dtype in ["fp16", "fp32"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            "diffp_cann",
            dtype,
            partial(baseline_diffp_benchmark, dtype=tdtype),
            sizes,
            sp_density,
        )
    elif bench == "diff" and dtype in ["fp16", "fp32"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            "diff",
            dtype,
            partial(diff_benchmark, dtype=tdtype),
            sizes,
            sp_density,
        )
    elif bench == "compress" and dtype in ["fp16", "fp32"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            f"compress_{s}",
            dtype,
            partial(compress_benchmark, dtype=tdtype, s=s, segm_density=sp_density),
            sizes,
            sp_density,
        )
    elif bench == "segmented_sum" and dtype in ["fp16"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            f"segmented_sum_{s}",
            dtype,
            partial(
                segmented_sum_benchmark, dtype=tdtype, s=s, segm_density=sp_density
            ),
            sizes,
            sp_density,
        )
    elif bench == "csr_gather":
        benchmark(device, "csr_gather", "fp16", csr_gather_benchmark, sizes)
    elif bench == "seg_scan_sc" and dtype in ["fp16"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            f"seg_scan_sc_{s}",
            "fp16",
            partial(segmented_scan_single_core_benchmark, s=s, segm_density=sp_density),
            sizes,
            sp_density,
        )
    elif bench == "mcscan" and dtype in ["fp16"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            f"mcscan_{s}",
            dtype,
            partial(mcscan_benchmark, dtype=tdtype, s=s),
            sizes,
            sp_density,
        )
    elif bench == "vec_seg_scan_sc":
        benchmark(
            device,
            f"vec_seg_scan_sc_{s}",
            "fp16",
            partial(
                vec_segmented_scan_single_core_benchmark, s=s, segm_density=sp_density
            ),
            sizes,
            sp_density,
        )
    elif bench == "scscan" and dtype in ["fp16"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            f"scscan_{s}",
            dtype,
            partial(scscan_benchmark, dtype=tdtype, s=s),
            sizes,
            sp_density,
        )
    else:
        raise RuntimeError(
            f"Unsupported benchmark setup: bench:{bench}, dtype:{dtype}, s:{s}"
        )
