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
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import random


def power_law_rvs(shape, exponent=2.0):
    u = np.random.uniform(0, 1, size=shape)
    return u ** (-1.0 / (exponent - 1))


def uniform_rvs(shape):
    return np.random.uniform(0, 1, size=shape)


def pad_to_multiple(x: torch.Tensor, s: int):
    N = x.shape[-1]
    target_size = ((N + s * s - 1) // (s * s)) * (s * s)
    pad_amount = target_size - N
    padded_x = F.pad(x, (0, pad_amount), mode="constant", value=0)
    return padded_x


DEVICE = os.environ.get("DEVICE_TYPE", "npu")
if DEVICE == "npu":
    import torch_npu  # noqa

    NPU_DEVICE = "npu:0"
    torch.npu.config.allow_internal_format = False
    torch.npu.set_device(NPU_DEVICE)
    assert torch.npu.is_available()

import tcuscan_ops  # noqa

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
    x_npu = x.npu()

    def run_diff() -> None:
        _ = torch.diff(x_npu)

    return _run_benchmark(device, run_diff)


def copy_benchmark(device: Device, x: torch.Tensor, s: int) -> float:
    """
    Benchmark torch.clone kernel.

    Args:
        device: Device to run benchmark on.
        size: Size of the arrays to use.
        dtype: Data type of the input array.

    Returns:
        Average time in microseconds.
    """
    x = x.npu()

    def run_copy() -> None:
        _ = tcuscan_ops.run_copy(x, s)

    return _run_benchmark(device, run_copy)


def benchmark(  # noqa
    device: Device,
    op_name: str,
    dtype: str,
    fn: typing.Callable,
    size: int,
    density: Optional[float],
    nnr: int,
    distr: str,
) -> None:
    """
    Benchmark a given function.

    Args:
        device: Device to run benchmark on.
        op_name: Operator name.
        dtype: Input data type.
        fn: Function to benchmark.
        size: total number of elements, aka nnz.
        benchname: name of the benchmark used.
        density: employed density to find nnz.
        nnr: number of rows.
        distr: employed probability distribution.
    """
    if density is not None:
        density_str = "density,"
    else:
        density_str = ""
    filename = f"random_matrices_{distr}_{op_name}_{dtype}_{ '' if (density is None) else str(density)}.csv"
    with open(
        filename,
        "a",
        encoding="UTF-8",
    ) as fd:
        global once
        if once is True:
            fd.write(f"operator,dtype,size,nrow,{density_str}time_us\n")
            once = False
        time = fn(device)
        op_name = f"{op_name}" + f"{'' if (density is None) else '_' + str(density)}"
        fd.write(
            f"{op_name},{dtype},{size},{nnr},{'' if (density is None) else str(density)+','}{time:.2f}\n"
        )
        logger.info(
            f" OP:{op_name}, dtype: {dtype}, device: {device.str}, density: { None if (density is None) else str(density)} size: {size}"
        )
    fd.close()


once = True

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
            "custom_copy",
        ],
    )
    parser.add_argument("--dtype", choices=["int8", "fp16", "int16", "fp32"])
    parser.add_argument("--s", type=int, default=64, required=False)
    parser.add_argument("--max_size", type=int, default=1e8, required=False)
    parser.add_argument("--num_cores", type=int, default=20, required=False)
    parser.add_argument("--density", type=float, default=1)
    parser.add_argument(
        "--prob", type=str, choices=["PowerLaw", "Uniform"], default="Uniform"
    )

    args = parser.parse_args()

    distr = args.prob
    dtype = args.dtype
    bench = args.bench
    max_size = args.max_size
    num_cores = args.num_cores
    s = args.s
    density = args.density
    tdtype = STR_TO_DTYPE[dtype]

    if DEVICE == "npu":
        device = Device(torch.npu, NPU_DEVICE)
    elif DEVICE == "cpu":
        device = Device(torch, "cpu")
    else:
        device = Device(torch.cuda, "cuda:0")

    for nnr in range(10000, 50000, 1000):
        vec_len = nnr * nnr * density
        B = []
        if "Uniform" == distr:
            B = random(
                nnr,
                nnr,
                density=density,
                format="csr",
                dtype=np.float32,
                data_rvs=uniform_rvs,
            )

        elif "PowerLaw" == distr:
            B = random(
                nnr,
                nnr,
                density=density,
                format="csr",
                dtype=np.float32,
                data_rvs=lambda shape: power_law_rvs(shape, exponent=2.5),
            )

        my_x = torch.tensor(B.data).to(torch.float16)
        f = np.zeros(B.nnz + 1)
        f[B.indptr] = 1
        my_f = f[:-1]
        my_f = torch.tensor(my_f).to(torch.int8)

        if bench == "seg_scan_sc" and dtype in ["fp16"]:
            my_x = pad_to_multiple(my_x, s)
            my_f = pad_to_multiple(my_f, s)
            benchmark(
                device,
                f"seg_scan_sc_{s}",
                dtype,
                partial(segmented_scan_single_core_benchmark, x=my_x, f=my_f, s=s),
                vec_len,
                density,
                nnr,
                distr,
            )
        elif bench == "custom_copy" and dtype in ["fp32", "fp16"]:
            tdtype = STR_TO_DTYPE[dtype]
            benchmark(
                device,
                f"custom_copy_{s}",
                dtype,
                partial(copy_benchmark, x=my_x, s=s),
                vec_len,
                density,
                nnr,
                distr,
            )
        elif bench == "vec_seg_scan_sc" and dtype in ["fp16"]:
            benchmark(
                device,
                f"vec_seg_scan_sc_{s}",
                dtype,
                partial(vec_segmented_scan_single_core_benchmark, x=my_x, f=my_f, s=s),
                vec_len,
                density,
                nnr,
                distr,
            )
        elif bench == "compress" and dtype in ["fp16", "fp32"]:
            benchmark(
                device,
                f"compress_{s}",
                dtype,
                partial(compress_benchmark, x=my_x, f=my_f, s=s),
                vec_len,
                density,
                nnr,
                distr,
            )
        elif bench == "mcscan" and dtype in ["fp16"]:
            benchmark(
                device,
                f"mcscan_{s}",
                dtype,
                partial(mcscan_benchmark, x=my_x, s=s),
                vec_len,
                density,
                nnr,
                distr,
            )
        elif bench == "diff" and dtype in ["fp16", "fp32"]:
            my_x = pad_to_multiple(my_x, s)
            my_f = pad_to_multiple(my_f, s)
            benchmark(
                device,
                "diff",
                dtype,
                partial(baseline_diff_benchmark, x=my_x),
                vec_len,
                density,
                nnr,
                distr,
            )
        else:
            raise RuntimeError(
                f"Unsupported benchmark setup: bench:{bench}, dtype:{dtype}, s:{s}"
            )
