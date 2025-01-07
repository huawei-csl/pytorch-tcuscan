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
from math import ceil

import torch

file_handler = logging.FileHandler(filename="torch_profiler.log")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=handlers,
)

logger = logging.getLogger(__name__)


DEVICE = os.environ.get("DEVICE_TYPE", "npu")
if DEVICE == "npu":
    import torch_npu

    import add_custom

    torch.npu.config.allow_internal_format = False

    NPU_DEVICE = "npu:1"
    torch.npu.config.allow_internal_format = False
    torch.npu.set_device(NPU_DEVICE)
    assert torch.npu.is_available()

WARMUP_ITERS = 10
BENCH_ITERS = 100

STR_TO_DTYPE = {"fp16": torch.float16, "int16": torch.int16, "int8": torch.int8}


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
        z = add_custom.run_add_custom(x, y)

    return _run_benchmark(device, run_vadd)


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
        fd.write("operator,dtype,size,time_us\n")
        for size in sizes:
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
        ],
    )
    parser.add_argument("--dtype", choices=["int8", "fp16", "int16"])
    parser.add_argument("--s", type=int, default=64, required=False)
    parser.add_argument("--max_size", type=int, default=1e8, required=False)
    parser.add_argument("--num_cores", type=int, default=20, required=False)
    args = parser.parse_args()

    bench = args.bench
    dtype = args.dtype
    max_size = args.max_size
    num_cores = args.num_cores
    s = args.s

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
    else:
        raise RuntimeError(
            f"Unsupported benchmark setup: bench:{bench}, dtype:{dtype}, s:{s}"
        )
