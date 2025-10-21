#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2024 Huawei Technologies Co., Ltd
# isort: skip_file

import argparse
import logging
import sys
import time
import types
import typing
from dataclasses import dataclass
from functools import partial
from math import ceil
from typing import Tuple

import numpy as np

import torch
import tcuscan_ops  # noqa : needed for tcuscan_ops.run_scan_cpu


file_handler = logging.FileHandler(filename="torch_profiler_cpu.log")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=handlers,
)

logger = logging.getLogger(__name__)


WARMUP_ITERS = 10
BENCH_ITERS = 100

STR_TO_DTYPE = {
    "fp16": torch.float16,
    "int16": torch.int16,
    "int8": torch.int8,
    "int32": torch.int32,
    "fp32": torch.float32,
}


# Input array size to benchmark sort
_SORT_SIZES = [
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1179648,
    4325376,
    16908288,
    25165824,
]

# Compress has same benchmark sizes as sort
_COMPRESS_SIZES = _SORT_SIZES


@dataclass
class Device:
    module: types.ModuleType
    str: str

    def sync(self) -> None:
        self.module.synchronize()

    def event(self) -> "typing.Self.module.Event":
        return self.module.Event(enable_timing=True)


def _run_benchmark(
    fn: typing.Callable,
    benchmark_iters: int = BENCH_ITERS,
) -> float:
    """
    Benchmark a given function with warmup.

    Args:
        fn: Function to benchmark.
        benchmark_iters: Number of benchmark runs.

    Returns:
        Average time in microseconds.
    """

    start_times = np.zeros(benchmark_iters)
    end_times = np.zeros(benchmark_iters)

    cache_size = 512 * 1024 * 1024
    cache = torch.ones(cache_size, dtype=torch.int8)

    for i in range(benchmark_iters):
        cache.zero_()
        start_times[i] = time.perf_counter_ns()
        fn()
        end_times[i] = time.perf_counter_ns()

    times_ns = [e - s for s, e in zip(start_times, end_times)]
    avg_time_ns = sum(times_ns) / len(times_ns)
    avg_time_us = avg_time_ns * 0.001
    return avg_time_us


def vadd_benchmark(device: Device, size: int) -> float:
    x = torch.rand(size, device=device.str, dtype=torch.float16)
    y = torch.rand(size, device=device.str, dtype=torch.float16)
    z = torch.zeros(size, device=device.str, dtype=torch.float16)

    def run_vadd() -> None:
        _ = torch.add(x, y, out=z)

    return _run_benchmark(run_vadd)


def clone_benchmark(device: Device, size: int, dtype: torch.dtype) -> float:
    if dtype in {torch.float16, torch.float32}:
        x = torch.rand(size, device=device.str, dtype=dtype)
    elif dtype == torch.int16:
        x = torch.randint(0, 2**7 - 1, (size,), device=device.str, dtype=dtype)
    else:
        raise ValueError("Incorrect copy data type")

    def run_clone() -> None:
        _ = torch.clone(x)

    return _run_benchmark(run_clone)


def cast_benchmark(device: Device, size: int, dtype: torch.dtype) -> float:
    if dtype in {torch.float16}:
        x = torch.rand(size, device=device.str, dtype=dtype)
    else:
        raise ValueError("Cast benchmark only supports fp16 for now")

    def run_cast() -> None:
        _ = x.to(torch.float32)

    return _run_benchmark(run_cast)


def scan_benchmark(device: Device, size: int, dtype: torch.dtype) -> float:
    if dtype in {torch.float16, torch.float32}:
        out_dtype = torch.float32
        x = torch.rand(size, device=device.str, dtype=dtype)
    elif dtype == torch.int8:
        out_dtype = torch.int32
        x = torch.randint(0, 2**7 - 1, (size,), device=device.str, dtype=dtype)
    else:
        raise ValueError("Incorrect scan data type")

    y = torch.zeros(size, device=device.str, dtype=out_dtype)  # noqa

    def run_scan() -> None:
        _ = tcuscan_ops.run_scan_cpu(x)
        # _ = torch.cumsum(x, dim=-1, dtype=out_dtype, out=y)

    return _run_benchmark(run_scan)


def sort_benchmark(device: Device, size: int, dtype: torch.dtype) -> float:
    if dtype in {torch.float16, torch.float32}:
        x = torch.rand(size, device=device.str, dtype=dtype)
    else:
        x = torch.randint(
            0, torch.iinfo(dtype).max, (size,), device=device.str, dtype=dtype
        )
    out = torch.zeros(size, device=device.str, dtype=dtype)
    ind = torch.zeros(size, device=device.str, dtype=torch.int64)

    def run_sort() -> None:
        torch.sort(x, dim=-1, descending=False, out=(out, ind))

    return _run_benchmark(run_sort)


def topk_benchmark(device: Device, size: int, dtype: torch.dtype, k: int) -> float:
    if dtype in {torch.float16, torch.float32}:
        x = torch.rand(size, device=device.str, dtype=dtype)
    else:
        x = torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            (size,),
            device=device.str,
            dtype=dtype,
        )

    def run_topk() -> None:
        _, _ = torch.topk(x, k)

    return _run_benchmark(run_topk)


def scan_batch_benchmark(
    device: Device, shape: Tuple[int, int], dtype: torch.dtype
) -> float:
    if dtype in {torch.int8, torch.float16}:
        x = torch.rand(shape, device=device.str, dtype=dtype)
    else:
        raise RuntimeError(
            f"dtype {dtype} is not supported in TCUSCAN batch scan operator"
        )

    def run_scan() -> None:
        _ = torch.cumsum(x, dim=1, dtype=torch.float)

    return _run_benchmark(run_scan)


def masked_select_benchmark(device: Device, size: int, dtype: torch.dtype) -> float:
    mask = (torch.randn(size, device=device.str) > 0).to(torch.bool)
    if dtype == torch.int16:
        x = torch.randint(0, 2**7 - 1, (size,), device=device.str).to(torch.int16)
    elif dtype == torch.float16:
        x = torch.rand(size, device=device.str, dtype=dtype)

    else:
        raise RuntimeError(
            f"dtype {dtype} is not supported in TCUSCAN masked_select operator"
        )

    def run_masked_select() -> None:
        _ = torch.masked_select(x, mask)

    return _run_benchmark(run_masked_select)


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def top_p_benchmark(device: Device, size: int, dtype: torch.dtype) -> float:
    """
    Benchmark top-p.

    Args:
        device: Device to run benchmark on.
        size: Size of the arrays to use.
        dtype: Data type of the input/output arrays.

    Returns:
        Average time in microseconds.
    """
    threshold = 0.1
    if dtype == torch.float16:
        probs = torch.rand(size, device=device.str, dtype=dtype)
    else:
        raise RuntimeError(f"dtype {dtype} is not supported in sample_top_p operator")

    def run_top_p() -> None:
        _ = sample_top_p(probs, threshold)

    return _run_benchmark(run_top_p)


def hist_benchmark(device: Device, size: int, dtype: torch.dtype) -> float:
    if dtype in {torch.float32}:
        x = torch.rand(size, device=device.str, dtype=dtype)
    else:
        raise ValueError("histogram benchmark only supports fp32 for now")

    def run_hist() -> None:
        _ = torch.histogram(x, bins=20)

    return _run_benchmark(run_hist)


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
    with open(
        f"bench_results_{op_name}_{device.str}_{dtype}.csv", "w", encoding="utf-8"
    ) as fd:
        fd.write("operator,dtype,size,time_us\n")
        for size in sizes:
            logger.info(
                f"OP:{op_name}, dtype: {dtype}, size: {size:,}, device: {device.str}"
            )
            time = fn(device, size)
            fd.write(f"{op_name},{dtype},{size},{time:.2f}\n")


def abs_benchmark(device: Device, size: int, dtype: torch.dtype) -> float:
    """
    Benchmark torch.abs kernel.

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
        raise ValueError("Incorrect scan data type")

    def run_abs() -> None:
        _ = torch.abs(x)

    return _run_benchmark(run_abs)


def multinomial_benchmark(device: Device, size: int, dtype: torch.dtype) -> float:
    """
    Benchmark torch.multinomial kernel.

    Args:
        device: Device to run benchmark on.
        size: Size of the arrays to use.
        dtype: Data type of the input array.

    Returns:
        Average time in microseconds.
    """

    probs = torch.rand((size,), device=device.str, dtype=dtype)
    probs = torch.exp(probs)

    def run_multinomial() -> None:
        _ = torch.multinomial(probs, num_samples=1)

    return _run_benchmark(run_multinomial)


def benchmark_batch(
    device: Device,
    op_name: str,
    dtype: str,
    fn: typing.Callable,
    shapes: typing.List[typing.Tuple[int, int]],
) -> None:
    """
    Benchmark a given function for batched operators.

    Args:
        device: Device to run benchmark on.
        op_name: Operator name.
        dtype: Input data type.
        fn: Function to benchmark.
        shapes: List of shapes to benchmark.
    """
    with open(f"bench_results_{op_name}_{dtype}.csv", "w", encoding="utf-8") as fd:
        fd.write("operator,dtype,batch_size,size,time_us\n")
        for shape in shapes:
            batch_size, size = shape
            logger.info(
                f"op_name:{op_name}, batch_size: {batch_size}, size: {size:,}, device: {device.str}"
            )
            time = fn(device, shape)
            fd.write(f"{op_name},{dtype},{batch_size},{size},{time:.2f}\n")


if __name__ == "__main__":  # noqa
    parser = argparse.ArgumentParser(
        prog="torch_profile", description="Profiler for torch with CPU backend"
    )

    parser.add_argument(
        "--bench",
        choices=[
            "vadd",
            "copy",
            "cast",
            "scan",
            "scan_batch",
            "sort",
            "masked_select",
            "top_p",
            "abs",
            "multinomial",
            "topk",
            "hist",
        ],
    )
    parser.add_argument("--dtype", choices=["int8", "fp16", "int16", "int32", "fp32"])
    parser.add_argument("--s", type=int, default=64, required=False)
    parser.add_argument("--k", type=int, default=4096, required=False)
    parser.add_argument("--max_size", type=int, default=1e8, required=False)
    parser.add_argument("--num_cores", type=int, default=20, required=False)
    args = parser.parse_args()

    bench = args.bench
    dtype = args.dtype
    max_size = args.max_size
    num_cores = args.num_cores
    s = args.s

    device = Device(torch, "cpu")

    # Maximum number of iterations
    max_iters = ceil(max_size / (num_cores * s * s))

    # Input sizes to benchmark
    sizes = [i * num_cores * s * s for i in range(1, max_iters, 16 * 128 // s)]

    tdtype = STR_TO_DTYPE[dtype]

    if bench == "scan" and dtype in ["int8", "fp16", "fp32"]:
        benchmark(
            device,
            "scan",
            dtype,
            partial(scan_benchmark, dtype=tdtype),
            sizes,
        )
    elif bench == "copy" and dtype in ["int16", "fp16", "fp32"]:
        benchmark(
            device,
            "copy",
            dtype,
            partial(clone_benchmark, dtype=tdtype),
            sizes,
        )
    elif bench == "cast" and dtype in ["fp16"]:
        benchmark(
            device,
            "cast",
            dtype,
            partial(cast_benchmark, dtype=tdtype),
            sizes,
        )
    elif bench == "hist" and dtype in ["fp32"]:
        benchmark(
            device,
            "hist",
            dtype,
            partial(hist_benchmark, dtype=tdtype),
            sizes,
        )
    elif bench == "abs" and dtype in ["int16", "fp16"]:
        benchmark(
            device,
            "abs",
            dtype,
            partial(abs_benchmark, dtype=tdtype),
            sizes,
        )
    elif bench == "multinomial" and dtype in ["fp16"]:
        # RuntimeError: call aclnnMultinomial failed, detail:EZ1001: number of categories cannot exceed 2^24
        benchmark(
            device,
            "multinomial",
            dtype,
            partial(multinomial_benchmark, dtype=tdtype),
            filter(lambda x: x < 2**24, sizes),
        )
    elif bench == "sort":
        assert dtype in [
            "int16",
            "fp16",
        ], "TCUSCAN sort only works for dtype 'int16' and positive 'fp16'"

        benchmark(device, "sort", dtype, partial(sort_benchmark, dtype=tdtype), sizes)
    elif bench == "topk":
        k = args.k
        assert dtype in [
            "int16",
            "fp16",
        ], "TCUSCAN topk only works for dtype 'int16' and positive 'fp16'"

        benchmark(
            device,
            f"topk_{k}",
            dtype,
            partial(topk_benchmark, dtype=tdtype, k=k),
            sizes,
        )
    elif bench.endswith("_batch") and dtype == "fp16":

        # Llama3 top-p sampler size. Corresponds to the vocabulary size
        llama_size = 65500
        # Override batch sizes
        shapes = []

        for batch_size in range(20, 256, 20):
            size = ceil(llama_size / (s * s)) * (s**2)
            shapes.append((batch_size, size))

        if bench == "scan_batch":
            benchmark_batch(
                device,
                "scan_batch",
                "fp16",
                partial(scan_batch_benchmark, dtype=torch.float16),
                shapes,
            )
    elif bench == "masked_select" and dtype in ["int16", "fp16"]:
        benchmark(
            device,
            "masked_select",
            dtype,
            partial(masked_select_benchmark, dtype=tdtype),
            _COMPRESS_SIZES,
        )
    elif bench == "top_p" and dtype in ["fp16"]:
        benchmark(
            device,
            "top_p",
            dtype,
            partial(top_p_benchmark, dtype=tdtype),
            filter(lambda x: x < 2**24, sizes),
        )
    elif bench == "vadd" and dtype in ["fp16", "fp32"]:
        benchmark(device, "vadd", dtype, vadd_benchmark, sizes)
    else:
        raise RuntimeError(
            f"Unsupported benchmark setup: bench:{bench}, dtype:{dtype}, s:{s}"
        )
