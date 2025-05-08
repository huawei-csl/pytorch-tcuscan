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
from typing import Optional, Tuple

import numpy as np
import torch.nn.functional as F

import torch


def pad_to_multiple(x: torch.Tensor, s: int):
    N = x.shape[-1]
    target_size = ((N + s * s - 1) // (s * s)) * (s * s)
    pad_amount = target_size - N
    padded_x = F.pad(x, (0, pad_amount), mode="constant", value=0)
    return padded_x


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


def vadd_benchmark(device: Device, size: int) -> Tuple[float, int]:
    x = torch.rand(size, device=device.str, dtype=torch.float16)
    y = torch.rand(size, device=device.str, dtype=torch.float16)

    def run_vadd() -> None:
        _ = tcuscan_ops.run_add_custom(x, y)

    return _run_benchmark(device, run_vadd), size


def copy_benchmark(
    device: Device, size: int, s: int, dtype: torch.dtype
) -> Tuple[float, int]:
    x = torch.rand(size, device=device.str, dtype=dtype)

    def run_copy() -> None:
        _ = tcuscan_ops.run_copy(x, s)

    return _run_benchmark(device, run_copy), size


def clone_benchmark(device: Device, size: int, dtype: torch.dtype) -> Tuple[float, int]:
    if dtype in {torch.float16, torch.float32}:
        x = torch.rand(size, device=device.str, dtype=dtype)
    elif dtype == torch.int16:
        x = torch.randint(0, 2**7 - 1, (size,), device=device.str, dtype=dtype)
    else:
        raise ValueError("Incorrect copy data type")

    def run_clone() -> None:
        _ = torch.clone(x)

    return _run_benchmark(device, run_clone), size


def cast_benchmark(device: Device, size: int, dtype: torch.dtype) -> Tuple[float, int]:
    if dtype in {torch.float16}:
        x = torch.rand(size, device=device.str, dtype=dtype)
    else:
        raise ValueError("Cast benchmark only supports fp16 for now")

    def run_cast() -> None:
        _ = x.to(torch.float32)

    return _run_benchmark(device, run_cast), size


def diff_benchmark(device: Device, size: int, dtype=torch.dtype) -> Tuple[float, int]:
    x = torch.rand(size, device=device.str, dtype=dtype)

    def run_diff() -> None:
        _ = tcuscan_ops.run_diff(x)

    return _run_benchmark(device, run_diff), size


def baseline_diff_benchmark(
    device: Device, size: int, dtype=torch.dtype
) -> Tuple[float, int]:
    if dtype in [torch.float16, torch.float32]:
        x = torch.rand(size, device=device.str, dtype=dtype)
    else:
        raise ValueError("Invalid diff_cann input data type")

    def run_diff() -> None:
        _ = torch.diff(x)

    return _run_benchmark(device, run_diff), size


def baseline_diffp_benchmark(
    device: Device, size: int, dtype=torch.dtype
) -> Tuple[float, int]:
    x = torch.rand(size, device=device.str, dtype=torch.float16)
    if dtype in [torch.float16, torch.float32]:
        x = torch.rand(size, device=device.str, dtype=dtype)
    else:
        raise ValueError("Invalid diff_cann input data type")

    def run_diff() -> None:
        _ = torch.diff(x, prepend=torch.zeros(1, device=device.str))

    return _run_benchmark(device, run_diff), size


def gather_spmv_benchmark(device: Device, vec_len: int, s: int) -> Tuple[float, int]:
    rng = np.random.default_rng()
    input_values = rng.uniform(1, 100, vec_len).astype(np.float32)
    idx_len = vec_len / s
    input_cols = rng.uniform(0, vec_len, int(idx_len))
    input_cols[0] = 0
    input_cols[1] = 0
    input_cols[2] = 0
    input_cols.sort()
    input_cols = input_cols.astype(np.uint32)
    val_torch = torch.Tensor(input_values).to(torch.float32).npu()
    idx_torch = torch.from_numpy(input_cols).npu()
    outputsize = idx_len

    def run_gather_spmv() -> None:
        _ = tcuscan_ops.run_gather_spmv(val_torch, idx_torch, s)

    return _run_benchmark(device, run_gather_spmv), outputsize


def mc_gather_benchmark(device: Device, vec_len: int, s: int) -> Tuple[float, int]:
    rng = np.random.default_rng()
    input_values = rng.uniform(1, 100, vec_len).astype(np.float32)
    idx_len = vec_len / s
    input_cols = rng.uniform(0, vec_len, int(idx_len))
    input_cols.sort()
    input_cols = input_cols.astype(np.uint32)
    val_torch = torch.Tensor(input_values).to(torch.float32).npu()
    idx_torch = torch.from_numpy(input_cols).npu()
    outputsize = idx_len
    torch.npu.synchronize()

    def run_mc_gather() -> None:
        _ = tcuscan_ops.run_mc_gather(val_torch, idx_torch, s)

    return _run_benchmark(device, run_mc_gather), outputsize


def csr_gather_benchmark(device: Device, vec_len: int) -> Tuple[float, int]:
    # Maximum value of x cannot exceed 20K (UB shared memory size)
    max_x_len = 2 * 1024

    input_x = torch.rand(max_x_len, device=device.str).half()

    input_values = torch.randn(vec_len).half().npu()
    input_cols = torch.randint(
        low=0, high=max_x_len, size=(vec_len,), dtype=torch.int32
    ).npu()
    outputsize = vec_len

    def run_csr_gather() -> None:
        _ = tcuscan_ops.run_csr_gather(input_values, input_cols, input_x)

    return _run_benchmark(device, run_csr_gather), outputsize


def segmented_scan_single_core_benchmark(
    device: Device, vec_len: int, s: int, segm_density: float
) -> Tuple[float, int]:
    x = torch.randn(vec_len).half().npu()
    f = torch.empty(vec_len).uniform_(0, 1) < segm_density
    f = f.to(torch.int8).npu()
    outputsize = vec_len

    def run_seg_scan_single_core() -> None:
        _ = tcuscan_ops.run_seg_scan(x, f, s)

    return _run_benchmark(device, run_seg_scan_single_core), outputsize


def vec_segmented_scan_single_core_benchmark(
    device: Device, vec_len: int, s: int, segm_density: float
) -> Tuple[float, int]:
    x = torch.randn(vec_len).half().npu()
    f = torch.empty(vec_len).uniform_(0, 1) < segm_density
    f = f.to(torch.int8).npu()
    outputsize = vec_len

    def run_vec_seg_scan_single_core() -> None:
        _ = tcuscan_ops.run_seg_scan_vec(x, f, s)

    return _run_benchmark(device, run_vec_seg_scan_single_core), outputsize


def compress_benchmark(
    device: Device, size: int, dtype: torch.dtype, s: int, segm_density: float
) -> Tuple[float, int]:

    mask = (torch.rand(size=(size,)) < segm_density).to(torch.int8).npu()
    if dtype in {torch.float16, torch.float32}:
        x = torch.rand(size, device=device.str, dtype=dtype)
    else:
        raise RuntimeError(f"dtype {dtype} is not supported in TCUSCAN scan operator")
    outputsize = torch.sum(mask)

    def run_compress() -> None:
        _ = tcuscan_ops.run_compress(x, mask, s)

    return _run_benchmark(device, run_compress), outputsize


def segmented_sum_benchmark(
    device: Device, vec_len: int, dtype: torch.dtype, segm_density: float, s: int
) -> Tuple[float, int]:

    x = torch.randint(-100, 100, size=(vec_len,), dtype=dtype)
    x = pad_to_multiple(x, s)
    f = torch.empty(vec_len).uniform_(0, 1) < segm_density
    f = f.to(torch.int8)
    f = pad_to_multiple(f, s)
    x_npu = x.npu()
    f_npu = f.npu()
    outputsize = torch.sum(f)

    def run_seg_sum() -> None:
        _ = tcuscan_ops.run_seg_sum(x_npu, f_npu, s)

    return _run_benchmark(device, run_seg_sum), outputsize


def scscan_benchmark(
    device: Device, size: int, dtype: torch.dtype, s: int
) -> Tuple[float, int]:
    x = torch.rand(size, device=device.str, dtype=dtype)

    def run_scan() -> None:
        _ = tcuscan_ops.run_scan_single_core(x, s)

    return _run_benchmark(device, run_scan), size


def radix_sort_benchmark(device: Device, vec_len: int, dtype: torch.dtype, s: int):
    if dtype == torch.int16:
        x = (
            torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, (vec_len,))
            .to(dtype)
            .npu()
        )
    else:
        x = torch.rand((vec_len,), dtype=dtype).npu()

    def run_radix_sort() -> None:
        _, _ = tcuscan_ops.run_radix_sort(x, s)

    return _run_benchmark(device, run_radix_sort), vec_len


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

    return _run_benchmark(device, run_sort), size


def mcscan_benchmark(
    device: Device, size: int, dtype: torch.dtype, s: int
) -> Tuple[float, int]:
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

    return _run_benchmark(device, run_scan), size


def mcscan_no_l2_benchmark(
    device: Device, size: int, dtype: torch.dtype, s: int
) -> Tuple[float, int]:
    if dtype == torch.float16:
        x = torch.rand(size, device=device.str, dtype=dtype)
    elif dtype == torch.int8:
        x = torch.randint(
            0, torch.iinfo(dtype).max, (size,), device=device.str, dtype=dtype
        )
    else:
        raise RuntimeError(f"dtype {dtype} is not supported in TCUSCAN scan operator")

    def run_scan_no_l2() -> None:
        _ = tcuscan_ops.run_scan_multi_core_no_l2(x, s)

    return _run_benchmark(device, run_scan_no_l2), size


def row_scan_benchmark(
    device: Device, size: int, dtype: torch.dtype, s: int
) -> Tuple[float, int]:
    if dtype == torch.float16:
        x = torch.rand(size, device=device.str, dtype=dtype)
    else:
        raise RuntimeError(
            f"dtype {dtype} is not supported in TCUSCAN row_scan operator"
        )

    def run_row_scan() -> None:
        _ = tcuscan_ops.run_row_scan(x.reshape(-1, s), s)

    return _run_benchmark(device, run_row_scan), size


def row_scan_cce_benchmark(
    device: Device, size: int, dtype: torch.dtype, s: int
) -> Tuple[float, int]:
    if dtype == torch.float16:
        x = torch.rand(size, device=device.str, dtype=dtype)
    else:
        raise RuntimeError(
            f"dtype {dtype} is not supported in TCUSCAN row_scan operator"
        )

    U_s = torch.tril(torch.ones((s, s), dtype=dtype)).npu()

    def run_row_scan_cce() -> None:
        _ = tcuscan_ops.run_matmul_cce(x.reshape(-1, s), U_s)

    return _run_benchmark(device, run_row_scan_cce), size


def seg_scan_mc_revert_benchmark(
    device: Device, size: int, dtype: torch.dtype, segm_density: float
) -> Tuple[float, int]:

    rng = np.random.default_rng(seed=42)
    input_x = rng.integers(0, 10, size).astype(dtype)
    input_f = torch.empty(size).uniform_(0, 1) < segm_density
    input_f = input_f.numpy()
    input_f[0] = 0
    scan_x = np.cumsum(input_x).astype(np.float32)
    scan_f = np.cumsum(input_f).astype(np.int32)
    diff = np.compress(np.append(input_f[1:], 1), scan_x).astype(np.float32)

    scan_x_npu = torch.Tensor(scan_x).npu()
    scan_f_npu = torch.Tensor(scan_f).to(torch.int32).npu()
    diff_npu = torch.Tensor(diff).npu()

    assert scan_x_npu.dtype == torch.float32
    assert scan_f_npu.dtype == torch.int32
    assert diff_npu.dtype == torch.float32

    def run_revert() -> None:
        _ = tcuscan_ops.run_seg_scan_mc_revert(scan_x_npu, scan_f_npu, diff_npu)

    # TODO: Here we abuse of outputsize param. Outputsize = Inputsize, but len(diff) is needed
    # for bandwidth measurements
    return _run_benchmark(device, run_revert), len(diff_npu)


def topk_benchmark(device: Device, size: int, dtype: torch.dtype, k: int) -> float:
    """
    Benchmark topk kernel.

    Args:
        device: Device to run benchmark on.
        size: Size of the arrays to use.
        dtype: Data type of the input/output arrays.
        k: topk parameter.

    Returns:
        Average time in microseconds.
    """
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

    return _run_benchmark(device, run_topk), k


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
    with open(
        f"bench_results_{op_name}_{dtype}.csv",
        "w",
        encoding="UTF-8",
    ) as fd:

        fd.write("operator,dtype,size,density,outputsize,time_us\n")

        for size in sizes:
            time, outputsize = fn(device, size)
            fd.write(f"{op_name},{dtype},{size},{density},{outputsize},{time:.2f}\n")
            logger.info(
                f"OP:{op_name}, dtype: {dtype}, size: {size:}, outputsize: {outputsize}, density: {density}, device: {device.str}"
            )


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
            "mcscan_no_l2",
            "compress",
            "segmented_sum",
            "custom_copy",
            "vec_seg_scan_sc",
            "scscan",
            "seg_scan_mc_revert",
            "topk",
            "mcgather",
            "gather_spmv",
            "sort",
            "radix_sort",
            "row_scan",
            "row_scan_cce",
            "cast",
        ],
    )
    parser.add_argument("--dtype", choices=["int8", "fp16", "int16", "int32", "fp32"])
    parser.add_argument("--s", type=int, default=64, required=False)
    parser.add_argument("--k", type=int, default=256, required=False)
    parser.add_argument("--max_size", type=int, default=1e8, required=False)
    parser.add_argument("--num_cores", type=int, default=20, required=False)
    parser.add_argument("--density", type=float, default=None, required=False)
    args = parser.parse_args()

    bench = args.bench
    dtype = args.dtype
    max_size = args.max_size
    num_cores = args.num_cores
    s = args.s
    density = args.density

    if DEVICE == "npu":
        device = Device(torch.npu, NPU_DEVICE)
    elif DEVICE == "cpu":
        device = Device(torch, "cpu")
    else:
        device = Device(torch.cuda, "cuda:0")

    # Maximum number of iterations
    max_iters = ceil(max_size / (num_cores * s * s))

    logger.info("*******************************")
    logger.info(f"* bench          : {bench}")
    logger.info(f"* dtype          : {dtype}")
    logger.info(f"* max_size       : {max_size}")
    logger.info(f"* Max iterations : {max_iters}")
    logger.info(f"* num_cores      : {num_cores}")
    logger.info(f"* s              : {s}")
    logger.info(f"* density        : {density}")
    logger.info(f"* device         : {device.str}")
    logger.info("*******************************")
    logger.info("*******************************")

    # Input sizes to benchmark
    sizes = [i * num_cores * s * s for i in range(1, max_iters, 16 * 128 // s)]

    if bench == "vadd":
        benchmark(device, "vadd", "fp16", vadd_benchmark, sizes)
    elif bench == "copy" and dtype in ["int16", "fp16", "fp32"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            "copy",
            dtype,
            partial(clone_benchmark, dtype=tdtype),
            sizes,
            density,
        )
    elif bench == "cast" and dtype in ["fp16"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            "cast",
            dtype,
            partial(cast_benchmark, dtype=tdtype),
            sizes,
            density,
        )
    elif bench == "seg_scan_mc_revert" and dtype in ["fp32"]:
        benchmark(
            device,
            f"seg_scan_mc_revert_{density}",
            dtype,
            partial(
                seg_scan_mc_revert_benchmark, dtype=np.float32, segm_density=density
            ),
            sizes,
            density,
        )
    elif bench == "custom_copy" and dtype in ["fp32", "fp16"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            f"custom_copy_{s}",
            dtype,
            partial(copy_benchmark, s=s, dtype=tdtype),
            sizes,
            density,
        )
    elif bench == "diff_cann" and dtype in ["fp16", "fp32"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            "diff_cann",
            dtype,
            partial(baseline_diff_benchmark, dtype=tdtype),
            sizes,
            density,
        )
    elif bench == "diffp_cann" and dtype in ["fp16", "fp32"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            "diffp_cann",
            dtype,
            partial(baseline_diffp_benchmark, dtype=tdtype),
            sizes,
            density,
        )
    elif bench == "diff" and dtype in ["fp16", "fp32"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            "diff",
            dtype,
            partial(diff_benchmark, dtype=tdtype),
            sizes,
            density,
        )
    elif bench == "compress" and dtype in ["fp16", "fp32"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            f"compress_{s}_{density}",
            dtype,
            partial(compress_benchmark, dtype=tdtype, s=s, segm_density=density),
            sizes,
            density,
        )
    elif bench == "segmented_sum" and dtype in ["fp16"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            f"segmented_sum_{s}_{density}",
            dtype,
            partial(segmented_sum_benchmark, dtype=tdtype, s=s, segm_density=density),
            sizes,
            density,
        )
    elif bench == "csr_gather":
        benchmark(device, "csr_gather", "fp16", csr_gather_benchmark, sizes)
    elif bench == "seg_scan_sc" and dtype in ["fp16"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            f"seg_scan_sc_{s}_{density}",
            "fp16",
            partial(segmented_scan_single_core_benchmark, s=s, segm_density=density),
            sizes,
            density,
        )
    elif bench == "mcgather":
        benchmark(
            device,
            f"mcgather_{s}",
            dtype,
            partial(mc_gather_benchmark, s=s),
            sizes,
        )
    elif bench == "gather_spmv":
        benchmark(
            device,
            f"gather_spmv_{s}",
            dtype,
            partial(mc_gather_benchmark, s=s),
            sizes,
        )
    elif bench == "mcscan" and dtype in ["fp16", "int8"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            f"mcscan_{s}",
            dtype,
            partial(mcscan_benchmark, dtype=tdtype, s=s),
            sizes,
            density,
        )
    elif bench == "mcscan_no_l2" and dtype in ["fp16", "int8"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            f"mcscan_no_l2_{s}",
            dtype,
            partial(mcscan_no_l2_benchmark, dtype=tdtype, s=s),
            sizes,
            density,
        )
    elif bench == "row_scan" and dtype in ["fp16"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            f"row_scan_{s}",
            dtype,
            partial(row_scan_benchmark, dtype=tdtype, s=s),
            sizes,
            density,
        )
    elif bench == "row_scan_cce" and dtype in ["fp16"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            "row_scan_cce_512",
            dtype,
            partial(row_scan_cce_benchmark, dtype=tdtype, s=512),
            sizes,
            density,
        )
    elif bench == "vec_seg_scan_sc":
        benchmark(
            device,
            f"vec_seg_scan_sc_{s}_{density}",
            "fp16",
            partial(
                vec_segmented_scan_single_core_benchmark, s=s, segm_density=density
            ),
            sizes,
            density,
        )
    elif bench == "topk":
        k = args.k
        assert dtype in [
            "int16",
            "fp16",
            "int32",
        ], "TCUSCAN topk only works for dtype 'int16', 'int32'"

        tdtype = torch.float16
        if dtype == "int16":
            tdtype = torch.int16
        elif dtype == "int32":
            tdtype = torch.int32

        benchmark(
            device,
            f"topk_{k}",
            dtype,
            partial(topk_benchmark, dtype=tdtype, k=k),
            sizes,
        )

    elif bench == "scscan" and dtype in ["fp16"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            f"scscan_{s}",
            dtype,
            partial(scscan_benchmark, dtype=tdtype, s=s),
            sizes,
            density,
        )
    elif bench == "sort" and dtype in ["int16", "fp16"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(device, "sort", dtype, partial(sort_benchmark, dtype=tdtype), sizes)
    elif bench == "radix_sort" and dtype in ["int16", "fp16"]:
        tdtype = STR_TO_DTYPE[dtype]
        benchmark(
            device,
            f"radix_sort_{s}",
            dtype,
            partial(radix_sort_benchmark, dtype=tdtype, s=s),
            sizes,
            density,
        )
    else:
        raise RuntimeError(
            f"Unsupported benchmark setup: bench:{bench}, dtype:{dtype}, s:{s}"
        )
