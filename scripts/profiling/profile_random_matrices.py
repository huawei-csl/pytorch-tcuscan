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
from typing import Optional

import numpy as np
import torch.nn.functional as F
from scipy.sparse import csr_matrix, random

import torch


def power_law_rvs(shape, exponent=2.0):
    return np.random.power(exponent, size=shape)


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

    NPU_DEVICE = "npu:1"
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


def gather_spmv_benchmark(device: Device, x: torch.Tensor, idx: torch.Tensor, s: int):
    x_npu = x.npu()
    idx_npu = idx.npu()

    def run_gather_spmv() -> None:
        _ = tcuscan_ops.run_gather_spmv(x_npu, idx_npu, s)

    return _run_benchmark(device, run_gather_spmv)


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


def spmv_benchmark(
    device: Device,
    B: csr_matrix,
    s: int,
):
    """
    Baseline for SPMV using different python calls.

    Args:
        device: Device to run benchmark on.
        B: CSR random matrix
        S: Tiling Size for the matrix unit, used for mcscan

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

    def run_spmv():
        _ = tcuscan_ops.run_spmv(vals_npu, idx_npu, col_npu, vec_npu, s)

    return _run_benchmark(device, run_spmv)


def spmv_v2_benchmark(
    device: Device,
    B: csr_matrix,
    s: int,
):
    """
    Baseline for SPMV v2 (segmented-sum based) kernel.

    Args:
        device: Device to run benchmark on.
        B: CSR random matrix
        s: Tiling Size for the matrix unit, used for mcscan

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

    def run_spmv_v2():
        _ = tcuscan_ops.run_spmv_v2(vals_npu, idx_npu, col_npu, vec_npu, s)

    return _run_benchmark(device, run_spmv_v2)


def spmv_multi_cube_benchmark(
    device: Device,
    B: csr_matrix,
    s: int,
):
    """
    Baseline for the multi-cube SPMV kernel.

    Args:
        device: Device to run benchmark on.
        B: CSR random matrix
        s: Tiling Size for the matrix unit, used for mcscan

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

    def run_spmv_multi_cube():
        _ = tcuscan_ops.run_spmv_multi_cube(
            vals_npu, idx_npu, col_npu, vec_npu, upper, lower_strict
        )

    return _run_benchmark(device, run_spmv_multi_cube)


def spmv_v2_multi_cube_benchmark(
    device: Device,
    B: csr_matrix,
    s: int,
):
    """
    Baseline for the SpMV v2 multi-cube kernel (segmented-sum based).

    Args:
        device: Device to run benchmark on.
        B: CSR random matrix
        s: Tiling Size for the matrix unit, used for mcscan

    Returns:
        Average time in microseconds.
    """
    rng = np.random.default_rng(seed=42)
    vals = torch.from_numpy((B.data).astype(np.float16))
    idx = torch.from_numpy((B.indptr).astype(np.int32))
    cols = torch.from_numpy((B.indices).astype(np.int32))
    vector = torch.from_numpy(rng.uniform(1, 9, len(idx) - 1).astype(np.float16))
    vals_npu = vals.npu()
    idx_npu = idx.npu()
    col_npu = cols.npu()
    vec_npu = vector.npu()
    ones = torch.ones((s, s), dtype=torch.float16, device=NPU_DEVICE)
    upper = torch.triu(ones)
    lower_strict = torch.tril(ones, -1)

    def run_spmv_v2_multi_cube():
        _ = tcuscan_ops.run_spmv_v2_multi_cube(
            vals_npu, idx_npu, col_npu, vec_npu, upper, lower_strict
        )

    return _run_benchmark(device, run_spmv_v2_multi_cube)


def benchmark(  # noqa
    device: Device,
    op_name: str,
    dtype: str,
    fn: typing.Callable,
    size: int,
    density: Optional[float],
    nnr: int,
    distr: str,
    alpha: float,
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

    if distr == "PowerLaw":
        alpha_str = "alpha,"
    else:
        alpha_str = ""
    filename = f"random_matrices_{distr}_{op_name}_{dtype}_{ '' if (density is None) else str(density)}.csv"
    with open(
        filename,
        "a",
        encoding="UTF-8",
    ) as fd:
        global once
        if once is True:
            fd.write(f"operator,dtype,size,nrow,{alpha_str}{density_str}time_us\n")
            once = False
        time = fn(device)
        op_name = f"{op_name}" + f"{'' if (density is None) else '_' + str(density)}"
        density_str = "" if (density is None) else f"{density},"
        dist_str = "" if (distr == "Uniform") else f"{alpha},"
        fd.write(f"{op_name},{dtype},{size},{nnr},{dist_str}{density_str}{time:.2f}\n")
        logger.info(
            f" OP:{op_name}, dtype: {dtype}, device: {device.str}, density: {density} size: {size}"
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
            "gather_spmv",
            "spmv",
            "spmv_v2",
            "spmv_multi_cube",
            "spmv_v2_multi_cube",
        ],
    )
    parser.add_argument("--dtype", choices=["int8", "fp16", "int16", "fp32"])
    parser.add_argument("--s", type=int, default=64, required=False)
    parser.add_argument("--max_size", type=int, default=1e8, required=False)
    parser.add_argument("--num_cores", type=int, default=20, required=False)
    parser.add_argument("--density", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument(
        "--prob", type=str, choices=["PowerLaw", "Uniform"], default="Uniform"
    )

    args = parser.parse_args()
    alpha = args.alpha
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
    # 71680 is the maximum supported nnr for spmv
    for nnr in range(s * num_cores, 71680, s * 20):
        vec_len = nnr * nnr * density
        B = []
        if "Uniform" == distr:
            B = random(
                nnr - 1,
                nnr - 1,
                density=density,
                format="csr",
                dtype=np.float32,
                data_rvs=uniform_rvs,
            )

        elif "PowerLaw" == distr:
            B = random(
                nnr - 1,
                nnr - 1,
                density=density,
                format="csr",
                dtype=np.float32,
                data_rvs=lambda shape: power_law_rvs(shape, exponent=alpha),
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
                alpha,
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
                alpha,
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
                alpha,
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
                alpha,
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
                alpha,
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
                alpha,
            )
        elif bench == "gather_spmv" and dtype in ["fp32"]:
            values = (B.data).astype(np.float32)
            indexes = (B.indptr).astype(np.uint32)
            benchmark(
                device,
                f"gather_spmv_{s}",
                dtype,
                partial(
                    gather_spmv_benchmark,
                    x=torch.from_numpy(values),
                    idx=torch.from_numpy(indexes),
                    s=s,
                ),
                len(values),
                density,
                nnr,
                distr,
                alpha,
            )
        elif bench == "spmv" and dtype in ["fp16"]:
            benchmark(
                device,
                f"spmv_{alpha}_{s}",
                dtype,
                partial(
                    spmv_benchmark,
                    B=B,
                    s=s,
                ),
                len(B.data),
                density,
                nnr,
                distr,
                alpha,
            )
        elif bench == "spmv_v2" and dtype in ["fp16"]:
            benchmark(
                device,
                f"spmv_v2_{alpha}_{s}",
                dtype,
                partial(
                    spmv_v2_benchmark,
                    B=B,
                    s=s,
                ),
                len(B.data),
                density,
                nnr,
                distr,
                alpha,
            )
        elif bench == "spmv_multi_cube" and dtype in ["fp16"]:
            benchmark(
                device,
                f"spmv_multi_cube_{alpha}_{s}",
                dtype,
                partial(
                    spmv_multi_cube_benchmark,
                    B=B,
                    s=s,
                ),
                len(B.data),
                density,
                nnr,
                distr,
                alpha,
            )
        elif bench == "spmv_v2_multi_cube" and dtype in ["fp16"]:
            benchmark(
                device,
                f"spmv_v2_multi_cube_{alpha}_{s}",
                dtype,
                partial(
                    spmv_v2_multi_cube_benchmark,
                    B=B,
                    s=s,
                ),
                len(B.data),
                density,
                nnr,
                distr,
                alpha,
            )
        else:
            raise RuntimeError(
                f"Unsupported benchmark setup: bench:{bench}, dtype:{dtype}, s:{s}"
            )
