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

import numpy as np
from scipy.sparse import random as sp_random

import torch

DEVICE = os.environ.get("DEVICE_TYPE", "npu")
if DEVICE == "npu":
    import torch_npu  # noqa

    NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:0")
    torch.npu.config.allow_internal_format = True
    torch.npu.set_device(NPU_DEVICE)
    assert torch.npu.is_available()

import tcuscan_ops  # noqa

file_handler = logging.FileHandler(filename="profile_spmv_comparison.log")
stdout_handler = logging.StreamHandler(stream=sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=[file_handler, stdout_handler],
)

logger = logging.getLogger(__name__)

WARMUP_ITERS = 10
BENCH_ITERS = 100

STR_TO_DTYPE = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "int16": torch.int16,
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
    start_events = [device.event() for _ in range(benchmark_iters)]
    end_events = [device.event() for _ in range(benchmark_iters)]

    device.sync()
    for _ in range(warmup_iters):
        fn()

    device.sync()

    # Clear L2 cache between runs (copied from triton testing utils)
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
    return sum(times_ms) / len(times_ms) * 1000  # microseconds


def _dtype_bytes(dtype: torch.dtype) -> int:
    if dtype.is_floating_point:
        return torch.finfo(dtype).bits // 8
    return torch.iinfo(dtype).bits // 8


def _spmv_bytes(nrow: int, nnz: int, dtype: torch.dtype) -> int:
    elem = _dtype_bytes(dtype)
    return (
        nnz * elem          # values
        + nnz * 4           # col indices (int32)
        + (nrow + 1) * 4    # row pointers (uint32)
        + nrow * elem       # input vector
        + nrow * elem       # output vector
    )


def uniform_rvs(shape, dtype: np.dtype, scale: int = 6):
    if np.issubsctype(dtype, np.integer):
        return np.random.randint(-scale, scale, size=shape)
    return scale * np.random.uniform(0, 1, size=shape) - (scale // 2)


def _build_csr_inputs(
    nrow: int, density: float, dtype: torch.dtype, device_str: str
) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build CSR matrix inputs on NPU for a random (nrow x nrow) sparse matrix."""
    rng = np.random.default_rng(seed=42)
    sp_dtype = np.int32 if dtype == torch.int16 else np.float32
    scale = 6 if dtype == torch.int16 else 2

    B = sp_random(
        nrow,
        nrow,
        density=density,
        format="csr",
        dtype=sp_dtype,
        data_rvs=partial(uniform_rvs, dtype=sp_dtype, scale=scale),
    )

    values = B.data.astype(sp_dtype)
    indexes = B.indptr.astype(np.uint32)
    cols = B.indices.astype(np.uint32)
    vector = rng.uniform(1, 9, nrow).astype(sp_dtype)

    torch_values = torch.from_numpy(values).to(dtype).to(device_str)
    torch_indexes = torch.from_numpy(indexes).to(device_str).to(torch.int32)
    torch_cols = torch.from_numpy(cols).to(torch.int32).to(device_str)
    torch_vector = torch.from_numpy(vector).to(dtype).to(device_str)

    return torch_values, torch_indexes, torch_cols, torch_vector, B.nnz


def _spmv_v2_tiling(nnz: int, s: int, max_aic_cores: int = 20) -> typing.Tuple[int, int]:
    matmul_tile_len = s * s
    num_tiles = ceil(nnz / matmul_tile_len)
    num_blocks = min(num_tiles, max_aic_cores)
    max_num_tiles_per_block = ceil(num_tiles / num_blocks)
    block_len = max_num_tiles_per_block * matmul_tile_len
    return block_len, num_blocks


def spmv_benchmark(
    device: Device,
    torch_values: torch.Tensor,
    torch_indexes: torch.Tensor,
    torch_cols: torch.Tensor,
    torch_vector: torch.Tensor,
    s: int,
) -> float:
    def run() -> None:
        _ = tcuscan_ops.run_spmv(torch_values, torch_indexes, torch_cols, torch_vector, s)

    return _run_benchmark(device, run)


def spmv_v2_benchmark(
    device: Device,
    torch_values: torch.Tensor,
    torch_indexes: torch.Tensor,
    torch_cols: torch.Tensor,
    torch_vector: torch.Tensor,
    s: int,
) -> float:
    nnz = torch_values.numel()
    block_len, num_blocks = _spmv_v2_tiling(nnz, s)
    sstart = torch.clamp(
        torch.arange(0, num_blocks + 1, dtype=torch.int32, device=device.str) * block_len,
        max=nnz,
    )
    device.sync()
    segm_offsets = torch.searchsorted(torch_indexes, sstart, out_int32=True)
    device.sync()

    def run() -> None:
        _ = tcuscan_ops.run_spmv_v2(
            torch_values, torch_indexes, torch_cols, torch_vector, s, segm_offsets
        )

    return _run_benchmark(device, run)


def run_comparison(
    device: Device,
    nrows: typing.List[int],
    densities: typing.List[float],
    s_values: typing.List[int],
    dtype: torch.dtype,
    dtype_str: str,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    comparison_path = os.path.join(
        output_dir, f"spmv_comparison_{dtype_str}.csv"
    )

    with open(comparison_path, "w", encoding="UTF-8") as cmp_fd:
        cmp_fd.write("nrow,nnz,density,s,dtype,time_v1_us,time_v2_us,speedup_v2_vs_v1,bw_v1_GBs,bw_v2_GBs\n")

        for density in densities:
            v1_path = os.path.join(output_dir, f"spmv_v1_{dtype_str}_density{density}.csv")
            v2_path = os.path.join(output_dir, f"spmv_v2_{dtype_str}_density{density}.csv")

            with open(v1_path, "w", encoding="UTF-8") as v1_fd, \
                 open(v2_path, "w", encoding="UTF-8") as v2_fd:

                v1_fd.write("operator,dtype,nrow,nnz,density,s,time_us,bw_GBs\n")
                v2_fd.write("operator,dtype,nrow,nnz,density,s,time_us,bw_GBs\n")

                for nrow in nrows:
                    logger.info(
                        f"Building CSR matrix: nrow={nrow}, density={density}, dtype={dtype_str}"
                    )
                    try:
                        vals, idx, cols, vec, nnz = _build_csr_inputs(
                            nrow, density, dtype, device.str
                        )
                    except Exception as e:
                        logger.warning(f"Skipping nrow={nrow}, density={density}: {e}")
                        continue

                    device.sync()

                    for s in s_values:
                        logger.info(
                            f"Benchmarking spmv: nrow={nrow}, nnz={nnz:,}, "
                            f"density={density}, s={s}, dtype={dtype_str}"
                        )
                        try:
                            time_v1 = spmv_benchmark(device, vals, idx, cols, vec, s)
                            time_v2 = spmv_v2_benchmark(device, vals, idx, cols, vec, s)
                        except Exception as e:
                            logger.warning(
                                f"Skipping s={s}: {e}"
                            )
                            continue

                        speedup = time_v1 / time_v2 if time_v2 > 0 else float("inf")
                        nbytes = _spmv_bytes(nrow, nnz, dtype)
                        bw_v1 = nbytes / (time_v1 * 1e-6) / 1e9
                        bw_v2 = nbytes / (time_v2 * 1e-6) / 1e9

                        v1_fd.write(f"spmv,{dtype_str},{nrow},{nnz},{density},{s},{time_v1:.2f},{bw_v1:.2f}\n")
                        v2_fd.write(f"spmv_v2,{dtype_str},{nrow},{nnz},{density},{s},{time_v2:.2f},{bw_v2:.2f}\n")
                        cmp_fd.write(
                            f"{nrow},{nnz},{density},{s},{dtype_str},"
                            f"{time_v1:.2f},{time_v2:.2f},{speedup:.4f},{bw_v1:.2f},{bw_v2:.2f}\n"
                        )

                        logger.info(
                            f"  spmv={time_v1:.1f}us ({bw_v1:.1f} GB/s)  "
                            f"spmv_v2={time_v2:.1f}us ({bw_v2:.1f} GB/s)  "
                            f"speedup={speedup:.3f}x"
                        )

    logger.info(f"Comparison results written to {comparison_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="profile_spmv_comparison",
        description="Compare run_spmv vs run_spmv_v2 across matrix sizes up to 70,000 x 70,000",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "fp32", "int16"],
        default="fp16",
    )
    parser.add_argument(
        "--s",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help="Tile size parameter(s). Multiple values allowed.",
    )
    parser.add_argument(
        "--density",
        type=float,
        nargs="+",
        default=[0.01, 0.001, 0.0001],
        help="Sparse matrix density (fraction of non-zeros). Multiple values allowed.",
    )
    parser.add_argument(
        "--max-nrow",
        type=int,
        default=70000,
        help="Maximum matrix dimension (rows = cols).",
    )
    parser.add_argument(
        "--min-nrow",
        type=int,
        default=1024,
        help="Minimum matrix dimension.",
    )
    parser.add_argument(
        "--nrow-step",
        type=int,
        default=0,
        help="Step between nrow values. Defaults to max(1024, max_nrow // 20).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="spmv_bench_results",
        help="Directory for output CSV files.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=WARMUP_ITERS,
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=BENCH_ITERS,
    )

    args = parser.parse_args()

    WARMUP_ITERS = args.warmup_iters
    BENCH_ITERS = args.bench_iters

    dtype_str = args.dtype
    dtype = STR_TO_DTYPE[dtype_str]
    s_values = args.s
    densities = args.density
    max_nrow = args.max_nrow
    min_nrow = args.min_nrow
    nrow_step = args.nrow_step if args.nrow_step > 0 else max(1024, max_nrow // 20)
    output_dir = args.output_dir

    nrows = list(range(min_nrow, max_nrow + 1, nrow_step))
    if max_nrow not in nrows:
        nrows.append(max_nrow)

    if DEVICE == "npu":
        device = Device(torch.npu, NPU_DEVICE)
    elif DEVICE == "cpu":
        device = Device(torch, "cpu")
    else:
        device = Device(torch.cuda, "cuda:0")

    logger.info("=" * 60)
    logger.info("SpMV v1 vs v2 Comparison Benchmark")
    logger.info(f"  dtype     : {dtype_str}")
    logger.info(f"  s values  : {s_values}")
    logger.info(f"  densities : {densities}")
    logger.info(f"  nrow range: {min_nrow} .. {max_nrow} (step {nrow_step})")
    logger.info(f"  nrows     : {nrows}")
    for d in densities:
        nnz_vals = [int(n * n * d) for n in [min_nrow, max_nrow]]
        logger.info(f"  exp. nnz (density={d}): {nnz_vals[0]} .. {nnz_vals[1]}")
    logger.info(f"  device    : {device.str}")
    logger.info(f"  output    : {output_dir}/")
    logger.info(f"  warmup    : {WARMUP_ITERS}  bench: {BENCH_ITERS}")
    logger.info("=" * 60)

    run_comparison(
        device=device,
        nrows=nrows,
        densities=densities,
        s_values=s_values,
        dtype=dtype,
        dtype_str=dtype_str,
        output_dir=output_dir,
    )
