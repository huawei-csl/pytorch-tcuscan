#!/usr/bin/env python3
# --------------------------------------------------------------------------------
# Copyright (c) 2023-2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pytorch-tcuscan/
# for the full License text.
# --------------------------------------------------------------------------------
"""Profile a single tcuscan kernel by name with the torch/NPU profiler.

Reads a kernel name from the command line, builds the inputs it needs, and
runs it under ``run_torch_profiler`` so the trace can be inspected in
TensorBoard.

Example:
    python scripts/profiling/profile_kernel.py --kernel spmv_v2 --nnr 4096 \\
        --density 0.01 --s 64
"""

import argparse
import os
import typing

import numpy as np
from scipy.sparse import csr_matrix, random

import torch

DEVICE = os.environ.get("DEVICE_TYPE", "npu")
if DEVICE == "npu":
    import torch_npu  # noqa

    NPU_DEVICE = "npu:1"
    torch.npu.config.allow_internal_format = False
    torch.npu.set_device(NPU_DEVICE)
    assert torch.npu.is_available()

import tcuscan_ops  # noqa


def run_torch_profiler(
    profile_dir: str,
    fn: typing.Callable,
    warmup_iters: int = 10,
):
    # Warm up so kernel compilation/caching doesn't pollute the trace.
    for _ in range(warmup_iters):
        fn()
    if DEVICE == "npu":
        torch.npu.synchronize()

    profiler = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_dir),
    )

    with profiler:
        fn()


def power_law_rvs(shape, exponent=2.0):
    return np.random.power(exponent, size=shape)


def uniform_rvs(shape):
    return np.random.uniform(0, 1, size=shape)


def make_csr(nnr: int, density: float, distr: str, alpha: float) -> csr_matrix:
    """Generate a random CSR matrix like the other profiling scripts do."""
    if distr == "Uniform":
        return random(
            nnr - 1,
            nnr - 1,
            density=density,
            format="csr",
            dtype=np.float32,
            data_rvs=uniform_rvs,
        )
    if distr == "PowerLaw":
        return random(
            nnr - 1,
            nnr - 1,
            density=density,
            format="csr",
            dtype=np.float32,
            data_rvs=lambda shape: power_law_rvs(shape, exponent=alpha),
        )
    raise ValueError(f"Unknown distribution: {distr}")


def _csr_to_npu(B: csr_matrix, indptr_dtype=np.uint32):
    """Common CSR -> NPU tensor conversion used by the SpMV kernels."""
    rng = np.random.default_rng(seed=42)
    vals = torch.from_numpy(B.data.astype(np.float16)).npu()
    idx = torch.from_numpy(B.indptr.astype(indptr_dtype)).npu()
    cols = torch.from_numpy(B.indices.astype(np.int32)).npu()
    vec = torch.from_numpy(
        rng.uniform(1, 9, len(B.indptr) - 1).astype(np.float16)
    ).npu()
    return vals, idx, cols, vec


# --------------------------------------------------------------------------------
# Kernel setups: each returns a zero-arg callable that invokes the kernel once.
# Add new kernels here.
# --------------------------------------------------------------------------------


def setup_spmv(B: csr_matrix, s: int) -> typing.Callable:
    vals, idx, cols, vec = _csr_to_npu(B, indptr_dtype=np.uint32)

    def run():
        _ = tcuscan_ops.run_spmv(vals, idx, cols, vec, s)

    return run


def setup_spmv_v2(B: csr_matrix, s: int) -> typing.Callable:
    vals, idx, cols, vec = _csr_to_npu(B, indptr_dtype=np.uint32)

    def run():
        _ = tcuscan_ops.run_spmv_v2(vals, idx, cols, vec, s)

    return run


def setup_spmv_multi_cube(B: csr_matrix, s: int) -> typing.Callable:
    vals, idx, cols, vec = _csr_to_npu(B, indptr_dtype=np.uint32)
    ones = torch.ones((s, s), dtype=torch.float16, device=NPU_DEVICE)
    upper = torch.triu(ones)
    lower_strict = torch.tril(ones, -1)

    def run():
        _ = tcuscan_ops.run_spmv_multi_cube(vals, idx, cols, vec, upper, lower_strict)

    return run


def setup_spmv_v2_multi_cube(B: csr_matrix, s: int) -> typing.Callable:
    vals, idx, cols, vec = _csr_to_npu(B, indptr_dtype=np.int32)
    ones = torch.ones((s, s), dtype=torch.float16, device=NPU_DEVICE)
    upper = torch.triu(ones)
    lower_strict = torch.tril(ones, -1)

    def run():
        _ = tcuscan_ops.run_spmv_v2_multi_cube(
            vals, idx, cols, vec, upper, lower_strict
        )

    return run


KERNEL_SETUPS = {
    "spmv": setup_spmv,
    "spmv_v2": setup_spmv_v2,
    "spmv_multi_cube": setup_spmv_multi_cube,
    "spmv_v2_multi_cube": setup_spmv_v2_multi_cube,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="profile_kernel",
        description="Profile a single tcuscan kernel with run_torch_profiler",
    )
    parser.add_argument(
        "--kernel", required=True, choices=sorted(KERNEL_SETUPS.keys())
    )
    parser.add_argument("--s", type=int, default=64)
    parser.add_argument("--nnr", type=int, default=4096, help="Number of rows")
    parser.add_argument("--density", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument(
        "--prob", choices=["PowerLaw", "Uniform"], default="Uniform"
    )
    parser.add_argument(
        "--profile-dir",
        default="./profiler_output",
        help="Directory for the TensorBoard trace",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations before profiling",
    )
    args = parser.parse_args()

    B = make_csr(args.nnr, args.density, args.prob, args.alpha)
    fn = KERNEL_SETUPS[args.kernel](B, args.s)

    profile_dir = os.path.join(args.profile_dir, args.kernel)
    run_torch_profiler(profile_dir, fn, warmup_iters=args.warmup)
    print(f"Wrote profiler trace for '{args.kernel}' to {profile_dir}")
