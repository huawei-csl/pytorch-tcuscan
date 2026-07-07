#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import os
from functools import partial

import numpy as np
import pytest
import torch_npu  # noqa
from scipy.sparse import random

import tcuscan_ops
import torch

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)

_NROW = [
    1024 * 1,
    1024 * 2,
    1024 * 3,
    1024 * 4,
    1024 * 5,
    1024 * 6,
    1024 * 7,
    1024 * 8,
    1024 * 9,
    1024 * 16,
]

# Absolute tolerance for the (float32 reference) vs (kernel) comparison.
_ATOL = {torch.float32: 1e-3, torch.float16: 1e-1}


def uniform_rvs(shape, dtype: np.dtype, scale: int = 6):
    if np.issubsctype(dtype, np.integer):
        return np.random.randint(-scale, scale, size=shape)
    return scale * np.random.uniform(0, 1, size=shape) - (scale // 2)


def _make_csr(nrow: int, ncol: int, density: float, scale_factor: int):
    return random(
        nrow,
        ncol,
        density=density,
        format="csr",
        dtype=np.float32,
        data_rvs=partial(uniform_rvs, dtype=np.float32, scale=scale_factor),
    )


def _test_tcuscan_spmv_ops_sparse(
    nrow: int,
    ncol: int,
    density: float,
    dtype: torch.dtype,
    scale_factor: int,
    alpha: float,
    beta: float,
):
    rng = np.random.default_rng(seed=42)

    B = _make_csr(nrow, ncol, density, scale_factor)

    values = (B.data).astype(np.float32)
    indexes = (B.indptr).astype(np.uint32)
    cols = (B.indices).astype(np.uint32)
    vector = rng.uniform(1, 9, ncol).astype(np.float32)
    y0 = rng.uniform(1, 9, nrow).astype(np.float32)

    # Reference: y = alpha * A @ x + beta * y (computed in float64).
    result = alpha * (B.astype(np.float64) @ vector.astype(np.float64))
    if beta != 0.0:
        result = result + beta * y0.astype(np.float64)

    torch_values = torch.from_numpy(values).to(dtype).npu()
    torch_indexes = torch.from_numpy(indexes).npu()
    torch_cols = torch.from_numpy(cols).to(torch.int32).npu()
    torch_vector = torch.from_numpy(vector).to(dtype).npu()
    torch_y = torch.from_numpy(y0).to(dtype).npu() if beta != 0.0 else None

    torch.npu.synchronize()
    actual = tcuscan_ops.run_spmv_ops_sparse(
        torch_values,
        torch_indexes,
        torch_cols,
        torch_vector,
        alpha=alpha,
        beta=beta,
        y=torch_y,
    )
    torch.npu.synchronize()

    actual_cpu = actual.cpu()
    expected = torch.from_numpy(result).to(actual_cpu.dtype)

    assert actual.shape == expected.shape
    # The output has the same dtype as the input values.
    assert actual.dtype == dtype

    assert torch.allclose(
        actual_cpu, expected, atol=_ATOL[dtype]
    ), f"Error spmv_ops_sparse ({dtype}). alpha={alpha} beta={beta}"


@pytest.mark.parametrize(("alpha", "beta"), [(1.0, 0.0), (2.0, 0.5)])
@pytest.mark.parametrize("density", [0.01, 0.001, 0.0001])
@pytest.mark.parametrize("nrow", _NROW)
@pytest.mark.parametrize(
    ("dtype", "scale_factor"),
    [
        pytest.param(torch.float32, 2, id="torch.float32"),
        pytest.param(torch.float16, 2, id="torch.float16"),
    ],
)
def test_tcuscan_spmv_ops_sparse(
    density: float,
    nrow: int,
    dtype: torch.dtype,
    scale_factor: int,
    alpha: float,
    beta: float,
):
    # Square matrix (A is (nrow-1) x (nrow-1)).
    _test_tcuscan_spmv_ops_sparse(
        nrow - 1, nrow - 1, density, dtype, scale_factor, alpha, beta
    )


@pytest.mark.parametrize(("nrow", "ncol"), [(2048, 1024), (1024, 4096)])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, id="torch.float32"),
        pytest.param(torch.float16, id="torch.float16"),
    ],
)
def test_tcuscan_spmv_ops_sparse_rectangular(nrow: int, ncol: int, dtype: torch.dtype):
    # Non-square A exercises the num_cols != nrow path.
    _test_tcuscan_spmv_ops_sparse(nrow, ncol, 0.01, dtype, 2, 1.0, 0.0)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, id="torch.float32"),
        pytest.param(torch.float16, id="torch.float16"),
    ],
)
def test_tcuscan_spmv_ops_sparse_empty_rows(dtype: torch.dtype):
    # An all-empty matrix must reduce to y = beta * y (here beta = 0.5).
    nrow, ncol = 512, 256
    indexes = torch.zeros(nrow + 1, dtype=torch.int32).npu()
    cols = torch.zeros(0, dtype=torch.int32).npu()
    values = torch.zeros(0, dtype=dtype).npu()
    vector = torch.ones(ncol, dtype=dtype).npu()
    y0 = torch.arange(1, nrow + 1, dtype=torch.float32)
    torch_y = y0.to(dtype).npu()

    torch.npu.synchronize()
    actual = tcuscan_ops.run_spmv_ops_sparse(
        values, indexes, cols, vector, alpha=1.0, beta=0.5, y=torch_y
    )
    torch.npu.synchronize()

    expected = (0.5 * y0).to(actual.cpu().dtype)
    assert torch.allclose(actual.cpu(), expected, atol=_ATOL[dtype])


def test_tcuscan_spmv_ops_sparse_trans_is_gated():
    # The transposed variant is migrated but gated on this platform; it must
    # raise cleanly rather than launch (and fault) the atomic-scatter kernel.
    B = _make_csr(64, 32, 0.1, 2)
    values = torch.from_numpy(B.data.astype(np.float32)).npu()
    indexes = torch.from_numpy(B.indptr.astype(np.uint32)).npu()
    cols = torch.from_numpy(B.indices.astype(np.uint32)).to(torch.int32).npu()
    vector = torch.ones(64, dtype=torch.float32).npu()

    with pytest.raises(RuntimeError):
        tcuscan_ops.run_spmv_ops_sparse(
            values, indexes, cols, vector, trans=True, num_cols=32
        )
