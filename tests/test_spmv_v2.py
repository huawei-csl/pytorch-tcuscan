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

import numpy as np
import pytest
import torch_npu  # noqa
from scipy.sparse import random
from functools import partial

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


def uniform_rvs(shape, dtype: np.dtype, scale: int = 6):
    if np.issubsctype(dtype, np.integer):
        return np.random.randint(-scale, scale, size=shape)
    return scale * np.random.uniform(0, 1, size=shape) - (scale // 2)


# SpMV variants under test: the default op splits the non-zeros into
# L2-cache-sized chunks; the ``_no_l2`` op forces a single chunk. Both must
# match the CPU reference (and each other).
_VARIANTS = [
    pytest.param("run_spmv_v2", id="l2_split"),
    pytest.param("run_spmv_v2_no_l2", id="no_l2"),
]


def _run_variant(
    variant: str, values, indexes, cols, vector, s: int, dtype: torch.dtype
):
    torch_values = torch.from_numpy(values).to(dtype).npu()
    torch_indexes = torch.from_numpy(indexes).npu()
    torch_cols = torch.from_numpy(cols).to(torch.int32).npu()
    torch_vector = torch.from_numpy(vector).to(dtype).npu()

    op = getattr(tcuscan_ops, variant)
    torch.npu.synchronize()
    actual = op(torch_values, torch_indexes, torch_cols, torch_vector, s)
    torch.npu.synchronize()
    return actual


def _make_matrix(nnr: int, density: float, out_dtype, scale_factor: int):
    rng = np.random.default_rng(seed=42)
    B = random(
        nnr - 1,
        nnr - 1,
        density=density,
        format="csr",
        dtype=out_dtype,
        data_rvs=partial(uniform_rvs, dtype=out_dtype, scale=scale_factor),
    )
    values = (B.data).astype(out_dtype)
    indexes = (B.indptr).astype(np.uint32)
    cols = (B.indices).astype(np.uint32)
    vector = rng.uniform(1, 9, nnr - 1).astype(out_dtype)
    return B, values, indexes, cols, vector


def _test_tcuscan_spmv_v2(
    nnr: int,
    s: int,
    density: float,
    dtype: torch.dtype,
    scale_factor: int,
    variant: str = "run_spmv_v2",
):
    out_dtype = np.int32 if dtype == torch.int16 else np.float32
    B, values, indexes, cols, vector = _make_matrix(
        nnr, density, out_dtype, scale_factor
    )

    result = B @ vector

    actual = _run_variant(variant, values, indexes, cols, vector, s, dtype)

    actual_cpu = actual.cpu()
    expected = torch.from_numpy(result)
    assert actual.shape == expected.shape

    expected_dtype = (
        torch.float32 if dtype in (torch.float16, torch.float32) else torch.int32
    )
    assert actual.dtype == expected_dtype

    assert torch.allclose(
        actual_cpu, expected, atol=1e-01
    ), f"Error spmv ({expected.dtype}). s={s}, variant={variant}"


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("s", [32, 64, 128])
@pytest.mark.parametrize("density", [0.01, 0.001, 0.0001])
@pytest.mark.parametrize("nrow", _NROW)
@pytest.mark.parametrize(
    ("dtype", "scale_factor"),
    [
        pytest.param(torch.float32, 2, id="torch.float32"),
        pytest.param(torch.float16, 2, id="torch.float16"),
    ],
)
def test_tcuscan_spmv_v2(
    s: int,
    density: float,
    nrow: int,
    dtype: torch.dtype,
    scale_factor: int,
    variant: str,
):
    _test_tcuscan_spmv_v2(nrow, s, density, dtype, scale_factor, variant)


# Large matrices whose working set (CSR products + fp32 scan output) exceeds L2,
# forcing run_spmv_v2 into multiple L2 chunks. This exercises the cross-chunk
# seam: rows straddling a chunk boundary must accumulate correctly via the
# atomic-add output. The no-L2 path (single chunk) is used as the oracle in
# addition to the CPU reference.
@pytest.mark.parametrize("s", [32, 64, 128])
@pytest.mark.parametrize(
    ("nrow", "density"),
    [
        pytest.param(1024 * 32, 0.02, id="32k_d0.02"),
        pytest.param(1024 * 48, 0.01, id="48k_d0.01"),
    ],
)
@pytest.mark.parametrize(
    ("dtype", "scale_factor"),
    [
        pytest.param(torch.float32, 2, id="torch.float32"),
        pytest.param(torch.float16, 2, id="torch.float16"),
    ],
)
def test_tcuscan_spmv_v2_l2_split(
    s: int, nrow: int, density: float, dtype: torch.dtype, scale_factor: int
):
    out_dtype = np.int32 if dtype == torch.int16 else np.float32
    B, values, indexes, cols, vector = _make_matrix(
        nrow, density, out_dtype, scale_factor
    )
    expected = torch.from_numpy(B @ vector)

    split = _run_variant("run_spmv_v2", values, indexes, cols, vector, s, dtype).cpu()
    no_split = _run_variant(
        "run_spmv_v2_no_l2", values, indexes, cols, vector, s, dtype
    ).cpu()

    atol = 1e-01 if dtype == torch.float32 else 1.0
    assert torch.allclose(
        split, expected, atol=atol
    ), f"L2-split spmv mismatch vs reference. s={s}, nrow={nrow}"
    assert torch.allclose(
        split, no_split, atol=atol
    ), f"L2-split spmv disagrees with no-L2 path. s={s}, nrow={nrow}"
