#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
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

import tcuscan_ops
import torch

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:0")
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


def uniform_rvs(shape):
    return 6 * np.random.uniform(0, 1, size=shape) - 1


def _test_tcuscan_spmv(nnr: int, s: int, density: float, dtype: torch.dtype):
    rng = np.random.default_rng(seed=42)
    out_dtype = np.int32 if dtype == torch.int16 else np.float32

    B = random(
        nnr - 1,
        nnr - 1,
        density=density,
        format="csr",
        dtype=out_dtype,
        data_rvs=uniform_rvs,
    )

    values = (B.data).astype(out_dtype)
    indexes = (B.indptr).astype(np.uint32)
    cols = (B.indices).astype(np.uint32)
    vector = rng.uniform(1, 9, nnr - 1).astype(out_dtype)

    result = B @ vector

    torch_values = torch.from_numpy(values).to(dtype).npu()
    torch_indexes = torch.from_numpy(indexes).npu()
    torch_cols = torch.from_numpy(cols).npu()

    torch_vector = torch.from_numpy(vector).to(dtype).npu()

    torch.npu.synchronize()
    actual = tcuscan_ops.run_spmv(
        torch_values, torch_indexes, torch_cols, torch_vector, s
    )
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    expected = torch.from_numpy(result)
    assert (
        actual.shape == expected.shape
    ), f"Output shape mismatch. Got {actual.shape}. Expected {expected.shape}"

    expected_dtype = torch.float32 if dtype == torch.float16 else torch.int32
    assert actual.dtype == expected_dtype

    assert torch.allclose(
        actual_cpu, expected, atol=1e-01
    ), f"Error spmv ({expected.dtype}). s={s}"


@pytest.mark.parametrize("s", [32, 64, 128])
@pytest.mark.parametrize("density", [0.01, 0.001, 0.0001])
@pytest.mark.parametrize("nrow", _NROW)
@pytest.mark.parametrize("dtype", [torch.int16, torch.float16], ids=str)
def test_tcuscan_spmv(s: int, density: float, nrow: int, dtype: torch.dtype):
    _test_tcuscan_spmv(nrow, s, density, dtype)
