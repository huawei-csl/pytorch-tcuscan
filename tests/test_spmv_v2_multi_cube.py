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

import tcuscan_ops
import torch

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)

_NROW = [64 * 64 - 1, 24 * 128 + 1, 16 * 128 - 13, 64 * 128 + 13, 32 * 128 - 133]


def uniform_rvs(shape):
    return 2 * np.random.uniform(0, 1, size=shape) - 1


def _test_tcuscan_spmv_v2_multi_cube(nnr: int, s: int, density: float):
    rng = np.random.default_rng(seed=42)

    B = random(
        nnr - 1,
        nnr - 1,
        density=density,
        format="csr",
        dtype=np.float32,
        data_rvs=uniform_rvs,
    )

    values = (B.data).astype(np.float16)
    indexes = (B.indptr).astype(np.int32)
    cols = (B.indices).astype(np.int32)
    vector = rng.uniform(1, 9, nnr - 1).astype(np.float16)

    result = B @ vector

    torch_values = torch.from_numpy(values).npu()
    torch_indexes = torch.from_numpy(indexes).npu()
    torch_cols = torch.from_numpy(cols).npu()
    torch_vector = torch.from_numpy(vector).npu()

    ones = torch.ones((s, s), dtype=torch.float16, device=NPU_DEVICE)
    upper = torch.triu(ones)
    lower_strict = torch.tril(ones, -1)

    torch.npu.synchronize()
    actual = tcuscan_ops.run_spmv_v2_multi_cube(
        torch_values, torch_indexes, torch_cols, torch_vector, upper, lower_strict
    )
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    expected = torch.from_numpy(result)
    assert (
        actual.shape == expected.shape
    ), f"Output shape mismatch. Got {actual.shape}. Expected {expected.shape}"

    # The multi-cube block scan is a half-only kernel, so the output is fp32.
    assert actual.dtype == torch.float32

    assert torch.allclose(
        actual_cpu, expected, atol=1e-0
    ), f"Error spmv_v2_multi_cube  ({expected.dtype}). s={s}"


@pytest.mark.parametrize("s", [32, 64, 128])
@pytest.mark.parametrize("density", [0.01, 0.001, 0.0001])
@pytest.mark.parametrize("nrow", _NROW)
def test_tcuscan_spmv_v2_multi_cube(s: int, density: float, nrow: int):
    _test_tcuscan_spmv_v2_multi_cube(nrow, s, density)
