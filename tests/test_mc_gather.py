#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================


import random

import numpy as np
import pytest
import torch
import torch_npu  # noqa

import tcuscan_ops

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

torch.npu.config.allow_internal_format = False

_MULTIPLIER = [1, 2, 3, 5, 8, 9, 12, 16, 20, 24, 40, 50]


def _test_tcuscan_mcgather(s: int, nnz: int, idx_len: int):
    data_type = np.float32
    index_type = np.uint32
    shape = nnz
    cols_id_shape = idx_len

    rng = np.random.default_rng(seed=42)
    input_values = rng.uniform(1, 100, shape).astype(data_type)
    input_cols = rng.uniform(0, nnz, cols_id_shape)

    input_cols.sort()
    input_cols = input_cols.astype(index_type)

    input_cols_idx = list(input_cols)
    golden = input_values[input_cols_idx]

    expected = torch.Tensor(golden).to(torch.float32)
    torch.npu.synchronize()
    val_torch = torch.Tensor(input_values).to(torch.float32).npu()
    idx_torch = torch.from_numpy(input_cols).npu()
    torch.npu.synchronize()
    actual = tcuscan_ops.run_mc_gather(val_torch, idx_torch, s)

    assert (
        actual.shape == expected.shape
    ), "Output shape mismatch. Got {actual.shape}. Expected {expected.shape}"
    assert (
        actual.dtype == expected.dtype
    ), f"Output dtype mismatch. Got {actual.dtype}. Expected {expected.dtype}"

    assert torch.allclose(
        actual.cpu(), expected, atol=1e-02
    ), f"Error gather mc ({expected.dtype}). s={s}"


@pytest.mark.parametrize("multiplier", _MULTIPLIER)
@pytest.mark.parametrize("s", [64, 128, 256, 512])
def test_tcuscan_mc_gather(multiplier: int, s: int):
    nnz = multiplier * 20 * s * s
    idx_len = s * s
    _test_tcuscan_mcgather(s, nnz, idx_len)
