# --------------------------------------------------------------------------------
# Copyright (c) 2023-2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pytorch-tcuscan/
# for the full License text.
# --------------------------------------------------------------------------------
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
import random

import numpy as np
import pytest
import torch_npu  # noqa

import tcuscan_ops
import torch

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)

_MULTIPLIER = [1, 2, 4, 5, 10, 15, 20, 25, 30, 45, 60]


def _test_tcuscan_gather_spmv(s: int, nnz: int, idx_len: int):
    data_type = np.float32
    index_type = np.uint32

    shape = nnz
    cols_id_shape = idx_len

    rng = np.random.default_rng(seed=42)
    input_values = rng.uniform(1, 100, shape).astype(data_type)
    input_cols = rng.uniform(0, nnz, cols_id_shape)

    input_cols.sort()
    input_cols = input_cols.astype(index_type)

    input_cols_idx = [x - 1 for x in input_cols]
    expected = np.zeros(len(input_cols_idx)).astype(np.float32)
    for i, idx in enumerate(input_cols_idx):
        if idx < 0:
            expected[i] = 0
        else:
            expected[i] = input_values[idx]

    expected = torch.from_numpy(expected)
    val_torch = torch.Tensor(input_values).to(torch.float32).npu()
    idx_torch = torch.from_numpy(input_cols).npu()

    torch.npu.synchronize()
    actual = tcuscan_ops.run_gather_spmv(val_torch, idx_torch, s)
    torch.npu.synchronize()

    assert (
        actual.shape == expected.shape
    ), "Output shape mismatch. Got {actual.shape}. Expected {expected.shape}"
    assert (
        actual.dtype == expected.dtype
    ), f"Output dtype mismatch. Got {actual.dtype}. Expected {expected.dtype}"

    assert torch.equal(
        actual.cpu(), expected
    ), f"Error gather spmv ({expected.dtype}). s={s}"


@pytest.mark.parametrize("offset", [0, 7, 11, 17])
@pytest.mark.parametrize("multiplier", _MULTIPLIER)
@pytest.mark.parametrize("s", [64, 128, 256, 512])
def test_tcuscan_gather_spmv(offset: int, multiplier: int, s: int):
    nnz = multiplier * 20 * s * s
    idx_len = s * s - offset
    _test_tcuscan_gather_spmv(s, nnz, idx_len)
