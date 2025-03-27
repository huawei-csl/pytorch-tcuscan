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
from scipy.sparse import random as sp_random

import tcuscan_ops

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

torch.npu.config.allow_internal_format = False

_NNR_ = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def uniform_rvs(shape):
    print(shape)
    return np.random.uniform(0, 1, size=shape)


def _test_tcuscan_gather_spmv(nnr: int, s: int, density: float):

    B = sp_random(
        nnr - 1,
        nnr - 1,
        density=density,
        format="csr",
        dtype=np.float32,
        data_rvs=uniform_rvs,
    )

    values = (B.data).astype(np.float32)
    indexes = (B.indptr).astype(np.uint32)

    input_cols_idx = [x - 1 for x in indexes]
    expected = np.zeros(len(input_cols_idx)).astype(np.float32)
    for i, idx in enumerate(input_cols_idx):
        if idx < 0:
            expected[i] = 0
        else:
            expected[i] = values[idx]

    expected = torch.from_numpy(expected)

    torch.npu.synchronize()

    values_npu = torch.from_numpy(values).npu()
    indexes_npu = torch.from_numpy(indexes).npu()
    torch.npu.synchronize()

    actual = tcuscan_ops.run_gather_spmv(values_npu, indexes_npu, s)
    assert actual.shape == expected.shape, "Output shape does not match expected shape."

    assert actual.dtype == expected.dtype, "Output dtype does not match expected dtype"

    assert torch.allclose(
        actual.cpu(), expected, atol=1e-02
    ), f"Error gather spmv ({expected.dtype}). s={s}, nnr={nnr}, density: {density}"


@pytest.mark.parametrize("nnr", _NNR_)
@pytest.mark.parametrize("s", [128, 256])
@pytest.mark.parametrize("density", [0.0001, 0.001, 0.001])
def test_tcuscan_gather_spmv(nnr: int, s: int, density: float):
    _test_tcuscan_gather_spmv(nnr, s, density)
