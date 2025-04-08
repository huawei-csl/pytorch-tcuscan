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
from scipy import sparse

import tcuscan_ops
import torch

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)


def _test_tcuscan_gather_spmv(n: int, s: int, density: float):
    B = sparse.random(
        n - 1,
        n - 1,
        density=density,
        format="csr",
        dtype=np.float32,
    )
    values = (B.data).astype(np.float32)
    indexes = (B.indptr).astype(np.uint32)

    expected = np.zeros(indexes.shape).astype(np.float32)
    mask = indexes != 0
    expected[1:] = values[[idx - 1 for idx in indexes[1:]]] * mask[1:]
    expected = torch.from_numpy(expected)

    torch.npu.synchronize()

    values_npu = torch.from_numpy(values).npu()
    indexes_npu = torch.from_numpy(indexes).npu()
    torch.npu.synchronize()

    actual = tcuscan_ops.run_gather_spmv(values_npu, indexes_npu, s)
    torch.npu.synchronize()

    assert actual.shape == expected.shape, "Output shape does not match expected shape."
    assert actual.dtype == expected.dtype, "Output dtype does not match expected dtype"

    assert np.allclose(actual.cpu(), expected, atol=1e-3)


@pytest.mark.parametrize("n", [256, 512, 1024, 2048, 4096, 8192, 16384])
@pytest.mark.parametrize("s", [128, 256])
@pytest.mark.parametrize("density", [0.01, 0.001, 0.0001])
def test_tcuscan_gather_spmv(n: int, s: int, density: float):
    _test_tcuscan_gather_spmv(n, s, density)
