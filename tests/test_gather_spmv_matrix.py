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
from scipy.sparse import random as sp_random
import torch
import torch_npu  # noqa

import tcuscan_ops

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

torch.npu.config.allow_internal_format = False
_n_rows = [256, 512, 1024, 2048, 4096, 8192, 16384]
_densities = [0.01, 0.001, 0.0001]
_matrices = {}
for density in _densities:
    _matrices[density] = {}
    for n in _n_rows:
        _matrices[density][n] = sp_random(
            n - 1,
            n - 1,
            density=density,
            format="csr",
            dtype=np.float32,
        )


def _test_tcuscan_gather_spmv(n: int, s: int, density: float):
    B = _matrices[density][n]
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

    actual_cpu = actual.cpu()
    error_indices = []
    for i in range(expected.shape[0]):
        if np.abs(actual_cpu[i] - expected[i]) > 1e-3:
            error_indices.append(i)
    assert len(error_indices) == 0, "\n".join(
        [f"Error for size n={n}, occured at:"]
        + [
            f"index i = {i}: value = {actual_cpu[i]:.4f}, expected = {expected[i]:.4f}"
            for i in error_indices
        ]
    )


@pytest.mark.parametrize("n", _n_rows)
@pytest.mark.parametrize("s", [128, 256])
@pytest.mark.parametrize("density", _densities)
def test_tcuscan_gather_spmv(n: int, s: int, density: float):
    _test_tcuscan_gather_spmv(n, s, density)
