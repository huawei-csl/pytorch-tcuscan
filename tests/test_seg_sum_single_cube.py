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
from scipy.sparse import random as sp_random

import tcuscan_ops
import torch

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)

_NUM_SEGMENTS = [513, 1025, 2011]  # 519, 2043 fails!
_NUM_COLUMNS = [64 * 64 - 1, 128 * 128, 128 * 128 - 13, 128 * 128 + 13, 128 * 128 - 133]


def uniform_rvs(shape):
    return 2 * np.random.uniform(0, 1, size=shape) - 1


def _test_seg_sum_single_cube(
    num_segments: int, num_cols: int, s: int, density: float, dtype: torch.dtype
):
    sp_dtype = np.float32 if dtype == torch.float16 else np.int32

    A = sp_random(
        num_segments,
        num_cols,
        density=density,
        format="csr",
        dtype=sp_dtype,
        data_rvs=uniform_rvs,
    )

    ones = np.ones(num_cols).astype(sp_dtype)

    values = (A.data).astype(sp_dtype)
    indices = (A.indptr).astype(np.uint32)
    nnz = A.nnz

    expected = A @ ones
    expected = torch.from_numpy(expected.flatten())

    # Drop last entry of indices.
    indices = indices[:-1]

    assert (
        len(indices) == num_segments
    ), f"Number of segments. vals: {len(values)}, indices: {len(indices)}"

    values_npu = torch.from_numpy(values).npu().to(dtype)
    indices_npu = torch.from_numpy(indices).npu().to(torch.uint32)
    ones_npu = torch.ones((s, s), dtype=dtype, device=NPU_DEVICE)
    upper_npu = torch.triu(ones_npu)
    lower_strict_npu = torch.tril(ones_npu, -1)

    torch.npu.synchronize()
    actual = tcuscan_ops.run_seg_sum_single_cube(
        values_npu, upper_npu, lower_strict_npu, indices_npu, s
    ).cpu()
    torch.npu.synchronize()

    print(f"# of segments : {num_segments}")
    print(f"# of columns  : {num_cols}")
    print(f"nnz           : {nnz}")
    print(f"indices       : {indices}")
    print(f"expected      : {expected}")
    print(f"actual        : {actual}")

    abs_error = torch.max(torch.abs(actual - expected))
    rel_error = torch.max(torch.abs((actual - expected) / expected))

    assert (
        actual.shape == expected.shape
    ), f"Output shape mismatch. Got {actual.shape}. Expected {expected.shape}"
    assert (
        actual.dtype == expected.dtype
    ), f"Output dtype mismatch. Got {actual.dtype}. Expected {expected.dtype}"
    assert torch.allclose(
        actual, expected, atol=2e-2
    ), f"Error seg_sum ({expected.dtype}). Abs-error: {abs_error} / {rel_error}, s={s}, num_cols={num_cols}"


@pytest.mark.parametrize("num_segments", _NUM_SEGMENTS)
@pytest.mark.parametrize("num_cols", _NUM_COLUMNS)
@pytest.mark.parametrize("s", [16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16], ids=str)
def test_seg_sum_single_cube(
    num_segments: int, num_cols: int, s: int, dtype: torch.dtype
):
    density = 0.01
    _test_seg_sum_single_cube(num_segments, num_cols, s, density, dtype)
