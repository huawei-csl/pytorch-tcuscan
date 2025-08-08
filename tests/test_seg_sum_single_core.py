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

_MULTIPLIER = [1, 2, 3, 4]


def uniform_rvs(shape):
    return 2 * np.random.uniform(0, 1, size=shape) - 1


def _test_tcuscan_seg_sum_single_core(num_cols: int, s: int, density: float):

    num_segments = 513
    A = sp_random(
        num_segments,
        num_cols,
        density=density,
        format="csr",
        dtype=np.float32,
        data_rvs=uniform_rvs,
    )

    ones = torch.ones(num_cols).half()
    values = (A.data).astype(np.float16)
    indices = (A.indptr).astype(np.uint32)

    expected = A @ ones
    expected = torch.from_numpy(expected.flatten())

    # Drop last entry of indices.
    indices = indices[:-1]

    assert (
        len(indices) == num_segments
    ), f"Number of segments. vals: {len(values)}, indices: {len(indices)}"

    values_npu = torch.from_numpy(values).npu().half()
    indices_npu = torch.from_numpy(indices).npu().to(torch.uint32)

    torch.npu.synchronize()
    actual = tcuscan_ops.run_seg_sum_single_core(values_npu, indices_npu, s).cpu()
    torch.npu.synchronize()

    print(f"indices: {indices}")
    print(f"expected: {expected}")
    print(f"actual: {actual}")

    abs_error = torch.max(torch.abs(actual - expected))
    rel_error = torch.max(torch.abs((actual - expected) / expected))

    assert (
        actual.shape == expected.shape
    ), f"Output shape mismatch. Got {actual.shape}. Expected {expected.shape}"
    assert (
        actual.dtype == expected.dtype
    ), f"Output dtype mismatch. Got {actual.dtype}. Expected {expected.dtype}"
    assert torch.allclose(
        actual, expected, atol=1e-2
    ), f"Error seg_sum ({expected.dtype}). Abs-error: {abs_error} / {rel_error}, s={s}, num_cols={num_cols}"


@pytest.mark.parametrize("multiplier", _MULTIPLIER)
def test_tcuscan_segmented_sum_s_32(multiplier: int):
    s = 32
    density = 0.01
    n = multiplier * s * s

    _test_tcuscan_seg_sum_single_core(n, s, density)


@pytest.mark.parametrize("multiplier", _MULTIPLIER)
def test_tcuscan_segmented_sum_s_64(multiplier: int):
    s = 64
    segm_density = 0.01
    n = multiplier * s * s

    _test_tcuscan_seg_sum_single_core(n, s, segm_density)


@pytest.mark.parametrize(
    "num_cols", [128 * 128, 128 * 128 - 13, 128 * 128 + 13, 2 * 128 * 128 + 133]
)
def test_tcuscan_segmented_sum_s_128(num_cols: int):
    s = 128
    segm_density = 0.01

    _test_tcuscan_seg_sum_single_core(num_cols, s, segm_density)
