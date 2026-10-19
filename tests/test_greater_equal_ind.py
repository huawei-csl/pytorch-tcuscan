# --------------------------------------------------------------------------------
# Copyright (c) 2023-2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pytorch-tcuscan/
# for the full License text.
# --------------------------------------------------------------------------------
import os
import random

import numpy as np
import pytest
import torch_npu  # noqa

import torch  # isort: skip
import tcuscan_ops  # isort: skip

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)

_OFFSETS = [-127, -13, 13, 65, 129]


def get_lengths(s: int, max_iters: int):
    NUM_AI_CORES = 20
    for multiplier in range(1, max_iters):
        yield multiplier * NUM_AI_CORES * s * s


def _test_greater_equal_ind(vec_len: int, s: int, dtype: torch.dtype, pivot: float):
    x = torch.randint(0, 2**7 - 1, (vec_len,), dtype=dtype, device=NPU_DEVICE)
    positions = torch.arange(0, len(x), 1, dtype=torch.int32, device=NPU_DEVICE)

    torch.npu.synchronize()
    mask = (x >= pivot).to(torch.uint8)
    expected = torch.masked_select(x, mask)
    expected_indices = torch.masked_select(positions, mask)
    torch.npu.synchronize()

    actual, actual_indices = tcuscan_ops.run_greater_equal_ind(x, pivot, s)
    torch.npu.synchronize()

    assert len(expected) == len(actual)
    assert torch.equal(expected, actual)

    assert len(expected_indices) == len(actual_indices)
    assert torch.equal(expected_indices, actual_indices)

    # The returned indices must address the returned values.
    assert torch.equal(x[actual_indices.long()], actual)


@pytest.mark.parametrize("vec_len", get_lengths(s=32, max_iters=12))
def test_greater_equal_ind_fp16_s_32(vec_len: int):
    _test_greater_equal_ind(vec_len, 32, torch.float16, 64.0)


@pytest.mark.parametrize("vec_len", get_lengths(s=64, max_iters=8))
def test_greater_equal_ind_fp16_s_64(vec_len: int):
    _test_greater_equal_ind(vec_len, 64, torch.float16, 64.0)


@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=6))
def test_greater_equal_ind_fp16_s_128(vec_len: int):
    _test_greater_equal_ind(vec_len, 128, torch.float16, 64.0)


@pytest.mark.parametrize("vec_len", get_lengths(s=32, max_iters=12))
@pytest.mark.parametrize("offset", _OFFSETS)
def test_greater_equal_ind_fp32_s_32(vec_len: int, offset: int):
    _test_greater_equal_ind(vec_len - offset, 32, torch.float32, 64.0)


@pytest.mark.parametrize("vec_len", get_lengths(s=64, max_iters=8))
@pytest.mark.parametrize("offset", _OFFSETS)
def test_greater_equal_ind_fp32_s_64(vec_len: int, offset: int):
    _test_greater_equal_ind(vec_len - offset, 64, torch.float32, 64.0)


# fp32 at s=128 is the largest tile the kernel handles and the tightest fit in
# the Unified Buffer, so it is covered explicitly.
@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=4))
@pytest.mark.parametrize("offset", _OFFSETS)
def test_greater_equal_ind_fp32_s_128(vec_len: int, offset: int):
    _test_greater_equal_ind(vec_len - offset, 128, torch.float32, 64.0)


@pytest.mark.parametrize("s", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=str)
def test_greater_equal_ind_negative_pivot(s: int, dtype: torch.dtype):
    # A pivot below every input selects everything, including any lane the
    # kernel pads a partial tile with.
    _test_greater_equal_ind(20 * s * s - 13, s, dtype, -1.0)


@pytest.mark.parametrize("pivot", [-1.0, 0.0, 128.0])
def test_greater_equal_ind_edge_pivots(pivot: float):
    # pivot below the minimum selects everything, above the maximum nothing.
    _test_greater_equal_ind(20 * 32 * 32, 32, torch.float16, pivot)
