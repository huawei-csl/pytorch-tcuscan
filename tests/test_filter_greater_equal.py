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

import tcuscan_ops
import torch

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


def tcuscan_filter_greater_equal(x, pivot: float, s: int):
    z = tcuscan_ops.run_filter_greater_eq(x, pivot, s)
    return z


def _test_filter_greater_equal(vec_len: int, s: int, dtype: torch.dtype):
    x = torch.randint(0, 2**7 - 1, (vec_len,), dtype=dtype, device=NPU_DEVICE)
    pivot = 13.0
    torch.npu.synchronize()
    mask = (x >= pivot).to(torch.uint8)

    torch.npu.synchronize()
    expected = torch.masked_select(x, mask)
    torch.npu.synchronize()
    actual = tcuscan_filter_greater_equal(x, float(pivot), s)
    torch.npu.synchronize()

    assert len(expected) == len(actual)
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("vec_len", get_lengths(s=32, max_iters=12))
def test_tcuscan_filter_ge_fp16_s_32(vec_len: int):
    _test_filter_greater_equal(vec_len, 32, torch.float16)


@pytest.mark.parametrize("vec_len", get_lengths(s=64, max_iters=8))
def test_tcuscan_filter_ge_fp16_s_64(vec_len: int):
    _test_filter_greater_equal(vec_len, 64, torch.float16)


@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=6))
def test_tcuscan_filter_ge_fp16_s_128(vec_len: int):
    _test_filter_greater_equal(vec_len, 128, torch.float16)


@pytest.mark.parametrize("vec_len", get_lengths(s=32, max_iters=12))
@pytest.mark.parametrize("offset", _OFFSETS)
def test_tcuscan_filter_ge_fp32_s_16(vec_len: int, offset: int):
    _test_filter_greater_equal(vec_len - offset, 32, torch.float32)


@pytest.mark.parametrize("vec_len", get_lengths(s=64, max_iters=8))
@pytest.mark.parametrize("offset", _OFFSETS)
def test_tcuscan_filter_ge_fp32_s_64(vec_len: int, offset: int):
    _test_filter_greater_equal(vec_len - offset, 64, torch.float32)


@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=6))
@pytest.mark.parametrize("offset", _OFFSETS)
def test_tcuscan_filter_ge_fp32_s_128(vec_len: int, offset: int):
    _test_filter_greater_equal(vec_len - offset, 128, torch.float32)
