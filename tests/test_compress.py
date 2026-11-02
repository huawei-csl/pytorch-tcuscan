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


def tcuscan_compress(x, mask, s: int):
    z = tcuscan_ops.run_compress(x, mask, s)
    return z


def _test_compress(vec_len: int, s: int, dtype: torch.dtype):
    x = torch.randint(0, 2**7 - 1, (vec_len,), dtype=dtype, device=NPU_DEVICE)
    mask = (torch.randn(vec_len) > 0).to(torch.int8).npu()

    expected = torch.masked_select(x, mask.to(torch.uint8))
    torch.npu.synchronize()
    actual = tcuscan_compress(x, mask, s)

    assert len(expected) == len(actual)
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("vec_len", get_lengths(s=32, max_iters=12))
def test_tcuscan_compress_fp16_s_32(vec_len: int):
    _test_compress(vec_len, 32, torch.float16)


@pytest.mark.parametrize("vec_len", get_lengths(s=64, max_iters=8))
def test_tcuscan_compress_fp16_s_64(vec_len: int):
    _test_compress(vec_len, 64, torch.float16)


@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=6))
def test_tcuscan_compress_fp16_s_128(vec_len: int):
    _test_compress(vec_len, 128, torch.float16)


@pytest.mark.parametrize("vec_len", get_lengths(s=32, max_iters=12))
@pytest.mark.parametrize("offset", _OFFSETS)
def test_tcuscan_compress_fp32_s_16(vec_len: int, offset: int):
    _test_compress(vec_len - offset, 32, torch.float32)


@pytest.mark.parametrize("vec_len", get_lengths(s=64, max_iters=8))
@pytest.mark.parametrize("offset", _OFFSETS)
def test_tcuscan_compress_fp32_s_64(vec_len: int, offset: int):
    _test_compress(vec_len - offset, 64, torch.float32)


@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=6))
@pytest.mark.parametrize("offset", _OFFSETS)
def test_tcuscan_compress_fp32_s_128(vec_len: int, offset: int):
    _test_compress(vec_len - offset, 128, torch.float32)


# --------------------------------------------------------------------------------
# Output length
#
# The output is as long as there are ones in the mask, which is a property of
# the data.  The wrapper no longer reduces the mask on the host to size the
# output: it allocates for the worst case, lets the kernel's own reduction
# report how many elements each core wrote, and returns that prefix.  The two
# extremes are what pin that down -- an empty result and one as long as the
# input.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("s", [32, 64, 128], ids=lambda x: f"s:{x}")
@pytest.mark.parametrize("ones", [0, 1], ids=lambda x: f"mask:{'all' if x else 'none'}")
def test_tcuscan_compress_uniform_mask(s: int, ones: int):
    vec_len = 20 * s * s
    x = torch.randint(0, 2**7 - 1, (vec_len,), dtype=torch.float16, device=NPU_DEVICE)
    mask = torch.full((vec_len,), ones, dtype=torch.int8, device=NPU_DEVICE)
    torch.npu.synchronize()

    actual = tcuscan_ops.run_compress(x, mask, s)
    torch.npu.synchronize()

    expected = torch.masked_select(x, mask.to(torch.uint8))
    assert actual.numel() == vec_len * ones
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("density", [0.001, 0.01, 0.5, 0.99], ids=lambda x: f"d:{x}")
def test_tcuscan_compress_sparse_and_dense_masks(density: float):
    """The reported length must track the mask at either end of the density range."""
    s = 128
    vec_len = 20 * s * s
    x = torch.randint(0, 2**7 - 1, (vec_len,), dtype=torch.float16, device=NPU_DEVICE)
    mask = (torch.rand(vec_len, device=NPU_DEVICE) < density).to(torch.int8)
    torch.npu.synchronize()

    actual = tcuscan_ops.run_compress(x, mask, s)
    torch.npu.synchronize()

    expected = torch.masked_select(x, mask.to(torch.uint8))
    assert actual.numel() == expected.numel()
    assert torch.equal(expected, actual)
