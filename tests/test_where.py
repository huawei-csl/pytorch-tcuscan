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

# TODO(anastasios): enable offsets
# _OFFSETS = [-127, -13, 13, 65, 129]


def get_lengths(s: int, max_iters: int):
    NUM_AI_CORES = 20
    for multiplier in range(1, max_iters):
        yield multiplier * NUM_AI_CORES * s * s


def _test_where(vec_len: int, s: int, dtype: torch.dtype):
    x = torch.randint(0, 2**7 - 1, (vec_len,), dtype=dtype, device=NPU_DEVICE)
    pivot = 13.0
    torch.npu.synchronize()

    torch.npu.synchronize()
    expected = torch.argwhere(x >= pivot).flatten()
    torch.npu.synchronize()
    actual = tcuscan_ops.run_where(x, float(pivot), s)
    torch.npu.synchronize()

    print(expected[:16])
    print(actual[:16])
    assert len(expected) == len(actual)
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("vec_len", get_lengths(s=32, max_iters=12))
def test_tcuscan_where_fp16_s_32(vec_len: int):
    _test_where(vec_len, 32, torch.float16)


@pytest.mark.parametrize("vec_len", get_lengths(s=64, max_iters=8))
def test_tcuscan_where_fp16_s_64(vec_len: int):
    _test_where(vec_len, 64, torch.float16)


@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=6))
def test_tcuscan_where_fp16_s_128(vec_len: int):
    _test_where(vec_len, 128, torch.float16)
