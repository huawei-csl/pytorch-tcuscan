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


# --------------------------------------------------------------------------------
# Output length
#
# The result holds one index per element at or above the pivot.  The wrapper no
# longer reduces the mask on the host to size it: it allocates for the worst
# case, lets the kernel's own reduction report how many indices each core
# wrote, and returns that prefix.  The pivots below put every element on one
# side or the other, and a sparse one in between.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("s", [32, 64, 128], ids=lambda x: f"s:{x}")
@pytest.mark.parametrize("pivot", [-1.0, 200.0], ids=lambda x: f"pivot:{x}")
def test_tcuscan_where_pivot_outside_the_data(s: int, pivot: float):
    vec_len = 20 * s * s
    x = torch.randint(0, 2**7 - 1, (vec_len,), dtype=torch.float16, device=NPU_DEVICE)
    torch.npu.synchronize()

    actual = tcuscan_ops.run_where(x, pivot, s)
    torch.npu.synchronize()

    expected = torch.argwhere(x >= pivot).flatten()
    assert actual.numel() == expected.numel()
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("pivot", [1.0, 64.0, 120.0], ids=lambda x: f"pivot:{x}")
def test_tcuscan_where_sparse_and_dense_results(pivot: float):
    """The reported length must track how many elements pass the pivot."""
    s = 128
    vec_len = 20 * s * s
    x = torch.randint(0, 2**7 - 1, (vec_len,), dtype=torch.float16, device=NPU_DEVICE)
    torch.npu.synchronize()

    actual = tcuscan_ops.run_where(x, pivot, s)
    torch.npu.synchronize()

    expected = torch.argwhere(x >= pivot).flatten()
    assert actual.numel() == expected.numel()
    assert torch.equal(expected, actual)


def test_tcuscan_where_rejects_unsupported_dtype():
    """Only float16 is implemented; the output length is otherwise undefined."""
    x = torch.rand(20 * 32 * 32, dtype=torch.float32, device=NPU_DEVICE)
    torch.npu.synchronize()

    with pytest.raises(RuntimeError, match="float16"):
        tcuscan_ops.run_where(x, 0.5, 32)
