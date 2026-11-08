# --------------------------------------------------------------------------------
# Copyright (c) 2023-2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pytorch-tcuscan/
# for the full License text.
# --------------------------------------------------------------------------------
import os
import random
from math import ceil

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

# Input array size to benchmark sort
_SORT_SIZES = [
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1179648,
    4325376,
    16908288,
    25165824,
]


def get_sizes():
    s = 128
    max_size = 1e7
    num_cores = 20

    # Maximum number of iterations
    # Remove '- 3' for more thorough testing
    max_iters = ceil(max_size / (num_cores * s * s)) - 5

    return [i * num_cores * s * s for i in range(1, max_iters, 16 * 128 // s)]


def generate_random_input(dtype, vec_len):
    if not dtype.is_floating_point:
        xmin = torch.iinfo(dtype).min
        xmax = torch.iinfo(dtype).max
        return torch.randint(xmin, xmax, (vec_len,), dtype=dtype, device=NPU_DEVICE)
    else:
        # Centered on zero, so that the encoding of the negative values - and
        # the reversed split on the sign bit - is exercised too.
        return torch.rand((vec_len,), dtype=dtype, device=NPU_DEVICE) - 0.5


def _test_sort(vec_len: int, dtype: torch.dtype, s: int):
    x = generate_random_input(dtype, vec_len)

    torch.npu.synchronize()
    expected, expected_indices = torch.sort(x, dim=-1, descending=False)
    torch.npu.synchronize()
    actual, actual_indices = tcuscan_ops.run_radix_sort(x, s)
    torch.npu.synchronize()

    assert len(expected) == len(
        actual
    ), f"Vector size must agree. Expected: {len(expected)}. Actual: {actual}"
    assert len(expected_indices) == len(
        actual_indices
    ), f"Indices size must agree. Expected: {len(expected_indices)}. Actual: {actual_indices}"

    assert torch.equal(expected, actual), "Output must be sorted"


@pytest.mark.parametrize("vec_len", _SORT_SIZES)
@pytest.mark.parametrize("s", [32, 64, 128])
def test_tcuscan_sort_fp16(vec_len: int, s: int):
    if vec_len >= s * s:
        _test_sort(vec_len, torch.float16, s)


# TODO: See ISSUE-52
# Radix sort fails on irregular length.
# @pytest.mark.parametrize("vec_len", _SORT_SIZES[1:])
# def test_tcuscan_sort_int16_s_128_irregular(vec_len):
#    s = 128
#    dtype = torch.int16
#    _test_sort(vec_len, dtype, s)


@pytest.mark.parametrize("vec_len", get_sizes())
@pytest.mark.parametrize("s", [64, 128])
def test_tcuscan_sort_int16(vec_len: int, s: int):
    dtype = torch.int16
    _test_sort(vec_len, dtype, s)


@pytest.mark.parametrize("vec_len", _SORT_SIZES)
@pytest.mark.parametrize("s", [32, 64, 128])
def test_tcuscan_sort_fp32(vec_len: int, s: int):
    if vec_len >= s * s:
        _test_sort(vec_len, torch.float32, s)


@pytest.mark.parametrize("vec_len", get_sizes())
@pytest.mark.parametrize("s", [64, 128])
def test_tcuscan_sort_int32(vec_len: int, s: int):
    _test_sort(vec_len, torch.int32, s)


# Lengths that are not a multiple of the tile size, so that the last tile of
# the last block is partial.
@pytest.mark.parametrize("vec_len", [1000, 3000, 100003, 262145])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.int16, torch.int32]
)
def test_tcuscan_sort_irregular_length(vec_len: int, dtype: torch.dtype):
    _test_sort(vec_len, dtype, 64)


def test_tcuscan_sort_fp32_special_values():
    specials = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        float("inf"),
        -float("inf"),
        1e-45,  # denormal
        -1e-45,
        3.4e38,
        -3.4e38,
    ]
    x = torch.tensor(specials * 103, dtype=torch.float32, device=NPU_DEVICE)

    values, indices = tcuscan_ops.run_radix_sort(x, 32)
    torch.npu.synchronize()

    expected, _ = torch.sort(x)
    assert torch.equal(values, expected)
    assert torch.equal(x[indices.long()], values)


def test_tcuscan_sort_int32_extreme_values():
    info = torch.iinfo(torch.int32)
    extremes = [info.min, info.max, 0, -1, 1, info.min + 1, info.max - 1]
    x = torch.tensor(extremes * 149, dtype=torch.int32, device=NPU_DEVICE)

    values, indices = tcuscan_ops.run_radix_sort(x, 32)
    torch.npu.synchronize()

    expected, _ = torch.sort(x)
    assert torch.equal(values, expected)
    assert torch.equal(x[indices.long()], values)


def test_tcuscan_sort_rejects_unsupported_dtype():
    x = torch.rand(4096, dtype=torch.float64, device=NPU_DEVICE)
    with pytest.raises(RuntimeError, match="unsupported dtype"):
        tcuscan_ops.run_radix_sort(x, 64)


# A prime number of tiles used to collapse the launch down to a single block,
# because the block count was required to divide `num_tiles` exactly.
@pytest.mark.parametrize("num_tiles", [23, 25, 641, 643])
@pytest.mark.parametrize("s", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_tcuscan_sort_awkward_tile_counts(num_tiles: int, s: int, dtype: torch.dtype):
    vec_len = num_tiles * s * s
    x = torch.rand(vec_len, dtype=dtype, device=NPU_DEVICE) - 0.5

    torch.npu.synchronize()
    values, indices = tcuscan_ops.run_radix_sort(x, s)
    torch.npu.synchronize()

    expected, _ = torch.sort(x)
    assert torch.equal(values, expected)
    # Ties make the index set itself ambiguous, so check the pairing instead.
    assert torch.equal(x[indices.long()], values)
