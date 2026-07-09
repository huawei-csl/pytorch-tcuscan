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

NUM_AI_CORES = 20


def get_lengths(s: int, max_iters: int):
    for multiplier in range(1, max_iters):
        yield multiplier * NUM_AI_CORES * s * s


def _test_dtype(vec_len: int, s: int, dtype: torch.dtype):

    ones = torch.ones((s, s), dtype=dtype, device=NPU_DEVICE)
    upper = torch.triu(ones)
    lower_strict = torch.tril(ones, -1)
    torch.npu.synchronize()

    out_dtype = None
    if dtype == torch.float16:
        x = torch.rand(vec_len, dtype=dtype, device=NPU_DEVICE)
        out_dtype = torch.float32
    elif dtype == torch.int8:
        x = torch.randint(0, 10, size=(vec_len,), dtype=dtype, device=NPU_DEVICE)
        out_dtype = torch.int32
    else:
        assert False, f"Unsupported dtype for MCSCAN. Got {dtype}."

    torch.npu.synchronize()
    expected = torch.cumsum(x, dim=-1, dtype=out_dtype)
    torch.npu.synchronize()
    actual = tcuscan_ops.run_scan_multi_cube(x, upper, lower_strict)
    torch.npu.synchronize()

    abs_error = torch.max(torch.abs(actual - expected))
    rel_error = torch.max(torch.abs(actual - expected) / torch.abs(actual))
    assert actual.dtype == expected.dtype
    assert torch.allclose(
        actual, expected, atol=0, rtol=1e-2
    ), f"multi-cube scan ({dtype}) is wrong. len / s : {vec_len} / {s}. Abs/rel error: {abs_error:.5f} / {rel_error:.7f}"


@pytest.mark.parametrize("multiplier", range(1, 11))
@pytest.mark.parametrize("s", [16])
@pytest.mark.parametrize("dtype", [torch.float16], ids=str)
def test_scan_multi_cube_s_16(multiplier: int, s: int, dtype: torch.dtype):
    vec_len = multiplier * NUM_AI_CORES * s * s
    _test_dtype(vec_len, s, dtype)


@pytest.mark.parametrize("multiplier", range(2, 10))
@pytest.mark.parametrize("s", [32])
@pytest.mark.parametrize("dtype", [torch.float16], ids=str)
def test_scan_multi_cube_s_32(multiplier: int, s: int, dtype: torch.dtype):
    vec_len = multiplier * NUM_AI_CORES * s * s
    _test_dtype(vec_len, s, dtype)


@pytest.mark.parametrize("multiplier", range(2, 10))
@pytest.mark.parametrize("s", [64])
@pytest.mark.parametrize("dtype", [torch.float16], ids=str)
def test_scan_multi_cube_s_64(multiplier: int, s: int, dtype: torch.dtype):
    vec_len = multiplier * NUM_AI_CORES * s * s
    _test_dtype(vec_len, s, dtype)


@pytest.mark.parametrize("multiplier", range(1, 10))
@pytest.mark.parametrize("s", [128])
@pytest.mark.parametrize("dtype", [torch.float16], ids=str)
@pytest.mark.parametrize("offset", [5, 13, 23, 33])
def test_scan_multi_cube_s_128(
    multiplier: int, s: int, dtype: torch.dtype, offset: int
):
    vec_len = multiplier * NUM_AI_CORES * s * s - offset
    _test_dtype(vec_len, s, dtype)


# Uncomment this test to verify that the input length used in profiling are correct.
# @pytest.mark.parametrize("multiplier", range(1, 39))
# @pytest.mark.parametrize("s", [128])
# @pytest.mark.parametrize("dtype", [torch.float16], ids=str)
# def test_scan_multi_cube_profiling_lengths(multiplier: int, s: int, dtype: torch.dtype):
#     vec_len = multiplier * NUM_AI_CORES * s * s
#     _test_dtype(vec_len, s, dtype)
