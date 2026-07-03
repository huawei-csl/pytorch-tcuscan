#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import os
from math import ceil

import pytest
import torch_npu  # noqa

import tcuscan_ops
import torch

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)

_CSR_GATHER_SIZES = [
    16 * 1024,
    20 * 1024,
    30 * 1024,
    40 * 1024,
    60 * 1024,
    61 * 1024,
    61 * 1024 - 13,
    62 * 1024 - 123,
    63 * 1024 - 7,
    66 * 1024 - 1,
    67 * 1024 + 1,
    68 * 1024 - 33,
    80 * 1024 - 63,
    1024 * 1024 - 31,
]

_X_LENS = [256, 16384, 16385, 170013, 1360000]


def get_profiling_lengths():
    num_cores = 20
    s = 128
    max_size = 1e8
    max_iters = ceil(max_size / (num_cores * s * s))

    return [i * num_cores * s * s for i in range(1, max_iters, 16 * 128 // s)]


def ref_csr_gather(values, cols, input_x):
    return values * input_x[cols]


def _test_tcuscan_csr_gather(col_len, x_len: int, in_dtype: torch.dtype):
    input_values = torch.randn(col_len, dtype=in_dtype, device=NPU_DEVICE)
    input_cols = torch.randint(
        low=0, high=x_len, size=(col_len,), dtype=torch.int32, device=NPU_DEVICE
    )
    input_x = torch.randn(x_len, dtype=in_dtype, device=NPU_DEVICE)

    expected = ref_csr_gather(input_values, input_cols, input_x)
    torch.npu.synchronize()
    actual = tcuscan_ops.run_csr_gather(input_values, input_cols, input_x)
    torch.npu.synchronize()

    assert actual.shape == expected.shape, "Output shape does not match expected shape."
    assert torch.equal(actual.float(), expected.float())


@pytest.mark.parametrize("col_len", _CSR_GATHER_SIZES)
@pytest.mark.parametrize("x_len", _X_LENS)
@pytest.mark.parametrize("dtype", [torch.int16, torch.float16], ids=str)
def test_tcuscan_csr_gather(col_len: int, x_len: int, dtype: torch.dtype):
    _test_tcuscan_csr_gather(col_len, x_len, dtype)


@pytest.mark.parametrize("col_len", get_profiling_lengths())
@pytest.mark.parametrize("x_len", _X_LENS)
@pytest.mark.parametrize("dtype", [torch.int16, torch.float16], ids=str)
def test_tcuscan_csr_gather_profiling_lens(
    col_len: int, x_len: int, dtype: torch.dtype
):
    _test_tcuscan_csr_gather(col_len, x_len, dtype)
