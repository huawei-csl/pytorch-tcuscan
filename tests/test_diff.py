#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import torch
import torch_npu  # noqa

import tcuscan_ops
import pytest
from math import ceil


torch.npu.config.allow_internal_format = False


_DIFF_SIZES = [
    10 * 1024,
    20 * 1024,
    30 * 1024,
    40 * 1024,
    80 * 1024,
    1024 * 1024,
    80 * 1024 - 10,
]


def get_profiling_lengths():
    num_cores = 20
    s = 64
    max_size = 1e8
    max_iters = ceil(max_size / (num_cores * s * s))

    return [i * num_cores * s * s for i in range(1, max_iters, 16 * 128 // s)]


def _test_tcuscan_diff(length: int, dtype: torch.dtype):
    x = torch.rand(length, device="cpu", dtype=dtype)

    x_cpu = torch.concat([torch.zeros(1, dtype=dtype), x])
    x_npu = x.npu()
    output = tcuscan_ops.run_diff(x_npu)
    cpuout = torch.diff(x_cpu).npu()

    assert output.shape == cpuout.shape, "Output shape does not match expected shape."
    assert torch.allclose(output, cpuout)


@pytest.mark.parametrize("length", _DIFF_SIZES)
def test_tcuscan_diff_fp16(length: int):
    _test_tcuscan_diff(length, torch.float16)


@pytest.mark.parametrize("length", _DIFF_SIZES)
def test_tcuscan_diff_fp32(length: int):
    _test_tcuscan_diff(length, torch.float32)


@pytest.mark.parametrize("length", get_profiling_lengths())
def test_tcuscan_diff_fp16_profiling_lens(length: int):
    _test_tcuscan_diff(length, torch.float16)


@pytest.mark.parametrize("length", get_profiling_lengths())
def test_tcuscan_diff_fp32_profiling_lens(length: int):
    _test_tcuscan_diff(length, torch.float32)
