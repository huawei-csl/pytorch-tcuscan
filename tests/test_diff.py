# --------------------------------------------------------------------------------
# Copyright (c) 2023-2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pytorch-tcuscan/
# for the full License text.
# --------------------------------------------------------------------------------
#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

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
    max_size = ceil(0.5 * 1e8)
    max_iters = ceil(max_size / (num_cores * s * s))

    return [i * num_cores * s * s for i in range(1, max_iters, 16 * 128 // s)]


def _test_tcuscan_diff(length: int, dtype: torch.dtype):
    x = torch.rand(length, device="cpu", dtype=dtype)

    x_cpu = torch.concat([torch.zeros(1, dtype=dtype), x])
    x_npu = x.npu()
    torch.npu.synchronize()
    output = tcuscan_ops.run_diff(x_npu)
    torch.npu.synchronize()
    cpuout = torch.diff(x_cpu).npu()
    torch.npu.synchronize()

    assert output.shape == cpuout.shape, "Output shape does not match expected shape."
    assert torch.equal(output, cpuout)


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
