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

_COPY_SIZES = [
    10 * 1024,
    20 * 1024,
    30 * 1024,
    40 * 1024,
    80 * 1024,
    1024 * 1024,
    80 * 1024 - 10,
]


def get_profiling_lengths():
    num_cores = 1
    s = 128
    max_size = 1e8
    max_iters = ceil(max_size / (num_cores * s * s))

    return [i * num_cores * s * s for i in range(1, max_iters, s * s)]


def _test_tcuscan_copy(length: int, dtype: torch.dtype, s: int):
    x = torch.rand(length, device="cpu", dtype=dtype)
    x_npu = x.npu()
    output = tcuscan_ops.run_copy(x_npu, s)
    cpuout = torch.clone(x).npu()

    assert output.shape == cpuout.shape, "Output shape does not match expected shape."
    assert torch.equal(output, cpuout)


@pytest.mark.parametrize("length", _COPY_SIZES)
def test_tcuscan_copy_fp32(length: int):
    _test_tcuscan_copy(length, torch.float32, 128)


@pytest.mark.parametrize("length", get_profiling_lengths())
def test_tcuscan_copy_fp32_profiling_lens(length: int):
    _test_tcuscan_copy(length, torch.float32, 128)
