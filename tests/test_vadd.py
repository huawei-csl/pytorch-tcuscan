#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import random

import numpy as np
import pytest
import torch_npu  # noqa

import tcuscan_ops
import torch

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

torch.npu.config.allow_internal_format = False

VEC_LENS = [
    256,
    1024,
    2048,
    4096,
    8192,
    1024 * 1024,
    256,
    1024,
    2048,
    4096,
    8192,
    1024 * 1024,
    1024 * 1024,
    1024 * 1024,
]


@pytest.mark.parametrize("vec_len", VEC_LENS)
def test_vadd(vec_len: int):
    x = torch.rand(vec_len, device="cpu", dtype=torch.float16)
    y = torch.rand(vec_len, device="cpu", dtype=torch.float16)

    x_npu = x.npu()
    y_npu = y.npu()
    output = tcuscan_ops.run_add(x_npu, y_npu)
    cpuout = torch.add(x, y).npu()

    assert output.shape == cpuout.shape, "Output shape does not match expected shape."
    assert torch.allclose(output, cpuout)
