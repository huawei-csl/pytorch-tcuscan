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
import torch
import torch_npu  # noqa

import tcuscan_ops

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

torch.npu.config.allow_internal_format = False


def test_vadd():
    length = [40 * 2048]
    x = torch.rand(length, device="cpu", dtype=torch.float16)
    y = torch.rand(length, device="cpu", dtype=torch.float16)

    x_npu = x.npu()
    y_npu = y.npu()
    output = tcuscan_ops.run_add(x_npu, y_npu)
    cpuout = torch.add(x, y).npu()

    assert output.shape == cpuout.shape, "Output shape does not match expected shape."
    assert torch.allclose(output, cpuout)
