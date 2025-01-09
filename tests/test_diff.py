#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import torch
import torch_npu  # noqa

import tcuscan_ops
import pytest

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


@pytest.mark.parametrize("length", _DIFF_SIZES)
def test_tcuscan_diff(length: int):

    x = torch.rand(length, device="cpu", dtype=torch.float16)

    x_cpu = torch.concat([torch.zeros(1, dtype=torch.float16), x])
    x_npu = x.npu()
    output = tcuscan_ops.run_diff(x_npu)
    cpuout = torch.diff(x_cpu).npu()

    assert output.shape == cpuout.shape, "Output shape does not match expected shape."
    assert torch.allclose(output, cpuout)
