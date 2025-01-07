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

torch.npu.config.allow_internal_format = False


def test_tcuscan_diff():
    length = [40 * 1024]
    x = torch.rand(length, device="cpu", dtype=torch.float16)

    x_cpu = torch.concat([torch.zeros(1, dtype=torch.float16), x])
    x_npu = x.npu()
    output = tcuscan_ops.run_diff(x_npu)
    cpuout = torch.diff(x_cpu).npu()

    assert output.shape == cpuout.shape, "Output shape does not match expected shape."
    assert torch.allclose(output, cpuout)
