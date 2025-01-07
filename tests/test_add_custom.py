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

import add_custom

torch.npu.config.allow_internal_format = False


def test_add_custom_ops():
    length = [40 * 2048]
    x = torch.rand(length, device="cpu", dtype=torch.float16)
    y = torch.rand(length, device="cpu", dtype=torch.float16)

    x_npu = x.npu()
    y_npu = y.npu()
    output = add_custom.run_add_custom(x_npu, y_npu)
    cpuout = torch.add(x, y).npu()

    assert output.shape == cpuout.shape, "Output shape does not match expected shape."
    assert torch.allclose(output, cpuout)
