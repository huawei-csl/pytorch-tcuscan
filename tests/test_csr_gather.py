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


_CSR_GATHER_SIZES = [
    10 * 1024,
    20 * 1024,
    30 * 1024,
    40 * 1024,
    80 * 1024,
    1024 * 1024,
]


@pytest.mark.parametrize("length", _CSR_GATHER_SIZES)
def test_tcuscan_csr_gather(length: int):

    max_x_len = 1024

    input_values = torch.randn(length).half().npu()
    input_cols = torch.randint(
        low=0, high=max_x_len, size=(length,), dtype=torch.int32
    ).npu()
    input_x = torch.randn(max_x_len).half().npu()

    gathered_x = input_x[input_cols]
    expected = input_values * gathered_x
    torch.npu.synchronize()

    actual = tcuscan_ops.run_csr_gather(input_values, input_cols, input_x)
    torch.npu.synchronize()

    assert actual.shape == expected.shape, "Output shape does not match expected shape."
    assert torch.allclose(actual.float(), expected.float())
