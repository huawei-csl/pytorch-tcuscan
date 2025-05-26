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

_ALIGN_SIZES = [128, 256, 512, 1024, 2048, 4096]
_OFFSET_SIZES = [7, 11, 23, 18, 1050, 2050, 4050]
_MULTIPLIER = [1, 2, 13, 17]


def _test_simple_pad(tile_len: int, offset: int, multiplier: int, dtype: torch.dtype):
    vec_len = multiplier * tile_len * tile_len - offset
    x = torch.rand(vec_len, device="cpu", dtype=dtype)
    x_npu = x.npu()
    torch.npu.synchronize()

    output = tcuscan_ops.run_simple_pad(x_npu, tile_len)
    torch.npu.synchronize()

    last_tail = tile_len - offset
    last_tail_offset = vec_len - last_tail
    last_tile = x[last_tail_offset:]
    cpuout = output.cpu()[:-offset]
    torch.npu.synchronize()

    print("x Shape: ", x.shape)
    print("cpuout Shape: ", cpuout.shape)

    assert len(output.cpu()) == tile_len
    assert (
        last_tile.shape == cpuout.shape
    ), "Output shape does not match expected shape."
    assert torch.equal(last_tile, cpuout)


@pytest.mark.parametrize("multiplier", _MULTIPLIER)
@pytest.mark.parametrize("tile_len", _ALIGN_SIZES)
@pytest.mark.parametrize("offset", _OFFSET_SIZES)
def test_simple_pad_fp16(multiplier: int, tile_len: int, offset: int):
    _test_simple_pad(tile_len, offset, multiplier, torch.float16)
