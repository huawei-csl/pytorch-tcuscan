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


def _test_tcuscan_gen_lower(matrix_size: int, dtype: torch.dtype):
    x = torch.tril(
        torch.ones((matrix_size, matrix_size), dtype=dtype, device=NPU_DEVICE)
    )
    torch.npu.synchronize()
    output = tcuscan_ops.run_gen_lower(matrix_size, x.device, dtype)
    torch.npu.synchronize()
    torch.npu.synchronize()
    assert output.shape == x.shape, "Output shape does not match expected shape."
    assert torch.equal(output, x)


@pytest.mark.parametrize("matrix_size", [32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.int8, torch.half], ids=str)
def test_tcuscan_gen_lower(matrix_size: int, dtype: torch.dtype):
    _test_tcuscan_gen_lower(matrix_size, dtype)
