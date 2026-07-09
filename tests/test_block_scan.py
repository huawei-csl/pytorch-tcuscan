# --------------------------------------------------------------------------------
# Copyright (c) 2023-2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pytorch-tcuscan/
# for the full License text.
# --------------------------------------------------------------------------------
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


def _test_block_scan(m: int, s: int, dtype: torch.dtype):
    matmul_tile = s * s
    A = 0.1 * torch.randn(m, matmul_tile, dtype=dtype).npu()
    B = torch.triu(torch.ones((matmul_tile, matmul_tile), dtype=dtype)).npu()

    ones = torch.ones((s, s), dtype=dtype, device=NPU_DEVICE)
    upper = torch.triu(ones)
    lower_strict = torch.tril(ones, -1)
    torch.npu.synchronize()

    actual = tcuscan_ops.run_block_scan(A.flatten(), upper, lower_strict)
    expected = torch.matmul(A.float(), B.float()).flatten()

    assert actual.dtype == expected.dtype
    assert torch.allclose(
        actual, expected, atol=1e-2, rtol=1e-3
    ), f"Block scan ({dtype}) is wrong. s={s}, m={m}"


@pytest.mark.parametrize("m", range(10, 21))
@pytest.mark.parametrize("s", [16, 32, 64])
def test_tcuscan_block_scan_fp16(m: int, s: int):
    _test_block_scan(m, s, torch.float16)


@pytest.mark.parametrize("m", range(20, 25))
def test_tcuscan_block_scan_fp16_s_128(m: int):
    _test_block_scan(m, 128, torch.float16)
