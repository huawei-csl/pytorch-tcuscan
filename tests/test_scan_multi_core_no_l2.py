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


def get_lengths(s: int, max_iters: int):
    NUM_AI_CORES = 20
    for multiplier in range(1, max_iters):
        yield multiplier * NUM_AI_CORES * s * s


def _test_dtype(vec_len: int, s: int, dtype: torch.dtype):
    out_dtype = None
    if dtype == torch.float16:
        x = torch.randn(vec_len, dtype=dtype, device=NPU_DEVICE)
        out_dtype = torch.float32
    elif dtype == torch.float32:
        x = torch.randn(vec_len, dtype=dtype, device=NPU_DEVICE)
        out_dtype = torch.float32
    elif dtype == torch.int8:
        x = torch.randint(0, 10, size=(vec_len,), dtype=dtype, device=NPU_DEVICE)
        out_dtype = torch.int32
    else:
        assert False, "Unsupported dtype for MCSCAN. Got {dtype}."

    torch.npu.synchronize()
    expected = torch.cumsum(x, dim=-1, dtype=out_dtype)
    torch.npu.synchronize()
    actual = tcuscan_ops.run_scan_multi_core_no_l2(x, s)
    torch.npu.synchronize()
    assert actual.dtype == expected.dtype
    assert torch.allclose(
        actual, expected, atol=1e-02
    ), f"multi-core scan ({dtype}) is wrong. s={s}, vec_len={vec_len}"


@pytest.mark.parametrize("vec_len", get_lengths(s=16, max_iters=16))
@pytest.mark.parametrize("s", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.int8, torch.float32], ids=str)
def test_mcscan(vec_len: int, s: int, dtype: torch.dtype):
    _test_dtype(vec_len, s, dtype)


@pytest.mark.parametrize("vec_len", get_lengths(s=32, max_iters=12))
def test_mcscan_fp16_s_16(vec_len: int):
    _test_dtype(vec_len, 16, torch.float16)
