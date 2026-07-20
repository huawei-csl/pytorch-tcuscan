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
torch.npu.set_device(NPU_DEVICE)


def _test_single_core_scan(vec_len: int, s: int, dtype: torch.dtype):
    out_dtype = None
    if dtype == torch.float16:
        x = torch.randint(
            0, 6, size=(vec_len,), dtype=torch.int8, device=NPU_DEVICE
        ).half()
        out_dtype = torch.float32
    elif dtype == torch.float32:
        x = torch.randint(
            0, 6, size=(vec_len,), dtype=torch.int8, device=NPU_DEVICE
        ).float()
        out_dtype = torch.float32
    elif dtype == torch.int8:
        x = torch.randint(0, 2, size=(vec_len,), dtype=torch.int8, device=NPU_DEVICE)
        out_dtype = torch.int32
    else:
        assert False, f"SingleCoreScan supports only fp16/fp32/int8. Got {dtype}."

    torch.npu.synchronize()
    x_cpu = x.cpu()
    expected = torch.cumsum(x_cpu, dim=-1, dtype=out_dtype)
    torch.npu.synchronize()
    actual = tcuscan_ops.run_scan_single_core(x, s)
    torch.npu.synchronize()
    max_abs_error = torch.max(torch.abs(expected - actual.cpu()))
    max_rel_error = torch.max(torch.abs(expected - actual.cpu()) / torch.abs(expected))
    print(f"Max abs error: {max_abs_error:.5f}")
    print(f"Max rel error: {max_rel_error:.5f}")

    expected_double = torch.cumsum(x_cpu.double(), dim=-1)
    expected_half = torch.cumsum(x_cpu, dim=-1, dtype=out_dtype)
    assert torch.equal(expected_double, expected_half.double())
    assert torch.allclose(
        expected, actual.cpu(), atol=0, rtol=1e-2
    ), f"single-core scan ({dtype}) is wrong. s={s}, vec_len={vec_len}"


@pytest.mark.parametrize("multiplier", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
@pytest.mark.parametrize("s", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.int8, torch.float16, torch.float32], ids=str)
@pytest.mark.parametrize("offset", [1, 3, 7, 17, 23, 33, 53, 113])
def test_single_core_scan_minus(
    offset: int, multiplier: int, s: int, dtype: torch.dtype
):
    vec_len = multiplier * s * s - offset
    _test_single_core_scan(vec_len, s, dtype)


@pytest.mark.parametrize("multiplier", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
@pytest.mark.parametrize("s", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.int8, torch.float16, torch.float32], ids=str)
@pytest.mark.parametrize("offset", [1, 3, 7, 17, 23, 33, 53, 113])
def test_single_core_scan_plus(
    offset: int, multiplier: int, s: int, dtype: torch.dtype
):
    vec_len = multiplier * s * s + offset
    _test_single_core_scan(vec_len, s, dtype)
