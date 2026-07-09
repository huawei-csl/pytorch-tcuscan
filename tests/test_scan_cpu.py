# --------------------------------------------------------------------------------
# Copyright (c) 2023-2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pytorch-tcuscan/
# for the full License text.
# --------------------------------------------------------------------------------
# isort: skip_file
import random

import numpy as np
import pytest

import torch
import torch_npu  # noqa
import tcuscan_ops

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def _test_dtype(vec_len: int, dtype: torch.dtype):
    out_dtype = None
    if dtype in {torch.float16, torch.float32}:
        x = 0.1 * torch.randn(vec_len, dtype=dtype)
        out_dtype = torch.float32
    elif dtype == torch.int8:
        x = torch.randint(0, 10, size=(vec_len,), dtype=dtype)
        out_dtype = torch.int32
    else:
        assert False, "Unsupported dtype for SCAN (cpu). Got {dtype}."

    expected = torch.cumsum(x, dim=-1, dtype=out_dtype)
    actual = tcuscan_ops.run_scan_cpu(x)

    abs_error = torch.max(torch.abs(actual - expected))
    rel_error = torch.max(torch.abs(actual - expected) / torch.abs(expected))
    assert actual.dtype == expected.dtype
    assert torch.allclose(
        actual, expected, atol=1e-3, rtol=1e-7
    ), f"Scan (CPU) ({dtype}) is wrong. vec_len={vec_len}. Abs/rel error: {abs_error:.5f} / {rel_error:.7f}"


@pytest.mark.parametrize("vec_len", [1024, 10 * 1024, 30 * 1024, 1_000_000])
@pytest.mark.parametrize("dtype", [torch.int8, torch.float16, torch.float32], ids=str)
def test_scan_cpu(vec_len: int, dtype: torch.dtype):
    _test_dtype(vec_len, dtype)
