# --------------------------------------------------------------------------------
# Copyright (c) 2023-2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pytorch-tcuscan/
# for the full License text.
# --------------------------------------------------------------------------------
import os
import random
from typing import List

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

_MULTIPLIERS = [1, 2, 3, 11, 63]
_BATCH_SIZES = [2, 16, 20, 32, 44, 256]
_S = [32, 64, 128]


def get_lengths_and_batch_sizes(
    matmul_sizes: List[int], multipliers: List[int], batch_sizes: List[int]
):
    for s in matmul_sizes:
        for multiplier in multipliers:
            for batch_size in batch_sizes:
                yield (s, multiplier * s * s, batch_size)
                yield (s, multiplier * s * s // 2, batch_size)
                yield (s, multiplier * s * s // 4, batch_size)
                yield (s, multiplier * s * s - 1, batch_size)
                yield (s, multiplier * s * s + 1, batch_size)


def _test_scan_batch(s: int, vec_len: int, batch_size: int, dtype: torch.dtype):

    x = torch.rand((batch_size, vec_len), dtype=dtype, device=NPU_DEVICE)

    torch.npu.synchronize()
    expected = torch.cumsum(x, dim=1, dtype=torch.float)
    torch.npu.synchronize()
    actual = tcuscan_ops.run_scan_batch(x, s)
    torch.npu.synchronize()

    assert torch.allclose(
        actual, expected, rtol=1e-03
    ), f"batch scan ({dtype}) wrong. s={s}, vec_len={vec_len}, batch_size= {batch_size}"


@pytest.mark.parametrize(
    "s, vec_len, batch_size",
    get_lengths_and_batch_sizes(
        matmul_sizes=_S, multipliers=_MULTIPLIERS, batch_sizes=_BATCH_SIZES
    ),
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=str)
def test_scan_batch(s: int, vec_len: int, batch_size: int, dtype: torch.dtype):
    _test_scan_batch(s, vec_len, batch_size, dtype)
