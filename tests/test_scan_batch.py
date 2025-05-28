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
_BATCH_SIZES = [2, 16, 20, 32, 44]


def get_lengths_and_batch_sizes(s: int, multipliers: List[int], batch_sizes: List[int]):
    for multiplier in multipliers:
        for batch_size in batch_sizes:
            yield (multiplier * s * s, batch_size)


def _test_scan_batch_fp16(s: int, vec_len: int, batch_size: int):

    x = torch.rand((batch_size, vec_len), dtype=torch.float16, device=NPU_DEVICE)

    torch.npu.synchronize()
    expected = torch.cumsum(x, dim=1, dtype=torch.float)
    torch.npu.synchronize()
    actual = tcuscan_ops.run_scan_batch(x, s)
    torch.npu.synchronize()

    assert torch.allclose(
        actual, expected, rtol=1e-03
    ), f"batch scan (fp16) wrong. s={s}, vec_len={vec_len}, batch_size= {batch_size}"


@pytest.mark.parametrize(
    "vec_len, batch_size",
    get_lengths_and_batch_sizes(
        s=16, multipliers=_MULTIPLIERS, batch_sizes=_BATCH_SIZES
    ),
)
def test_scan_batch_fp16_s_16(vec_len: int, batch_size: int):
    _test_scan_batch_fp16(16, vec_len, batch_size)


@pytest.mark.parametrize(
    "vec_len, batch_size",
    get_lengths_and_batch_sizes(
        s=32, multipliers=_MULTIPLIERS, batch_sizes=_BATCH_SIZES
    ),
)
def test_scan_batch_fp16_s_32(vec_len: int, batch_size: int):
    _test_scan_batch_fp16(32, vec_len, batch_size)


@pytest.mark.parametrize(
    "vec_len, batch_size",
    get_lengths_and_batch_sizes(
        s=64, multipliers=_MULTIPLIERS, batch_sizes=_BATCH_SIZES
    ),
)
def test_scan_batch_fp16_s_64(vec_len: int, batch_size: int):
    _test_scan_batch_fp16(64, vec_len, batch_size)


@pytest.mark.parametrize(
    "vec_len, batch_size",
    get_lengths_and_batch_sizes(
        s=128, multipliers=_MULTIPLIERS, batch_sizes=_BATCH_SIZES
    ),
)
def test_scan_batch_fp16_s_128(vec_len: int, batch_size: int):
    _test_scan_batch_fp16(128, vec_len, batch_size)
