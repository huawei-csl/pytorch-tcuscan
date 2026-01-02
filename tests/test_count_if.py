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

_SIZES = [1024 * 8, 1024 * 16, 2048 * 32, 1024 * 128]


def _test_tcuscan_count_if(length: int, dtype: torch.dtype, tile_len: int):
    x = torch.randn(length, device="cpu", dtype=dtype)
    x_npu = x.npu()

    pivot = 0.1
    output = tcuscan_ops.run_count_if(x_npu, pivot, tile_len)
    actual = torch.sum(output).cpu()
    expected = torch.count_nonzero(x <= pivot)

    assert torch.equal(actual, expected), f"Expected {expected}. Got {actual}"


@pytest.mark.parametrize("length", _SIZES)
@pytest.mark.parametrize("tile_len", [128, 1024])
def test_tcuscan_count_if_fp16(length: int, tile_len: int):
    _test_tcuscan_count_if(length, torch.float16, tile_len)
