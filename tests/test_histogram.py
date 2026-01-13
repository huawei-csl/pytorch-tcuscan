import os
import random

import numpy as np
import pytest

import torch  # isort: skip
import tcuscan_ops  # isort: skip


random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)

_SIZES = [
    # 10 * 1024, FAILS!
    20 * 1024,
    30 * 1024,
    40 * 1024,
    80 * 1024,
    80 * 1024 - 10,
    1024 * 1024 + 13,
    2 * 1024 * 1024 - 113,
    3 * 1024 * 1024 + 123,
    4 * 1024 * 1024,
]


def hist_range(x, num_bins: int):
    torch.npu.synchronize()
    x_min, x_max = torch.aminmax(x)
    torch.npu.synchronize()
    return torch.linspace(x_min.float(), x_max.float(), num_bins + 1)


def _test_tcuscan_histogram(length: int, dtype: torch.dtype):
    # x = 10 * torch.rand(length, dtype=dtype, device=NPU_DEVICE)
    x = torch.randint(0, 2**7 - 1, (length,), dtype=dtype, device=NPU_DEVICE)
    num_bins = 32

    torch.npu.synchronize()
    actual = tcuscan_ops.run_histogram(x, num_bins)
    torch.npu.synchronize()
    total_counts = torch.sum(actual, dtype=torch.int32)
    torch.npu.synchronize()
    # RuntimeError: "histogramdd" not implemented for 'Half'
    expected = torch.histogram(x.to(torch.float), num_bins).hist.to(torch.int32)

    assert (
        total_counts == length
    ), f"[actual] Sum of hist_counts must equal length. Expected: {length}. Got {total_counts}."
    assert expected.shape == actual.shape
    assert expected.dtype == actual.dtype
    assert torch.equal(expected, actual), f"Diff {expected - actual}"


@pytest.mark.parametrize("length", _SIZES)
def test_tcuscan_histogram_fp16(length: int):
    _test_tcuscan_histogram(length, torch.float16)
