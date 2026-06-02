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
    10 * 1024,
    20 * 1024,
    30 * 1024,
    40 * 1024,
    80 * 1024 + 13,
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


def _test_tcuscan_histogram(length: int, hist_bins: int, dtype: torch.dtype):
    # x = 10 * torch.rand(length, dtype=dtype, device=NPU_DEVICE)
    x = torch.randint(-2**7 +1, 2**7 - 1, (length,), dtype=dtype, device=NPU_DEVICE)

    torch.npu.synchronize()
    actual = tcuscan_ops.run_histogram(x, hist_bins)
    torch.npu.synchronize()
    total_counts = torch.sum(actual, dtype=torch.int32)
    torch.npu.synchronize()
    # RuntimeError: "histogramdd" not implemented for 'Half'
    expected = torch.histogram(x.to(torch.float), hist_bins).hist.to(torch.int32)

    assert (
        total_counts == length
    ), f"[actual] Sum of hist_counts must equal length. Expected: {length}. Got {total_counts}."
    assert expected.shape == actual.shape
    assert expected.dtype == actual.dtype
    assert torch.equal(expected, actual), f"Diff {expected - actual}"


@pytest.mark.parametrize("length", _SIZES, ids=lambda x: f"length:{x:,}")
@pytest.mark.parametrize("hist_bins", [5, 8, 16, 32, 24], ids=lambda x: f"hist_bins:{x}")
def test_tcuscan_histogram_fp16(length: int, hist_bins: int):
    _test_tcuscan_histogram(length, hist_bins, torch.float16)
