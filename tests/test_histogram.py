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
    x = torch.randint(-(2**7) + 1, 2**7 - 1, (length,), dtype=dtype, device=NPU_DEVICE)

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
@pytest.mark.parametrize(
    "hist_bins", [5, 8, 16, 32, 24], ids=lambda x: f"hist_bins:{x}"
)
def test_tcuscan_histogram_fp16(length: int, hist_bins: int):
    _test_tcuscan_histogram(length, hist_bins, torch.float16)


# --------------------------------------------------------------------------------
# Full-precision reference
#
# `torch.histogram` is not a sound oracle for this kernel: it assigns bins by
# scaling and truncating in float32 rather than by comparing against its own bin
# edges, so it carries rounding of its own and disagrees with exact arithmetic on
# some inputs.  The reference below instead reproduces the kernel's *contract*:
# the kernel derives its bin edges in float32 as `x_max - i * step`, and must then
# bin every element as though it compared against those edges in full precision.
#
# Comparing in float64 is exact here, because every float16 input and every
# float32 edge is representable in float64.  A disagreement therefore always means
# the kernel's comparison is wrong -- which is what happened when the bin edge was
# rounded to float16 with round-to-nearest: inputs sitting exactly on an edge that
# rounded *down* were pushed one bin up.
# --------------------------------------------------------------------------------


def reference_histogram(x: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Counts the kernel must produce for ``x``, computed exactly.

    Mirrors the kernel's own float32 bin edges but performs every comparison in
    float64, so the result is the mathematically correct count for those edges.
    """
    values = x.to(torch.float64).cpu().numpy()
    x_min = np.float32(values.min())
    x_max = np.float32(values.max())
    step = np.float32((x_max - x_min) / np.float32(num_bins))

    counts = np.zeros(num_bins, dtype=np.int64)
    if num_bins == 1:
        counts[0] = values.size
        return torch.from_numpy(counts).to(torch.int32)

    # The kernel probes edge `x_max - i * step` and credits the result to bin
    # `num_bins - i`, so the lower edge of bin `j` is `x_max - (num_bins - j) *
    # step` for j = 1 .. num_bins - 1. Ascending in j, as searchsorted requires.
    j = np.arange(1, num_bins, dtype=np.float32)
    edges = (x_max - (np.float32(num_bins) - j) * step).astype(np.float64)

    # `side="left"` yields, for each edge, the number of inputs strictly below it.
    below = np.searchsorted(np.sort(values), edges, side="left")

    counts[0] = below[0]
    counts[1:-1] = np.diff(below)
    counts[-1] = values.size - below[-1]
    return torch.from_numpy(counts).to(torch.int32)


def _make_input(length: int, distribution: str) -> torch.Tensor:
    """Input tensors for the sweeps below.

    ``integers`` is the case that exposed the bin-edge bug: many elements land
    exactly on a bin edge, so any error in the edge comparison moves counts
    between neighbouring bins.
    """
    if distribution == "integers":
        return torch.randint(
            -(2**7) + 1, 2**7 - 1, (length,), dtype=torch.float16, device=NPU_DEVICE
        )
    return 10 * torch.rand(length, dtype=torch.float16, device=NPU_DEVICE) - 5


# Bin counts deliberately include values that do *not* divide the value range
# cleanly (3, 7, 13, 24, 100, 255); those are the ones that produce a bin edge
# whose float16 rounding falls below the true edge.
_REF_BINS = [1, 2, 3, 5, 7, 8, 13, 16, 24, 32, 64, 100, 128, 255, 256]

# Ragged lengths exercise the partial-tile path (tile_len is 16384).
_REF_SIZES = [1024, 16385, 100003, 1024 * 1024 + 13]


@pytest.mark.parametrize("length", _REF_SIZES, ids=lambda x: f"length:{x:,}")
@pytest.mark.parametrize("hist_bins", _REF_BINS, ids=lambda x: f"hist_bins:{x}")
@pytest.mark.parametrize("distribution", ["integers", "uniform"], ids=str)
def test_tcuscan_histogram_matches_reference(
    length: int, hist_bins: int, distribution: str
):
    """The kernel must match exact-precision binning of its own float32 edges."""
    x = _make_input(length, distribution)
    torch.npu.synchronize()
    actual = tcuscan_ops.run_histogram(x, hist_bins)
    torch.npu.synchronize()

    expected = reference_histogram(x, hist_bins)

    assert actual.cpu().sum().item() == length, (
        f"Counts must sum to the input length. Expected {length}, "
        f"got {actual.cpu().sum().item()}."
    )
    assert torch.equal(expected, actual.cpu()), f"Diff {expected - actual.cpu()}"


@pytest.mark.parametrize(
    "length", [16 * 1024 * 1024 + 1], ids=lambda x: f"length:{x:,}"
)
@pytest.mark.parametrize("hist_bins", [8, 64], ids=lambda x: f"hist_bins:{x}")
def test_tcuscan_histogram_matches_reference_large(length: int, hist_bins: int):
    """Same contract on an input spanning many tiles and all vector cores."""
    x = _make_input(length, "integers")
    torch.npu.synchronize()
    actual = tcuscan_ops.run_histogram(x, hist_bins)
    torch.npu.synchronize()

    expected = reference_histogram(x, hist_bins)
    assert torch.equal(expected, actual.cpu()), f"Diff {expected - actual.cpu()}"


@pytest.mark.parametrize("hist_bins", [8, 32, 100], ids=lambda x: f"hist_bins:{x}")
def test_tcuscan_histogram_is_deterministic(hist_bins: int):
    """Repeated launches on identical input must agree bit for bit.

    The kernel synchronises the vector and scalar pipes with explicit events
    (`PIPE_V` barriers plus V_S / S_V flags) rather than a blanket `PIPE_ALL`.
    A missing or mis-scoped sync typically shows up as a result that varies
    between launches rather than one that is consistently wrong, so a single
    comparison against a reference would not reliably catch it.
    """
    x = _make_input(1024 * 1024 + 13, "integers")
    torch.npu.synchronize()
    first = tcuscan_ops.run_histogram(x, hist_bins).cpu()
    torch.npu.synchronize()

    for _ in range(8):
        again = tcuscan_ops.run_histogram(x, hist_bins).cpu()
        torch.npu.synchronize()
        assert torch.equal(first, again), f"Diff {first - again}"
