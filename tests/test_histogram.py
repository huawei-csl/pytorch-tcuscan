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


def _edges_agree(expected: torch.Tensor, actual: torch.Tensor) -> bool:
    """Whether two sets of bin edges agree to within float32 rounding.

    Torch interpolates its edges with `linspace`, while the kernel walks down
    from the top of the range as `x_max - i * step`, so the two differ by an
    ulp on some ranges.  That is some four thousand times finer than the
    float16 resolution the counts are decided at, so it cannot move an element
    between bins; against the kernel's own contract the edges are compared
    exactly instead, in `_assert_matches_reference`.

    The tolerance is an absolute one taken from the widest edge: the rounding
    comes from arithmetic carried out at the scale of the range, so the edges
    near zero carry the same absolute error as the ones at its ends.
    """
    scale = max(expected.abs().max().item(), 1.0)
    return torch.allclose(expected, actual, rtol=0.0, atol=1e-6 * scale)


def _test_tcuscan_histogram(length: int, hist_bins: int, dtype: torch.dtype):
    x = torch.randint(-(2**7) + 1, 2**7 - 1, (length,), dtype=dtype, device=NPU_DEVICE)

    torch.npu.synchronize()
    actual, actual_edges = tcuscan_ops.run_histogram(x, hist_bins)
    torch.npu.synchronize()
    total_counts = torch.sum(actual)
    torch.npu.synchronize()
    # RuntimeError: "histogramdd" not implemented for 'Half'
    expected, expected_edges = torch.histogram(x.to(torch.float).cpu(), hist_bins)

    assert (
        total_counts == length
    ), f"[actual] Sum of hist_counts must equal length. Expected: {length}. Got {total_counts}."
    assert expected.shape == actual.shape
    assert expected.dtype == actual.dtype
    assert torch.equal(expected, actual.cpu()), f"Diff {expected - actual.cpu()}"
    assert _edges_agree(expected_edges, actual_edges.cpu())


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


def reference_range(x: torch.Tensor, hist_range=None):
    """The float32 range the kernel histograms over, as the wrapper derives it."""
    if hist_range is None:
        if x.numel() == 0:
            x_min, x_max = np.float32(0.0), np.float32(1.0)
        else:
            values = x.reshape(-1).to(torch.float64).cpu().numpy()
            x_min, x_max = np.float32(values.min()), np.float32(values.max())
    else:
        x_min, x_max = np.float32(hist_range[0]), np.float32(hist_range[1])

    # An empty range is widened by half a unit on each side, as torch does.
    if x_min == x_max:
        x_min, x_max = x_min - np.float32(0.5), x_max + np.float32(0.5)

    return x_min, x_max


def reference_bin_edges(x: torch.Tensor, num_bins: int, hist_range=None):
    """The edges the kernel bins against, in the float32 it computes them in.

    The kernel probes edge `x_max - i * step` and credits the result to bin
    `num_bins - i`, so edge `j` is `x_max - (num_bins - j) * step`.  The lowest
    edge is `x_min` itself rather than `x_max - num_bins * step`, which rounding
    would place a hair to either side of it.
    """
    x_min, x_max = reference_range(x, hist_range)
    step = np.float32((x_max - x_min) / np.float32(num_bins))

    edges = np.empty(num_bins + 1, dtype=np.float32)
    edges[0] = x_min
    j = np.arange(1, num_bins + 1, dtype=np.float32)
    edges[1:] = x_max - (np.float32(num_bins) - j) * step
    return edges


def reference_histogram(
    x: torch.Tensor, num_bins: int, hist_range=None
) -> torch.Tensor:
    """Counts the kernel must produce for ``x``, computed exactly.

    Mirrors the kernel's own float32 bin edges but performs every comparison in
    float64, so the result is the mathematically correct count for those edges.
    Bin `j` spans `[edges[j], edges[j + 1])`, the last bin is closed on both
    ends, and elements outside the range are not counted at all.
    """
    values = np.sort(x.reshape(-1).to(torch.float64).cpu().numpy())
    edges = reference_bin_edges(x, num_bins, hist_range).astype(np.float64)

    # `side="left"` yields, for each edge, the number of inputs strictly below
    # it; the closed upper end of the last bin takes `side="right"` instead.
    below = np.searchsorted(values, edges, side="left")
    counts = np.diff(below).astype(np.int64)
    counts[-1] = np.searchsorted(values, edges[-1], side="right") - below[-2]

    return torch.from_numpy(counts).to(torch.float32)


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


def _assert_matches_reference(x, hist_bins, hist_range=None, expect_sum=None):
    """Counts and edges must both match the exact-precision reference."""
    torch.npu.synchronize()
    if hist_range is None:
        actual, actual_edges = tcuscan_ops.run_histogram(x, hist_bins)
    else:
        actual, actual_edges = tcuscan_ops.run_histogram(x, hist_bins, hist_range)
    torch.npu.synchronize()

    expected = reference_histogram(x, hist_bins, hist_range)
    expected_edges = torch.from_numpy(reference_bin_edges(x, hist_bins, hist_range))

    assert actual.dtype == torch.float32
    assert torch.equal(expected, actual.cpu()), f"Diff {expected - actual.cpu()}"
    # Exact: the kernel writes out the edges it compared against, computed
    # with the same float32 expressions the reference uses.
    assert torch.equal(
        expected_edges, actual_edges.cpu()
    ), f"Diff {expected_edges - actual_edges.cpu()}"
    if expect_sum is not None:
        # Summed in float64: each count is exact in float32, but their total
        # is not once it passes 2 ** 24.
        total = actual.cpu().to(torch.float64).sum().item()
        assert total == expect_sum, f"Counts sum to {total}, expected {expect_sum}."

    return actual.cpu(), actual_edges.cpu()


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
    _assert_matches_reference(x, hist_bins, expect_sum=length)


@pytest.mark.parametrize(
    "length", [16 * 1024 * 1024 + 1], ids=lambda x: f"length:{x:,}"
)
@pytest.mark.parametrize("hist_bins", [8, 64], ids=lambda x: f"hist_bins:{x}")
def test_tcuscan_histogram_matches_reference_large(length: int, hist_bins: int):
    """Same contract on an input spanning many tiles and all vector cores."""
    x = _make_input(length, "integers")
    _assert_matches_reference(x, hist_bins, expect_sum=length)


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
    first = tcuscan_ops.run_histogram(x, hist_bins)[0].cpu()
    torch.npu.synchronize()

    for _ in range(8):
        again = tcuscan_ops.run_histogram(x, hist_bins)[0].cpu()
        torch.npu.synchronize()
        assert torch.equal(first, again), f"Diff {first - again}"


# --------------------------------------------------------------------------------
# Input shape and layout
#
# `torch.histogram` histograms every element of its input whatever the shape,
# and the kernel reads a flat row-major buffer, so the wrapper has to flatten
# and materialize whatever it is handed.  Views are the interesting case: a
# transposed or strided view shares storage with a larger tensor, and reading
# `numel()` elements straight from that storage would histogram the wrong
# values.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(1024,), (32, 64), (7, 11, 13), (2, 3, 5, 7), (1, 4096), (4096, 1), ()],
    ids=str,
)
def test_tcuscan_histogram_any_shape(shape):
    """Any shape is histogrammed as its flattened self."""
    x = _make_input(int(np.prod(shape, dtype=np.int64)), "integers").reshape(shape)
    _assert_matches_reference(x, 8, expect_sum=x.numel())


def test_tcuscan_histogram_transposed_view():
    """A transposed view must be materialized, not read along its storage."""
    base = _make_input(128 * 97, "uniform").reshape(128, 97)
    x = base.t()

    assert not x.is_contiguous()
    _assert_matches_reference(x, 16, expect_sum=x.numel())


def test_tcuscan_histogram_strided_view():
    """Every other element of a row-major tensor, and nothing in between."""
    base = _make_input(2 * 100003, "uniform")
    x = base[::2]

    assert not x.is_contiguous()
    counts, _ = _assert_matches_reference(x, 16, expect_sum=x.numel())
    # Guard the oracle itself: the odd elements must have been left out.
    assert counts.sum().item() == base.numel() // 2


def test_tcuscan_histogram_offset_view():
    """A contiguous slice starts part-way into its storage."""
    base = _make_input(100003, "integers")
    x = base[1000:]

    assert x.is_contiguous() and x.storage_offset() == 1000
    _assert_matches_reference(x, 13, expect_sum=x.numel())


def test_tcuscan_histogram_empty_input():
    """An empty input gives empty bins over torch's fallback range of [0, 1]."""
    x = torch.empty(0, dtype=torch.float16, device=NPU_DEVICE)
    torch.npu.synchronize()

    counts, edges = tcuscan_ops.run_histogram(x, 4)
    torch.npu.synchronize()

    expected, expected_edges = torch.histogram(torch.empty(0), 4)
    assert torch.equal(expected, counts.cpu())
    assert torch.equal(expected_edges, edges.cpu())


def test_tcuscan_histogram_constant_input():
    """A constant input has an empty range, which torch widens by half a unit.

    Every element then lands in the middle bin.  The kernel has to apply the
    widening itself: on the default path the range is a device-side reduction
    that the host never reads back, so it cannot tell that it degenerated.
    """
    x = torch.full((100003,), 3.0, dtype=torch.float16, device=NPU_DEVICE)
    torch.npu.synchronize()

    counts, edges = tcuscan_ops.run_histogram(x, 3)
    torch.npu.synchronize()

    expected, expected_edges = torch.histogram(torch.full((100003,), 3.0), 3)
    assert torch.equal(expected, counts.cpu()), f"Got {counts.cpu()}"
    assert _edges_agree(expected_edges, edges.cpu()), f"Got {edges.cpu()}"


# --------------------------------------------------------------------------------
# Histogram range
#
# The range spans the input, so by default it is a reduction over `x`.  The
# wrapper leaves that reduction on the device and the kernel loads the two
# values itself; reading them back to the host would cost a blocking
# synchronisation far larger than the kernel.  A caller who already knows the
# range can pass it instead and skip the reduction entirely.
#
# Both paths must agree: the device-side path reads the same float16 min and
# max the host would have read, so the counts are identical, not merely close.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("hist_bins", _REF_BINS, ids=lambda x: f"hist_bins:{x}")
@pytest.mark.parametrize("distribution", ["integers", "uniform"], ids=str)
def test_tcuscan_histogram_explicit_range_matches_default(
    hist_bins: int, distribution: str
):
    """Passing the input's own range must reproduce the default exactly."""
    x = _make_input(1024 * 1024 + 13, distribution)
    torch.npu.synchronize()
    x_min, x_max = torch.aminmax(x)
    torch.npu.synchronize()

    default, default_edges = tcuscan_ops.run_histogram(x, hist_bins)
    torch.npu.synchronize()
    explicit, explicit_edges = tcuscan_ops.run_histogram(
        x, hist_bins, (x_min.item(), x_max.item())
    )
    torch.npu.synchronize()

    assert torch.equal(
        default.cpu(), explicit.cpu()
    ), f"Diff {default.cpu() - explicit.cpu()}"
    assert torch.equal(default_edges.cpu(), explicit_edges.cpu())


@pytest.mark.parametrize("hist_bins", [8, 32], ids=lambda x: f"hist_bins:{x}")
def test_tcuscan_histogram_explicit_range_matches_reference(hist_bins: int):
    """An explicit range equal to the input's range must match the oracle."""
    x = _make_input(100003, "integers")
    torch.npu.synchronize()
    x_min, x_max = torch.aminmax(x)
    torch.npu.synchronize()

    _assert_matches_reference(
        x, hist_bins, (x_min.item(), x_max.item()), expect_sum=x.numel()
    )


@pytest.mark.parametrize("hist_bins", [1, 4, 13, 64], ids=lambda x: f"hist_bins:{x}")
def test_tcuscan_histogram_narrow_range_drops_outliers(hist_bins: int):
    """Values outside an explicit range are not counted, as torch does.

    Only the elements inside `[-1, 1]` are binned, and the element equal to the
    largest edge belongs to the last bin: that end of the range is closed.
    """
    x = 10 * torch.rand(1024 * 1024, dtype=torch.float16, device=NPU_DEVICE) - 5
    torch.npu.synchronize()

    counts, _ = _assert_matches_reference(x, hist_bins, (-1.0, 1.0))

    values = x.to(torch.float64).cpu()
    in_range = ((values >= -1.0) & (values <= 1.0)).sum().item()
    assert counts.sum().item() == in_range
    assert in_range < x.numel(), "the range must actually exclude something"


def test_tcuscan_histogram_range_beyond_the_input():
    """A range wider than the input leaves the outer bins empty."""
    x = _make_input(100003, "uniform")
    counts, _ = _assert_matches_reference(x, 8, (-100.0, 100.0), expect_sum=x.numel())

    assert counts[0].item() == 0
    assert counts[-1].item() == 0


def test_tcuscan_histogram_empty_explicit_range():
    """An explicit range of zero width is widened, exactly as torch widens it."""
    x = torch.full((4096,), 2.0, dtype=torch.float16, device=NPU_DEVICE)
    torch.npu.synchronize()

    counts, edges = tcuscan_ops.run_histogram(x, 3, (2.0, 2.0))
    torch.npu.synchronize()

    expected, expected_edges = torch.histogram(
        torch.full((4096,), 2.0), 3, range=(2.0, 2.0)
    )
    assert torch.equal(expected, counts.cpu()), f"Got {counts.cpu()}"
    assert _edges_agree(expected_edges, edges.cpu()), f"Got {edges.cpu()}"


@pytest.mark.parametrize(
    "hist_range,hist_bins",
    [
        ((1000.0, 1000.0001), 64),
        ((999.99995, 1000.00018), 64),
        ((1000.00006, 1000.0002), 255),
        ((1000.0, 1000.002), 128),
        ((1000.5, 1000.5005), 100),
    ],
    ids=str,
)
def test_tcuscan_histogram_subulp_range_stays_sane(hist_range, hist_bins: int):
    """A range far narrower than its own float32 resolution must not go wrong.

    When the bin width falls below an ulp of the range's magnitude the edges
    stop being distinguishable, and the lowest two -- the only pair the kernel
    does not derive from one another -- can even come out of order.  No
    sensible counts exist at that point, but they must still be counts: never
    negative, and never more than there are elements.
    """
    x = torch.cat(
        [
            torch.full((2048,), 1000.0, dtype=torch.float16, device=NPU_DEVICE),
            torch.full((2048,), 1001.0, dtype=torch.float16, device=NPU_DEVICE),
        ]
    )
    torch.npu.synchronize()

    counts, _ = tcuscan_ops.run_histogram(x, hist_bins, hist_range)
    torch.npu.synchronize()

    counts = counts.cpu()
    assert (counts >= 0).all(), f"Negative counts: {counts}"
    assert counts.sum().item() <= x.numel()


def test_tcuscan_histogram_rejects_inverted_range():
    """An explicit range must not be empty in the wrong direction."""
    x = _make_input(1024, "uniform")
    torch.npu.synchronize()

    with pytest.raises(RuntimeError, match="must not exceed"):
        tcuscan_ops.run_histogram(x, 8, (1.0, -1.0))


def test_tcuscan_histogram_rejects_malformed_range():
    """The range is a pair; anything else is a mistake, not a default."""
    x = _make_input(1024, "uniform")
    torch.npu.synchronize()

    with pytest.raises(RuntimeError, match="exactly two values"):
        tcuscan_ops.run_histogram(x, 8, (0.0,))

    with pytest.raises(RuntimeError, match="exactly two values"):
        tcuscan_ops.run_histogram(x, 8, (0.0, 1.0, 2.0))


@pytest.mark.parametrize("hist_bins", [0, -1], ids=lambda x: f"hist_bins:{x}")
def test_tcuscan_histogram_rejects_non_positive_bins(hist_bins: int):
    """`torch.histogram` requires at least one bin."""
    x = _make_input(1024, "uniform")
    torch.npu.synchronize()

    with pytest.raises(RuntimeError, match="bins must be > 0"):
        tcuscan_ops.run_histogram(x, hist_bins)


def test_tcuscan_histogram_rejects_unsupported_dtype():
    """Only float16 is implemented; anything else must say so rather than
    return silent zeros."""
    x = torch.rand(1024, dtype=torch.float32, device=NPU_DEVICE)
    torch.npu.synchronize()

    with pytest.raises(RuntimeError, match="float16"):
        tcuscan_ops.run_histogram(x, 8)
