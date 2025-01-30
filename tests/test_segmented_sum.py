#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import torch
import torch_npu  # noqa

import tcuscan_ops
import pytest

torch.npu.config.allow_internal_format = False

_MULTIPLIER = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]


def ref_segsum(x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Returns the segmented sum of input vector x and segment-descriptor vector f.

    Reference implementation of segmented sum.

    Returns torch.Tensor(dtype=torch.float32).
    """

    # Perform unsegmented scan on f
    scan_f = torch.cumsum(f, dim=-1, dtype=torch.float32)

    # last entry equals to # of segments
    num_segments = int(scan_f[-1].item())
    sums = torch.zeros(num_segments + 1, dtype=torch.float32)

    # For each segment, perform a scan
    offset = 0
    for segment_id in range(0, num_segments + 1):
        segment_size = torch.sum(scan_f == segment_id)
        end = offset + segment_size

        # Scan on (segment_id)-th segment
        sums[segment_id] = torch.sum(x[offset:end])

        # Move to the next segment
        offset = offset + segment_size

    return sums


def _test_tcuscan_seg_sum(n: int, s: int, segm_density: float):
    x = torch.randint(0, 10, size=(n,)).half()
    f = (torch.rand(n) < segm_density).to(torch.int8)
    f[0] = 0

    x_npu = x.npu()
    f_npu = torch.concat([f[1:], torch.ones(1, dtype=torch.int8)]).contiguous().npu()
    torch.npu.synchronize()
    actual = tcuscan_ops.run_seg_sum(x_npu, f_npu, s).cpu()
    expected = ref_segsum(x, f)
    torch.npu.synchronize()
    assert (
        actual.shape == expected.shape
    ), "Output shape mismatch. Got {actual.shape}. Expected {expected.shape}"
    assert (
        actual.dtype == expected.dtype
    ), f"Output dtype mismatch. Got {actual.dtype}. Expected {expected.dtype}"
    assert torch.allclose(
        actual, expected, atol=1e-02
    ), f"Error seg_sum ({expected.dtype}). s={s}, vec_len={n}"


@pytest.mark.parametrize("multiplier", _MULTIPLIER)
def test_tcuscan_segmented_sum_s_32_density_0_01(multiplier: int):
    s = 32
    segm_density = 0.01
    n = multiplier * 20 * s * s

    _test_tcuscan_seg_sum(n, s, segm_density)


@pytest.mark.parametrize("multiplier", _MULTIPLIER[:3])
def test_tcuscan_segmented_sum_s_64_density_0_01(multiplier: int):
    s = 64
    segm_density = 0.01
    n = multiplier * 20 * s * s

    _test_tcuscan_seg_sum(n, s, segm_density)


@pytest.mark.parametrize("multiplier", _MULTIPLIER[:2])
def test_tcuscan_segmented_sum_s_128_density_0_01(multiplier: int):
    s = 128
    n = multiplier * s * s
    segm_density = 0.01

    _test_tcuscan_seg_sum(n, s, segm_density)
