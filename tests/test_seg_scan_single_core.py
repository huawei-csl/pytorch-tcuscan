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
import numpy as np
import numpy.typing as npt

import tcuscan_ops
import pytest

torch.npu.config.allow_internal_format = False

_MULTIPLIER = [1, 2, 3, 5, 8, 9, 12, 16, 20, 24]


def ref_segscan(x: npt.ArrayLike, f: npt.ArrayLike) -> npt.ArrayLike:
    """Returns the segmented scan of input vector x given flag vector f.

    Reference implementation of segmented scan.
    """
    y = np.copy(x.cpu().numpy())

    # Perform unsegmented scan on f; last entry equals to # of segments
    scan_f = np.cumsum(f.cpu().numpy())
    num_segments = int(scan_f[-1])

    # For each segment, perform a scan
    offset = 0
    for segment_id in range(0, num_segments + 1):
        segment_size = sum(scan_f == segment_id)
        end = offset + segment_size

        # Scan on (segment_id)-th segment
        y[offset:end] = np.cumsum(x[offset:end])

        # Move to the next segment
        offset = offset + segment_size

    return torch.Tensor(y)


def _test_tcuscan_segscan_single_core(n: int, s: int, segm_density: float):
    rng = np.random.default_rng(seed=42)
    input_x = rng.integers(0, 10, n).astype(np.float16)
    x = torch.Tensor(input_x).half()
    f = torch.empty(n).uniform_(0, 1) < segm_density
    f = f.to(torch.int8)

    print(f)
    print(f" # of segments: {torch.sum(f)}")
    x_npu = x.npu()
    f_npu = f.npu()
    actual = tcuscan_ops.run_seg_scan(x_npu, f_npu, s).cpu()
    expected = ref_segscan(x.float(), f)

    assert actual.shape == expected.shape, "Output shape does not match expected shape."
    assert torch.allclose(
        actual, expected, atol=1e-02
    ), f"segmented scan single core (fp16) wrong. s={s}, vec_len={n}"


@pytest.mark.parametrize("multiplier", _MULTIPLIER)
def test_tcuscan_segscan_single_score_s_32_density_0_01(multiplier: int):
    s = 32
    segm_density = 0.01
    n = multiplier * s * s

    _test_tcuscan_segscan_single_core(n, s, segm_density)


@pytest.mark.parametrize("multiplier", _MULTIPLIER[:-3])
def test_tcuscan_segscan_single_score_s_64_density_0_01(multiplier: int):
    s = 64
    segm_density = 0.01
    n = multiplier * s * s

    _test_tcuscan_segscan_single_core(n, s, segm_density)


@pytest.mark.parametrize("multiplier", _MULTIPLIER[:-6])
def test_tcuscan_segscan_single_score_s_128_density_0_01(multiplier: int):
    s = 128
    segm_density = 0.01
    n = multiplier * s * s

    _test_tcuscan_segscan_single_core(n, s, segm_density)


@pytest.mark.parametrize("multiplier", _MULTIPLIER)
def test_tcuscan_segscan_single_score_s_32_density_0_05(multiplier: int):
    s = 32
    segm_density = 0.05
    n = multiplier * s * s

    _test_tcuscan_segscan_single_core(n, s, segm_density)


@pytest.mark.parametrize("multiplier", _MULTIPLIER[:-3])
def test_tcuscan_segscan_single_score_s_64_density_0_05(multiplier: int):
    s = 64
    segm_density = 0.05
    n = multiplier * s * s

    _test_tcuscan_segscan_single_core(n, s, segm_density)


@pytest.mark.parametrize("multiplier", _MULTIPLIER[:-6])
def test_tcuscan_segscan_single_score_s_128_density_0_05(multiplier: int):
    s = 128
    segm_density = 0.05
    n = multiplier * s * s

    _test_tcuscan_segscan_single_core(n, s, segm_density)
