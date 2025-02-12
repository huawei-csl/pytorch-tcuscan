#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np
import pytest
import torch
import torch_npu  # noqa

import tcuscan_ops

torch.npu.config.allow_internal_format = False

_MULTIPLIER = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]  # , 2, 3, 5]


def ref_segscan(x, f):
    z = np.zeros(len(x))
    scan_f = np.cumsum(f)
    num_segments = scan_f[-1]
    index = 0
    for seg_id in range(0, num_segments + 1):
        seg_size = np.sum(scan_f == seg_id)
        z[index : index + seg_size] = np.cumsum(x[scan_f == seg_id])
        index += seg_size

    return z


def _test_tcuscan_segscan_revert(n: int, segm_density: float):

    data_type = np.float32

    rng = np.random.default_rng(seed=42)
    input_x = rng.integers(0, 10, n).astype(data_type)
    input_f = np.zeros((n,), dtype=np.int32)
    num_ones = int(n * segm_density)
    input_f[rng.choice(n, num_ones, replace=False)] = 1
    input_f[0] = 0

    # Scan(x/f)
    scan_x = np.cumsum(input_x).astype(data_type)
    scan_f = np.cumsum(input_f).astype(np.int32)

    # Collect largest value of each segment
    diff = np.compress(np.append(input_f[1:], 1), scan_x).astype(data_type)

    scan_x_npu = torch.Tensor(scan_x).npu()
    scan_f_npu = torch.Tensor(scan_f).to(torch.int32).npu()
    diff_npu = torch.Tensor(diff).npu()

    assert scan_x_npu.dtype == torch.float32
    assert scan_f_npu.dtype == torch.int32
    assert diff_npu.dtype == torch.float32

    torch.npu.synchronize()

    actual = tcuscan_ops.run_seg_scan_mc_revert(scan_x_npu, scan_f_npu, diff_npu).cpu()
    golden = ref_segscan(input_x, input_f).astype(data_type)

    assert actual.shape == golden.shape, "Output shape does not match expected shape."
    assert torch.allclose(
        actual, torch.Tensor(golden), atol=1e-02
    ), f"segmented scan revertion (fp16) vec_len={n}"


@pytest.mark.parametrize("multiplier", _MULTIPLIER)
def test_tcuscan_segscan_single_score_s_32_density_0_01(multiplier: int):
    segm_density = 0.01
    num_cores = 40
    n = num_cores * 2 * 1024 * multiplier

    _test_tcuscan_segscan_revert(n, segm_density)


@pytest.mark.parametrize("multiplier", _MULTIPLIER)
def test_tcuscan_segscan_single_score_s_32_density_0_1(multiplier: int):
    segm_density = 0.1
    num_cores = 40
    n = num_cores * 2 * 1024 * multiplier

    _test_tcuscan_segscan_revert(n, segm_density)


@pytest.mark.parametrize("multiplier", _MULTIPLIER)
def test_tcuscan_segscan_single_score_s_32_density_0_001(multiplier: int):
    segm_density = 0.001
    num_cores = 40
    n = num_cores * 2 * 1024 * multiplier

    _test_tcuscan_segscan_revert(n, segm_density)


@pytest.mark.parametrize("multiplier", _MULTIPLIER)
def test_tcuscan_segscan_single_score_s_32_density_0_002(multiplier: int):
    segm_density = 0.003
    num_cores = 40
    n = num_cores * 2 * 1024 * multiplier

    _test_tcuscan_segscan_revert(n, segm_density)
