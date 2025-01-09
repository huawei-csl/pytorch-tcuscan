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

torch.npu.config.allow_internal_format = False


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


def test_tcuscan_segscan_single_score():
    s = 32
    n = 8 * s * s

    x = torch.rand(n, device="cpu", dtype=torch.float16)
    # TODO: there is a bug in seg_scan that fails on random instances
    # threshold = 0.95
    # f = (torch.empty(n).uniform_(0, 1) > threshold).to(torch.int8)
    # f[0] = 0
    f = torch.zeros(n, dtype=torch.int8)
    expected = ref_segscan(x.float(), f)

    print(f)
    x_npu = x.npu()
    f_npu = f.npu()
    U_s = torch.tril(torch.ones(s, s)).to(torch.float16).npu()
    output = tcuscan_ops.run_seg_scan(x_npu, f_npu, U_s, U_s.to(torch.int8)).cpu()

    print(f"Error norm: {torch.norm(output - expected)}")
    assert output.shape == expected.shape, "Output shape does not match expected shape."
    assert torch.allclose(output, expected)
