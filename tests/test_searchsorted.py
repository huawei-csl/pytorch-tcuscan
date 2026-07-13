#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import os

import numpy as np
import pytest
import torch_npu  # noqa

import tcuscan_ops
import torch

np.random.seed(42)
torch.manual_seed(42)

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)


def _reference(sorted_npu: torch.Tensor, values_npu: torch.Tensor) -> torch.Tensor:
    # side='left' (lower_bound), int32 output -- what run_searchsorted reproduces.
    return torch.searchsorted(sorted_npu, values_npu, out_int32=True)


def _run_and_check(sorted_cpu: torch.Tensor, values_cpu: torch.Tensor):
    sorted_npu = sorted_cpu.to(torch.int32).npu()
    values_npu = values_cpu.to(torch.int32).npu()
    torch.npu.synchronize()

    output = tcuscan_ops.run_searchsorted(sorted_npu, values_npu)
    torch.npu.synchronize()

    reference = _reference(sorted_npu, values_npu)

    assert output.dtype == torch.int32
    assert output.shape == values_npu.shape
    assert torch.equal(
        output, reference
    ), f"mismatch\n got: {output.cpu()}\n exp: {reference.cpu()}"


@pytest.mark.parametrize("data_len", [1, 8, 127, 1024, 65537])
@pytest.mark.parametrize("num_values", [1, 3, 21, 64])
def test_tcuscan_searchsorted_random(data_len, num_values):
    # Random sorted haystack (with duplicates) and random needles.
    sorted_cpu = torch.sort(
        torch.randint(-50, 50, (data_len,), dtype=torch.int32)
    ).values
    values_cpu = torch.randint(-60, 60, (num_values,), dtype=torch.int32)
    _run_and_check(sorted_cpu, values_cpu)


def test_tcuscan_searchsorted_edges():
    # Contiguous run with duplicates; exercise below-min, above-max, exact ties,
    # and interior boundaries.
    sorted_cpu = torch.tensor([0, 0, 2, 2, 2, 5, 9, 9, 10], dtype=torch.int32)
    values_cpu = torch.tensor([-1, 0, 1, 2, 3, 5, 8, 9, 10, 11], dtype=torch.int32)
    _run_and_check(sorted_cpu, values_cpu)


def test_tcuscan_searchsorted_csr_indptr_like():
    # Mimics the spmv_v2_multi_cube use: monotonic CSR row-pointer haystack and
    # a handful of block-boundary needles.
    nnz = 100000
    num_segments = 4000

    indptr = torch.sort(
        torch.randint(0, nnz, (num_segments,), dtype=torch.int32)
    ).values
    indptr = torch.cat([torch.zeros(1, dtype=torch.int32), indptr])
    sstart = torch.tensor(
        [0, nnz // 4, nnz // 2, (3 * nnz) // 4, nnz], dtype=torch.int32
    )
    _run_and_check(indptr, sstart)
