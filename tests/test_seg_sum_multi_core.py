#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import os
import random

import numpy as np
import pytest
import torch_npu  # noqa
from scipy.sparse import csr_matrix

import tcuscan_ops
import torch

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)

NUM_SEGMENTS = [131]
MAX_SEGMENT_LEN = [5000]
NUM_BLOCKS = 4
VEC_LENS = [ 4 * 128 * 128]


def uniform_rvs(shape):
    return 2 * np.random.uniform(0, 1, size=shape) - 1

def random_csr(rows: int, cols: int, nnz: int, dtype: np.dtype) -> csr_matrix:
    flat = np.random.choice(rows * cols, size=nnz, replace=False)
    row = flat // cols
    col = flat % cols
    data = uniform_rvs(nnz).astype(dtype)
    return csr_matrix((data, (row, col)), shape=(rows, cols))


def _test_tcuscan_seg_sum_multi_core(
    vec_len: int, num_segments: int, max_seg_len: int, s: int, dtype: torch.dtype
):
    sp_dtype = np.float32 if dtype == torch.float16 else np.int32

    A = random_csr(num_segments, max_seg_len, vec_len, sp_dtype)

    ones = np.ones(max_seg_len).astype(sp_dtype)
    values = (A.data).astype(sp_dtype)
    indices = (A.indptr).astype(np.uint32)
    nnz = A.nnz

    expected = A @ ones
    expected = torch.from_numpy(expected.flatten())

    assert (
        len(indices) - 1 == num_segments
    ), f"Got num_segments: {num_segments}, len(indices): {len(indices)}"

    values_npu = torch.from_numpy(values).to(dtype).npu()
    indices_npu = torch.from_numpy(indices).to(torch.int32).npu()

    print(f"nnz: {nnz} | sqrt(nnz): {nnz**0.5}")
    print(f"values: {values[:10]} ...")
    print(f"indices: {indices}")

    assert nnz % (s*s) == 0, "Number of non-zeros must be aligned"

    sstart = torch.arange(0, nnz, nnz / NUM_BLOCKS, dtype=torch.int32).npu()
    bstart = torch.searchsorted(indices_npu, sstart, right=False)
    print(f"sstart: {sstart}")
    print(f"bstart: {bstart}")

    torch.npu.synchronize()
    actual = tcuscan_ops.run_seg_sum_multi_core(values_npu, indices_npu, s, NUM_BLOCKS).cpu()
    torch.npu.synchronize()

    print(f"# of segments : {num_segments}")
    print(f"# of columns  : {max_seg_len}")
    print(f"nnz           : {nnz}")
    print(f"indices       : {indices}")
    print(f"expected      : {expected}")
    print(f"actual        : {actual}")

    abs_error = torch.max(torch.abs(actual - expected))
    rel_error = torch.max(torch.abs((actual - expected) / expected))

    assert (
        actual.shape == expected.shape
    ), f"Output shape mismatch. Got {actual.shape}. Expected {expected.shape}"
    assert (
        actual.dtype == expected.dtype
    ), f"Output dtype mismatch. Got {actual.dtype}. Expected {expected.dtype}"
    assert torch.allclose(
        actual, expected, atol=1e-2
    ), f"Error seg_sum ({expected.dtype}). Abs-error: {abs_error} / {rel_error}, s={s}, max_seg_len={max_seg_len}"


@pytest.mark.parametrize("vec_len", VEC_LENS)
@pytest.mark.parametrize("num_segments", NUM_SEGMENTS)
@pytest.mark.parametrize("max_seg_len", MAX_SEGMENT_LEN)
@pytest.mark.parametrize("s", [128])
@pytest.mark.parametrize("dtype", [torch.float16], ids=str)
def test_tcuscan_seg_sum_multi_core(
    vec_len: int, num_segments: int, max_seg_len: int, s: int, dtype: torch.dtype
):
    density = 0.01
    _test_tcuscan_seg_sum_multi_core(vec_len, num_segments, max_seg_len, s, dtype)
