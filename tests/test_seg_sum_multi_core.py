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
import math

import numpy as np
import pytest
import torch_npu  # noqa
from scipy.sparse import csr_matrix

import tcuscan_ops
import torch

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:0")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)

MAX_SEGMENT_LEN = [5000]


def uniform_rvs(shape):
    return 2 * np.random.uniform(0, 1, size=shape) - 1

def random_csr(rows: int, cols: int, nnz: int, dtype: np.dtype) -> csr_matrix:
    flat = np.random.choice(rows * cols, size=nnz, replace=False)
    row = flat // cols
    col = flat % cols
    data = uniform_rvs(nnz).astype(dtype)
    return csr_matrix((data, (row, col)), shape=(rows, cols))


def _test_tcuscan_seg_sum_multi_core(
    vec_len: int, num_segments: int, max_seg_len: int, s: int, num_blocks: int, dtype: torch.dtype
):
    sp_dtype = np.float32 if dtype == torch.float16 else np.int32

    A = random_csr(num_segments, max_seg_len, vec_len, sp_dtype)

    ones = np.ones(max_seg_len).astype(sp_dtype)
    values = (A.data).astype(sp_dtype)
    indices = (A.indptr).astype(np.uint32)
    nnz = A.nnz

    expected = A @ ones
    expected = torch.from_numpy(expected.flatten())

    values_npu = torch.from_numpy(values).to(dtype).npu()
    indices_npu = torch.from_numpy(indices).to(torch.int32).npu()
    torch.npu.synchronize()

    print(f"nnz: {nnz} | sqrt(nnz): {nnz**0.5}")
    print(f"values: {values[:10]} ...")
    print(f"indices: {indices}")

    assert nnz % (s*s) == 0, "Number of non-zeros must be aligned"

    torch.npu.synchronize()
    sstart = torch.arange(0, nnz + 1, nnz // num_blocks, dtype=torch.int32).npu()
    torch.npu.synchronize()
    bstart = torch.searchsorted(indices_npu, sstart, right=False).to(torch.int32)
    torch.npu.synchronize()
    segm_len_per_block = torch.diff(bstart)
    torch.npu.synchronize()

    print(f"num_blocks                : {num_blocks}")
    print(f"block_len                  : {nnz // num_blocks}")
    print(f"block_offsets (len: {len(sstart)}): {sstart}")
    print(f"bstart (len: {len(bstart)}): {bstart}")
    print(f"segm_len_per_block (len:{len(segm_len_per_block)}): {segm_len_per_block}")

    torch.npu.synchronize()
    actual = tcuscan_ops.run_seg_sum_multi_core(values_npu, indices_npu, bstart, segm_len_per_block, s, num_blocks).cpu()
    torch.npu.synchronize()

    print(f"# of segments : {num_segments}")
    print(f"# of columns  : {max_seg_len}")
    print(f"nnz           : {nnz}")
    print(f"indices       : {indices}")
    print(f"expected      : {expected}")
    print(f"actual        : {actual}")
    diff = torch.abs(actual - expected) < 1e-2
    print(f"diff          : {diff}")
    print(f"ratio of wrong entries: {1 - diff.float().mean():.4f}")
    np_where = torch.where(diff == False)
    print(f"np.where: {np_where}")
    print(f"actual: {actual[np_where]}")
    print(f"expected: {expected[np_where]}")


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


@pytest.mark.parametrize("max_seg_len", MAX_SEGMENT_LEN)
@pytest.mark.parametrize("s", [128])
@pytest.mark.parametrize("num_blocks", [2, 4, 8, 16, 20])
@pytest.mark.parametrize("dtype", [torch.float16], ids=str)
def test_tcuscan_seg_sum_multi_core(
    max_seg_len: int, s: int, num_blocks: int, dtype: torch.dtype
):
    num_segments = int(num_blocks * math.sqrt(s))
    vec_len = num_blocks * s * s
    _test_tcuscan_seg_sum_multi_core(vec_len, num_segments, max_seg_len, s, num_blocks, dtype)
