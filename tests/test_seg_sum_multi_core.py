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
from math import ceil

import numpy as np
import pytest
import torch_npu  # noqa
from scipy.sparse import csr_matrix
from scipy.sparse import random as sp_random
from functools import partial

import tcuscan_ops
import torch

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)

_NUM_SEGMENTS = [513, 1025, 2011]  # 519, 2043 fails!
_NUM_COLUMNS = [64 * 64 - 1, 128 * 128, 128 * 128 - 13, 128 * 128 + 13, 128 * 128 - 133]


def uniform_rvs(shape, dtype: np.dtype):
    if np.issubsctype(dtype, np.integer):
        return np.random.randint(-2, 2, size=shape)
    return 0.5 * np.random.uniform(0, 1, size=shape) - 0.25


def random_csr(rows: int, cols: int, nnz: int, dtype: np.dtype) -> csr_matrix:
    flat = np.random.choice(rows * cols, size=nnz, replace=False)
    row = flat // cols
    col = flat % cols
    data = uniform_rvs(nnz).astype(dtype)
    return csr_matrix((data, (row, col)), shape=(rows, cols))


def tiling_function(nnz: int, s: int, max_aic_cores: int = 20):
    "Return the tiling parameters 'block_len' and 'num_blocks' of seg_sum_multi_core."
    matmul_tile_len = s * s
    num_tiles = ceil(nnz / matmul_tile_len)
    num_blocks = max_aic_cores
    if num_tiles < num_blocks:
        num_blocks = num_tiles

    max_num_tiles_per_block = ceil(num_tiles / num_blocks)
    block_len = max_num_tiles_per_block * matmul_tile_len
    return block_len, num_blocks


def _test_seg_sum_multi_core(
    num_rows: int,
    num_cols: int,
    s: int,
    density: float,
    dtype: torch.dtype,
    use_segm_offsets: bool,
):
    sp_dtype = np.float32 if dtype == torch.float16 else np.int32

    num_segments = num_rows

    A = sp_random(
        num_rows,
        num_cols,
        density=density,
        format="csr",
        dtype=sp_dtype,
        data_rvs=partial(uniform_rvs, dtype=sp_dtype),
    )

    ones = np.ones(num_cols).astype(sp_dtype)
    values = (A.data).astype(sp_dtype)
    indices = (A.indptr).astype(np.uint32)
    nnz = A.nnz

    expected = A @ ones
    expected = torch.from_numpy(expected.flatten())

    # Drop first (always zero) and the last (always len(x)) entry of indices.
    assert indices[0] == 0, "First entry must be zero."
    assert indices[-1] == nnz, "Last entry equals to the nnzs."

    values_npu = torch.from_numpy(values).npu().to(dtype)
    indices_npu = torch.from_numpy(indices).npu().to(torch.int32)
    nnz = A.nnz

    torch.npu.synchronize()
    if use_segm_offsets:
        block_len, num_blocks = tiling_function(nnz, s)
        sstart = torch.clamp(
            torch.arange(0, num_blocks + 1, dtype=torch.int32) * block_len,
            max=nnz,
        ).npu()
        torch.npu.synchronize()
        segm_offsets = torch.searchsorted(indices_npu, sstart, out_int32=True, right=False)
        torch.npu.synchronize()

        actual = tcuscan_ops.run_seg_sum_multi_core(
            values_npu, indices_npu, s, segm_offsets
        ).cpu()
    else:
        actual = tcuscan_ops.run_seg_sum_multi_core(values_npu, indices_npu, s).cpu()
    torch.npu.synchronize()

    print(f"# of segments : {num_segments}")
    print(f"# of columns  : {num_cols}")
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
    if dtype == torch.int8:
        assert torch.equal(
            actual, expected
        ), f"Error seg_sum ({expected.dtype}). Abs-error: {abs_error} / {rel_error}, s={s}, num_cols={num_cols}"
    elif dtype == torch.float16:
        assert torch.allclose(
            actual, expected, atol=1e-2
        ), f"Error seg_sum ({expected.dtype}). Abs-error: {abs_error} / {rel_error}, s={s}, num_cols={num_cols}"


@pytest.mark.parametrize(
    "num_segments", _NUM_SEGMENTS, ids=lambda x: f"num_segms:({x})"
)
@pytest.mark.parametrize("num_cols", _NUM_COLUMNS, ids=lambda x: f"num_cols:({x})")
@pytest.mark.parametrize("s", [32, 64, 128], ids=lambda s: f"s:({s})")
@pytest.mark.parametrize("dtype", [torch.int8, torch.float16], ids=str)
@pytest.mark.parametrize(
    "use_segm_offsets",
    [False, True],
    ids=lambda x: "with_segm_offsets" if x else "no_segm_offsets",
)
def test_seg_sum_multi_core(
    num_segments: int, num_cols: int, s: int, dtype: torch.dtype, use_segm_offsets: bool
):
    density = 0.01
    _test_seg_sum_multi_core(
        num_segments, num_cols, s, density, dtype, use_segm_offsets
    )
