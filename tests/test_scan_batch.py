# --------------------------------------------------------------------------------
# Copyright (c) 2023-2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pytorch-tcuscan/
# for the full License text.
# --------------------------------------------------------------------------------
import os
import random
from typing import List

import numpy as np
import pytest
import torch_npu  # noqa

import tcuscan_ops
import torch

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)

_MULTIPLIERS = [1, 2, 3, 11, 63]
_BATCH_SIZES = [2, 16, 20, 32, 44, 256]
_S = [32, 64, 128]


def get_lengths_and_batch_sizes(
    matmul_sizes: List[int], multipliers: List[int], batch_sizes: List[int]
):
    for s in matmul_sizes:
        for multiplier in multipliers:
            for batch_size in batch_sizes:
                yield (s, multiplier * s * s, batch_size)
                yield (s, multiplier * s * s // 2, batch_size)
                yield (s, multiplier * s * s // 4, batch_size)
                yield (s, multiplier * s * s - 1, batch_size)
                yield (s, multiplier * s * s + 1, batch_size)


def _test_scan_batch(s: int, vec_len: int, batch_size: int, dtype: torch.dtype):

    x = torch.rand((batch_size, vec_len), dtype=dtype, device=NPU_DEVICE)

    torch.npu.synchronize()
    expected = torch.cumsum(x, dim=1, dtype=torch.float)
    torch.npu.synchronize()
    actual = tcuscan_ops.run_scan_batch(x, s)
    torch.npu.synchronize()

    assert torch.allclose(
        actual, expected, rtol=1e-03
    ), f"batch scan ({dtype}) wrong. s={s}, vec_len={vec_len}, batch_size= {batch_size}"


@pytest.mark.parametrize(
    "s, vec_len, batch_size",
    get_lengths_and_batch_sizes(
        matmul_sizes=_S, multipliers=_MULTIPLIERS, batch_sizes=_BATCH_SIZES
    ),
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=str)
def test_scan_batch(s: int, vec_len: int, batch_size: int, dtype: torch.dtype):
    _test_scan_batch(s, vec_len, batch_size, dtype)


# Layout coverage: the kernel itself only consumes a row-major
# `batch_size x vec_len` buffer, so the host wrapper is what has to cope with
# transposed views, storage offsets and a scanned dimension other than the last.
_LAYOUT_S = 32
_DTYPES = [torch.float32, torch.float16]


def _check_scan_batch(x: torch.Tensor, s: int, dim: int, msg: str) -> None:
    expected = torch.cumsum(x, dim=dim, dtype=torch.float)
    torch.npu.synchronize()
    actual = tcuscan_ops.run_scan_batch(x, s, dim)
    torch.npu.synchronize()

    assert (
        actual.shape == x.shape
    ), f"{msg}: got shape {actual.shape}, expected {x.shape}"
    # The layout of the result is deliberately not forced to row-major, so it
    # is only compared by value.
    assert torch.allclose(actual, expected, rtol=1e-03), msg


@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
def test_scan_batch_transposed_input(dtype: torch.dtype):
    # In `base.t()` the first dimension is the fastest running one.
    base = torch.rand((256, 64), dtype=dtype, device=NPU_DEVICE)
    x = base.t()

    assert not x.is_contiguous()
    _check_scan_batch(x, _LAYOUT_S, -1, "transposed input")


@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
def test_scan_batch_sliced_input(dtype: torch.dtype):
    # A row slice stays row-major but starts at a non-zero storage offset.
    base = torch.rand((66, 256), dtype=dtype, device=NPU_DEVICE)
    x = base[2:]

    assert x.is_contiguous() and x.storage_offset() != 0
    _check_scan_batch(x, _LAYOUT_S, -1, "sliced input")


@pytest.mark.parametrize("dim", [0, 1, -1, -2])
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
def test_scan_batch_dim_2d(dim: int, dtype: torch.dtype):
    x = torch.rand((64, 256), dtype=dtype, device=NPU_DEVICE)
    _check_scan_batch(x, _LAYOUT_S, dim, f"2D dim={dim}")


@pytest.mark.parametrize("dim", [0, 1, 2, -1])
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
def test_scan_batch_dim_3d(dim: int, dtype: torch.dtype):
    x = torch.rand((64, 64, 128), dtype=dtype, device=NPU_DEVICE)
    _check_scan_batch(x, _LAYOUT_S, dim, f"3D dim={dim}")


@pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
def test_scan_batch_odd_batch_size(batch_size: int, dtype: torch.dtype):
    # The kernel walks over pairs of vectors, so an odd batch needs padding.
    x = torch.rand((batch_size, 256), dtype=dtype, device=NPU_DEVICE)
    _check_scan_batch(x, _LAYOUT_S, -1, f"batch_size={batch_size}")


@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
def test_scan_batch_1d_input(dtype: torch.dtype):
    x = torch.rand(256, dtype=dtype, device=NPU_DEVICE)
    _check_scan_batch(x, _LAYOUT_S, -1, "1-D input")


def test_scan_batch_does_not_force_output_layout():
    # Scanning the last dimension produces a row-major result; scanning any
    # other one hands back a transposed view instead of copying it row-major.
    x = torch.rand((64, 256), dtype=torch.float32, device=NPU_DEVICE)

    assert tcuscan_ops.run_scan_batch(x, _LAYOUT_S, -1).is_contiguous()
    assert not tcuscan_ops.run_scan_batch(x, _LAYOUT_S, 0).is_contiguous()


@pytest.mark.parametrize("dim", [2, -3])
def test_scan_batch_rejects_out_of_range_dim(dim: int):
    x = torch.rand((64, 256), dtype=torch.float32, device=NPU_DEVICE)
    with pytest.raises(RuntimeError, match="out of range"):
        tcuscan_ops.run_scan_batch(x, _LAYOUT_S, dim)
