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
import torch_npu  # noqa

import tcuscan_ops
import torch

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Only shapes the test lengths below: how many blocks the kernel splits the
# work over is decided by the tiling heuristic, not by the caller.
_LENGTH_MULTIPLE = 20

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)


def get_lengths(s: int, max_iters: int):
    for multiplier in range(1, max_iters):
        yield multiplier * _LENGTH_MULTIPLE * s * s


def _make_input(vec_len: int, dtype: torch.dtype):
    if dtype == torch.float16:
        return 0.1 * torch.randn(vec_len, dtype=dtype, device=NPU_DEVICE)
    if dtype == torch.int8:
        return torch.randint(
            -3, 3, size=(vec_len,), dtype=torch.int8, device=NPU_DEVICE
        )
    raise AssertionError(f"Unsupported dtype for cube_reduce. Got {dtype}.")


def _out_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype == torch.float16 else torch.int32


def _assert_matches_torch_sum(x: torch.Tensor) -> None:
    """The kernel returns the total as a single element.

    The number of blocks it splits the work over is a tiling decision made
    inside the wrapper, so nothing here depends on it.
    """
    out_dtype = _out_dtype(x.dtype)
    torch.npu.synchronize()
    expected = x.reshape(-1).sum(dtype=out_dtype).reshape(1)
    torch.npu.synchronize()
    actual = tcuscan_ops.run_cube_reduce(x)
    torch.npu.synchronize()

    assert expected.dtype == actual.dtype
    assert actual.shape == (1,), f"Expected a single element, got {actual.shape}."
    assert torch.allclose(
        actual, expected, atol=1e-0, rtol=1e-2
    ), f"Expected : {expected}. Actual: {actual}"


def _test_cube_reduce(vec_len: int, dtype: torch.dtype):
    _assert_matches_torch_sum(_make_input(vec_len, dtype))


@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=16))
@pytest.mark.parametrize("offset", [0])  # TODO(anastasios): test unpadded cases
@pytest.mark.parametrize("dtype", [torch.int8, torch.float16], ids=str)
def test_cube_reduce(vec_len: int, offset: int, dtype: torch.dtype):
    _test_cube_reduce(vec_len + offset, dtype)


# --------------------------------------------------------------------------------
# Shape, length and dtype
#
# The result is one element and the block count is internal, so nothing about
# the input has to line up with it: any shape, any length -- including ones
# that need the kernel's internal padding pass -- and the empty input.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.int8, torch.float16], ids=str)
@pytest.mark.parametrize(
    "shape", [(327680,), (20, 16384), (4, 5, 16384), (7, 11, 13), ()], ids=str
)
def test_cube_reduce_any_shape(shape, dtype: torch.dtype):
    numel = int(np.prod(shape, dtype=np.int64))
    _assert_matches_torch_sum(_make_input(numel, dtype).reshape(shape))


@pytest.mark.parametrize("dtype", [torch.int8, torch.float16], ids=str)
@pytest.mark.parametrize("vec_len", [1, 13, 1000, 16385])
def test_cube_reduce_short_lengths(vec_len: int, dtype: torch.dtype):
    """Inputs shorter than a full tile, where the heuristic drops to one block."""
    _assert_matches_torch_sum(_make_input(vec_len, dtype))


@pytest.mark.parametrize("dtype", [torch.int8, torch.float16], ids=str)
@pytest.mark.parametrize(
    "vec_len", [327679, 327681, 344063, 1000003, 16 * 1024 * 1024 + 7]
)
def test_cube_reduce_ragged_lengths(vec_len: int, dtype: torch.dtype):
    """Lengths that do not fill whole tiles take the kernel's padding path.

    The padded input is written by every vector core and read by every cube
    core, so it needs a device-wide barrier between the two; with only a
    group-wide one the cube cores reduced tiles their peers had not written.
    """
    _assert_matches_torch_sum(_make_input(vec_len, dtype))


def test_cube_reduce_non_contiguous_view():
    """A strided view must be materialized, not read along its storage."""
    base = _make_input(2 * 327680, torch.float16)
    x = base[::2]

    assert not x.is_contiguous()
    _assert_matches_torch_sum(x)


@pytest.mark.parametrize("dtype", [torch.int8, torch.float16], ids=str)
def test_cube_reduce_empty_input(dtype: torch.dtype):
    """An empty input sums to zero and launches no kernel."""
    x = torch.empty(0, dtype=dtype, device=NPU_DEVICE)
    torch.npu.synchronize()

    actual = tcuscan_ops.run_cube_reduce(x)
    torch.npu.synchronize()

    assert actual.shape == (1,)
    assert actual.item() == 0
    assert actual.dtype == _out_dtype(dtype)


def test_cube_reduce_rejects_unsupported_dtype():
    """Only fp16 and int8 are implemented; others used to return zeros."""
    x = torch.rand(327680, dtype=torch.float32, device=NPU_DEVICE)
    torch.npu.synchronize()

    with pytest.raises(RuntimeError, match="float16 and int8"):
        tcuscan_ops.run_cube_reduce(x)
