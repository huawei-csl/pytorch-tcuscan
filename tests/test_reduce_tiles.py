import os
import random
from math import ceil

import numpy as np
import pytest
import torch_npu  # noqa

import tcuscan_ops
import torch

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

NUM_BLOCKS = 40

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)


def get_lengths(s: int, max_iters: int):
    for multiplier in range(
        1, max_iters
    ):  # FIXME(anastasios) range(1,max_iters) fails on torch.int8!
        yield multiplier * NUM_BLOCKS * s * s


def ref_reduce_tiles(x, dtype: torch.dtype, tile_len: int, num_blocks: int):
    n = len(x)
    num_tiles = ceil(n / tile_len)
    max_num_tiles_per_block = ceil(num_tiles / num_blocks)
    block_len = tile_len * max_num_tiles_per_block
    sums = torch.zeros(num_blocks, dtype=dtype, device=NPU_DEVICE)
    for i in range(num_blocks):
        end = min((i + 1) * block_len, len(x))
        sums[i] = torch.sum(x[i * block_len : end])

    return sums


def _test_reduce_tiles(vec_len: int, tile_len: int, dtype: torch.dtype):
    out_dtype = None
    if dtype == torch.float16:
        x = 0.1 * torch.randn(vec_len, dtype=dtype, device=NPU_DEVICE)
        out_dtype = torch.float32
    elif dtype == torch.int8:
        x = torch.randint(-3, 3, size=(vec_len,), dtype=torch.int8, device=NPU_DEVICE)
        out_dtype = torch.int32
    else:
        assert False, "Unsupported dtype for reduce_tiles. Got {dtype}."

    torch.npu.synchronize()
    expected = ref_reduce_tiles(x, out_dtype, tile_len, NUM_BLOCKS)
    torch.npu.synchronize()
    actual = tcuscan_ops.run_reduce_tiles(x, tile_len, NUM_BLOCKS)
    torch.npu.synchronize()

    assert expected.dtype == actual.dtype
    assert expected.shape == actual.shape
    assert torch.allclose(
        actual, expected, atol=1e-0, rtol=1e-3
    ), f"Expected : {expected}. Actual: {actual}"


@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=16))
@pytest.mark.parametrize("s", [16, 32, 64, 128])
@pytest.mark.parametrize("offset", [-127, -34, -11, -1, 0, 1, 11, 17, 34])
@pytest.mark.parametrize("dtype", [torch.int8, torch.float16], ids=str)
def test_reduce_tiles(vec_len: int, s: int, offset: int, dtype: torch.dtype):
    _test_reduce_tiles(vec_len + offset, s * s // 2, dtype)
