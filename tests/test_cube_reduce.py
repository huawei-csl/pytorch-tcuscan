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

NUM_BLOCKS = 20

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)


def get_lengths(s: int, max_iters: int):
    for multiplier in range(1, max_iters):
        yield multiplier * NUM_BLOCKS * s * s


def ref_cube_reduce(x, dtype: torch.dtype, tile_len: int, num_blocks: int):
    n = len(x)
    num_tiles = ceil(n / tile_len)
    max_num_tiles_per_block = ceil(num_tiles / num_blocks)
    block_len = tile_len * max_num_tiles_per_block
    sums = torch.zeros(num_blocks, dtype=dtype, device=NPU_DEVICE)
    for i in range(num_blocks):
        end = min((i + 1) * block_len, len(x))
        sums[i] = torch.sum(x[i * block_len : end])

    return sums


def _test_cube_reduce(vec_len: int, dtype: torch.dtype):
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
    expected = x.reshape(NUM_BLOCKS, -1).sum(dim=1, dtype=out_dtype).flatten()
    torch.npu.synchronize()
    actual = tcuscan_ops.run_cube_reduce(x, NUM_BLOCKS)
    torch.npu.synchronize()

    assert expected.dtype == actual.dtype
    assert expected.shape == actual.shape
    assert torch.allclose(
        actual, expected, atol=1e-0, rtol=1e-2
    ), f"Expected : {expected}. Actual: {actual}"


@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=16))
@pytest.mark.parametrize("offset", [0])  # TODO(anastasios): test unpadded cases
@pytest.mark.parametrize("dtype", [torch.int8, torch.float16], ids=str)
def test_cube_reduce(vec_len: int, offset: int, dtype: torch.dtype):
    _test_cube_reduce(vec_len + offset, dtype)
