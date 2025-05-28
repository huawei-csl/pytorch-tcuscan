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

NUM_BLOCKS = 20

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)


def get_lengths(s: int, max_iters: int):
    for multiplier in range(1, max_iters):
        yield multiplier * NUM_BLOCKS * s * s


def _test_reduce_tiles(vec_len: int, s: int, dtype: torch.dtype):
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
    expected = torch.sum(x.reshape(NUM_BLOCKS, -1), dim=1, dtype=out_dtype).flatten()
    torch.npu.synchronize()
    actual = tcuscan_ops.run_reduce_tiles(x, s, NUM_BLOCKS)
    torch.npu.synchronize()

    assert expected.dtype == actual.dtype
    assert expected.shape == actual.shape
    assert torch.allclose(
        actual, expected, atol=1e-0, rtol=1e-3
    ), f"Input: {x}, {actual}"


@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=16))
@pytest.mark.parametrize("s", [32, 64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.int8, torch.float16], ids=str)
def test_reduce_tiles(vec_len: int, s: int, dtype: torch.dtype):
    _test_reduce_tiles(vec_len, s, dtype)
