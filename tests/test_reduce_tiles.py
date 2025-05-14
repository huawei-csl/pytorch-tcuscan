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

# Ascend 910B specifics
NUM_CORES = 20
AIV_TO_AIC_RATIO = 2

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)


def get_lengths(s: int, max_iters: int):
    NUM_AI_CORES = 20
    for multiplier in range(1, max_iters):
        yield multiplier * NUM_AI_CORES * AIV_TO_AIC_RATIO * s * s


def _test_dtype(vec_len: int, s: int, dtype: torch.dtype):
    out_dtype = None
    if dtype == torch.float16:
        x = 0.1 * torch.randn(vec_len).half().npu()
        out_dtype = torch.float32
    elif dtype == torch.int8:
        x = torch.randint(-3, 3, size=(vec_len,), dtype=torch.int8).npu()
        out_dtype = torch.int32
    else:
        assert False, "Unsupported dtype for reduce_tiles. Got {dtype}."

    expected = torch.sum(x.reshape(NUM_CORES, -1), dim=1, dtype=out_dtype).flatten()
    torch.npu.synchronize()
    actual = tcuscan_ops.run_reduce_tiles(x, s, NUM_CORES)
    torch.npu.synchronize()

    # Keep only the first entries of the output tensor (GM alignment restrictions to 32B)
    actual = actual[:NUM_CORES]
    assert expected.dtype == actual.dtype
    assert expected.shape == actual.shape
    assert torch.allclose(
        actual, expected, atol=1e-2, rtol=1e-2
    ), f"Input: {x}, {actual}"


@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=16))
@pytest.mark.parametrize("dtype", [torch.int8, torch.float16], ids=str)
def test_reduce_tiles_128(vec_len: int, dtype: torch.dtype):
    _test_dtype(vec_len, 128, dtype)


@pytest.mark.parametrize("vec_len", get_lengths(s=64, max_iters=10))
@pytest.mark.parametrize("dtype", [torch.int8, torch.float16], ids=str)
def test_reduce_tiles_64(vec_len: int, dtype: torch.dtype):
    _test_dtype(vec_len, 64, dtype)
