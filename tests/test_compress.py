import random

import numpy as np
import pytest
import torch
import torch_npu  # noqa

import tcuscan_ops

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

torch.npu.config.allow_internal_format = False


def get_lengths(s: int, max_iters: int):
    NUM_AI_CORES = 20
    for multiplier in range(1, max_iters):
        yield multiplier * NUM_AI_CORES * s * s


def tcuscan_compress(x, mask, s: int):
    z = tcuscan_ops.run_compress(x, mask, s)
    output_size = torch.sum(mask)
    return z[:output_size]


def _test_compress(vec_len: int, s: int, dtype: torch.dtype):
    x = torch.randint(0, 2**7 - 1, (vec_len,)).to(dtype).npu()
    mask = (torch.randn(vec_len) > 0).to(torch.int8).npu()

    expected = torch.masked_select(x, mask.to(torch.uint8))
    torch.npu.synchronize()
    actual = tcuscan_compress(x, mask, s)

    assert len(expected) == len(actual)
    assert torch.allclose(expected, actual)


@pytest.mark.parametrize("vec_len", get_lengths(s=32, max_iters=12))
def test_tcuscan_compress_fp16_s_32(vec_len: int):
    _test_compress(vec_len, 32, torch.float16)


@pytest.mark.parametrize("vec_len", get_lengths(s=64, max_iters=8))
def test_tcuscan_compress_fp16_s_64(vec_len: int):
    _test_compress(vec_len, 64, torch.float16)


@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=6))
def test_tcuscan_compress_fp16_s_128(vec_len: int):
    _test_compress(vec_len, 128, torch.float16)


@pytest.mark.parametrize("vec_len", get_lengths(s=32, max_iters=12))
def test_tcuscan_compress_fp32_s_16(vec_len: int):
    _test_compress(vec_len, 32, torch.float32)


@pytest.mark.parametrize("vec_len", get_lengths(s=64, max_iters=8))
def test_tcuscan_compress_fp32_s_64(vec_len: int):
    _test_compress(vec_len, 64, torch.float32)


@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=6))
def test_tcuscan_compress_fp32_s_128(vec_len: int):
    _test_compress(vec_len, 128, torch.float32)
