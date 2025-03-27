import random

import numpy as np
import pytest
import torch
import torch_npu  # noqa

import tcuscan_ops

torch.npu.config.allow_internal_format = False


random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def get_lengths(s: int, max_iters: int):
    NUM_AI_CORES = 20
    for multiplier in range(1, max_iters):
        yield multiplier * NUM_AI_CORES * s * s


def _test_fp16(vec_len: int, s: int):
    x = torch.randn(vec_len).half().npu()

    expected = torch.cumsum(x, dim=-1, dtype=torch.float32)
    torch.npu.synchronize()
    actual = tcuscan_ops.run_scan_multi_core(x, s)
    torch.npu.synchronize()
    assert actual.dtype == expected.dtype
    assert torch.allclose(
        actual, expected, atol=1e-02
    ), f"multi-core scan (fp16) is wrong. s={s}, vec_len={vec_len}"


def _test_int8(vec_len: int, s: int):
    x = torch.randint(0, 10, size=(vec_len,), dtype=torch.int8).npu()

    assert x.dtype == torch.int8

    expected = torch.cumsum(x, dim=-1, dtype=torch.int32)
    torch.npu.synchronize()
    actual = tcuscan_ops.run_scan_multi_core(x, s)
    torch.npu.synchronize()
    assert actual.dtype == expected.dtype
    assert torch.allclose(
        actual, expected, atol=1e-02
    ), f"single-core scan (int8) is wrong. s={s}, vec_len={vec_len}"


@pytest.mark.parametrize("vec_len", get_lengths(s=16, max_iters=16))
def test_mcscan_fp16_s_16(vec_len: int):
    _test_fp16(vec_len, 16)


@pytest.mark.parametrize("vec_len", get_lengths(s=32, max_iters=12))
def test_mcscan_fp16_s_32(vec_len: int):
    _test_fp16(vec_len, 32)


@pytest.mark.parametrize("vec_len", get_lengths(s=64, max_iters=8))
def test_mcscan_fp16_s_64(vec_len: int):
    _test_fp16(vec_len, 64)


@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=6))
def test_mcscan_fp16_s_128(vec_len: int):
    _test_fp16(vec_len, 128)


@pytest.mark.parametrize("vec_len", get_lengths(s=32, max_iters=24))
def test_mcscan_int8_s_32(vec_len: int):
    _test_int8(vec_len, 32)


@pytest.mark.parametrize("vec_len", get_lengths(s=64, max_iters=8))
def test_mcscan_int8_s_64(vec_len: int):
    _test_int8(vec_len, 64)


@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=6))
def test_mcscan_int8_s_128(vec_len: int):
    _test_int8(vec_len, 128)
