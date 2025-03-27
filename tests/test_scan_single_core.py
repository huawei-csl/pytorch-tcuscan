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
    for multiplier in range(1, max_iters):
        yield multiplier * s * s


def _test_fp16(vec_len: int, s: int):
    x = torch.randn(vec_len).half().npu()

    expected = torch.cumsum(x, dim=-1, dtype=torch.float32)
    torch.npu.synchronize()
    actual = tcuscan_ops.run_scan_single_core(x, s)
    torch.npu.synchronize()
    assert torch.allclose(
        actual, expected, atol=1e-02
    ), f"single-core scan (fp16) is wrong. s={s}, vec_len={vec_len}"


@pytest.mark.parametrize("vec_len", get_lengths(s=16, max_iters=32))
def test_sc_scan_fp16_s_16(vec_len: int):
    _test_fp16(vec_len, 16)


@pytest.mark.parametrize("vec_len", get_lengths(s=32, max_iters=24))
def test_sc_scan_fp16_s_32(vec_len: int):
    _test_fp16(vec_len, 32)


@pytest.mark.parametrize("vec_len", get_lengths(s=64, max_iters=8))
def test_sc_scan_fp16_s_64(vec_len: int):
    _test_fp16(vec_len, 64)


@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=6))
def test_sc_scan_fp16_s_128(vec_len: int):
    _test_fp16(vec_len, 128)
