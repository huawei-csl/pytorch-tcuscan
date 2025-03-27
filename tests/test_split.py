import random

import numpy as np
import pytest
import torch
import torch_npu  # noqa

import tcuscan_ops

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

NUM_CORES = 20

VEC_LENS = [
    256 - 1,
    1024 - 1,
    2048 - 1,
    4096 - 1,
    8192 - 1,
    1024 * 1024 - 1,
    256 + 1,
    1024 + 1,
    2048 + 1,
    4096 + 1,
    8192 + 1,
    1024 * 1024 + 1,
    1024 * 1024 + 13,
    1024 * 1024 + 27,
]


def _test_split_ind(vec_len: int, s: int, dtype: torch.dtype = torch.int16):
    "Unit tests split_ind operator for given input length, s and dtype."
    x = torch.randint(0, 2**7 - 1, (vec_len,)).to(dtype).npu()
    mask = (torch.randn(vec_len) > 0).to(torch.int8).npu()

    z = tcuscan_ops.run_split(x, mask, s)

    assert len(z) == len(x), "Input and output vector dimensions must agree."

    expected_left_part = torch.masked_select(x, mask == 1)
    num_selected = len(expected_left_part)
    assert torch.allclose(expected_left_part, z[:num_selected])

    expected_right_part = torch.masked_select(x, mask == 0)
    num_tail = len(expected_right_part)
    assert torch.allclose(expected_right_part, z[-num_tail:])


def test_tcuscan_split_ind_int16_s32():
    s = 32
    vec_len = 8 * NUM_CORES * s * s
    _test_split_ind(vec_len, s)


def test_tcuscan_split_ind_int16_s64():
    s = 64
    vec_len = 8 * NUM_CORES * s * s
    _test_split_ind(vec_len, s)


def test_tcuscan_split_int16_s128():
    s = 128
    vec_len = 8 * NUM_CORES * s * s
    _test_split_ind(vec_len, s)


def test_tcuscan_split_ind_fp16_s32():
    s = 32
    vec_len = 8 * NUM_CORES * s * s
    _test_split_ind(vec_len, s, torch.float16)


def test_tcuscan_split_ind_fp16_s64():
    s = 64
    vec_len = 8 * NUM_CORES * s * s
    _test_split_ind(vec_len, s, torch.float16)


def test_tcuscan_split_fp16_s128():
    s = 128
    vec_len = 8 * NUM_CORES * s * s
    _test_split_ind(vec_len, s, torch.float16)


@pytest.mark.parametrize("vec_len", VEC_LENS)
def test_tcuscan_split_s_32_padded(vec_len):
    s = 32
    _test_split_ind(vec_len, s, torch.int16)


@pytest.mark.parametrize("vec_len", VEC_LENS)
def test_tcuscan_split_fp16_s32_padded(vec_len):
    s = 32
    _test_split_ind(vec_len, s, torch.float16)
