from math import ceil

import pytest
import torch
import torch_npu  # noqa

import tcuscan_ops

# Input array size to benchmark sort
_SORT_SIZES = [
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1179648,
    4325376,
    16908288,
    25165824,
]


def get_sizes():
    s = 128
    max_size = 1e7
    num_cores = 20

    # Maximum number of iterations
    # Remove '- 3' for more thorough testing
    max_iters = ceil(max_size / (num_cores * s * s)) - 5

    return [i * num_cores * s * s for i in range(1, max_iters, 16 * 128 // s)]


def generate_random_int(dtype, vec_len):
    if dtype == torch.int16:
        return (
            torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, (vec_len,))
            .to(dtype)
            .npu()
        )
    else:
        return torch.rand((vec_len,), dtype=dtype).npu()


def _test_sort(vec_len: int, dtype: torch.dtype, s: int):
    torch.manual_seed(42)
    x = generate_random_int(dtype, vec_len)

    expected, expected_indices = torch.sort(x, dim=-1, descending=False)
    actual, actual_indices = tcuscan_ops.run_radix_sort(x, s)

    assert len(expected) == len(
        actual
    ), f"Vector size must agree. Expected: {len(expected)}. Actual: {actual}"
    assert len(expected_indices) == len(
        actual_indices
    ), f"Indices size must agree. Expected: {len(expected_indices)}. Actual: {actual_indices}"

    assert torch.allclose(expected, actual), "Output must be sorted"


@pytest.mark.parametrize("vec_len", _SORT_SIZES)
def test_tcuscan_sort_fp16_s_32(vec_len):
    _test_sort(vec_len, torch.float16, 32)


@pytest.mark.parametrize("vec_len", _SORT_SIZES)
def test_tcuscan_sort_fp16_s_64(vec_len):
    s = 64
    if vec_len >= s * s:
        _test_sort(vec_len, torch.float16, s)


@pytest.mark.parametrize("vec_len", _SORT_SIZES)
def test_tcuscan_sort_fp16_s_128(vec_len):
    s = 128
    if vec_len >= s * s:
        _test_sort(vec_len, torch.float16, s)


# TODO: See ISSUE-52
# Radix sort fails on irregular length.
# @pytest.mark.parametrize("vec_len", _SORT_SIZES[1:])
# def test_tcuscan_sort_int16_s_128_irregular(vec_len):
#    s = 128
#    dtype = torch.int16
#    _test_sort(vec_len, dtype, s)


@pytest.mark.parametrize("vec_len", get_sizes())
def test_tcuscan_sort_int16_s_64(vec_len):
    s = 64
    dtype = torch.int16
    _test_sort(vec_len, dtype, s)


@pytest.mark.parametrize("vec_len", get_sizes())
def test_tcuscan_sort_int16_s_128(vec_len):
    s = 128
    dtype = torch.int16
    _test_sort(vec_len, dtype, s)
