# --------------------------------------------------------------------------------
# Copyright (c) 2023-2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pytorch-tcuscan/
# for the full License text.
# --------------------------------------------------------------------------------
import os

import numpy as np
import pytest
import torch_npu  # noqa

import tcuscan_ops
import torch

_VEC_LENS = [5 * 1024, 10 * 1024, 10 * 1024 * 1024 - 1, 20 * 1024 * 1024 - 10]
NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.set_device(NPU_DEVICE)


def generate_random_int(dtype, vec_len, low=None, high=None):
    low = torch.iinfo(dtype).min if low is None else low
    high = torch.iinfo(dtype).max if high is None else high
    return torch.randint(low, high, (vec_len,)).to(dtype).npu()


# `compare::LessThan<int16_t>` evaluates `pivot - x_i` in int16 arithmetic and
# reads the sign bit of the result, so the comparison inverts once the operands
# are further apart than the int16 range. Inputs bounded by half of that range
# keep every `pivot - x_i` representable.
_INT16_SAFE_BOUND = 1 << 14


def _generate_input(dtype: torch.dtype, size: int, distribution: str) -> torch.Tensor:
    if dtype == torch.int16:
        # The int16 input already spans both signs, so `distribution` only
        # applies to the floating point inputs.
        return generate_random_int(
            torch.int16, size, low=-_INT16_SAFE_BOUND, high=_INT16_SAFE_BOUND - 1
        )
    if distribution == "positive":
        values = torch.rand(size)
    elif distribution == "negative":
        values = -torch.rand(size)
    elif distribution == "mixed":
        values = 2.0 * torch.rand(size) - 1.0
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    return values.to(torch.half).npu()


def _test_topk(
    size: int, k: int, dtype: torch.dtype, distribution: str = "positive"
) -> None:

    # TODO (ISSUE-53: anastasios): randomized testing fails with TopK :-(
    torch.manual_seed(42)

    s = 128
    x = _generate_input(dtype, size, distribution)

    min_x = torch.min(x)
    max_x = torch.max(x)

    torch.npu.synchronize()
    if dtype == torch.int16:
        actual_topk, actual_indices = tcuscan_ops.run_topk_int16(x, k, min_x, max_x, s)
    else:
        actual_topk, actual_indices = tcuscan_ops.run_topk_fp16(x, k, min_x, max_x, s)

    torch.npu.synchronize()

    expected_topk, _ = torch.topk(x, k, largest=True)

    assert len(actual_topk) == k
    assert (
        len(actual_indices) == k
    ), f"Output indices must have size K. K={len(actual_indices)}"

    # Every returned index must point at the value returned next to it. Values
    # can repeat, so this pairing is checked before the tensors get sorted.
    assert torch.equal(x[actual_indices.long()].float(), actual_topk.float())

    # Sort the topk elements and indices to compare them
    actual_topk, _ = torch.sort(actual_topk, dim=-1)
    expected_topk, _ = torch.sort(expected_topk, dim=-1)

    print(f"min / max: {torch.min(x)} / {torch.max(x)}")
    print(f"actual_topk: {actual_topk}")
    print(f"expected_topk: {expected_topk}")
    assert torch.allclose(actual_topk.float(), expected_topk.float())

    assert actual_indices.unique().numel() == k, "Output indices must be unique"

    kth_value = expected_topk[0]  # `expected_topk` is sorted ascending.
    above_kth = (x > kth_value).nonzero().flatten()
    assert set(above_kth.tolist()) <= set(actual_indices.long().tolist())


dtype = torch.float16


@pytest.mark.parametrize("vec_len", _VEC_LENS)
def test_topk_K_1(vec_len):
    k = 1
    _test_topk(vec_len, k, dtype)


@pytest.mark.parametrize("vec_len", _VEC_LENS)
def test_topk_K_2(vec_len):
    k = 2
    _test_topk(vec_len, k, dtype)


@pytest.mark.parametrize("vec_len", _VEC_LENS)
def test_topk_K_5(vec_len):
    k = 5
    _test_topk(vec_len, k, dtype)


@pytest.mark.parametrize("vec_len", _VEC_LENS)
def test_topk_K_SQRT(vec_len):
    k = int(np.sqrt(vec_len))
    _test_topk(vec_len, k, dtype)


# The bisection bounds are halved in the value domain, so inputs that are not
# strictly positive exercise a different convergence path.
@pytest.mark.parametrize("vec_len", _VEC_LENS)
@pytest.mark.parametrize("k", [1, 5, 1024])
@pytest.mark.parametrize("distribution", ["negative", "mixed"])
def test_topk_signed_values(vec_len, k, distribution):
    _test_topk(vec_len, k, dtype, distribution=distribution)


@pytest.mark.parametrize("vec_len", _VEC_LENS)
@pytest.mark.parametrize("k", [1, 5, 1024])
def test_topk_int16(vec_len, k):
    _test_topk(vec_len, k, torch.int16)


@pytest.mark.xfail(
    strict=True,
    reason="`compare::LessThan<int16_t>` overflows when `pivot - x_i` leaves "
    "the int16 range, so inputs spanning more than half of that range are "
    "misclassified.",
)
def test_topk_int16_full_range():
    torch.manual_seed(42)

    x = generate_random_int(torch.int16, 10 * 1024)
    actual_topk, _ = tcuscan_ops.run_topk_int16(
        x, 1024, torch.min(x), torch.max(x), 128
    )
    expected_topk, _ = torch.topk(x, 1024, largest=True)

    torch.npu.synchronize()
    assert torch.equal(
        torch.sort(actual_topk)[0].int(), torch.sort(expected_topk)[0].int()
    )


# `num_tiles` is a prime number here, which used to collapse the kernel launch
# down to a single block.
@pytest.mark.parametrize("k", [1, 5])
def test_topk_prime_number_of_tiles(k):
    tile_elems = 128 * 128
    _test_topk(641 * tile_elems, k, dtype)
