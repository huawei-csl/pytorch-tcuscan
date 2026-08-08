# --------------------------------------------------------------------------------
# Copyright (c) 2023-2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pytorch-tcuscan/
# for the full License text.
# --------------------------------------------------------------------------------
import random

import numpy as np
import pytest
import torch_npu  # noqa

import tcuscan_ops
import torch

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

_VEC_LENS = [
    10 * 1024 * 1024,
    20 * 1024 * 1024,
    30 * 1024 * 1024,
    40 * 1024 * 1024,
]


def generate_random_int(dtype, vec_len):
    return (
        torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, (vec_len,))
        .to(dtype)
        .npu()
    )


def _test_topk_pivot(size: int, k: int, dtype: torch.dtype) -> None:
    if dtype == torch.int16:
        x = generate_random_int(torch.int16, size)
    else:
        x = 10 * torch.randn(size).to(torch.half).npu()

    torch.npu.synchronize()
    # FIXME(anastasios): k_inner must be 8 here.
    k_is_ignored_for_now = 8
    pivot_estimates = tcuscan_ops.run_topk_pivot_fp16(x, k_is_ignored_for_now)
    print(len(pivot_estimates))
    print(pivot_estimates.shape)
    print(f"Unsorted : {pivot_estimates}")
    estimate, _ = torch.sort(pivot_estimates)
    print(f"Sorted   : {estimate}")
    assert True
    estimate = estimate[-1]
    torch.npu.synchronize()

    estimator_len = torch.sum(x >= estimate)
    estimator_ratio = estimator_len / k

    expected_topk, _ = torch.topk(x, k, largest=True)
    # Sort the topk elements and indices to compare them
    expected_topk, _ = torch.sort(expected_topk, dim=-1)
    assert len(expected_topk) == k
    expected_k_largest = expected_topk[-1]
    assert (
        expected_k_largest > estimate
    ), f"Estimator is not a top-K lower-bound. Expected: {expected_k_largest}, estimated: {estimate}. Ratio: {estimator_ratio}"


@pytest.mark.parametrize("vec_len", _VEC_LENS)
def test_topk_pivot(vec_len):
    k = 2048
    _test_topk_pivot(vec_len, k, torch.float16)
