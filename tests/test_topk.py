import numpy as np
import pytest
import torch
import torch_npu  # noqa

import tcuscan_ops

_VEC_LENS = [5 * 1024, 10 * 1024, 10 * 1024 * 1024 - 1, 20 * 1024 * 1024 - 10]


def generate_random_int(dtype, vec_len):
    return (
        torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, (vec_len,))
        .to(dtype)
        .npu()
    )


def _test_topk(size: int, k: int, dtype: torch.dtype) -> None:

    # TODO (ISSUE-53: anastasios): randomized testing fails with TopK :-(
    torch.manual_seed(42)

    s = 128
    if dtype == torch.int16:
        x = generate_random_int(torch.int16, size)
    else:
        x = torch.rand(size).to(torch.half).npu()

    min_x = torch.min(x)
    max_x = torch.max(x)

    torch.npu.synchronize()
    actual_topk, actual_indices = tcuscan_ops.run_topk_int16(x, k, min_x, max_x, s)
    torch.npu.synchronize()

    (
        expected_topk,
        expected_indices,
    ) = torch.topk(x, k, largest=True)

    # Sort the topk elements and indices to compare them
    actual_topk, _ = torch.sort(actual_topk, dim=-1)
    expected_topk, _ = torch.sort(expected_topk, dim=-1)

    assert len(actual_topk) == k
    print(f"min / max: {torch.min(x)} / {torch.max(x)}")
    print(f"actual_topk: {actual_topk}")
    print(f"expected_topk: {expected_topk}")
    assert torch.allclose(actual_topk.float(), expected_topk.float())

    actual_indices, _ = torch.sort(actual_indices, dim=-1)
    expected_indices, _ = torch.sort(expected_indices, dim=-1)

    assert (
        len(actual_indices) == k
    ), f"Output indices must have size K. K={len(actual_indices)}"
    assert torch.allclose(expected_indices.float(), actual_indices.float())


@pytest.mark.parametrize("vec_len", _VEC_LENS)
def test_topk_K_1(vec_len):
    k = 1
    _test_topk(vec_len, k, torch.int16)


@pytest.mark.parametrize("vec_len", _VEC_LENS)
def test_topk_K_2(vec_len):
    k = 2
    _test_topk(vec_len, k, torch.int16)


@pytest.mark.parametrize("vec_len", _VEC_LENS)
def test_topk_K_5(vec_len):
    k = 5
    _test_topk(vec_len, k, torch.int16)


@pytest.mark.parametrize("vec_len", _VEC_LENS)
def test_topk_K_SQRT(vec_len):
    k = int(np.sqrt(vec_len))
    _test_topk(vec_len, k, torch.int16)
