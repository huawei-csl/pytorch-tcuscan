import pytest
import torch_npu  # noqa

import tcuscan_ops
import torch

_VEC_LENS = [
    100 * 1024,
    200 * 1024,
    300 * 1024,
    400 * 1024,
    10 * 1024 * 1024 - 1,
    20 * 1024 * 1024 - 10,
]


def generate_random_int(dtype, vec_len):
    return (
        torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, (vec_len,))
        .to(dtype)
        .npu()
    )


def _test_topk_pivot(size: int, k: int, dtype: torch.dtype) -> None:

    # TODO (ISSUE-53: anastasios): randomized testing fails with TopK :-(
    torch.manual_seed(42)

    if dtype == torch.int16:
        x = generate_random_int(torch.int16, size)
    else:
        x = 10 * torch.randn(size).to(torch.half).npu()

    torch.npu.synchronize()
    pivot = tcuscan_ops.run_topk_pivot_fp16(x, k)
    estimate = pivot[0]
    torch.npu.synchronize()

    expected_topk, _ = torch.topk(x, k, largest=True)

    # Sort the topk elements and indices to compare them
    expected_topk, _ = torch.sort(expected_topk, dim=-1)

    assert len(expected_topk) == k
    print(f"min / max: {torch.min(x)} / {torch.max(x)}")
    print(f"expected_topk: {expected_topk}")

    expected_topk_pivot = expected_topk[-1]
    estimator_len = torch.sum(x >= estimate)
    assert (
        expected_topk_pivot < estimate
    ), f"Expected: {expected_topk_pivot}, estimated: {estimate}, est_len: {estimator_len}"


@pytest.mark.parametrize("vec_len", _VEC_LENS)
def test_topk_pivot(vec_len):
    k = 64
    _test_topk_pivot(vec_len, k, torch.float16)
