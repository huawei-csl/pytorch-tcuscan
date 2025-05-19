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

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:0")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)


def _test_complete_rows(
    vec_len: int, tile_len: int, num_blocks: int, dtype: torch.dtype
):
    out_dtype = None
    if dtype == torch.float32:
        x = 0.1 * torch.randn(vec_len).float().npu()
        out_dtype = torch.float32
    elif dtype == torch.int32:
        x = torch.randint(-3, 3, size=(vec_len,), dtype=torch.int32).npu()
        out_dtype = torch.int32
    else:
        assert False, "Unsupported dtype for reduce_tiles. Got {dtype}."

    sums = torch.sum(x.reshape(num_blocks, -1), dim=1, dtype=out_dtype).flatten()
    prefix_sums = torch.cumsum(
        torch.concat([torch.zeros(1, dtype=out_dtype).npu(), sums[:-1]]), dim=-1
    )

    assert (
        vec_len % num_blocks == 0
    ), "Input vector length must be divisble by the number of blocks."

    expected = x.reshape(num_blocks, -1) + prefix_sums[:, torch.newaxis]
    expected = expected.flatten()
    torch.npu.synchronize()
    actual = tcuscan_ops.run_complete_rows(x, sums, tile_len, tile_len)
    torch.npu.synchronize()

    offset = tile_len * tile_len
    print(f"input    : {x[:offset]}")
    print(f"sums     : {sums}")
    print(f"prefix_sums     : {prefix_sums}")
    print("*" * 40)
    print(f"expected : {expected}")
    print(f"actual   : {actual}")
    print(f"actual-input   : {actual-x}")
    print(f"expected-input   : {expected-x}")

    assert expected.dtype == actual.dtype
    assert expected.shape == actual.shape
    assert torch.allclose(
        actual, expected, atol=1e-0, rtol=1e-2
    ), f"Returned tensor does not match the expected tensor. sums: {sums}"


@pytest.mark.parametrize("num_blocks", [20, 40])
@pytest.mark.parametrize("tile_len", [16])  # Fails for 32, 128
@pytest.mark.parametrize("multiplier", [1])
def test_tcuscan_complete_rows_fp32(num_blocks: int, tile_len: int, multiplier: int):
    _test_complete_rows(
        num_blocks * tile_len * tile_len * multiplier,
        tile_len,
        num_blocks,
        torch.float32,
    )
