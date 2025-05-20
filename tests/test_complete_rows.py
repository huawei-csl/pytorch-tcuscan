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


def ref_complete_rows(x, sums, tile_len: int, num_blocks: int):
    "Reference implementation of `KernelCompleteRows` AscendC kernel."
    vec_len = len(x)

    prefix_sums = torch.cumsum(
        torch.concat([torch.zeros(1, dtype=x.dtype).npu(), sums[:-1]]), dim=-1
    )

    assert sums.numel() == num_blocks
    assert prefix_sums.numel() == num_blocks

    assert (
        vec_len % num_blocks == 0
    ), "Input vector length must be divisble by the number of blocks."

    block_len = vec_len // num_blocks
    tile_width = tile_len
    tile_height = block_len // tile_width
    expected = torch.clone(x)
    for b in range(num_blocks):
        block_offset = b * block_len
        acc = prefix_sums[b]
        for h in range(tile_height):
            expected[
                block_offset + h * tile_width : block_offset + (h + 1) * tile_width
            ] += acc
            acc = expected[block_offset + (h + 1) * tile_width - 1]

    return expected.flatten()


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

    # Sum-reductions per block
    sums = torch.sum(x.reshape(num_blocks, -1), dim=1, dtype=out_dtype).flatten()
    expected = ref_complete_rows(x, sums, tile_len, num_blocks)
    torch.npu.synchronize()
    actual = tcuscan_ops.run_complete_rows(x, sums, tile_len, tile_len)
    torch.npu.synchronize()

    assert expected.dtype == actual.dtype
    assert expected.shape == actual.shape
    assert torch.allclose(
        actual, expected, atol=1e-0, rtol=1e-2
    ), f"Returned tensor does not match the expected tensor. sums: {sums}"


@pytest.mark.parametrize("num_blocks", [2, 10, 20, 40])
@pytest.mark.parametrize("tile_len", [32, 128, 512])  # Fails for 32, 128
@pytest.mark.parametrize("multiplier", [3])
def test_tcuscan_complete_rows_fp32(num_blocks: int, tile_len: int, multiplier: int):
    _test_complete_rows(
        num_blocks * tile_len * tile_len * multiplier,
        tile_len,
        num_blocks,
        torch.float32,
    )
