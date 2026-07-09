# --------------------------------------------------------------------------------
# Copyright (c) 2023-2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pytorch-tcuscan/
# for the full License text.
# --------------------------------------------------------------------------------
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

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)


def ref_complete_blocks(x, sums, num_blocks: int):
    "Reference implementation of `KernelCompleteBlocks` AscendC kernel."
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
    expected = torch.clone(x)
    for b in range(num_blocks):
        block_offset = b * block_len
        acc = prefix_sums[b]
        expected[block_offset : block_offset + block_len] += acc

    return expected.flatten()


def _test_complete_blocks(
    vec_len: int, tile_len: int, num_blocks: int, dtype: torch.dtype
):
    if dtype == torch.float32:
        x = 0.1 * torch.randn(vec_len, device=NPU_DEVICE).float()
    elif dtype == torch.int32:
        x = torch.randint(0, 3, size=(vec_len,), dtype=torch.int32, device=NPU_DEVICE)
    else:
        assert False, f"Unsupported dtype for reduce_tiles. Got {dtype}."

    # Sum-reductions per block
    sums = torch.sum(x.reshape(num_blocks, -1), dim=1, dtype=dtype).flatten()
    expected = ref_complete_blocks(x, sums, num_blocks)
    torch.npu.synchronize()
    actual = tcuscan_ops.run_complete_blocks(x, sums, tile_len)
    torch.npu.synchronize()

    assert expected.dtype == actual.dtype
    assert expected.shape == actual.shape
    assert torch.allclose(
        actual, expected, atol=1e-0, rtol=1e-5
    ), "Returned tensor does not match expected tensor."


@pytest.mark.parametrize("num_blocks", [2, 10, 20, 40, 80])
@pytest.mark.parametrize("multiplier", [3, 5, 7])
@pytest.mark.parametrize(
    "tile_ratio", [32, 16, 8, 4, 2]
)  # Fails when tile_ratio = 1 and s=128
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.int32], ids=str
)  # TODO: add support for int32
@pytest.mark.parametrize("s", [32, 64, 128])
def test_tcuscan_complete_blocks_fp32(
    num_blocks: int, s: int, multiplier: int, tile_ratio: int, dtype: torch.dtype
):
    tile_len = s * s // tile_ratio
    _test_complete_blocks(
        num_blocks * s * s * multiplier,
        tile_len,
        num_blocks,
        dtype,
    )
