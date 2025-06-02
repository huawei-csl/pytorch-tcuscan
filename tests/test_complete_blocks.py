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


def ref_complete_blocks(x, tile_len: int):
    "Reference implementation of `KernelCompleteBlocks` AscendC kernel."
    vec_len = len(x)
    expected = x.clone()
    running_sum = 0
    for offset in range(0, vec_len, tile_len):
        this_tile_len = tile_len
        if offset + this_tile_len >= vec_len:
            this_tile_len = vec_len - offset
        expected[offset : offset + this_tile_len] = (
            expected[offset : offset + this_tile_len] + running_sum
        )
        running_sum = expected[offset + this_tile_len - 1]

    return expected


def _test_complete_blocks(vec_len: int, tile_len: int, dtype: torch.dtype):
    if dtype == torch.float32:
        x_np_cpu = np.random.randn(vec_len).astype(np.float32)
        x_cpu = torch.from_numpy(x_np_cpu)
    else:
        assert False, f"Unsupported dtype for reduce_tiles. Got {dtype}."

    expected_cpu = ref_complete_blocks(x_cpu, tile_len)

    x = x_cpu.npu()
    torch.npu.synchronize()
    actual = tcuscan_ops.run_complete_blocks(x, tile_len)
    actual_cpu = actual.cpu()
    torch.npu.synchronize()
    assert expected_cpu.dtype == actual_cpu.dtype
    assert expected_cpu.shape == actual_cpu.shape
    assert torch.allclose(
        actual_cpu, expected_cpu, atol=0, rtol=1e-15
    ), "Returned tensor does not match the expected tensor."


@pytest.mark.parametrize("matrix_size", [32, 64, 128])
@pytest.mark.parametrize("multiplier", range(1, 10))
def test_tcuscan_complete_blocks_fp32(matrix_size: int, multiplier: int):
    tile_len = matrix_size * matrix_size
    vec_len = tile_len * multiplier
    _test_complete_blocks(
        vec_len,
        tile_len,
        torch.float32,
    )
