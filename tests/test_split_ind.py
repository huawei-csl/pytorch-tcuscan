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
    indices_in = torch.cumsum(torch.ones(vec_len), dim=-1).to(torch.int32).npu()

    z, indices_out = tcuscan_ops.run_split_ind(x, mask, indices_in, s)

    assert len(z) == len(x), "Input and output vector dimensions must agree."
    assert len(z) == len(indices_out)

    expected_left_part = torch.masked_select(x, mask == 1)
    expected_left_indices = (
        torch.nonzero(mask == 1, as_tuple=True)[0].to(torch.int32) + 1
    )
    num_selected = len(expected_left_part)
    assert torch.allclose(expected_left_part, z[:num_selected])
    assert torch.allclose(expected_left_indices, indices_out[:num_selected])

    expected_right_part = torch.masked_select(x, mask == 0)
    expected_right_indices = (
        torch.nonzero(mask == 0, as_tuple=True)[0].to(torch.int32) + 1
    )

    num_tail = len(expected_right_part)
    assert torch.equal(expected_right_part, z[-num_tail:])
    assert torch.equal(expected_right_indices, indices_out[-num_tail:])


@pytest.mark.parametrize("dtype", [torch.int16, torch.float16])
@pytest.mark.parametrize("s", [32, 64, 128])
@pytest.mark.parametrize("vec_len", VEC_LENS)
def test_tcuscan_split(dtype: torch.dtype, s, vec_len):
    _test_split_ind(vec_len, s, dtype)
