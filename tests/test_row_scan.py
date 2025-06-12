import os
import random

import numpy as np
import pytest
import torch.nn.functional as F
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
S = 128

M_LIST = [NUM_CORES * S * S * i - 13 for i in range(1, 10)]


def _test_row_scan(m: int, dtype: torch.dtype):
    A = torch.ones(m, S, dtype=dtype, device=NPU_DEVICE)
    B = torch.tril(torch.ones((S, S), dtype=dtype, device=NPU_DEVICE))

    torch.npu.synchronize()
    actual = tcuscan_ops.run_row_scan(A, S).half()
    torch.npu.synchronize()
    expected = F.linear(A, B).flatten()
    torch.npu.synchronize()

    assert actual.dtype == expected.dtype
    assert torch.allclose(
        actual, expected, atol=0, rtol=1e-3
    ), f"Row scan ({dtype}) is wrong. s={S}, m={m}"


@pytest.mark.parametrize("m", M_LIST)
def test_tcuscan_row_scan_fp16(m: int):
    _test_row_scan(m, torch.float16)
