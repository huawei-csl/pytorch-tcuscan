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


def _test_single_core_scan(vec_len: int, s: int, dtype: torch.dtype):
    out_dtype = None
    if dtype == torch.float16:
        x = torch.randn(vec_len).half().npu()
        out_dtype = torch.float32
    elif dtype == torch.int8:
        x = torch.randint(-2, 2, size=(vec_len,), dtype=torch.int8).npu()
        out_dtype = torch.int32
    else:
        assert False, "SingleCoreScan supports only fp16/int8. Got {dtype}."

    expected = torch.cumsum(x, dim=-1, dtype=out_dtype)
    torch.npu.synchronize()
    actual = tcuscan_ops.run_scan_single_core(x, s)
    torch.npu.synchronize()
    assert torch.allclose(
        actual, expected, atol=1e-02
    ), f"single-core scan ({dtype}) is wrong. s={s}, vec_len={vec_len}"


@pytest.mark.parametrize("multiplier", range(1, 10))
@pytest.mark.parametrize("s", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.int8], ids=str)
def test_single_core_scan(multiplier: int, s: int, dtype: torch.dtype):
    vec_len = multiplier * s * s
    _test_single_core_scan(vec_len, s, dtype)
