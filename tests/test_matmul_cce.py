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


M_LIST = [NUM_CORES * 128 * i for i in range(1, 10)]


def _test_matmul_cce(m: int, dtype: torch.dtype):
    N, K = 512, 512
    A = torch.randn(m, K, dtype=dtype).npu()
    B = torch.randn(N, K, dtype=dtype).npu()

    actual = tcuscan_ops.run_matmul_cce(A, B)
    expected = F.linear(A, B)

    avg_diff = torch.mean(torch.abs(actual - expected))
    assert (
        avg_diff < 1e-3
    ), f"Matrix multiplication produce has average error: {avg_diff}"


@pytest.mark.parametrize("m", M_LIST)
def test_tcuscan_matmul_cce_fp16(m: int):
    _test_matmul_cce(m, torch.float16)
