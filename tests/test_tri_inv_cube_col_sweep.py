#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

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


def triu_inv_cs(input_x, dtype: np.dtype = np.float16):
    """Return the matrix inverse of a 3d tensor where first dimension is batch dimension.
    Each batch dimension returns U_inv_n of input U_n, i.e., U_n U_inv_n = I_n.

    The algorithm is a matrix-formulation of the column sweep's algorithm.
    """
    output = np.zeros_like(input_x)
    batch_dim, n, _ = input_x.shape
    I_n = np.eye(n, dtype=np.float32)

    for idx in range(batch_dim):
        U_n = input_x[idx, :, :]
        U_n = 2 * I_n - U_n
        U_inv_n = I_n.copy()

        for k in reversed(range(n)):
            M = I_n.copy()
            M[:, k] = U_n[:, k]
            U_inv_n = M.astype(np.float32) @ U_inv_n.astype(np.float32)
            # FIXPIPE fp32 -> fp16
            U_inv_n = U_inv_n.astype(dtype)

        output[idx, :, :] = U_inv_n

    return output.astype(dtype)


def rand_triu_tensor(batch_size: int, n: int, dtype: np.dtype):
    "Returns a random unit upper triangular matrix of size n."
    A = 0.2 * np.random.rand(batch_size, n, n)
    A = np.triu(A)
    for k in range(batch_size):
        np.fill_diagonal(A[k, :, :], 1.0)
    return A.astype(dtype)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("matrix_size", [16, 32, 64, 128])
def test_tri_inv_cube_col_sweep(batch_size: int, matrix_size: int):

    data_type = np.float16
    input_x_cpu = rand_triu_tensor(batch_size, matrix_size, data_type)
    expected_cpu = triu_inv_cs(input_x_cpu)

    # Convert input matrices from row-major order to column-major order
    input_x_cpu = input_x_cpu.transpose(0, 2, 1)
    torch.npu.synchronize()
    input_x = torch.from_numpy(input_x_cpu).half().npu()
    torch.npu.synchronize()
    expected = torch.from_numpy(expected_cpu).half().npu()
    torch.npu.synchronize()

    print("*" * 40)
    print(f"Input_x (shape: {input_x.shape})")
    print("*" * 40)

    torch.npu.synchronize()
    actual = tcuscan_ops.run_tri_inv_cube_col_sweep(input_x)
    torch.npu.synchronize()

    assert actual.shape == expected.shape
    assert torch.allclose(actual.float(), expected.float(), atol=1e-1, rtol=1e-2)
