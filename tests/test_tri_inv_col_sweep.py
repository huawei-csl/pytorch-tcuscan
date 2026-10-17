# --------------------------------------------------------------------------------
# Copyright (c) 2023-2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pytorch-tcuscan/
# for the full License text.
# --------------------------------------------------------------------------------
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


def np_triu_inv_cs(input_x, dtype: np.dtype = np.float16):
    """Return the matrix inverse of a 3d tensor where first dimension is batch dimension.
    Each batch dimension returns U_inv_n of input U_n, i.e., U_n U_inv_n = I_n.

    The algorithm is the column sweep's algorithm (vectorized) applied on each column of I, i.e.,
    Ax=e_j for j=0,1,...,(n-1).
    """
    output = np.zeros_like(input_x)
    batch_dim = input_x.shape[0]
    for reps in range(batch_dim):
        U_n = input_x[reps, :, :].copy()
        n = U_n.shape[1]
        U_inv_n = np.zeros_like(U_n)

        for j in range(n):

            b = np.zeros(n, dtype=dtype)
            b[j] = 1  # b = e_j

            x = np.zeros(n, dtype=dtype)
            for k in range(n - 1, -1, -1):
                x[k] = b[k]  # must be b[k] / U_n[k,k]

                if k > 0:
                    b[:k] -= U_n[:k, k] * x[k]
            U_inv_n[:, j] = x

        output[reps, :, :] = U_inv_n

    return output.astype(dtype)


def rand_np_tril(batch_size: int, n: int, dtype: np.dtype):
    "Returns a random strictly lower triangular matrix of size n."
    A = np.random.rand(batch_size, n, n).astype(dtype)
    A = np.tril(A, k=-1)
    return A.astype(dtype)


def ones_np_tril(batch_size: int, n: int, dtype: np.dtype):
    "Returns an all-ones strictly lower triangular matrix of size n."
    A = np.ones((batch_size, n, n)).astype(dtype)
    A = np.tril(A, k=-1)
    return A.astype(dtype)


def compute_inv_with_numpy(A: np.array):
    A_copy = A.astype(np.float64)
    for k in range(A.shape[0]):
        np.fill_diagonal(A_copy[k, :, :], 1.0)
    return np.linalg.inv(A_copy).astype(A.dtype)


@pytest.mark.parametrize("batch_size", [2, 4, 40, 256])
@pytest.mark.parametrize("matrix_size", [16, 32, 64, 128])
@pytest.mark.parametrize("data_type", [np.float16, np.float32], ids=str)
@pytest.mark.parametrize(
    "mat_gen",
    (rand_np_tril, ones_np_tril),
)
def test_tri_inv_col_sweep(
    batch_size: int,
    matrix_size: int,
    data_type: np.dtype,
    mat_gen: callable,
):

    input_x_cpu = mat_gen(batch_size, matrix_size, data_type)
    expected_cpu = np_triu_inv_cs(input_x_cpu.transpose(0, 2, 1), data_type)

    # Convert input matrices from row-major order to column-major order
    input_x_cpu = input_x_cpu.transpose(0, 2, 1)
    input_x = torch.from_numpy(input_x_cpu).npu()
    expected = torch.from_numpy(expected_cpu).npu()

    torch.npu.synchronize()
    actual = tcuscan_ops.run_tri_inv_col_sweep(input_x)
    torch.npu.synchronize()
    # Transpose matrices back to row-major order
    actual = actual.transpose(2, 1)
    torch.npu.synchronize()

    assert actual.shape == expected.shape, "Output shape does not match expected shape."
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 40, 256])
@pytest.mark.parametrize("matrix_size", [16, 32, 64, 128])
@pytest.mark.parametrize("data_type", [np.float32], ids=str)
@pytest.mark.parametrize(
    "mat_gen,atol,rtol",
    [
        (rand_np_tril, 1e-5, 1e-5),
        (ones_np_tril, 0, 0),
    ],
)
def test_tri_inv_col_sweep_np_linalg_inv(
    batch_size: int,
    matrix_size: int,
    data_type: np.dtype,
    mat_gen: callable,
    atol: float,
    rtol: float,
):

    input_x_cpu = mat_gen(batch_size, matrix_size, data_type)
    golden_numpy_cpu = compute_inv_with_numpy(input_x_cpu)

    # Convert input matrices from row-major order to column-major order
    input_x_cpu = input_x_cpu.transpose(0, 2, 1)
    input_x = torch.from_numpy(input_x_cpu).npu()
    golden_cpu = torch.from_numpy(golden_numpy_cpu)

    torch.npu.synchronize()
    actual = tcuscan_ops.run_tri_inv_col_sweep(input_x)
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    torch.npu.synchronize()

    # rtol must be scaled w.r.t to the input size, see Higham's paper, Eq. (2.3)
    # https://nhigham.com/wp-content/uploads/2023/08/high89t.pdf
    scaled_rtol = min([0.05, 10 * (matrix_size + batch_size) * rtol])
    assert torch.allclose(actual_cpu, golden_cpu, atol=atol, rtol=scaled_rtol)


@pytest.mark.parametrize("batch_size", [2, 4, 40, 256])
@pytest.mark.parametrize("matrix_size", [16, 32, 64, 128])
@pytest.mark.parametrize("data_type", [np.float16, np.float32], ids=str)
@pytest.mark.parametrize(
    "mat_gen",
    (rand_np_tril, ones_np_tril),
)
def test_tri_inv_col_sweep_diagonal_robustness(
    batch_size: int,
    matrix_size: int,
    data_type: np.dtype,
    mat_gen: callable,
):

    x_zero_diagonal = mat_gen(batch_size, matrix_size, data_type)
    x_unit_diagonal = x_zero_diagonal.copy()
    for k in range(batch_size):
        np.fill_diagonal(x_unit_diagonal[k, :, :], 1.0)
    x_zero_diagonal = x_zero_diagonal.transpose(0, 2, 1)
    x_unit_diagonal = x_unit_diagonal.transpose(0, 2, 1)

    # Convert input matrices from row-major order to column-major order
    x_zero_npu = torch.from_numpy(x_zero_diagonal).npu()
    x_unit_npu = torch.from_numpy(x_unit_diagonal).npu()

    torch.npu.synchronize()
    actual_zero = tcuscan_ops.run_tri_inv_col_sweep(x_zero_npu)
    torch.npu.synchronize()
    actual_unit = tcuscan_ops.run_tri_inv_col_sweep(x_unit_npu)
    torch.npu.synchronize()
    actual_zero_cpu = actual_zero.cpu()
    actual_unit_cpu = actual_unit.cpu()
    torch.npu.synchronize()

    assert torch.equal(actual_zero_cpu, actual_unit_cpu)


# `num_out_tiles` controls how many column tiles each output matrix is split
# into, one per vector core. It changes only the work distribution, so every
# valid value has to produce the same inverse.
@pytest.mark.parametrize(
    "matrix_size, num_out_tiles",
    [
        (16, 1),
        (16, 2),
        (16, 4),
        (16, 8),
        (16, 16),
        (32, 1),
        (32, 2),
        (32, 4),
        (32, 8),
        (32, 16),
        (32, 32),
        (64, 1),
        (64, 4),
        (64, 16),
        (64, 64),
        (128, 1),
        (128, 8),
        (128, 32),
        (128, 128),
    ],
)
@pytest.mark.parametrize("data_type", [np.float16, np.float32], ids=str)
def test_tri_inv_col_sweep_num_out_tiles(
    matrix_size: int, num_out_tiles: int, data_type: np.dtype
):
    batch_size = 4
    input_x_cpu = rand_np_tril(batch_size, matrix_size, data_type).transpose(0, 2, 1)
    input_x = torch.from_numpy(input_x_cpu).npu()

    torch.npu.synchronize()
    expected = tcuscan_ops.run_tri_inv_col_sweep(input_x)
    actual = tcuscan_ops.run_tri_inv_col_sweep(input_x, num_out_tiles)
    torch.npu.synchronize()

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected), (
        f"num_out_tiles={num_out_tiles} disagrees with the default for "
        f"matrix_size={matrix_size}, dtype={data_type}"
    )


def test_tri_inv_col_sweep_default_matches_explicit():
    # The default is 16 columns per tile, i.e. `matrix_size / 16`.
    matrix_size = 64
    x = torch.from_numpy(
        rand_np_tril(2, matrix_size, np.float32).transpose(0, 2, 1)
    ).npu()

    torch.npu.synchronize()
    assert torch.equal(
        tcuscan_ops.run_tri_inv_col_sweep(x),
        tcuscan_ops.run_tri_inv_col_sweep(x, matrix_size // 16),
    )


@pytest.mark.parametrize("num_out_tiles", [-1, 3, 5, 7, 24, 129])
def test_tri_inv_col_sweep_rejects_invalid_num_out_tiles(num_out_tiles: int):
    # 128 is not divisible by any of these, and negatives are rejected outright.
    x = torch.from_numpy(rand_np_tril(2, 128, np.float32).transpose(0, 2, 1)).npu()
    with pytest.raises(RuntimeError):
        tcuscan_ops.run_tri_inv_col_sweep(x, num_out_tiles)
