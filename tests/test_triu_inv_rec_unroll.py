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
torch.npu.set_device(NPU_DEVICE)


def block_ones_matrix(n, block_dim):
    U_ = np.ones((16, 16))
    n_blocks = n // 16
    U = np.zeros((block_dim, n, n))
    for k in range(block_dim):
        for i in range(n_blocks):
            start = i * 16
            end = i * 16 + 16
            U[k, start:end, start:end] = U_
    return torch.from_numpy(U - np.tril(U))


def zeros_matrix(n, block_dim):
    return torch.zeros((block_dim, n, n))


def ones_matrix(n, block_dim):
    U = np.ones((block_dim, n, n))
    return torch.from_numpy(U - np.tril(U))


def big_bad_matrix(n, block_dim):
    Z = np.zeros((block_dim, n, n))
    for k in range(block_dim):
        Z[k] = 0.5 * np.identity(n).astype(np.float64)
    U = np.triu(0.5 * np.ones((block_dim, n, n)))
    U = U - Z
    return torch.from_numpy(U)


def random_matrix(n, block_dim=2, scale=0.1):
    U = scale * np.random.rand(block_dim, n, n)
    U = U - np.tril(U)
    return torch.from_numpy(U)


def _test_triu_inv_rec_unroll(U: torch.tensor, atol, rtol):

    n = U.shape[-1]
    assert U.stride() == (n * n, n, 1)
    golden_numpy = np.linalg.inv(np.identity(n) + U.numpy().astype(np.float64))
    assert golden_numpy.shape == U.numpy().shape
    U_npu = U.to(torch.half)
    U_npu = U_npu.npu()
    torch.npu.synchronize()
    actual = tcuscan_ops.run_triu_inv_rec_unroll(U_npu[0:1, :, :])
    torch.npu.synchronize()
    actual_cpu = actual.cpu().to(torch.float64)
    golden_cpu = torch.from_numpy(golden_numpy)
    print(f"max abs error: {torch.max(torch.abs(golden_cpu - actual_cpu))}")
    assert np.allclose(
        actual_cpu.numpy(), golden_cpu.numpy(), atol=atol, rtol=rtol
    ), f"Error at matrix size: {n}."


@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("block_dim", [2, 4, 6, 20])
def test_triu_inv_rec_unroll_random(n: int, block_dim: int):
    atol = 1e-1
    rtol = 1
    U = random_matrix(n, block_dim)
    _test_triu_inv_rec_unroll(U, atol, rtol)


@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("block_dim", [2, 4, 6, 20])
def test_triu_inv_rec_unroll_ones(n: int, block_dim: int):
    atol = 0
    rtol = 0
    U = ones_matrix(n, block_dim)
    _test_triu_inv_rec_unroll(U, atol, rtol)


@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("block_dim", [2, 4, 6, 20])
def test_triu_inv_rec_unroll_block_ones(n: int, block_dim: int):
    atol = 0
    rtol = 0
    U = block_ones_matrix(n, block_dim)
    _test_triu_inv_rec_unroll(U, atol, rtol)


@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("block_dim", [2, 4, 6, 20])
def test_triu_inv_rec_unroll_big_bad_matrix(n: int, block_dim: int):
    atol = 1e-7
    rtol = 1e-4
    U = big_bad_matrix(n, block_dim)
    _test_triu_inv_rec_unroll(U, atol, rtol)


@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("block_dim", [2, 4, 6, 20])
def test_triu_inv_rec_unroll_zeros(n: int, block_dim: int):
    atol = 0
    rtol = 0
    U = zeros_matrix(n, block_dim)
    _test_triu_inv_rec_unroll(U, atol, rtol)
