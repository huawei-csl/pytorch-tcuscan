import os
import random

import numpy as np
import pytest

import torch
import torch_npu  # noqa
import tcuscan_ops

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:0")
torch.npu.set_device(NPU_DEVICE)


def arithmetic_prog_matrix(n, block_dim):
    U = np.zeros((block_dim, n, n))
    for i in range(n):
        for k in range(block_dim):
            U[k, i, i:] = range(1, n - i + 1)
    U = np.linalg.inv(U)
    return torch.from_numpy(np.triu(U, k=1))


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


def block_random_matrix(n, block_dim, scale=0.1):
    U_ = scale * np.random.rand(16, 16)
    U_ = np.triu(U_, k=1)
    U = np.zeros((block_dim, n, n))
    for k in range(block_dim):
        for i in range(0, n, 16):
            U[k, i : i + 16, i : i + 16] = U_.copy()
    return torch.from_numpy(U)


def zeros_matrix(n, block_dim):
    return torch.zeros((block_dim, n, n))


def ones_matrix(n, block_dim):
    U = np.ones((block_dim, n, n))
    return torch.from_numpy(np.triu(U, k=1))


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


def _test_triu_inv_rec_unroll(U: torch.tensor, atol: float, rtol: float, ftol: float):

    n = U.shape[-1]
    U = U.to(torch.half)
    U_npu = U.npu()
    torch.npu.synchronize()

    assert U.stride() == (n * n, n, 1)
    Identity = np.ones((U.shape[0], n, n))
    Identity = np.triu(Identity)
    Identity = np.tril(Identity)
    U_plus_I = Identity.astype(np.float64) + U.numpy().astype(np.float64)
    golden_numpy = np.zeros((U_plus_I.shape))
    for k in range(U_plus_I.shape[0]):
        golden_numpy[k] = np.linalg.inv(U_plus_I[k])
    golden_cpu = torch.from_numpy(golden_numpy)

    torch.npu.synchronize()
    actual = tcuscan_ops.run_triu_inv_rec_unroll(U_npu)
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    torch.npu.synchronize()
    actual_cpu = actual_cpu.to(torch.float64)
    frob_error = torch.sqrt(
        torch.sum((golden_cpu - actual_cpu) * (golden_cpu - actual_cpu))
        / torch.sum(golden_cpu * golden_cpu)
    )
    # print(f"max_error:  {torch.max(torch.abs(golden_cpu - actual_cpu)):.3e}")
    # print(f"frob_error: {frob_error:.3e}")
    actual_numpy = actual_cpu.numpy()
    golden_numpy = golden_cpu.numpy()
    assert np.allclose(
        actual_numpy, golden_numpy, atol=atol, rtol=rtol
    ), f"Error at allclose - tensor shape: {U.shape} - rtol: {rtol}."
    assert frob_error <= ftol, f"frob_error: {frob_error}"


@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("block_dim", [1, 2, 4, 6, 20, 32, 64, 128, 256])
def test_triu_inv_rec_unroll_random(n: int, block_dim: int):
    atol = 3e-5
    rtol = min([0.5, 1e-2 * (n + block_dim)])
    ftol = 1e-4
    U = random_matrix(n, block_dim)
    _test_triu_inv_rec_unroll(U, atol, rtol, ftol)


@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("block_dim", [1, 2, 4, 6, 20, 32, 64, 128, 256])
def test_triu_inv_rec_unroll_block_random(n: int, block_dim: int):
    atol = 1e-5
    rtol = min([0.5, 1e-2 * (n + block_dim)])
    ftol = 1e-4
    U = block_random_matrix(n, block_dim)
    _test_triu_inv_rec_unroll(U, atol, rtol, ftol)


@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("block_dim", [1, 2, 4, 6, 20, 32, 64, 128, 256])
def test_triu_inv_rec_unroll_ones(n: int, block_dim: int):
    atol = 0
    rtol = 0
    ftol = 0
    U = ones_matrix(n, block_dim)
    _test_triu_inv_rec_unroll(U, atol, rtol, ftol)


@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("block_dim", [1, 2, 4, 6, 20, 32, 64, 128, 256])
def test_triu_inv_rec_unroll_block_ones(n: int, block_dim: int):
    atol = 0
    rtol = 0
    ftol = 0
    U = block_ones_matrix(n, block_dim)
    _test_triu_inv_rec_unroll(U, atol, rtol, ftol)


@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("block_dim", [1, 2, 4, 6, 20, 32, 64, 128, 256])
def test_triu_inv_rec_unroll_big_bad_matrix(n: int, block_dim: int):
    atol = 1e-7
    rtol = 1e-4
    ftol = 1e-7
    U = big_bad_matrix(n, block_dim)
    _test_triu_inv_rec_unroll(U, atol, rtol, ftol)


@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("block_dim", [1, 2, 4, 6, 20, 32, 64, 128, 256])
def test_triu_inv_rec_unroll_zeros(n: int, block_dim: int):
    atol = 0
    rtol = 0
    ftol = 0
    U = zeros_matrix(n, block_dim)
    _test_triu_inv_rec_unroll(U, atol, rtol, ftol)


@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("block_dim", [1, 2, 4, 6, 20, 32, 64, 128, 256])
def test_triu_inv_rec_unroll_ar_prog(n: int, block_dim: int):
    atol = 0
    rtol = 0
    ftol = 0
    U = arithmetic_prog_matrix(n, block_dim)
    _test_triu_inv_rec_unroll(U, atol, rtol, ftol)
