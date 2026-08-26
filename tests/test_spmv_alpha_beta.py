# --------------------------------------------------------------------------------
# Copyright (c) 2023-2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pytorch-tcuscan/
# for the full License text.
# --------------------------------------------------------------------------------
"""BLAS-style scaling for the SpMV entry points: ``y = alpha * A @ x + beta * y``."""

import os

import numpy as np
import pytest
import torch_npu  # noqa
from scipy.sparse import random

import tcuscan_ops
import torch

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)

_NROW = [1024 * 1, 1024 * 4, 1024 * 9]
_ALPHA_BETA = [
    (1.0, 0.0),  # default: plain A @ x
    (2.5, 0.0),  # alpha only
    (1.0, 1.0),  # accumulate onto y
    (-0.5, 2.0),  # both, negative alpha
    (0.0, 3.0),  # alpha == 0: pure scaling of y
]


def _csr_problem(nrow: int, density: float, dtype: torch.dtype, seed: int = 42):
    """Builds a random CSR problem and its NPU tensors."""
    rng = np.random.default_rng(seed=seed)
    B = random(
        nrow - 1,
        nrow - 1,
        density=density,
        format="csr",
        dtype=np.float32,
        data_rvs=lambda n: 2 * rng.uniform(0, 1, size=n) - 1,
        random_state=seed,
    )
    vector = rng.uniform(1, 9, nrow - 1).astype(np.float32)
    y0 = rng.uniform(-3, 3, B.shape[0]).astype(np.float32)

    np_dtype = np.float16 if dtype == torch.float16 else np.float32
    tensors = {
        "vals": torch.from_numpy(B.data.astype(np_dtype)).npu(),
        "indptr": torch.from_numpy(B.indptr.astype(np.int32)).npu(),
        "cols": torch.from_numpy(B.indices.astype(np.int32)).npu(),
        "x": torch.from_numpy(vector.astype(np_dtype)).npu(),
    }
    return B, vector, y0, tensors


def _expected(B, vector, y0, alpha: float, beta: float):
    return torch.from_numpy(alpha * (B @ vector) + beta * y0)


def _assert_close(actual, expected, label: str, atol: float):
    actual_cpu = actual.cpu()
    assert actual.dtype == torch.float32
    assert (
        actual.shape == expected.shape
    ), f"{label}: shape mismatch, got {actual.shape} expected {expected.shape}"
    assert torch.allclose(
        actual_cpu, expected, atol=atol
    ), f"{label}: max |diff| = {(actual_cpu - expected).abs().max().item()}"


def _scan_matrices(s: int):
    ones = torch.ones((s, s), dtype=torch.float16, device=NPU_DEVICE)
    return torch.triu(ones), torch.tril(ones, -1)


@pytest.mark.parametrize(("alpha", "beta"), _ALPHA_BETA)
@pytest.mark.parametrize("s", [64, 128])
@pytest.mark.parametrize("nrow", _NROW)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_spmv_v2_alpha_beta(alpha: float, beta: float, s: int, nrow: int, dtype):
    B, vector, y0, t = _csr_problem(nrow, 0.001, dtype)
    y = torch.from_numpy(y0).npu()

    torch.npu.synchronize()
    actual = tcuscan_ops.run_spmv_v2(
        t["vals"], t["indptr"], t["cols"], t["x"], s, alpha=alpha, beta=beta, y=y
    )
    torch.npu.synchronize()

    _assert_close(actual, _expected(B, vector, y0, alpha, beta), "run_spmv_v2", 1e-1)
    # `y` is updated in place and returned.
    assert actual.data_ptr() == y.data_ptr()


@pytest.mark.parametrize("s", [128])
@pytest.mark.parametrize("nrow", _NROW)
def test_spmv_v2_beta_zero_ignores_garbage(s: int, nrow: int):
    """BLAS semantics: ``beta == 0`` must overwrite ``y`` without reading it."""
    B, vector, _, t = _csr_problem(nrow, 0.001, torch.float32)
    y = torch.full((B.shape[0],), float("nan"), dtype=torch.float32).npu()

    torch.npu.synchronize()
    actual = tcuscan_ops.run_spmv_v2(
        t["vals"], t["indptr"], t["cols"], t["x"], s, alpha=1.0, beta=0.0, y=y
    )
    torch.npu.synchronize()

    _assert_close(actual, torch.from_numpy(B @ vector), "run_spmv_v2 beta=0", 1e-1)


@pytest.mark.parametrize("s", [128])
@pytest.mark.parametrize("nrow", _NROW)
def test_spmv_v2_no_y_matches_default(s: int, nrow: int):
    """Omitting ``y`` returns ``alpha * A @ x``, whatever ``beta`` says."""
    _, vector, _, t = _csr_problem(nrow, 0.001, torch.float32)

    torch.npu.synchronize()
    actual = tcuscan_ops.run_spmv_v2(
        t["vals"], t["indptr"], t["cols"], t["x"], s, alpha=3.0, beta=7.0
    )
    baseline = tcuscan_ops.run_spmv_v2(t["vals"], t["indptr"], t["cols"], t["x"], s)
    torch.npu.synchronize()

    _assert_close(actual, (3.0 * baseline).cpu(), "run_spmv_v2 no-y", 1e-1)


@pytest.mark.parametrize(("alpha", "beta"), _ALPHA_BETA)
@pytest.mark.parametrize("s", [64, 128])
@pytest.mark.parametrize("nrow", _NROW)
def test_spmv_v2_multi_cube_alpha_beta(alpha: float, beta: float, s: int, nrow: int):
    B, vector, y0, t = _csr_problem(nrow, 0.001, torch.float16)
    upper, lower_strict = _scan_matrices(s)
    y = torch.from_numpy(y0).npu()

    torch.npu.synchronize()
    actual = tcuscan_ops.run_spmv_v2_multi_cube(
        t["vals"],
        t["indptr"],
        t["cols"],
        t["x"],
        upper,
        lower_strict,
        alpha=alpha,
        beta=beta,
        y=y,
    )
    torch.npu.synchronize()

    _assert_close(
        actual, _expected(B, vector, y0, alpha, beta), "run_spmv_v2_multi_cube", 1e0
    )
    assert actual.data_ptr() == y.data_ptr()


@pytest.mark.parametrize(("alpha", "beta"), _ALPHA_BETA)
@pytest.mark.parametrize("s", [64, 128])
@pytest.mark.parametrize("nrow", _NROW)
def test_spmv_alpha_beta(alpha: float, beta: float, s: int, nrow: int):
    B, vector, y0, t = _csr_problem(nrow, 0.001, torch.float16)
    y = torch.from_numpy(y0).npu()

    torch.npu.synchronize()
    actual = tcuscan_ops.run_spmv(
        t["vals"], t["indptr"], t["cols"], t["x"], s, alpha=alpha, beta=beta, y=y
    )
    torch.npu.synchronize()

    _assert_close(actual, _expected(B, vector, y0, alpha, beta), "run_spmv", 1e0)
    assert actual.data_ptr() == y.data_ptr()


@pytest.mark.parametrize(("alpha", "beta"), _ALPHA_BETA)
@pytest.mark.parametrize("s", [64, 128])
@pytest.mark.parametrize("nrow", _NROW)
def test_spmv_multi_cube_alpha_beta(alpha: float, beta: float, s: int, nrow: int):
    B, vector, y0, t = _csr_problem(nrow, 0.001, torch.float16)
    upper, lower_strict = _scan_matrices(s)
    y = torch.from_numpy(y0).npu()

    torch.npu.synchronize()
    actual = tcuscan_ops.run_spmv_multi_cube(
        t["vals"],
        t["indptr"],
        t["cols"],
        t["x"],
        upper,
        lower_strict,
        alpha=alpha,
        beta=beta,
        y=y,
    )
    torch.npu.synchronize()

    _assert_close(
        actual, _expected(B, vector, y0, alpha, beta), "run_spmv_multi_cube", 1e0
    )
    assert actual.data_ptr() == y.data_ptr()
