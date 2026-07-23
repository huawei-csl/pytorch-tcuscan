# --------------------------------------------------------------------------------
# Copyright (c) 2023-2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pytorch-tcuscan/
# for the full License text.
# --------------------------------------------------------------------------------
"""Standalone runner for the SpMV v2 kernel.

Unlike ``tests/test_spmv_v2.py`` (which goes through the ``tcuscan_ops`` pybind
extension), this script loads the compiled kernel directly from
``libkernel_spmv_v2_<arch>.so`` with ``ctypes`` and launches it by hand --
mirroring the style of ``run_radix_sort.py``.

Build the shared object first, e.g.::

    make compile_spmv_v2        # -> build/lib/libkernel_spmv_v2_a2.so
    make compile_a5_spmv_v2     # -> build/lib/libkernel_spmv_v2_a5.so

Then run it::

    python run_spmv_v2.py                 # loads the a2 build
    TCUSCAN_ARCH=a5 python run_spmv_v2.py  # loads the a5 build

``LIBKERNEL_SPMV_V2`` still overrides the path outright.
"""
import ctypes
import os
from functools import partial

import numpy as np
import torch_npu  # noqa: F401
from scipy.sparse import random as sparse_random

import torch

# Select device, e.g. "npu:0" or "npu:1".
DEVICE = os.environ.get("NPU_DEVICE", "npu:0")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(DEVICE)

np.random.seed(42)
torch.manual_seed(42)

# Number of AI (cube) cores; used to pick the block dimension, matching
# ascendc_platform->GetCoreNumAic() in src/torch/torch_spmv.h.
NUM_AI_CORES = int(getattr(torch.npu.get_device_properties("npu"), "cube_core_num", 20))


def ceil_div(a: int, b: int) -> int:
    """Reproduces host_utils::CeilDiv."""
    return (a + b - 1) // b


def align_up(value: int, alignment: int) -> int:
    """Reproduces host_utils::AlignUp."""
    return ceil_div(value, alignment) * alignment


def compute_block_dim(nnz: int, tile_len: int) -> int:
    """Reproduces the block-dim heuristic from run_spmv_v2 in torch_spmv.h."""
    align_size = tile_len * tile_len
    num_tiles = ceil_div(nnz, align_size)
    block_dim = NUM_AI_CORES
    if num_tiles < block_dim:
        block_dim = num_tiles
    return block_dim


def compute_block_len(nnz: int, tile_len: int, block_dim: int) -> int:
    """Reproduces block_len from run_spmv_v2 in torch_spmv.h."""
    align_size = tile_len * tile_len
    num_tiles = ceil_div(nnz, align_size)
    max_num_tiles_per_block = ceil_div(num_tiles, block_dim)
    return max_num_tiles_per_block * align_size


def make_tiling(
    nnz: int, num_segments: int, x_len: int, tile_len: int, block_len: int
) -> torch.Tensor:
    """Builds the SpMVTiling struct (5 x uint32) on device.

    Layout (see src/tiling/tiling_spmv.h):
        {nnz, num_segments, x_len, tile_len, block_len}
    """
    return torch.tensor(
        [nnz, num_segments, x_len, tile_len, block_len],
        dtype=torch.int32,
        device=DEVICE,
    )


def compute_segment_offsets(
    indptr: torch.Tensor, block_dim: int, block_len: int, nnz: int
) -> torch.Tensor:
    """Reproduces the searchsorted segment-offset-per-block from
    run_seg_sum_multi_core in torch_seg_ops.h.

    ``indptr`` is the full CSR row-pointer array (with the leading zero).
    Returns an int32 tensor of length ``block_dim + 1``.
    """
    sstart = torch.clamp(
        torch.arange(0, block_dim + 1, dtype=torch.int32, device=DEVICE) * block_len,
        max=nnz,
    )
    return torch.searchsorted(indptr.to(torch.int32), sstart, out_int32=True)


def torch_to_ctypes(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def run_spmv_v2(
    lib,
    values: torch.Tensor,
    indptr: torch.Tensor,
    cols: torch.Tensor,
    vector: torch.Tensor,
    s: int,
) -> torch.Tensor:
    """Launch the SpMV v2 kernel: computes A @ x for A in CSR form.

    ``indptr`` is the full CSR row-pointer array (with the leading zero).
    Returns the dense result vector (float32).
    """
    nnz = values.numel()
    x_len = vector.numel()
    num_segments = indptr.numel() - 1
    tile_len = s

    if values.dtype == torch.float16:
        kernel = lib.launch_spmv_v2_fp16
        input_elem_size = 2  # sizeof(int16_t)
    elif values.dtype == torch.float32:
        kernel = lib.launch_spmv_v2_fp32
        input_elem_size = 4  # sizeof(float)
    else:
        raise ValueError(
            f"Unsupported dtype {values.dtype}; expected float16 or float32."
        )

    # Host launcher ABI (see launch_spmv_v2_* in src/spmv_v2.cpp):
    #   void launch_spmv_v2_*(uint32_t block_dim, void* stream,
    #                         GM_ADDR vec_in, GM_ADDR cols_in, GM_ADDR indptr,
    #                         GM_ADDR x_in, GM_ADDR segment_offsets,
    #                         GM_ADDR vec_out, GM_ADDR workspace,
    #                         GM_ADDR tiling)
    kernel.restype = None
    kernel.argtypes = [
        ctypes.c_uint32,  # block_dim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # vec_in (values)
        ctypes.c_void_p,  # cols_in
        ctypes.c_void_p,  # indptr
        ctypes.c_void_p,  # x_in (vector)
        ctypes.c_void_p,  # segment_offsets
        ctypes.c_void_p,  # vec_out
        ctypes.c_void_p,  # workspace
        ctypes.c_void_p,  # tiling
    ]

    block_dim = compute_block_dim(nnz, tile_len)
    block_len = compute_block_len(nnz, tile_len, block_dim)
    align_size = tile_len * tile_len
    padded_nnz = align_up(nnz, align_size)

    z = torch.zeros(num_segments, dtype=torch.float32, device=DEVICE)

    tiling = make_tiling(nnz, num_segments, x_len, tile_len, block_len)
    segment_offsets = compute_segment_offsets(indptr, block_dim, block_len, nnz)

    # workspace: padded_nnz * sizeof(input_dtype) for CSR products
    #          + padded_nnz * sizeof(float) for cube scan output.
    # (system LibApi reserve is appended in the pybind path via
    #  GetLibApiWorkSpaceSize(); 16 MiB is the standard reserve on A2/A5.)
    workspace_size = padded_nnz * (input_elem_size + 4) + 16 * 1024 * 1024
    workspace = torch.zeros(workspace_size, dtype=torch.uint8, device=DEVICE)

    # Offset indptr by one element: the kernel expects the row-pointer array
    # without the leading zero (see run_spmv_v2 in torch_spmv.h).
    indptr_kernel = indptr.to(torch.int32)[1:]

    stream_ptr = torch.npu.current_stream()._as_parameter_  # noqa: SLF001

    kernel(
        block_dim,
        stream_ptr,
        torch_to_ctypes(values),
        torch_to_ctypes(cols),
        torch_to_ctypes(indptr_kernel),
        torch_to_ctypes(vector),
        torch_to_ctypes(segment_offsets),
        torch_to_ctypes(z),
        torch_to_ctypes(workspace),
        torch_to_ctypes(tiling),
    )
    # The launch is asynchronous; block here so the scratch buffers local to
    # this function are not reclaimed by the caching allocator while the kernel
    # is still reading them.
    torch.npu.synchronize()
    return z


def uniform_rvs(shape, dtype: np.dtype, scale: int = 6):
    if np.issubsctype(dtype, np.integer):
        return np.random.randint(-scale, scale, size=shape)
    return scale * np.random.uniform(0, 1, size=shape) - (scale // 2)


def generate_csr(nrow: int, density: float, dtype: torch.dtype, scale_factor: int):
    """Builds a random CSR matrix and dense vector (mirrors test_spmv_v2.py)."""
    rng = np.random.default_rng(seed=42)
    out_dtype = np.float32

    B = sparse_random(
        nrow - 1,
        nrow - 1,
        density=density,
        format="csr",
        dtype=out_dtype,
        data_rvs=partial(uniform_rvs, dtype=out_dtype, scale=scale_factor),
    )

    values = B.data.astype(out_dtype)
    indexes = B.indptr.astype(np.uint32)
    cols = B.indices.astype(np.uint32)
    vector = rng.uniform(1, 9, nrow - 1).astype(out_dtype)

    expected = torch.from_numpy(B @ vector)

    torch_values = torch.from_numpy(values).to(dtype).npu()
    torch_indexes = torch.from_numpy(indexes).to(torch.int32).npu()
    torch_cols = torch.from_numpy(cols).to(torch.int32).npu()
    torch_vector = torch.from_numpy(vector).to(dtype).npu()

    return torch_values, torch_indexes, torch_cols, torch_vector, expected


if __name__ == "__main__":
    arch = os.environ.get("TCUSCAN_ARCH", "a2")
    lib_path = os.environ.get(
        "LIBKERNEL_SPMV_V2",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "build",
            "lib",
            f"libkernel_spmv_v2_{arch}.so",
        ),
    )
    lib_path = os.path.abspath(lib_path)

    try:
        lib = ctypes.CDLL(lib_path)
        print(f"Loaded library from {lib_path}")

        # Tile size, matrix rows, density and dtype.
        s = 128
        nrow = 1024 * 4
        density = 0.01
        dtype = torch.float16

        values, indptr, cols, vector, expected = generate_csr(
            nrow, density, dtype, scale_factor=2
        )

        torch.npu.synchronize()
        actual = run_spmv_v2(lib, values, indptr, cols, vector, s)
        torch.npu.synchronize()

        actual_cpu = actual.cpu()

        print(f"Expected (first 10): {expected[:10]}")
        print(f"Actual   (first 10): {actual_cpu[:10]}")

        assert (
            actual_cpu.shape == expected.shape
        ), f"Shape mismatch: {actual_cpu.shape} vs {expected.shape}"
        is_close = torch.allclose(actual_cpu, expected, atol=1e-01)
        print(
            f"nrow={nrow}, nnz={values.numel()}, s={s}, "
            f"density={density}, dtype={dtype}"
        )
        print(f"Is SpMV output correct? {is_close}")
    finally:
        del lib  # triggers dlclose in CPython
