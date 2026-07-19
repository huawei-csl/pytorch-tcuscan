# --------------------------------------------------------------------------------
# Copyright (c) 2023-2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pytorch-tcuscan/
# for the full License text.
# --------------------------------------------------------------------------------
"""Standalone runner for the radix-sort kernel.

Unlike ``tests/test_radix_sort.py`` (which goes through the ``tcuscan_ops``
pybind extension), this script loads the compiled kernel directly from
``libkernel_radix_sort_<arch>.so`` with ``ctypes`` and launches it by hand --
mirroring the style of ``run_abs.py`` in the ``pto-kernels`` repository.

Build the shared object first, e.g.::

    make compile_radix_sort        # -> build/lib/libkernel_radix_sort_a2.so
    make compile_a5_radix_sort     # -> build/lib/libkernel_radix_sort_a5.so

Then run it::

    python run_radix_sort.py                 # loads the a2 build
    TCUSCAN_ARCH=a5 python run_radix_sort.py  # loads the a5 build

``LIBKERNEL_RADIX_SORT`` still overrides the path outright.
"""
import ctypes
import os

import torch_npu  # noqa: F401

import torch

# Select device, e.g. "npu:0" or "npu:1".
DEVICE = os.environ.get("NPU_DEVICE", "npu:0")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(DEVICE)

torch.manual_seed(42)

# Number of AI (cube) cores; used to pick the block dimension, matching
# ascendc_platform->GetCoreNumAic() in src/torch/torch_sort.h.
NUM_AI_CORES = int(getattr(torch.npu.get_device_properties("npu"), "cube_core_num", 20))

# System (LibApi) reserved workspace appended to the user workspace, matching
# GetLibApiWorkSpaceSize() in src/torch/commons.h::alloc_workspace. 16 MiB is the
# standard reserve on Atlas A2 / 910B; over-allocating is harmless.
SYSTEM_WORKSPACE_SIZE = 16 * 1024 * 1024

# GM_ALIGNMENT from src/host_utils.h.
GM_ALIGNMENT = 256


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def compute_block_dim(vec_len: int, matmul_size: int) -> int:
    """Reproduces the block-dim heuristic from src/torch/torch_sort.h."""
    tile_elems = matmul_size * matmul_size
    num_tiles = vec_len // tile_elems
    block_dim = NUM_AI_CORES
    while num_tiles % block_dim != 0:
        block_dim -= 1
    return max(block_dim, 1)


def user_workspace_size(vec_len: int, num_blocks: int) -> int:
    """Reproduces get_workspace_size<int16_t>() from src/torch/workspace.h.

    InputT is int16_t (2 bytes) for both the fp16 and int16 kernels.
    """
    tmp_output_size = vec_len * 2  # vec_len * sizeof(int16_t)
    radices_size = align_up(vec_len * 1, 4)  # AlignUp(vec_len, sizeof(int32_t))
    indices_size = vec_len * 4  # vec_len * sizeof(int32_t)
    split_ws_size = align_up(num_blocks * 4, GM_ALIGNMENT)
    return tmp_output_size + radices_size + indices_size + split_ws_size


def make_tiling(vec_len: int, matmul_size: int, num_blocks: int) -> torch.Tensor:
    """Builds the RadixSortTiling struct (4 x uint32) on device.

    Layout (see src/tiling/tiling_radix_sort.h):
        {num_blocks, vec_len, matmul_size, vec_tile_size}
    """
    vec_tile_size = (matmul_size * matmul_size) // 2
    return torch.tensor(
        [num_blocks, vec_len, matmul_size, vec_tile_size],
        dtype=torch.int32,
        device=DEVICE,
    )


def torch_to_ctypes(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def generate_input(dtype, vec_len):
    if dtype == torch.int16:
        info = torch.iinfo(dtype)
        return torch.randint(info.min, info.max, (vec_len,), dtype=dtype, device=DEVICE)
    return torch.rand((vec_len,), dtype=dtype, device=DEVICE)


def run_radix_sort(lib, x: torch.Tensor, s: int):
    """Launch the radix-sort kernel on ``x`` with tile size ``s``.

    Returns (sorted_values, sorted_indices).
    """
    vec_len = x.numel()
    matmul_size = s
    block_dim = compute_block_dim(vec_len, matmul_size)

    if x.dtype == torch.float16:
        kernel = lib.launch_radix_sort_fp16
    elif x.dtype == torch.int16:
        kernel = lib.launch_radix_sort_int16
    else:
        raise ValueError(f"Unsupported dtype {x.dtype}; expected float16 or int16.")

    # Host launcher ABI (see launch_radix_sort_* in src/radix_sort.cpp):
    #   void launch_radix_sort_*(uint32_t block_dim, void* stream,
    #                            GM_ADDR in, GM_ADDR out, GM_ADDR indices,
    #                            GM_ADDR workspace, GM_ADDR tiling)
    kernel.restype = None
    kernel.argtypes = [
        ctypes.c_uint32,  # block_dim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # in
        ctypes.c_void_p,  # out
        ctypes.c_void_p,  # indices
        ctypes.c_void_p,  # workspace
        ctypes.c_void_p,  # tiling
    ]

    values_out = torch.empty(vec_len, dtype=x.dtype, device=DEVICE)
    indices_out = torch.empty(vec_len, dtype=torch.int32, device=DEVICE)

    tiling = make_tiling(vec_len, matmul_size, block_dim)
    workspace = torch.zeros(
        user_workspace_size(vec_len, block_dim) + SYSTEM_WORKSPACE_SIZE,
        dtype=torch.uint8,
        device=DEVICE,
    )

    stream_ptr = torch.npu.current_stream()._as_parameter_  # noqa: SLF001

    kernel(
        block_dim,
        stream_ptr,
        torch_to_ctypes(x),
        torch_to_ctypes(values_out),
        torch_to_ctypes(indices_out),
        torch_to_ctypes(workspace),
        torch_to_ctypes(tiling),
    )
    # The launch is asynchronous; block here so the `workspace` and `tiling`
    # scratch buffers (local to this function) are not reclaimed by the caching
    # allocator while the kernel is still reading them.
    torch.npu.synchronize()
    return values_out, indices_out


if __name__ == "__main__":
    arch = os.environ.get("TCUSCAN_ARCH", "a2")
    lib_path = os.environ.get(
        "LIBKERNEL_RADIX_SORT",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "build",
            "lib",
            f"libkernel_radix_sort_{arch}.so",
        ),
    )
    lib_path = os.path.abspath(lib_path)

    try:
        lib = ctypes.CDLL(lib_path)
        print(f"Loaded library from {lib_path}")

        # Tile size (must divide the input and satisfy vec_len >= s * s).
        s = 128
        dtype = torch.float16
        vec_len = 65536

        x = generate_input(dtype, vec_len)

        torch.npu.synchronize()
        expected, expected_indices = torch.sort(x, dim=-1, descending=False)
        torch.npu.synchronize()
        actual, actual_indices = run_radix_sort(lib, x, s)
        torch.npu.synchronize()

        print(f"Input (first 10):    {x[:10]}")
        print(f"Expected (first 10): {expected[:10]}")
        print(f"Actual (first 10):   {actual[:10]}")

        is_sorted = torch.equal(expected, actual)
        print(f"vec_len={vec_len}, s={s}, dtype={dtype}")
        print(f"Is output sorted correctly? {is_sorted}")
    finally:
        del lib  # triggers dlclose in CPython
