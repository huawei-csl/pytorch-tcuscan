/**
 * @file torch_linalg.h
 * @brief Torch wrapper for linear algebra operators.
 * @date 2025-12-03
 *
 * @copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../host_utils.h"
#include "../tiling/tiling_tri_inv_col_sweep.h"
#include "../tiling/tiling_tri_inv_cube_col_sweep.h"
#include "../tiling/tiling_triu_inv_rec_unroll.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "workspace.h"

// Kernel launchers; documented at their definitions in src/*.cpp
/// @cond
extern "C" void launch_tri_inv_col_sweep_fp16(uint32_t blockDim, void* stream,
                                              void* vec_in, void* vec_out,
                                              void* workspace, void* tiling_gm);
extern "C" void launch_tri_inv_col_sweep_fp32(uint32_t blockDim, void* stream,
                                              void* vec_in, void* vec_out,
                                              void* workspace, void* tiling_gm);
extern "C" void launch_tri_inv_cube_col_sweep_fp16(uint32_t blockDim,
                                                   void* stream, void* vec_in,
                                                   void* vec_out,
                                                   void* workspace,
                                                   void* tiling_gm);
extern "C" void launch_triu_inv_rec_unroll_fp16(uint32_t blockDim, void* stream,
                                                void* vec_in, void* vec_out,
                                                void* workspace, void* tiling);
/// @endcond

namespace tcuscan {

/**
 * @brief Torch wrapper for vector `tri_inv_col_sweep` kernel.
 *
 * Each output matrix is produced in `num_out_tiles` column tiles, one per
 * vector core. The parameter therefore trades parallelism against redundant
 * work: every core sweeps the whole matrix but writes back only its own
 * `n / num_out_tiles` columns, and the kernel is launched with
 * `batch_dim * num_out_tiles` blocks.
 *
 * @param [in] x Input Tensor of (batch_dim, n, n ). Matrices are assumed to be
 * unit upper triangular matrices.
 * @param [in] num_out_tiles Number of column tiles each output matrix is split
 * into. Must divide `n`. Pass `0` to keep the default of 16 columns per tile,
 * i.e. `n / 16`.
 * @return Returns Matrix inverse for each matrix over the `batch_dim`.
 */
at::Tensor run_tri_inv_col_sweep(const at::Tensor& x,
                                 int64_t num_out_tiles = 0) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  if (x.dim() < 2) {
    throw std::runtime_error("Input tensor must have at least 2 dimensions.\n");
  }

  const uint32_t matrix_size = static_cast<uint32_t>(x.size(-1));
  if (matrix_size != x.size(-2)) {
    throw std::runtime_error("Only square matrices are supported.\n");
  }

  const uint32_t num_elems = static_cast<uint32_t>(x.numel());
  const uint32_t num_matrices = num_elems / (matrix_size * matrix_size);

  if (num_out_tiles < 0) {
    throw std::runtime_error("num_out_tiles must not be negative.\n");
  }
  // `0` keeps the default tiling of 16 columns per output tile.
  const uint32_t out_tiles = num_out_tiles == 0
                                 ? matrix_size / 16
                                 : static_cast<uint32_t>(num_out_tiles);
  if (out_tiles == 0 || matrix_size % out_tiles != 0) {
    throw std::runtime_error(
        "num_out_tiles must be a positive divisor of the matrix size.\n");
  }

  const uint32_t block_dim = num_matrices * out_tiles;

  const at::Tensor z = at::empty_like(x);
  const at::Tensor workspace_tensor = tcuscan::alloc_workspace(0, device);

  const TriInvColumnSweepTiling tiling{block_dim, num_elems, matrix_size,
                                       out_tiles};
  const at::Tensor tiling_device =
      tcuscan::alloc_copy_tiling_tensor(tiling, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
  if (dtype == torch::kHalf) {
    launch_tri_inv_col_sweep_fp16(
        block_dim, acl_stream, const_cast<void*>(x.storage().data()),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()),
        tiling_device.data_ptr());
  } else {
    launch_tri_inv_col_sweep_fp32(
        block_dim, acl_stream, const_cast<void*>(x.storage().data()),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()),
        tiling_device.data_ptr());
  }

  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Torch wrapper for vector `tri_inv_cube_col_sweep` kernel.
 * @param [in] x Input Tensor of (batch_dim, n, n ). Matrices are assumed to be
 * unit upper triangular matrices.
 * @return Returns Matrix inverse for each matrix over the `batch_dim`.
 */
at::Tensor run_tri_inv_cube_col_sweep(const at::Tensor& x) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out = torch::kFloat32;
  if (x.dim() < 2)
    throw std::runtime_error("Input tensor must have at least 2 dimensions.\n");

  const uint32_t matrix_size = static_cast<uint32_t>(x.size(-1));
  if (matrix_size != x.size(-2))
    throw std::runtime_error("Only square matrices are supported.\n");

  const uint32_t num_elems = static_cast<uint32_t>(x.numel());
  const uint32_t block_dim =
      static_cast<uint32_t>(num_elems / (matrix_size * matrix_size));

  const at::Tensor z =
      at::empty({block_dim, matrix_size, matrix_size},
                at::TensorOptions().dtype(dtype_out).device(device));

  constexpr uint32_t WS_CIRCULAR_BUFFER_LEN = 4;
  const TriInvCubeColSweepTiling tiling{block_dim, matrix_size,
                                        WS_CIRCULAR_BUFFER_LEN};
  const at::Tensor tiling_device =
      tcuscan::alloc_copy_tiling_tensor(tiling, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == torch::kHalf) {
    const size_t workspace_size =
        tcuscan::get_workspace_size<uint16_t /* half */>(tiling);
    const at::Tensor workspace_tensor =
        tcuscan::alloc_workspace(workspace_size, device);

    launch_tri_inv_cube_col_sweep_fp16(
        block_dim, acl_stream, const_cast<void*>(x.storage().data()),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()),
        tiling_device.data_ptr());
  } else {
    throw std::runtime_error("Currently only half precision is supported.\n");
  }

  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief
 *
 * @param x: tensor containing strictly upper triangular matrices of the same
 * size
 *
 * @return Tensor containing the inverses of the triangular matrices plus
 * identity
 *
 */
at::Tensor run_triu_inv_rec_unroll(const at::Tensor& x) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out = torch::kFloat32;
  if (x.dim() < 2)
    throw std::runtime_error("Input tensor must have at least 2 dimensions.\n");

  const uint32_t matrix_size = static_cast<uint32_t>(x.size(-1));
  if (matrix_size != x.size(-2))
    throw std::runtime_error("Only square matrices are supported.\n");

  const uint32_t num_elems = static_cast<uint32_t>(x.numel());
  const uint32_t block_dim =
      static_cast<uint32_t>(num_elems / (matrix_size * matrix_size));

  const at::Tensor z =
      at::empty({block_dim, matrix_size, matrix_size},
                at::TensorOptions().dtype(dtype_out).device(device));

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  const TriuInvRecUnrollTiling tiling{block_dim, matrix_size};
  const at::Tensor tiling_device =
      tcuscan::alloc_copy_tiling_tensor(tiling, device);

  if (dtype == torch::kHalf) {
    uint32_t workspace_size = 3 * num_elems * sizeof(uint16_t);
    const at::Tensor workspace_tensor =
        tcuscan::alloc_zeros_workspace(workspace_size, device);
    launch_triu_inv_rec_unroll_fp16(
        block_dim, acl_stream, const_cast<void*>(x.storage().data()),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()),
        tiling_device.data_ptr());
  } else {
    throw std::runtime_error("Currently only half precision is supported.\n");
  }

  aclrtSynchronizeStream(acl_stream);

  return z;
}

}  // namespace tcuscan
