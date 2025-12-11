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
#include "aclrtlaunch_tri_inv_col_sweep_fp16.h"
#include "aclrtlaunch_tri_inv_col_sweep_fp32.h"
#include "aclrtlaunch_tri_inv_cube_col_sweep_fp16.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "workspace.h"

namespace tcuscan {

/**
 * @brief Torch wrapper for vector `tri_inv_col_sweep` kernel.
 * @param [in] x Input Tensor of (batch_dim, n, n ). Matrices are assumed to be
 * unit upper triangular matrices.
 * @return Returns Matrix inverse for each matrix over the `batch_dim`.
 */
at::Tensor run_tri_inv_col_sweep(const at::Tensor &x) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t block_dim = static_cast<uint32_t>(x.size(0));
  const uint32_t matrix_size = static_cast<uint32_t>(x.size(1));
  const uint32_t num_elems = static_cast<uint32_t>(x.numel());

  const at::Tensor z = at::empty_like(x);
  const at::Tensor workspace_tensor = tcuscan::alloc_workspace(0, device);
  const TriInvColumnSweepTiling tiling{block_dim, num_elems, matrix_size};
  uint8_t *tiling_device = tcuscan::alloc_copy_tiling(tiling);

  if (dtype == torch::kHalf) {
    ACLRT_LAUNCH_KERNEL(tri_inv_col_sweep_fp16)
    (block_dim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  } else {
    ACLRT_LAUNCH_KERNEL(tri_inv_col_sweep_fp32)
    (block_dim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Torch wrapper for vector `tri_inv_cube_col_sweep` kernel.
 * @param [in] x Input Tensor of (batch_dim, n, n ). Matrices are assumed to be
 * unit upper triangular matrices.
 * @return Returns Matrix inverse for each matrix over the `batch_dim`.
 */
at::Tensor run_tri_inv_cube_col_sweep(const at::Tensor &x) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();

  const uint32_t block_dim = static_cast<uint32_t>(x.size(0));
  const uint32_t matrix_size = static_cast<uint32_t>(x.size(1));

  const at::Tensor z =
      at::empty({block_dim, matrix_size, matrix_size},
                at::TensorOptions().dtype(at::kFloat).device(device));

  const TriInvCubeColSweepTiling tiling{block_dim, matrix_size};
  uint8_t *tiling_device = tcuscan::alloc_copy_tiling(tiling);
  const size_t workspace_size =
      workspace::get_workspace_size<uint16_t /* half */>(tiling);
  const at::Tensor workspace_tensor =
      tcuscan::alloc_workspace(workspace_size, device);

  ACLRT_LAUNCH_KERNEL(tri_inv_cube_col_sweep_fp16)
  (block_dim, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tiling_device);

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

}  // namespace tcuscan
