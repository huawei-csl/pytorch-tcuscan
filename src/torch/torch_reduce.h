/**
 * @file torch_reduce.h
 * @brief Torch wrapper for reduce kernels.
 * @date 2025-05-13
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/tiling_reduce_tiles.h"
#include "aclrtlaunch_reduce_tiles_fp16.h"
#include "aclrtlaunch_reduce_tiles_int8.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "workspace.h"

namespace asc {

namespace reduce {

/**
 * @brief Returns the sum-reductions over each tile of an input 1D vector.
 *
 * @param x Input 1D vector.
 * @param tile_len Length of tiles to reduce on.
 * @param block_num Number of AI-cores to used
 * @return Returns a vector whose i-th entry is the reductions of XXX (TODO:
 * anastasios) of length aligned up to 32Bs
 */
at::Tensor run_reduce_tiles(const at::Tensor &x, int tile_len, int block_num) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out =
      dtype == torch::kHalf ? torch::kFloat32 : torch::kInt32;

  const uint32_t s = 2 * static_cast<uint32_t>(tile_len);
  const uint32_t block_dim = static_cast<uint32_t>(block_num);
  const uint32_t total_len = x.numel();

  const at::Tensor z = at::empty(
      {block_num}, at::TensorOptions().dtype(dtype_out).device(device));

  const ReduceTilesTiling tiling{total_len, s};
  uint8_t *tiling_device = allocCopyTiling(tiling);

  const at::Tensor workspace_tensor = alloc_workspace(0, device);
  if (dtype == torch::kInt8) {
    ACLRT_LAUNCH_KERNEL(reduce_tiles_int8)
    (block_dim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  } else if (dtype == torch::kHalf) {
    ACLRT_LAUNCH_KERNEL(reduce_tiles_fp16)
    (block_dim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

}  // namespace reduce

}  // namespace asc
