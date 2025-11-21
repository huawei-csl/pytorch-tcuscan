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

#include "../tiling/tiling_complete_blocks.h"
#include "../tiling/tiling_complete_rows.h"
#include "../tiling/tiling_reduce_tiles.h"
#include "aclrtlaunch_complete_blocks_fp32.h"
#include "aclrtlaunch_complete_blocks_int32.h"
#include "aclrtlaunch_complete_rows_fp32.h"
#include "aclrtlaunch_complete_rows_int32.h"
#include "aclrtlaunch_reduce_tiles_fp16.h"
#include "aclrtlaunch_reduce_tiles_int8.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace tcuscan {

/**
 * @brief Returns the sum-reductions over each tile of an input 1D vector.
 *
 * @param x Input 1D vector.
 * @param tile_len Length of tiles to reduce on.
 * @param block_num Number of AI-cores to used
 * @return Returns a vector whose i-th entry is the reductions of XXX (TODO:
 * anastasios) of length aligned up to 32Bs
 */
at::Tensor run_reduce_tiles(const at::Tensor& x, int tile_len, int block_num) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out =
      dtype == torch::kHalf ? torch::kFloat32 : torch::kInt32;

  const uint32_t s = static_cast<uint32_t>(tile_len);
  const uint32_t block_dim = static_cast<uint32_t>(block_num);
  const uint32_t total_len = x.numel();

  const at::Tensor z = at::empty(
      {block_num}, at::TensorOptions().dtype(dtype_out).device(device));

  const ReduceTilesTiling tiling{total_len, s};
  uint8_t* tiling_device = tcuscan::alloc_copy_tiling(tiling);

  const at::Tensor workspace_tensor = tcuscan::alloc_workspace(0, device);
  if (dtype == torch::kInt8) {
    ACLRT_LAUNCH_KERNEL(reduce_tiles_int8)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else if (dtype == torch::kHalf) {
    ACLRT_LAUNCH_KERNEL(reduce_tiles_fp16)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Broadcast-adds the block-sums into each block on the second phase
 * (down-sweep) of MCSCAN.
 *
 * Assumption: input vector x is split into `len(sums)` blocks.
 *
 * @param [in] x Input 1D vector.
 * @param [in] sums Tensor containing sum-reduction of each block of x.
 * len(sums) == number of blocks
 * @param [in] tile_width Length of the row used by `KernelRowScan` kernel.
 * @param [in] tile_height Number of rows processed in a single iteration
 * `KernelRowScan` kernel.
 * @return Returns a vector where each i-th block of x has been added sims[i].
 */
at::Tensor run_complete_rows(const at::Tensor& x, const at::Tensor& sums,
                             int tile_width, int tile_height) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out =
      dtype == torch::kFloat32 ? torch::kFloat32 : torch::kInt32;

  const uint32_t block_dim = sums.numel();
  const uint32_t total_len = x.numel();
  const uint32_t width = static_cast<uint32_t>(tile_width);
  const uint32_t height = static_cast<uint32_t>(tile_height);

  const at::Tensor z = at::empty(
      {total_len}, at::TensorOptions().dtype(dtype_out).device(device));

  const CompleteRowsTiling tiling{total_len, width, height};
  uint8_t* tiling_device = tcuscan::alloc_copy_tiling(tiling);

  const at::Tensor workspace_tensor = tcuscan::alloc_workspace(0, device);
  if (dtype == torch::kLong) {
    ACLRT_LAUNCH_KERNEL(complete_rows_int32)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(sums.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else if (dtype == torch::kFloat) {
    ACLRT_LAUNCH_KERNEL(complete_rows_fp32)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(sums.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Broadcast-adds the block-sums into each block on the second phase
 * (down-sweep) of block scan.
 *
 * Assumption: input vector x is split into blocks.
 *
 * @param [in] x Input 1D vector.
 * @param [in] sums Tensor containing sum-reduction of consecutive blocks of x.
 * @param [in] tile_length Length of kernel tiles to be used for performance
 * optimization.
 * @return Returns a vector where each i-th block of x has been added
 * sum(sums[:i]).
 */
at::Tensor run_complete_blocks(const at::Tensor& x, const at::Tensor& sums,
                               int tile_length) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t vec_len = x.numel();
  const uint32_t num_blocks = sums.numel();
  const uint32_t tile_len = static_cast<uint32_t>(tile_length);
  const at::Tensor z = at::empty_like(x);

  const CompleteBlocksTiling tiling{vec_len, num_blocks, tile_len};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  const at::Tensor workspace_tensor = alloc_workspace(0, device);
  if (dtype == torch::kFloat) {
    ACLRT_LAUNCH_KERNEL(complete_blocks_fp32)
    (num_blocks, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(sums.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else if (dtype == torch::kInt32) {
    ACLRT_LAUNCH_KERNEL(complete_blocks_int32)
    (num_blocks, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(sums.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

}  // namespace tcuscan
