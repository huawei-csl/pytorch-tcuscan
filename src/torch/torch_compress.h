/**
 * @file torch_compress.h
 * @brief Torch wrapper for compaction/compress.
 * @date 2025-03-27
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/tiling_compress.h"
#include "aclrtlaunch_compress_fp16.h"
#include "aclrtlaunch_compress_fp32.h"
#include "aclrtlaunch_compress_pos_fp16.h"
#include "aclrtlaunch_compress_pos_fp32.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "workspace.h"

namespace asc {

namespace compress {

/**
 * @brief Parallel vector compaction / compress.
 *
 * @param x Input data 1D vector.
 * @param mask Input boolean mask vector.
 * @param S Tiling parameter. Typical values: 32, 64, 128.
 * @return The subvector of `x`: `x[f == 1]`. Output length is `sum(f == 1)`.
 */
at::Tensor run_compress(const at::Tensor &x, const at::Tensor &mask, int S) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(SOC_VERSION);
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t total_length = x.numel();

  const uint32_t tile_elems = matmul_size * matmul_size;
  const uint32_t vec_tile_size = tile_elems / 2;

  const uint32_t num_tiles = total_length / tile_elems;

  uint32_t block_dim = ascendc_platform->GetCoreNum() / 2;
  while (num_tiles % block_dim != 0) {
    block_dim--;
  }
  if (block_dim <= 1) {
    block_dim = 1;
  }

  const at::Tensor z = at::empty(
      {total_length}, at::TensorOptions().dtype(dtype).device(device));

  const CompressTiling tiling{total_length, matmul_size, vec_tile_size};
  uint8_t *tiling_device = alloc_copy_tiling(tiling);

  const uint32_t user_workspace_size =
      workspace::compress::get_workspace_size<int8_t>(tiling, block_dim);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  if (dtype == torch::kHalf or dtype == torch::kInt16) {
    ACLRT_LAUNCH_KERNEL(compress_fp16)
    (block_dim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(mask.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  } else {
    ACLRT_LAUNCH_KERNEL(compress_fp32)
    (block_dim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(mask.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Run the compress kernel with positions. Kernel also takes a position
 * vector on the input which is an output of inclusive scan on the binary input
 * mask.
 *
 * @param x Input data 1D vector.
 * @param mask Input boolean mask vector.
 * @param pos Input position vector.
 * @param S Tiling parameter. Typical values: 32, 64, 128.
 * @return The subvector of `x`: `x[f == 1]`. Output length is `sum(f == 1)`.
 */
at::Tensor run_compress_pos(const at::Tensor &x, const at::Tensor &mask,
                            const at::Tensor &pos, int S) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(SOC_VERSION);
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t total_length = x.numel();

  const uint32_t tile_elems = matmul_size * matmul_size;
  const uint32_t vec_tile_size = tile_elems / 2;

  const uint32_t num_tiles = total_length / tile_elems;

  uint32_t block_dim = ascendc_platform->GetCoreNum() / 2;
  while (num_tiles % block_dim != 0) {
    block_dim--;
  }
  if (block_dim <= 1) {
    block_dim = 1;
  }

  // Last entry of pos tensor contains number of output elements.
  const at::Tensor z =
      at::empty({pos[total_length - 1].item<int32_t>()},
                at::TensorOptions().dtype(dtype).device(device));

  const CompressTiling tiling{total_length, matmul_size, vec_tile_size};
  uint8_t *tiling_device = alloc_copy_tiling(tiling);

  const uint32_t user_workspace_size =
      workspace::compress::get_workspace_size<int8_t>(tiling, block_dim);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  if (dtype == torch::kHalf or dtype == torch::kInt16) {
    ACLRT_LAUNCH_KERNEL(compress_pos_fp16)
    (block_dim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(mask.storage().data()),
     const_cast<void *>(pos.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  } else {
    ACLRT_LAUNCH_KERNEL(compress_pos_fp32)
    (block_dim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(mask.storage().data()),
     const_cast<void *>(pos.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}
}  // namespace compress

}  // namespace asc
