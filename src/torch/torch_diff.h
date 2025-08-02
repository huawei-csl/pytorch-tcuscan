/**
 * @file torch_diff.h
 * @brief Torch wrapper for vector diff.
 * @date 2025-03-27
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/tiling_diff.h"
#include "aclrtlaunch_diff_fp16.h"
#include "aclrtlaunch_diff_fp32.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace asc {

namespace diff {

/**
 * @brief Torch wrapper of diff kernel.
 *
 * @param x Input vector.
 * @param max_size Number of elements of x to apply diff on.
 * @return Return vector so that z[i] = x[i+1] - x[i] where x[-1] = 0
 */
at::Tensor run_diff(const at::Tensor &x, int64_t max_size) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const auto dtype = x.options().dtype();
  const at::Device device = x.options().device();
  // Tiling length must be around 10,000
  const uint32_t tile_len = 20 * 1024 / byte_size(x);

  uint32_t total_length = x.numel();
  if (max_size > 0) {
    total_length = max_size;
  }

  const at::Tensor z = at::empty(
      {total_length}, at::TensorOptions().dtype(dtype).device(device));
  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  uint32_t block_dim =
      static_cast<uint32_t>((total_length + tile_len - 1) / tile_len);

  if (block_dim < 1) {
    block_dim = 1;
  }

  const DiffTiling tiling{total_length, tile_len};
  uint8_t *tiling_device = alloc_copy_tiling(tiling);

  if (dtype == torch::kHalf) {
    ACLRT_LAUNCH_KERNEL(diff_fp16)
    (block_dim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  } else {
    ACLRT_LAUNCH_KERNEL(diff_fp32)
    (block_dim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

}  // namespace diff

}  // namespace asc
