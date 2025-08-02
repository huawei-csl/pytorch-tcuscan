/**
 * @file torch_copy.h
 * @brief Torch wrapper for memcopy.
 * @date 2025-03-27
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/tiling_copy.h"
#include "aclrtlaunch_copy_fp16.h"
#include "aclrtlaunch_copy_fp32.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace asc {

namespace copy {

/**
 * @brief Memory copy (single AI core)
 *
 * @param x Input data vector.
 * @param s Tiling length
 * @return Copy of input vector `x`.
 */
at::Tensor run_copy(const at::Tensor &x, int s) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t total_length = x.numel();
  const at::Tensor z = at::empty(
      {total_length}, at::TensorOptions().dtype(dtype).device(device));

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const uint32_t num_block = 1;  // required for 1 core
  const uint32_t tile_size = static_cast<uint32_t>(s);
  const CopyTiling tiling{num_block, total_length, tile_size};
  uint8_t *tiling_device = alloc_copy_tiling(tiling);

  if (dtype == at::kFloat) {
    ACLRT_LAUNCH_KERNEL(copy_fp32)
    (1 /* single core*/, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  } else {
    ACLRT_LAUNCH_KERNEL(copy_fp16)
    (1 /* single core*/, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}
}  // namespace copy

}  // namespace asc
