/**
 * @file torch_vadd.h
 * @brief Torch wrapper for vector add.
 * @date 2025-03-27
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../host_utils.h"
#include "../tiling/tiling_vadd.h"
#include "aclrtlaunch_vadd_custom.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace asc {

namespace add {

/**
 * @brief Torch wrapper for vector add kernel.
 * @param x Input tensor.
 * @param y Input tensor.
 * @return Returns element-wise addition, i.e., x+y.
 */
at::Tensor run_add(const at::Tensor &x, const at::Tensor &y) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Tensor z = at::empty_like(x);
  const at::Device device = x.options().device();
  const uint32_t tile_len = 128;
  const uint32_t total_len = x.numel();
  const uint32_t block_dim = host_utils::CeilDiv(total_len, 2 * tile_len);
  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const VaddTiling tiling{total_len, tile_len};
  uint8_t *tiling_device = allocCopyTiling(tiling);

  ACLRT_LAUNCH_KERNEL(vadd_custom)
  (block_dim, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(y.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tiling_device);

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

}  // namespace add

}  // namespace asc
