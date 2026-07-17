/**
 * @file torch_pad.h
 * @brief Torch wrapper for vector pad.
 * @date 2025-03-27
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../host_utils.h"
#include "../tiling/tiling_simple_pad.h"
#include "aclrtlaunch_simple_pad_fp16.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace tcuscan {

/**
 * @brief Torch wrapper for vector add kernel.
 * @param x Input tensor.
 * @param align_len tile size to pad the vector
 * @return Returns the x input tensor padded up to align_len
 */
at::Tensor run_simple_pad(const at::Tensor& x, const uint32_t align_len) {
  at::Tensor z = at::empty({align_len}, x.options());
  const at::Device device = x.options().device();

  const uint32_t vec_len = x.numel();
  const uint32_t total_len = align_len;
  const uint32_t block_dim = 1;

  const SimplePadTiling tiling{block_dim, vec_len, align_len};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);
  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  ACLRT_LAUNCH_KERNEL(simple_pad_fp16)
  (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
   const_cast<void*>(z.storage().data()),
   const_cast<void*>(workspace_tensor.storage().data()), tiling_device);

  aclrtFree(tiling_device);

  return z;
}

}  // namespace tcuscan
