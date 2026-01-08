/**
 * @file torch_compare.h
 * @brief Torch wrapper for comparison kernels.
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/tiling_count_if.h"
#include "../tiling/tiling_greater_equal.h"
#include "aclrtlaunch_count_if_fp16.h"
#include "aclrtlaunch_greater_equal_fp16.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "workspace.h"

namespace tcuscan {

/**
 * @brief Vector count_if kernel
 *
 * @param [in] x Input data vector.
 * @param [in] pivot Input pivot
 * @param [in] tile_len Tile length that is assigned on each AIV core.
 * @return Copy of input vector `x`.
 */
at::Tensor run_count_if(const at::Tensor& x, float pivot, uint32_t tile_len) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t vec_len = x.numel();
  const uint32_t block_dim = host_utils::CeilDiv(vec_len, tile_len);

  const at::Tensor z = at::empty(
      {block_dim}, at::TensorOptions().dtype(torch::kInt32).device(device));

  const CountIfTiling tiling{block_dim, vec_len, tile_len, pivot};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  if (dtype == at::kHalf) {
    ACLRT_LAUNCH_KERNEL(count_if_fp16)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else {
    /* Unsupported */
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Returns binary tensor `{ x_i >= pivot }` of type `int8_t`.
 *
 * @param [in] x Input data vector.
 * @param [in] pivot Input pivot
 * @param [in] tile_len Tile length that is assigned on each AIV core.
 * @return Copy of input vector `x`.
 */
at::Tensor run_greater_equal(const at::Tensor& x, float pivot,
                             uint32_t tile_len) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t num_elems = x.numel();

  const uint32_t block_dim = host_utils::CeilDiv(num_elems, tile_len);
  const at::Tensor z = at::empty(
      {num_elems}, at::TensorOptions().dtype(torch::kInt8).device(device));

  const GreaterEqualTiling tiling{block_dim, num_elems, tile_len, pivot};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  if (dtype == at::kHalf) {
    ACLRT_LAUNCH_KERNEL(greater_equal_fp16)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else {
    /* Not supported */
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}
}  // namespace tcuscan
