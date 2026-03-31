/**
 * @file torch_sort.h
 * @brief Torch wrapper for sorting.
 * @date 2025-03-27
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/heuristics/heuristics_radix_sort.h"
#include "../tiling/tiling_radix_sort.h"
#include "aclrtlaunch_radix_sort_fp16.h"
#include "aclrtlaunch_radix_sort_int16.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "workspace.h"

namespace tcuscan {

/**
 * @brief Radix sort on 1D vector with indices.
 *
 * @param x Input 1D vector.
 * @param S Tiling parameter. Typical values: 32,64, 128.
 * @return Returns 2-tuple of sorted values of `x` along with its indices.
 */
std::tuple<at::Tensor, at::Tensor> run_radix_sort(const at::Tensor& x, int S) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t total_length = x.numel();
  const uint32_t matmul_size = static_cast<uint32_t>(S);

  const uint32_t tile_elems = matmul_size * matmul_size;
  const size_t num_tiles = total_length / tile_elems;

  uint32_t block_dim = ascendc_platform->GetCoreNumAic();
  while (num_tiles % block_dim != 0) {
    block_dim--;
  }
  if (block_dim <= 1) {
    block_dim = 1;
  }

  const at::Tensor vec_out = at::empty(
      {total_length}, at::TensorOptions().dtype(dtype).device(device));
  const at::Tensor indices_out = at::empty(
      {total_length}, at::TensorOptions().dtype(torch::kInt32).device(device));

  RadixSortTiling tiling{block_dim, total_length, matmul_size, tile_elems / 2};
  uint8_t* tiling_device = tcuscan::alloc_copy_tiling(tiling);

  const uint32_t user_workspace_size =
      tcuscan::get_workspace_size<int16_t>(tiling);
  const at::Tensor workspace_tensor =
      tcuscan::alloc_workspace(user_workspace_size, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == torch::kHalf) {
    ACLRT_LAUNCH_KERNEL(radix_sort_fp16)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(vec_out.storage().data()),
     const_cast<void*>(indices_out.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else if (dtype == torch::kInt16) {
    ACLRT_LAUNCH_KERNEL(radix_sort_int16)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(vec_out.storage().data()),
     const_cast<void*>(indices_out.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);

  return std::make_tuple(vec_out, indices_out);
}

}  // namespace tcuscan
