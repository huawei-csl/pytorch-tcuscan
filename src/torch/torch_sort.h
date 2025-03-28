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

namespace asc {

namespace sort {

/**
 * @brief Radix sort on 1D vector with indices.
 *
 * @param x Input 1D vector.
 * @param S Tiling parameter. Typical values: 32,64, 128.
 * @return Returns 2-tuple of sorted values of `x` along with its indices.
 */
std::tuple<at::Tensor, at::Tensor> run_radix_sort(const at::Tensor &x, int S) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(SOC_VERSION);
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t totalLength = x.numel();
  const uint32_t matmul_size = static_cast<uint32_t>(S);

  const uint32_t tile_elems = matmul_size * matmul_size;
  const size_t num_tiles = totalLength / tile_elems;

  uint32_t blockDim = ascendc_platform->GetCoreNum() / 2;
  while (num_tiles % blockDim != 0) {
    blockDim--;
  }
  if (blockDim <= 1) {
    blockDim = 1;
  }

  const at::Tensor vec_out =
      at::empty({totalLength}, at::TensorOptions().dtype(dtype).device(device));
  const at::Tensor indices_out = at::empty(
      {totalLength}, at::TensorOptions().dtype(torch::kInt32).device(device));

  RadixSortTiling tiling{blockDim, totalLength, matmul_size, tile_elems / 2};
  uint8_t *tiling_device = allocCopyTiling(tiling);

  const uint32_t user_workspace_size =
      workspace::radix_sort::GetWorkspaceSize<int16_t>(tiling);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  if (dtype == torch::kHalf) {
    ACLRT_LAUNCH_KERNEL(radix_sort_fp16)
    (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(vec_out.storage().data()),
     const_cast<void *>(indices_out.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  } else if (dtype == torch::kInt16) {
    ACLRT_LAUNCH_KERNEL(radix_sort_int16)
    (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(vec_out.storage().data()),
     const_cast<void *>(indices_out.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return std::make_tuple(vec_out, indices_out);
}
}  // namespace sort

}  // namespace asc
