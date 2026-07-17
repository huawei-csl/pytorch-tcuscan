/**
 * @file torch_histogram.h
 * @brief Torch wrapper for `histogram` kernel.
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/heuristics/heuristics_histogram.h"
#include "../tiling/tiling_histogram.h"
#include "aclrtlaunch_histogram_fp16.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace tcuscan {

/**
 * @brief Histogram using vector cores.
 *
 *
 * @param [in] x Input data vector.
 * @param [in] num_bins Number of bins.
 * @return Tensor of length `bin_count` containing the histogram of range
 * [x_min, x_max].
 */
at::Tensor run_histogram(const at::Tensor& x, uint32_t num_bins) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const auto x_minmax = torch::aminmax(x);
  const float x_min = std::get<0>(x_minmax).item<float>();
  const float x_max = std::get<1>(x_minmax).item<float>();

  const uint32_t vec_len = x.numel();
  const at::Tensor z = at::zeros(
      {num_bins}, at::TensorOptions().dtype(torch::kInt32).device(device));

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const uint32_t num_aiv_cores = ascendc_platform->GetCoreNumAiv();
  const HistogramTiling tiling =
      tcuscan::tiling::heuristics::histogram::CalculateTiling(
          vec_len, num_aiv_cores, num_bins, x_min, x_max);
  uint32_t block_dim = tiling.num_blocks;

  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == at::kHalf) {
    ACLRT_LAUNCH_KERNEL(histogram_fp16)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else {
    /* Unsupported */
  }

  aclrtFree(tiling_device);

  return z;
}
}  // namespace tcuscan
