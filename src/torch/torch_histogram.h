/**
 * @file torch_histogram.h
 * @brief Torch wrapper for `histogram` kernel.
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/tiling_histogram.h"
#include "aclrtlaunch_histogram_fp16.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace tcuscan {

/**
 * @brief Vector histogram kernel
 *
 * @param [in] x Input data vector.
 * @param [in] num_bins Number of bins.
 * @param [in] s Tiling length
 * @return Tensor of length `bin_count` containing the histogram of range
 * [x_min, x_max].
 */
at::Tensor run_histogram(const at::Tensor& x, uint32_t num_bins, int s) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const auto x_minmax = torch::aminmax(x);
  const float x_min = std::get<0>(x_minmax).item<float>();
  const float x_max = std::get<1>(x_minmax).item<float>();

  const uint32_t vec_len = x.numel();
  // TODO(anastasios): fix to num_bins and `torch::kInt32`
  const at::Tensor z =
      at::empty({vec_len}, at::TensorOptions().dtype(dtype).device(device));

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const uint32_t block_dim = 8;
  const uint32_t tile_size = static_cast<uint32_t>(s);
  const HistogramTiling tiling{block_dim, vec_len, tile_size,
                               num_bins,  x_min,   x_max};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  if (dtype == at::kHalf) {
    ACLRT_LAUNCH_KERNEL(histogram_fp16)
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
}  // namespace tcuscan
