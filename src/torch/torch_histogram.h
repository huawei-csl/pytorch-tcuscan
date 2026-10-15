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
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

// Kernel launchers; documented at their definitions in src/*.cpp
/// @cond
extern "C" void launch_histogram_fp16(uint32_t blockDim, void* stream,
                                      void* vec_in, void* bounds, void* vec_out,
                                      void* workspace, void* tiling_gm);
/// @endcond

namespace tcuscan {

/**
 * @brief Histogram using vector cores.
 *
 * The histogram spans `[hist_min, hist_max]`. When those are not given they
 * default to the range of `x`, which is computed on the device and handed to
 * the kernel as a two-element tensor: reading the reduction back to the host
 * would need a blocking synchronization costing far more than the kernel
 * itself.
 *
 * @param [in] x Input data vector.
 * @param [in] num_bins Number of bins.
 * @param [in] hist_min Lower end of the histogram range. Defaults to `min(x)`.
 * @param [in] hist_max Upper end of the histogram range. Defaults to `max(x)`.
 * @return Tensor of length `bin_count` containing the histogram of range
 * [hist_min, hist_max].
 */
at::Tensor run_histogram(const at::Tensor& x, uint32_t num_bins,
                         c10::optional<double> hist_min = c10::nullopt,
                         c10::optional<double> hist_max = c10::nullopt) {
  TORCH_CHECK(hist_min.has_value() == hist_max.has_value(),
              "run_histogram: hist_min and hist_max must be given together, "
              "or neither of them.");

  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  // Without caller-supplied bounds the range is a device-side reduction that
  // the kernel reads itself; `bounds` must outlive the launch below.
  const bool bounds_in_gm = !hist_min.has_value();
  at::Tensor bounds;
  float x_min = 0.0F;
  float x_max = 0.0F;

  if (bounds_in_gm) {
    const auto x_minmax = torch::aminmax(x);
    bounds = torch::stack({std::get<0>(x_minmax), std::get<1>(x_minmax)});
  } else {
    x_min = static_cast<float>(hist_min.value());
    x_max = static_cast<float>(hist_max.value());
    TORCH_CHECK(x_min <= x_max,
                "run_histogram: hist_min must not exceed hist_max, got ", x_min,
                " > ", x_max, ".");
  }

  const uint32_t vec_len = x.numel();
  const at::Tensor z = at::zeros(
      {num_bins}, at::TensorOptions().dtype(torch::kInt32).device(device));

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const uint32_t num_aiv_cores = ascendc_platform->GetCoreNumAiv();
  const HistogramTiling tiling =
      tcuscan::tiling::heuristics::histogram::CalculateTiling(
          vec_len, num_aiv_cores, num_bins, x_min, x_max, bounds_in_gm);
  uint32_t block_dim = tiling.num_blocks;

  const at::Tensor tiling_device =
      alloc_copy_tiling_tensor(tiling, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == at::kHalf) {
    launch_histogram_fp16(
        block_dim, acl_stream, const_cast<void*>(x.storage().data()),
        bounds_in_gm ? const_cast<void*>(bounds.storage().data()) : nullptr,
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device.data_ptr());
  } else {
    /* Unsupported */
  }

  aclrtSynchronizeStream(acl_stream);

  return z;
}
}  // namespace tcuscan
