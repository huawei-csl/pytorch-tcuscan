/**
 * @file torch_histogram.h
 * @brief Torch wrapper for `histogram` kernel.
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <tuple>
#include <vector>

#include "../tiling/heuristics/heuristics_histogram.h"
#include "../tiling/tiling_histogram.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

// Kernel launchers; documented at their definitions in src/*.cpp
/// @cond
extern "C" void launch_histogram_fp16(uint32_t blockDim, void* stream,
                                      void* vec_in, void* bounds, void* vec_out,
                                      void* bin_edges, void* workspace,
                                      void* tiling_gm);
/// @endcond

namespace tcuscan {

/**
 * @brief Histogram using vector cores; the `torch.histogram` of `x`.
 *
 * Every element of `x` is histogrammed whatever its shape or layout, as
 * `torch.histogram` does. Bin `j` spans `[bin_edges[j], bin_edges[j + 1])`,
 * the last bin is closed on both ends, and elements outside the range are not
 * counted.
 *
 * The histogram spans `range` when it is given and `[min(x), max(x)]`
 * otherwise. The default range is computed on the device and handed to the
 * kernel as a two-element tensor: reading the reduction back to the host would
 * need a blocking synchronization costing far more than the kernel itself.
 *
 * The edges come back from the kernel rather than being reconstructed here:
 * they are the float32 values it compared against, and on the default path
 * the host does not even know the range they span.
 *
 * Unlike `torch.histogram`, the counts come back as float32 rather than in the
 * dtype of `x`: float16 cannot represent counts above 2048 exactly, so it
 * would corrupt the result of any input larger than that.
 *
 * @param [in] x Input data tensor of any shape; only `float16` is supported.
 * @param [in] bins Number of equal-width bins.
 * @param [in] range Two values bounding the bins. Defaults to the range of
 * `x`. An empty range, whether given or degenerate, is widened by half a unit
 * on each side, as `torch.histogram` does.
 * @return Pair of the `bins` counts and the `bins + 1` bin edges.
 */
std::tuple<at::Tensor, at::Tensor> run_histogram(
    const at::Tensor& x, int64_t bins,
    c10::optional<std::vector<double>> range = c10::nullopt) {
  TORCH_CHECK(bins > 0, "run_histogram: bins must be > 0, but got ", bins, ".");
  TORCH_CHECK(x.scalar_type() == at::kHalf,
              "run_histogram: only float16 input is supported, but got ",
              x.scalar_type(), ".");
  TORCH_CHECK(!range.has_value() || range->size() == 2,
              "run_histogram: range must hold exactly two values, but got ",
              range.has_value() ? range->size() : 0, ".");

  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  const at::Device device = x.options().device();
  const auto float_options =
      at::TensorOptions().dtype(torch::kFloat32).device(device);

  // `torch.histogram` histograms every element whatever the shape, and the
  // kernel reads its input as a flat row-major buffer, ignoring both strides
  // and the storage offset. `contiguous()` is a no-op when the layout already
  // is one, and the reshape is then a view.
  const at::Tensor x_flat = x.contiguous().reshape({-1});
  const uint32_t vec_len = static_cast<uint32_t>(x_flat.numel());
  const uint32_t num_bins = static_cast<uint32_t>(bins);

  // Without caller-supplied bounds the range is a device-side reduction that
  // the kernel reads itself; `bounds` must outlive the launch below. An empty
  // input has no range to reduce over, and torch falls back to [0, 1].
  const bool bounds_in_gm = !range.has_value() && vec_len > 0;
  at::Tensor bounds;
  float x_min = 0.0F;
  float x_max = 1.0F;

  if (range.has_value()) {
    x_min = static_cast<float>((*range)[0]);
    x_max = static_cast<float>((*range)[1]);
    TORCH_CHECK(x_min <= x_max,
                "run_histogram: range min must not exceed range max, got ",
                x_min, " > ", x_max, ".");
    // An empty range would collapse every edge onto one value and make the
    // step zero; torch widens it so that the input lands in the middle bin.
    if (x_min == x_max) {
      x_min -= 0.5F;
      x_max += 0.5F;
    }
  }

  if (bounds_in_gm) {
    const auto x_minmax = torch::aminmax(x_flat);
    bounds = torch::stack({std::get<0>(x_minmax), std::get<1>(x_minmax)});
  }

  const at::Tensor z = at::zeros(
      {bins}, at::TensorOptions().dtype(torch::kInt32).device(device));

  // An empty input leaves every bin at zero, and there is no tile to hand to
  // a core; the kernel that would have written the edges never runs, so they
  // are interpolated here instead. Nothing was counted, so the handful of ulp
  // between `linspace` and the kernel's own edges cannot matter.
  if (vec_len == 0) {
    return {z.to(torch::kFloat32),
            at::linspace(x_min, x_max, bins + 1, float_options)};
  }

  const at::Tensor bin_edges = at::empty({bins + 1}, float_options);
  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const uint32_t num_aiv_cores = ascendc_platform->GetCoreNumAiv();
  const HistogramTiling tiling =
      tcuscan::tiling::heuristics::histogram::CalculateTiling(
          vec_len, num_aiv_cores, num_bins, x_min, x_max, bounds_in_gm);
  uint32_t block_dim = tiling.num_blocks;

  const at::Tensor tiling_device = alloc_copy_tiling_tensor(tiling, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  launch_histogram_fp16(block_dim, acl_stream, x_flat.data_ptr(),
                        bounds_in_gm ? bounds.data_ptr() : nullptr,
                        z.data_ptr(), bin_edges.data_ptr(),
                        const_cast<void*>(workspace_tensor.storage().data()),
                        tiling_device.data_ptr());

  aclrtSynchronizeStream(acl_stream);

  return {z.to(torch::kFloat32), bin_edges};
}
}  // namespace tcuscan
