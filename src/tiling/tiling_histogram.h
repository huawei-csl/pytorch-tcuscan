/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file tiling_histogram.h
 * @brief Tiling structure for Vector histogram kernel operation.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Vector histogram kernel tiling parameter structure.
 */
struct HistogramTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Total number of input elements.
  uint32_t vec_len;
  /// @brief Tiling length.
  uint32_t tile_len;
  /// @brief Number of histogram bins.
  uint32_t num_bins;
  /// @brief Non-zero if the kernel must read the histogram range from global
  /// memory instead of from `x_min` / `x_max`.
  ///
  /// The range is a property of the data, so computing it on the host would
  /// require reading a device-side reduction back, which drains the stream.
  /// When this flag is set, the host leaves the reduction on the device and
  /// the kernel loads the two values itself; `x_min` and `x_max` are then
  /// unused.
  uint32_t bounds_in_gm;
  /// @brief Minimum value of the input vector. Unused if `bounds_in_gm`.
  float x_min;
  /// @brief Maximum value of the input vector. Unused if `bounds_in_gm`.
  float x_max;
};
#pragma pack(pop)

}  // namespace tcuscan
