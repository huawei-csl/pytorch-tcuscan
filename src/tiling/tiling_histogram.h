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
  /// @brief Minimum value of the input vector.
  float x_min;
  /// @brief Maximum value of the input vector.
  float x_max;
};
#pragma pack(pop)

}  // namespace tcuscan
