/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * @file heuristics_histogram.h
 * @brief Heuristics for calculating histogram tiling.
 */

#pragma once

#include "../../host_utils.h"
#include "../tiling_histogram.h"

namespace tcuscan::tiling::heuristics::histogram {

/**
 * @brief Determine tiling parameters for histogram.
 *
 * @param [in] vec_len Number of elements in the input vector.
 * @param [in] num_aiv_cores Number of AIV cores.
 * @param [in] num_bins Number of histogram bins.
 * @param [in] x_min Minimum value over input.
 * @param [in] x_max Maximum value over input.
 * @return Tiling parameter structure.
 */
HistogramTiling CalculateTiling(uint32_t vec_len, uint32_t num_aiv_cores,
                                uint32_t num_bins, float x_min, float x_max) {
  HistogramTiling tiling{num_aiv_cores, vec_len, 0, num_bins, x_min, x_max};

  // Experimentally selected thresholds
  if (vec_len >= 128 * 128 * num_aiv_cores) {
    tiling.tile_len = 128 * 128;
  } else if (vec_len >= 112 * 112 * num_aiv_cores) {
    tiling.tile_len = 112 * 112;
  } else if (vec_len >= 96 * 96 * num_aiv_cores) {
    tiling.tile_len = 96 * 96;
  } else if (vec_len >= 64 * 64 * num_aiv_cores) {
    tiling.tile_len = 64 * 64;
  } else {
    tiling.tile_len = 32 * 32;
  }

  const uint32_t num_tiles = host_utils::CeilDiv(vec_len, tiling.tile_len);
  if (num_tiles < num_aiv_cores) {
    tiling.num_blocks = num_tiles;
  }

  return tiling;
}

}  // namespace tcuscan::tiling::heuristics::histogram
