/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file heuristics_radix_sort.h
 * @brief Heuristics for calculating radix sort tiling.
 */

#pragma once

#include "../../host_utils.h"
#include "../tiling_radix_sort.h"

namespace tiling::heuristics::radix_sort {

/**
 * @brief Determine tiling parameters for radix sort.
 *
 * @param [in] vec_len Number of elements in the input vector.
 * @param [in] num_ai_cores Number of HW AI cores.
 * @param [in] verbose Indicates if logging should be enabled.
 * @return Tiling parameter structure.
 */
RadixSortTiling CalculateTiling(uint32_t vec_len, uint32_t num_ai_cores,
                                bool verbose = false) {
  RadixSortTiling tiling;
  tiling.num_elems = vec_len;

  // Experimentally selected thresholds
  if (vec_len >= 128 * 128 * 4) {
    tiling.matmul_size = 128;
  } else if (vec_len >= 64 * 64) {
    tiling.matmul_size = 64;
  } else {
    tiling.matmul_size = 32;
  }
  if (verbose) {
    std::cout << "[RadixSort][Tiling] Vector length: " << vec_len << std::endl;
    std::cout << "[RadixSort][Tiling] Selected tile size: "
              << tiling.matmul_size << std::endl;
  }

  const uint32_t tile_elems = tiling.matmul_size * tiling.matmul_size;
  tiling.vec_tile_size = tile_elems / 2;
  const size_t num_tiles = vec_len / tile_elems;

  tiling.num_blocks = num_ai_cores;
  while (num_tiles % tiling.num_blocks != 0) {
    tiling.num_blocks--;
  }

  if (verbose) {
    std::cout << "[RadixSort][Tiling] Selected number of blocks: "
              << tiling.num_blocks << std::endl;
  }
  return tiling;
}

}  // namespace tiling::heuristics::radix_sort
