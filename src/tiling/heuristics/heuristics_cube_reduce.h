/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file heuristics_cube_reduce.h
 * @brief Heuristics for calculating cube-based reductions tiling.
 */

#pragma once

#include "../../host_utils.h"
#include "../tiling_cube_reduce.h"

namespace tcuscan::tiling::heuristics::cube_reduce {

/**
 * @brief Determine tiling parameters for radix sort.
 *
 * @param [in] vec_len Number of elements in the input vector.
 * @param [in] num_blocks Number of input blocks.
 * @param [in] verbose Indicates if logging should be enabled.
 * @return Tiling parameter structure.
 */
CubeReduceTiling CalculateTiling(uint32_t vec_len, uint32_t num_blocks,
                                 bool verbose = false) {
  CubeReduceTiling tiling;
  tiling.vec_len = vec_len;
  tiling.num_blocks = num_blocks;

  // Experimentally selected thresholds
  if (vec_len >= 128 * 128 * num_blocks) {
    tiling.matmul_size = 128;
  } else if (vec_len >= 64 * 64 * num_blocks) {
    tiling.matmul_size = 64;
  } else {
    tiling.matmul_size = 32;
  }
  if (verbose) {
    std::cout << "[CubeReduce][Tiling] Vector length: " << vec_len << std::endl;
    std::cout << "[CubeReduce][Tiling] Selected tile size: "
              << tiling.matmul_size << std::endl;
  }

  if (verbose) {
    std::cout << "[RadixSort][Tiling] Selected number of blocks: "
              << tiling.num_blocks << std::endl;
  }
  return tiling;
}

}  // namespace tcuscan::tiling::heuristics::cube_reduce
