
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file heuristics_cube_reduce.h
 * @brief Heuristics for calculating cube-based reductions tiling.
 */

#pragma once

#include "../../host_utils.h"
#include "../tiling_csr_gather.h"

namespace tcuscan::tiling::heuristics::csr_gather {

/**
 * @brief Determine tiling parameters for radix sort.
 *
 * @param [in] vec_len Number of elements in the input vector.
 * @param [in] num_blocks Number of input blocks.
 * @param [in] verbose Indicates if logging should be enabled.
 * @return Tiling parameter structure.
 */
CsrGatherTiling CalculateTiling(uint32_t vec_len, uint32_t num_blocks,
                                bool verbose = false) {
  CsrGatherTiling tiling;
  tiling.vec_len = vec_len;
  tiling.num_blocks = num_blocks;

  constexpr uint32_t kUbBudget = tcuscan::UB_SIZE_BYTES;
  constexpr uint32_t kTileByteCost =
      2 * (2 * sizeof(T) + sizeof(uint32_t)) + sizeof(uint32_t);
  const uint32_t x_bytes = x_len * sizeof(T);
  const uint32_t ub_bound_tile =
      x_bytes < kUbBudget ? (kUbBudget - x_bytes) / kTileByteCost : 0;
  // No point tiling larger than the whole padded vector; keep 32B UB alignment.
  const uint32_t csr_gather_tile_len = scalar::AlignDown<uint32_t>(
      scalar::Min<uint32_t>(ub_bound_tile, padded_vec_len),
      UB_ALIGNMENT / sizeof(T));

  return tiling;
}

}  // namespace tcuscan::tiling::heuristics::csr_gather
