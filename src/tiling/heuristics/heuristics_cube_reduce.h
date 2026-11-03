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
 * @brief Determine tiling parameters for cube reduce.
 *
 * The block count is derived here rather than taken from the caller: it is a
 * property of the machine and the input, not of the reduction being asked
 * for. Every cube core is used, except on inputs too short to give each one a
 * matmul tile.
 *
 * @param [in] vec_len Number of elements in the input vector.
 * @param [in] num_aic_cores Number of cube cores available.
 * @param [in] verbose Indicates if logging should be enabled.
 * @return Tiling parameter structure.
 */
CubeReduceTiling CalculateTiling(uint32_t vec_len, uint32_t num_aic_cores,
                                 bool verbose = false) {
  CubeReduceTiling tiling;
  tiling.vec_len = vec_len;
  tiling.num_blocks = num_aic_cores > 0 ? num_aic_cores : 1;

  // The largest tile the input can fill. The block count no longer scales
  // these thresholds: it is derived from the tile count below instead, so a
  // long input keeps the big tile and simply spreads it over more blocks.
  if (vec_len >= 128 * 128) {
    tiling.matmul_size = 128;
  } else if (vec_len >= 64 * 64) {
    tiling.matmul_size = 64;
  } else {
    tiling.matmul_size = 32;
  }

  // Every block must end up with at least one tile. The cube stage writes one
  // output region per block and the vector stage reads all of them, so a block
  // left idle contributes an unwritten region -- whatever the workspace
  // happened to hold -- to the total. Work is handed out as
  // `CeilDiv(num_tiles, num_blocks)` tiles per block, so drop blocks until
  // that distribution reaches the last one.
  const uint32_t tile_elems = tiling.matmul_size * tiling.matmul_size;
  // The kernel pads the input up to whole tiles, so count the padded tiles.
  const uint32_t num_tiles = host_utils::CeilDiv(vec_len, tile_elems);

  if (num_tiles < tiling.num_blocks) {
    tiling.num_blocks = num_tiles;
  }
  while (tiling.num_blocks > 1 &&
         host_utils::CeilDiv(num_tiles, tiling.num_blocks) *
                 (tiling.num_blocks - 1) >=
             num_tiles) {
    tiling.num_blocks--;
  }

  if (verbose) {
    std::cout << "[CubeReduce][Tiling] Vector length: " << vec_len << std::endl;
    std::cout << "[CubeReduce][Tiling] Selected tile size: "
              << tiling.matmul_size << std::endl;
    std::cout << "[CubeReduce][Tiling] Selected number of blocks: "
              << tiling.num_blocks << std::endl;
  }
  return tiling;
}

}  // namespace tcuscan::tiling::heuristics::cube_reduce
