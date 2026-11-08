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

/// @brief Shortest input that still pays for the cube pipeline's setup.
///
/// The cube path carries a far larger fixed cost than the vector one -- a
/// 277 us floor against 205 us on this part -- for only a slightly better
/// streaming rate (600 vs 560 GB/s), so it earns its setup back only on long
/// inputs. Measured on `fp16`, vector-only against the split: 275 vs 292 us
/// at 16.7M elements, 347 vs 353 at 33.6M, 376 vs 367 at 41.9M and 586 vs
/// 549 at 99.9M -- so the two cross between 33.6M and 41.9M. See
/// `CalculateTiling`.
constexpr uint32_t CUBE_MIN_ELEMS = 40u * 1024u * 1024u;

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

  const uint32_t tile_elems = tiling.matmul_size * tiling.matmul_size;

  // Who reduces what: the cube units take whole tiles off the front and the
  // vector units take the rest, the two running concurrently in the one
  // launch. A short input skips the cube pipeline altogether, and its ragged
  // tail always belongs to the vector side, which reduces partial tiles
  // natively -- so the cube side never needs the padding path.
  tiling.cube_len =
      vec_len >= CUBE_MIN_ELEMS ? (vec_len / 2 / tile_elems) * tile_elems : 0;

  const uint32_t num_tiles = tiling.cube_len / tile_elems;

  if (num_tiles == 0) {
    // Vector units only; they are spread over every core.
    tiling.cube_len = 0;
    return tiling;
  }

  // No block may be left idle: the cube stage writes one output region per
  // block and the vector stage reads all of them, so an untouched region --
  // whatever the workspace happened to hold -- would land in the total.
  //
  // The kernel hands out `CeilDiv(num_tiles, num_blocks)` tiles per block
  // (see `GetWorkDistribution`), and only `CeilDiv(num_tiles, per_block)`
  // blocks get one. Using that count is the fixed point of the distribution:
  // the chunk size it implies is unchanged, and the last block is non-empty.
  if (num_tiles < tiling.num_blocks) {
    tiling.num_blocks = num_tiles;
  }
  const uint32_t tiles_per_block =
      host_utils::CeilDiv(num_tiles, tiling.num_blocks);
  tiling.num_blocks = host_utils::CeilDiv(num_tiles, tiles_per_block);

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
