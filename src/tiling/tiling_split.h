/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_split.h
 * @brief Tiling structure for split.
 */

#pragma once

#include <cstdint>
namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Split tiling parameter structure.
 */
struct SplitTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Total number of input elements in a single vector.
  uint32_t num_elems;
  /// @brief Size of the matmul tile used in the cube part of scan.
  uint32_t scan_matmul_size;
  /// @brief Size of the vector tiles.
  uint32_t vec_tile_size;
};
#pragma pack(pop)
}  // namespace tcuscan
