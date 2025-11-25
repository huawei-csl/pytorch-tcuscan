/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_copy.h
 * @brief Tiling structure for copy.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Copy tiling parameter structure.
 */
struct CopyTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Total number of input elements in a single vector.
  uint32_t num_elems;
  /// @brief Size of the tiles.
  uint32_t tile_size;
};
#pragma pack(pop)

}  // namespace tcuscan
