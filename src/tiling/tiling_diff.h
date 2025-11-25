/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_diff.h
 * @brief Tiling structure for vector diff.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Vector diff tiling parameter structure.
 */
struct DiffTiling {
  /// @brief Total number of input elements.
  uint32_t vec_len;
  /// @brief Tile length.
  uint32_t tile_len;
};
#pragma pack(pop)
}  // namespace tcuscan
