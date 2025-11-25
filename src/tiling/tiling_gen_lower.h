/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_gen_lower.h
 * @brief Tiling structure for gen_lower.
 */

#pragma once

#include <cstdint>
namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Copy tiling parameter structure.
 */
struct GenLowerTiling {
  /// @brief Size of the triangular matrix (number of rows / columns)
  uint32_t matrix_size;
  /// @brief Size of the tiles.
  uint32_t tile_size;
};
#pragma pack(pop)
}  // namespace tcuscan
