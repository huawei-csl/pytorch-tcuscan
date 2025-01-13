/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_compress.h
 * @brief Tiling structure for compress kernel.
 */

#pragma once

#include <cstdint>

#pragma pack(push, 8)
/**
 * @brief Compress tiling parameter structure.
 */
struct CompressTiling {
  /// @brief Total number of input elements.
  uint32_t size;
  /// @brief Tile length.
  uint32_t scan_tile_size;
  /// @brief Compress tile.
  uint32_t compress_tile_size;
};
#pragma pack(pop)