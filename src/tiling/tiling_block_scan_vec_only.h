/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * @file tiling_block_scan_vec_only.h
 * @brief Tiling structure for Vector block_scan_vec_only kernel operation.
 */

#pragma once

#include <cstdint>

#pragma pack(push, 8)
/**
 * @brief Vector block_scan_vec_only kernel tiling parameter structure.
 */
struct BlockScanVecOnlyTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Total number of input elements.
  uint32_t num_elems;
  /// @brief Tiling length.
  uint32_t tile_len;
};
#pragma pack(pop)