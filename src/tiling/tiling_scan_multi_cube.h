

/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file tiling_scan_multi_cube.h
 * @brief Tiling structure for SCAN multi-cube.
 */

#pragma once

#include <cstdint>

#pragma pack(push, 8)
/**
 * @brief Scan multi-cube tiling parameter structure.
 */
struct ScanMultiCubeTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Number of elements.
  uint32_t num_elems;
  /// @brief Matmul size used. Each block has length `matmul_size *
  /// matmul_size`.
  uint32_t matmul_size;
};
#pragma pack(pop)
