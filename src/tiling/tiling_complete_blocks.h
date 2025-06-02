/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file tiling_complete_blocks.h
 * @brief Tiling structure for AIV complete blocks kernel.
 */

#pragma once

#include <cstdint>

#pragma pack(push, 8)
/**
 * @brief Tiling struct of `complete_blocks` kernel.
 */
struct CompleteBlocksTiling {
  /// @brief Number of elements
  uint32_t num_elems;
  /// @brief block_size
  uint32_t block_size;
};
#pragma pack(pop)
