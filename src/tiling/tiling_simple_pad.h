/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_simple_pad.h
 * @brief Tiling structure for Vector simple pad kernel operation.
 */

#pragma once

#include <cstdint>

#pragma pack(push, 8)
/**
 * @brief Vector simple pad kernel tiling parameter structure.
 */
struct SimplePadTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Total number of input elements.
  uint32_t num_elems;
  /// @brief Length of alignment.
  uint32_t align_len;
};
#pragma pack(pop)