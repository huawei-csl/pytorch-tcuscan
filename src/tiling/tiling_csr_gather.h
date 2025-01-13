/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_csr_gather.h
 * @brief Tiling structure for CSR gather operation.
 */

#pragma once

#include <cstdint>

#pragma pack(push, 8)
/**
 * @brief  CSR gather tiling parameter structure.
 */
struct CSRGatherTiling {
  /// @brief Total number of input value elements.
  uint32_t num_elems;
  /// @brief Total number of input x vector  elements.
  uint32_t num_x_elems;
  /// @brief Width of the tile: length of the vectors processed by CumSum
  uint32_t tile_len;
};
#pragma pack(pop)
