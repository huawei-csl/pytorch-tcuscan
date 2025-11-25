/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file tiling_topk_pivot.h
 * @brief Tiling structure for Vector K-largest value estimator.
 */

#pragma once

#include <cstdint>
namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Vector K-largest estimator kernel tiling parameter structure.
 */
struct TopKPivotTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Total number of input elements.
  uint32_t num_elems;
  /// @brief Tiling length.
  uint32_t tile_len;
  /// @brief Top-K parameter.
  uint32_t k;
};
#pragma pack(pop)
}  // namespace tcuscan
