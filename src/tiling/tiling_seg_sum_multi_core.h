/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026 All rights reserved.
 *
 * @file tiling_seg_sum_multi_core.h
 * @brief Tiling structure for segmented sum single core.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief `seg_sum_multi_core` kernel tiling parameter structure.
 */
struct SegSumMultiCoreTiling {
  /// @brief Total number of input elements.
  uint32_t num_elems;
  /// @brief Total number of segments.
  uint32_t num_segments;
  /// @brief Tiling length.
  uint32_t tile_len;
  /// @brief Workspace size for single core segmented sum per block.
  uint32_t single_core_ws_size;
};

#pragma pack(pop)

}  // namespace tcuscan
