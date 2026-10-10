/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * @file tiling_searchsorted.h
 * @brief Tiling structure for the searchsorted kernel.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief `searchsorted` kernel tiling parameter structure.
 */
struct SearchsortedTiling {
  /// @brief Length of the sorted haystack array.
  uint32_t data_len;
  /// @brief Number of needle values to search for.
  uint32_t num_values;
};
#pragma pack(pop)

}  // namespace tcuscan
