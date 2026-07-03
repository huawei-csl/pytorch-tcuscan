/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026 All rights reserved.
 *
 * @file tiling_spmv.h
 * @brief Tiling structure for CSR SpMV.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief `spmv` kernel tiling parameter structure.
 */
struct SpMVTiling {
  /// @brief Number of non-zeros elements.
  uint32_t nnz;
  /// @brief Total number of segments.
  uint32_t num_segments;
  /// @brief Length of the dense input vector.
  uint32_t x_len;
  /// @brief Tiling length.
  uint32_t tile_len;
  /// @brief Launch grid size (number of AI Core groups).
  uint32_t block_dim;
  /// @brief L2 cache size in bytes. Pass a very large value to disable L2 splitting.
  uint64_t l2_cache_size;
};

#pragma pack(pop)

}  // namespace tcuscan
