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
  /// @brief Block length (per-block, per-L2-chunk non-zero slab).
  uint32_t block_len;
  /// @brief Number of serial L2 chunks the non-zeros are split into. A value of
  /// `1` disables L2 splitting (single chunk covering all non-zeros).
  uint32_t num_l2_splits;
  /// @brief Launch grid size (number of AI Core groups). Passed explicitly so
  /// the device derives `fitting_len = block_dim * block_len` identically on
  /// Cube and Vector cores.
  uint32_t block_dim;
};

#pragma pack(pop)

}  // namespace tcuscan
