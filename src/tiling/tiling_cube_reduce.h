/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file tiling_cube_reduce.h
 * @brief Tiling structure for Cube block reducer.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Cube reduce kernel tiling parameter structure.
 */
struct CubeReduceTiling {
  /// @brief Number of blocks.
  uint32_t num_blocks;
  /// @brief Total number of input elements.
  uint32_t vec_len;
  /// @brief Matrix multiplication size.
  uint32_t matmul_size;
  /// @brief How many of the leading elements the cube units reduce.
  ///
  /// The rest is reduced by the vector units in the same launch and the two
  /// partial sums are added at the end. Always a whole number of
  /// `matmul_size * matmul_size` tiles, so the cube side never needs padding;
  /// any ragged tail falls to the vector side, which handles it natively.
  /// Zero means the vector units do all of it -- the fast path for small
  /// inputs, where the cube pipeline's fixed cost dominates.
  uint32_t cube_len;
};
#pragma pack(pop)

}  // namespace tcuscan
