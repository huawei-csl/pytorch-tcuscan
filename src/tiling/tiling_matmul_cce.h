

/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file tiling_matmul_cce.h
 * @brief Tiling structure for MatMul CCE kernel.
 */

#pragma once

#include <cstdint>
namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief MatMul CCE tiling parameter structure. Matrix multiplication C=AB.
 */
struct MatMulCCETiling {
  /// @brief Number of rows of A.
  uint32_t M;
  /// @brief Number of columns of B.
  uint32_t N;
  /// @brief Number of columns of A (and rows of B).
  uint32_t K;
};
#pragma pack(pop)
}  // namespace tcuscan
