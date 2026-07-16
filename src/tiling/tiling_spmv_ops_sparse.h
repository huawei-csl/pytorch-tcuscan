/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026 All rights reserved.
 *
 * @file tiling_spmv_ops_sparse.h
 * @brief Tiling structure for the row-parallel CSR SpMV kernel migrated from
 * CANN ops-sparse.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief `spmv_ops_sparse` kernel tiling parameter structure.
 *
 * Encodes the shape of the sparse matrix and the GEMV scaling coefficients for
 * the operation \f$ y \leftarrow \alpha \cdot A x + \beta \cdot y \f$ (or its
 * transpose when @ref trans is non-zero).
 */
struct SpMVOpsSparseTiling {
  /// @brief Number of rows of the CSR matrix (`indptr.numel() - 1`).
  uint32_t total_rows_num;
  /// @brief Number of columns of the CSR matrix.
  uint32_t total_cols_num;
  /// @brief Scaling factor applied to the matrix-vector product.
  float alpha;
  /// @brief Scaling factor applied to the pre-existing output vector.
  float beta;
  /// @brief Non-zero to compute the transposed product \f$ A^\top x \f$.
  uint32_t trans;
};

#pragma pack(pop)

}  // namespace tcuscan
