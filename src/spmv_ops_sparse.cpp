/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * @file spmv_ops_sparse.cpp
 * @brief Entrypoints for the row-parallel CSR SpMV kernel migrated from CANN
 * ops-sparse.
 */

#include "kernels/kernel_spmv_ops_sparse.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_spmv_ops_sparse.h"

using namespace AscendC;
using namespace tcuscan;

/**
 * @brief Run the `spmv_ops_sparse` kernel with float32 dtype.
 *
 * Computes \f$ y \leftarrow \alpha \cdot A x + \beta \cdot y \f$ where \f$ A
 * \f$ is a CSR matrix. The row-pointer convention follows the scipy CSR format
 * (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).
 *
 * @param [in] csrRowPtr CSR row-pointer array (length rows + 1).
 * @param [in] csrColInd CSR column-index array (length nnz).
 * @param [in] csrVal CSR non-zero values (length nnz).
 * @param [in] xVec Dense input vector.
 * @param [in,out] yVec Dense output vector.
 * @param [in,out] workspace Pointer to workspace (unused).
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void spmv_ops_sparse_fp32(
    GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec,
    GM_ADDR yVec, GM_ADDR workspace, GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  (void)workspace;

  tcuscan::SpMVOpsSparseTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  run_spmv_ops_sparse<float, float, float>(
      csrRowPtr, csrColInd, csrVal, xVec, yVec, tiling.total_rows_num,
      tiling.total_cols_num, tiling.alpha, tiling.beta, tiling.trans != 0);
}

/**
 * @brief Run the `spmv_ops_sparse` kernel with float16 storage and float32
 * compute.
 *
 * The CSR values, dense vector `x` and output `y` are stored as float16, while
 * the multiply / reduce math is carried out in float32 for accuracy.
 *
 * @param [in] csrRowPtr CSR row-pointer array (length rows + 1).
 * @param [in] csrColInd CSR column-index array (length nnz).
 * @param [in] csrVal CSR non-zero values (length nnz).
 * @param [in] xVec Dense input vector.
 * @param [in,out] yVec Dense output vector.
 * @param [in,out] workspace Pointer to workspace (unused).
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void spmv_ops_sparse_fp16(
    GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec,
    GM_ADDR yVec, GM_ADDR workspace, GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  (void)workspace;

  tcuscan::SpMVOpsSparseTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  run_spmv_ops_sparse<float, half, half>(
      csrRowPtr, csrColInd, csrVal, xVec, yVec, tiling.total_rows_num,
      tiling.total_cols_num, tiling.alpha, tiling.beta, tiling.trans != 0);
}
