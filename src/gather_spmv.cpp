/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file gather_spmv.cpp
 * @brief Entrypoint for gather_spmv kernel operation.
 */

#include "kernels/kernel_gather_spmv.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_gather_spmv.h"

/**
 * @brief Gather SpMV kernel for CSR sparse matrices.
 *
 * @param values_in Input data vector.
 * @param cols_in Input column indices of CSR sparse format.
 * @param vec_out Output vector.
 * @param workspace Pointer to workspace.
 * @param tilingGm Pointer to tiling structure.
 */
extern "C" __global__ __aicore__ void gather_spmv(GM_ADDR values_in,
                                                  GM_ADDR cols_in,
                                                  GM_ADDR vec_out,
                                                  GM_ADDR workspace,
                                                  GM_ADDR tilingGm) {
  (void)workspace;
  GatherSpmvTiling tiling;
  tiling::GetTilingData(&tiling, tilingGm);

  run_gather_spmv(values_in, cols_in, vec_out, tiling.idx_len, tiling.val_len,
                  tiling.tile_len);
}
