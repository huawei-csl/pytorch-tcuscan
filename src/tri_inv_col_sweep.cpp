/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file tri_inv_col_sweep.cpp
 * @brief Entrypoint for vector-only `tri_inv_col_sweep` kernel.
 */

#include "kernels/kernel_tri_inv_col_sweep.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_tri_inv_col_sweep.h"

/**
 * @brief Run the `tri_inv_col_sweep` kernel.
 *
 * @param [in] vec_in Pointer to input vector.
 * @param [in] vec_out Pointer to output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to tiling vector.
 */
extern "C" __global__ __aicore__ void tri_inv_col_sweep_fp16(
    GM_ADDR vec_in, GM_ADDR vec_out, GM_ADDR workspace, GM_ADDR tiling_gm) {
  (void)workspace;
  tcuscan::TriInvColumnSweepTiling tiling;
  tiling::GetTilingData(&tiling, tiling_gm);

  tcuscan::tri_inv_col_sweep<half, false>(vec_in, vec_out, tiling.num_elems,
                                          tiling.matrix_size);
}
