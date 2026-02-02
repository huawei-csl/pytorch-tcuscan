/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file tri_inv_cube_col_sweep.cpp
 * @brief Entrypoint for vector-only `tri_inv_cube_col_sweep` kernel.
 */

#include "kernels/kernel_tri_inv_cube_col_sweep.h"
#include "kernels/kernel_vec_col_sweep_mat_gen.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_tri_inv_cube_col_sweep.h"

/**
 * @brief Run the `tri_inv_cube_col_sweep_fp16` kernel.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void tri_inv_cube_col_sweep_fp16(
    GM_ADDR vec_in, GM_ADDR vec_out, GM_ADDR workspace, GM_ADDR tiling_gm) {
  using tcuscan::KernelTriInvCubeColSweep;
  using tcuscan::KernelVecColSweepMatGen;

  tcuscan::TriInvCubeColSweepTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t matrix_size = tiling.matrix_size;
  const uint32_t ws_circular_buffer_len = tiling.ws_circular_buffer_len;

  if ASCEND_IS_AIV {
    KernelVecColSweepMatGen<half> op(matrix_size, ws_circular_buffer_len);
    op.Init(vec_in, workspace);
    op.Process();
  }

  if ASCEND_IS_AIC {
    KernelTriInvCubeColSweep<half> op(matrix_size, ws_circular_buffer_len);
    op.Init(workspace, vec_out);
    op.Process();
  }
}
