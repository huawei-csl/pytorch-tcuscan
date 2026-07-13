/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file triu_inv_rec_unroll.cpp
 * @brief Kernel implementing the triangular matrix inverse kernel operation.
 */

#include "kernels/ascendc_kernel_operator.h"
#include "kernels/kernel_triu_inv_rec_unroll.h"
#include "tiling/tiling_triu_inv_rec_unroll.h"

/**
 * @brief Run the `triu_inv_rec_unroll` kernel.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to the workspace.
 * @param [in] tiling Pointer to the tiling structure.
 */

extern "C" __global__ __aicore__ void triu_inv_rec_unroll_fp16(
    GM_ADDR vec_in, GM_ADDR vec_out, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  TriuInvRecUnrollTiling tiling_data;
  tcuscan::GetTilingData(&tiling_data, tiling);

  run_triu_inv_rec_unroll<half>(vec_in, vec_out, tiling_data.matrix_size,
                                tiling_data.num_blocks, workspace);
}
