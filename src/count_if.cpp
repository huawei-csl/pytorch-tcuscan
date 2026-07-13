/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file count_if.cpp
 * @brief Kernel implementing Vector `count_if` operation.
 */

#include "kernels/ascendc_kernel_operator.h"
#include "kernels/kernel_count_if.h"
#include "tiling/tiling_count_if.h"

/**
 * @brief Run the `count_if` kernel with input fp16.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling structure.
 */

extern "C" __global__ __aicore__ void count_if_fp16(GM_ADDR vec_in,
                                                    GM_ADDR vec_out,
                                                    GM_ADDR workspace,
                                                    GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::CountIfTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  tcuscan::run_count_if<false, half>(vec_in, vec_out, tiling.num_elems,
                                     tiling.tile_len, tiling.pivot);
}
