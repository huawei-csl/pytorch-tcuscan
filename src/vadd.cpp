/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file vadd.cpp
 * @brief Entrypoint for `vadd` kernel.
 */

#include "kernels/kernel_vadd.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_vadd.h"

/**
 * @brief Run the `vadd` kernel.
 *
 * @param [in] x Pointer to the input vector.
 * @param [in] y Pointer to the input vector.
 * @param [in] z Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void vadd_custom(GM_ADDR x, GM_ADDR y,
                                                  GM_ADDR z, GM_ADDR workspace,
                                                  GM_ADDR tilingGm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::VaddTiling tiling_data;
  GetTilingData(&tiling_data, tilingGm);

  if ASCEND_IS_AIV {
    tcuscan::KernelAdd op(tiling_data.vec_len, tiling_data.tile_len);
    op.Init(x, y, z);
    op.Process();
  }
}
