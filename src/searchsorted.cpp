/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * @file searchsorted.cpp
 * @brief Entrypoint for the searchsorted kernel operation.
 */

#include "kernels/kernel_searchsorted.h"
#include "tiling/tiling_searchsorted.h"

/**
 * @brief Run the `searchsorted` kernel with int32 data.
 *
 * @param [in] sorted Pointer to the sorted haystack.
 * @param [in] values Pointer to the needle values.
 * @param [in] out Pointer to the output indices.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void searchsorted_int32(GM_ADDR sorted,
                                                         GM_ADDR values,
                                                         GM_ADDR out,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tilingGm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::SearchsortedTiling tiling;
  GetTilingData(&tiling, tilingGm);

  tcuscan::run_searchsorted<true, int32_t>(sorted, values, out, tiling.data_len,
                                           tiling.num_values);
}
