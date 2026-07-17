/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file mc_gather.cpp
 * @brief Kernel implementing the Vector MCGather kernel operation.
 */

#include "kernels/kernel_mc_gather.h"
#include "tiling/tiling_mc_gather.h"

/**
 * @brief Run the `mc_gather` kernel for dtype fp32.
 *
 * @param [in] values_in Pointer to input vector.
 * @param [in] indexes_in Pointer to column indices input vector.
 * @param [in] z_out Pointer to output vector.
 * @param [in] workspace Length of the input values vector.
 * @param [in] tiling_gm Length of the tile processed in a single iteration.
 */
extern "C" __global__ __aicore__ void mc_gather_fp32(GM_ADDR values_in,
                                                     GM_ADDR indexes_in,
                                                     GM_ADDR z_out,
                                                     GM_ADDR workspace,
                                                     GM_ADDR tiling_gm) {
  (void)workspace;
  tcuscan::McGatherTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  tcuscan::run_mc_gather<float>(values_in, indexes_in, z_out, tiling.idx_len,
                                tiling.val_len, tiling.tile_len);
}

/**
 * @brief Run the `mc_gather` kernel for dtype fp32.
 *
 * @param [in] values_in Pointer to input vector.
 * @param [in] indexes_in Pointer to column indices input vector.
 * @param [in] z_out Pointer to output vector.
 * @param [in] workspace Length of the input values vector.
 * @param [in] tiling_gm Length of the tile processed in a single iteration.
 */
extern "C" __global__ __aicore__ void mc_gather_fp16(GM_ADDR values_in,
                                                     GM_ADDR indexes_in,
                                                     GM_ADDR z_out,
                                                     GM_ADDR workspace,
                                                     GM_ADDR tiling_gm) {
  (void)workspace;
  tcuscan::McGatherTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  tcuscan::run_mc_gather<half>(values_in, indexes_in, z_out, tiling.idx_len,
                               tiling.val_len, tiling.tile_len);
}
