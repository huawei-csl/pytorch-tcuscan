/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file gen_lower.cpp
 * @brief gen_lower launcher
 */

#include "kernels/kernel_gen_lower.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_gen_lower.h"

/**
 * @brief Run the `gen_lower` kernel.
 *
 * @param [in] dst Pointer to the destination buffer in global memory.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to tiling vector.
 */
extern "C" __global__ __aicore__ void gen_lower_fp16(GM_ADDR dst,
                                                     GM_ADDR workspace,
                                                     GM_ADDR tiling) {
  (void)workspace;
  tcuscan::GenLowerTiling tiling_data;
  GetTilingData(&tiling_data, tiling);
  run_gen_lower<half, false>(dst, tiling_data.matrix_size);
}

// /**
//  * @brief Run the `gen_lower` kernel.
//  *
//  * @param [in] dst Pointer to the destination buffer in global memory.
//  * @param [in] workspace Pointer to workspace.
//  * @param [in] tiling Pointer to tiling vector.
//  */
// extern "C" __global__ __aicore__ void gen_lower_fp32(GM_ADDR dst,
//                                                      GM_ADDR workspace,
//                                                      GM_ADDR tiling) {
//   (void)workspace;
//   GenLowerTiling tiling_data;
//   GetTilingData(&tiling_data, tiling);
//   run_gen_lower<float>(dst, tiling_data.matrix_size);
// }

/**
 * @brief Run the `gen_lower` kernel.
 *
 * @param [in] dst Pointer to the destination buffer in global memory.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to tiling vector.
 */
extern "C" __global__ __aicore__ void gen_lower_int8(GM_ADDR dst,
                                                     GM_ADDR workspace,
                                                     GM_ADDR tiling) {
  (void)workspace;
  tcuscan::GenLowerTiling tiling_data;
  GetTilingData(&tiling_data, tiling);
  run_gen_lower<int8_t, false>(dst, tiling_data.matrix_size);
}
