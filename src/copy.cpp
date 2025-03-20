/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file copy.cpp
 * @brief Kernel implementing a copy operation.
 */

#include "kernels/kernel_copy.h"
#include "tiling/tiling_copy.h"

/**
 * @brief Run the `copy_fp16` kernel.
 *
 * @param [in] in Pointer to input vector.
 * @param [in] out Pointer to output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to tiling vector.
 */
extern "C" __global__ __aicore__ void copy_fp16(GM_ADDR in, GM_ADDR out,
                                                GM_ADDR workspace,
                                                GM_ADDR tiling) {
  (void)workspace;
  CopyTiling tiling_data;
  tiling::GetTilingData(&tiling_data, tiling);
  run_copy<half>(in, out, tiling_data.num_elems, tiling_data.tile_size);
}

/**
 * @brief Run the `copy_fp32` kernel.
 *
 * @param [in] in Pointer to input vector.
 * @param [in] out Pointer to output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to tiling vector.
 */
extern "C" __global__ __aicore__ void copy_fp32(GM_ADDR in, GM_ADDR out,
                                                GM_ADDR workspace,
                                                GM_ADDR tiling) {
  (void)workspace;
  CopyTiling tiling_data;
  tiling::GetTilingData(&tiling_data, tiling);
  run_copy<float>(in, out, tiling_data.num_elems, tiling_data.tile_size);
}
