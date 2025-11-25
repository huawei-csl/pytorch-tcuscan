/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file complete_blocks.cpp
 * @brief Kernel implementing a multi-core AIV complete blocks phase of block
 * scan.
 */

#include "kernels/kernel_complete_blocks.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_complete_blocks.h"

template <typename T>
__aicore__ inline void _run_complete_blocks(GM_ADDR input_vec,
                                            GM_ADDR input_sums,
                                            GM_ADDR output_vec,
                                            GM_ADDR tiling_gm) {
  tcuscan::CompleteBlocksTiling tiling;
  tiling::GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t num_blocks = tiling.num_blocks;
  const uint32_t tile_len = tiling.tile_len;

  if ASCEND_IS_AIV {
    const uint32_t block_len = vec_len / num_blocks;
    KernelCompleteBlocks<T> op_complete_blocks(vec_len, block_len, tile_len);
    op_complete_blocks.Init(input_vec, input_sums, output_vec);
    op_complete_blocks.Process();
  }
}

/**
 * @brief Run the multi core vector complete blocks kernel with dtype fp32.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] input_sums Pointer to vector with partial block reductions
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void complete_blocks_fp32(GM_ADDR input_vec,
                                                           GM_ADDR input_sums,
                                                           GM_ADDR output_vec,
                                                           GM_ADDR workspace,
                                                           GM_ADDR tiling_gm) {
  (void)workspace;
  _run_complete_blocks<float>(input_vec, input_sums, output_vec, tiling_gm);
}

/**
 * @brief Run the `complete_blocks` kernel.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] input_sums Pointer to vector with partial block reductions
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void complete_blocks_int32(GM_ADDR input_vec,
                                                            GM_ADDR input_sums,
                                                            GM_ADDR output_vec,
                                                            GM_ADDR workspace,
                                                            GM_ADDR tiling_gm) {
  (void)workspace;
  _run_complete_blocks<int32_t>(input_vec, input_sums, output_vec, tiling_gm);
}
