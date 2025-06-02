/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file complete_blocks.cpp
 * @brief Kernel implementing a multi-core AIV complete blocks phase of block
 * scan.
 */

#include "kernel_utils.h"
#include "kernels/kernel_complete_blocks.h"
#include "tiling/tiling_complete_blocks.h"

__aicore__ inline void _run_complete_blocks(GM_ADDR input_vec,
                                            GM_ADDR output_vec,
                                            GM_ADDR tiling_gm) {
  CompleteBlocksTiling tiling;
  tiling::GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t tile_size = tiling.block_size;

  if ASCEND_IS_AIV {
    KernelCompleteBlocks op_complete_blocks(vec_len, tile_size);
    op_complete_blocks.Init(input_vec, output_vec);
    op_complete_blocks.Process();
  }
}

/**
 * @brief Run the multi core vector complete blocks kernel with dtype fp32.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void complete_blocks_fp32(GM_ADDR input_vec,
                                                           GM_ADDR output_vec,
                                                           GM_ADDR workspace,
                                                           GM_ADDR tiling_gm) {
  (void)workspace;
  _run_complete_blocks(input_vec, output_vec, tiling_gm);
}
