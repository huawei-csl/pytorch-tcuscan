/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file block_scan.cpp
 * @brief Kernel implementing a multi-core inclusive block scan.
 */

#include "kernels/constants.h"
#include "kernels/kernel_block_scan.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_block_scan.h"

template <typename InputT>
__aicore__ inline void _run_block_scan(GM_ADDR input_vec, GM_ADDR lower,
                                       GM_ADDR upper_strict, GM_ADDR output_vec,
                                       GM_ADDR tilingGm) {
  BlockScanTiling tiling;
  tiling::GetTilingData(&tiling, tilingGm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t matmul_size = tiling.matmul_size;

  if ASCEND_IS_AIC {
    KernelBlockScan<InputT> op_cube(vec_len, matmul_size);
    op_cube.Init(input_vec, lower, upper_strict, output_vec);
    op_cube.Process();
  }
}

/**
 * @brief Run the multi core inclusive block scan kernel with dtype fp16
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] lower Pointer to upper-triangular matrix filled with ones.
 * @param [in] upper_strict Pointer to strict upper-triangular matrix filled
 * with ones
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void block_scan_fp16(
    GM_ADDR input_vec, GM_ADDR lower, GM_ADDR upper_strict, GM_ADDR output_vec,
    GM_ADDR workspace, GM_ADDR tilingGm) {
  (void)workspace;
  _run_block_scan<half>(input_vec, lower, upper_strict, output_vec, tilingGm);
}
