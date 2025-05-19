/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file complete_rows.cpp
 * @brief Kernel implementing a multi-core AIV complete rows phase of MCSCAN.
 */

#include "kernel_utils.h"
#include "kernels/kernel_complete_rows.h"
#include "tiling/tiling_complete_rows.h"

template <typename InputT>
__aicore__ inline void _run_complete_rows(GM_ADDR input_vec, GM_ADDR input_sums,
                                          GM_ADDR output_vec,
                                          GM_ADDR tilingGm) {
  CompleteRowsTiling tiling;
  tiling::GetTilingData(&tiling, tilingGm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t tile_width = tiling.tile_width;
  const uint32_t tile_height = tiling.tile_height;

  if ASCEND_IS_AIV {
    KernelCompleteRows<InputT> op_reduce(tile_width, tile_height, vec_len);
    op_reduce.Init(input_vec, input_sums, output_vec);
    op_reduce.Process();
  }
}

/**
 * @brief Run the multi core vector reduce tiles kernel with dtype fp32.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] input_sums Pointer to vector with partial sums
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void complete_rows_fp32(GM_ADDR input_vec,
                                                         GM_ADDR input_sums,
                                                         GM_ADDR output_vec,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tilingGm) {
  (void)workspace;
  _run_complete_rows<float>(input_vec, input_sums, output_vec, tilingGm);
}

/**
 * @brief Run the multi core vector reduce tiles kernel with dtype int32.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] input_sums Pointer to vector with partial sums
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void complete_rows_int32(GM_ADDR input_vec,
                                                          GM_ADDR input_sums,
                                                          GM_ADDR output_vec,
                                                          GM_ADDR workspace,
                                                          GM_ADDR tilingGm) {
  (void)workspace;
  _run_complete_rows<int32_t>(input_vec, input_sums, output_vec, tilingGm);
}
