/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file reduce_tiles.cpp
 * @brief Kernel implementing a multi-core AIV reduce tiles.
 */

#include "kernel_utils.h"
#include "kernels/kernel_reduce_tiles.h"
#include "tiling/tiling_reduce_tiles.h"

template <typename InputT>
__aicore__ inline void _run_reduce_tiles(GM_ADDR input_vec, GM_ADDR output_vec,
                                         GM_ADDR tilingGm) {
  ReduceTilesTiling tiling;
  tiling::GetTilingData(&tiling, tilingGm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t tile_len = tiling.tile_len;

  if ASCEND_IS_AIV {
    KernelReduceTiles<InputT> op_reduce(tile_len, vec_len);
    op_reduce.Init(input_vec, output_vec);
    op_reduce.Process();
  }
}

/**
 * @brief Run the multi core vector reduce tiles kernel with dtype fp16.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void reduce_tiles_fp16(GM_ADDR input_vec,
                                                        GM_ADDR output_vec,
                                                        GM_ADDR workspace,
                                                        GM_ADDR tilingGm) {
  (void)workspace;
  _run_reduce_tiles<half>(input_vec, output_vec, tilingGm);
}

/**
 * @brief Run the multi core vector reduce tiles kernel with dtype int8.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void reduce_tiles_int8(GM_ADDR input_vec,
                                                        GM_ADDR output_vec,
                                                        GM_ADDR workspace,
                                                        GM_ADDR tilingGm) {
  (void)workspace;
  _run_reduce_tiles<int8_t>(input_vec, output_vec, tilingGm);
}
