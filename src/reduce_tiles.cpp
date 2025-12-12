/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * @file reduce_tiles.cpp
 * @brief Kernel implementing a multi-core AIV reduce tiles.
 */

#include "kernels/kernel_reduce_tiles.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_reduce_tiles.h"

/**
 * @brief Run the reduce_tiles kernel.
 *
 * @tparam T Input data type
 *
 * @param [in] vec_in Pointer to an input vector.
 * @param [in] vec_out Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] vec_len Number of elements to process.
 * @param [in] tile_len Tile length.
 * output vector.
 */
template <typename InputT>
__aicore__ inline void _run_reduce_tiles(GM_ADDR vec_in, GM_ADDR vec_out,
                                         GM_ADDR workspace, uint32_t vec_len,
                                         uint32_t tile_len) {
  (void)workspace;

  if ASCEND_IS_AIV {
    KernelReduceTiles<InputT> op_reduce(tile_len, vec_len);
    op_reduce.Init(vec_in, vec_out);
    op_reduce.Process();
  }
}

/**
 * @brief Run the multi core vector reduce tiles kernel with dtype fp16.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void reduce_tiles_fp16(GM_ADDR input_vec,
                                                        GM_ADDR output_vec,
                                                        GM_ADDR workspace,
                                                        GM_ADDR tiling_gm) {
  tcuscan::ReduceTilesTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t tile_len = tiling.tile_len;

  _run_reduce_tiles<half>(input_vec, output_vec, workspace, vec_len, tile_len);
}

/**
 * @brief Run the multi core vector reduce tiles kernel with dtype int8.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void reduce_tiles_int8(GM_ADDR input_vec,
                                                        GM_ADDR output_vec,
                                                        GM_ADDR workspace,
                                                        GM_ADDR tiling_gm) {
  tcuscan::ReduceTilesTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t tile_len = tiling.tile_len;

  _run_reduce_tiles<int8_t>(input_vec, output_vec, workspace, vec_len,
                            tile_len);
}
