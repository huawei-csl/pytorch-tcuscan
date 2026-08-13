/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file block_scan_vec_only.cpp
 * @brief Kernel implementing a multi-core inclusive block scan using vector
 * cores.
 */

#include "kernels/kernel_block_scan_vec_only.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_block_scan_vec_only.h"

namespace tcuscan {

/**
 * @brief Run the multi core inclusive block scan kernel
 *
 *
 * @tparam InputT Input data type.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
template <typename InputT>
__aicore__ inline void run_block_scan_vec_only(GM_ADDR input_vec,
                                               GM_ADDR output_vec,
                                               GM_ADDR tiling_gm) {
  BlockScanVecOnlyTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t tile_len = tiling.tile_len;

  if ASCEND_IS_AIV {
    KernelBlockScanVecOnly<InputT> op_vec(vec_len, tile_len);
    op_vec.Init(input_vec, output_vec);
    op_vec.Process();
  }
}

}  // namespace tcuscan

/**
 * @brief Run the multi core inclusive block scan kernel with dtype fp16
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void block_scan_vec_only_fp16(
    GM_ADDR input_vec, GM_ADDR output_vec, GM_ADDR workspace,
    GM_ADDR tilingGm) {
  (void)workspace;
  tcuscan::run_block_scan_vec_only<half>(input_vec, output_vec, tilingGm);
}
