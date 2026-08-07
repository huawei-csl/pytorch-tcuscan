/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file scan_vector_only.cpp
 * @brief Entrypoint for scan vector-only kernel operation.
 */

#include "kernels/kernel_scan_vec_only.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_scan_vec_only.h"

/**
 * @brief Run the scan vec only kernel.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void scan_vec_only_fp16(GM_ADDR input_vec,
                                                         GM_ADDR output_vec,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tiling) {
  (void)workspace;
  tcuscan::ScanVecOnlyTiling tiling_data;
  tcuscan::GetTilingData(&tiling_data, tiling);

  if ASCEND_IS_AIV {
    tcuscan::KernelScanVecOnly op(
        tiling_data.num_elems, tiling_data.tile_height, tiling_data.tile_width);
    op.Init(input_vec, output_vec);
    op.Process();
  }
}
