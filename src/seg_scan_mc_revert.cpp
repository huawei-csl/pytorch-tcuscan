/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file seg_scan_mc_revert.cpp
 * @brief Kernel implementing the vector multi-core segmented scan revertion
 * kernel operation.
 */

#include "kernels/ascendc_kernel_operator.h"
#include "kernels/kernel_seg_scan_mc_revert.h"
#include "tiling/tiling_seg_scan_mc_revert.h"

/**
 * @brief Run the `seg_scan_mc_revert` kernel.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_f_in Pointer to the input flag vector.
 * @param [in] vec_diff_in Pointer to the input diff vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to the workspace in GM.
 * @param [in] tiling Pointer to the tiling structure.
 */

extern "C" __global__ __aicore__ void seg_scan_mc_revert(
    GM_ADDR vec_in, GM_ADDR vec_f_in, GM_ADDR vec_diff_in, GM_ADDR vec_out,
    GM_ADDR workspace, GM_ADDR tiling) {
  (void)workspace;
  SegScanMcRevertTiling tiling_data;
  tiling::GetTilingData(&tiling_data, tiling);

  run_seg_scan_mc_revert<true, float>(
      vec_in, vec_f_in, vec_diff_in, vec_out, tiling_data.num_elems,
      tiling_data.num_segments, tiling_data.tile_len);
}
