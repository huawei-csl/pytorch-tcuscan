/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file seg_scan_vec_single_core.cpp
 * @brief Launcher of seg_scan_vec_single_core
 */

#include "kernels/kernel_seg_scan_vec_single_core.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_seg_scan_vec_single_core.h"

/**
 * @brief Run the single core segmented scan vector-only kernel.
 *
 * @param [in] vec_in Pointer to an input data vector.
 * @param [in] f_in Pointer to an input flag vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to the workspace struct.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void seg_scan_vec_single_core(
    GM_ADDR vec_in, GM_ADDR f_in, GM_ADDR vec_out, GM_ADDR workspace,
    GM_ADDR tilingGm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::SegScanVecSingleCoreTiling tiling_data;
  GetTilingData(&tiling_data, tilingGm);

  if ASCEND_IS_AIV {
    tcuscan::KernelSegScanVecSingleCore<half, int8_t> op_vec(
        tiling_data.num_elems, tiling_data.tile_len);
    op_vec.Init(vec_in, f_in, vec_out);
    op_vec.Process();
  }
}
