/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file seg_scan_vec_single_core.cpp
 * @brief Launcher of seg_scan_vec_single_core
 */

#include "kernels/kernel_seg_scan_vec_single_core.h"
#include "tiling/tiling_seg_scan_vec_single_core.h"

__aicore__ inline void CopyTiling(SegScanVecSingleCoreTiling *tiling,
                                  GM_ADDR tilingGM) {
  uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
  auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

  for (uint32_t i = 0;
       i < sizeof(SegScanVecSingleCoreTiling) / sizeof(uint32_t); i++, ptr++) {
    *ptr = *(tiling32 + i);
  }
}

/**
 * @brief Run the single core segmented scan vector-only kernel.
 *
 * @param [in] vec_in Pointer to an input data vector.
 * @param [in] f_in Pointer to an input flag vector.
 * @param [in] output_vec Pointer to the output vector.
 * @param [in] workspace Pointer to the workspace struct.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void seg_scan_vec_single_core(
    GM_ADDR vec_in, GM_ADDR f_in, GM_ADDR vec_out, GM_ADDR workspace,
    GM_ADDR tilingGm) {
  (void)workspace;
  SegScanVecSingleCoreTiling tiling;
  CopyTiling(&tiling, tilingGm);

  if ASCEND_IS_AIV {
    KernelSegScanVecSingleCore<half, int8_t> op_vec(tiling.num_elems,
                                                    tiling.tile_len);
    op_vec.Init(vec_in, f_in, vec_out);
    op_vec.Process();
  }
}
