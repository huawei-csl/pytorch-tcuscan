/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file scan_multi_cube.cpp
 * @brief Kernel implementing a multi-cube inclusive scan.
 */

#include "kernel_utils.h"
#include "kernels/constants.h"
#include "kernels/kernel_scan_multi_cube.h"
#include "tiling/tiling_scan_multi_cube.h"

/**
 * @brief Run the multi-cube inclusive block scan kernel with dtype fp16
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] lower Pointer to upper-triangular matrix filled with ones.
 * @param [in] upper_strict Pointer to strict upper-triangular matrix filled
 * with ones
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void scan_multi_cube_fp16(
    GM_ADDR input_vec, GM_ADDR lower, GM_ADDR upper_strict, GM_ADDR output_vec,
    GM_ADDR workspace, GM_ADDR tilingGm) {
  ScanMultiCubeTiling tiling;
  tiling::GetTilingData(&tiling, tilingGm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t matmul_size = tiling.matmul_size;

  run_scan_multi_cube_kernel<half>(input_vec, lower, upper_strict, output_vec,
                                   workspace, vec_len, matmul_size);
}
