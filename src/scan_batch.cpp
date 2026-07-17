/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file scan_batch.cpp
 * @brief Kernel implementing a multi-core inclusive scan using either
 * cube-vector or cube-only approach on a batched input.
 */

#include "kernels/constants.h"
#include "kernels/kernel_scan_batch.h"
#include "tiling/tiling_scan_batch.h"

/**
 *  @brief Run the multi core inclusive scan kernel on a batched input.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to a workspace vector (empty vector!).
 * @param [in] tiling Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void scan_batch_fp16(GM_ADDR input_vec,
                                                      GM_ADDR output_vec,
                                                      GM_ADDR workspace,
                                                      GM_ADDR tiling) {
  using tcuscan::run_row_scan_kernel;
  using tcuscan::run_scan_batch_kernel;

  tcuscan::ScanBatchTiling tiling_data;
  GetTilingData(&tiling_data, tiling);
  GM_ADDR const lower = load_tril_matrix<half>(tiling_data.matmul_size);

  ASCENDC_ASSERT(tiling_data.batch_size % tiling_data.vec_cube_ratio == 0, {
    KERNEL_LOG(KERNEL_ERROR,
               "The batch size (%d) must be "
               "divisible by the ratio between vector and cube cores (%d).",
               tiling_data.batch_size, tiling_data.vec_cube_ratio);
  });

  if (tiling_data.num_elems <= tiling_data.matmul_size) {
    run_row_scan_kernel<half>(input_vec, lower, output_vec,
                              tiling_data.num_elems, tiling_data.batch_size,
                              tiling_data.matmul_size, workspace);
  } else {
    run_scan_batch_kernel<half>(input_vec, lower, output_vec,
                                tiling_data.num_elems, tiling_data.batch_size,
                                tiling_data.matmul_size,
                                tiling_data.vec_cube_ratio, workspace);
  }
}

/**
 *  @brief Run the multi core inclusive scan kernel on a batched input.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to a workspace vector (empty vector!).
 * @param [in] tiling Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void scan_batch_fp32(GM_ADDR input_vec,
                                                      GM_ADDR output_vec,
                                                      GM_ADDR workspace,
                                                      GM_ADDR tiling) {
  using tcuscan::run_row_scan_kernel;
  using tcuscan::run_scan_batch_kernel;

  tcuscan::ScanBatchTiling tiling_data;
  GetTilingData(&tiling_data, tiling);
  GM_ADDR const lower = load_tril_matrix<float>(tiling_data.matmul_size);

  ASCENDC_ASSERT(tiling_data.batch_size % tiling_data.vec_cube_ratio == 0, {
    KERNEL_LOG(KERNEL_ERROR,
               "The batch size (%d) must be "
               "divisible by the ratio between vector and cube cores (%d).",
               tiling_data.batch_size, tiling_data.vec_cube_ratio);
  });

  if (tiling_data.num_elems <= tiling_data.matmul_size) {
    run_row_scan_kernel<float>(input_vec, lower, output_vec,
                               tiling_data.num_elems, tiling_data.batch_size,
                               tiling_data.matmul_size, workspace);
  } else {
    run_scan_batch_kernel<float>(input_vec, lower, output_vec,
                                 tiling_data.num_elems, tiling_data.batch_size,
                                 tiling_data.matmul_size,
                                 tiling_data.vec_cube_ratio, workspace);
  }
}
