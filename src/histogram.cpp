/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * @file histogram.cpp
 * @brief Kernel implementing the Vector histogram kernel operation.
 */

#include "kernels/ascendc_kernel_operator.h"
#include "kernels/kernel_histogram.h"
#include "tiling/tiling_histogram.h"

/**
 * @brief Run the `histogram` kernel.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] bounds Pointer to the `{min, max}` histogram range. Only read
 * when the tiling sets `bounds_in_gm`; may be null otherwise.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] bin_edges Pointer to the `num_bins + 1` bin edges.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void histogram_fp16(
    GM_ADDR vec_in, GM_ADDR bounds, GM_ADDR vec_out, GM_ADDR bin_edges,
    GM_ADDR workspace, GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  (void)workspace;
  tcuscan::HistogramTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t tile_len = tiling.tile_len;
  const uint32_t num_bins = tiling.num_bins;
  const float x_min = tiling.x_min;
  const float x_max = tiling.x_max;
  const bool bounds_in_gm = tiling.bounds_in_gm != 0;

  tcuscan::run_histogram<half>(vec_in, bounds, vec_out, bin_edges, vec_len,
                               tile_len, num_bins, x_min, x_max, bounds_in_gm);
}

/**
 * @brief Call the `histogram` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] bounds Pointer to the `{min, max}` histogram range buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] bin_edges Pointer to the bin edges output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_histogram_fp16(uint32_t blockDim, void* stream,
                                      uint8_t* vec_in, uint8_t* bounds,
                                      uint8_t* vec_out, uint8_t* bin_edges,
                                      uint8_t* workspace, uint8_t* tiling_gm) {
  histogram_fp16<<<blockDim, nullptr, stream>>>(
      vec_in, bounds, vec_out, bin_edges, workspace, tiling_gm);
}
