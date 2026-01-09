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
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void histogram_fp16(GM_ADDR vec_in,
                                                     GM_ADDR vec_out,
                                                     GM_ADDR workspace,
                                                     GM_ADDR tiling_gm) {
  (void)workspace;
  tcuscan::HistogramTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t tile_len = tiling.tile_len;
  const uint32_t num_bins = tiling.num_bins;
  const float x_min = tiling.x_min;
  const float x_max = tiling.x_max;

  tcuscan::run_histogram<half>(vec_in, vec_out, vec_len, tile_len, num_bins,
                               x_min, x_max);
}
