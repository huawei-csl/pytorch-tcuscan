/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file simple_pad.cpp
 * @brief Kernel implementing the Vector simple pad kernel operation.
 */

#include "kernels/ascendc_kernel_operator.h"
#include "kernels/kernel_simple_pad.h"
#include "tiling/tiling_simple_pad.h"

/**
 * @brief Run the `simple_pad_fp16` kernel.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace workspace param. Zero in this case.
 * @param [in] tiling Pointer to the tiling structure.
 */

extern "C" __global__ __aicore__ void simple_pad_fp16(GM_ADDR vec_in,
                                                      GM_ADDR vec_out,
                                                      GM_ADDR workspace,
                                                      GM_ADDR tiling) {
  (void)workspace;
  tcuscan::SimplePadTiling tiling_data;
  GetTilingData(&tiling_data, tiling);

  run_simple_pad<false, half>(vec_in, vec_out, tiling_data.num_elems,
                              tiling_data.align_len);
}
