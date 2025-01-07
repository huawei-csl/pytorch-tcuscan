/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_diff.cpp
 * @brief Kernel implementing the diff kernel operation.
 */

#include "kernels/diff.h"

/**
 * @brief Run the `diff` kernel.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] vec_len Vector length.
 * @param [in] tile_len Tile length.
 */

extern "C" __global__ __aicore__ void diff(GM_ADDR vec_in, GM_ADDR vec_out,
                                           uint32_t vec_len,
                                           uint32_t tile_len) {
  run_diff<true, half>(vec_in, vec_out, vec_len, tile_len);
}
