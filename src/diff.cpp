/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file diff.cpp
 * @brief Entrypoint for diff kernel operation.
 */

#include "kernels/kernel_diff.h"
#include "tiling/tiling_diff.h"

/**
 * @brief Convert tiling struct to global memory.
 *
 * @param [in] tiling Input tiling struct.
 * @param [in] tilingGM Output global memory point to write tiling struct.
 */
__aicore__ inline void CopyTiling(DiffTiling *tiling, GM_ADDR tilingGM) {
  uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
  auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

  for (uint32_t i = 0; i < sizeof(DiffTiling) / sizeof(uint32_t); i++, ptr++) {
    *ptr = *(tiling32 + i);
  }
}

/**
 * @brief Run the `diff` kernel with data type float16/half.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */

extern "C" __global__ __aicore__ void diff_fp16(GM_ADDR vec_in, GM_ADDR vec_out,
                                                GM_ADDR workspace,
                                                GM_ADDR tilingGm) {
  (void)workspace;
  DiffTiling tiling;
  CopyTiling(&tiling, tilingGm);

  run_diff<true, half>(vec_in, vec_out, tiling.vec_len, tiling.tile_len);
}

/**
 * @brief Run the `diff` kernel with data type float32.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */

extern "C" __global__ __aicore__ void diff_fp32(GM_ADDR vec_in, GM_ADDR vec_out,
                                                GM_ADDR workspace,
                                                GM_ADDR tilingGm) {
  (void)workspace;
  DiffTiling tiling;
  CopyTiling(&tiling, tilingGm);

  run_diff<true, float>(vec_in, vec_out, tiling.vec_len, tiling.tile_len);
}
