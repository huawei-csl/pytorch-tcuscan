/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file gather_spmv.cpp
 * @brief Entrypoint for gather_spmv kernel operation.
 */

#include "kernels/kernel_gather_spmv.h"
#include "tiling/tiling_gather_spmv.h"

/**
 * @brief Run the `gather_spmv` kernel.
 *
 * @param [in] values_in Pointer to input vector.
 * @param [in] cols_in Pointer to column indices input vector.
 * @param [in] z_out Pointer to output vector.
 * @param [in] values_in_len Length of the input values vector.
 * @param [in] tile_len Length of the tile processed in a single iteration.
 */

__aicore__ inline void CopyTiling(GatherSpmvTiling *tiling, GM_ADDR tilingGM) {
  uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
  auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

  for (uint32_t i = 0; i < sizeof(GatherSpmvTiling) / sizeof(uint32_t);
       i++, ptr++) {
    *ptr = *(tiling32 + i);
  }
}

extern "C" __global__ __aicore__ void gather_spmv(GM_ADDR values_in,
                                                  GM_ADDR cols_in,
                                                  GM_ADDR z_out,
                                                  GM_ADDR workspace,
                                                  GM_ADDR tilingGm) {
  (void)workspace;
  GatherSpmvTiling tiling;
  CopyTiling(&tiling, tilingGm);
  run_gather_spmv(values_in, cols_in, z_out, tiling.idx_len, tiling.tile_len);
}