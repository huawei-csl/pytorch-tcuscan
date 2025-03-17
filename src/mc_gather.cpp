/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_mc_gather.cpp
 * @brief Kernel implementing the Vector MCGather kernel operation.
 */

#include "kernels/kernel_mc_gather.h"
#include "tiling/tiling_mc_gather.h"

__aicore__ inline void CopyTiling(McGatherTiling *tiling, GM_ADDR tilingGM) {
  uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
  auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

  for (uint32_t i = 0; i < sizeof(McGatherTiling) / sizeof(uint32_t);
       i++, ptr++) {
    *ptr = *(tiling32 + i);
  }
}

/**
 * @brief Run the `mc_gather` kernel.
 *
 * @param [in] values_in Pointer to input vector.
 * @param [in] cols_in Pointer to column indices input vector.
 * @param [in] z_out Pointer to output vector.
 * @param [in] values_in_len Length of the input values vector.
 * @param [in] tile_len Length of the tile processed in a single iteration.
 */

extern "C" __global__ __aicore__ void mc_gather(GM_ADDR values_in,
                                                GM_ADDR indexes_in,
                                                GM_ADDR z_out,
                                                GM_ADDR workspace,
                                                GM_ADDR tilingGm) {
  (void)workspace;
  McGatherTiling tiling;
  CopyTiling(&tiling, tilingGm);

  run_mc_gather(values_in, indexes_in, z_out, tiling.num_elems,
                tiling.tile_len);
}
