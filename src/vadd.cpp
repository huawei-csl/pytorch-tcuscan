/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file vadd.cpp
 * @brief vadd launcher
 */

#include "kernels/kernel_vadd.h"
#include "tiling/tiling_vadd.h"

__aicore__ inline void CopyTiling(VaddTiling *tiling, GM_ADDR tilingGM) {
  uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
  auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

  for (uint32_t i = 0; i < sizeof(VaddTiling) / sizeof(uint32_t); i++, ptr++) {
    *ptr = *(tiling32 + i);
  }
}

/**
 * @brief Run the `vadd` kernel.
 *
 * @param [in] x Pointer to the input vector.
 * @param [in] y Pointer to the input vector.
 * @param [in] z Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y,
                                                 GM_ADDR z, GM_ADDR workspace,
                                                 GM_ADDR tilingGm) {
  VaddTiling tiling;
  CopyTiling(&tiling, tilingGm);

  if ASCEND_IS_AIV {
    KernelAdd op(tiling.vec_len, tiling.tile_len);
    op.Init(x, y, z);
    op.Process();
  }
}
