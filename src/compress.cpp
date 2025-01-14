/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file compress.cpp
 * @brief Multi-core compress approach.
 */

#include "kernels/constants.h"
#include "kernels/kernel_compress.h"
#include "lib/matmul_intf.h"
#include "tiling/tiling_compress.h"

using namespace AscendC;

__aicore__ inline void CopyTiling(CompressTiling *tiling, GM_ADDR tilingGM) {
  uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
  auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

  for (uint32_t i = 0; i < sizeof(CompressTiling) / sizeof(uint32_t);
       i++, ptr++) {
    *ptr = *(tiling32 + i);
  }
}

extern "C" __global__ __aicore__ void compress(GM_ADDR x, GM_ADDR mask,
                                               GM_ADDR z, GM_ADDR workspace,
                                               GM_ADDR tilingGm) {
  CompressTiling tiling;
  CopyTiling(&tiling, tilingGm);

  // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha003/quickstart/quickstart/quickstart_18_0001.html?sub_id=%2Fzh%2FCANNCommunityEdition%2F80RC2alpha003%2Fdevguide%2Fopdevg%2Fascendcopdevg%2Fatlas_ascendc_10_0083.html
  GM_ADDR usrWorkspace =
      AscendC::GetUserWorkspace(workspace);  // Get the user workspace pointer.

  // Select lower-triangular all-ones matrix staticly initialized on device
  // See `kernel_constants.h`
  GM_ADDR lower = load_tril_matrix<int8_t>(tiling.scan_tile_size);

  const uint32_t in_size = tiling.size;
  const uint32_t scan_tile_size = tiling.scan_tile_size;
  const uint32_t compress_tile_size = tiling.compress_tile_size;

  run_compress_uint16(x, mask, z, lower, usrWorkspace, in_size, scan_tile_size,
                      compress_tile_size);
}