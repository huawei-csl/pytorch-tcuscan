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

extern "C" __global__ __aicore__ void compress_fp16(GM_ADDR x, GM_ADDR mask,
                                                    GM_ADDR z,
                                                    GM_ADDR workspace,
                                                    GM_ADDR tilingGm) {
  CompressTiling tiling;
  CopyTiling(&tiling, tilingGm);

  GM_ADDR const usrWorkspace = AscendC::GetUserWorkspace(workspace);
  GM_ADDR const lower = load_tril_matrix<int8_t>(tiling.scan_tile_size);

  const uint32_t in_size = tiling.size;
  const uint32_t scan_tile_size = tiling.scan_tile_size;
  const uint32_t compress_tile_size = tiling.compress_tile_size;

  _run_compress<half>(x, mask, z, lower, usrWorkspace, in_size, scan_tile_size,
                      compress_tile_size);
}

extern "C" __global__ __aicore__ void compress_pos_fp16(
    GM_ADDR vec_in, GM_ADDR mask, GM_ADDR pos, GM_ADDR vec_out,
    GM_ADDR workspace, GM_ADDR tilingGm) {
  CompressTiling tiling;
  CopyTiling(&tiling, tilingGm);

  const uint32_t in_size = tiling.size;
  const uint32_t scan_tile_size = tiling.scan_tile_size;
  const uint32_t compress_tile_size = tiling.compress_tile_size;

  if ASCEND_IS_AIV {
    SyncAll<true /*isAIVOnly*/>();

    KernelCompress<half> op(in_size, compress_tile_size);
    op.Init(vec_in, mask, pos, vec_out);
    op.Process();
  }
}

extern "C" __global__ __aicore__ void compress_fp32(GM_ADDR x, GM_ADDR mask,
                                                    GM_ADDR z,
                                                    GM_ADDR workspace,
                                                    GM_ADDR tilingGm) {
  CompressTiling tiling;
  CopyTiling(&tiling, tilingGm);

  GM_ADDR const usrWorkspace = AscendC::GetUserWorkspace(workspace);
  GM_ADDR const lower = load_tril_matrix<int8_t>(tiling.scan_tile_size);

  const uint32_t in_size = tiling.size;
  const uint32_t scan_tile_size = tiling.scan_tile_size;
  const uint32_t compress_tile_size = tiling.compress_tile_size;

  _run_compress<float>(x, mask, z, lower, usrWorkspace, in_size, scan_tile_size,
                       compress_tile_size);
}

extern "C" __global__ __aicore__ void compress_pos_fp32(
    GM_ADDR vec_in, GM_ADDR mask, GM_ADDR pos, GM_ADDR vec_out,
    GM_ADDR workspace, GM_ADDR tilingGm) {
  CompressTiling tiling;
  CopyTiling(&tiling, tilingGm);

  const uint32_t in_size = tiling.size;
  const uint32_t scan_tile_size = tiling.scan_tile_size;
  const uint32_t compress_tile_size = tiling.compress_tile_size;

  if ASCEND_IS_AIV {
    SyncAll<true /*isAIVOnly*/>();

    KernelCompress<float> op(in_size, compress_tile_size);
    op.Init(vec_in, mask, pos, vec_out);
    op.Process();
  }
}
