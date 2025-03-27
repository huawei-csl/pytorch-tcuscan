/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file compress.cpp
 * @brief Multi-core compress approach.
 */

#include "kernel_utils.h"
#include "kernels/constants.h"
#include "kernels/kernel_compress.h"
#include "lib/matmul_intf.h"
#include "tiling/tiling_compress.h"

using namespace AscendC;

/**
 * @brief Compress kernel for input dtype fp16
 *
 * @param x Input data vector
 * @param mask Input mask vector
 * @param z Output vector
 * @param workspace Pointer to workspace.
 * @param tilingGm Pointer to tiling structure.
 */
extern "C" __global__ __aicore__ void compress_fp16(GM_ADDR x, GM_ADDR mask,
                                                    GM_ADDR z,
                                                    GM_ADDR workspace,
                                                    GM_ADDR tilingGm) {
  CompressTiling tiling;
  tiling::GetTilingData(&tiling, tilingGm);

  GM_ADDR const usrWorkspace = AscendC::GetUserWorkspace(workspace);
  GM_ADDR const lower = load_tril_matrix<int8_t>(tiling.scan_tile_size);

  const uint32_t in_size = tiling.size;
  const uint32_t scan_tile_size = tiling.scan_tile_size;
  const uint32_t compress_tile_size = tiling.compress_tile_size;

  _run_compress<half>(x, mask, z, lower, usrWorkspace, in_size, scan_tile_size,
                      compress_tile_size);
}

/**
 * @brief Compress kernel with positions for dtype fp16
 *
 * @param vec_in Input data vector
 * @param mask Input mask vector
 * @param pos Input mask vector
 * @param vec_out Output vector
 * @param workspace Pointer to workspace.
 * @param tilingGm Pointer to tiling structure.
 */
extern "C" __global__ __aicore__ void compress_pos_fp16(
    GM_ADDR vec_in, GM_ADDR mask, GM_ADDR pos, GM_ADDR vec_out,
    GM_ADDR workspace, GM_ADDR tilingGm) {
  CompressTiling tiling;
  tiling::GetTilingData(&tiling, tilingGm);

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

/**
 * @brief Compress kernel for dtype fp32
 *
 * @param x Input data vector
 * @param mask Input mask vector
 * @param z Output vector
 * @param workspace Pointer to workspace.
 * @param tilingGm Pointer to tiling structure.
 */
extern "C" __global__ __aicore__ void compress_fp32(GM_ADDR x, GM_ADDR mask,
                                                    GM_ADDR z,
                                                    GM_ADDR workspace,
                                                    GM_ADDR tilingGm) {
  CompressTiling tiling;
  tiling::GetTilingData(&tiling, tilingGm);

  GM_ADDR const usrWorkspace = AscendC::GetUserWorkspace(workspace);
  GM_ADDR const lower = load_tril_matrix<int8_t>(tiling.scan_tile_size);

  const uint32_t in_size = tiling.size;
  const uint32_t scan_tile_size = tiling.scan_tile_size;
  const uint32_t compress_tile_size = tiling.compress_tile_size;

  _run_compress<float>(x, mask, z, lower, usrWorkspace, in_size, scan_tile_size,
                       compress_tile_size);
}

/**
 * @brief Compress kernel with positions for dtype fp32
 *
 * @param vec_in Input data vector
 * @param mask Input mask vector
 * @param pos Input mask vector
 * @param vec_out Output vector
 * @param workspace Pointer to workspace.
 * @param tilingGm Pointer to tiling structure.
 */
extern "C" __global__ __aicore__ void compress_pos_fp32(
    GM_ADDR vec_in, GM_ADDR mask, GM_ADDR pos, GM_ADDR vec_out,
    GM_ADDR workspace, GM_ADDR tilingGm) {
  CompressTiling tiling;
  tiling::GetTilingData(&tiling, tilingGm);

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
