/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file compress.cpp
 * @brief Multi-core compress approach.
 */

#include "kernels/constants.h"
#include "kernels/kernel_compress.h"
#include "kernels/tcuscan_utils.h"
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
  tcuscan::CompressTiling tiling;
  GetTilingData(&tiling, tilingGm);

  const uint32_t in_size = tiling.size;
  const uint32_t scan_tile_size = tiling.scan_tile_size;

  tcuscan::run_compress<half>(x, mask, z, workspace, in_size, scan_tile_size);
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
  tcuscan::CompressTiling tiling;
  GetTilingData(&tiling, tilingGm);

  const uint32_t in_size = tiling.size;
  const uint32_t scan_tile_size = tiling.scan_tile_size;

  tcuscan::run_compress<float>(x, mask, z, workspace, in_size, scan_tile_size);
}
