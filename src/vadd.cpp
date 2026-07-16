/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file vadd.cpp
 * @brief Entrypoint for `vadd` kernel.
 */

#include "kernels/kernel_vadd.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_vadd.h"

/**
 * @brief Run the `vadd` kernel.
 *
 * @param [in] x Pointer to the input vector.
 * @param [in] y Pointer to the input vector.
 * @param [in] z Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void vadd_fp16(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                GM_ADDR workspace,
                                                GM_ADDR tilingGm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  (void)workspace;
  tcuscan::VaddTiling tiling_data;
  GetTilingData(&tiling_data, tilingGm);

  if ASCEND_IS_AIV {
    tcuscan::KernelAdd op(tiling_data.vec_len, tiling_data.tile_len);
    op.Init(x, y, z);
    op.Process();
  }
}

/**
 * @brief Call the `vadd` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream CUDA stream.
 * @param [in] x Pointer to the input vector.
 * @param [in] y Pointer to the input vector.
 * @param [in] z Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to the tiling buffer.
 */
extern "C" void launch_vadd_fp16(uint32_t blockDim, void* stream, uint8_t* x,
                                 uint8_t* y, uint8_t* z, uint8_t* workspace,
                                 uint8_t* tiling) {
  vadd_fp16<<<blockDim, nullptr, stream>>>(x, y, z, workspace, tiling);
}
