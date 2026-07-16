/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file copy.cpp
 * @brief Kernel implementing a copy operation.
 */

#include "kernels/kernel_copy.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_copy.h"

/**
 * @brief Run the `copy_fp16` kernel.
 *
 * @param [in] in Pointer to input vector.
 * @param [in] out Pointer to output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to tiling vector.
 */
extern "C" __global__ __aicore__ void copy_fp16(GM_ADDR in, GM_ADDR out,
                                                GM_ADDR workspace,
                                                GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::CopyTiling tiling_data;
  GetTilingData(&tiling_data, tiling);
  tcuscan::run_copy<half>(in, out, tiling_data.num_elems,
                          tiling_data.tile_size);
}

/**
 * @brief Run the `copy_fp32` kernel.
 *
 * @param [in] in Pointer to input vector.
 * @param [in] out Pointer to output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to tiling vector.
 */
extern "C" __global__ __aicore__ void copy_fp32(GM_ADDR in, GM_ADDR out,
                                                GM_ADDR workspace,
                                                GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::CopyTiling tiling_data;
  GetTilingData(&tiling_data, tiling);
  tcuscan::run_copy<float>(in, out, tiling_data.num_elems,
                           tiling_data.tile_size);
}

/**
 * @brief Call the `copy` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream CUDA stream.
 * @param [in] in Pointer to an input buffer.
 * @param [in] out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to the tiling buffer.
 */
extern "C" void launch_copy_fp16(uint32_t blockDim, void* stream, uint8_t* in,
                                 uint8_t* out, uint8_t* workspace,
                                 uint8_t* tiling) {
  copy_fp16<<<blockDim, nullptr, stream>>>(in, out, workspace, tiling);
}

/**
 * @brief Call the `copy` kernel for FP32 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream CUDA stream.
 * @param [in] in Pointer to an input buffer.
 * @param [in] out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to the tiling buffer.
 */
extern "C" void launch_copy_fp32(uint32_t blockDim, void* stream, uint8_t* in,
                                 uint8_t* out, uint8_t* workspace,
                                 uint8_t* tiling) {
  copy_fp32<<<blockDim, nullptr, stream>>>(in, out, workspace, tiling);
}
