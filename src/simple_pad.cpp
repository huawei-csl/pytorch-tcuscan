/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file simple_pad.cpp
 * @brief Kernel implementing the Vector simple pad kernel operation.
 */

#include "kernels/ascendc_kernel_operator.h"
#include "kernels/kernel_simple_pad.h"
#include "tiling/tiling_simple_pad.h"

/**
 * @brief Run the `simple_pad_fp16` kernel.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace workspace param. Zero in this case.
 * @param [in] tiling Pointer to the tiling structure.
 */

extern "C" __global__ __aicore__ void simple_pad_fp16(GM_ADDR vec_in,
                                                      GM_ADDR vec_out,
                                                      GM_ADDR workspace,
                                                      GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::SimplePadTiling tiling_data;
  GetTilingData(&tiling_data, tiling);

  tcuscan::run_simple_pad<false, half>(vec_in, vec_out, tiling_data.num_elems,
                                       tiling_data.align_len);
}

/**
 * @brief Call the `simple_pad` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream CUDA stream.
 * @param [in] vec_in Pointer to an input buffer.
 * @param [in] vec_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to the tiling buffer.
 */
extern "C" void launch_simple_pad_fp16(uint32_t blockDim, void* stream,
                                       uint8_t* vec_in, uint8_t* vec_out,
                                       uint8_t* workspace, uint8_t* tiling) {
  simple_pad_fp16<<<blockDim, nullptr, stream>>>(vec_in, vec_out, workspace,
                                                 tiling);
}
