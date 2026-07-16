/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * @file searchsorted.cpp
 * @brief Entrypoint for the searchsorted kernel operation.
 */

#include "kernels/kernel_searchsorted.h"
#include "tiling/tiling_searchsorted.h"

/**
 * @brief Run the `searchsorted` kernel with int32 data.
 *
 * @param [in] sorted Pointer to the sorted haystack.
 * @param [in] values Pointer to the needle values.
 * @param [in] out Pointer to the output indices.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void searchsorted_int32(GM_ADDR sorted,
                                                         GM_ADDR values,
                                                         GM_ADDR out,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tilingGm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::SearchsortedTiling tiling;
  GetTilingData(&tiling, tilingGm);

  tcuscan::run_searchsorted<true, int32_t>(sorted, values, out, tiling.data_len,
                                           tiling.num_values);
}

/**
 * @brief Call the `searchsorted` kernel for INT32 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream CUDA stream.
 * @param [in] sorted Pointer to an input buffer.
 * @param [in] values Pointer to an input buffer.
 * @param [in] out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" void launch_searchsorted_int32(uint32_t blockDim, void* stream,
                                          uint8_t* sorted, uint8_t* values,
                                          uint8_t* out, uint8_t* workspace,
                                          uint8_t* tilingGm) {
  searchsorted_int32<<<blockDim, nullptr, stream>>>(sorted, values, out,
                                                    workspace, tilingGm);
}
