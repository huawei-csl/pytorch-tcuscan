/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file topk.cpp
 * @brief Entrypoint for parallel topk kernel.
 */

#include "kernels/constants.h"
#include "kernels/kernel_topk.h"
#include "kernels/kernel_topk_pivot.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_topk.h"
#include "tiling/tiling_topk_pivot.h"

/**
 * @brief Run the `topk` (int16) kernel.
 *
 * @param [in] vec_in Pointer to input vector.
 * @param [in] vec_out Pointer to output vector.
 * @param [in] indices_out Pointer to output indices vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_ptr Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void topk_int16(GM_ADDR vec_in,
                                                 GM_ADDR vec_out,
                                                 GM_ADDR indices_out,
                                                 GM_ADDR workspace,
                                                 GM_ADDR tiling_ptr) {
  TopKTiling tiling;
  tiling::GetTilingData(&tiling, tiling_ptr);

  GM_ADDR const usrWorkspace = AscendC::GetUserWorkspace(workspace);
  GM_ADDR const lower = load_tril_matrix<int8_t>(tiling.matmul_size);

  const int32_t x_min = tiling.x_min.value_i32;
  const int32_t x_max = tiling.x_max.value_i32;

  run_topk<int16_t>(vec_in, tiling.k, vec_out, indices_out, lower, usrWorkspace,
                    x_min, x_max, tiling.num_elems, tiling.vec_tile_size,
                    tiling.matmul_size);
}

/**
 * @brief Run the `topk` (fp16) kernel.
 *
 * @param [in] vec_in Pointer to input vector.
 * @param [in] vec_out Pointer to output vector.
 * @param [in] indices_out Pointer to output indices vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_ptr Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void topk_fp16(GM_ADDR vec_in, GM_ADDR vec_out,
                                                GM_ADDR indices_out,
                                                GM_ADDR workspace,
                                                GM_ADDR tiling_ptr) {
  TopKTiling tiling;
  tiling::GetTilingData(&tiling, tiling_ptr);

  GM_ADDR const usrWorkspace = AscendC::GetUserWorkspace(workspace);
  GM_ADDR const lower = load_tril_matrix<half>(tiling.matmul_size);

  const half x_min = static_cast<half>(tiling.x_min.value_fp32);
  const half x_max = static_cast<half>(tiling.x_max.value_fp32);

  run_topk<half>(vec_in, tiling.k, vec_out, indices_out, lower, usrWorkspace,
                 x_min, x_max, tiling.num_elems, tiling.vec_tile_size,
                 tiling.matmul_size);
}

/**
 * @brief Run the `topk_pivot` (fp16) kernel.
 *
 * @param [in] vec_in Pointer to input vector.
 * @param [in] vec_out Pointer to output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_ptr Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void topk_pivot_fp16(GM_ADDR vec_in,
                                                      GM_ADDR vec_out,
                                                      GM_ADDR workspace,
                                                      GM_ADDR tiling_ptr) {
  (void)workspace;
  TopKPivotTiling tiling;
  tiling::GetTilingData(&tiling, tiling_ptr);

  run_pivot_topk_estimator<true, half>(vec_in, vec_out, tiling.num_elems,
                                       tiling.tile_len, tiling.k);
}
