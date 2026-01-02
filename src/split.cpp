/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file split.cpp
 * @brief Entrypoint for parallel split kernel.
 */

#include "kernels/constants.h"
#include "kernels/kernel_split.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_split.h"

/**
 * @brief Run the `split_uint16` kernel.
 *
 * @param [in] in Pointer to input vector.
 * @param [in] mask Pointer to mask vector.
 * @param [in] out Pointer to output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to tiling vector.
 */
extern "C" __global__ __aicore__ void split_uint16(GM_ADDR in, GM_ADDR mask,
                                                   GM_ADDR out,
                                                   GM_ADDR workspace,
                                                   GM_ADDR tiling) {
  constexpr bool zeros_first = false;
  tcuscan::SplitTiling tiling_data;
  GetTilingData(&tiling_data, tiling);

  GM_ADDR const usrWorkspace = AscendC::GetUserWorkspace(workspace);

  tcuscan::run_split_uint16(in, mask, out, usrWorkspace, tiling_data.num_elems,
                            tiling_data.vec_tile_size, zeros_first);
}

/**
 * @brief Run the `split_ind_uint16` kernel.
 *
 * @param [in] vec_in Pointer to input vector.
 * @param [in] mask_in Pointer to mask vector.
 * @param [in] indices_in Pointer to input indices vector.
 * @param [in] vec_out Pointer to output vector.
 * @param [in] indices_out Pointer to output indices vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to tiling vector.
 */
extern "C" __global__ __aicore__ void split_ind_uint16(
    GM_ADDR vec_in, GM_ADDR mask_in, GM_ADDR indices_in, GM_ADDR vec_out,
    GM_ADDR indices_out, GM_ADDR workspace, GM_ADDR tiling) {
  constexpr bool zeros_first = false;
  tcuscan::SplitTiling tiling_data;
  GetTilingData(&tiling_data, tiling);

  GM_ADDR const usrWorkspace = AscendC::GetUserWorkspace(workspace);

  tcuscan::run_split_ind_uint16(
      vec_in, mask_in, indices_in, vec_out, indices_out, usrWorkspace,
      tiling_data.num_elems, tiling_data.vec_tile_size, zeros_first);
}
