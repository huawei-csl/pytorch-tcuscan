/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file gather_spmv_diff.cpp
 * @brief Entrypoint for the gather_spmv kernel with a fused `torch::diff`.
 */

#include "kernels/kernel_gather_spmv.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_gather_spmv.h"

namespace {
/**
 * @brief Shared body for the gather-with-fused-diff entrypoints.
 *
 * The gather itself is a type-agnostic element move, but the fused diff is
 * arithmetic and must run in the data's native type: `float` for fp16/fp32
 * SpMV (scan promotes to fp32) and `int32_t` for int16 SpMV (scan promotes to
 * int32), matching `torch::diff`.
 */
template <typename DataType>
__aicore__ inline void gather_spmv_diff_impl(GM_ADDR values_in, GM_ADDR cols_in,
                                             GM_ADDR vec_out, GM_ADDR workspace,
                                             GM_ADDR tilingGm) {
  tcuscan::GatherSpmvTiling tiling;
  GetTilingData(&tiling, tilingGm);

  GM_ADDR gathered_scratch = AscendC::GetUserWorkspace(workspace);

  tcuscan::run_gather_spmv</*ForceMixMode=*/true, /*EnableDiff=*/true,
                           DataType>(values_in, cols_in, vec_out,
                                     gathered_scratch, tiling.idx_len,
                                     tiling.val_len, tiling.tile_len);
}
}  // namespace

/**
 * @brief Gather SpMV kernel with fused diff (float32 data).
 *
 * Equivalent to `torch::diff(gather_spmv(values_in, cols_in))` computed in a
 * single launch. The output vector has length `idx_len - 1`.
 *
 * @param values_in Input data vector.
 * @param cols_in Input column indices of CSR sparse format.
 * @param vec_out Output vector (length `idx_len - 1`).
 * @param workspace Pointer to workspace; its user portion stages the
 * intermediate full-length gather (length `idx_len`).
 * @param tilingGm Pointer to tiling structure.
 */
extern "C" __global__ __aicore__ void gather_spmv_diff_fp32(GM_ADDR values_in,
                                                            GM_ADDR cols_in,
                                                            GM_ADDR vec_out,
                                                            GM_ADDR workspace,
                                                            GM_ADDR tilingGm) {
  gather_spmv_diff_impl<float>(values_in, cols_in, vec_out, workspace,
                               tilingGm);
}

/**
 * @brief Gather SpMV kernel with fused diff (int32 data, for int16 SpMV).
 *
 * @param values_in Input data vector.
 * @param cols_in Input column indices of CSR sparse format.
 * @param vec_out Output vector (length `idx_len - 1`).
 * @param workspace Pointer to workspace; its user portion stages the
 * intermediate full-length gather (length `idx_len`).
 * @param tilingGm Pointer to tiling structure.
 */
extern "C" __global__ __aicore__ void gather_spmv_diff_int32(GM_ADDR values_in,
                                                             GM_ADDR cols_in,
                                                             GM_ADDR vec_out,
                                                             GM_ADDR workspace,
                                                             GM_ADDR tilingGm) {
  gather_spmv_diff_impl<int32_t>(values_in, cols_in, vec_out, workspace,
                                 tilingGm);
}
