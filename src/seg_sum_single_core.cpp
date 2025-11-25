/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file seg_sum_single_core.cpp
 * @brief Entrypoint for segmented sum single core kernel operation.
 */

#include "kernels/constants.h"
#include "kernels/kernel_pad.h"
#include "kernels/kernel_row_scan.h"
#include "kernels/kernel_seg_sum_vec_revert.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_seg_sum_single_core.h"

using namespace AscendC;
using namespace kernel_utils;

/**
 * @brief Run the `seg_sum_single_core` kernel.
 *
 * @tparam T input data type
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] upper Pointer to an upper-triangular all-ones square matrix of
 * size \f$\textit{matmul_size}\f$.
 * @param [in] segm_ind_in Pointer to the segment indices vector.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] workspace Pointer to a memory region used as workspace.
 * @param [in] vec_len Input vector length.
 * @param [in] num_segments Number of segments.
 * @param [in] tile_len Tile length.
 */
template <typename T>
__aicore__ inline void run_seg_sum_single_core(
    GM_ADDR vec_in, GM_ADDR upper, GM_ADDR segm_ind_in, GM_ADDR vec_out,
    GM_ADDR workspace, uint32_t vec_len, uint32_t num_segments,
    uint32_t tile_len) {
  using OutputT = kernel_utils::cube_unit::CubeOutType_t<T>;

  const uint32_t align_size = tile_len * tile_len;
  const uint32_t padded_vec_len = scalar::AlignUp(vec_len, align_size);
  const uint32_t pad_size = padded_vec_len * sizeof(T);

  GM_ADDR const padded_input = workspace;
  GM_ADDR const spec_block_scan = workspace + pad_size;

  run_pad_kernel<T, false>(vec_in, padded_input, vec_len, align_size);

  sync::SyncGroup<sync::GroupSyncDirection::FULL>();

  if ASCEND_IS_AIC {
    KernelRowScan<T, true> op_cube(tile_len, tile_len, padded_vec_len);
    op_cube.Init(padded_input, upper, spec_block_scan);
    op_cube.Process();
  }

  if ASCEND_IS_AIV {
    KernelSegSumVecRevert<OutputT, true> op(vec_len, num_segments, tile_len);
    op.Init(spec_block_scan, segm_ind_in, vec_out);
    op.Process();
  }
}

/**
 * @brief Run the `seg_sum_single_core` kernel with half/float16 dtype.
 *
 * The segment indices format follows the scipy Compressed Sparse Row Matrix
 * convention
 * (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] indptr Pointer to the segment indices vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void seg_sum_single_core_fp16(
    GM_ADDR vec_in, GM_ADDR indptr, GM_ADDR vec_out, GM_ADDR workspace,
    GM_ADDR tiling_gm) {
  tcuscan::SegSumSingleCoreTiling tiling;
  tiling::GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t num_segments = tiling.num_segments;
  const uint32_t matmul_size = tiling.tile_len;

  GM_ADDR const lower = load_tril_matrix<half>(matmul_size);

  run_seg_sum_single_core<half>(vec_in, lower, indptr, vec_out, workspace,
                                vec_len, num_segments, matmul_size);
}

/**
 * @brief Run the `seg_sum_single_core` kernel with `int8_t` dtype.
 *
 * The segment indices format follows the scipy Compressed Sparse Row Matrix
 * convention
 * (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] indptr Pointer to the segment indices vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void seg_sum_single_core_int8(
    GM_ADDR vec_in, GM_ADDR indptr, GM_ADDR vec_out, GM_ADDR workspace,
    GM_ADDR tiling_gm) {
  tcuscan::SegSumSingleCoreTiling tiling;
  tiling::GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t num_segments = tiling.num_segments;
  const uint32_t matmul_size = tiling.tile_len;

  GM_ADDR const lower = load_tril_matrix<int8_t>(matmul_size);

  run_seg_sum_single_core<int8_t>(vec_in, lower, indptr, vec_out, workspace,
                                  vec_len, num_segments, matmul_size);
}
