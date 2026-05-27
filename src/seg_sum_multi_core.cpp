/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file seg_sum_multi_core.cpp
 * @brief Entrypoint for segmented sum multi core kernel.
 */

#include "kernels/constants.h"
#include "kernels/kernel_pad.h"
#include "kernels/kernel_seg_sum_single_core.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_seg_sum_multi_core.h"

using namespace AscendC;
using namespace tcuscan;

/**
 * @brief Run the `seg_sum_multi_core` kernel.
 *
 * @tparam T input data type
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] upper Pointer to an upper-triangular all-ones square matrix of
 * size \f$\textit{matmul_size}\f$.
 * @param [in] segm_ind_in Pointer to the segment indices vector.
 * @param [in] bstart_in Pointer to start indices per block.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] workspace Pointer to a memory region used as workspace.
 * @param [in] vec_len Input vector length.
 * @param [in] num_segments Number of segments.
 * @param [in] tile_len Tile length.
 */
template <typename T>
__aicore__ inline void run_seg_sum_multi_core(
    GM_ADDR vec_in, GM_ADDR upper, GM_ADDR segm_ind_in, GM_ADDR bstart_in,
    GM_ADDR vec_out, GM_ADDR workspace, uint32_t vec_len, uint32_t num_segments,
    uint32_t tile_len) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<T>;

  const uint32_t align_size = tile_len * tile_len;
  const uint32_t padded_vec_len = scalar::AlignUp(vec_len, align_size);
  const uint32_t pad_size = padded_vec_len * sizeof(T);

  GM_ADDR const padded_input = workspace;
  GM_ADDR const spec_block_scan = workspace + pad_size;

  const uint32_t num_blocks = AscendC::GetBlockNum();
  const uint32_t ell = scalar::CeilDiv(vec_len, num_blocks);

  const auto id = AscendC::GetBlockIdx();
  const uint32_t p = scalar::GetGMValue<int32_t>(bstart_in, id, num_blocks);
  const uint32_t q = scalar::GetGMValue<int32_t>(bstart_in, id + 1, num_blocks);

  const uint32_t s = id * ell;
  const uint32_t e = (id + 1) * ell > vec_len ? vec_len : (id + 1) * ell;

  // Call single core segmented per AI core
  // First block needs special offsetting
  if (id > 0) {
    run_seg_sum_single_core<T, true /* UseAtomicWrite */>(
        vec_in + (s - 1) * sizeof(T), upper, segm_ind_in + p * sizeof(int32_t),
        vec_out + p * sizeof(OutputT), workspace /* TODO*/, e - s, q - p + 1,
        tile_len);
  } else {
    run_seg_sum_single_core<T, true /* UseAtomicWrite */>(
        vec_in + s * sizeof(T), upper, segm_ind_in + p * sizeof(int32_t),
        vec_out + p * sizeof(OutputT), workspace /* TODO*/, e - s, q - p + 1,
        tile_len);
  }
}

/**
 * @brief Run the `seg_sum_multi_core` kernel with half/float16 dtype.
 *
 * The segment indices format follows the scipy Compressed Sparse Row Matrix
 * convention
 * (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] indptr Pointer to the segment indices vector.
 * @param [in] bstart Pointer to the segment bstart vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void seg_sum_multi_core_fp16(
    GM_ADDR vec_in, GM_ADDR indptr, GM_ADDR bstart, GM_ADDR vec_out,
    GM_ADDR workspace, GM_ADDR tiling_gm) {
  tcuscan::SegSumMultiCoreTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t num_segments = tiling.num_segments;
  const uint32_t matmul_size = tiling.tile_len;

  GM_ADDR const lower = load_tril_matrix<half>(matmul_size);

  run_seg_sum_multi_core<half>(vec_in, lower, indptr, bstart, vec_out,
                               workspace, vec_len, num_segments, matmul_size);
}
