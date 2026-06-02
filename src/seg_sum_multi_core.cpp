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
 * @param [in] segm_len_per_block_in Pointer to segment lengths per block.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] workspace Pointer to a memory region used as workspace.
 * @param [in] vec_len Input vector length.
 * @param [in] num_segments Number of segments.
 * @param [in] block_len Block length.
 */
template <typename T>
__aicore__ inline void run_seg_sum_multi_core(
    GM_ADDR vec_in, GM_ADDR upper, GM_ADDR segm_ind_in, GM_ADDR bstart_in,
    GM_ADDR segm_len_per_block_in, GM_ADDR vec_out, GM_ADDR workspace,
    uint32_t vec_len, uint32_t num_segments, uint32_t tile_len,
    uint32_t block_len) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<T>;

  const uint32_t num_blocks = AscendC::GetBlockNum();

  const auto id = AscendC::GetBlockIdx();
  const int32_t segment_block_offset =
      scalar::GetGMValue<int32_t>(bstart_in, id, num_blocks);
  const int32_t segment_block_len =
      scalar::GetGMValue<int32_t>(segm_len_per_block_in, id, num_blocks);

  if ASCEND_IS_AIC {
    KernelRowScan<T> op_cube(tile_len, tile_len, vec_len);
    op_cube.Init(vec_in, upper, vec_out);
    op_cube.Process();
  }

  scalar::SetGMValue<int32_t>(workspace, id, segment_block_len, num_segments);

  /*
    if (false) {
      run_seg_sum_single_core_aligned<T, true>(
          vec_in, upper, segm_ind_in + p * sizeof(int32_t),
          vec_out + p * sizeof(OutputT),
          workspace + id * workspace_size_per_block, block_len, segment_len,
          tile_len);
    } else {
      run_seg_sum_single_core_aligned<T, true>(vec_in, upper, segm_ind_in,
                                               vec_out, workspace, block_len,
                                               segment_len, tile_len);
    }
                                               */
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
 * @param [in] segm_len_per_block Pointer to the segment length per block
 * vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void seg_sum_multi_core_fp16(
    GM_ADDR vec_in, GM_ADDR indptr, GM_ADDR bstart, GM_ADDR segm_len_per_block,
    GM_ADDR vec_out, GM_ADDR workspace, GM_ADDR tiling_gm) {
  tcuscan::SegSumMultiCoreTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t num_segments = tiling.num_segments;
  const uint32_t matmul_size = tiling.tile_len;
  const uint32_t block_len = tiling.block_len;

  GM_ADDR const lower = load_tril_matrix<half>(matmul_size);

  run_seg_sum_multi_core<half>(vec_in, lower, indptr, bstart,
                               segm_len_per_block, vec_out, workspace, vec_len,
                               num_segments, matmul_size, block_len);
}
