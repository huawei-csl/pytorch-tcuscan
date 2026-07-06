/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file spmv.cpp
 * @brief Entrypoint for SpMV kernel.
 */

#include "kernels/constants.h"
#include "kernels/kernel_csr_gather.h"
#include "kernels/kernel_row_scan.h"
#include "kernels/kernel_seg_sum_vec_revert.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_spmv.h"

using namespace AscendC;
using namespace tcuscan;

/**
 * @brief Run the SpMV (Sparse Matrix-Vector multiplication) kernel.
 *
 * @tparam T input data type
 *
 * @param [in] vec_in Pointer to the input vector (sparse matrix non-zero
 * values).
 * @param [in] cols_in Pointer to the CSR column indices array.
 * @param [in] segm_ind_in Pointer to the segment indices vector.
 * @param [in] x_in Pointer to the dense input vector.
 * @param [in] upper Pointer to an upper-triangular all-ones square matrix of
 * size \f$\textit{tile\_len} \times \textit{tile\_len}\f$.
 * @param [in] segm_offset_per_block Pointer to segment index offset per block.
 * @param [out] vec_out Pointer to the output vector.
 * @param [in,out] workspace Pointer to a memory region used as workspace.
 * @param [in] vec_len Input vector length (number of non-zeros).
 * @param [in] num_segments Number of segments.
 * @param [in] x_len Length of the dense input vector.
 * @param [in] tile_len Tile size used for the matrix multiplication step.
 * @param [in] block_len Block length assigned to each AI Core group.
 */
template <typename T>
__aicore__ inline void run_spmv_v2(
    GM_ADDR vec_in, GM_ADDR cols_in, GM_ADDR segm_ind_in, GM_ADDR x_in,
    GM_ADDR upper_in, GM_ADDR segm_offset_per_block, GM_ADDR vec_out,
    GM_ADDR workspace, uint32_t vec_len, uint32_t num_segments, uint32_t x_len,
    uint32_t tile_len, uint32_t block_len) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<T>;

  const uint32_t align_size = tile_len * tile_len;
  const uint32_t padded_vec_len = scalar::AlignUp(vec_len, align_size);
  const uint32_t pad_size = padded_vec_len * sizeof(T);

  GM_ADDR const csr_products_ws = workspace;
  GM_ADDR const spec_block_scan_ws = workspace + pad_size;

  const uint32_t csr_gather_tile_len = align_size > 1024 ? 1024 : align_size;
  run_csr_gather<T, false>(vec_in, cols_in, x_in, csr_products_ws, vec_len,
                           x_len, csr_gather_tile_len);

  sync::SyncGroup<sync::GroupSyncDirection::FULL>();
  sync::SyncAllCores();

  if ASCEND_IS_AIC {
    KernelRowScan<T, true> op_cube(tile_len, tile_len, padded_vec_len);
    op_cube.Init(csr_products_ws, upper_in, spec_block_scan_ws);
    op_cube.Process();
  }

  if ASCEND_IS_AIV {
    const uint32_t num_blocks = AscendC::GetBlockNum();

    // id is the id of each AI Core (2 AIVs and 1 AIC core)
    const auto id = GetBlockIdx() / GetTaskRation();
    int32_t segm_ind_offset =
        scalar::GetGMValue<int32_t>(segm_offset_per_block, id, num_blocks + 1);
    const int32_t next_offset = scalar::GetGMValue<int32_t>(
        segm_offset_per_block, id + 1, num_blocks + 1);
    const int32_t num_segments_per_block = next_offset - segm_ind_offset;

    // The boundaries of each segment must overlap
    if (id > 0) {
      segm_ind_offset--;
    }

    // Each AI Core group is responsible (offsets) starting from `block_len`
    const uint32_t block_vec_offset = id * block_len;
    if (block_vec_offset >= vec_len) {
      return;
    }
    const bool is_overflow_block = block_vec_offset + block_len > vec_len;
    if (is_overflow_block) {
      block_len = vec_len - block_vec_offset;
    }

    KernelSegSumVecRevert<OutputT, true, true> op(
        block_len, num_segments_per_block, tile_len, block_vec_offset);
    op.Init(spec_block_scan_ws, segm_ind_in + segm_ind_offset * sizeof(int32_t),
            vec_out + segm_ind_offset * sizeof(OutputT));
    op.Process();
  }
}

/**
 * @brief Run the SpMV v2 kernel with half/float16 dtype.
 *
 * The segment indices format follows the scipy Compressed Sparse Row Matrix
 * convention
 * (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).
 *
 * @param [in] vec_in Pointer to the sparse matrix non-zero values.
 * @param [in] cols_in Pointer to the CSR column indices array.
 * @param [in] indptr Pointer to the segment indices vector (CSR row pointers).
 * @param [in] x_in Pointer to the dense input vector.
 * @param [in] segment_offsets Pointer to the segment offset per block.
 * @param [out] vec_out Pointer to the output vector.
 * @param [in,out] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void spmv_v2_fp16(
    GM_ADDR vec_in, GM_ADDR cols_in, GM_ADDR indptr, GM_ADDR x_in,
    GM_ADDR segment_offsets, GM_ADDR vec_out, GM_ADDR workspace,
    GM_ADDR tiling_gm) {
  tcuscan::SpMVTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.nnz;
  const uint32_t num_segments = tiling.num_segments;
  const uint32_t x_len = tiling.x_len;
  const uint32_t tile_len = tiling.tile_len;
  const uint32_t block_len = tiling.block_len;

  GM_ADDR const upper = load_tril_matrix<half>(tile_len);

  run_spmv_v2<half>(vec_in, cols_in, indptr, x_in, upper, segment_offsets,
                    vec_out, workspace, vec_len, num_segments, x_len, tile_len,
                    block_len);
}

/**
 * @brief Run the SpMV v2 kernel with float32 dtype.
 *
 * The segment indices format follows the scipy Compressed Sparse Row Matrix
 * convention
 * (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).
 *
 * @param [in] vec_in Pointer to the sparse matrix non-zero values.
 * @param [in] cols_in Pointer to the CSR column indices array.
 * @param [in] indptr Pointer to the segment indices vector (CSR row pointers).
 * @param [in] x_in Pointer to the dense input vector.
 * @param [in] segment_offsets Pointer to the segment offset per block.
 * @param [out] vec_out Pointer to the output vector.
 * @param [in,out] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void spmv_v2_fp32(
    GM_ADDR vec_in, GM_ADDR cols_in, GM_ADDR indptr, GM_ADDR x_in,
    GM_ADDR segment_offsets, GM_ADDR vec_out, GM_ADDR workspace,
    GM_ADDR tiling_gm) {
  tcuscan::SpMVTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.nnz;
  const uint32_t num_segments = tiling.num_segments;
  const uint32_t x_len = tiling.x_len;
  const uint32_t tile_len = tiling.tile_len;
  const uint32_t block_len = tiling.block_len;

  GM_ADDR const upper = load_tril_matrix<float>(tile_len);

  run_spmv_v2<float>(vec_in, cols_in, indptr, x_in, upper, segment_offsets,
                     vec_out, workspace, vec_len, num_segments, x_len, tile_len,
                     block_len);
}
