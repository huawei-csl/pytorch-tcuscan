/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
 *
 * @file kernel_compress_ind.h
 * @brief Kernel implementing a compress with returning indices operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "kernel_compress.h"
#include "kernel_reduce_tiles.h"
#include "kernel_where.h"
#include "tcuscan_utils.h"

using namespace AscendC;
using namespace tcuscan;

#pragma once

namespace tcuscan {

/**
 * @brief Run the `compress_ind` kernel. Compression/compaction that returns the
 * corresponding input indices. Support `fp16/half` and `float32` input data
 * types. Indices must be `int32_t` or `uint32_t`.
 *
 * @tparam T Data type of the input vector.
 *
 * @param [in] vec_in Pointer to input data vector.
 * @param [in] indices_in Pointer to input indices vector.
 * @param [in] mask Pointer to boolean flag/mask vector of dtype int8.
 * @param [in] vec_out Pointer to output data vector.
 * @param [in] indices_out Pointer to output data vector.
 * @param [in] workspace Pointer to workspace. The number of mask ones per
 * vector core is left at its start; see `run_compress`.
 * @param [in] vec_len Length of the input vector.
 * @param [in] tile_len Tile length.
 */
template <typename T>
__aicore__ inline void run_compress_ind(GM_ADDR vec_in, GM_ADDR indices_in,
                                        GM_ADDR mask, GM_ADDR vec_out,
                                        GM_ADDR indices_out, GM_ADDR workspace,
                                        uint32_t vec_len, uint32_t tile_len) {
  GM_ADDR const num_ones_per_block = workspace;
  const uint32_t block_len = tile_len * tile_len / 2;

  // Both compactions below need to know where each core's output starts,
  // which is the running total of mask ones before it. Reducing the mask here
  // rather than on the host keeps the count on the device, where it is only
  // ever used.
  if ASCEND_IS_AIV {
    KernelReduceTiles<int8_t> op_reduce(block_len, vec_len);
    op_reduce.Init(mask, num_ones_per_block);
    op_reduce.Process();
  }
  SyncAll<true /*isAIVOnly*/>();

  if ASCEND_IS_AIV {
    KernelCompress<T> data_op(vec_len, block_len);
    data_op.Init(vec_in, mask, num_ones_per_block, vec_out);
    data_op.Process();
  }
  SyncAll<true /*isAIVOnly*/>();

  if ASCEND_IS_AIV {
    KernelCompress<int32_t> indices_op(vec_len, block_len);
    indices_op.Init(indices_in, mask, num_ones_per_block, indices_out);
    indices_op.Process();
  }
}

/**
 * @brief Run the `compress_ind` kernel. Compression/compaction that returns the
 * corresponding input indices. Support `fp16/half` and `float32` input data
 * types. Returned indices have `int32_t` dtype.
 *
 * @tparam T Data type of the input vector.
 *
 * @param [in] vec_in Pointer to input data vector.
 * @param [in] mask_in Pointer to boolean flag/mask vector of dtype int8.
 * @param [in] vec_out Pointer to output data vector.
 * @param [in] indices_out Pointer to output data vector.
 * @param [in] workspace Pointer to workspace. The number of mask ones per
 * vector core is left at its start; see `run_compress`.
 * @param [in] vec_len Length of the input vector.
 * @param [in] tile_len Tile length.
 */
template <typename T>
__aicore__ inline void run_compress_ind_no_arange(
    GM_ADDR vec_in, GM_ADDR mask_in, GM_ADDR vec_out, GM_ADDR indices_out,
    GM_ADDR workspace, uint32_t vec_len, uint32_t tile_len) {
  GM_ADDR const num_ones_per_block = workspace;
  const uint32_t block_len = tile_len * tile_len / 2;

  // The compaction and the index generation below both start from each core's
  // running total of mask ones, so the mask is reduced here rather than on
  // the host; the count never has to leave the device.
  if ASCEND_IS_AIV {
    KernelReduceTiles<int8_t> op_reduce(block_len, vec_len);
    op_reduce.Init(mask_in, num_ones_per_block);
    op_reduce.Process();
  }
  SyncAll<true /*isAIVOnly*/>();

  if ASCEND_IS_AIV {
    KernelCompress<T> data_op(vec_len, block_len);
    data_op.Init(vec_in, mask_in, num_ones_per_block, vec_out);
    data_op.Process();
  }
  SyncAll<true /*isAIVOnly*/>();

  if ASCEND_IS_AIV {
    KernelWhere op(vec_len, block_len);
    op.Init(mask_in, num_ones_per_block, indices_out);
    op.Process();
  }
}

}  // namespace tcuscan
