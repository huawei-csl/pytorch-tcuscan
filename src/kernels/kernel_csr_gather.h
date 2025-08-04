#pragma once
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * @file kernel_csr_gather.h
 * @brief Kernel implementing a CSR gather operation.
 */

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;
using namespace kernel_utils;

/**
 * @brief Performs the CSR gather operation as described in SEGMV algorithm in
 * [1].
 *
 * [1] Segmented Operations for Sparse Matrix Computation on Vector
 * Multiprocessors: https://dl.acm.org/doi/10.5555/865221.
 */
template <typename DataType>
class KernelCSRGather {
  constexpr static uint32_t BUFFER_NUM = 2;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] values_in_len Length of the input values vector.
   * @param [in] rows_in_len Length of the input `row_ptr` vector.
   * @param [in] x_in_len Length of the input x vector.
   * @param [in] tile_len Length of the tile processed in a single iteration.
   */

  __aicore__ inline KernelCSRGather(uint32_t values_in_len,
                                    uint32_t rows_in_len, uint32_t x_in_len,
                                    uint32_t tile_len)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        values_in_len_(values_in_len),
        rows_in_len_(rows_in_len),
        x_in_len_(x_in_len),
        tile_len_(tile_len),
        num_tiles_(scalar::CeilDiv(values_in_len, tile_len_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, vec_core_num_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] values_in  Pointer to input values vector.
   * @param [in] cols_in Pointer to input column indices vector.
   * @param [in] rows_in Pointer to input rows indices vector.
   * @param [in] x_in Pointer to input x vector.
   * @param [in] z_out Point to output z vector.
   */
  __aicore__ inline void Init(GM_ADDR values_in, GM_ADDR cols_in,
                              GM_ADDR rows_in, GM_ADDR x_in, GM_ADDR z_out) {
    // CSR Matrix (values, columns, row_ptr)
    global_values_.SetGlobalBuffer((__gm__ DataType *)values_in,
                                   values_in_len_);
    global_cols_.SetGlobalBuffer((__gm__ uint32_t *)cols_in, values_in_len_);
    global_rows_.SetGlobalBuffer((__gm__ uint32_t *)rows_in, rows_in_len_);

    global_x_.SetGlobalBuffer((__gm__ DataType *)x_in, x_in_len_);
    global_z_.SetGlobalBuffer((__gm__ DataType *)z_out, values_in_len_);

    pipe.InitBuffer(values_q_, BUFFER_NUM, tile_len_ * sizeof(DataType));
    pipe.InitBuffer(cols_q_, BUFFER_NUM, tile_len_ * sizeof(uint32_t));
    pipe.InitBuffer(rows_q_, BUFFER_NUM, tile_len_ * sizeof(uint32_t));
    pipe.InitBuffer(x_q_, 1, x_in_len_ * sizeof(DataType));
    pipe.InitBuffer(output_q_, BUFFER_NUM, tile_len_ * sizeof(DataType));

    pipe.InitBuffer(tbuf_, tile_len_ * sizeof(uint32_t));
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    uint32_t gm_offset = GetBlockIdx() * tile_len_ * max_num_tiles_per_block_;
    const uint32_t num_tiles_to_process =
        kernel_utils::scalar::GetWorkDistribution(values_in_len_, tile_len_,
                                                  vec_core_num_);

    if (num_tiles_to_process == 0) {
      return;
    }

    // Read whole vector x
    copy::CopyGmToVec(x_q_, global_x_, x_in_len_);
    x_lt_ = x_q_.DeQue<DataType>();

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const bool is_full_tile = gm_offset + tile_len_ <= values_in_len_;
      const uint32_t num_elems =
          is_full_tile ? tile_len_ : values_in_len_ - gm_offset;

      // CopyIN
      copy::CopyGmToVec(values_q_, global_values_[gm_offset], num_elems);
      copy::CopyGmToVec(cols_q_, global_cols_[gm_offset], num_elems);

      // Compute
      ProcessTile(num_elems);

      // CopyOut
      copy::CopyVecToGm(global_z_[gm_offset], output_q_, num_elems);

      gm_offset += tile_len_;
    }

    x_q_.FreeTensor<DataType>(x_lt_);
  }

  /**
   * @brief Process tile assumes that `values_q_` is populated with a tile
   * and `x_lt_` is also populated with the whole input.
   *
   *    @param num_elems Number of elements to process
   */
  __aicore__ inline void ProcessTile(uint32_t num_elems) {
    LocalTensor<uint32_t> cols_lt = cols_q_.DeQue<uint32_t>();
    LocalTensor<DataType> z_lt = output_q_.AllocTensor<DataType>();

    LocalTensor<uint32_t> cols_uint32_t = tbuf_.Get<uint32_t>();

    ShiftLeft<uint32_t>(cols_uint32_t, cols_lt, 1, cols_lt.GetSize());
    cols_q_.FreeTensor<uint32_t>(cols_lt);
    AscendC::Gather(z_lt, x_lt_, cols_uint32_t, (uint32_t)0, num_elems);
    LocalTensor<DataType> values_lt = values_q_.DeQue<DataType>();
    Mul(z_lt, z_lt, values_lt, num_elems);
    values_q_.FreeTensor<DataType>(values_lt);

    output_q_.EnQue<DataType>(z_lt);
  }

 private:
  TPipe pipe;
  TQue<QuePosition::VECIN, BUFFER_NUM> values_q_;
  TQue<QuePosition::VECIN, BUFFER_NUM> cols_q_;
  TQue<QuePosition::VECIN, BUFFER_NUM> rows_q_;
  TQue<QuePosition::VECIN, BUFFER_NUM> x_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> output_q_;

  TBuf<QuePosition::VECCALC> tbuf_;

  GlobalTensor<DataType> global_values_;
  GlobalTensor<uint32_t> global_cols_;
  GlobalTensor<uint32_t> global_rows_;
  GlobalTensor<DataType> global_x_;
  GlobalTensor<DataType> global_z_;

  LocalTensor<DataType> x_lt_;

  const uint32_t vec_core_num_;
  const uint32_t values_in_len_;
  const uint32_t rows_in_len_;
  const uint32_t x_in_len_;
  const uint32_t tile_len_;
  const uint16_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;
};

/**
 * @brief Run the `csr_gather` kernel.
 *
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 *
 * @param [in] values_in Pointer to input vector.
 * @param [in] cols_in Pointer to column indices input vector.
 * @param [in] rows_in Pointer to rows indices input vector.
 * @param [in] x_in Pointer to x input vector.
 * @param [in] z_out Pointer to output vector.
 * @param [in] values_in_len Length of the input values vector.
 * @param [in] rows_in_len Length of the input rows indices vector.
 * @param [in] x_in_len Length of the input x vector.
 * @param [in] tile_len Length of the tile processed in a single iteration.
 */
template <bool ForceMixMode = true>
__aicore__ inline void run_csr_gather(GM_ADDR values_in, GM_ADDR cols_in,
                                      GM_ADDR rows_in, GM_ADDR x_in,
                                      GM_ADDR z_out, uint32_t values_in_len,
                                      uint32_t rows_in_len, uint32_t x_in_len,
                                      uint32_t tile_len) {
  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }

  if ASCEND_IS_AIV {
    KernelCSRGather<half> op(values_in_len, rows_in_len, x_in_len, tile_len);
    op.Init(values_in, cols_in, rows_in, x_in, z_out);
    op.Process();
  }
}
