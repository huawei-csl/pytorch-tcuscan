/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file kernel_tri_inv_col_sweep.h
 * @brief Kernel implementing a Vector matrix inverse kernel operation.
 */

#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Returns the matrix inverse of the matrix I+A, where I
 * is the identity and A is the input strictly lower triangular matrix
 * of size matrix_size. The matrix A must be in column-major format.
 * The diagonal elements of the input are always ignored and treated
 * as one, e.g., it does not matter if the input is A or I+A. The
 * column sweep algorithm is used for the linear system (I+A)x=e_j where
 * e_j is the j-th standard vector.
 *
 * @tparam T Input data type. Supports only half dtype.
 *
 */
template <typename T>
class KernelTriInvColumnSweep {
  constexpr static uint32_t BUFFER_NUM = 1;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Dimension of the input vectors.
   * @param [in] matrix_size Input square matrix size.
   */
  __aicore__ inline KernelTriInvColumnSweep(uint32_t vec_len,
                                            uint32_t matrix_size)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        vec_len_(vec_len),
        matrix_size_(matrix_size),
        tile_len_(matrix_size * matrix_size) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_in Pointer to the input vector in global memory.
   * @param [in] vec_out Pointer to the output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR vec_out) {
    global_in_.SetGlobalBuffer((__gm__ T*)vec_in, vec_len_);
    global_out_.SetGlobalBuffer((__gm__ T*)vec_out, vec_len_);

    pipe_.InitBuffer(in_q_, BUFFER_NUM, tile_len_ * sizeof(T));
    pipe_.InitBuffer(out_q_, BUFFER_NUM, tile_len_ * sizeof(T));
    pipe_.InitBuffer(b_buf_, matrix_size_ * sizeof(T));
  }

  /**
   * @brief Run the kernel.
   *
   */
  __aicore__ inline void Process() {
    const uint32_t global_offset = GetBlockIdx() * tile_len_;
    copy::CopyGmToVec(in_q_, global_in_[global_offset], tile_len_);
    InvertMatrix();
    copy::CopyVecToGm(global_out_[global_offset], out_q_, tile_len_);
  }

 private:
  __aicore__ inline void InvertMatrix() {
    const int32_t n_rows = matrix_size_;
    const int32_t n_cols = matrix_size_;

    LocalTensor<T> vec_in_lt = in_q_.DeQue<T>();
    const LocalTensor<T> vec_out_lt = out_q_.AllocTensor<T>();

    // Left-hand side Ax=b.
    LocalTensor<T> b = b_buf_.Get<T>();

    // Transpose(vec_in_lt, vec_in_lt);
    Duplicate(vec_out_lt, static_cast<T>(0), tile_len_);

    // For every output column j-th
    for (int32_t j = 0; j < n_cols; j++) {
      // Column sweep on each column.

      // `b` vector is e_j standar vector.
      Duplicate(b, static_cast<T>(0), matrix_size_);
      b.SetValue(j, static_cast<T>(1));

      // Ax=b
      LocalTensor<T> x = vec_out_lt[j * n_rows];
      for (int32_t k = n_rows - 1; k >= 0; k--) {
        const LocalTensor<T> A_k = vec_in_lt[k * n_rows];

        // x[k] = b[k] / A[k, k]
        x.SetValue(k, b.GetValue(k));

        if (k > 0) {
          // b[:k] -= A[:k, k] * x[k]
          const float x_k = -static_cast<float>(x.GetValue(k));
          AscendC::Axpy<T>(b, A_k, static_cast<T>(x_k), k);
        }
      }
    }

    out_q_.EnQue<T>(vec_out_lt);
    in_q_.FreeTensor<T>(vec_in_lt);
  }

  TPipe pipe_;

  TQue<QuePosition::VECIN, BUFFER_NUM> in_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> out_q_;

  TBuf<QuePosition::VECCALC> b_buf_;

  GlobalTensor<T> global_in_;
  GlobalTensor<T> global_out_;

  const uint32_t vec_core_num_;
  const uint32_t vec_len_;
  const uint32_t matrix_size_;
  const uint32_t tile_len_;
};

/**
 * @brief Run the `tri_inv_col_sweep` kernel.
 *
 * @tparam T Input data type. Supports fp16/half.
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] vec_len Dimension of the input vector.
 * @param [in] matrix_size Matrix size to invert.
 */
template <typename T, bool ForceMixMode = false>
__aicore__ inline void tri_inv_col_sweep(GM_ADDR vec_in, GM_ADDR vec_out,
                                         uint32_t vec_len,
                                         uint32_t matrix_size) {
  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }
  if ASCEND_IS_AIV {
    KernelTriInvColumnSweep<T> op(vec_len, matrix_size);
    op.Init(vec_in, vec_out);
    op.Process();
  }
}

}  // namespace tcuscan
