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
 * @brief Returns the number of prefetch pipelines that are allowed, given the
 * input matrix size.
 *
 * @param matrix_size Input matrix size for triangular matrix inversion.
 * @return Number of pipeline stages (AscendC queue length)
 */
constexpr uint32_t GetPipeNumStages(unsigned matrix_size) {
  if (matrix_size == 16 || matrix_size == 32) {
    return 8;
  } else if (matrix_size == 64) {
    return 4;
  }
  return 1;
}

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
 * @tparam matrix_size Input square matrix size. Supports 16, 32, 64, 128.
 */
template <typename T, unsigned MATRIX_SIZE>
class KernelTriInvColumnSweep {
  constexpr static uint32_t BUFFER_NUM = GetPipeNumStages(MATRIX_SIZE);

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Dimension of the input vectors.
   */
  __aicore__ inline KernelTriInvColumnSweep(uint32_t vec_len)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        vec_len_(vec_len),
        max_num_matrices_per_core_(
            scalar::CeilDiv(vec_len, tile_len_ * vec_core_num_)) {}

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
    pipe_.InitBuffer(b_buf_, MATRIX_SIZE * sizeof(T));
  }

  /**
   * @brief Run the kernel.
   *
   */
  __aicore__ inline void Process() {
    const uint32_t global_offset =
        GetBlockIdx() * max_num_matrices_per_core_ * tile_len_;

    const uint32_t num_matrices_to_process =
        tcuscan::scalar::GetWorkDistribution(vec_len_, tile_len_,
                                             vec_core_num_);

    for (uint32_t matrix_idx = 0; matrix_idx < num_matrices_to_process;
         matrix_idx++) {
      const uint32_t offset = global_offset + matrix_idx * tile_len_;
      copy::CopyGmToVec(in_q_, global_in_[offset], tile_len_);
      InvertMatrix();
      copy::CopyVecToGm(global_out_[offset], out_q_, tile_len_);
    }
  }

 private:
  __aicore__ inline void InvertMatrix() {
    constexpr int32_t n_rows = MATRIX_SIZE;
    constexpr int32_t n_cols = MATRIX_SIZE;

    LocalTensor<T> vec_in_lt = in_q_.template DeQue<T>();
    const LocalTensor<T> vec_out_lt = out_q_.template AllocTensor<T>();

    // Left-hand side Ax=b.
    LocalTensor<T> b = b_buf_.Get<T>();

    Duplicate(vec_out_lt, static_cast<T>(0), tile_len_);

    // For every output column j-th
    for (int32_t j = 0; j < n_cols; j++) {
      // Column sweep on each column.

      // `b` vector is e_j standard vector.
      Duplicate(b, static_cast<T>(0), MATRIX_SIZE);
      b.SetValue(j, static_cast<T>(1));

      // Ax=b
      LocalTensor<T> x = vec_out_lt[j * n_rows];
      for (int32_t k = n_rows - 1; k >= 0; k--) {
        const LocalTensor<T> A_k = vec_in_lt[k * n_rows];

        // x[k] = b[k] / A[k, k]
        const T x_k = b.GetValue(k);
        x.SetValue(k, x_k);

        if (k > 0) {
          // b[:k] -= A[:k, k] * x[k]
          const float x_k_minus_fp32 = -static_cast<float>(x_k);
          AscendC::Axpy<T>(b, A_k, static_cast<T>(x_k_minus_fp32), k);
        }
      }
    }

    out_q_.template EnQue<T>(vec_out_lt);
    in_q_.template FreeTensor<T>(vec_in_lt);
  }

  TPipe pipe_;

  TQue<QuePosition::VECIN, BUFFER_NUM> in_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> out_q_;

  TBuf<QuePosition::VECCALC> b_buf_;

  GlobalTensor<T> global_in_;
  GlobalTensor<T> global_out_;

  const uint32_t vec_core_num_;
  const uint32_t vec_len_;
  constexpr static uint32_t tile_len_ = MATRIX_SIZE * MATRIX_SIZE;
  const uint32_t max_num_matrices_per_core_;
};

/**
 * @brief Run the `tri_inv_col_sweep` kernel.
 *
 * @tparam T Input data type. Supports fp16/half.
 * @tparam MATRIX_SIZE Input square matrix size.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] vec_len Dimension of the input vector.
 */
template <typename T, unsigned MATRIX_SIZE>
__aicore__ inline void tri_inv_col_sweep(GM_ADDR vec_in, GM_ADDR vec_out,
                                         uint32_t vec_len) {
  if ASCEND_IS_AIV {
    KernelTriInvColumnSweep<T, MATRIX_SIZE> op(vec_len);
    op.Init(vec_in, vec_out);
    op.Process();
  }
}

}  // namespace tcuscan
