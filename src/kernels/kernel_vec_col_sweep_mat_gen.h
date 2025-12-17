/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file kernel_vec_col_sweep_mat_gen.h
 * @brief Kernel implementing an AIV matrix generator that generates the matrix
 * formulation of column sweep.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;
using namespace kernel_utils;

namespace tcuscan {

/**
 * @brief Returns a sequence of matrices that encode the column-sweep steps in
 * matrix notation. On the first iteration, it returns the identity matrix.
 *
 * @tparam Input data type. Support fp16/half.
 *
 * See discussion on Section 3.2.1 of [1], in particular Equations 3.8 and 3.9
 * on page 54.
 *
 * \code{.python}
  def aiv_matrix_gen(A: npt.ArrayLike):
    n = A.shape[0]
    I_n = np.eye(n, dtype=A.dtype)

    # Your transformation A = 2I_n - A
    A = 2 * I_n - A

    for k in reversed(range(n)):
      M = I_n.copy()
      M[:, k] = A[:, k]
      yield M

  * \endcode
  *
  *  [1] Parallelism in Matrix Computations.E.Gallopoulos, B.Philippe and
 A.H.Sameh.
  * Hard cover(ISBN : 978 - 94 - 017 - 7187 - 0),
  * Soft cover(ISBN : 978 - 94 - 024 - 0317 - 6),
  * Electronic(ISBN : 978 - 94 - 017 - 7188 - 7)
  **/
template <typename T = half>
class KernelVecColSweepMatGen {
 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] matrix_size Input square matrix size.
   */
  __aicore__ inline KernelVecColSweepMatGen(uint32_t matrix_size)
      : matrix_size_(matrix_size),
        tile_len_(matrix_size * matrix_size),
        aic_id_(scalar::FloorDiv(GetBlockIdx(), GetTaskRation())),
        global_offset_(aic_id_ * tile_len_) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_in Pointer to the input vector in global memory.
   * @param [in] vec_out Pointer to the output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR vec_out) {
    const uint32_t vec_len = GetBlockNum() * tile_len_;
    global_in_.SetGlobalBuffer((__gm__ T*)vec_in, vec_len);
    global_out_.SetGlobalBuffer((__gm__ T*)vec_out, vec_len);

    pipe_.InitBuffer(in_q_, 1, tile_len_ * sizeof(T));
    pipe_.InitBuffer(out_q_, 1, tile_len_ * sizeof(T));
    pipe_.InitBuffer(work_buf_, tile_len_ * sizeof(T));
  }

  /**
   * @brief Run the kernel.
   *
   */
  __aicore__ inline void Process() {
    // Read input matrix into work_buf_.
    copy::CopyGmToVec(in_q_, global_in_[global_offset_]);
    ReadInputMatrixInUB();

    // AIV-0 writes identity matrix for AIC
    if (GetSubBlockIdx() == 0) {
      EnQueueIdentityMatrix();
      copy::CopyVecToGm(global_out_[global_offset_], out_q_);
    }

    //  Sync with all AIVs in group, to write the matrix.
    sync::SyncGroup<sync::GroupSyncDirection::FULL>();

    // First matrix is identity (just wait one more round)
    sync::SyncGroup<sync::GroupSyncDirection::FULL>();

    // Matrix column sweep algorithm requires `matrix_size_` iterations.
    for (int32_t col_index = matrix_size_ - 2; col_index >= 0; col_index--) {
      // AIV-0: writes the  (col_index + 1)-th column of the identity matrix
      // AIV-1: writes the "column-sweep" column of matrix `M`.
      if (GetSubBlockIdx() == 0) {
        const uint32_t out_offset =
            global_offset_ + (col_index + 1) * matrix_size_;
        const LocalTensor<T> vec_out_lt = out_q_.AllocTensor<T>();

        // Write the (col_index + 1)-th column of identity matrix on first
        // `matrix_size_` elements of vec_out_lt.
        FillStandardVector(vec_out_lt, matrix_size_, col_index + 1);

        out_q_.EnQue<T>(vec_out_lt);
        copy::CopyVecToGm(global_out_[out_offset], out_q_, matrix_size_);

      } else if (GetSubBlockIdx() == 1) {
        const uint32_t out_offset = global_offset_ + col_index * matrix_size_;
        const LocalTensor<T> vec_out_lt = out_q_.AllocTensor<T>();

        // Write the (col_index)-th column of matrix M.
        LocalTensor<T> work_lt = work_buf_.Get<T>();
        DataCopy(vec_out_lt, work_lt[col_index * matrix_size_], matrix_size_);
        out_q_.EnQue<T>(vec_out_lt);
        copy::CopyVecToGm(global_out_[out_offset], out_q_, matrix_size_);
      }

      // Sync with all AIVs in group, to write the matrix.
      sync::SyncGroup<sync::GroupSyncDirection::FULL>();
    }
  }

 private:
  /**
   * @brief Read the input triangular matrix A into the `work_buf_`.
   */
  __aicore__ inline void ReadInputMatrixInUB() {
    LocalTensor<T> vec_in_lt = in_q_.DeQue<T>();
    LocalTensor<T> work_lt = work_buf_.Get<T>();
    Duplicate(work_lt, static_cast<T>(0), work_lt.GetSize());
    Muls(work_lt, vec_in_lt, static_cast<T>(-1), vec_in_lt.GetSize());
    kernel_utils::FillDiagonal(work_lt, matrix_size_, static_cast<T>(1));
    in_q_.FreeTensor<T>(vec_in_lt);
  }

  /**
   * @brief EnQue identity matrix on output queue.
   *
   */
  __aicore__ inline void EnQueueIdentityMatrix() {
    const LocalTensor<T> vec_out_lt = out_q_.AllocTensor<T>();
    kernel_utils::FillIdentity(vec_out_lt, matrix_size_);
    out_q_.EnQue<T>(vec_out_lt);
  }

  TPipe pipe_;

  TQue<QuePosition::VECIN, 1> in_q_;
  TQue<QuePosition::VECOUT, 1> out_q_;

  TBuf<QuePosition::VECCALC> work_buf_;

  GlobalTensor<T> global_in_;
  GlobalTensor<T> global_out_;

  const uint32_t matrix_size_;
  const uint32_t tile_len_;
  const uint32_t aic_id_;
  const uint32_t global_offset_;
};

}  // namespace tcuscan
