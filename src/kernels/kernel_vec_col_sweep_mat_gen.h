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
 * [1] Parallelism in Matrix Computations. E. Gallopoulos, B. Philippe and A .H.
 Sameh. Hard cover (ISBN: 978-94-017-7187-0), Soft cover (ISBN:
 978-94-024-0317-6), Electronic (ISBN: 978-94-017-7188-7)
 */
template <typename T = half>
class KernelVecColSweepMatGen {
  constexpr static uint32_t BUFFER_NUM = 1;

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
    global_in_.SetGlobalBuffer((__gm__ T *)vec_in, vec_len);
    global_out_.SetGlobalBuffer((__gm__ T *)vec_out, vec_len);

    pipe_.InitBuffer(in_q_, BUFFER_NUM, tile_len_ * sizeof(T));
    pipe_.InitBuffer(out_q_, BUFFER_NUM, tile_len_ * sizeof(T));
    pipe_.InitBuffer(work_buf_, tile_len_ * sizeof(T));
  }

  /**
   * @brief Run the kernel - process all tiles.
   *
   */
  __aicore__ inline void Process() {
    // TODO(anastasios): optimization so that AIV0 and AIV handle the even and
    // odd iterations, respectively.
    if (GetSubBlockIdx() == 0) {
      ProcessWorker();
    } else {
      ProcessDummy();
    }
  }

  /**
   * @brief Noop work by AIV. Synchronizes `matrix_size_ + 1` times with AI
   * group to avoid deadlocks.
   *
   */
  __aicore__ inline void ProcessDummy() {
    sync::SyncGroup<sync::GroupSyncDirection::FULL>();
    for (uint32_t iter = 0; iter < matrix_size_; iter++) {
      (void)iter;
      // Sync with all AIVs in group, to write the matrix.
      sync::SyncGroup<sync::GroupSyncDirection::FULL>();
    }
  }

  /**
   * @brief Run the kernel.
   *
   */
  __aicore__ inline void ProcessWorker() {
    // Read input matrix into work_buf_.
    copy::CopyGmToVec(in_q_, global_in_[global_offset_]);
    ReadInputMatrixInUB();

    // Write identity matrix for AIC
    EnQueueIdentityMatrix();
    copy::CopyVecToGm(global_out_[global_offset_], out_q_);

    //  Sync with all AIVs in group, to write the matrix.
    sync::SyncGroup<sync::GroupSyncDirection::FULL>();

    // Matrix column sweep algorithm requires `matrix_size_` iterations.
    for (uint32_t column_index = 0; column_index < matrix_size_;
         column_index++) {
      ColumnSweepMatrixGenerator(matrix_size_ - (column_index + 1));
      copy::CopyVecToGm(global_out_[global_offset_], out_q_);

      // Sync with all AIVs in group, to write the matrix.
      sync::SyncGroup<sync::GroupSyncDirection::FULL>();
    }
  }

 private:
  /**
   * @brief Write the identity square matrix into the local tensor.
   *
   * @param [in] lt Input local tensor of length \f$ matrix_size_ \times
   * matrix_size_\f$ where the identity matrix will be written.
   */
  __aicore__ inline void FillIdentityMatrix(const LocalTensor<T> &lt) {
    Duplicate(lt, static_cast<T>(0), lt.GetSize());

    // Set one on the main diagonal
    for (uint32_t i = 0; i < matrix_size_; i++) {
      lt.SetValue(i * matrix_size_ + i, static_cast<T>(1));
    }
  }

  /**
   * @brief Set (in-place) j-th column of input matrix to be the j-th column
   * of the identity matrix.
   *
   * @param [in] lt Input local tensor of length \f$ matrix_size_ \times
   * matrix_size_\f$.
   * @param [in] j Column index to set.
   */
  __aicore__ inline void FillColumnWithStandardVector(const LocalTensor<T> &lt,
                                                      uint32_t j) {
    const uint32_t offset = j * matrix_size_;
    // Write zeros on j-th column of input matrix
    Duplicate(lt[offset], static_cast<T>(0), matrix_size_);

    // Write one on the diagonal (j,j)-th element of input matrix.
    lt.SetValue(offset + j, static_cast<T>(1));
  }

  /**
   * @brief Generates a "column-sweep" matrix and enqueues it into
   * `out_q_`.
   *
   * Example of enqueued matrix in `out_q_` for matrix_size_ = 4 and col_index =
   * 1
   *
   * output = [[1   0  0 0],
   *            [0   1  0 0],
   *            [0 -w_1 1 0],
   *            [0 -w_2 0 1]]
   *
   *  @param [in] col_index Column index on which to generate the column sweep
   * matrix.
   */
  __aicore__ inline void ColumnSweepMatrixGenerator(uint32_t col_index) {
    const LocalTensor<T> vec_out_lt = out_q_.AllocTensor<T>();

    // Set identity matrix on output buffer.
    FillIdentityMatrix(vec_out_lt);

    // Write vector on the "col_index"-th columnn.
    LocalTensor<T> work_lt = work_buf_.Get<T>();
    const uint32_t offset = col_index * matrix_size_;
    DataCopy(vec_out_lt[offset], work_lt[offset], matrix_size_);
    // Make sure "diagonal" element is one.
    vec_out_lt.SetValue(offset + col_index, static_cast<T>(1));

    out_q_.EnQue<T>(vec_out_lt);
  }

  /**
   * @brief Read the input triangular matrix A into the `work_buf_`.
   */
  __aicore__ inline void ReadInputMatrixInUB() {
    LocalTensor<T> vec_in_lt = in_q_.DeQue<T>();
    LocalTensor<T> work_lt = work_buf_.Get<T>();
    Duplicate(work_lt, static_cast<T>(0), work_lt.GetSize());
    Muls(work_lt, vec_in_lt, static_cast<T>(-1), vec_in_lt.GetSize());
    in_q_.FreeTensor<T>(vec_in_lt);
  }

  /**
   * @brief EnQue identity matrix on output queue.
   *
   */
  __aicore__ inline void EnQueueIdentityMatrix() {
    const LocalTensor<T> vec_out_lt = out_q_.AllocTensor<T>();
    FillIdentityMatrix(vec_out_lt);
    out_q_.EnQue<T>(vec_out_lt);
  }

  TPipe pipe_;

  TQue<QuePosition::VECIN, BUFFER_NUM> in_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> out_q_;

  TBuf<QuePosition::VECCALC> work_buf_;

  GlobalTensor<T> global_in_;
  GlobalTensor<T> global_out_;

  const uint32_t matrix_size_;
  const uint32_t tile_len_;
  const uint32_t aic_id_;
  const uint32_t global_offset_;
};

}  // namespace tcuscan
