/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file kernel_tri_inv_cube_col_sweep.h
 * @brief Kernel implementing a Cube matrix inverse column sweep.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;
using namespace kernel_utils;

namespace tcuscan {

/**
 * @brief Returns the matrix inverse of an upper triangular square matrix of
 * size `matrix_size`. The matrix has ones on the main diagonal.
 *
 * The column sweep algorithm is used for the linear system Ax=e_j where e_j is
 * the standard vector.
 *
 * The kernel assumes \f$ matrix_size_ + 1 \f$ synchronizations with the AIVs.
 *
 * @tparam InputT Input data type. Default value fp16/half.
 *
 */
template <typename InputT = half>
class KernelTriInvCubeColSweep {
  using OutputT = kernel_utils::cube_unit::CubeOutType_t<InputT>;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] matrix_size Input square matrix size to invert.
   */
  __aicore__ inline KernelTriInvCubeColSweep(uint32_t matrix_size)
      : matrix_size_(matrix_size),
        tile_len_(matrix_size * matrix_size),
        global_offset_(GetBlockIdx() * tile_len_) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] matrix_stream_in Pointer where the input matrices will be
   * written by AIVs in global memory.
   * @param [in] inv_matrix_out Pointer where the output matrix inverses
   * will be written.
   */
  __aicore__ inline void Init(GM_ADDR matrix_stream_in,
                              GM_ADDR inv_matrix_out) {
    const uint32_t vec_len = GetBlockNum() * tile_len_;
    global_A_.SetGlobalBuffer((__gm__ InputT*)matrix_stream_in, vec_len);
    global_C_.SetGlobalBuffer((__gm__ OutputT*)inv_matrix_out, vec_len);

    pipe_.InitBuffer(a1_q_, 1, tile_len_ * sizeof(InputT));
    pipe_.InitBuffer(a2_q_, 1, tile_len_ * sizeof(InputT));
    pipe_.InitBuffer(b1_q_, 1, tile_len_ * sizeof(InputT));
    pipe_.InitBuffer(b2_q_, 1, tile_len_ * sizeof(InputT));
    pipe_.InitBuffer(co1_q_, 1, tile_len_ * sizeof(OutputT));
  }

  /**
   * @brief Run kernel.
   *
   */
  __aicore__ inline void Process() {
    // On the first iteration, the AIVs will send the identity matrix to AIC
    sync::SyncGroup<sync::GroupSyncDirection::FULL>();
    LoadIdentityMatrixinL0C();

    // Matrix column sweep algorithm requires `matrix_size_` iterations.
    for (uint32_t iter = 0; iter < matrix_size_; iter++) {
      (void)iter;
      // Sync with all AIVs in group, to write the matrix.
      sync::SyncGroup<sync::GroupSyncDirection::FULL>();

      // Load next matrix A and perform C = A @ C
      LoadMatrixAintoL0A();
      MultiplyAWithC();
      AscendC::PipeBarrier<PIPE_ALL>();
    }

    // Write L0C matrix to global memory
    copy::CopyCL0ToGlobal(global_C_[global_offset_], co1_q_, M_, N_);
  }

 private:
  /**
   * @brief Loads matrix from global memory into L0A (`a1_q_` queue).
   */
  __aicore__ inline void LoadMatrixAintoL0A() {
    // Load matrix from global_A_ into L0A
    copy::CopyGmToL1A(a1_q_, global_A_[global_offset_], m_blocks_, k_blocks_);
    copy::CopyL1ToL0A<InputT, true>(a2_q_, a1_q_, m_blocks_, k_blocks_);
  }
  /**
   * @brief Perform matrix multiplication in Cube unit, like C = A @ C
   *
   * Assumes that the matrices A and C are enqueued.
   */
  __aicore__ inline void MultiplyAWithC() {
    cube_unit::MultiplyAWithC<InputT, true /* free_a*/>(a2_q_, b1_q_, b2_q_,
                                                        co1_q_, M_, N_, K_);
  }

  /**
   * @brief Loads the identity matrix from global memory to L0C (`co1_q_`
   * queue).
   */
  __aicore__ inline void LoadIdentityMatrixinL0C() {
    LoadIdentityMatrixinL0A();
    LoadIdentityMatrixinL0B();
    cube_unit::Multiply<InputT, false /* accumulate_c */, true /* free_a*/,
                        true /* free_b */>(a2_q_, b2_q_, co1_q_, M_, N_, K_);
  }

  /**
   * @brief Loads the identity matrix from global memory to L0A (`a1_q_`
   * queue).
   */
  __aicore__ inline void LoadIdentityMatrixinL0A() {
    copy::CopyGmToL1A(a1_q_, global_A_[global_offset_], m_blocks_, k_blocks_);
    copy::CopyL1ToL0A<InputT, true>(a2_q_, a1_q_, m_blocks_, k_blocks_);
  }

  /**
   * @brief Loads the identity matrix from global memory to L0B (`b1_q_`
   * queue).
   */
  __aicore__ inline void LoadIdentityMatrixinL0B() {
    // Here, we "abuse" the 'global_A_' pointer
    copy::CopyTransposedGmToL0B(b2_q_, b1_q_, global_A_[global_offset_],
                                k_blocks_, n_blocks_);
  }

  TPipe pipe_;

  TQue<QuePosition::A1, 1> a1_q_;
  TQue<QuePosition::A2, 1> a2_q_;
  TQue<QuePosition::B1, 1> b1_q_;
  TQue<QuePosition::B2, 1> b2_q_;

  TQue<QuePosition::CO1, 1> co1_q_;

  GlobalTensor<InputT> global_A_;
  GlobalTensor<OutputT> global_C_;

  const uint32_t matrix_size_;
  const uint32_t tile_len_;
  const uint32_t global_offset_;

  constexpr static uint32_t M_CUBE_BLOCK_SIZE =
      kernel_utils::GetFractalMN<InputT>();
  constexpr static uint32_t N_CUBE_BLOCK_SIZE =
      kernel_utils::GetFractalMN<InputT>();
  constexpr static uint32_t K_CUBE_BLOCK_SIZE =
      kernel_utils::GetFractalK<InputT>();

  const uint32_t M_ = matrix_size_;
  const uint32_t K_ = matrix_size_;
  const uint32_t N_ = matrix_size_;

  const uint32_t n_blocks_ = N_ / N_CUBE_BLOCK_SIZE;
  const uint32_t k_blocks_ = K_ / K_CUBE_BLOCK_SIZE;
  const uint32_t m_blocks_ = M_ / M_CUBE_BLOCK_SIZE;
};

/**
 * @brief Run the `tri_inv_cube_col_sweep` kernel.
 *
 * @tparam InputT Input data type. Supports fp16/half.
 *
 * @param [in] matrix_stream_in Pointer where the input matrices will be
 * written by AIVs in global memory.
 * @param [in] inv_matrix_out Pointer where the output matrix inverses
 * will be written.
 * @param [in] matrix_size Input square matrix size to invert.
 */
template <typename InputT>
__aicore__ inline void run_tri_inv_cube_col_sweep(GM_ADDR matrix_stream_in,
                                                  GM_ADDR inv_matrix_out,
                                                  uint32_t matrix_size) {
  if ASCEND_IS_AIC {
    KernelTriInvCubeColSweep<InputT> op(matrix_size);
    op.Init(matrix_stream_in, inv_matrix_out);
    op.Process();
  }
}

}  // namespace tcuscan
