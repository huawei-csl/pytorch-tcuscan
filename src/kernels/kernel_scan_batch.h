/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_scan_batch.h
 * @brief Kernel implementing different variants of a scan over a batched input
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "kernel_row_scan.h"
#include "kernel_utils.h"

using namespace AscendC;
using namespace kernel_utils;

/**
 * @brief Run the batched scan kernel using only the cube core.
 *
 * @tparam InputT Data type of the input vectors.
 * @tparam OutputT Data type of the output vectors.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] upper_triangular Pointer to an upper-triangular matrix filled
 * with ones of size \f$\textit{matmul_size} \times \textit{matmul_size}\f$.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] vec_len Number of elements in one vector.
 * @param [in] batch_size Number of vectors in a batch.
 */
template <typename InputT>
__aicore__ inline void run_scan_cube_only_kernel(GM_ADDR input_vec,
                                                 GM_ADDR upper_triangular,
                                                 GM_ADDR output_vec,
                                                 uint32_t vec_len,
                                                 uint32_t batch_size) {
  if ASCEND_IS_AIC {
    KernelRowScan<InputT> op_cube(vec_len, vec_len, vec_len * batch_size);
    op_cube.Init(input_vec, upper_triangular, output_vec);
    op_cube.Process();
  }
}
/**
 * @brief Performs a "row-wise" inclusive-scan on multiple input vectors.
 *
 * The algorithm assigns pairs of independent vectors to the cube cores
 *
 * The algorithm splits the input vectors into chunks ("rows") of size
 * \f$K = \textit{matmul_k_size}\f$ and performs local inclusive scans on all
 * the chunks separately. Local scan is performed by transforming the problem
 * into matrix multiplication. Matrix \f$B\f$ is an upper triangular matrix of
 * size \f$K \times K\f$ with ones on the main diagonal and above. Matrix
 * \f$A\f$ is created from each input vector by reshaping them into a matrix
 * with \f$K\f$ dimension equal to \f$K\f$ and \f$M\f$ dimension equal to
 * \f$\textit{matmul_m_size}\f$. Then matrix \f$A\f$ is splitted into \f$M
 * \times K\f$ tiles that are multiplied by the same matrix \f$B\f$.
 *
 * The algorithm assumes that the length of the input vectors is divisible by
 * \f$M K\f$.
 */
class KernelRowScanBatch {
 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] matmul_k_size Size of the K dimension of A matrix.
   * @param [in] matmul_m_size Size of the M dimension of A matrix.
   * @param [in] vec_len Number of elements in a single input vector.
   * @param [in] vec_cube_ratio Specifies the vector to cube ratio to use
   * (ranges from 1 to 2 in 910B).
   */
  __aicore__ inline KernelRowScanBatch(uint16_t matmul_k_size,
                                       uint16_t matmul_m_size, uint32_t vec_len,
                                       uint32_t vec_cube_ratio)
      : matmul_k_size_(matmul_k_size),
        matmul_m_size_(matmul_m_size),
        num_tiles_per_vec_(vec_len / (matmul_m_size_ * matmul_k_size_)),
        num_vec_(vec_cube_ratio) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] a Pointer to input vectors in global memory.
   * @param [in] b Pointer to upper triangular matrix filled with ones in
   * global memory.
   * @param [in] c Pointer to output vectors in global memory.
   */
  __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c) {
    global_A_.SetGlobalBuffer(
        (__gm__ half *)a +
            (GetBlockIdx() * a_cube_tile_size_ * num_tiles_per_vec_ * num_vec_),
        a_cube_tile_size_ * num_tiles_per_vec_ * num_vec_);
    global_B_.SetGlobalBuffer((__gm__ half *)b);
    global_C_.SetGlobalBuffer(
        (__gm__ float *)c +
            (GetBlockIdx() * c_cube_tile_size_ * num_tiles_per_vec_ * num_vec_),
        c_cube_tile_size_ * num_tiles_per_vec_ * num_vec_);

    pipe.InitBuffer(a1_q_, 1, a_cube_tile_size_ * sizeof(half));
    pipe.InitBuffer(a2_q_, 1, a_cube_tile_size_ * sizeof(half));
    pipe.InitBuffer(b1_q_, 1, b_cube_tile_size_ * sizeof(half));
    pipe.InitBuffer(b2_q_, 1, b_cube_tile_size_ * sizeof(half));

    pipe.InitBuffer(co1_q_, 1, c_cube_tile_size_ * sizeof(float));
  }

  /**
   * @brief Run the kernel - process all tiles.
   */
  __aicore__ inline void Process() {
    // Load the B matrix only once from global memory
    LoadBToL0();
    for (uint32_t idx = 0; idx < num_tiles_per_vec_; ++idx) {
      // Compute multiple output vectors at the time
      for (uint32_t i = 0; i < num_vec_; i++)
        CubeIter(idx + i * num_tiles_per_vec_);
      sync::SyncGroup<sync::GroupSyncDirection::FULL>();
    }
    queue::FreeFromQ<half>(b2_q_);
  }

 private:
  __aicore__ inline void CubeIter(uint32_t iter_idx) {
    copy::CopyGmToL1A(a1_q_, global_A_[iter_idx * a_cube_tile_size_], m_blocks_,
                      k_blocks_);
    LoadA1ToA2();
    cube_unit::Multiply<half, false /* accumulate_c */, true /* free_a*/,
                        false /* free_b */>(a2_q_, b2_q_, co1_q_,
                                            matmul_m_size_, N_, K_);
    copy::CopyCL0ToGlobal(global_C_[iter_idx * c_cube_tile_size_], co1_q_,
                          matmul_m_size_, N_);
  }
  __aicore__ inline void LoadBToL0() {
    copy::CopyTransposedGmToL0B(b2_q_, b1_q_, global_B_, k_blocks_, n_blocks_);
  }

  __aicore__ inline void LoadA1ToA2() {
    copy::CopyL1ToL0A<half, true>(a2_q_, a1_q_, m_blocks_, k_blocks_);
  }

  TPipe pipe;

  TQue<QuePosition::A1, 1> a1_q_;
  TQue<QuePosition::A2, 1> a2_q_;
  TQue<QuePosition::B1, 1> b1_q_;
  TQue<QuePosition::B2, 1> b2_q_;

  TQue<QuePosition::CO1, 1> co1_q_;

  GlobalTensor<half> global_A_, global_B_;
  GlobalTensor<float> global_C_;

  const uint16_t matmul_k_size_;
  const uint16_t matmul_m_size_;
  const uint16_t cube_block_size_ = 16;
  const uint16_t K_ = matmul_k_size_;
  const uint16_t N_ = matmul_k_size_;

  const uint16_t n_blocks_ = N_ / cube_block_size_;
  const uint16_t k_blocks_ = K_ / cube_block_size_;
  const uint16_t m_blocks_ = matmul_m_size_ / cube_block_size_;

  const uint32_t a_cube_tile_size_ = matmul_m_size_ * K_;
  const uint32_t b_cube_tile_size_ = K_ * N_;
  const uint32_t c_cube_tile_size_ = matmul_m_size_ * N_;

  const uint32_t num_tiles_per_vec_;
  const uint32_t num_vec_;
};

/**
 * @brief Transforms an inclusive row-wise scan of multiple vectors into an
 * inclusive scan of them.
 *
 * The algorithm assigns each independent vector to a vector core.
 *
 * The algorithm takes an inclusive row-wise scan (row size being equal to
 * `tile_width`) of multiple input vectors (for example the ones produced by
 * `KernelRowScan`). Then the algorithm transforms the row-wise scan into a
 * full inclusive scan by iterating over chunks of size `tile_width` and
 * adding to them the sum of all the previous chunks.
 */
class KernelCompleteRowsBatched {
 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] tile_width Size of the row used by `KernelRowScan`.
   * @param [in] tile_height Number of rows processed in a single
   * iteration.
   * @param [in] vec_len Number of elements in a single input vector.
   * @param [in] vec_cube_ratio Specifies the vector to cube ratio to use
   * (ranges from 1 to 2 in 910B).
   */
  __aicore__ inline KernelCompleteRowsBatched(uint16_t tile_width,
                                              uint16_t tile_height,
                                              uint32_t vec_len,
                                              uint32_t vec_cube_ratio)
      : tile_width_(tile_width),
        tile_height_(tile_height),
        tile_size_(tile_width * tile_height),
        num_tiles_(vec_len / tile_size_),
        vec_cube_ratio_(vec_cube_ratio) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] input Pointer to row-wise scan in global memory.
   * @param [in] output Pointer to output vectors in global memory.
   */
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
    global_input_.SetGlobalBuffer(
        (__gm__ float *)input + (GetBlockIdx() * tile_size_ * num_tiles_ *
                                 vec_cube_ratio_ / GetTaskRation()),
        tile_size_ * num_tiles_);
    global_output_.SetGlobalBuffer(
        (__gm__ float *)output + (GetBlockIdx() * tile_size_ * num_tiles_ *
                                  vec_cube_ratio_ / GetTaskRation()),
        tile_size_ * num_tiles_);

    pipe.InitBuffer(vecin_q_, 1, tile_size_ * sizeof(float));
    pipe.InitBuffer(vecout_q_, 1, tile_size_ * sizeof(float));
    pipe.InitBuffer(work_buf_, tile_size_ * sizeof(float));
  }

  /**
   * @brief Run the kernel - process all tiles.
   */
  __aicore__ inline void Process() {
    float running_sum = 0.f;
    for (uint32_t idx = 0; idx < num_tiles_; ++idx) {
      // Run the reduction on all vector cores

      sync::SyncGroup<sync::GroupSyncDirection::FULL>();
      if (vec_cube_ratio_ == 1)
        if (GetBlockIdx() % 2) continue;

      running_sum = VecIter(idx, running_sum);
    }
  }

 private:
  __aicore__ inline float VecIter(uint32_t iter_idx, float initial_sum) {
    copy::CopyGmToVec(vecin_q_, global_input_[iter_idx * tile_size_]);
    const float sum = ReduceWithVec(initial_sum);
    StoreVecToGlobal(iter_idx);
    return sum;
  }

  __aicore__ inline float ReduceWithVec(float initial_sum) {
    LocalTensor<float> vec_lt = vecin_q_.DeQue<float>();
    // Get 2 local tensors pointing to the same buffer: this is needed to
    // trick the compiler into considering operations on vec_buf1 and
    // vec_buf2 independent.
    const LocalTensor<float> vec_buf1 = work_buf_.Get<float>();
    const LocalTensor<float> vec_buf2 = work_buf_.Get<float>();
    DataCopy(vec_buf1, vec_lt, vec_lt.GetSize());
    vecin_q_.FreeTensor(vec_lt);

    sync::ScalarWaitForVec();

    uint32_t first_offset = 0;
    uint32_t second_offset = tile_width_;
    float first_sum = initial_sum;
    float second_sum = vec_buf1.GetValue(second_offset - 1) + first_sum;
    for (uint32_t i = 0; i < tile_height_; i += 2) {
      // The Adds instructions can be overlapped because they are
      // independent.
      Adds(vec_buf1[first_offset], vec_buf1[first_offset], first_sum,
           tile_width_);
      Adds(vec_buf2[second_offset], vec_buf2[second_offset], second_sum,
           tile_width_);
      first_offset += tile_width_ * 2;
      second_offset += tile_width_ * 2;
      first_sum = vec_buf2.GetValue(first_offset - 1);
      if (i + 2 < tile_height_)
        second_sum = vec_buf1.GetValue(second_offset - 1) + first_sum;
    }

    return first_sum;
  }

  __aicore__ inline void StoreVecToGlobal(uint32_t iter_idx) {
    const uint32_t dst_offset = iter_idx * tile_size_;
    const LocalTensor<float> lt = work_buf_.Get<float>();
    copy::CopyVecToGm<float>(global_output_[dst_offset], vecout_q_, lt);
  }

  TPipe pipe;

  TQue<QuePosition::VECIN, 1> vecin_q_;
  TQue<QuePosition::VECOUT, 1> vecout_q_;
  TBuf<QuePosition::VECCALC> work_buf_;

  GlobalTensor<float> global_input_;
  GlobalTensor<float> global_output_;

  const uint16_t tile_width_;
  const uint16_t tile_height_;
  const uint32_t tile_size_;
  const uint32_t num_tiles_;
  const uint32_t vec_cube_ratio_;
};

/**
 * @brief Run the batched scan kernel.
 *
 * @param [in] input_vec Pointer to an input tensor.
 * @param [in] upper_triangular Pointer to an upper-triangular matrix filled
 * with ones of size \f$\textit{matmul_size} \times \textit{matmul_size}\f$.
 * @param [in] output_vec Pointer to an output tensor.
 * @param [in] vec_len Number of elements in a single vector.
 * @param [in] batch_size Number of vectors in a batch.
 * @param [in] matmul_size Size of the matmul tile used in the kernel.
 * @param [in] vec_cube_ratio Specifies the vector to cube ratio to use (ranges
 * from 1 to 2 in 910B).
 */

__aicore__ inline void run_scan_batch_kernel(
    GM_ADDR input_vec, GM_ADDR upper_triangular, GM_ADDR output_vec,
    uint32_t vec_len, uint32_t batch_size, uint16_t matmul_size,
    uint32_t vec_cube_ratio) {
  const uint32_t block_n = GetBlockNum();
  const uint32_t num_vectors_to_process =
      scalar::GetBatchDistribution(batch_size / vec_cube_ratio, block_n);

  for (uint32_t i = 0; i < num_vectors_to_process; i++) {
    const uint32_t offset = vec_len * i * block_n * vec_cube_ratio;
    if ASCEND_IS_AIC {
      KernelRowScanBatch op_cube(matmul_size, matmul_size, vec_len,
                                 vec_cube_ratio);
      op_cube.Init(input_vec + offset * sizeof(half), upper_triangular,
                   output_vec + offset * sizeof(float));
      op_cube.Process();
    }

    if ASCEND_IS_AIV {
      KernelCompleteRowsBatched op_vec(matmul_size, matmul_size, vec_len,
                                       vec_cube_ratio);
      op_vec.Init(output_vec + offset * sizeof(float),
                  output_vec + offset * sizeof(float));
      op_vec.Process();
    }
  }
}
