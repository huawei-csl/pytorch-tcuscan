/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_row_scan.h
 * @brief Kernel implementing row scan using right matrix multiplication by U_s
 (upper tringular all-ones matrix).
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "kernel_utils.h"

using namespace AscendC;
using namespace kernel_utils;

/**
 * @brief Performs a "row-wise" inclusive scan on an input vector.
 *
 * The algorithm splits the input vector into chunks ("rows") of size
 * \f$K = \textit{matmul_k_size}\f$ and performs local inclusive scans on all
 * the chunks separately. Local scan is performed by transforming the problem
 * into matrix multiplication. Matrix \f$B\f$ is an upper triangular matrix of
 * size \f$K \times K\f$ with ones on the main diagonal and above. Matrix
 * \f$A\f$ is created from the input vector by reshaping it into a matrix with
 * \f$K\f$ dimension equal to \f$K\f$ and \f$M\f$ dimension equal to
 * \f$\textit{matmul_m_size}\f$. Then matrix \f$A\f$ is splitted into \f$M
 * \times K\f$ tiles that are multiplied by the same matrix \f$B\f$. The tiles
 * are equally distributed into different cube cores.
 *
 * The algorithm assumes that the length of the input vector is divisible by
 * \f$M K\f$ times the number of blocks.
 *
 * The algorithm is a first step in the full-scan calculation.
 *
 * The algorithm supports following data type configurations:
 *   - inputs: `int8_t`; output: `int32_t`;
 *   - inputs: `uint8_t`; output: `uint32_t`;
 *   - inputs: `half`; output: `float`.
 *
 * @tparam InputT Data type of the input vector.
 */
template <typename InputT>
class KernelRowScan {
  using OutputT = kernel_utils::cube_unit::CubeOutType_t<InputT>;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] matmul_k_size Size of the K dimension of A matrix.
   * @param [in] matmul_m_size Size of the M dimension of A matrix.
   * @param [in] vec_len Number of elements in an input vector.
   */
  __aicore__ inline KernelRowScan(uint32_t matmul_k_size,
                                  uint32_t matmul_m_size, uint32_t vec_len)
      : block_num_(GetBlockNum()),
        matmul_k_size_(matmul_k_size),
        matmul_m_size_(matmul_m_size),
        tile_size_(matmul_k_size * matmul_m_size),
        vec_len_(vec_len),
        num_tiles_(scalar::FloorDiv(vec_len, tile_size_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, block_num_)) {
    static_assert(kernel_utils::cube_unit::IsCubeSupported<InputT>,
                  "Unsupported input Cube dtype. Please use half or int8_t.");
    ASCENDC_ASSERT((vec_len % (matmul_m_size_ * matmul_k_size_) == 0), {
      KERNEL_LOG(KERNEL_ERROR,
                 "The length of the input vector (%d) must be "
                 "divisible by the tile size (%d)",
                 vec_len, matmul_m_size_ * matmul_k_size_);
    });
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] input Pointer to input vector in global memory.
   * @param [in] b Pointer to upper triangular matrix filled with ones in
   * global memory.
   * @param [in] output Pointer to output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR b, GM_ADDR output) {
    global_A_.SetGlobalBuffer((__gm__ InputT *)input, vec_len_);
    global_B_.SetGlobalBuffer((__gm__ InputT *)b);
    global_C_.SetGlobalBuffer((__gm__ OutputT *)output, vec_len_);

    pipe.InitBuffer(a1_q_, 1, a_cube_tile_size_ * sizeof(InputT));
    pipe.InitBuffer(a2_q_, 1, a_cube_tile_size_ * sizeof(InputT));
    pipe.InitBuffer(b1_q_, 1, b_cube_tile_size_ * sizeof(InputT));
    pipe.InitBuffer(b2_q_, 1, b_cube_tile_size_ * sizeof(InputT));

    pipe.InitBuffer(co1_q_, 1, c_cube_tile_size_ * sizeof(OutputT));
  }

  /**
   * @brief Run the kernel - process all tiles.
   */
  __aicore__ inline void Process() {
    // Load the B matrix only once from global memory
    const uint32_t num_tiles_to_process =
        kernel_utils::scalar::GetWorkDistribution(vec_len_, tile_size_,
                                                  block_num_);

    LoadBToL0();
    for (uint32_t idx = 0; idx < num_tiles_to_process; ++idx) {
      CubeIter(idx);
    }
    queue::FreeFromQ<InputT>(b2_q_);
  }

 private:
  __aicore__ inline void CubeIter(uint32_t iter_idx) {
    copy::CopyGmToL1A(
        a1_q_,
        global_A_[GetBlockIdx() * a_cube_tile_size_ * max_num_tiles_per_block_ +
                  iter_idx * a_cube_tile_size_],
        m_blocks_, k_blocks_);
    LoadA1ToA2();
    cube_unit::Multiply<InputT, false /* accumulate_c */, true /* free_a*/,
                        false /* free_b */>(a2_q_, b2_q_, co1_q_,
                                            matmul_m_size_, N_, K_);
    copy::CopyCL0ToGlobal(
        global_C_[GetBlockIdx() * c_cube_tile_size_ * max_num_tiles_per_block_ +
                  iter_idx * c_cube_tile_size_],
        co1_q_, matmul_m_size_, N_);
  }

  __aicore__ inline void LoadBToL0() {
    copy::CopyTransposedGmToL0B(b2_q_, b1_q_, global_B_, k_blocks_, n_blocks_);
  }

  __aicore__ inline void LoadA1ToA2() {
    copy::CopyL1ToL0A<InputT, true>(a2_q_, a1_q_, m_blocks_, k_blocks_);
  }

  TPipe pipe;

  TQue<QuePosition::A1, 1> a1_q_;
  TQue<QuePosition::A2, 1> a2_q_;
  TQue<QuePosition::B1, 1> b1_q_;
  TQue<QuePosition::B2, 1> b2_q_;

  TQue<QuePosition::CO1, 1> co1_q_;

  GlobalTensor<InputT> global_A_, global_B_;
  GlobalTensor<OutputT> global_C_;

  const uint32_t block_num_;
  const uint32_t matmul_k_size_;
  const uint32_t matmul_m_size_;
  const uint32_t tile_size_;
  const uint32_t vec_len_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;

  constexpr static uint32_t M_CUBE_BLOCK_SIZE =
      kernel_utils::GetFractalMN<InputT>();
  constexpr static uint32_t N_CUBE_BLOCK_SIZE =
      kernel_utils::GetFractalMN<InputT>();
  // Fractal size is 16x32 for 8-bit input data types instead of the standard
  // 16x16 one
  constexpr static uint32_t K_CUBE_BLOCK_SIZE =
      kernel_utils::GetFractalK<InputT>();
  const uint32_t K_ = matmul_k_size_;
  const uint32_t N_ = matmul_k_size_;

  const uint32_t n_blocks_ = N_ / N_CUBE_BLOCK_SIZE;
  const uint32_t k_blocks_ = K_ / K_CUBE_BLOCK_SIZE;
  const uint32_t m_blocks_ = matmul_m_size_ / M_CUBE_BLOCK_SIZE;

  const uint32_t a_cube_tile_size_ = matmul_m_size_ * K_;
  const uint32_t b_cube_tile_size_ = K_ * N_;
  const uint32_t c_cube_tile_size_ = matmul_m_size_ * N_;
};
