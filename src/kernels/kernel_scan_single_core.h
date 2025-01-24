/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_scan_single_core.h
 * @brief Kernel implementing a single-core scan using cube-vector approach.
 */

#include "ascendc_kernel_operator.h"
#include "kernel_pad.h"
#include "kernel_utils.h"

using namespace AscendC;
using namespace kernel_utils;
/**
 * @brief Performs a "row-wise" inclusive-scan on an input vector.
 *
 * The algorithm utilizes only a single cube core.
 *
 * The algorithm splits the input vector into chunks ("rows") of size
 * \f$K = \textit{matmul_k_size}\f$ and performs local inclusive scans on all
 * the chunks separately. Local scan is performed by transforming the problem
 * into matrix multiplication. Matrix \f$B\f$ is an upper triangular matrix of
 * size \f$K \times K\f$ with ones on the main diagonal and above. Matrix
 * \f$A\f$ is created from the input vector by reshaping it into a matrix with
 * \f$K\f$ dimension equal to \f$K\f$ and \f$M\f$ dimension equal to
 * \f$\textit{matmul_m_size}\f$. Then matrix \f$A\f$ is splitted into \f$M
 * \times K\f$ tiles that are multiplied by the same matrix \f$B\f$.
 *
 * The algorithm assumes that the length of the input vector is divisible by
 * \f$M K\f$.
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
  __aicore__ inline KernelRowScan(uint16_t matmul_k_size,
                                  uint16_t matmul_m_size, uint32_t vec_len)
      : matmul_k_size_(matmul_k_size),
        matmul_m_size_(matmul_m_size),
        num_tiles_(vec_len / (matmul_m_size_ * matmul_k_size_)) {
    static_assert(kernel_utils::cube_unit::IsCubeSupported<InputT>,
                  "Unsupported input Cube dtype. Please use half or int8_t.");
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] a Pointer to input vector in global memory.
   * @param [in] b Pointer to upper triangular matrix filled with ones in
   * global memory.
   * @param [in] c Pointer to output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c) {
    global_A_.SetGlobalBuffer((__gm__ InputT *)a);
    global_B_.SetGlobalBuffer((__gm__ InputT *)b);
    global_C_.SetGlobalBuffer((__gm__ OutputT *)c);

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
    LoadBToL0();
    for (uint32_t idx = 0; idx < num_tiles_; ++idx) {
      CubeIter(idx);
      sync::SyncGroup<sync::GroupSyncDirection::FULL>();
    }
    queue::FreeFromQ<InputT>(b2_q_);
  }

 private:
  __aicore__ inline void CubeIter(uint32_t iter_idx) {
    copy::CopyGmToL1A(a1_q_, global_A_[iter_idx * a_cube_tile_size_], m_blocks_,
                      k_blocks_);
    LoadA1ToA2();
    cube_unit::Multiply<InputT, false /* accumulate_c */, true /* free_a*/,
                        false /* free_b */>(a2_q_, b2_q_, co1_q_,
                                            matmul_m_size_, N_, K_);
    copy::CopyCL0ToGlobal(global_C_[iter_idx * c_cube_tile_size_], co1_q_,
                          matmul_m_size_, N_);
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

  const uint16_t matmul_k_size_;
  const uint16_t matmul_m_size_;
  const uint32_t num_tiles_;

  constexpr static uint16_t M_CUBE_BLOCK_SIZE =
      kernel_utils::GetFractalMN<InputT>();
  constexpr static uint16_t N_CUBE_BLOCK_SIZE =
      kernel_utils::GetFractalMN<InputT>();
  // Fractal size is 16x32 for 8-bit dtypes, instead of 16x16
  constexpr static uint16_t K_CUBE_BLOCK_SIZE =
      kernel_utils::GetFractalK<InputT>();
  const uint16_t K_ = matmul_k_size_;
  const uint16_t N_ = matmul_k_size_;

  const uint16_t n_blocks_ = N_ / N_CUBE_BLOCK_SIZE;
  const uint16_t k_blocks_ = K_ / K_CUBE_BLOCK_SIZE;
  const uint16_t m_blocks_ = matmul_m_size_ / M_CUBE_BLOCK_SIZE;

  const uint32_t a_cube_tile_size_ = matmul_m_size_ * K_;
  const uint32_t b_cube_tile_size_ = K_ * N_;
  const uint32_t c_cube_tile_size_ = matmul_m_size_ * N_;
};

/**
 * @brief Transforms an inclusive row-wise scan of a vector into an
 * inclusive scan of it.
 *
 * The algorithm utilizes only a single vector core.
 *
 * The algorithm takes an inclusive row-wise scan (row size being equal to
 * `tile_width`) of an input vector (for example the one produced by
 * `KernelRowScan`). Then the algorithm transforms the row-wise scan into a
 * full inclusive scan by iterating over chunks of size `tile_width` and
 * adding to them the sum of all the previous chunks.
 *
 * @tparam InputT Data type of the input vector.
 */
template <typename InputT>
class KernelCompleteRows {
 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] tile_width Size of the row used by `KernelRowScan`.
   * @param [in] tile_height Number of rows processed in a single
   * iteration.
   * @param [in] vec_len Number of elements in an input vector.
   */
  __aicore__ inline KernelCompleteRows(uint16_t tile_width,
                                       uint16_t tile_height, uint32_t vec_len)
      : tile_width_(tile_width),
        tile_height_(tile_height),
        vec_len_(vec_len),
        tile_size_(tile_width * tile_height),
        num_tiles_((vec_len_ + tile_size_ - 1) / tile_size_) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] input Pointer to row-wise scan in global memory.
   * @param [in] output Pointer to output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
    global_input_.SetGlobalBuffer((__gm__ InputT *)input);
    global_output_.SetGlobalBuffer((__gm__ InputT *)output);

    pipe.InitBuffer(vecin_q_, 1, tile_size_ * sizeof(InputT));
    pipe.InitBuffer(vecout_q_, 1, tile_size_ * sizeof(InputT));
    pipe.InitBuffer(work_buf_, tile_size_ * sizeof(InputT));
  }

  /**
   * @brief Run the kernel - process all tiles.
   */
  __aicore__ inline void Process() {
    InputT running_sum = 0;
    for (uint32_t idx = 0; idx < num_tiles_; ++idx) {
      sync::SyncGroup<sync::GroupSyncDirection::FULL>();
      if (GetBlockIdx() == 0) {
        running_sum = VecIter(idx, running_sum);
      }
    }
  }

 private:
  __aicore__ inline InputT VecIter(uint32_t iter_idx, InputT initial_sum) {
    copy::CopyGmToVec(vecin_q_, global_input_[iter_idx * tile_size_]);
    const InputT sum = ReduceWithVec(initial_sum);
    StoreVecToGlobal(iter_idx);
    return sum;
  }

  __aicore__ inline InputT ReduceWithVec(InputT initial_sum) {
    LocalTensor<InputT> vec_lt = vecin_q_.DeQue<InputT>();
    // Get 2 local tensors pointing to the same buffer: this is needed to
    // trick the compiler into considering operations on vec_buf1 and
    // vec_buf2 independent.
    const LocalTensor<InputT> vec_buf1 = work_buf_.Get<InputT>();
    const LocalTensor<InputT> vec_buf2 = work_buf_.Get<InputT>();
    DataCopy(vec_buf1, vec_lt, vec_lt.GetSize());
    vecin_q_.FreeTensor(vec_lt);

    sync::ScalarWaitForVec();

    uint32_t first_offset = 0;
    uint32_t second_offset = tile_width_;
    InputT first_sum = initial_sum;
    InputT second_sum = vec_buf1.GetValue(second_offset - 1) + first_sum;
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
      if (i + 2 < tile_height_) {
        second_sum = vec_buf1.GetValue(second_offset - 1) + first_sum;
      }
    }

    return first_sum;
  }

  __aicore__ inline void StoreVecToGlobal(uint32_t iter_idx) {
    const uint32_t dst_offset = iter_idx * tile_size_;
    const LocalTensor<InputT> lt = work_buf_.Get<InputT>();
    if (iter_idx == num_tiles_ - 1 && vec_len_ % tile_size_) {
      copy::CopyVecToGm<InputT>(global_output_[dst_offset], vecout_q_, lt,
                                vec_len_ % tile_size_);
    } else {
      copy::CopyVecToGm<InputT>(global_output_[dst_offset], vecout_q_, lt);
    }
  }

  TPipe pipe;

  TQue<QuePosition::VECIN, 1> vecin_q_;
  TQue<QuePosition::VECOUT, 1> vecout_q_;
  TBuf<QuePosition::VECCALC> work_buf_;

  GlobalTensor<InputT> global_input_;
  GlobalTensor<InputT> global_output_;

  const uint16_t tile_width_;
  const uint16_t tile_height_;
  const uint32_t vec_len_;
  const uint32_t tile_size_;
  const uint32_t num_tiles_;
};

namespace sc_scan {

/**
 * @brief Calculate the workspace size for single core scan.
 *
 * @tparam InputT Input data type.
 *
 * @param [in] input_elems Number of elements in the input vector.
 * @param [in] matmul_size Size of the matmul used in scan.
 * @return Size of the workspace in bytes.
 */
template <typename InputT>
__aicore__ inline uint32_t get_workspace_size(uint32_t input_elems,
                                              uint32_t matmul_size) {
  const uint32_t align_size = matmul_size * matmul_size;
  const uint32_t padded_input_len = scalar::AlignUp(input_elems, align_size);

  const uint32_t padded_input_size = padded_input_len * sizeof(InputT);
  const uint32_t padded_rowwise_size =
      padded_input_len * sizeof(kernel_utils::cube_unit::CubeOutType_t<InputT>);

  return padded_input_size + padded_rowwise_size;
}

}  // namespace sc_scan

/**
 * @brief Run the single core scan kernel.
 *
 * @tparam InputT Input data type.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] upper_triangular Pointer to an upper-triangular matrix filled
 * with ones of size \f$\textit{matmul_size} \times \textit{matmul_size}\f$.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] vec_len Number of elements to process.
 * @param [in] matmul_size Size of the matmul tile used in the kernel.
 * @param [in] workspace Pointer to the kernel workspace.
 */
template <typename InputT>
__aicore__ inline void run_scan_single_core(
    GM_ADDR input_vec, GM_ADDR upper_triangular, GM_ADDR output_vec,
    uint32_t vec_len, uint16_t matmul_size, GM_ADDR workspace) {
  using OutputT = kernel_utils::cube_unit::CubeOutType_t<InputT>;

  if (vec_len % (matmul_size * matmul_size) || vec_len % UB_ALIGNMENT) {
    const uint32_t align_size = matmul_size * matmul_size;
    const uint32_t padded_vec_len = scalar::AlignUp(vec_len, align_size);

    GM_ADDR const padded_input = workspace;
    GM_ADDR const padded_rowwise_scan =
        workspace + padded_vec_len * sizeof(InputT);

    run_pad_kernel<InputT, false>(input_vec, padded_input, vec_len, align_size);

    sync::SyncGroup<sync::GroupSyncDirection::FULL>();

    if ASCEND_IS_AIC {
      KernelRowScan<InputT> op_cube(matmul_size, matmul_size, padded_vec_len);
      op_cube.Init(padded_input, upper_triangular, padded_rowwise_scan);
      op_cube.Process();
    }

    if ASCEND_IS_AIV {
      KernelCompleteRows<OutputT> op_vec(matmul_size, matmul_size, vec_len);
      op_vec.Init(padded_rowwise_scan, output_vec);
      op_vec.Process();
    }
  } else {
    if ASCEND_IS_AIC {
      KernelRowScan<InputT> op_cube(matmul_size, matmul_size, vec_len);
      op_cube.Init(input_vec, upper_triangular, output_vec);
      op_cube.Process();
    }

    if ASCEND_IS_AIV {
      KernelCompleteRows<OutputT> op_vec(matmul_size, matmul_size, vec_len);
      op_vec.Init(output_vec, output_vec);
      op_vec.Process();
    }
  }
}
