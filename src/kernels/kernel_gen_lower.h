/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
 *
 * @file kernel_gen_lower.h
 * @brief Kernel generating a lower triangular matrix in global memory.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;
using namespace kernel_utils;

namespace tcuscan {

/**
 * @brief Returns a torch.tril(torch.ones(s, s)) for matrix length s.
 *
 */
template <typename T = half>
class KernelGenerateLower {
  constexpr static bool NEEDS_CAST =
      std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>;
  using IntermediateT = half;
  using DuplicateT =
      typename std::conditional<NEEDS_CAST, IntermediateT, T>::type;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] matrix_size Size of triangular square matrix.
   *
   */
  __aicore__ inline KernelGenerateLower(uint32_t matrix_size)
      : block_num_(GetBlockNum() * GetTaskRation()),
        matrix_size_(matrix_size),
        total_matrix_elements_(matrix_size_ * matrix_size_) {
    rows_per_block_ = scalar::CeilDiv(matrix_size_, block_num_);
    num_matrix_elements_per_block_ = rows_per_block_ * matrix_size_;
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_out Pointer to the output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR vec_out) {
    global_out_.SetGlobalBuffer((__gm__ T *)vec_out, total_matrix_elements_);

    pipe.InitBuffer(vec_buf_,
                    num_matrix_elements_per_block_ * sizeof(DuplicateT));

    pipe.InitBuffer(out_q_, NUM_BUFFERS,
                    num_matrix_elements_per_block_ * sizeof(T));
  }

  /**
   * @brief Run the kernel.
   *
   */
  __aicore__ inline void Process() {
    if (GetBlockIdx() >= block_num_) {
      return;
    }
    const LocalTensor<DuplicateT> lt = vec_buf_.Get<DuplicateT>();
    uint32_t lt_offset = 0;
    Duplicate(lt, static_cast<DuplicateT>(0), lt.GetSize());

    const uint32_t start = rows_per_block_ * GetBlockIdx();
    const uint32_t end = scalar::Min(start + rows_per_block_, matrix_size_);
    for (uint32_t row_idx = start; row_idx < end; ++row_idx) {
      const uint32_t num_ones_in_row = row_idx + 1;
      Duplicate(lt[lt_offset], static_cast<DuplicateT>(1), num_ones_in_row);
      lt_offset += matrix_size_;
    }

    const LocalTensor<T> res_lt = out_q_.template AllocTensor<T>();
    if constexpr (NEEDS_CAST) {
      Cast(res_lt, lt, RoundMode::CAST_NONE, num_matrix_elements_per_block_);
    } else {
      DataCopy<T>(res_lt, lt, num_matrix_elements_per_block_);
    }
    out_q_.EnQue<T>(res_lt);

    copy::CopyVecToGm(
        global_out_[num_matrix_elements_per_block_ * GetBlockIdx()], out_q_,
        num_matrix_elements_per_block_);
  }

 private:
  TPipe pipe;

  constexpr static uint32_t NUM_BUFFERS = 1;

  GlobalTensor<T> global_out_;
  TBuf<TPosition::VECCALC> vec_buf_;
  TQue<QuePosition::VECOUT, NUM_BUFFERS> out_q_;

  const uint32_t block_num_;
  const uint32_t matrix_size_;
  const uint32_t total_matrix_elements_;
  uint32_t rows_per_block_;
  uint32_t num_matrix_elements_per_block_;
};

/**
 * @brief Run the `gen_lower` kernel.
 *
 * @tparam T Data type of the matrix elements.
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 *
 * @param [in] dst Pointer to the destination buffer in global memory.
 * @param [in] matrix_size Size of the matrix row and column.
 */
template <typename T, bool ForceMixMode = true>
__aicore__ inline void run_gen_lower(GM_ADDR dst, uint32_t matrix_size) {
  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }
  if ASCEND_IS_AIV {
    KernelGenerateLower<T> op_gen(matrix_size);
    op_gen.Init(dst);
    op_gen.Process();
  }
}

}  // namespace tcuscan
