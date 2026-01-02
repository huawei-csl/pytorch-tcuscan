/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file kernel_count_if.h
 * @brief Kernel implementing a Vector `count_if` kernel operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;
using namespace kernel_utils;

namespace tcuscan {

/**
 * @brief Returns `torch.count_nonzero(x <= pivot)` given an input vector.
 *
 */
template <typename T = half>
class KernelCountIf {
  /// @brief Accumulation data type for the counts.
  using AccT = int32_t;
  /// @brief Minimum size of queue size (UB restriction)
  constexpr static int32_t MIN_VEC_SIZE = UB_ALIGNMENT / sizeof(AccT);

  constexpr static int32_t RED2_SIZE = 8;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Dimension of the input vectors.
   * @param [in] tile_len Tile length.
   */
  __aicore__ inline KernelCountIf(uint32_t vec_len, uint32_t tile_len)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        vec_len_(vec_len),
        tile_len_(tile_len),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_len_)) {
    ASCENDC_ASSERT(vec_len_ < tile_len_ * vec_core_num_, {
      KERNEL_LOG(KERNEL_ERROR,
                 "KernelCountIf only works for input vectors of size at most "
                 "(number of AIV cores) x (maximum UB size, ~ 10K elements). "
                 "Got (tile_len=%d, AIVs=%d)",
                 tile_len_, vec_core_num_);
    });
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_in Pointer to the input vector in global memory.
   * @param [in] vec_out Pointer to the output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR vec_out) {
    const uint32_t out_queue_len =
        vec_core_num_ < MIN_VEC_SIZE ? MIN_VEC_SIZE : vec_core_num_;

    global_in_.SetGlobalBuffer((__gm__ T*)vec_in, vec_len_);
    global_out_.SetGlobalBuffer((__gm__ int32_t*)vec_out, vec_core_num_);

    pipe_.InitBuffer(in_q_, 1, tile_len_ * sizeof(T));
    pipe_.InitBuffer(out_q_, 1, out_queue_len * sizeof(int32_t));

    pipe_.InitBuffer(vec_in_buf_, tile_len_ * sizeof(T));
    pipe_.InitBuffer(work_buf_, tile_len_ * sizeof(AccT));
    pipe_.InitBuffer(comparisons_buf_, tile_len_ * sizeof(AccT));
    pipe_.InitBuffer(red2_buf_, RED2_SIZE * sizeof(AccT));
  }

  /**
   * @brief Load input tiles into UB.
   *
   * Each AIV core loads one tile inside `work_buf_` TBuf.
   */
  __aicore__ inline void LoadInputTileInUB() {
    const uint32_t global_offset = GetBlockIdx() * tile_len_;

    copy::CopyGmToVec(in_q_, global_in_[global_offset], tile_len_);

    LocalTensor<T> lt = in_q_.DeQue<T>();
    const LocalTensor<T> vec_in_lt = vec_in_buf_.Get<T>();
    DataCopy(vec_in_lt, lt, tile_len_);
    in_q_.FreeTensor(lt);
  }

  /**
   * @brief Run the kernel.
   *
   * @param [in] pivot Pivot on which to count the cardinality of the set {x_i <
   * pivot} where \f$ x_i \f$ are the input elements.
   *
   */
  __aicore__ inline void LessThan(T pivot) {
    LocalTensor<T> tile_lt = vec_in_buf_.Get<T>();
    LocalTensor<T> work_lt = work_buf_.Get<T>();
    LocalTensor<uint8_t> binary_lt = comparisons_buf_.Get<uint8_t>();

    compare::LessThan<T, uint8_t>(binary_lt, tile_lt, pivot, work_lt);

    Duplicate(work_lt, static_cast<T>(0), work_lt.GetSize());
    AscendC::PipeBarrier<PIPE_V>();

    LocalTensor<half> count_fp16_lt = work_lt.template ReinterpretCast<half>();
    AscendC::Cast(count_fp16_lt, binary_lt, RoundMode::CAST_NONE, tile_len_);

    LocalTensor<int32_t> count_i32_lt =
        work_lt.template ReinterpretCast<int32_t>();
    AscendC::Cast(count_i32_lt, count_fp16_lt, RoundMode::CAST_RINT, tile_len_);

    const AccT num_elems_less_than = ReduceToScalar(count_i32_lt, tile_len_);
    const auto id = GetBlockIdx();
    copy::CopyScalarToGm(global_out_[id], out_q_, num_elems_less_than);
  }

  /**
   * @brief Returns the sum-reduction of the input vector.
   *
   * @param [in] binary_lt Input tensor
   * @param [in] length Input length
   * @return Sum-reduction of first `length` elements of input
   **/
  __aicore__ inline AccT ReduceToScalar(const LocalTensor<AccT>& binary_lt,
                                        uint32_t length) {
    const LocalTensor<AccT> red2_lt = red2_buf_.Get<AccT>();
    reduce::ReduceVecAdd<true /*AllocAcc*/, AccT>(red2_lt, binary_lt, length);
    return reduce::ReduceScalarAdd<AccT>(red2_lt, red2_lt.GetSize());
  }

 private:
  TPipe pipe_;

  TQue<QuePosition::VECIN, 1> in_q_;
  TQue<QuePosition::VECOUT, 1> out_q_;

  TBuf<QuePosition::VECCALC> vec_in_buf_;
  TBuf<QuePosition::VECCALC> work_buf_;
  TBuf<QuePosition::VECCALC> red2_buf_;
  TBuf<QuePosition::VECCALC> comparisons_buf_;

  GlobalTensor<T> global_in_;
  GlobalTensor<int32_t> global_out_;

  const uint32_t vec_core_num_;
  const uint32_t vec_len_;
  const uint32_t tile_len_;
  const uint32_t num_tiles_;
};

/**
 * @brief Run the `count_if` kernel.
 *
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] vec_len Dimension of the input vector.
 * @param [in] tile_len Tile length.
 * @param [in] pivot Initial pivot.
 */
template <bool ForceMixMode = true, typename T>
__aicore__ inline void run_count_if(GM_ADDR vec_in, GM_ADDR vec_out,
                                    uint32_t vec_len, uint32_t tile_len,
                                    T pivot) {
  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }
  if ASCEND_IS_AIV {
    KernelCountIf<T> op(vec_len, tile_len);
    op.Init(vec_in, vec_out);
    op.LoadInputTileInUB();
    op.LessThan(static_cast<float>(pivot));

    // TODO: binary search on UB
    // constexpr uint32_t recursion_steps = 4;
    // for (uint32_t idx = 0; idx < recursion_steps; idx++) {
    //   op.LessThan(((float)0.5 + (float)pivot));
    //  SyncAll<true /* isAIVOnly */>();
    //}
  }
}

}  // namespace tcuscan
