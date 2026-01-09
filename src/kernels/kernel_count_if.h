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
  using PackedMaskT = uint8_t;
  using GatherMaskT =
      typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type;
  constexpr static uint16_t IN_ELEMS_PER_MASK_ELEM = sizeof(PackedMaskT) * 8;

  /// @brief Minimum size of queue size (UB restriction)
  constexpr static int32_t MIN_VEC_SIZE = UB_ALIGNMENT / sizeof(AccT);

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
    global_out_.SetGlobalBuffer((__gm__ int32_t*)vec_out, 1);

    pipe_.InitBuffer(in_q_, 1, tile_len_ * sizeof(T));
    pipe_.InitBuffer(out_q_, 1, out_queue_len * sizeof(int32_t));

    pipe_.InitBuffer(vec_in_buf_, tile_len_ * sizeof(T));
    pipe_.InitBuffer(work_buf_, tile_len_ * sizeof(T));
    pipe_.InitBuffer(packed_mask_buf_, tile_len_ * sizeof(PackedMaskT) / 8);
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

    const LocalTensor<uint8_t> packed_mask_8b = packed_mask_buf_.Get<uint8_t>();
    CompareScalar(packed_mask_8b, tile_lt, pivot, CMPMODE::LE, tile_len_);

    const LocalTensor<GatherMaskT> mask_lt =
        packed_mask_buf_.Get<GatherMaskT>();

    uint64_t num_gathered_elems = 0;
    GatherMask(work_lt, tile_lt, mask_lt, true, tile_len_, {1, 1, 8, 8},
               num_gathered_elems);

    AtomicAddWrite(static_cast<int32_t>(num_gathered_elems));
  }

 private:
  __aicore__ inline void AtomicAddWrite(int32_t value) {
    AscendC::PipeBarrier<PIPE_MTE3>();
    AscendC::SetAtomicAdd<int32_t>();
    copy::CopyScalarToGm(global_out_, out_q_, value);
    AscendC::SetAtomicNone();
  }

  TPipe pipe_;

  TQue<QuePosition::VECIN, 1> in_q_;
  TQue<QuePosition::VECOUT, 1> out_q_;

  TBuf<QuePosition::VECCALC> vec_in_buf_;
  TBuf<QuePosition::VECCALC> work_buf_;
  TBuf<QuePosition::VECCALC> packed_mask_buf_;

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
    op.LessThan(pivot);

    // TODO: binary search on UB
    // constexpr uint32_t recursion_steps = 4;
    // for (uint32_t idx = 0; idx < recursion_steps; idx++) {
    //   op.LessThan(((float)0.5 + (float)pivot));
    //  SyncAll<true /* isAIVOnly */>();
    //}
  }
}

}  // namespace tcuscan
