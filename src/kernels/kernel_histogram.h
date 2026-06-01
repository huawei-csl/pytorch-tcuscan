/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * @file kernel_histogram.h
 * @brief Kernel implementing a Vector histogram kernel operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Returns `torch.histogram(x, bins=num_bins)`.
 *
 * @tparam T Input data type. Supported types are `float16`.
 */
template <typename T>
class KernelHistogram {
  constexpr static uint32_t BUFFER_NUM = 2;

  using PackedMaskT = uint8_t;
  using GatherMaskT =
      typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Dimension of the input vectors.
   * @param [in] tile_len Tile length.
   * @param [in] num_bins Number of histogram bins.
   * @param [in] hist_min Smallest value where the histogram will be built on.
   * @param [in] hist_max Largest value on which the histogram will be built on.
   */
  __aicore__ inline KernelHistogram(uint32_t vec_len, uint32_t tile_len,
                                    uint32_t num_bins, float hist_min,
                                    float hist_max)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        vec_len_(vec_len),
        tile_len_(tile_len),
        num_bins_(num_bins),
        hist_min_(hist_min),
        hist_max_(hist_max),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_len_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, vec_core_num_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_in Pointer to the input vector in global memory.
   * @param [in] vec_out Pointer to the output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR vec_out) {
    global_in_.SetGlobalBuffer((__gm__ T*)vec_in, vec_len_);
    global_out_.SetGlobalBuffer((__gm__ int32_t*)vec_out, num_bins_);

    pipe_.InitBuffer(in_q_, BUFFER_NUM, tile_len_ * sizeof(T));
    pipe_.InitBuffer(out_q_, 1, num_bins_ * sizeof(int32_t));

    pipe_.InitBuffer(out_buf_, num_bins_ * sizeof(int32_t));
    pipe_.InitBuffer(work_buf_, tile_len_ * sizeof(T));
    pipe_.InitBuffer(packed_mask_buf_, tile_len_ * sizeof(PackedMaskT) / 8);
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    uint32_t global_offset =
        GetBlockIdx() * tile_len_ * max_num_tiles_per_block_;
    const uint32_t num_tiles_to_process =
        scalar::GetWorkDistribution(vec_len_, tile_len_, vec_core_num_);

    const LocalTensor<int32_t> vec_out_lt = out_q_.AllocTensor<int32_t>();
    AscendC::Duplicate<int32_t>(vec_out_lt, 0, num_bins_);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const bool full_tile = global_offset + tile_len_ <= vec_len_;
      const uint32_t num_elems_to_process =
          full_tile ? tile_len_ : vec_len_ - global_offset;

      copy::CopyGmToVec(in_q_, global_in_[global_offset], num_elems_to_process);
      ProcessTile(vec_out_lt, num_elems_to_process);
      global_offset += tile_len_;
    }
    out_q_.EnQue<int32_t>(vec_out_lt);
    AtomicAddWritePartialHist();
  }

 private:
  __aicore__ inline void ProcessTile(const LocalTensor<int32_t>& vec_out_lt,
                                     uint32_t size) {
    LocalTensor<T> vec_in_lt = in_q_.DeQue<T>();
    LocalTensor<int32_t> partial_out_lt = out_buf_.Get<int32_t>();

    AscendC::Duplicate<int32_t>(partial_out_lt, 0, num_bins_);
    AscendC::PipeBarrier<PIPE_ALL>();

    const float step = (hist_max_ - hist_min_) / num_bins_;

    int32_t current_size = static_cast<int32_t>(size);
    for (uint32_t i = 1; i < num_bins_; i++) {
      const T bin_edge = static_cast<T>(hist_max_ - i * step);
      const uint32_t less_than_elems = CountLessThan(vec_in_lt, bin_edge, size);
      AscendC::PipeBarrier<PIPE_ALL>();
      partial_out_lt.SetValue(num_bins_ - i, current_size - less_than_elems);
      AscendC::PipeBarrier<PIPE_ALL>();
      current_size = less_than_elems;
    }
    partial_out_lt.SetValue(0, current_size);

    AscendC::PipeBarrier<PIPE_ALL>();

    Add(vec_out_lt, vec_out_lt, partial_out_lt, num_bins_);

    in_q_.FreeTensor<T>(vec_in_lt);
  }

  /**
   * @brief Run the kernel.
   *
   * @param [in] vec_lt Input vector tile.
   * @param [in] pivot Pivot on which to count the cardinality of the set {x_i <
   * pivot} where \f$ x_i \f$ are the input elements.
   * @param [in] size Input tile size to consider; first `size` elements.
   */
  __aicore__ inline int32_t CountLessThan(LocalTensor<T>& vec_lt, T pivot,
                                          uint32_t size) {
    LocalTensor<T> work_lt = work_buf_.Get<T>();

    const LocalTensor<uint8_t> packed_mask_8b = packed_mask_buf_.Get<uint8_t>();
    AscendC::Duplicate(packed_mask_8b.template ReinterpretCast<int32_t>(), 0,
                       packed_mask_8b.GetSize() / 4);
    AscendC::Duplicate(work_lt, static_cast<T>(0), work_lt.GetSize());
    AscendC::PipeBarrier<PIPE_ALL>();
    CompareScalar(packed_mask_8b, vec_lt, pivot, CMPMODE::LT, tile_len_);
    AscendC::PipeBarrier<PIPE_ALL>();

    const LocalTensor<GatherMaskT> mask_lt =
        packed_mask_buf_.Get<GatherMaskT>();

    uint64_t num_gathered_elems = 0;
    GatherMask(work_lt, vec_lt, mask_lt, true, size, {1, 1, 8, 8},
               num_gathered_elems);

    return static_cast<int32_t>(num_gathered_elems);
  }

 private:
  __aicore__ inline void AtomicAddWritePartialHist() {
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SetAtomicAdd<int32_t>();
    copy::CopyVecToGm(global_out_, out_q_, num_bins_);
    AscendC::SetAtomicNone();
  }

  TPipe pipe_;

  TQue<QuePosition::VECIN, BUFFER_NUM> in_q_;
  TQue<QuePosition::VECOUT, 1> out_q_;

  TBuf<QuePosition::VECCALC> work_buf_;
  TBuf<QuePosition::VECCALC> out_buf_;
  TBuf<QuePosition::VECCALC> packed_mask_buf_;

  GlobalTensor<T> global_in_;
  GlobalTensor<int32_t> global_out_;

  const uint32_t vec_core_num_;
  const uint32_t vec_len_;
  const uint32_t tile_len_;
  const uint32_t num_bins_;
  const float hist_min_;
  const float hist_max_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;
};

/**
 * @brief Run the `histogram` kernel.
 *
 * @tparam T Input data type.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] vec_len Dimension of the input vector.
 * @param [in] tile_len Tile length.
 * @param [in] num_bins Number of histogram bins.
 * @param [in] hist_min Smallest value where the histogram will be built on.
 * @param [in] hist_max Largest value on which the histogram will be built on.
 */
template <typename T>
__aicore__ inline void run_histogram(GM_ADDR vec_in, GM_ADDR vec_out,
                                     uint32_t vec_len, uint32_t tile_len,
                                     uint32_t num_bins, float hist_min,
                                     float hist_max) {
  if ASCEND_IS_AIV {
    KernelHistogram<T> op(vec_len, tile_len, num_bins, hist_min, hist_max);
    op.Init(vec_in, vec_out);
    op.Process();
  }
}

}  // namespace tcuscan
