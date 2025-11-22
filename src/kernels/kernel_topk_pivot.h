/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file kernel_topk_pivot.h
 * @brief Kernel implementing a Vector k-largest value estimator.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"
#include "tiling/tiling_topk.h"

using namespace AscendC;
using namespace kernel_utils;

/**
 * @brief Returns an estimation \f$\theta\f$ of the k-th largest value of an
 * input array \f$x\f$. The estimation always lower-bounds the k-largest value,
 * i.e., \f$ topk\_value(x) \geq \theta\f$.
 *
 * @tparam T Input data type.
 */
template <typename T = half>
class KernelPivotTopkEstimator {
  constexpr static uint32_t BUFFER_NUM = 2;
  constexpr static uint32_t MIN_VEC_SIZE = UB_ALIGNMENT / sizeof(T);

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Dimension of the input vectors.
   * @param [in] tile_len Tile length.
   * @param [in] k Parameter of top-k.
   */
  __aicore__ inline KernelPivotTopkEstimator(uint32_t vec_len,
                                             uint32_t tile_len, uint32_t k)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        vec_len_(vec_len),
        tile_len_(static_cast<int32_t>(tile_len)),
        k_{static_cast<int32_t>(k)},
        inner_k_{scalar::CeilDiv(k_, vec_core_num_)},
        num_tiles_(scalar::CeilDiv(vec_len_, tile_len_)),
        num_samples_per_core_(1),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, vec_core_num_)) {
    ASCENDC_ASSERT(k_ % vec_core_num_ == 0, {
      KERNEL_LOG(KERNEL_ERROR,
                 "Parameter k of top-k must divide the number of AIVs. "
                 "Got (k=%d, AIVs=%d)",
                 k_, vec_core_num_);
    });
    ASCENDC_ASSERT(inner_k_ <= 128, {
      KERNEL_LOG(KERNEL_ERROR,
                 "K-largest (top-k) value estimator supports k at most "
                 "128. Got (k=%d, inner_k_=%d, AIVs=%d)",
                 k_, inner_k_, vec_core_num_);
    });

    ASCENDC_ASSERT(num_samples_per_core_ <= max_num_tiles_per_block_, {
      KERNEL_LOG(KERNEL_ERROR,
                 "Number of samples per core must be smaller than the maximum "
                 "number of tiles per block. Got (tile_len=%d, "
                 "num_samples_per_core_=%d, max_num_tiles_per_block_=%d)",
                 tile_len_, num_samples_per_core_, max_num_tiles_per_block_);
    });
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_in Pointer to the input vector in global memory.
   * @param [in] vec_out Pointer to the output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR vec_out) {
    global_in_.SetGlobalBuffer((__gm__ T*)vec_in, vec_len_);
    global_out_.SetGlobalBuffer((__gm__ T*)vec_out, MIN_VEC_SIZE);

    pipe.InitBuffer(in_q_, BUFFER_NUM, tile_len_ * sizeof(T));
    pipe.InitBuffer(out_q_, BUFFER_NUM, tile_len_ * sizeof(T));

    // Sort32 writes into a Tuple 2 of (value, index). Each output element
    // of Sort32 is 8-bytes.
    constexpr uint32_t SORT_ELEM_SIZE = 8;
    pipe.InitBuffer(concat_values_indices_buf_, tile_len_ * SORT_ELEM_SIZE);
    pipe.InitBuffer(unused_indices_buf_, tile_len_ * sizeof(uint32_t));
    pipe.InitBuffer(tmp_sorted_buf_, tile_len_ * SORT_ELEM_SIZE);
  }

  /**
   * @brief Run the kernel.
   *
   */
  __aicore__ inline void Process() {
    FillMinOutput<half>();

    uint32_t global_offset =
        GetBlockIdx() * tile_len_ * max_num_tiles_per_block_;
    const uint32_t max_num_tiles = kernel_utils::scalar::GetWorkDistribution(
        vec_len_, tile_len_, vec_core_num_);

    const uint32_t num_tiles_to_process =
        kernel_utils::scalar::Min(max_num_tiles, num_samples_per_core_);

    const LocalTensor<T> vec_out_lt = out_q_.AllocTensor<T>();
    T estimate = std::numeric_limits<T>::min();
    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const bool full_tile = global_offset + tile_len_ < vec_len_;
      const uint32_t num_elems_to_process_ =
          full_tile ? tile_len_ : vec_len_ - global_offset;

      copy::CopyGmToVec(in_q_, global_in_[global_offset],
                        num_elems_to_process_);
      ProcessTile(vec_out_lt, estimate);
    }

    AscendC::Duplicate(vec_out_lt, static_cast<T>(0), tile_len_);
    AscendC::PipeBarrier<PIPE_ALL>();
    vec_out_lt.SetValue(0, estimate);
    out_q_.EnQue<T>(vec_out_lt);

    // Write k-largest estimation value atomically in GM
    WriteToGmAtomicMin();
  }

 private:
  /**
   * @brief Fill output vector with fill_value using AIV core 0.
   */
  template <typename FillT>
  __aicore__ inline void FillMinOutput() {
    if (GetBlockIdx() == 0) {
      constexpr FillT max_val = std::numeric_limits<FillT>::min();
      const LocalTensor<FillT> vec_out_lt = out_q_.AllocTensor<FillT>();
      Duplicate(vec_out_lt, max_val, MIN_VEC_SIZE);
      out_q_.EnQue<FillT>(vec_out_lt);
      LocalTensor<T> lt = out_q_.template DeQue<T>();
      DataCopy(global_out_, lt, MIN_VEC_SIZE);
      out_q_.FreeTensor(lt);
    }

    // Sync all AIV cores
    SyncAll<true /*isAIVOnly*/>();
  }

  __aicore__ inline void ProcessTile(const LocalTensor<T>& vec_out_lt,
                                     T& estimate) {
    LocalTensor<T> vec_in_lt = in_q_.DeQue<T>();

    LocalTensor<T> concat_values_indices_lt =
        concat_values_indices_buf_.Get<T>();

    LocalTensor<uint32_t> unused_index_lt = unused_indices_buf_.Get<uint32_t>();

    LocalTensor<T> tmp_sorted_lt = tmp_sorted_buf_.Get<T>();

    // Sort32 bundles together the values (half/float) and indices
    // (uint32_t) into 8-bytes
    const int32_t repeats_of_32_elems = tile_len_ / 32;
    AscendC::Sort32<T>(concat_values_indices_lt, vec_in_lt, unused_index_lt,
                       repeats_of_32_elems);

    // Extracts the values and indices from (value, index) 8-bytes
    const uint32_t offset = AscendC::GetSortOffset<T>(32);
    AscendC::MrgSortSrcList sort_list = AscendC::MrgSortSrcList(
        concat_values_indices_lt[0], concat_values_indices_lt[offset],
        concat_values_indices_lt[2 * offset],
        concat_values_indices_lt[3 * offset]);
    const uint16_t elementCountList[4] = {32, 32, 32, 32};
    uint32_t sortedNum[4];
    AscendC::MrgSort<T, false>(tmp_sorted_lt, sort_list, elementCountList,
                               sortedNum, 0b1111, 1);
    AscendC::Extract(vec_out_lt, unused_index_lt, tmp_sorted_lt, 4);

    // T current = vec_out_lt(inner_k_ - 1);
    T current = vec_out_lt(k_ - 1);
    if (static_cast<float>(estimate) < static_cast<float>(current)) {
      estimate = current;
    }

    in_q_.FreeTensor<T>(vec_in_lt);
  }

  __aicore__ void WriteToGmAtomicMin() {
    AscendC::PipeBarrier<PIPE_MTE3>();
    AscendC::SetAtomicMax<T>();
    LocalTensor<T> lt = out_q_.template DeQue<T>();
    DataCopy(global_out_, lt, MIN_VEC_SIZE);
    out_q_.FreeTensor(lt);
    AscendC::SetAtomicNone();
  }

  TPipe pipe;

  TQue<QuePosition::VECIN, BUFFER_NUM> in_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> out_q_;

  TBuf<QuePosition::VECCALC> concat_values_indices_buf_;
  TBuf<QuePosition::VECCALC> unused_indices_buf_;
  TBuf<QuePosition::VECCALC> tmp_sorted_buf_;

  GlobalTensor<T> global_in_;
  GlobalTensor<T> global_out_;

  const uint32_t vec_core_num_;
  const uint32_t vec_len_;
  const int32_t tile_len_;
  const int32_t k_;
  const int32_t inner_k_;
  const uint32_t num_tiles_;
  const uint32_t num_samples_per_core_;
  const uint32_t max_num_tiles_per_block_;
};

/**
 * @brief Run the `pivot_topk_estimator` kernel.
 *
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 * @tparam T Data type used in kernel.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] vec_len Dimension of the input vector.
 * @param [in] tile_len Tile length.
 * @param [in] k Top-K parameter.
 */
template <bool ForceMixMode = true, typename T>
__aicore__ inline void run_pivot_topk_estimator(GM_ADDR vec_in, GM_ADDR vec_out,
                                                uint32_t vec_len,
                                                uint32_t tile_len, uint32_t k) {
  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }
  if ASCEND_IS_AIV {
    KernelPivotTopkEstimator<T> op(vec_len, tile_len, k);
    op.Init(vec_in, vec_out);
    op.Process();
  }
}
