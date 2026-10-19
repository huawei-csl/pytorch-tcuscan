/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * @file kernel_greater_equal_ind.h
 * @brief Kernel implementing a fused `greater-equal` filter that also returns
 * the indices of the selected input elements.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "kernel_count_if.h"
#include "tcuscan_utils.h"

using namespace AscendC;
using namespace tcuscan;

namespace tcuscan {

/**
 * @brief Filter the input by `{x_i >= pivot}` and return both the selected
 * values and their positions in the input.
 *
 * The kernel is a fusion of `greater_equal` and `compress_ind_no_arange`:
 * the predicate is evaluated directly on the data tile, so no dense `int8`
 * mask is ever materialized in global memory, and the values and the indices
 * are compacted in the same tile iteration (they share the same output
 * offsets, since they are selected by the same bitmask).
 *
 * The kernel takes the per-core number of elements satisfying the predicate,
 * which is used to compute the output offset of each core. That vector must be
 * produced with the *same* `vec_len`, `tile_len` and launch `blockDim` as this
 * kernel (see \c run_count_if_per_block), otherwise the outputs of the cores
 * interleave incorrectly.
 *
 * The compaction is stable: the relative order of the selected elements is the
 * same as in the input.
 *
 * @tparam T Data type of the input vector. Must be at least 2 bytes wide.
 */
template <typename T>
class KernelGreaterEqualInd {
  using PosT = int32_t;

  using PackedMaskT = uint8_t;

  template <typename _InputT>
  using _GatherMaskT =
      typename std::conditional<sizeof(_InputT) == 2, uint16_t, uint32_t>::type;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Number of elements in an input vector.
   * @param [in] tile_len Tile length.
   */
  __aicore__ inline KernelGreaterEqualInd(uint32_t vec_len, uint32_t tile_len)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        vec_len_(vec_len),
        tile_len_(tile_len),
        mask_required_elems_(tile_len_ / IN_ELEMS_PER_MASK_ELEM),
        packed_mask_tile_size_(mask_required_elems_ < SMALLEST_MASK
                                   ? SMALLEST_MASK
                                   : mask_required_elems_),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_len_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, vec_core_num_)),
        max_num_elems_per_block_(tile_len_ * max_num_tiles_per_block_),
        num_elems_before_block_(GetBlockIdx() * max_num_elems_per_block_) {
    // GatherMask requires either 2- or 4-byte elements
    static_assert(sizeof(T) >= 2);
  }

 public:
  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_in Pointer to input vector in global memory.
   * @param [in] num_ges_per_block Pointer to the per-core number of elements
   * satisfying `{x_i >= pivot}`.
   * @param [in] vec_out Pointer to output data vector in global memory.
   * @param [in] indices_out Pointer to output indices vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR num_ges_per_block,
                              GM_ADDR vec_out, GM_ADDR indices_out) {
    global_input_.SetGlobalBuffer((__gm__ T*)vec_in, vec_len_);
    global_output_.SetGlobalBuffer((__gm__ T*)vec_out, vec_len_);
    global_indices_.SetGlobalBuffer((__gm__ PosT*)indices_out, vec_len_);
    global_pos_.SetGlobalBuffer((__gm__ PosT*)num_ges_per_block, vec_core_num_);

    // `pos_in_q_` only ever holds the per-core counts, one `PosT` per vector
    // core, not a whole tile.
    const uint32_t pos_bytes = scalar::AlignUp(
        vec_core_num_ * static_cast<uint32_t>(sizeof(PosT)), UB_ALIGNMENT);
    const uint32_t fixed_bytes =
        tile_len_ * static_cast<uint32_t>(sizeof(T)) + pos_bytes +
        tile_len_ * static_cast<uint32_t>(sizeof(PosT)) +
        packed_mask_tile_size_ * static_cast<uint32_t>(sizeof(PackedMaskT));
    const uint32_t gathered_bytes =
        tile_len_ * static_cast<uint32_t>(sizeof(T) + sizeof(PosT));
    // Double buffer the gathered outputs when they fit. A 4-byte input at the
    // largest tile does not, and overflowing the Unified Buffer faults the
    // vector core rather than failing to build.
    const uint8_t num_gathered_bufs =
        (fixed_bytes + 2 * gathered_bytes <= UB_SIZE_BYTES) ? 2 : 1;

    pipe_.InitBuffer(vec_in_q_, 1, tile_len_ * sizeof(T));
    pipe_.InitBuffer(pos_in_q_, 1, pos_bytes);
    pipe_.InitBuffer(arange_buf_, tile_len_ * sizeof(PosT));
    pipe_.InitBuffer(packed_mask_buf_,
                     packed_mask_tile_size_ * sizeof(PackedMaskT));
    pipe_.InitBuffer(gathered_vec_q_, num_gathered_bufs, tile_len_ * sizeof(T));
    pipe_.InitBuffer(gathered_ind_q_, num_gathered_bufs,
                     tile_len_ * sizeof(PosT));
  }

  /**
   * @brief Run the kernel.
   *
   * @param [in] pivot Value to compare the input with.
   */
  __aicore__ inline void Process(T pivot) {
    const uint32_t num_tiles_to_process = tcuscan::scalar::GetWorkDistribution(
        vec_len_, tile_len_, vec_core_num_);

    if (num_tiles_to_process == 0) {
      return;
    }

    uint32_t offset_within_block = 0;
    uint32_t output_offset = CalculatePartsOffsets();

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const uint32_t global_offset =
          num_elems_before_block_ + offset_within_block;

      const uint32_t num_elems_to_process =
          scalar::NextTileLen(tile_len_, global_offset, vec_len_);

      if (num_elems_to_process == 0) {
        return;
      }

      copy::CopyGmToVec(vec_in_q_, global_input_[global_offset],
                        num_elems_to_process);

      LocalTensor<T> input_lt = vec_in_q_.DeQue<T>();

      // Evaluate the predicate straight on the data tile: no int8 mask is
      // read from / written to global memory.
      BuildPackedMask(input_lt, pivot, num_elems_to_process);

      // Gather the selected values.
      const uint32_t num_gathered_elems =
          GatherAndStore(input_lt, gathered_vec_q_,
                         global_output_[output_offset], num_elems_to_process);

      vec_in_q_.FreeTensor(input_lt);

      // Gather the positions of the selected values. The bitmask is the same,
      // hence the output offset and the gathered count are the same too.
      const LocalTensor<PosT> arange_lt = arange_buf_.Get<PosT>();
      AscendC::ArithProgression<PosT>(
          arange_lt, static_cast<PosT>(global_offset), 1, num_elems_to_process);
      GatherAndStore(arange_lt, gathered_ind_q_, global_indices_[output_offset],
                     num_elems_to_process);

      output_offset += num_gathered_elems;
      offset_within_block += num_elems_to_process;
    }
  }

 private:
  __aicore__ inline uint32_t CalculatePartsOffsets() {
    uint32_t output_offset = 0;
    if (num_elems_before_block_ > 0) {
      copy::CopyGmToVec(pos_in_q_, global_pos_, vec_core_num_);
      LocalTensor<PosT> pos_lt = pos_in_q_.DeQue<PosT>();

      output_offset =
          reduce::ReduceScalarAdd(pos_lt, static_cast<uint32_t>(GetBlockIdx()));
      pos_in_q_.FreeTensor(pos_lt);
    }
    return output_offset;
  }

  /**
   * @brief Compute the packed (1 bit per element) mask of `{x_i >= pivot}`.
   *
   * `CompareScalar` is issued over the whole tile, so the lanes past the
   * copied-in data are zeroed first to keep the comparison well defined. Those
   * lanes are never gathered anyway, since `GatherMask` is given the exact
   * element count.
   */
  __aicore__ inline void BuildPackedMask(const LocalTensor<T>& input_lt,
                                         T pivot,
                                         uint32_t num_elems_to_process) {
    const uint32_t aligned_len = scalar::AlignUp(
        num_elems_to_process, static_cast<uint32_t>(UB_ALIGNMENT / sizeof(T)));
    if (aligned_len < tile_len_) {
      Duplicate(input_lt[aligned_len], static_cast<T>(0),
                tile_len_ - aligned_len);
    }

    const LocalTensor<PackedMaskT> packed_mask_8b =
        packed_mask_buf_.Get<PackedMaskT>();
    CompareScalar(packed_mask_8b, input_lt, pivot, CMPMODE::GE, tile_len_);
  }

  template <typename GatherT, int32_t _NumBuf>
  __aicore__ inline uint32_t GatherAndStore(
      const LocalTensor<GatherT>& input_lt,
      TQue<QuePosition::VECOUT, _NumBuf>& out_q,
      const GlobalTensor<GatherT>& global, uint32_t num_elems_to_process) {
    using GatherMaskT = _GatherMaskT<GatherT>;

    uint64_t num_gathered_elems = 0;
    {
      const LocalTensor<GatherMaskT> mask_lt =
          packed_mask_buf_.Get<GatherMaskT>();
      const LocalTensor<GatherT> output_lt =
          out_q.template AllocTensor<GatherT>();

      GatherMask(output_lt, input_lt, mask_lt, true, num_elems_to_process,
                 {1, 1, 8, 8}, num_gathered_elems);
      out_q.EnQue(output_lt);
    }

    {
      LocalTensor<GatherT> output_lt = out_q.template DeQue<GatherT>();

      if (num_gathered_elems == 0) {
        out_q.FreeTensor(output_lt);
        return static_cast<uint32_t>(0);
      }

      DataCopyExtParams params;
      params.blockCount = 1;
      params.blockLen = num_gathered_elems * sizeof(GatherT);
      params.srcStride = 0;
      params.dstStride = 0;
      DataCopyPad(global, output_lt, params);

      out_q.FreeTensor(output_lt);
      return static_cast<uint32_t>(num_gathered_elems);
    }
  }

  TPipe pipe_;

  GlobalTensor<T> global_input_;
  GlobalTensor<T> global_output_;
  GlobalTensor<PosT> global_indices_;
  GlobalTensor<PosT> global_pos_;

  TQue<QuePosition::VECIN, 1> vec_in_q_;
  TQue<QuePosition::VECIN, 1> pos_in_q_;
  TBuf<QuePosition::VECCALC> arange_buf_;
  TBuf<QuePosition::VECCALC> packed_mask_buf_;
  TQue<QuePosition::VECOUT, 2> gathered_vec_q_;
  TQue<QuePosition::VECOUT, 2> gathered_ind_q_;

  const uint32_t vec_core_num_;
  const uint32_t vec_len_;
  const uint32_t tile_len_;
  const uint32_t mask_required_elems_;
  const uint32_t packed_mask_tile_size_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;
  const uint32_t max_num_elems_per_block_;
  const uint32_t num_elems_before_block_;

  constexpr static uint16_t SMALLEST_MASK =
      tcuscan::UB_ALIGNMENT / sizeof(PackedMaskT);
  constexpr static uint16_t IN_ELEMS_PER_MASK_ELEM = sizeof(PackedMaskT) * 8;
};

/**
 * @brief Run the `greater_equal_ind` kernel. Supports `fp16/half` and
 * `float32` input dtypes. Returned indices have `int32_t` dtype.
 *
 * @tparam T Data type of the input vector.
 *
 * @param [in] vec_in Pointer to the input data vector.
 * @param [in] pivot Value to compare the input with.
 * @param [in] num_ges_per_block Pointer to the per-core number of elements
 * satisfying `{x_i >= pivot}`.
 * @param [in] vec_out Pointer to the output data vector.
 * @param [in] indices_out Pointer to the output indices vector.
 * @param [in] vec_len Length of the input vector.
 * @param [in] tile_len Tile length.
 */
template <typename T>
__aicore__ inline void run_greater_equal_ind(
    GM_ADDR vec_in, T pivot, GM_ADDR num_ges_per_block, GM_ADDR vec_out,
    GM_ADDR indices_out, uint32_t vec_len, uint32_t tile_len) {
  if ASCEND_IS_AIV {
    KernelGreaterEqualInd<T> op(vec_len, tile_len);
    op.Init(vec_in, num_ges_per_block, vec_out, indices_out);
    op.Process(pivot);
  }
}

}  // namespace tcuscan
