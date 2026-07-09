/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_gather_spmv.h
 * @brief Kernel implementing a Vector gather_spmv kernel operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Performs a multi-core AIV gather operation given an 1D vector
 *   `values_in` and an 1D indices vector `indices_in` that returns a vector:
 *       z_out = values_in[x - 1 for x in indices_in] (in numpy notation)
 *
 *   Notice that indices_in = 0 then z_out = 0
 *   The output vector has length `len(indices_in)`
 *
 *   Example:
 *     values_in = [11 12 13 14 15 16 17 18 19]
 *     indices_in = [0  0  2  5  5  5  9  9]
 *     output = [0  0 12 15 15 15 19 19]
 *
 * @tparam DataType The data type of the tensor elements that must be gathered.
 * @tparam EnableDiff When `true`, the kernel fuses `torch::diff` into the same
 * launch: phase 1 gathers into a scratch buffer, a cross-core barrier makes the
 * gather visible, then phase 2 writes `out[i] = gathered[i + 1] - gathered[i]`
 * (length `idx_in_len - 1`) directly to the output. When `false`, the kernel
 * behaves exactly as a plain gather (unchanged public contract).
 *
 */
template <typename DataType, bool EnableDiff = false>
class KernelGatherSpmv {
  constexpr static uint32_t BUFFER_NUM = 1;
  /// @brief * represents the maximum tile length handled by the LocalTensor
  /// that contains the fetched values.
  constexpr static uint32_t VALUE_TILE_LEN = 32768;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] idx_in_len Length of the input index vector.
   * @param [in] values_in_len Length of the input values_in_len vector.
   * @param [in] tile_len Length of the tile processed in a single iteration.
   */

  __aicore__ inline KernelGatherSpmv(uint32_t idx_in_len,
                                     uint32_t values_in_len, uint32_t tile_len)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        values_in_len_(values_in_len),
        idx_in_len_(idx_in_len),
        tile_len_(tile_len),
        num_tiles_idx_(scalar::CeilDiv(idx_in_len, tile_len_)),
        max_num_tiles_per_block_idx_(
            scalar::CeilDiv(num_tiles_idx_, vec_core_num_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] values_in  Pointer to input values vector.
   * @param [in] idx_in Pointer to input column indices vector.
   * @param [in] z_out Pointer to output vector. Length `idx_in_len_` for a
   * plain gather, `idx_in_len_ - 1` when `EnableDiff`.
   * @param [in] gathered_scratch Pointer to a scratch buffer of length
   * `idx_in_len_` used to hold the phase-1 gather. Only read when `EnableDiff`.
   */
  __aicore__ inline void Init(GM_ADDR values_in, GM_ADDR idx_in, GM_ADDR z_out,
                              GM_ADDR gathered_scratch) {
    global_values_.SetGlobalBuffer((__gm__ DataType*)values_in, values_in_len_);
    global_idx_.SetGlobalBuffer((__gm__ uint32_t*)idx_in, idx_in_len_);
    if constexpr (EnableDiff) {
      // Phase 1 gathers into the scratch buffer; phase 2 writes the fused
      // `torch::diff` result (length `idx_in_len_ - 1`) into `z_out`.
      global_gathered_.SetGlobalBuffer((__gm__ DataType*)gathered_scratch,
                                       idx_in_len_);
      global_diff_out_.SetGlobalBuffer((__gm__ DataType*)z_out,
                                       idx_in_len_ - 1);
    } else {
      (void)gathered_scratch;
      global_gathered_.SetGlobalBuffer((__gm__ DataType*)z_out, idx_in_len_);
    }

    pipe.InitBuffer(values_q_gather_, BUFFER_NUM,
                    VALUE_TILE_LEN * sizeof(DataType));
    pipe.InitBuffer(idx_q_, BUFFER_NUM, tile_len_ * sizeof(int32_t));
    pipe.InitBuffer(output_q_, BUFFER_NUM, tile_len_ * sizeof(DataType));
    pipe.InitBuffer(threshold_up_buf_, tile_len_ * sizeof(uint32_t));
    pipe.InitBuffer(threshold_down_buf_, tile_len_ * sizeof(uint32_t));
    pipe.InitBuffer(mask_buf_, tile_len_ * sizeof(uint32_t));
    pipe.InitBuffer(mask_up_buf_, tile_len_ * sizeof(uint32_t));
    pipe.InitBuffer(mask_down_buf_, tile_len_ * sizeof(uint32_t));
    pipe.InitBuffer(gathered_mask_buff_, tile_len_ * sizeof(uint32_t));

    if constexpr (EnableDiff) {
      pipe.InitBuffer(diff_in_q_, BUFFER_NUM,
                      (tile_len_ + 1) * sizeof(DataType));
      pipe.InitBuffer(diff_out_q_, BUFFER_NUM, tile_len_ * sizeof(DataType));
      pipe.InitBuffer(diff_tbuf_, (tile_len_ + 1) * sizeof(uint32_t));
    }
  }

  /**
   * @brief Run the kernel.
   *
   *  This kernel tiles on the indexes and fetches from memory only the needed
   * values, putting them in a LocalTensor of maximum size "values_q_gather_"
   */
  __aicore__ inline void Process() {
    uint32_t gm_offset =
        GetBlockIdx() * tile_len_ * max_num_tiles_per_block_idx_;

    const uint32_t num_idx_tiles_to_process =
        tcuscan::scalar::GetWorkDistribution(idx_in_len_, tile_len_,
                                             vec_core_num_);
    for (uint32_t tile_idx = 0; tile_idx < num_idx_tiles_to_process;
         tile_idx++) {
      const bool is_full_tile = gm_offset + tile_len_ <= idx_in_len_;
      const uint32_t this_tile_len =
          is_full_tile ? tile_len_ : idx_in_len_ - gm_offset;
      copy::CopyGmToVec(idx_q_, global_idx_[gm_offset], this_tile_len);
      ProcessTile(gm_offset, this_tile_len);
      gm_offset += this_tile_len;
    }

    if constexpr (EnableDiff) {
      // Fuse `torch::diff` into this launch. The barrier guarantees every
      // core's phase-1 gather is committed to `global_gathered_` before any
      // core reads it back in phase 2 (same producer -> consumer pattern used
      // by the multi-core scan kernels).
      sync::SyncAllCores();
      RunDiff();
    }
  }

  /**
   * @brief Process tile performs the gather reducing the `idx_in` fetched
   * from GM by 1
   *
   * @param [in] output_gm global memory offset for writing the output
   * @param [in] this_tile_len the length of the current tile (always the
   * same as tile_len_, except possibly for the last tile)
   *
   * */
  __aicore__ inline void ProcessTile(uint32_t output_gm,
                                     uint32_t this_tile_len) {
    LocalTensor<uint32_t> idx_lt = idx_q_.DeQue<uint32_t>();

    if (idx_lt.GetValue(0) == 0) {
      HandleTileWithInitialZeros(idx_lt, output_gm, this_tile_len);
    } else {
      const uint32_t start = idx_lt.GetValue(0);
      const uint32_t end = idx_lt.GetValue(this_tile_len - 1);
      const uint32_t range = end - start + 1;

      if (range < VALUE_TILE_LEN) {
        HandleSingleTile(idx_lt, output_gm, start, end, this_tile_len);
      } else {
        HandleMultipleTiles(idx_lt, output_gm, start, end, this_tile_len);
      }
      idx_q_.FreeTensor<uint32_t>(idx_lt);
    }
  }

 private:
  /**
   * @brief As this gather should gather from Index - 1, index 0 must be
   * handled different. To each 0, it must correspond a 0 on the output. This
   * is usually done at the first tile, which is partially handled like this.
   * Then, the remaining tile is handled as usual
   *
   * @param idx_lt The input tensor whose elements are to be filtered.
   * @param output_gm global memory offset for writing the gathered values
   * @param this_tile_len length of current tile (always the
   * same as tile_len_, except possibly for the last tile)
   */
  __aicore__ inline void HandleTileWithInitialZeros(
      LocalTensor<uint32_t>& idx_lt, uint32_t output_gm,
      uint32_t this_tile_len) {
    LocalTensor<DataType> z_lt = output_q_.AllocTensor<DataType>();
    uint32_t i = 0;
    while (idx_lt.GetValue(i) == 0) {
      z_lt.SetValue(i, static_cast<DataType>(0));
      i = i + 1;
    }
    const uint32_t num_leading_zeros = i;

    output_q_.EnQue<DataType>(z_lt);
    copy::CopyVecToGm(global_gathered_[output_gm], output_q_,
                      num_leading_zeros);

    if (i < this_tile_len) {
      output_gm = output_gm + num_leading_zeros;
      const uint32_t start = idx_lt.GetValue(i);
      const uint32_t end = idx_lt.GetValue(this_tile_len - 1);
      const uint32_t range = end - start + 1;

      if (range < VALUE_TILE_LEN) {
        idx_q_.FreeTensor<uint32_t>(idx_lt);
        const uint32_t partial_tile_len = this_tile_len - num_leading_zeros;
        copy::CopyGmToVec(idx_q_, global_idx_[output_gm], partial_tile_len);
        LocalTensor<uint32_t> sub_idx_lt = idx_q_.DeQue<uint32_t>();
        HandleSingleTile(sub_idx_lt, output_gm, start, end, partial_tile_len);
        idx_q_.FreeTensor<uint32_t>(sub_idx_lt);
      } else {
        HandleMultipleTiles(idx_lt, output_gm, start, end, this_tile_len);
        idx_q_.FreeTensor<uint32_t>(idx_lt);
      }
    }
  }
  /**
   * @brief Handle a single tile case, avoiding any masking as the whole
   * gather will fit the LocalTensor of length VALUE_TILE_LEN
   *
   *
   * @param [in] idx_lt The input tensor whose elements are to be filtered.
   * @param [in] output_gm Output GM index to be written with gathered values
   * @param [in] start first value of the idx_array
   * @param [in] end Last value of the idx_array
   * @param [in] this_tile_len the length of the current tile (always the
   * same as tile_len_, except possibly for the last tile)
   */
  __aicore__ inline void HandleSingleTile(LocalTensor<uint32_t>& idx_lt,
                                          uint32_t output_gm, uint32_t start,
                                          uint32_t end,
                                          uint32_t this_tile_len) {
    const uint32_t range = end - start + 1;
    LocalTensor<DataType> z_lt = output_q_.AllocTensor<DataType>();
    copy::CopyGmToVec(values_q_gather_, global_values_[start - 1], range);
    LocalTensor<DataType> sync_fetched_values =
        values_q_gather_.DeQue<DataType>();
    GatherWithOffset(z_lt, idx_lt, sync_fetched_values, start, this_tile_len);
    values_q_gather_.FreeTensor<DataType>(sync_fetched_values);
    output_q_.EnQue<DataType>(z_lt);
    copy::CopyVecToGm(global_gathered_[output_gm], output_q_, this_tile_len);
  }

  /**
   * @brief Handle a multiple tiles case, requiring several subtiles of
   * length tile_len_ computed through masking, and a final subtile,
   * possibly smaller than tile_len_
   *
   *
   * @param idx_lt The input tensor whose elements are to be filtered.
   * @param [in] output_gm Output GM index to be written with gathered values
   * @param [in] start first value of the idx_lt array
   * @param [in] end Last value of the idx_lt array
   * @param [in] this_tile_len the length of the current tile (always the
   * same as tile_len_, except possibly for the last tile)
   */
  __aicore__ inline void HandleMultipleTiles(LocalTensor<uint32_t>& idx_lt,
                                             uint32_t output_gm, uint32_t start,
                                             uint32_t end,
                                             uint32_t this_tile_len) {
    uint32_t new_start = start;
    uint64_t gathered_size = 0;
    uint32_t gathered_count = 0;
    if (start != idx_lt.GetValue(0)) {
      gathered_count = output_gm;
    }

    while (end - new_start >= VALUE_TILE_LEN) {
      const uint32_t threshold_down = idx_lt.GetValue(gathered_count) - 1;
      const uint32_t threshold_up = threshold_down + VALUE_TILE_LEN;

      LocalTensor<uint32_t> gathered_idx_lt = FilterTileInterval(
          idx_lt, threshold_up, threshold_down, gathered_size, this_tile_len);

      const uint32_t subtile_size = static_cast<uint32_t>(gathered_size);
      new_start = gathered_idx_lt.GetValue(0);
      const uint32_t local_end = gathered_idx_lt.GetValue(subtile_size - 1);
      HandleSingleTile(gathered_idx_lt, output_gm, new_start, local_end,
                       subtile_size);

      output_gm = output_gm + subtile_size;
      gathered_count += gathered_size;
      new_start = idx_lt.GetValue(gathered_count);
      gathered_mask_buff_.FreeTensor<uint32_t>(gathered_idx_lt);
    }
    idx_q_.FreeTensor<uint32_t>(idx_lt);
    const uint32_t partial_tile_len = this_tile_len - gathered_count;
    copy::CopyGmToVec(idx_q_, global_idx_[output_gm], partial_tile_len);
    LocalTensor<uint32_t> sub_idx_lt = idx_q_.DeQue<uint32_t>();
    HandleSingleTile(sub_idx_lt, output_gm, new_start, end, partial_tile_len);
  }

  /**
   * @brief Filters an input tensor based on a upper and lower threshold
   * values. The output is a sub_tensor of the input tensor containing
   * only values in the specified interval. The number of returned
   * elements is stored in the output parameter
   * @p gathered_size.
   *
   * * Example 1:
   *
   * Threshold_up 17
   * Threshold_down 10
   *
   * Input: 1 3 5 7 9 10 11 12 14 17 19 20
   *
   * Output: 11 12 14
   *
   *
   * @param idx_lt The input tensor whose elements are to be filtered.
   * @param threshold_up The upper value; elements must be lower than
   * this value.
   * @param threshold_down The lower value; elements must be bigger than
   * this value.
   * @param gathered_size A reference to a variable where the count of
   * valid (gathered) elements will be stored.
   * @param this_tile_len the length of the current tile (always the
   * same as tile_len_, except possibly for the last tile)
   * @return LocalTensor<T> A tensor containing the indices of elements
   * that meet the filtering criteria.
   */
  __aicore__ inline LocalTensor<uint32_t> FilterTileInterval(
      LocalTensor<uint32_t> idx_lt, uint32_t threshold_up,
      uint32_t threshold_down, uint64_t& gathered_size,
      uint32_t this_tile_len) {
    LocalTensor<uint32_t> threshold_up_lt = threshold_up_buf_.Get<uint32_t>();
    LocalTensor<uint32_t> threshold_down_lt =
        threshold_down_buf_.Get<uint32_t>();

    Duplicate<uint32_t>(threshold_up_lt, threshold_up, this_tile_len);
    Duplicate<uint32_t>(threshold_down_lt, threshold_down, this_tile_len);

    LocalTensor<uint8_t> mask_up_lt = mask_up_buf_.Get<uint8_t>();
    LocalTensor<uint8_t> mask_down_lt = mask_down_buf_.Get<uint8_t>();
    LocalTensor<uint16_t> mask_lt = mask_buf_.Get<uint16_t>();

    if (this_tile_len < tile_len_) {
      Duplicate<uint16_t>(mask_up_lt.template ReinterpretCast<uint16_t>(), 0,
                          tile_len_);
      Duplicate<uint16_t>(mask_down_lt.template ReinterpretCast<uint16_t>(), 0,
                          tile_len_);
      Duplicate<uint16_t>(mask_lt, 0, tile_len_);
    }

    Compare(mask_up_lt, idx_lt.template ReinterpretCast<float>(),
            threshold_up_lt.template ReinterpretCast<float>(), CMPMODE::LT,
            tile_len_);  // Compares needs to handle the whole tile_len_. The
                         // AND takes care of partial tiles.

    Compare(mask_down_lt, idx_lt.template ReinterpretCast<float>(),
            threshold_down_lt.template ReinterpretCast<float>(), CMPMODE::GT,
            tile_len_);  // Compares needs to handle the whole tile_len_. The
                         // AND takes care of partial tiles.

    And(mask_lt, mask_up_lt.template ReinterpretCast<uint16_t>(),
        mask_down_lt.template ReinterpretCast<uint16_t>(), this_tile_len);
    LocalTensor<uint32_t> gathered_idx_lt = gathered_mask_buff_.Get<uint32_t>();
    GatherMask(gathered_idx_lt, idx_lt,
               mask_lt.template ReinterpretCast<uint32_t>(), true,
               this_tile_len, {1, 1, 8, 8}, gathered_size);

    return gathered_idx_lt;
  }

  /**
   * @brief Gathers elements from an input tensor using an index tensor
   * with an offset adjustment.
   *
   * @param output_tensor is the LocalTensor moved by reference, that will
   * contain the gather result
   * @param idx_lt The local tensor containing indexes that will be
   * adjusted and used for gathering.
   * @param input_tensor The local tensor from which elements will be
   * gathered.
   * @param offset The offset value to subtract from each index.
   * @param subtile_size The number of elements to process.
   */
  __aicore__ inline void GatherWithOffset(LocalTensor<DataType>& output_tensor,
                                          LocalTensor<uint32_t> idx_lt,
                                          LocalTensor<DataType> input_tensor,
                                          int32_t offset,
                                          uint64_t subtile_size) {
    LocalTensor<int32_t> idx_int32_lt =
        idx_lt.template ReinterpretCast<int32_t>();
    AscendC::Adds(idx_int32_lt, idx_int32_lt, -offset, subtile_size);
    AscendC::Muls(idx_int32_lt, idx_int32_lt,
                  static_cast<int32_t>(sizeof(DataType)), subtile_size);
    AscendC::Gather(output_tensor, input_tensor,
                    idx_int32_lt.template ReinterpretCast<uint32_t>(),
                    static_cast<uint32_t>(0), subtile_size);
  }

  /**
   * @brief Phase 2 (only instantiated when `EnableDiff`): computes
   * `torch::diff` of the phase-1 gather.
   *
   * Reads the gather output from `global_gathered_` (length `idx_in_len_`) and
   * writes `out[i] = gathered[i + 1] - gathered[i]` to `global_diff_out_`
   * (length `idx_in_len_ - 1`). Each core reads back only the contiguous chunk
   * it gathered itself in phase 1, so every scratch read is self-coherent. The
   * one right-overlap element per tile, `gathered[global_offset + num_elems]`,
   * may belong to a neighbouring core; reading it back from scratch is a
   * cross-core read-after-write race, so it is instead recomputed from the
   * read-only inputs in `DiffTile` (see below).
   *
   * Mirrors `KernelDiff`, but writes the segment-sum result directly (no
   * `prepend` shift / first-element fixup), fusing the standalone `diff`
   * launch.
   */
  __aicore__ inline void RunDiff() {
    uint32_t global_offset =
        GetBlockIdx() * tile_len_ * max_num_tiles_per_block_idx_;
    const uint32_t num_tiles_to_process = tcuscan::scalar::GetWorkDistribution(
        idx_in_len_, tile_len_, vec_core_num_);
    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      // A tile produces `num_elems` diffs. The last tile stops one short of
      // `idx_in_len_`.
      const bool full_tile = global_offset + tile_len_ < idx_in_len_;
      const uint32_t num_elems =
          full_tile ? tile_len_ : idx_in_len_ - global_offset - 1;
      copy::CopyGmToVec(diff_in_q_, global_gathered_[global_offset], num_elems);
      DiffTile(num_elems, global_offset);
      copy::CopyVecToGm(global_diff_out_[global_offset], diff_out_q_,
                        num_elems);
      global_offset += tile_len_;
    }
  }

  /**
   * @brief Computes one diff tile: `out[k] = in[k + 1] - in[k]` for `k in [0,
   * num_elems)`, given `num_elems` gathered values queued in `diff_in_q_`. The
   * result is enqueued in `diff_out_q_`.
   *
   * @param [in] num_elems Number of diff outputs this tile produces.
   * @param [in] global_offset Global position of the first element of the tile.
   */
  __aicore__ inline void DiffTile(uint32_t num_elems, uint32_t global_offset) {
    LocalTensor<DataType> vec_in_lt = diff_in_q_.DeQue<DataType>();
    const LocalTensor<DataType> vec_out_lt =
        diff_out_q_.AllocTensor<DataType>();

    LocalTensor<uint32_t> cols_uint32_lt = diff_tbuf_.Get<uint32_t>();
    LocalTensor<int32_t> cols_int32_lt =
        cols_uint32_lt.template ReinterpretCast<int32_t>();

    // Byte offsets sizeof, 2*sizeof, ... select the shifted-by-one input, so
    // vec_out_lt[k] = vec_in_lt[k + 1] for k in [0, num_elems - 1).
    ArithProgression<int32_t>(
        cols_int32_lt, static_cast<int32_t>(sizeof(DataType)),
        static_cast<int32_t>(sizeof(DataType)), num_elems - 1);
    AscendC::Gather(vec_out_lt, vec_in_lt, cols_uint32_lt,
                    static_cast<uint32_t>(0), num_elems - 1);
    Sub(vec_out_lt, vec_out_lt, vec_in_lt, num_elems);

    // The final diff needs gathered[global_offset + num_elems], which may live
    // in another core's scratch region. Recompute it from the read-only inputs
    // (always coherent) using the gather definition itself:
    //   gathered[p] = idx[p] == 0 ? 0 : values[idx[p] - 1].
    const uint32_t p = global_offset + num_elems;
    const uint32_t idx_p = global_idx_.GetValue(p);
    const DataType overlap = (idx_p == 0) ? static_cast<DataType>(0)
                                          : global_values_.GetValue(idx_p - 1);
    // Native-type subtraction: int32 accumulators (int16 SpMV) do not fit in
    // float32, so the diff must stay in DataType, matching torch::diff.
    const DataType last_in = vec_in_lt(num_elems - 1);
    vec_out_lt.SetValue(num_elems - 1,
                        static_cast<DataType>(overlap - last_in));

    diff_out_q_.EnQue<DataType>(vec_out_lt);
    diff_in_q_.FreeTensor<DataType>(vec_in_lt);
  }

  TPipe pipe;

  TQue<QuePosition::VECIN, BUFFER_NUM> values_q_gather_;
  TQue<QuePosition::VECIN, BUFFER_NUM> idx_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> output_q_;

  // Phase-2 diff queues (only initialized when `EnableDiff`).
  TQue<QuePosition::VECIN, BUFFER_NUM> diff_in_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> diff_out_q_;

  TBuf<QuePosition::VECCALC> threshold_up_buf_;
  TBuf<QuePosition::VECCALC> threshold_down_buf_;
  TBuf<QuePosition::VECCALC> mask_buf_;
  TBuf<QuePosition::VECCALC> mask_up_buf_;
  TBuf<QuePosition::VECCALC> mask_down_buf_;
  TBuf<QuePosition::VECCALC> gathered_mask_buff_;
  TBuf<QuePosition::VECCALC> diff_tbuf_;

  GlobalTensor<DataType> global_values_;
  GlobalTensor<uint32_t> global_idx_;
  // Phase-1 gather destination. Aliases the kernel output for a plain gather,
  // or a scratch buffer when `EnableDiff` (phase 2 reads it back).
  GlobalTensor<DataType> global_gathered_;
  // Fused-diff output (length `idx_in_len_ - 1`); only used when `EnableDiff`.
  GlobalTensor<DataType> global_diff_out_;

  const uint32_t vec_core_num_;
  const uint32_t values_in_len_;
  const uint32_t idx_in_len_;
  const uint32_t tile_len_;
  const uint32_t num_tiles_idx_;
  const uint32_t max_num_tiles_per_block_idx_;
};

/**
 * @brief Run the `gather_spmv` kernel.
 *
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 * @tparam EnableDiff When `true`, fuses `torch::diff` into the kernel: the
 * output is `out[i] = gather[i + 1] - gather[i]` with length `idx_in_len - 1`,
 * and `gathered_scratch` must point to a buffer of length `idx_in_len`.
 *
 * @param [in] values_in Pointer to input vector.
 * @param [in] idx_in Pointer to column indices input vector.
 * @param [in] z_out Pointer to output vector.
 * @param [in] gathered_scratch Scratch buffer of length `idx_in_len` used to
 * stage the gather when `EnableDiff`; ignored otherwise.
 * @param [in] idx_in_len length of the indices array
 * @param [in] val_in_len length of the val_len array
 * @param [in] tile_len Length of the tile processed in a single iteration.
 */

template <bool ForceMixMode = true, bool EnableDiff = false,
          typename DataType = float>
__aicore__ inline void run_gather_spmv(GM_ADDR values_in, GM_ADDR idx_in,
                                       GM_ADDR z_out, GM_ADDR gathered_scratch,
                                       uint32_t idx_in_len, uint32_t val_in_len,
                                       uint32_t tile_len) {
  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }
  if ASCEND_IS_AIV {
    KernelGatherSpmv<DataType, EnableDiff> op(idx_in_len, val_in_len, tile_len);
    op.Init(values_in, idx_in, z_out, gathered_scratch);
    op.Process();
  }
}

}  // namespace tcuscan
