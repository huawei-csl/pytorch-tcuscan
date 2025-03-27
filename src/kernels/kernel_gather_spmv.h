/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_gather_spmv.h
 * @brief Kernel implementing a Vector gather_spmv kernel operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "kernel_utils.h"

using namespace AscendC;
using namespace kernel_utils;

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
 *
 */
template <typename DataType>
class KernelGatherSpmv {
  constexpr static uint32_t BUFFER_NUM = 1;
  constexpr static int32_t SHIFT_BYTES = sizeof(DataType);
  constexpr static int32_t START_OFFSET_BYTES = sizeof(DataType);

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] idx_in_len Length of the input index vector.
   * @param [in] tile_len Length of the tile processed in a single iteration.
   */

  __aicore__ inline KernelGatherSpmv(uint32_t idx_in_len, uint32_t tile_len)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        idx_in_len_(idx_in_len),
        tile_len_(tile_len),
        num_tiles_idx_(idx_in_len / tile_len_),
        max_num_tiles_per_block_idx_(
            scalar::CeilDiv(num_tiles_idx_, vec_core_num_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] values_in  Pointer to input values vector.
   * @param [in] idx_in Pointer to input column indices vector.
   * @param [in] z_out Pointer to output z vector.
   *
   * value_tile_size_ = 32768 represents the maximum size handled by the
   * LocalBuffer that contains fetched values.
   *
   */
  __aicore__ inline void Init(GM_ADDR values_in, GM_ADDR idx_in,
                              GM_ADDR z_out) {
    global_values_.SetGlobalBuffer((__gm__ DataType *)values_in);
    global_idx_.SetGlobalBuffer((__gm__ uint32_t *)idx_in);
    global_z_.SetGlobalBuffer((__gm__ DataType *)z_out);

    pipe.InitBuffer(values_q_gather_, BUFFER_NUM,
                    value_tile_size_ * sizeof(DataType));
    pipe.InitBuffer(idx_q_, BUFFER_NUM, tile_len_ * sizeof(int32_t));
    pipe.InitBuffer(output_q_, BUFFER_NUM, tile_len_ * sizeof(DataType));
    pipe.InitBuffer(idx_buf_, tile_len_ * sizeof(uint32_t));
    pipe.InitBuffer(tensor_up_buf_, tile_len_ * sizeof(uint32_t));
    pipe.InitBuffer(tensor_down_buf_, tile_len_ * sizeof(uint32_t));
    pipe.InitBuffer(mask_buf_, tile_len_ * sizeof(uint32_t));
    pipe.InitBuffer(mask_up_buf_, tile_len_ * sizeof(uint32_t));
    pipe.InitBuffer(mask_down_buf_, tile_len_ * sizeof(uint32_t));
    pipe.InitBuffer(gathered_mask_buff_, tile_len_ * sizeof(uint32_t));
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
        kernel_utils::scalar::GetWorkDistribution(idx_in_len_, tile_len_,
                                                  vec_core_num_);
    for (uint32_t tile_idx = 0; tile_idx < num_idx_tiles_to_process;
         tile_idx++) {
      const bool is_full_tile = gm_offset + tile_len_ <= idx_in_len_;
      const uint32_t num_elems_idx =
          is_full_tile ? tile_len_ : idx_in_len_ - gm_offset;
      copy::CopyGmToVec(idx_q_, global_idx_[gm_offset], num_elems_idx);
      ProcessTile(gm_offset);
      gm_offset += tile_len_;
    }
  }

  /**
   * @brief Process tile performs the gather reducing the col_idx fetched
   * from GM by 1
   * @param [in] output_gm  global memory offset for writing the output
   *
   * */
  __aicore__ inline void ProcessTile(uint32_t output_gm) {
    LocalTensor<uint32_t> idx_lt = idx_q_.DeQue<uint32_t>();

    if (idx_lt.GetValue(0) == 0) {
      HandleTileWithInitialZeros(idx_lt, output_gm);
    } else {
      const uint32_t start = idx_lt.GetValue(0);
      const uint32_t end = idx_lt.GetValue(tile_len_ - 1);
      const uint32_t es_diff = (end - start + 1);

      if (es_diff < value_tile_size_) {
        HandleSingleTile(idx_lt, output_gm, start, es_diff, tile_len_);
      } else {
        HandleMultipleTiles(idx_lt, output_gm, start, end, tile_len_, false);
      }
    }
  }

  /**
   * @brief As this gather should gather from Index - 1, index 0 must be
   * handled different. To each 0, it must correspond a 0 on the output. This
   * is usually done at the first tile, which is partially handled like this.
   * Then, the remaining tile is handled as usual
   *
   * @param idx_lt The input tensor whose elements are to be filtered.
   * @param output_gm Output GM index to be written with gathered values
   */
 private:
  __aicore__ inline void HandleTileWithInitialZeros(
      LocalTensor<uint32_t> &idx_lt, uint32_t output_gm) {
    LocalTensor<DataType> z_lt = output_q_.AllocTensor<DataType>();
    uint32_t i = 0;
    while (idx_lt.GetValue(i) == 0) {
      z_lt.SetValue(i, static_cast<float>(0));
      i = i + 1;
    }

    output_q_.EnQue<DataType>(z_lt);
    copy::CopyVecToGm(global_z_[output_gm], output_q_, i);

    if (i < idx_lt.GetSize()) {
      output_gm = output_gm + i;
      uint32_t start = idx_lt.GetValue(i);
      uint32_t end = idx_lt.GetValue(tile_len_ - 1);
      uint32_t es_diff = (end - start + 1);

      if (es_diff < value_tile_size_) {
        idx_q_.FreeTensor<uint32_t>(idx_lt);
        const uint32_t partial_tile = tile_len_ - i;
        copy::CopyGmToVec(idx_q_, global_idx_[output_gm], partial_tile);
        LocalTensor<uint32_t> sub_idx_lt = idx_q_.DeQue<uint32_t>();
        HandleSingleTile(sub_idx_lt, output_gm, start, es_diff, partial_tile);
      } else {
        HandleMultipleTiles(idx_lt, output_gm, start, end, tile_len_, true);
      }
    }
  }
  /**
   * @brief Handle a single tile case, avoiding any masking as the whole
   * gather will fit the LocalTensor value_tile_size_
   *
   *
   * @tparam T The data type of the tensor elements.
   * @param idx_lt The input tensor whose elements are to be filtered.
   * @param output_gm Output GM index to be written with gathered values
   * @param offset index offset to be gathered
   * @param es_diff Number of values to be gathered
   * @param tile_len Number of elements currently evaluated in the tile
   */

  __aicore__ inline void HandleSingleTile(LocalTensor<uint32_t> &idx_lt,
                                          uint32_t output_gm, uint32_t offset,
                                          uint32_t es_diff, uint32_t tile_len) {
    LocalTensor<DataType> z_lt = output_q_.AllocTensor<DataType>();

    copy::CopyGmToVec(values_q_gather_, global_values_[offset - 1], es_diff);
    LocalTensor<DataType> sync_fetched_values =
        values_q_gather_.DeQue<DataType>();
    GatherWithOffset<uint32_t, DataType>(z_lt, idx_lt, sync_fetched_values,
                                         offset, tile_len);
    values_q_gather_.FreeTensor<DataType>(sync_fetched_values);
    output_q_.EnQue<DataType>(z_lt);
    idx_q_.FreeTensor<uint32_t>(idx_lt);
    copy::CopyVecToGm(global_z_[output_gm], output_q_, tile_len);
  }

  /**
   * @brief Handle a multiple tiles case, requiring several subtiles of
   * length tile_len_ computed through masking, and a final subtile,
   * possibly smaller than tile_len_
   *
   *
   * @tparam T The data type of the tensor elements.
   * @param idx_lt The input tensor whose elements are to be filtered.
   * @param output_gm Output GM index to be written with gathered values
   * @param start first value of the idx_array
   * @param end Last value of the idx_array
   * @param tile_len remaining size to be processed
   * @param partial_handle this is true if the current index tile has been
   * partially handled before entering this function. For example, to handle 0
   * indexes
   *
   */
  __aicore__ inline void HandleMultipleTiles(LocalTensor<uint32_t> &idx_lt,
                                             uint32_t output_gm, uint32_t start,
                                             uint32_t end, uint32_t tile_len,
                                             bool partial_handle) {
    uint32_t new_start = start;

    uint32_t subtile_size = 0;
    uint32_t threshold_up;
    uint32_t threshold_down;

    uint64_t gathered_size = 0;
    uint32_t gathered_count = 0;

    if (partial_handle) {
      gathered_count = output_gm;
    }
    LocalTensor<DataType> z_lt = output_q_.AllocTensor<DataType>();
    LocalTensor<uint32_t> gathered_idx_lt;
    while (end - new_start >= value_tile_size_) {
      threshold_down = idx_lt.GetValue(gathered_count) - 1;
      threshold_up = threshold_down + value_tile_size_;

      // Filter Indexes -> find idx subtile
      LocalTensor<uint32_t> gathered_idx_lt;

      gathered_idx_lt = FilterTileInterval<uint32_t>(
          idx_lt, threshold_up, threshold_down, gathered_size, tile_len);

      new_start = gathered_idx_lt.GetValue(0);
      subtile_size = static_cast<uint32_t>(gathered_size);
      copy::CopyGmToVec(values_q_gather_, global_values_[new_start - 1],
                        value_tile_size_);
      LocalTensor<DataType> sync_fetched_values =
          values_q_gather_.DeQue<DataType>();

      // Gather values using idx subtile
      GatherWithOffset<uint32_t, DataType>(
          z_lt, gathered_idx_lt, sync_fetched_values, new_start, subtile_size);

      output_q_.EnQue<DataType>(z_lt);
      copy::CopyVecToGm(global_z_[output_gm], output_q_, subtile_size);

      // Update Addresses and prepare next operation
      values_q_gather_.FreeTensor<DataType>(sync_fetched_values);
      output_gm = output_gm + subtile_size;
      z_lt = output_q_.AllocTensor<DataType>();
      gathered_count += gathered_size;
      new_start = idx_lt.GetValue(gathered_count);
    }
    // Compute last idx_subtile
    uint32_t remaining_tile = end - new_start + 1;
    threshold_down = idx_lt.GetValue(gathered_count) - 1;
    gathered_idx_lt = FilterTile<uint32_t>(
        idx_lt, threshold_down, gathered_size, CMPMODE::GT, tile_len);
    new_start = gathered_idx_lt.GetValue(0);
    subtile_size = static_cast<uint32_t>(gathered_size);
    copy::CopyGmToVec(values_q_gather_, global_values_[new_start - 1],
                      remaining_tile);
    LocalTensor<DataType> sync_fetched_values =
        values_q_gather_.DeQue<DataType>();

    // Gather values using the last subtiles
    GatherWithOffset<uint32_t, DataType>(
        z_lt, gathered_idx_lt, sync_fetched_values, new_start, subtile_size);

    // Write the output and free mem
    output_q_.EnQue<DataType>(z_lt);
    copy::CopyVecToGm(global_z_[output_gm], output_q_, subtile_size);
    values_q_gather_.FreeTensor<DataType>(sync_fetched_values);
    idx_q_.FreeTensor<uint32_t>(idx_lt);
  }

  /**
   * @brief Filters an input tensor based on a threshold values.
   * The output is a sub-tensor of the input tensor containing only values
   * bigger (if CMPMODE::GT) or smaller (if CMPMODE::LT) than the given
   * threshold. The number of returned elements is stored in the output
   * parameter
   * @p gathered_size.
   *
   * Example 1:
   *
   * Threshold 10
   * Operation AscendC::CMPMODE LT
   * Input: 1 3 5 7 9 10 11 12 14 17 19 20
   *
   * Output: 1 3 5 7 9
   *
   * Example 2:
   *
   * Threshold 10
   * Operation AscendC::CMPMODE GT
   * Input: 1 3 5 7 9 10 11 12 14 17 19 20
   *
   * Output: 10 11 12 14 17 19 20
   *
   *
   * @tparam T The data type of the tensor elements.
   * @param idx_lt The input tensor whose elements are to be filtered.
   * @param threshold The upper value; elements must be compared with
   * this value.
   * @param gathered_size A reference to a variable where the count of
   * valid (gathered) elements will be stored.
   * @param mode Ascend::CMPMODE for comparing threshold and tensor
   * @param tile_len Number of elements currently evaluated in the tile
   * @return LocalTensor<T> A tensor containing the indices of elements
   * that meet the filtering criteria.
   *
   */

  template <typename T>
  __aicore__ inline LocalTensor<T> FilterTile(LocalTensor<T> idx_lt,
                                              uint32_t threshold,
                                              uint64_t &gathered_size,
                                              AscendC::CMPMODE mode,
                                              uint32_t tile_len) {
    LocalTensor<T> tensor = tensor_down_buf_.Get<uint32_t>();
    Duplicate<uint32_t>(tensor, threshold, tile_len);
    LocalTensor<uint8_t> mask_tensor = mask_down_buf_.Get<uint8_t>();
    Compare(mask_tensor, idx_lt.template ReinterpretCast<float>(),
            tensor.template ReinterpretCast<float>(), mode, tile_len);
    LocalTensor<T> gathered_idx_lt = gathered_mask_buff_.Get<T>();
    GatherMask(gathered_idx_lt, idx_lt,
               mask_tensor.template ReinterpretCast<uint32_t>(), true, tile_len,
               {1, 1, 8, 8}, gathered_size);
    return gathered_idx_lt;
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
   * @tparam T The data type of the tensor elements.
   * @param idx_lt The input tensor whose elements are to be filtered.
   * @param threshold_up The upper value; elements must be lower than
   * this value.
   * @param threshold_down The lower value; elements must be bigger than
   * this value.
   * @param gathered_size A reference to a variable where the count of
   * valid (gathered) elements will be stored.
   * @param tile_len Number of elements currently evaluated in the tile
   * @return LocalTensor<T> A tensor containing the indices of elements
   * that meet the filtering criteria.
   *
   */
  template <typename T>
  __aicore__ inline LocalTensor<T> FilterTileInterval(LocalTensor<T> idx_lt,
                                                      T threshold_up,
                                                      T threshold_down,
                                                      uint64_t &gathered_size,
                                                      uint32_t tile_len) {
    LocalTensor<T> tensor_up = tensor_up_buf_.Get<T>();
    LocalTensor<T> tensor_down = tensor_down_buf_.Get<T>();

    Duplicate<uint32_t>(tensor_up, threshold_up, tile_len);
    Duplicate<uint32_t>(tensor_down, threshold_down, tile_len);

    LocalTensor<uint8_t> mask_tensor_up = mask_up_buf_.Get<uint8_t>();
    LocalTensor<uint8_t> mask_tensor_down = mask_down_buf_.Get<uint8_t>();
    LocalTensor<uint32_t> mask_lt = mask_buf_.Get<uint32_t>();

    Compare(mask_tensor_up, idx_lt.template ReinterpretCast<float>(),
            tensor_up.template ReinterpretCast<float>(), CMPMODE::LT, tile_len);

    Compare(mask_tensor_down, idx_lt.template ReinterpretCast<float>(),
            tensor_down.template ReinterpretCast<float>(), CMPMODE::GT,
            tile_len);

    And(mask_lt, mask_tensor_up.template ReinterpretCast<uint32_t>(),
        mask_tensor_down.template ReinterpretCast<uint32_t>(), tile_len);
    LocalTensor<T> gathered_idx_lt = gathered_mask_buff_.Get<T>();
    GatherMask(gathered_idx_lt, idx_lt, mask_lt, true, tile_len, {1, 1, 8, 8},
               gathered_size);
    return gathered_idx_lt;
  }

  /**
   * @brief Gathers elements from an input tensor using an index tensor
   * with an offset adjustment.
   *
   * @tparam T The type of the elements in the index tensor.
   * @tparam U The type of the elements in the input and output
   * tensors.
   * @param output_tensor is the LocalTensor moved by reference, that will
   * contain the gather result
   * @param idx_lt The local tensor containing indexes that will be
   * adjusted and used for gathering.
   * @param input_tensor The local tensor from which elements will be
   * gathered.
   * @param offset The offset value to subtract from each index.
   * @param subtile_size The number of elements to process.
   *
   */
  template <typename T, typename U>
  __aicore__ inline void GatherWithOffset(LocalTensor<U> &output_tensor,
                                          LocalTensor<T> idx_lt,
                                          LocalTensor<U> input_tensor,
                                          int32_t offset,
                                          uint64_t subtile_size) {
    AscendC::Adds(idx_lt.template ReinterpretCast<int32_t>(),
                  idx_lt.template ReinterpretCast<int32_t>(),
                  -static_cast<int32_t>(offset), subtile_size);
    LocalTensor<int32_t> indexes = idx_buf_.Get<int32_t>();
    AscendC::Muls(indexes, idx_lt.template ReinterpretCast<int32_t>(),
                  static_cast<int32_t>(sizeof(U)), subtile_size);
    AscendC::Gather(output_tensor, input_tensor,
                    indexes.template ReinterpretCast<uint32_t>(),
                    static_cast<uint32_t>(0), subtile_size);
  }

  TPipe pipe;

  TQue<QuePosition::VECIN, BUFFER_NUM> values_q_gather_;
  TQue<QuePosition::VECIN, BUFFER_NUM> idx_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> output_q_;
  TBuf<QuePosition::VECCALC> idx_buf_;
  TBuf<QuePosition::VECCALC> tensor_up_buf_;
  TBuf<QuePosition::VECCALC> tensor_down_buf_;
  TBuf<QuePosition::VECCALC> mask_buf_;
  TBuf<QuePosition::VECCALC> mask_up_buf_;
  TBuf<QuePosition::VECCALC> mask_down_buf_;
  TBuf<QuePosition::VECCALC> gathered_mask_buff_;

  GlobalTensor<DataType> global_values_;
  GlobalTensor<uint32_t> global_idx_;
  GlobalTensor<DataType> global_z_;

  const uint32_t vec_core_num_;
  const uint32_t idx_in_len_;
  const uint32_t tile_len_;
  const uint32_t num_tiles_idx_;
  const uint32_t max_num_tiles_per_block_idx_;
  const uint32_t value_tile_size_ = 32768;
};

/**
 * @brief Run the `gather_spmv` kernel.
 *
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 *
 * @param [in] values_in Pointer to input vector.
 * @param [in] idx_in Pointer to column indices input vector.
 * @param [in] z_out Pointer to output vector.
 * @param [in] idx_in_len length of the indexes array
 * @param [in] tile_len Length of the tile processed in a single iteration.
 */

template <bool ForceMixMode = true>
__aicore__ inline void run_gather_spmv(GM_ADDR values_in, GM_ADDR idx_in,
                                       GM_ADDR z_out, uint32_t idx_in_len,
                                       uint32_t tile_len) {
  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }
  if ASCEND_IS_AIV {
    KernelGatherSpmv<float> op(idx_in_len, tile_len);
    op.Init(values_in, idx_in, z_out);
    op.Process();
  }
}