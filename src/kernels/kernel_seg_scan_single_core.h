/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * @file kernel_seg_scan_single_core.h
 * @brief Kernel implementing a single-core segmented scan using cube-vector
 * approach.
 */

#pragma once

#include "ascendc_kernel_operator.h"
#include "kernel_pad.h"
#include "kernel_scan2p_single_core.h"
#include "kernel_utils.h"

using namespace AscendC;
using namespace kernel_utils;

/**
 * @brief Reverts speculative scan operation for each segment.
 *
 * The algorithm utilizes only a single vector core.
 *
 * The algorithm takes an inclusive row-wise speculative scan (row size being
 * equal to `tile_width`) of an input vector and a flag vector (for example the
 * one produced by `KernelRowScan`). Then the algorithm reverts the unnecessary
 * speculative scan parts by iterating over chunks of size `tile_width` and
 * adding to them the sum of all the previous chunks.
 *
 * @tparam DataTypeT Data type of input and output vectors
 * @tparam FlagT Data type of flag vector
 */
template <typename DataTypeT, typename FlagOutputT>
class KernelSegScanRevertSpec {
  constexpr static uint32_t BUFFER_NUM = 1;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] matmul_size Size of the dimension of A matrix.
   * @param [in] vec_len Input vector length.
   */
  __aicore__ inline KernelSegScanRevertSpec(uint16_t matmul_size,
                                            uint32_t vec_len)
      : vec_len_(vec_len),
        tile_len_(matmul_size),
        num_matrix_tiles_(scalar::CeilDiv(vec_len_, matmul_size * matmul_size)),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_len_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_in Pointer to row-wise block scan of input in GM.
   * @param [in] flag_in Pointer to row-wise block scan of flags in GM.
   * @param [in] vec_out Pointer to output vector in GM.
   */
  __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR flag_in,
                              GM_ADDR vec_out) {
    global_input_.SetGlobalBuffer((__gm__ DataTypeT *)vec_in, vec_len_);
    global_scanned_flag_.SetGlobalBuffer((__gm__ FlagOutputT *)flag_in,
                                         vec_len_);
    global_output_.SetGlobalBuffer((__gm__ DataTypeT *)vec_out, vec_len_);

    pipe.InitBuffer(vecin_input_q_, BUFFER_NUM, tile_len_ * sizeof(DataTypeT));
    pipe.InitBuffer(vecin_scanned_flag_q_, BUFFER_NUM,
                    tile_len_ * sizeof(FlagOutputT));
    pipe.InitBuffer(vecout_q_, BUFFER_NUM, tile_len_ * sizeof(DataTypeT));

    pipe.InitBuffer(tmp_uint8_buf_,
                    tile_len_ * sizeof(uint8_t) /
                        AscendCUtils::GetBitSize(sizeof(uint8_t)));
    pipe.InitBuffer(tmp2_uint8_buf_, tile_len_ * sizeof(uint8_t));
    pipe.InitBuffer(tmp3_uint8_buf_, tile_len_ * sizeof(uint8_t));
    pipe.InitBuffer(tmp1_int16_buf_, tile_len_ * sizeof(int16_t));
    pipe.InitBuffer(dst_int16_buf_, tile_len_ * sizeof(int16_t));
    pipe.InitBuffer(dst2_int16_buf_, tile_len_ * sizeof(int16_t));
    pipe.InitBuffer(tmp2_int16_buf_, tile_len_ * sizeof(int16_t));
    pipe.InitBuffer(tmp1_half_buf_, tile_len_ * sizeof(half));
    pipe.InitBuffer(tmp2_half_buf_, tile_len_ * sizeof(half));
    pipe.InitBuffer(tmp3_half_buf_, tile_len_ * sizeof(half));
    pipe.InitBuffer(tmp_get_start_half_buf_, tile_len_ * sizeof(half));
    pipe.InitBuffer(tmp_popzero_half_buf_, tile_len_ * sizeof(half));
    pipe.InitBuffer(tmp_float_buf_, tile_len_ * sizeof(float));
    pipe.InitBuffer(tmp2_float_buf_, tile_len_ * sizeof(float));
    pipe.InitBuffer(tmp_get_start_float_buf_, tile_len_ * sizeof(float));
  }

  /**
   * @brief Run the kernel - process all tiles.
   */
  __aicore__ inline void Process() {
    uint32_t tile_idx = 0;
    DataTypeT running_sum = 0;
    for (uint32_t mat_tile_idx = 0; mat_tile_idx < num_matrix_tiles_;
         mat_tile_idx++) {
      // Sync with Cube core
      sync::SyncGroup<sync::GroupSyncDirection::FULL>();

      // FIXME: uses only one vector core
      if (GetSubBlockIdx() > 0) {
        continue;
      }

      for (uint32_t i = 0; i < tile_len_; i++) {
        VecIter(tile_idx, running_sum);
        tile_idx++;
      }
    }
  }

  /**
   * @brief Executes an iteration on tile with index `tile_idx`.
   *
   * @param tile_idx Tile index.
   * @param running_sum Running sum accumulation value of previous tile.
   */
  __aicore__ inline void VecIter(uint32_t tile_idx, DataTypeT &running_sum) {
    const bool is_last_tile = (tile_idx + 1) * tile_len_ < vec_len_;
    const uint32_t num_elems_to_process =
        is_last_tile ? tile_len_ : vec_len_ - tile_idx * tile_len_;

    copy::CopyGmToVec<DataTypeT>(vecin_input_q_,
                                 global_input_[tile_idx * tile_len_],
                                 num_elems_to_process);
    copy::CopyGmToVec<FlagOutputT>(vecin_scanned_flag_q_,
                                   global_scanned_flag_[tile_idx * tile_len_],
                                   num_elems_to_process);

    RevertSpecOnSegmentsWithinBlock(tile_idx, running_sum,
                                    num_elems_to_process);

    copy::CopyVecToGm<DataTypeT>(global_output_[tile_idx * tile_len_],
                                 vecout_q_, num_elems_to_process);
  }

  /**
   * @brief Reverts speculation on a tile of both the input data and flag
   * vectors.
   *
   * @param tile_idx Index of input tile.
   * @param running_sum Accumulation value of last element of previous tile.
   * @param num_elems Number of element to process.
   */
  __aicore__ inline void RevertSpecOnSegmentsWithinBlock(uint32_t tile_idx,
                                                         DataTypeT &running_sum,
                                                         uint32_t num_elems) {
    LocalTensor<DataTypeT> input_vec_lt = vecin_input_q_.DeQue<DataTypeT>();
    LocalTensor<FlagOutputT> flag_vec_lt =
        vecin_scanned_flag_q_.DeQue<FlagOutputT>();

    LocalTensor<DataTypeT> output_vec_lt = vecout_q_.AllocTensor<DataTypeT>();

    (void)tile_idx;

    // Go over each segment and revert speculation, if necessary
    const int32_t num_segments = flag_vec_lt(tile_len_ - 1);
    const int16_t first_flag = flag_vec_lt(0);
    if (num_segments == 1 and first_flag == 1) {
      DataCopy(output_vec_lt, input_vec_lt, num_elems);
      PipeBarrier<PIPE_V>();
    } else if (num_segments == 0) {
      Adds(output_vec_lt, input_vec_lt, running_sum, num_elems);
      PipeBarrier<PIPE_V>();
    } else {
      Adds(output_vec_lt, input_vec_lt, running_sum, num_elems);
      PipeBarrier<PIPE_V>();
      for (int16_t seg_idx = 1; seg_idx <= num_segments; seg_idx++) {
        const DataTypeT delta =
            GetDelta(output_vec_lt, flag_vec_lt, seg_idx, running_sum);

        PipeBarrier<PIPE_V>();

        FixSpecInPlace(output_vec_lt, flag_vec_lt, seg_idx, delta);
        PipeBarrier<PIPE_V>();
      }
    }

    vecout_q_.EnQue<DataTypeT>(output_vec_lt);
    vecin_input_q_.FreeTensor<DataTypeT>(input_vec_lt);
    vecin_scanned_flag_q_.FreeTensor<FlagOutputT>(flag_vec_lt);

    if (num_elems == tile_len_) {
      running_sum = output_vec_lt(tile_len_ - 1);
    }
  }

  /**
   * @brief Returns the segment delta correction value
   *
   * @param input Input scanned tile of data.
   * @param flag Input scanned tile of flags.
   * @param segment_id The id of the input segment.
   * @param running_sum Running sum from previous tile.
   * @return Delta correction value
   */
  __aicore__ inline DataTypeT GetDelta(const LocalTensor<DataTypeT> &input,
                                       const LocalTensor<FlagOutputT> &flag,
                                       int16_t segment_id,
                                       DataTypeT running_sum) {
    const int16_t segment_start_index =
        GetStartIndexOfSegment(flag, segment_id);
    PipeBarrier<PIPE_V>();

    const int16_t delta_index = segment_start_index - 1;
    const DataTypeT delta = delta_index >= 0 ? input(delta_index) : running_sum;

    return delta;
  }

  /**
   * @brief Returns the start index of a segment given a `segment_id`.
   *
   * @param flag Tile of input scanned flag vector.
   * @param segment_id The id of the input segment.
   * @return Start index of the segment with segment_id.
   */
  __aicore__ inline int16_t GetStartIndexOfSegment(
      const LocalTensor<FlagOutputT> &flag, int16_t segment_id) {
    const half threshold = static_cast<half>(static_cast<float>(segment_id) -
                                             static_cast<float>(0.1));

    LocalTensor<float> float_buf = tmp_get_start_float_buf_.Get<float>();
    LocalTensor<half> f_lt = tmp_get_start_half_buf_.Get<half>();

    const LocalTensor<uint16_t> f_uint16_lt =
        f_lt.template ReinterpretCast<uint16_t>();
    const LocalTensor<int16_t> f_int16_lt =
        f_lt.template ReinterpretCast<int16_t>();
    const LocalTensor<half> f_half_lt = f_lt.template ReinterpretCast<half>();

    const uint32_t size = f_lt.GetSize();

    Cast(float_buf, flag, RoundMode::CAST_NONE, size);
    Cast(f_lt, float_buf, RoundMode::CAST_NONE, size);
    Duplicate(float_buf, static_cast<float>(0), float_buf.GetSize());

    Muls(f_lt, f_lt, static_cast<half>(-1), size);

    Adds(f_lt, f_lt, threshold, size);

    PipeBarrier<PIPE_V>();

    Not(f_uint16_lt, f_uint16_lt, size);

    ShiftRight(f_uint16_lt, f_uint16_lt, (uint16_t)15, size);

    PipeBarrier<PIPE_V>();

    Cast(f_half_lt, f_int16_lt, RoundMode::CAST_NONE, size);
    Cast(float_buf, f_half_lt, RoundMode::CAST_NONE, size);
    PipeBarrier<PIPE_V>();

    ReduceSum(float_buf /*dst*/, float_buf /*src*/, float_buf /*work*/,
              float_buf.GetSize());

    // Returns the reduction value of ReduceSum
    return static_cast<int16_t>(AscendC::GetAccVal<float>());
  }

  /**
   * @brief Compare vector with a scalar using abs(x - scalar) and count
   * number of zeros
   *
   * @param dst Tile of output vector containing 1 iff element equals to
   * `scalar`
   * @param src Tile of input vector.
   * @param scalar Scalar to compare vector against.
   */
  __aicore__ inline void CustomCompareScalarEQ(LocalTensor<half> &dst,
                                               LocalTensor<int16_t> &src,
                                               int16_t scalar) {
    LocalTensor<half> tmp_half_lt = tmp1_half_buf_.Get<half>();
    LocalTensor<uint16_t> tmp_uint16_lt =
        tmp_half_lt.template ReinterpretCast<uint16_t>();
    LocalTensor<int16_t> tmp_int16_lt =
        tmp_half_lt.template ReinterpretCast<int16_t>();

    Cast(tmp_half_lt, src, RoundMode::CAST_NONE, tile_len_);
    Adds(tmp_half_lt, tmp_half_lt, static_cast<half>(-scalar), tile_len_);

    Not(tmp_uint16_lt, tmp_uint16_lt, tile_len_);

    ShiftRight(tmp_uint16_lt, tmp_uint16_lt, (uint16_t)15, tile_len_);

    PipeBarrier<PIPE_V>();

    Cast(dst, tmp_int16_lt, RoundMode::CAST_NONE, tile_len_);
  }

  /**
   * @brief Fixes the speculation in-place
   *
   * FixSpecInPlace performs the following vector operation,
   * output = output - (flag == segment_id) * delta
   *
   * @param output Output local tensor
   * @param flag Scanned flag vector
   * @param segment_id The segment id
   * @param delta Speculation correction value.
   */
  __aicore__ inline void FixSpecInPlace(LocalTensor<float> &output,
                                        LocalTensor<FlagOutputT> &flag,
                                        int16_t segment_id, DataTypeT delta) {
    if (delta == 0) {
      return;
    }

    LocalTensor<half> half_buf = tmp1_half_buf_.Get<half>();
    LocalTensor<int16_t> flag_int16_lt = tmp1_int16_buf_.Get<int16_t>();

    Cast(flag_int16_lt, flag, RoundMode::CAST_NONE, tile_len_);
    CustomCompareScalarEQ(half_buf, flag_int16_lt, segment_id);

    LocalTensor<float> float_buf = tmp_float_buf_.Get<float>();

    Muls(half_buf, half_buf, static_cast<half>(delta), tile_len_);
    Cast(float_buf, half_buf, RoundMode::CAST_NONE, tile_len_);
    Sub(output, output, float_buf, tile_len_);
  }

 private:
  TPipe pipe;

  TQue<QuePosition::VECIN, BUFFER_NUM> vecin_input_q_;
  TQue<QuePosition::VECIN, BUFFER_NUM> vecin_scanned_flag_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> vecout_q_;

  TBuf<QuePosition::VECCALC> tmp_uint8_buf_;
  TBuf<QuePosition::VECCALC> tmp2_uint8_buf_;
  TBuf<QuePosition::VECCALC> tmp3_uint8_buf_;
  TBuf<QuePosition::VECCALC> tmp1_int16_buf_;
  TBuf<QuePosition::VECCALC> tmp2_int16_buf_;
  TBuf<QuePosition::VECCALC> dst_int16_buf_;
  TBuf<QuePosition::VECCALC> dst2_int16_buf_;
  TBuf<QuePosition::VECCALC> tmp1_half_buf_;
  TBuf<QuePosition::VECCALC> tmp2_half_buf_;
  TBuf<QuePosition::VECCALC> tmp3_half_buf_;
  TBuf<QuePosition::VECCALC> tmp_get_start_float_buf_;
  TBuf<QuePosition::VECCALC> tmp_get_start_half_buf_;
  TBuf<QuePosition::VECCALC> tmp_popzero_half_buf_;
  TBuf<QuePosition::VECCALC> tmp_float_buf_;
  TBuf<QuePosition::VECCALC> tmp2_float_buf_;

  GlobalTensor<DataTypeT> global_input_;
  GlobalTensor<FlagOutputT> global_scanned_flag_;
  GlobalTensor<DataTypeT> global_output_;

  const uint32_t vec_len_;
  const uint32_t tile_len_;
  const uint32_t num_matrix_tiles_;
  const uint32_t num_tiles_;
};