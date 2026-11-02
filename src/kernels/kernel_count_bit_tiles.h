/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_count_bit_tiles.h
 * @brief Kernel counting, per block, how many inputs have a given bit set.
 */
#pragma once

#include "kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Counts, per block, how many inputs have a given bit set.
 *
 * This is the counting half of a one-bit radix pass. `KernelReduceTiles` cannot
 * serve here because it would have to read a materialised byte mask, and its
 * accumulator type for 16-bit input is 16-bit, which overflows a block's count.
 * Reading the keys directly and extracting the bit in UB removes a full
 * round-trip of the mask through global memory.
 *
 * The block partitioning matches `KernelSplit` exactly, so entry `i` of the
 * output is the number of ones in the range block `i` will later process.
 */
class KernelCountBitTiles {
  using AccT = int32_t;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] tile_len Tile length.
   * @param [in] vec_len Number of elements in the input vector.
   * @param [in] bit_idx Index of the bit to count.
   */
  __aicore__ inline KernelCountBitTiles(uint32_t tile_len, uint32_t vec_len,
                                        uint16_t bit_idx)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        tile_len_(tile_len),
        vec_len_(vec_len),
        bit_idx_(bit_idx),
        num_tiles_(scalar::CeilDiv(vec_len, tile_len)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, vec_core_num_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] input Pointer to the keys in global memory.
   * @param [in] output Pointer to the per-block counts in global memory.
   */
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
    global_input_.SetGlobalBuffer((__gm__ uint16_t*)input, vec_len_);
    global_output_.SetGlobalBuffer((__gm__ AccT*)output, vec_core_num_);

    pipe_.InitBuffer(in_q_, 2, tile_len_ * sizeof(uint16_t));
    pipe_.InitBuffer(acc_buf_, tile_len_ * sizeof(float));
    pipe_.InitBuffer(res_q_, 1, MIN_VEC_SIZE * sizeof(AccT));
  }

  /**
   * @brief Run the kernel - count over all tiles of this block.
   */
  __aicore__ inline void Process() {
    const uint32_t num_tiles_to_process = tcuscan::scalar::GetWorkDistribution(
        vec_len_, tile_len_, vec_core_num_);
    uint32_t global_offset =
        GetBlockIdx() * tile_len_ * max_num_tiles_per_block_;

    const LocalTensor<float> acc_lt = acc_buf_.Get<float>();
    float sum = 0.f;

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const uint32_t num_elems =
          scalar::NextTileLen(tile_len_, global_offset, vec_len_);
      if (num_elems > 0) {
        copy::CopyGmToVec(in_q_, global_input_[global_offset], num_elems);
        LocalTensor<uint16_t> in_lt = in_q_.DeQue<uint16_t>();

        // Move the wanted bit into the sign position and back down, leaving
        // 0 or 1 per element. Both casts below are exact for those values.
        ShiftLeft(in_lt, in_lt, static_cast<uint16_t>(15 - bit_idx_),
                  num_elems);
        PipeBarrier<PIPE_V>();
        ShiftRight(in_lt, in_lt, static_cast<uint16_t>(15), num_elems);
        PipeBarrier<PIPE_V>();

        const LocalTensor<half> half_lt = in_lt.ReinterpretCast<half>();
        const LocalTensor<int16_t> int_lt = in_lt.ReinterpretCast<int16_t>();
        Cast(half_lt, int_lt, RoundMode::CAST_NONE, num_elems);
        PipeBarrier<PIPE_V>();
        Cast(acc_lt, half_lt, RoundMode::CAST_NONE, num_elems);

        ReduceSum(acc_lt, acc_lt, acc_lt, num_elems);
        sum += AscendC::GetAccVal<float>();

        in_q_.FreeTensor(in_lt);
      }
      global_offset += num_elems;
    }

    copy::CopyScalarToGm(global_output_[GetBlockIdx()], res_q_,
                         static_cast<AccT>(sum));
  }

 private:
  TPipe pipe_;

  TQue<QuePosition::VECIN, 2> in_q_;
  TQue<QuePosition::VECOUT, 1> res_q_;
  TBuf<QuePosition::VECCALC> acc_buf_;

  GlobalTensor<uint16_t> global_input_;
  GlobalTensor<AccT> global_output_;

  constexpr static int32_t MIN_VEC_SIZE = UB_ALIGNMENT / sizeof(AccT);

  const uint32_t vec_core_num_;
  const uint32_t tile_len_;
  const uint32_t vec_len_;
  const uint16_t bit_idx_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;
};

}  // namespace tcuscan
