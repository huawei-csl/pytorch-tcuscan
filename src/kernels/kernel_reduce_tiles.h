/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_reduce_tiles.h
 * @brief Kernel implementing a tile reducer using the vector cores.
 *
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "kernel_utils.h"

using namespace AscendC;
using namespace kernel_utils;

/**
 * @brief Performs add-reduce operation on input tiles.
 *
 * Divides an input vector into tiles of size `tile_size` and reduces each of
 * them separately using addition operation. Because of the UB alignment
 * constraints, each result scalar is represented by a 32 bytes vector, where
 * the first `sizeof(AccT)` bytes is the actual result and the rest are dummy
 * padding data.
 *
 * @tparam InputT Data type of the input vector.
 * @tparam BufferNum Number of buffers. If set to `2`, double-buffering is
 * enabled.
 */
template <typename InputT, int32_t BufferNum = 2>
class KernelReduceTiles {
  using IntermediateT = half;
  using AccT = cube_unit::CubeOutType_t<InputT>;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] tile_size Size of the tile.
   * @param [in] vec_len Number of elements in an input vector.
   */
  __aicore__ inline KernelReduceTiles(uint32_t tile_size, uint32_t vec_len)
      : block_num_(GetBlockNum() * GetTaskRation()),
        tile_size_(tile_size),
        vec_len_(vec_len),
        num_tiles_(scalar::FloorDiv(vec_len, tile_size_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, block_num_)),
        red1_size_((tile_size_ / 16 > RED2_SIZE) ? tile_size_ / 16
                                                 : tile_size_ / 4) {
    static_assert(IS_ACC_T_SUPPORTED, "Unsupported accumulator type.");
    static_assert(IS_IN_T_8BIT || !IS_IN_T_UNSIGNED, "Unsupported input type.");
    ASCENDC_ASSERT((vec_len % tile_size_ == 0), {
      KERNEL_LOG(KERNEL_ERROR,
                 "The length of the input vector (%d) must be "
                 "divisible by the minimum amount of elements "
                 "processed by every blocks (%d).",
                 vec_len, tile_size_);
    });
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] input Pointer to input vector in global memory.
   * @param [in] output Pointer to output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
    global_input_.SetGlobalBuffer((__gm__ InputT *)input, vec_len_);
    global_output_.SetGlobalBuffer((__gm__ AccT *)output, block_num_);

    pipe.InitBuffer(vec_tile_input_q_, BufferNum, tile_size_ * sizeof(InputT));
    if constexpr (REQ_INTERMEDIATE_CAST) {
      pipe.InitBuffer(vec_tile_intermediate_buf_,
                      tile_size_ * sizeof(IntermediateT));
    }
    pipe.InitBuffer(vec_tile_q_, tile_size_ * sizeof(AccT));

    pipe.InitBuffer(red1_buf_, red1_size_ * sizeof(AccT));
    pipe.InitBuffer(red2_buf_, RED2_SIZE * sizeof(AccT));
    pipe.InitBuffer(res_q_, BufferNum, MIN_VEC_SIZE * sizeof(AccT));
  }

  /**
   * @brief Run the kernel - process all tiles.
   */
  __aicore__ inline void Process() {
    const uint32_t num_tiles_to_process =
        kernel_utils::scalar::GetWorkDistribution(vec_len_, tile_size_,
                                                  block_num_);

    if (num_tiles_to_process == 0) {
      return;
    }

    const LocalTensor<AccT> input_lt = vec_tile_q_.Get<AccT>();

    AccT sum = 0;

    if constexpr (reduce::IsAscendReduceSumSupported<AccT>) {
      for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
        LoadToAccT(input_lt, tile_idx);
        ReduceSum(input_lt, input_lt, input_lt, tile_size_);

        sum += AscendC::GetAccVal<AccT>();
      }
    } else {
      LoadToAccT(input_lt, 0);
      const LocalTensor<AccT> red1_lt = red1_buf_.Get<AccT>();
      reduce::ReduceVecAdd<true /*AllocateAcc*/, AccT>(red1_lt, input_lt);

      for (uint32_t tile_idx = 1; tile_idx < num_tiles_to_process; tile_idx++) {
        LoadToAccT(input_lt, tile_idx);
        reduce::ReduceVecAdd<false /*AllocateAcc*/, AccT>(red1_lt, input_lt);
      }
      const LocalTensor<AccT> red2_lt = red2_buf_.Get<AccT>();
      reduce::ReduceVecAdd<true /*AllocateAcc*/, AccT>(red2_lt, red1_lt);
      sum = reduce::ReduceScalarAdd<AccT>(red2_lt, red2_lt.GetSize());
    }
    copy::CopyScalarToGm(global_output_[GetBlockIdx()], res_q_, sum);
  }

 private:
  __aicore__ inline void LoadToAccT(const LocalTensor<AccT> &dst_lt,
                                    uint32_t tile_idx) {
    const uint32_t global_offset =
        GetBlockIdx() * tile_size_ * max_num_tiles_per_block_;

    copy::CopyGmToVec(vec_tile_input_q_,
                      global_input_[global_offset + tile_idx * tile_size_]);
    LocalTensor<InputT> input_lt = vec_tile_input_q_.template DeQue<InputT>();
    if constexpr (REQ_INTERMEDIATE_CAST) {
      const LocalTensor<IntermediateT> intermediate_lt =
          vec_tile_intermediate_buf_.Get<IntermediateT>();
      Cast(intermediate_lt, input_lt, RoundMode::CAST_NONE, tile_size_);

      constexpr auto cast_mode = std::is_same_v<AccT, float>
                                     ? RoundMode::CAST_NONE
                                     : RoundMode::CAST_RINT;
      Cast(dst_lt, intermediate_lt, cast_mode, tile_size_);
    } else {
      Cast(dst_lt, input_lt, RoundMode::CAST_NONE, tile_size_);
    }
    vec_tile_input_q_.FreeTensor(input_lt);
  }
  TPipe pipe;

  TQue<QuePosition::VECIN, BufferNum> vec_tile_input_q_;
  TBuf<QuePosition::VECCALC> vec_tile_intermediate_buf_;
  TBuf<QuePosition::VECCALC> vec_tile_q_;
  TBuf<QuePosition::VECCALC> red1_buf_;
  TBuf<QuePosition::VECCALC> red2_buf_;
  TQue<QuePosition::VECOUT, BufferNum> res_q_;

  GlobalTensor<InputT> global_input_;
  GlobalTensor<AccT> global_output_;

  constexpr static int32_t MIN_VEC_SIZE = UB_ALIGNMENT / sizeof(AccT);
  constexpr static bool IS_IN_T_8BIT =
      std::is_same_v<InputT, uint8_t> || std::is_same_v<InputT, int8_t>;
  // AscendC casts do not support unsigned integers in many configurations.
  constexpr static bool IS_IN_T_UNSIGNED = std::is_same_v<InputT, uint8_t> ||
                                           std::is_same_v<InputT, uint16_t> ||
                                           std::is_same_v<InputT, uint32_t>;
  constexpr static bool IS_ACC_T_SUPPORTED =
      std::is_same_v<AccT, float> || std::is_same_v<AccT, int32_t>;
  constexpr static bool REQ_INTERMEDIATE_CAST = IS_IN_T_8BIT;

  const uint32_t block_num_;
  const uint32_t tile_size_;
  const uint32_t vec_len_;

  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;

  const int32_t red1_size_;
  constexpr static int32_t RED2_SIZE = 8;
};
