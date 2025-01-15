/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * @file kernel_vadd.h
 * @brief Kernel implementing a Vector add kernel operation.
 */
#pragma once

#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
__aicore__ inline T CeilDiv(T value, T divisor) {
  return (value + divisor - 1) / divisor;
}

__aicore__ inline uint32_t GetWorkDistribution(uint32_t vec_len,
                                               uint32_t tile_size,
                                               uint32_t block_n) {
  const uint32_t num_tiles = CeilDiv(vec_len, tile_size);
  const uint32_t max_num_tiles_per_block = CeilDiv(num_tiles, block_n);
  uint32_t num_tiles_to_process = max_num_tiles_per_block;
  const int tiles_left =
      (int)num_tiles - (int)(GetBlockIdx() * max_num_tiles_per_block);

  if (tiles_left < 0) {
    num_tiles_to_process = 0;
  } else if (tiles_left < static_cast<int>(max_num_tiles_per_block)) {
    num_tiles_to_process = tiles_left;
  }
  return num_tiles_to_process;
}

/**
 * @brief Adds two vectors element-wise.
 *
 * Function : z = x + y
 * This sample is a very basic sample that implements vector add on Ascend
 * plaform. In this sample: Length of x / y / z is 8*2048. Num of vector core
 * used in sample is 8. Length for each core to compute is 2048. Tiles for each
 * core is 8 which means we add 2048/8=256 elements in one loop.
 *
 * This is just a tile strategy for demonstration, in fact we can compute at
 * most 128*255 elements in one loop for b16 type.
 */
class KernelAdd {
  /// Number of buffers for each queue.
  constexpr static int32_t BUFFER_NUM = 2;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Dimension of the input vectors.
   * @param [in] tile_len Tile length.
   */
  __aicore__ inline KernelAdd(uint32_t vec_len, uint32_t tile_len)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        vec_len_(vec_len),
        tile_len_(tile_len),
        num_tiles_(CeilDiv(vec_len_, tile_len_)),
        max_num_tiles_per_block_(CeilDiv(num_tiles_, vec_core_num_)),
        global_offset_(GetBlockIdx() * tile_len_ * max_num_tiles_per_block_) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] x Pointer to the first input vector in global memory.
   * @param [in] y Pointer to the second input vector in global memory.
   * @param [in] z Pointer to output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
    const uint32_t block_length = tile_len_ * max_num_tiles_per_block_;
    global_x_.SetGlobalBuffer((__gm__ half *)x + global_offset_, block_length);
    global_y_.SetGlobalBuffer((__gm__ half *)y + global_offset_, block_length);
    global_z_.SetGlobalBuffer((__gm__ half *)z + global_offset_, block_length);

    pipe.InitBuffer(x_q_, BUFFER_NUM, tile_len_ * sizeof(half));
    pipe.InitBuffer(y_q_, BUFFER_NUM, tile_len_ * sizeof(half));
    pipe.InitBuffer(z_q_, BUFFER_NUM, tile_len_ * sizeof(half));
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    const uint32_t num_tiles_to_process =
        GetWorkDistribution(vec_len_, tile_len_, vec_core_num_);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const bool full_tile = global_offset_ + tile_len_ < vec_len_;
      const uint32_t num_elems_to_process_ =
          full_tile ? tile_len_ : vec_len_ % tile_len_;

      CopyIn(tile_idx, num_elems_to_process_);
      Compute(num_elems_to_process_);
      CopyOut(tile_idx, num_elems_to_process_);
    }
  }

 private:
  __aicore__ inline void CopyIn(uint32_t progress, uint32_t num_elems) {
    // alloc tensor from queue memory
    LocalTensor<half> x_lt = x_q_.AllocTensor<half>();
    LocalTensor<half> y_lt = y_q_.AllocTensor<half>();
    // copy progress_th tile from global tensor to local tensor
    DataCopy(x_lt, global_x_[progress * tile_len_], num_elems);
    DataCopy(y_lt, global_y_[progress * tile_len_], num_elems);
    // enque input tensors to VECIN queue
    x_q_.EnQue(x_lt);
    y_q_.EnQue(y_lt);
  }
  __aicore__ inline void Compute(uint32_t num_elems) {
    // deque input tensors from VECIN queue
    LocalTensor<half> x_lt = x_q_.DeQue<half>();
    LocalTensor<half> y_lt = y_q_.DeQue<half>();
    LocalTensor<half> z_lt = z_q_.AllocTensor<half>();
    // call Add instr for computation
    Add(z_lt, x_lt, y_lt, num_elems);
    // enque the output tensor to VECOUT queue
    z_q_.EnQue<half>(z_lt);
    // free input tensors for reuse
    x_q_.FreeTensor(x_lt);
    y_q_.FreeTensor(y_lt);
  }
  __aicore__ inline void CopyOut(uint32_t progress, uint32_t num_elems) {
    // deque output tensor from VECOUT queue
    LocalTensor<half> z_lt = z_q_.DeQue<half>();
    // copy progress_th tile from local tensor to global tensor
    DataCopy(global_z_[progress * tile_len_], z_lt, num_elems);
    // free output tensor for reuse
    z_q_.FreeTensor(z_lt);
  }

  TPipe pipe;

  TQue<QuePosition::VECIN, BUFFER_NUM> x_q_;
  TQue<QuePosition::VECIN, BUFFER_NUM> y_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> z_q_;

  GlobalTensor<half> global_x_;
  GlobalTensor<half> global_y_;
  GlobalTensor<half> global_z_;

  const uint32_t vec_core_num_;
  const uint32_t vec_len_;
  const uint32_t tile_len_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;
  const uint32_t global_offset_;
};
