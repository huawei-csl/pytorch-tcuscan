/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_radix_enc.h
 * @brief Kernel implementing the encoding/decoding step to enable floating
 * point sorting with radix sort.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Encodes/decodes floating point values into a format that can be
 * sorted by the integer radix-sort.
 *
 * Radix sort correctly orders positive floating point values, but the order of
 * negative values is reversed. This kernel performs a bit-wise not on the
 * negative numbers so that they are correctly ordered, but preserves the value
 * of the sign bit. It preserves the sign bit because the integer radix-sort
 * already correclty handles it, by reversing the last split operation.
 *
 * In other words the key is `x ^ (sign(x) ? ~SIGN_MASK : 0)`, which is its
 * own inverse: running the kernel on already encoded values decodes them back
 * to the original ones.
 */
template <typename FloatT = half>
class KernelRadixEnc {
  constexpr static uint32_t BUFFER_NUM = 1;

  /// @brief Signed integer as wide as the float, for the arithmetic shift.
  using IntT =
      typename std::conditional<sizeof(FloatT) == 2, int16_t, int32_t>::type;
  /// @brief Unsigned view of the same width, for the logical shift.
  using UIntT =
      typename std::conditional<sizeof(FloatT) == 2, uint16_t, uint32_t>::type;
  /// @brief Index of the sign bit.
  constexpr static IntT SIGN_BIT = sizeof(FloatT) * 8 - 1;
  /// @brief Number of 16-bit words a key occupies.
  constexpr static uint32_t WORDS_PER_KEY = sizeof(FloatT) / sizeof(uint16_t);

  static_assert(sizeof(FloatT) == 2 || sizeof(FloatT) == 4,
                "Radix encoding handles `half` and `float`.");

 public:
  /**
   * @brief Class constructor.
   * @param [in] vec_len Dimension of the input vectors.
   * @param [in] tile_size Tile size.
   */
  __aicore__ inline KernelRadixEnc(uint32_t vec_len, uint32_t tile_size)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        vec_len_(vec_len),
        tile_size_(tile_size),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_size_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, vec_core_num_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] x Pointer to the input vector in global memory.
   * @param [in] y Pointer to the output matrix in global memory.
   */
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
    global_x_.SetGlobalBuffer((__gm__ FloatT*)x, vec_len_);
    global_y_.SetGlobalBuffer((__gm__ FloatT*)y, vec_len_);

    pipe.InitBuffer(in_q_, BUFFER_NUM, tile_size_ * sizeof(FloatT));
    pipe.InitBuffer(out_q_, BUFFER_NUM, tile_size_ * sizeof(FloatT));
    pipe.InitBuffer(mask_buf_, tile_size_ * sizeof(IntT));
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    uint32_t global_offset =
        GetBlockIdx() * tile_size_ * max_num_tiles_per_block_;
    const uint32_t num_tiles_to_process = tcuscan::scalar::GetWorkDistribution(
        vec_len_, tile_size_, vec_core_num_);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const bool full_tile = global_offset + tile_size_ <= vec_len_;
      const uint32_t num_elems_to_process_ =
          full_tile ? tile_size_ : vec_len_ - global_offset;

      copy::CopyGmToVec(in_q_, global_x_[global_offset], num_elems_to_process_);
      RadixEncTile();
      copy::CopyVecToGm(global_y_[global_offset], out_q_,
                        num_elems_to_process_);
      global_offset += tile_size_;
    }
  }

 private:
  __aicore__ inline void RadixEncTile() {
    LocalTensor<FloatT> x_lt = in_q_.template DeQue<FloatT>();
    const LocalTensor<FloatT> y_lt = out_q_.template AllocTensor<FloatT>();
    // Views of the same scratch buffer; the shifts below switch between the
    // signed and the unsigned one, so they need a barrier in between.
    const LocalTensor<IntT> mask_int_lt = mask_buf_.template Get<IntT>();
    const LocalTensor<UIntT> mask_uint_lt =
        mask_int_lt.template ReinterpretCast<UIntT>();

    const uint32_t size = x_lt.GetSize();
    // `Not`/`Xor` only exist for 16-bit elements, but xor is bitwise, so a
    // 32-bit key can be combined two 16-bit words at a time.
    const uint32_t num_words = size * WORDS_PER_KEY;

    DataCopy(mask_int_lt, x_lt.template ReinterpretCast<IntT>(), size);
    PipeBarrier<PIPE_V>();

    // Arithmetic shift: all ones for a negative key, all zeros otherwise.
    ShiftRight(mask_int_lt, mask_int_lt, SIGN_BIT, size);
    PipeBarrier<PIPE_V>();

    // Logical shift: drops the sign bit, leaving the mask of the bits to
    // flip, so that the sign bit itself survives the xor untouched.
    ShiftRight(mask_uint_lt, mask_uint_lt, static_cast<UIntT>(1), size);
    PipeBarrier<PIPE_V>();

    Xor(y_lt.template ReinterpretCast<uint16_t>(),
        x_lt.template ReinterpretCast<uint16_t>(),
        mask_int_lt.template ReinterpretCast<uint16_t>(), num_words);

    out_q_.template EnQue<FloatT>(y_lt);
    in_q_.FreeTensor(x_lt);
  }
  TPipe pipe;

  TQue<QuePosition::VECIN, BUFFER_NUM> in_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> out_q_;
  TBuf<QuePosition::VECCALC> mask_buf_;

  GlobalTensor<FloatT> global_x_;
  GlobalTensor<FloatT> global_y_;

  const uint32_t vec_core_num_;
  const uint32_t vec_len_;
  const uint32_t tile_size_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;
};

/**
 * @brief Run the `radix encode` kernel.
 *
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 *
 * @param [in] x Pointer to the input vector.
 * @param [in] y Pointer ot the output vector.
 * @param [in] vec_len Dimension of the input vector.
 * @param [in] tile_size Tile size.
 */
template <typename FloatT = half, bool ForceMixMode = true>
__aicore__ inline void run_radix_encode(GM_ADDR x, GM_ADDR y, uint32_t vec_len,
                                        uint32_t tile_size) {
  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }
  if ASCEND_IS_AIV {
    KernelRadixEnc<FloatT> op(vec_len, tile_size);
    op.Init(x, y);
    op.Process();
  }
}

}  // namespace tcuscan
