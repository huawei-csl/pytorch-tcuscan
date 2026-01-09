/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_diff.h
 * @brief Kernel implementing a Vector diff kernel operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Returns the difference give an input vector x.
 *
 * In particular, the output vector z has entries z[i] = x[i] - x[i-1] where
 * x[-1] = 0. In short, the kernel implements (numpy notation) np.diff(x,
 * prepend=[0])
 *
 *  The kernel expects the datatype to be `half` or `float32`.
 */
template <typename DataTypeT = half>
class KernelDiff {
  constexpr static uint32_t BUFFER_NUM = 2;
  constexpr static int32_t SHIFT_BYTES = sizeof(DataTypeT);
  constexpr static int32_t START_OFFSET_BYTES = sizeof(DataTypeT);

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Dimension of the input vectors.
   * @param [in] tile_len Tile length.
   */
  __aicore__ inline KernelDiff(uint32_t vec_len, uint32_t tile_len)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        vec_len_(vec_len),
        tile_len_(tile_len),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_len_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, vec_core_num_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_in Pointer to the input vector in global memory.
   * @param [in] vec_out Pointer to the output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR vec_out) {
    global_in_.SetGlobalBuffer((__gm__ DataTypeT*)vec_in, vec_len_);
    global_out_.SetGlobalBuffer((__gm__ DataTypeT*)vec_out + 1, vec_len_ - 1);
    global_out_first_elem_.SetGlobalBuffer((__gm__ DataTypeT*)vec_out, 1);

    pipe.InitBuffer(in_q_, BUFFER_NUM, (tile_len_ + 1) * sizeof(DataTypeT));
    pipe.InitBuffer(out_q_, BUFFER_NUM, tile_len_ * sizeof(DataTypeT));
    pipe.InitBuffer(tbuf_, (tile_len_ + 1) * sizeof(uint32_t));
  }

  /**
   * @brief Run the kernel.
   *
   */
  __aicore__ inline void Process() {
    uint32_t global_offset =
        GetBlockIdx() * tile_len_ * max_num_tiles_per_block_;
    const uint32_t num_tiles_to_process = tcuscan::scalar::GetWorkDistribution(
        vec_len_, tile_len_, vec_core_num_);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const bool full_tile = global_offset + tile_len_ < vec_len_;
      const uint32_t num_elems_to_process_ =
          full_tile ? tile_len_ : vec_len_ - global_offset - 1;

      copy::CopyGmToVec(in_q_, global_in_[global_offset],
                        num_elems_to_process_ + 1);
      ProcessTile(tile_idx, num_elems_to_process_);
      copy::CopyVecToGm(global_out_[global_offset], out_q_,
                        num_elems_to_process_);
      global_offset += tile_len_;
    }
  }

 private:
  __aicore__ inline void ProcessTile(uint32_t progress, uint32_t num_elems_) {
    LocalTensor<DataTypeT> vec_in_lt = in_q_.DeQue<DataTypeT>();
    const LocalTensor<DataTypeT> vec_out_lt = out_q_.AllocTensor<DataTypeT>();

    LocalTensor<uint32_t> cols_uint32_lt = tbuf_.Get<uint32_t>();
    LocalTensor<int32_t> cols_int32_lt =
        cols_uint32_lt.template ReinterpretCast<int32_t>();

    // Create index vector: 2,4,6,... (for 16-bits) or 4,8,12,... (for
    // 32-bits)
    ArithProgression<int32_t>(cols_int32_lt, SHIFT_BYTES, START_OFFSET_BYTES,
                              num_elems_ - 1);

    // Skip first element of input vector vec_in_lt,
    // i.e., vec_out_lt = [vec_in_lt(1), vec_in_lt(2), ...]
    AscendC::Gather(vec_out_lt, vec_in_lt, cols_uint32_lt,
                    static_cast<uint32_t>(0), num_elems_ - 1);

    Sub(vec_out_lt, vec_out_lt, vec_in_lt, num_elems_);

    // Fix last values with scalar unit
    // vec_in_lt has length 'num_elems_ + 1'
    const float delta = static_cast<float>(vec_in_lt(num_elems_)) -
                        static_cast<float>(vec_in_lt(num_elems_ - 1));
    vec_out_lt.SetValue(num_elems_ - 1, static_cast<DataTypeT>(delta));

    // Set manually the first element in GM
    if (GetBlockIdx() == 0 && progress == 0) {
      global_out_first_elem_(0) = vec_in_lt(0);
      DataCacheCleanAndInvalid<DataTypeT, CacheLine::SINGLE_CACHE_LINE,
                               DcciDst::CACHELINE_OUT>(global_out_first_elem_);
    }

    out_q_.EnQue<DataTypeT>(vec_out_lt);
    in_q_.FreeTensor<DataTypeT>(vec_in_lt);
  }

  TPipe pipe;

  TQue<QuePosition::VECIN, BUFFER_NUM> in_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> out_q_;

  TBuf<QuePosition::VECCALC> tbuf_;

  GlobalTensor<DataTypeT> global_in_;
  GlobalTensor<DataTypeT> global_out_;
  GlobalTensor<DataTypeT> global_out_first_elem_;

  const uint32_t vec_core_num_;
  const uint32_t vec_len_;
  const uint32_t tile_len_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;
};

/**
 * @brief Run the `diff` kernel.
 *
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 * @tparam DataTypeT Data type of kernel
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] vec_len Dimension of the input vector.
 * @param [in] tile_len Tile length.
 */
template <bool ForceMixMode = true, typename DataTypeT>
__aicore__ inline void run_diff(GM_ADDR vec_in, GM_ADDR vec_out,
                                uint32_t vec_len, uint32_t tile_len) {
  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }
  if ASCEND_IS_AIV {
    KernelDiff<DataTypeT> op(vec_len, tile_len);
    op.Init(vec_in, vec_out);
    op.Process();
  }
}

}  // namespace tcuscan
