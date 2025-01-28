/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_compress.h
 * @brief Kernel implementing a compress operation.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_scan_multi_core.h"
#include "kernel_utils.h"

using namespace AscendC;
using namespace kernel_utils;

/**
 * @brief Compress the input based on the binary mask.
 *
 * Kernel compresses the input vector into a smaller vector that contains only
 * the input value whose position corresponds to a 1 in the mask. The compress
 * is stable, so the relative order between the elements with the same value in
 * mask is consistent between the input and the output.
 *
 * Kernel also takes a position vector on the input which is an output of
 * inclusive scan on the binary input mask.
 *
 * The algorithm expects the mask datatype to be `int8_t` or `uint8_t`, the
 * positions to be `int32_t` and the input vector's elements to be at least 2
 * bytes.
 *
 * @tparam T Data type of the input vector.
 */
template <typename T>
class KernelCompress {
  using MaskT = uint8_t;
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
   * @param [in] tile_size Number of elements processed by each block in a
   * single iteration.
   */
  __aicore__ inline KernelCompress(uint32_t vec_len, uint32_t tile_size)
      : vec_len_(vec_len),
        block_num_(GetBlockNum() * GetTaskRation()),
        tile_size_(tile_size),
        mask_required_elems_(tile_size / IN_ELEMS_PER_MASK_ELEM),
        packed_mask_tile_size_(mask_required_elems_ < SMALLEST_MASK
                                   ? SMALLEST_MASK
                                   : mask_required_elems_),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_size_ * block_num_)),
        num_elems_per_block_(tile_size_ * num_tiles_),
        num_elems_before_block_(GetBlockIdx() * num_elems_per_block_) {
    // GatherMask requires either 2- or 4-byte elements
    static_assert(sizeof(T) >= 2);
  }

 public:
  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] input Pointer to input vector in global memory.
   * @param [in] mask Pointer to mask vector in global memory.
   * @param [in] pos Pointer to position vector in global memory.
   * @param [in] output Pointer to output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR mask, GM_ADDR pos,
                              GM_ADDR output) {
    global_input_.SetGlobalBuffer((__gm__ T *)input + num_elems_before_block_,
                                  num_elems_per_block_);
    global_output_.SetGlobalBuffer((__gm__ T *)output, vec_len_);
    global_mask_.SetGlobalBuffer((__gm__ MaskT *)mask + num_elems_before_block_,
                                 num_elems_per_block_);
    global_pos_.SetGlobalBuffer((__gm__ PosT *)pos, vec_len_);

    pipe.InitBuffer(vec_in_q_, 1, tile_size_ * sizeof(T));
    pipe.InitBuffer(mask_in_q_, 1, tile_size_ * sizeof(MaskT));
    pipe.InitBuffer(mask_fp16_buf_, tile_size_ * sizeof(half));
    pipe.InitBuffer(packed_mask_buf_,
                    packed_mask_tile_size_ * sizeof(PackedMaskT));
    pipe.InitBuffer(gathered_out_q_, 2, tile_size_ * sizeof(T));
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    for (uint32_t block_offset = 0; block_offset < num_elems_per_block_;
         block_offset += tile_size_) {
      LoadAndConvertMask(block_offset);
      copy::CopyGmToVec(vec_in_q_, global_input_[block_offset]);

      uint32_t output_offset;
      CalculatePartsOffsets(block_offset, output_offset);

      LocalTensor<T> input_lt = vec_in_q_.DeQue<T>();

      // Gather ones.
      GatherAndStore(input_lt, gathered_out_q_, global_output_[output_offset]);

      vec_in_q_.FreeTensor(input_lt);
    }
  }

 private:
  __aicore__ inline void CalculatePartsOffsets(uint32_t offset_in_block,
                                               uint32_t &output_offset) {
    const uint32_t num_elems_before_tile =
        num_elems_before_block_ + offset_in_block;

    output_offset = 0;
    if (num_elems_before_tile > 0) {
      output_offset = global_pos_.GetValue(num_elems_before_tile - 1);
      data_cache::InvalidateLine(global_pos_[num_elems_before_tile - 1]);
    }
  }

  __aicore__ inline void LoadAndConvertMask(uint32_t block_offset) {
    copy::CopyGmToVec(mask_in_q_, global_mask_[block_offset]);
    LocalTensor<MaskT> mask_lt = mask_in_q_.DeQue<MaskT>();

    // Cast mask to fp16 since compare instructions require fp16 or fp32
    // input.
    const LocalTensor<half> mask_fp16_lt = mask_fp16_buf_.Get<half>();
    Cast(mask_fp16_lt, mask_lt, RoundMode::CAST_NONE, tile_size_);
    mask_in_q_.FreeTensor(mask_lt);

    // Create a packed mask in uint8 datatype.
    const LocalTensor<PackedMaskT> packed_mask_8b =
        packed_mask_buf_.Get<PackedMaskT>();
    CompareScalar(packed_mask_8b, mask_fp16_lt, static_cast<half>(1),
                  CMPMODE::EQ, tile_size_);
  }

  template <typename GatherT, int32_t _NumBuf>
  __aicore__ inline void GatherAndStore(
      const LocalTensor<GatherT> &input_lt,
      TQue<QuePosition::VECOUT, _NumBuf> &out_q,
      const GlobalTensor<GatherT> &global) {
    using GatherMaskT = _GatherMaskT<GatherT>;

    uint64_t num_gathered_elems = 0;
    {
      const LocalTensor<GatherMaskT> mask_lt =
          packed_mask_buf_.Get<GatherMaskT>();
      const LocalTensor<GatherT> output_lt =
          out_q.template AllocTensor<GatherT>();

      const uint32_t vector_mask = input_lt.GetSize();
      GatherMask(output_lt, input_lt, mask_lt, true, vector_mask, {1, 1, 8, 8},
                 num_gathered_elems);
      out_q.EnQue(output_lt);
    }

    {
      LocalTensor<GatherT> output_lt = out_q.template DeQue<GatherT>();

      if (num_gathered_elems == 0) {
        out_q.FreeTensor(output_lt);
        return;
      }

      DataCopyExtParams params;
      params.blockCount = 1;
      params.blockLen = num_gathered_elems * sizeof(GatherT);
      params.srcStride = 0;
      params.dstStride = 0;
      DataCopyPad(global, output_lt, params);

      out_q.FreeTensor(output_lt);
    }
  }

  TPipe pipe;

  GlobalTensor<T> global_input_;
  GlobalTensor<T> global_output_;
  GlobalTensor<MaskT> global_mask_;
  GlobalTensor<PosT> global_pos_;

  TQue<QuePosition::VECIN, 1> vec_in_q_;
  TQue<QuePosition::VECIN, 1> mask_in_q_;
  TBuf<QuePosition::VECCALC> mask_fp16_buf_;
  TBuf<QuePosition::VECCALC> packed_mask_buf_;
  TQue<QuePosition::VECOUT, 2> gathered_out_q_;

  const uint32_t vec_len_;
  const uint32_t block_num_;
  const uint32_t tile_size_;
  const uint32_t mask_required_elems_;
  const uint32_t packed_mask_tile_size_;
  const uint32_t num_tiles_;
  const uint32_t num_elems_per_block_;
  const uint32_t num_elems_before_block_;

  constexpr static uint16_t SMALLEST_MASK =
      kernel_utils::UB_ALIGNMENT / sizeof(PackedMaskT);
  constexpr static uint16_t IN_ELEMS_PER_MASK_ELEM = sizeof(PackedMaskT) * 8;
};

/**
 * @brief Run the `compress` kernel. Support `fp16/half` and `float32` input
 * dtypes.
 *
 * @tparam T Data type of the input vector.
 * @param [in] in Pointer to input vector.
 * @param [in] mask Pointer to mask vector.
 * @param [in] out Pointer to output vector.
 * @param [in] upper Pointer to an upper-triangular matrix filled
 * with ones of size `scan_tile_size` x `scan_tile_size`.
 * @param [in] workspace Pointer to workspace.
 * @param [in] in_size Length of the input vector.
 * @param [in] scan_tile_size Size of the tile processed in a single
 * iteration of the scan kernel.
 * @param [in] compress_tile_size Size of the tile processed in a single
 * iteration of the compress kernel.
 */
template <typename InputT>
__aicore__ inline void _run_compress(GM_ADDR in, GM_ADDR mask, GM_ADDR out,
                                     GM_ADDR upper, GM_ADDR workspace,
                                     uint32_t in_size, uint16_t scan_tile_size,
                                     uint32_t compress_tile_size) {
  using OutputT = kernel_utils::cube_unit::CubeOutType_t<InputT>;

  const uint32_t scan_res_size =
      scalar::AlignUp(in_size * sizeof(OutputT), GM_ALIGNMENT);

  GM_ADDR const scan_res = workspace;
  GM_ADDR const scan_workspace = scan_res + scan_res_size;

  run_scan_multi_core_kernel<int8_t, true>(
      mask, upper, scan_res, scan_workspace, in_size, scan_tile_size);

  if ASCEND_IS_AIV {
    SyncAll<true /*isAIVOnly*/>();

    KernelCompress<InputT> op(in_size, compress_tile_size);
    op.Init(in, mask, scan_res, out);
    op.Process();
  }
}
