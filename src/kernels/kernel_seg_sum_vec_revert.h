/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file kernel_seg_sum_vec_revert.h
 * @brief Kernel implementing a revert speculation of segmented sum cube
 * operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;
using namespace kernel_utils;

/**
 * @brief Corrects the (speculative block scan) output of the `KernelRowScan` so
 * that the segmented sum of a vector is returned.
 *
 *
 * ### Example:
 * Given, vec_in = [1,2,3,4,5,6,7,8,9,10] and segm_ind_in = [0, 4, 7]
 * returns vec_out = (1+2+3+4, 5+6, 7+8+9+10) = (10, 11, 34)
 *
 * @tparam T
 * @tparam SyncBefore If true, the `KernelSegSumVecRevert` (this kernel) waits
 * for the cube unit to send a synchronization signal after each matrix tile is
 * ready. Matrix tile has length `tile_len * tile_len`.
 */
template <typename T = float, bool SyncBefore = false>
class KernelSegSumVecRevert {
  constexpr static uint32_t BUFFER_NUM = 2;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Input vector length.
   * @param [in] num_segments Number of segments.
   * @param [in] tile_len Tile length.
   */
  __aicore__ inline KernelSegSumVecRevert(uint32_t vec_len,
                                          uint32_t num_segments,
                                          uint32_t tile_len)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        vec_len_(vec_len),
        num_segments_(num_segments),
        tile_len_(tile_len),
        matrix_tile_len_(tile_len * tile_len),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_len_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_in Pointer to the input vector in global memory.
   * @param [in] segm_ind_in Pointer to the segment indices vector in global
   * memory.
   * @param [in] vec_out Pointer to the output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR segm_ind_in,
                              GM_ADDR vec_out) {
    global_in_.SetGlobalBuffer((__gm__ T *)vec_in, vec_len_);
    global_segm_in_.SetGlobalBuffer((__gm__ uint32_t *)segm_ind_in,
                                    num_segments_);
    global_out_.SetGlobalBuffer((__gm__ T *)vec_out, vec_len_);

    pipe_.InitBuffer(in_q_, BUFFER_NUM, tile_len_ * sizeof(T));
    pipe_.InitBuffer(segm_q_, BUFFER_NUM, tile_len_ * sizeof(uint32_t));
    pipe_.InitBuffer(out_q_, BUFFER_NUM, num_segments_ * sizeof(T));
  }

  /**
   * @brief Returns the next tile length, given the global memory offset. The
   * returned tile length equals typically to `tile_len_`, expect the last
   * iteration where the tile length is smaller than `tile_len_`.
   *
   * @param global_offset Global memory offset
   * @param length Total vector length
   * @return Length of "next" tile, given the current global memory offset.
   */
  __aicore__ inline uint32_t NextTileLen(uint32_t global_offset,
                                         uint32_t length) {
    const bool full_tile = global_offset + tile_len_ <= length;
    const uint32_t num_elems_to_process =
        full_tile ? tile_len_ : length - global_offset;

    return num_elems_to_process;
  }

  /**
   * @brief Returns a local tensor that contains the next segment indices tile
   * (of length up to `tile_len_`)
   *
   * @return Segments index vector of tile length at most `tile_len_`.
   */
  __aicore__ inline LocalTensor<uint32_t> LoadNextSegmentTile() {
    const uint32_t next_tile_len = NextTileLen(segments_offset_, num_segments_);
    copy::CopyGmToVec(segm_q_, global_segm_in_[segments_offset_],
                      next_tile_len);

    return segm_q_.DeQue<uint32_t>();
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    LocalTensor<uint32_t> segm_ind_lt = LoadNextSegmentTile();
    LocalTensor<T> vec_out_lt = out_q_.AllocTensor<T>();

    T accumulation = 0;
    uint32_t offset = 0;
    uint32_t out_lt_offset = 0;
    uint32_t segm_idx = 1;
    uint32_t segm_end = segm_ind_lt.GetValue(segm_idx);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_; tile_idx++) {
      if constexpr (SyncBefore) {
        if ((tile_idx * tile_len_) % matrix_tile_len_ == 0) {
          sync::SyncGroup<sync::GroupSyncDirection::FULL>();
        }
      }

      const uint32_t num_elems_to_process = NextTileLen(offset, vec_len_);
      copy::CopyGmToVec(in_q_, global_in_[offset], num_elems_to_process);
      LocalTensor<T> vec_in_lt = in_q_.DeQue<T>();

      while (segm_end < offset + num_elems_to_process) {
        // Last segment value. Zero if lies on tile boundary.
        const T delta = (segm_end == offset)
                            ? 0
                            : vec_in_lt.GetValue(segm_end - offset - 1);
        vec_out_lt.SetValue(out_lt_offset + segm_idx - 1, accumulation + delta);

        accumulation = -delta;
        segm_idx++;

        if (segments_offset_ + segm_idx >= num_segments_) {
          // Run out-of-segments, set the segment end to maximum
          // vector length.
          segm_end = vec_len_;
          out_lt_offset = num_segments_;
        } else if (segm_idx >= tile_len_) {
          // Run out-of-segments in the segment tiles. Just load the
          // next segment tile and update counters
          segm_q_.FreeTensor<uint32_t>(segm_ind_lt);
          segments_offset_ += tile_len_;
          segm_ind_lt = LoadNextSegmentTile();

          // "Local-coordinate" segment index must be zeroed and keep
          // track of the output vector location.
          segm_idx = 0;
          out_lt_offset += tile_len_;
          segm_end = segm_ind_lt.GetValue(0);
        } else {
          // Read next segment
          segm_end = segm_ind_lt.GetValue(segm_idx);
        }
      }

      accumulation += vec_in_lt.GetValue(num_elems_to_process - 1);

      in_q_.FreeTensor<T>(vec_in_lt);
      offset += num_elems_to_process;
    }

    segm_q_.FreeTensor<uint32_t>(segm_ind_lt);

    // The last segment contains the remaining accumulation values.
    vec_out_lt.SetValue(out_lt_offset - 1, accumulation);

    out_q_.EnQue<T>(vec_out_lt);
    copy::CopyVecToGm(global_out_, out_q_, num_segments_);
  }

 private:
  TPipe pipe_;

  TQue<QuePosition::VECIN, BUFFER_NUM> in_q_;
  TQue<QuePosition::VECIN, BUFFER_NUM> segm_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> out_q_;

  GlobalTensor<T> global_in_;
  GlobalTensor<uint32_t> global_segm_in_;
  GlobalTensor<T> global_out_;

  const uint32_t vec_core_num_;
  const uint32_t vec_len_;
  const uint32_t num_segments_;
  const uint32_t tile_len_;
  const uint32_t matrix_tile_len_;
  const uint32_t num_tiles_;

  uint32_t segments_offset_ = 0;
};
