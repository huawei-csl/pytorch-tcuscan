/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_scan_multi_core.h
 * @brief Kernel implementing different variants of a scan operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "kernel_pad.h"
#include "kernel_reduce_tiles.h"
#include "kernel_row_scan.h"
#include "kernel_utils.h"

using namespace AscendC;
using namespace kernel_utils;

/**
 * @brief Finish the scan operation based on row-wise partial scans and reduced
 * tiles.
 *
 * This kernel takes outputs of `KernelRowScan` and `KernelReduceTiles` and
 * finishes calculating the entire scan of the vector.
 *
 * Because of the limited vector core's support for data types, the input data
 * must be `half`, `int16_t`, `float` or `int32_t`.
 *
 * @tparam T Data type of the input and output vectors.
 * @tparam IsInclusive Indicates whether the scan is inclusive or exclusive.
 */
template <typename T, bool IsInclusive = true>
class KernelCompleteRows {
  constexpr static int32_t MIN_VEC_SIZE = UB_ALIGNMENT / sizeof(T);

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] tile_width Size of the row used by `KernelRowScan`.
   * @param [in] tile_height Number of rows processed in a single iteration.
   * @param [in] vec_len Number of elements in an input vector.
   */
  __aicore__ inline KernelCompleteRows(uint32_t tile_width,
                                       uint32_t tile_height, uint32_t vec_len)
      : block_num_(GetBlockNum() * GetTaskRation()),
        tile_width_(tile_width),
        tile_height_(tile_height),
        tile_size_(tile_width_ * tile_height_),
        sums_len_(block_num_),
        vec_len_(vec_len),
        output_real_elems_(IsInclusive ? vec_len_ : vec_len_ - 1),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_size_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, block_num_)) {
    constexpr bool IS_DT_SUPPORTED =
        std::is_same_v<T, half> || std::is_same_v<T, int16_t> ||
        std::is_same_v<T, float> || std::is_same_v<T, int32_t>;
    static_assert(IS_DT_SUPPORTED, "Unsupported data type.");
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] input_rows Pointer to input vector in global memory.
   * @param [in] sums Pointer to vector with partial sums in global
   * memory.
   * @param [in] output Pointer to output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR input_rows, GM_ADDR sums,
                              GM_ADDR output) {
    global_input_rows_.SetGlobalBuffer((__gm__ T *)input_rows,
                                       AlignUp(vec_len_, tile_size_));
    global_sums_.SetGlobalBuffer((__gm__ T *)sums, sums_len_);
    // The very first element of the output where a zero value must go
    // in case of exclusive scan.
    global_output_first_elem_.SetGlobalBuffer((__gm__ T *)output, 1);
    global_output_.SetGlobalBuffer((__gm__ T *)output + global_shift_,
                                   output_real_elems_);

    pipe.InitBuffer(vec_tile_q_, 2, tile_size_ * sizeof(T));
    pipe.InitBuffer(vec_tile_out_q_, 2, tile_size_ * sizeof(T));

    pipe.InitBuffer(sums_q_, 1, AlignUp(block_num_, MIN_VEC_SIZE) * sizeof(T));

    pipe.InitBuffer(work_buf_, tile_size_ * sizeof(T));
  }

  /**
   * @brief Run the kernel - process all tiles.
   *
   * @param [in] starting_value Starting value added to all entries of the
   * output vector.
   */
  __aicore__ inline void Process(T starting_value = 0) {
    const T previous_sum = ScanReducedTiles() + starting_value;
    AddSumsToMatmulTiles(previous_sum);
    if constexpr (!IsInclusive) {
      if (GetBlockIdx() == 0) {
        global_output_first_elem_(0) = starting_value;
        DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE,
                                 DcciDst::CACHELINE_OUT>(
            global_output_first_elem_);
      }
    }
  }

 private:
  __aicore__ inline T ScanReducedTiles() {
    // Reduce the sums of all the previous tiles.
    copy::CopyGmToVec(sums_q_, global_sums_, sums_len_);
    LocalTensor<T> sums_lt = sums_q_.DeQue<T>();
    const T previous_sum = reduce::ReduceScalarAdd(sums_lt, GetBlockIdx());
    sums_q_.FreeTensor(sums_lt);
    return previous_sum;
  }

  __aicore__ inline void AddSumsToMatmulTiles(T previous_sum) {
    uint32_t global_offset =
        GetBlockIdx() * tile_size_ * max_num_tiles_per_block_;
    const uint32_t num_tiles_to_process =
        scalar::GetWorkDistribution(vec_len_, tile_size_, block_num_);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      copy::CopyGmToVec(vec_tile_q_, global_input_rows_[global_offset]);
      previous_sum = VectorAdds(previous_sum);

      const bool full_tile = global_offset + tile_size_ <= output_real_elems_;
      const uint32_t num_elems_to_write =
          full_tile ? tile_size_ : output_real_elems_ - global_offset;
      const LocalTensor<T> result_lt = work_buf_.Get<T>();
      copy::CopyVecToGm(global_output_[global_offset], vec_tile_out_q_,
                        result_lt, num_elems_to_write);

      global_offset += tile_size_;
    }
  }

  __aicore__ inline T VectorAdds(T initial_sum) {
    LocalTensor<T> vec_lt = vec_tile_q_.DeQue<T>();
    // Get 2 local tensors pointing to the same buffer: this is needed to
    // trick the compiler into considering operations on vec_buf1 and
    // vec_buf2 independent.
    const LocalTensor<T> vec_buf1 = work_buf_.Get<T>();
    const LocalTensor<T> vec_buf2 = work_buf_.Get<T>();
    DataCopy(vec_buf1, vec_lt, vec_lt.GetSize());
    vec_tile_q_.FreeTensor(vec_lt);

    sync::ScalarWaitForVec();

    uint32_t first_offset = 0;
    uint32_t second_offset = tile_width_;
    T first_sum = initial_sum;
    T second_sum = vec_buf1.GetValue(second_offset - 1) + first_sum;
    for (uint32_t i = 0; i < tile_height_; i += 2) {
      // The Adds instructions can be overlapped because they are
      // independent.
      Adds(vec_buf1[first_offset], vec_buf1[first_offset], first_sum,
           tile_width_);
      Adds(vec_buf2[second_offset], vec_buf2[second_offset], second_sum,
           tile_width_);
      first_offset += tile_width_ * 2;
      second_offset += tile_width_ * 2;
      first_sum = vec_buf2.GetValue(first_offset - 1);
      if (i + 2 < tile_height_)
        second_sum = vec_buf1.GetValue(second_offset - 1) + first_sum;
    }
    return first_sum;
  }

  TPipe pipe;

  TQue<QuePosition::VECIN, 2> vec_tile_q_;
  TQue<QuePosition::VECOUT, 2> vec_tile_out_q_;

  TQue<QuePosition::VECIN, 1> sums_q_;

  TBuf<QuePosition::VECCALC> work_buf_;

  GlobalTensor<T> global_input_rows_;
  GlobalTensor<T> global_sums_;
  GlobalTensor<T> global_output_;
  GlobalTensor<T> global_output_first_elem_;

  constexpr static uint16_t VEC_LOOP_UNROLL_ = 8;
  constexpr static uint32_t global_shift_ = IsInclusive ? 0 : 1;

  const uint32_t block_num_;
  const uint32_t tile_width_;
  const uint32_t tile_height_;
  const uint32_t tile_size_;
  const uint32_t sums_len_;

  const uint32_t vec_len_;
  const uint32_t output_real_elems_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;
};

namespace mc_scan {

/**
 * @brief Calculate the workspace size for multi core scan.
 *
 * @tparam InputT Input data type.
 * @tparam OutputT Output data type.
 *
 * @param [in] input_elems Number of elements in the input vector.
 * @param [in] matmul_size Size of the matmul used in scan.
 * @return Size of the workspace in bytes.
 */
template <typename InputT, typename OutputT, bool IsInclusive = true>
__aicore__ inline uint32_t get_workspace_size(uint32_t input_elems,
                                              uint32_t matmul_size) {
  const uint32_t align_size = matmul_size * matmul_size;
  const uint32_t padded_input_len = scalar::AlignUp(input_elems, align_size);

  const uint32_t padded_input_size = padded_input_len * sizeof(InputT);
  const uint32_t padded_rowwise_size = padded_input_len * sizeof(OutputT);
  const uint32_t sums_len = GetBlockNum() * GetTaskRation();
  const uint32_t sums_size =
      scalar::AlignUp(sums_len * sizeof(OutputT), GM_ALIGNMENT);

  if (padded_input_len == input_elems) {
    if constexpr (IsInclusive)
      return sums_size;
    else
      return padded_rowwise_size + sums_size;
  } else
    return padded_input_size + padded_rowwise_size + sums_size;
}

}  // namespace mc_scan

/**
 * @brief Run the multi core scan kernel.
 *
 * @tparam IsInclusive Indicates whether the scan is inclusive or exclusive.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] upper_triangular Pointer to an upper-triangular matrix filled
 * with ones of size \f$\textit{matmul_size} \times \textit{matmul_size}\f$.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] vec_len Number of elements to process.
 * @param [in] matmul_size Size of the matmul tile used in the kernel.
 * @param [in] starting_value Starting value added to all entries of the
 * output vector.
 */

template <typename InputT, bool IsInclusive = true>
__aicore__ inline void run_scan_multi_core_kernel(
    GM_ADDR input_vec, GM_ADDR upper_triangular, GM_ADDR output_vec,
    GM_ADDR workspace, uint32_t vec_len, uint32_t matmul_size,
    cube_unit::CubeOutType_t<InputT> starting_value = 0) {
  using OutputT = cube_unit::CubeOutType_t<InputT>;

  const uint32_t align_size = matmul_size * matmul_size;
  const uint32_t padded_vec_len_c = scalar::AlignUp(vec_len, align_size);
  const uint32_t padded_vec_len_v = scalar::AlignUp(vec_len, align_size / 2);

  const uint32_t padded_input_size = padded_vec_len_c * sizeof(InputT);
  const uint32_t padded_rowwise_size = padded_vec_len_c * sizeof(OutputT);

  if (vec_len % align_size || vec_len % UB_ALIGNMENT) {
    GM_ADDR const padded_input = workspace;
    GM_ADDR const padded_rowwise_scan = padded_input + padded_input_size;
    GM_ADDR const sums = padded_rowwise_scan + padded_rowwise_size;

    run_pad_kernel<InputT, false>(input_vec, padded_input, vec_len, align_size);

    sync::SyncAllCores();
    sync::SyncGroup<sync::GroupSyncDirection::FULL>();
    PipeBarrier<PIPE_ALL>();

    if ASCEND_IS_AIC {
      KernelRowScan<InputT> op_cube(matmul_size, matmul_size, padded_vec_len_c);
      op_cube.Init(padded_input, upper_triangular, padded_rowwise_scan);
      op_cube.Process();
      PipeBarrier<PIPE_ALL>();
    }

    if ASCEND_IS_AIV {
      KernelReduceTiles<InputT> op_reduce(matmul_size * matmul_size / 2,
                                          padded_vec_len_v);
      op_reduce.Init(padded_input, sums);
      op_reduce.Process();
    }

    SyncAll<false /*isAIVOnly*/>();

    if ASCEND_IS_AIV {
      KernelCompleteRows<OutputT, IsInclusive> op(matmul_size, matmul_size / 2,
                                                  vec_len);
      op.Init(padded_rowwise_scan, sums, output_vec);
      op.Process(starting_value);
    }
  } else {
    // If possible, we try to execute phase 2 in place, as it might
    // utilize L2 better.
    GM_ADDR const padded_rowwise_scan = workspace;

    GM_ADDR const intermediate_res =
        IsInclusive ? output_vec : padded_rowwise_scan;
    GM_ADDR const sums =
        IsInclusive ? workspace : padded_rowwise_scan + padded_rowwise_size;

    if ASCEND_IS_AIC {
      KernelRowScan<InputT> op_cube(matmul_size, matmul_size, vec_len);
      op_cube.Init(input_vec, upper_triangular, intermediate_res);
      op_cube.Process();
      PipeBarrier<PIPE_ALL>();
    }

    if ASCEND_IS_AIV {
      KernelReduceTiles<InputT> op_reduce(matmul_size * matmul_size / 2,
                                          vec_len);
      op_reduce.Init(input_vec, sums);
      op_reduce.Process();
    }

    SyncAll<false /*isAIVOnly*/>();

    if ASCEND_IS_AIV {
      KernelCompleteRows<OutputT, IsInclusive> op(matmul_size, matmul_size / 2,
                                                  vec_len);
      op.Init(intermediate_res, sums, output_vec);
      op.Process(starting_value);
    }
  }
}
