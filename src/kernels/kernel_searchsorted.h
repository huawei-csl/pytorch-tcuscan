#pragma once
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * @file kernel_searchsorted.h
 * @brief Kernel implementing searchsorted (lower_bound binary search) into a
 * sorted array.
 */

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Reproduces `torch.searchsorted(sorted, values, side='left')`, a.k.a.
 * lower_bound: for each needle `values[i]`, returns the first index `j` such
 * that `sorted[j] >= values[i]` (equivalently, the count of elements
 * `< values[i]`).
 *
 * The haystack `sorted` is searched directly in global memory through scalar
 * reads (it is typically far too large to stage in the Unified Buffer), so each
 * needle costs \f$O(\log(\textit{data\_len}))\f$ GM reads.
 *
 * The needles are independent, so they are distributed across the vector cores
 * in `BLOCK`-sized (32-byte-aligned) output tiles: each core searches a
 * contiguous run of needles and writes them with one `CopyVecToGm`.
 *
 * @tparam T Data type of the sorted array and needle values. Only `int32_t` is
 * currently instantiated.
 */
template <typename T = int32_t>
class KernelSearchsorted {
  constexpr static uint32_t BUFFER_NUM = 2;
  // Needles are handed out in cache-line-aligned blocks so that the per-core
  // output writes land on disjoint 32-byte GM granules (no false sharing).
  constexpr static uint32_t BLOCK = UB_ALIGNMENT / sizeof(T);

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] data_len Length of the sorted haystack array.
   * @param [in] num_values Number of needle values to search for.
   */
  __aicore__ inline KernelSearchsorted(uint32_t data_len, uint32_t num_values)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        data_len_(data_len),
        num_values_(num_values),
        num_blocks_(scalar::CeilDiv(num_values_, BLOCK)),
        max_num_blocks_per_core_(scalar::CeilDiv(num_blocks_, vec_core_num_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] sorted_in Pointer to the sorted haystack in global memory.
   * @param [in] values_in Pointer to the needle values in global memory.
   * @param [in] out Pointer to the output indices in global memory.
   */
  __aicore__ inline void Init(GM_ADDR sorted_in, GM_ADDR values_in,
                              GM_ADDR out) {
    sorted_in_ = sorted_in;
    values_in_ = values_in;
    global_out_.SetGlobalBuffer((__gm__ T*)out, num_values_);

    pipe_.InitBuffer(out_q_, BUFFER_NUM, BLOCK * sizeof(T));
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    uint32_t block_idx = GetBlockIdx() * max_num_blocks_per_core_;
    const uint32_t num_blocks_to_process =
        scalar::GetWorkDistribution(num_values_, BLOCK, vec_core_num_);

    for (uint32_t k = 0; k < num_blocks_to_process; ++k) {
      const uint32_t needle_start = block_idx * BLOCK;
      const uint32_t block_len =
          scalar::Min<uint32_t>(BLOCK, num_values_ - needle_start);

      LocalTensor<T> out_lt = out_q_.template AllocTensor<T>();
      for (uint32_t j = 0; j < block_len; ++j) {
        const T value =
            scalar::GetGMValue<T>(values_in_, needle_start + j, num_values_);
        out_lt.SetValue(j, static_cast<T>(LowerBound(value)));
      }
      out_q_.template EnQue<T>(out_lt);
      copy::CopyVecToGm<T>(global_out_[needle_start], out_q_, block_len);

      block_idx++;
    }
  }

 private:
  /**
   * @brief Returns the lower_bound of @p value in the sorted haystack: the
   * first index `j` with `sorted[j] >= value`.
   */
  __aicore__ inline uint32_t LowerBound(T value) {
    return scalar::LowerBoundGM<T>(sorted_in_, data_len_, value);
  }

  TPipe pipe_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> out_q_;
  GlobalTensor<T> global_out_;

  GM_ADDR sorted_in_;
  GM_ADDR values_in_;

  const uint32_t vec_core_num_;
  const uint32_t data_len_;
  const uint32_t num_values_;
  const uint32_t num_blocks_;
  const uint32_t max_num_blocks_per_core_;
};

/**
 * @brief Run the `searchsorted` kernel.
 *
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 * @tparam T Data type of the sorted array and needle values.
 *
 * @param [in] sorted_in Pointer to the sorted haystack.
 * @param [in] values_in Pointer to the needle values.
 * @param [in] out Pointer to the output indices.
 * @param [in] data_len Length of the sorted haystack.
 * @param [in] num_values Number of needle values.
 */
template <bool ForceMixMode = true, typename T>
__aicore__ inline void run_searchsorted(GM_ADDR sorted_in, GM_ADDR values_in,
                                        GM_ADDR out, uint32_t data_len,
                                        uint32_t num_values) {
  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }
  if ASCEND_IS_AIV {
    KernelSearchsorted<T> op(data_len, num_values);
    op.Init(sorted_in, values_in, out);
    op.Process();
  }
}

}  // namespace tcuscan
