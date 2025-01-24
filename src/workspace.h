/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file workspace.h
 * @brief Workspace size calculcations.
 */
#pragma once

#include <cstdint>

#include "host_utils.h"
#include "tiling/tiling_scan_multi_core.h"
#include "tiling/tiling_scan_single_core.h"

namespace workspace {

namespace seg_scan {
template <typename InputVecT, typename OutputVecT, typename FlagVecT,
          typename FlagOutputVecT>
constexpr uint32_t GetWorkspaceSize(uint32_t vec_len, uint32_t matmul_size) {
  const uint32_t padded_vec_len =
      host_utils::AlignUp(vec_len, matmul_size * matmul_size);
  const uint32_t padded_input_size = padded_vec_len * sizeof(InputVecT);
  const uint32_t padded_input_rowwise_size =
      padded_vec_len * sizeof(OutputVecT);

  const uint32_t padded_flag_size = padded_vec_len * sizeof(FlagVecT);
  const uint32_t padded_rowwise_flag_size =
      padded_vec_len * sizeof(FlagOutputVecT);

  const uint32_t total_size = padded_input_size + padded_input_rowwise_size +
                              padded_flag_size + padded_rowwise_flag_size;
  return total_size;
}

}  // namespace seg_scan

namespace mc_scan {

/**
 * @brief Calculate the workspace size for multi core scan.
 *
 * @tparam InputT Input data type.
 * @tparam OutputT Output data type.
 *
 * @param [in] input_elems Number of elements in the input vector.
 * @param [in] matmul_size Size of the matmul used in scan.
 * @param [in] num_blocks Number of blocks.
 * @return Size of the workspace in bytes.
 */
template <typename InputT, typename OutputT, bool IsInclusive = true>
constexpr uint32_t GetWorkspaceSize(uint32_t input_elems, uint32_t matmul_size,
                                    uint32_t num_blocks) {
  const uint32_t align_size = matmul_size * matmul_size;
  const uint32_t padded_input_len =
      host_utils::AlignUp(input_elems, align_size);

  const uint32_t padded_input_size = padded_input_len * sizeof(InputT);
  const uint32_t padded_rowwise_size = padded_input_len * sizeof(OutputT);

  constexpr uint32_t NUM_AIV_PER_AI_CORE = 2;
  const uint32_t sums_len = num_blocks * NUM_AIV_PER_AI_CORE;
  const uint32_t sums_size =
      host_utils::AlignUp(sums_len * sizeof(OutputT), host_utils::GM_ALIGNMENT);

  if (padded_input_len == input_elems) {
    if constexpr (IsInclusive)
      return sums_size;
    else
      return padded_rowwise_size + sums_size;
  } else
    return padded_input_size + padded_rowwise_size + sums_size;
}
}  // namespace mc_scan

namespace compress {

/**
 * @brief Calculate the workspace size for compress.
 *
 * @tparam InputT Input data type.
 *
 * @param [in] input_elems Number of elements in the input vector.
 * @param [in] scan_tile_size Size of the matmul used in scan.
 * @param [in] num_blocks Number of blocks.
 * @return Size of the workspace in bytes.
 */
constexpr uint32_t GetWorkspaceSize(size_t input_elems, size_t scan_tile_size,
                                    size_t num_blocks) {
  const uint32_t scan_res_size = host_utils::AlignUp(
      input_elems * sizeof(int32_t), host_utils::GM_ALIGNMENT);
  const uint32_t scan_ws_size = mc_scan::GetWorkspaceSize<int8_t, int32_t>(
      input_elems, scan_tile_size, num_blocks);
  return scan_res_size + scan_ws_size;
};

}  // namespace compress

namespace sc_scan {

/**
 * @brief Calculate the workspace size for single core scan.
 *
 * @tparam InputT Input data type.
 *
 * @param [in] tiling Tiling parameters used in the kernel.
 * @return Size of the workspace in bytes.
 */
template <typename InputT>
constexpr uint32_t GetWorkspaceSize(const SingleCoreScanTiling &tiling) {
  const uint32_t padded_input_len = host_utils::AlignUp(
      tiling.num_elems, tiling.matmul_size * tiling.matmul_size);
  const uint32_t padded_input_size = padded_input_len * sizeof(InputT);
  const uint32_t padded_rowwise_size = padded_input_len * sizeof(int32_t);

  const uint32_t total_size = padded_input_size + padded_rowwise_size;
  return total_size;
}
}  // namespace sc_scan

}  // namespace workspace
