/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file workspace.h
 * @brief Workspace size calculcations.
 */
#pragma once

#include <cstdint>

#include "../host_utils.h"
#include "../tiling/tiling_compress.h"
#include "../tiling/tiling_copy.h"
#include "../tiling/tiling_radix_sort.h"
#include "../tiling/tiling_scan_multi_core.h"
#include "../tiling/tiling_scan_single_core.h"
#include "../tiling/tiling_seg_sum_single_core.h"
#include "../tiling/tiling_split.h"

namespace workspace {

namespace seg_scan {
/**
 * @brief
 *
 * @tparam InputVecT Input data type.
 * @tparam FlagVecT Input flag type.
 * @param vec_len Input vector length.
 * @param matmul_size Matrix multiplication tile size.
 * @return
 */
template <typename InputVecT, typename FlagVecT>
constexpr uint32_t get_workspace_size(uint32_t vec_len, uint32_t matmul_size) {
  using OutputVecT = host_utils::CubeOutType_t<InputVecT>;
  using FlagOutputVecT = host_utils::CubeOutType_t<FlagVecT>;

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
 *
 * @param [in] input_elems Number of elements in the input vector.
 * @param [in] matmul_size Size of the matmul used in scan.
 * @param [in] num_blocks Number of blocks.
 * @return Size of the workspace in bytes.
 */
template <typename InputT, bool IsInclusive = true>
constexpr uint32_t get_workspace_size(uint32_t input_elems,
                                      uint32_t matmul_size,
                                      uint32_t num_blocks) {
  using OutputT = host_utils::CubeOutType_t<InputT>;

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
 * @param [in] tiling Tiling parameters used in the kernel.
 * @param [in] num_blocks Number of blocks.
 * @return Size of the workspace in bytes.
 */
template <typename InputT>
constexpr uint32_t get_workspace_size(const CompressTiling& tiling,
                                      uint32_t num_blocks) {
  const uint32_t scan_res_size = host_utils::AlignUp(
      tiling.size * sizeof(int32_t), host_utils::GM_ALIGNMENT);
  const uint32_t scan_ws_size = mc_scan::get_workspace_size<int8_t>(
      tiling.size, tiling.scan_tile_size, num_blocks);
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
constexpr uint32_t get_workspace_size(const SingleCoreScanTiling& tiling) {
  using OutputT = host_utils::CubeOutType_t<InputT>;

  const uint32_t total_size =
      2 * tiling.matmul_size * tiling.matmul_size * sizeof(OutputT);
  return total_size;
}
}  // namespace sc_scan

namespace split {

/**
 * @brief Calculate the workspace size for split.
 *
 * @param [in] input_elems Number of elements in the input vector.
 * @param [in] matmul_size Size of the matmul used in scan.
 * @param [in] num_blocks Number of blocks.
 * @return Size of the workspace in bytes.
 */
constexpr uint32_t get_workspace_size(size_t input_elems, size_t matmul_size,
                                      size_t num_blocks) {
  const uint32_t scan_res_size = host_utils::AlignUp(
      input_elems * sizeof(int32_t), host_utils::GM_ALIGNMENT);
  const uint32_t scan_ws_size =
      mc_scan::get_workspace_size<int8_t>(input_elems, matmul_size, num_blocks);

  return scan_res_size + scan_ws_size;
}

/**
 * @brief Calculate the workspace size for split.
 *
 * @tparam InputT Input data type.
 *
 * @param [in] tiling Tiling parameters used in the kernel.
 * @return Size of the workspace in bytes.
 */
constexpr uint32_t get_workspace_size(const SplitTiling& tiling) {
  return workspace::split::get_workspace_size(
      tiling.num_elems, tiling.scan_matmul_size, tiling.num_blocks);
}
}  // namespace split

namespace radix_sort {

/**
 * @brief Calculate the workspace size for radix sort.
 *
 * @tparam InputT Input data type.
 *
 * @param [in] t Radix sort tiling struct.
 * @return Size of the workspace in bytes.
 */
template <typename InputT>
uint32_t get_workspace_size(const RadixSortTiling& t) {
  const uint32_t tmp_output_size = t.num_elems * sizeof(InputT);
  // Arrays in workspace have to be aligned to their data type size. Therefore
  // we align the size of the radices array to 4 bytes, so that the
  // indices array that comes after starts at a valid address for int32_t.
  const uint32_t radices_size =
      host_utils::AlignUp(t.num_elems * sizeof(uint8_t), sizeof(int32_t));
  const uint32_t indices_size = t.num_elems * sizeof(int32_t);
  const uint32_t split_ws_size =
      split::get_workspace_size(t.num_elems, t.matmul_size, t.num_blocks);

  const uint32_t total_size =
      tmp_output_size + radices_size + indices_size + split_ws_size;
  return total_size;
}
}  // namespace radix_sort

namespace topk {

/**
 * @brief Calculate the workspace size for topk.
 *
 * @tparam InputT Input data type.
 *
 * @param [in] input_elems Number of elements in the input vector.
 * @param [in] matmul_size Size of the matmul used in scan.
 * @param [in] num_blocks Number of blocks.
 * @return Size of the workspace in bytes.
 */
template <typename InputT>
constexpr uint32_t get_workspace_size(size_t input_elems, size_t matmul_size,
                                      size_t num_blocks) {
  const uint32_t mask_size = host_utils::AlignUp(input_elems * sizeof(uint8_t),
                                                 host_utils::GM_ALIGNMENT);
  const uint32_t split_input_size = host_utils::AlignUp(
      input_elems * sizeof(int16_t), host_utils::GM_ALIGNMENT);
  const uint32_t split_output_size = split_input_size;
  const uint32_t indices_ws_size = host_utils::AlignUp(
      input_elems * sizeof(uint32_t) * 2, host_utils::GM_ALIGNMENT);
  const bool is_input_size_aligned =
      input_elems % (matmul_size * matmul_size) == 0;
  // if the input size is aligned, the workspace of the first split may be
  // smaller than the workspace of the second split; so we force it to be
  // unaligned to avoid memory access errors
  const uint32_t split_ws_size =
      is_input_size_aligned ? workspace::split::get_workspace_size(
                                  input_elems + 1, matmul_size, num_blocks)
                            : workspace::split::get_workspace_size(
                                  input_elems, matmul_size, num_blocks);

  const uint32_t total_size = mask_size + split_ws_size + indices_ws_size +
                              split_input_size + split_output_size;
  return total_size;
}

}  // namespace topk

namespace seg_sum {

/**
 * @brief Calculate the workspace size for `seg_sum_single_core`.
 *
 * @param [in] input_elems Number of elements in the input vector.
 * @param [in] matmul_size Size of the matmul used in scan.
 * @return Size of the workspace in bytes.
 */
constexpr uint32_t get_workspace_size(size_t input_elems, size_t matmul_size) {
  const uint32_t padded_input_len =
      host_utils::AlignUp(input_elems, matmul_size * matmul_size);
  const uint32_t total_size = padded_input_len * sizeof(float);

  return total_size;
}

/**
 * @brief Calculate the workspace size for `seg_sum_single_core`.
 *
 * @param [in] tiling Tiling parameters used in the kernel.
 * @return Size of the workspace in bytes.
 */
constexpr uint32_t get_workspace_size(const SegSumSingleCoreTiling& tiling) {
  return seg_sum::get_workspace_size(tiling.num_elems, tiling.tile_len);
}

}  // namespace seg_sum

}  // namespace workspace
