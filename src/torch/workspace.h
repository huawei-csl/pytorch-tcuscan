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
#include "../tiling/tiling_cube_reduce.h"
#include "../tiling/tiling_radix_sort.h"
#include "../tiling/tiling_scan_batch.h"
#include "../tiling/tiling_scan_multi_core.h"
#include "../tiling/tiling_scan_single_core.h"
#include "../tiling/tiling_seg_sum_single_core.h"
#include "../tiling/tiling_seg_sum_single_cube.h"
#include "../tiling/tiling_split.h"
#include "../tiling/tiling_tri_inv_cube_col_sweep.h"

namespace tcuscan::workspace {

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
 * *
 * @param [in] tiling Tiling parameters used in the kernel.
 * @param [in] num_blocks Number of blocks.
 * @return Size of the workspace in bytes.
 */
constexpr uint32_t get_workspace_size(const CompressTiling& tiling,
                                      uint32_t num_blocks) {
  const uint32_t block_counts_size = host_utils::AlignUp(
      num_blocks * sizeof(int32_t), host_utils::GM_ALIGNMENT);
  return block_counts_size;
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

namespace scan_batch {

/**
 * @brief Calculate the workspace size for batched scan.
 *
 * @tparam InputT Input data type.
 *
 * @param [in] tiling Tiling parameters used in the kernel.
 * @return Size of the workspace in bytes.
 */
template <typename InputT>
constexpr uint32_t get_workspace_size(const ScanBatchTiling& tiling) {
  using OutputT = host_utils::CubeOutType_t<InputT>;

  if (tiling.num_elems > tiling.matmul_size) {
    const uint32_t padded_input_len =
        host_utils::AlignUp(tiling.num_elems,
                            tiling.matmul_size * tiling.matmul_size) *
        tiling.batch_size;
    const uint32_t padded_input_size = padded_input_len * sizeof(InputT);
    const uint32_t padded_rowwise_size = padded_input_len * sizeof(OutputT);

    return padded_input_size + padded_rowwise_size;
  } else {
    const uint32_t padded_input_len =
        host_utils::AlignUp(tiling.num_elems, tiling.matmul_size) *
        host_utils::AlignUp(tiling.batch_size,
                            tiling.matmul_size * tiling.matmul_size);

    const uint32_t padded_input_size = padded_input_len * sizeof(InputT);
    const uint32_t padded_rowwise_size = padded_input_len * sizeof(OutputT);
    return padded_input_size + padded_rowwise_size;
  }
}

}  // namespace scan_batch

namespace split {

/**
 * @brief Calculate the workspace size for split.
 *
 * @param [in] num_blocks Number of blocks.
 * @return Size of the workspace in bytes.
 */
constexpr uint32_t get_workspace_size(size_t num_blocks) {
  const uint32_t red_res_size = host_utils::AlignUp(
      num_blocks * sizeof(int32_t), host_utils::GM_ALIGNMENT);

  return red_res_size;
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
  return workspace::split::get_workspace_size(tiling.num_blocks);
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
  const uint32_t split_ws_size = split::get_workspace_size(t.num_blocks);

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
  // if the input size is aligned, the workspace of the first split may be
  // smaller than the workspace of the second split; so we force it to be
  // unaligned to avoid memory access errors
  const uint32_t split_ws_size =
      workspace::split::get_workspace_size(num_blocks);

  const uint32_t total_size = mask_size + split_ws_size + indices_ws_size +
                              split_input_size + split_output_size;
  return total_size;
}

}  // namespace topk

namespace seg_sum {

/**
 * @brief Calculate the workspace size for `seg_sum_single_core` and
 * `seg_sum_single_cube`.
 *
 * @tparam InputT Element input data type
 * @tparam OutputT Element output data type
 * @param [in] input_elems Number of elements in the input vector.
 * @param [in] matmul_size Size of the matmul used in scan.
 * @return Size of the workspace in bytes.
 */
template <typename InputT, typename OutputT>
constexpr uint32_t get_workspace_size(size_t input_elems, size_t matmul_size) {
  const uint32_t padded_input_len =
      host_utils::AlignUp(input_elems, matmul_size * matmul_size);
  // Keep a copy of the input (padded) and the output.
  const uint32_t total_size =
      padded_input_len * (sizeof(InputT) + sizeof(OutputT));

  return total_size;
}

/**
 * @brief Calculate the workspace size for `seg_sum_single_core`.
 *
 * @tparam InputT Element input data type
 * @tparam OutputT Element output data type
 * @param [in] tiling Tiling parameters used in the kernel.
 * @return Size of the workspace in bytes.
 */
template <typename InputT, typename OutputT>
constexpr uint32_t get_workspace_size(const SegSumSingleCoreTiling& tiling) {
  return seg_sum::get_workspace_size<InputT, OutputT>(tiling.num_elems,
                                                      tiling.tile_len);
}

/**
 * @brief Calculate the workspace size for `seg_sum_single_cube`.
 *
 * @tparam InputT Element input data type
 * @tparam OutputT Element output data type
 * @param [in] tiling Tiling parameters used in the kernel.
 * @return Size of the workspace in bytes.
 */
template <typename InputT, typename OutputT>
constexpr uint32_t get_workspace_size(const SegSumSingleCubeTiling& tiling) {
  return seg_sum::get_workspace_size<InputT, OutputT>(tiling.num_elems,
                                                      tiling.tile_len);
}

}  // namespace seg_sum

/**
 * @brief Calculate the workspace size for `cube_reduce`.
 *
 * @tparam T Input data type.
 *
 * @param [in] tiling Tiling parameters used in the kernel.
 * @return Size of the workspace in bytes.
 */
template <typename T>
constexpr uint32_t get_workspace_size(const CubeReduceTiling& tiling) {
  using OutputT = host_utils::CubeOutType_t<T>;

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t matmul_size = tiling.matmul_size;
  const uint32_t num_blocks = tiling.num_blocks;

  const uint32_t align_size = matmul_size * matmul_size;
  const uint32_t padded_input_len = host_utils::AlignUp(vec_len, align_size);

  const uint32_t padded_input_size = padded_input_len * sizeof(T);
  const uint32_t cube_reductions_size =
      num_blocks * matmul_size * matmul_size * sizeof(OutputT);

  return padded_input_size + cube_reductions_size;
}

/**
 * @brief Calculate the workspace size for `tri_inv_cube_col_sweep`.
 *
 * @tparam T Input data type.
 *
 * @param [in] tiling Tiling parameters used in the kernel.
 * @return Size of the workspace in bytes.
 */
template <typename T>
constexpr uint32_t get_workspace_size(const TriInvCubeColSweepTiling& tiling) {
  const uint32_t num_elems =
      tiling.num_blocks * tiling.matrix_size * tiling.matrix_size;
  const uint32_t workspace_size = num_elems * sizeof(T);

  return workspace_size;
}

}  // namespace tcuscan::workspace
