/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file radix_sort.cpp
 * @brief Kernel implementing a radix sort operation for fp16 / int16.
 */

#include "kernels/constants.h"
#include "kernels/kernel_arithmetic_progression.h"
#include "kernels/kernel_radix_enc.h"
#include "kernels/kernel_single_radix.h"
#include "kernels/kernel_split.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_radix_sort.h"

__aicore__ inline void _radix_sort_iter(GM_ADDR in, GM_ADDR radices_ws,
                                        GM_ADDR out, GM_ADDR indices_in,
                                        GM_ADDR indices_out, GM_ADDR lower,
                                        GM_ADDR split_ws, uint32_t in_len,
                                        uint16_t cube_tile_size,
                                        uint32_t vec_tile_size,
                                        uint16_t bit_idx, bool zeros_first) {
  run_single_radix<false>(in, radices_ws, in_len, vec_tile_size, bit_idx);
  SyncAll<false /*isAIVOnly*/>();
  run_split_ind_uint16(in, radices_ws, indices_in, out, indices_out, lower,
                       split_ws, in_len, cube_tile_size, vec_tile_size,
                       zeros_first);
}

/**
 * @brief Run the `radix_sort_fp16` kernel.
 *
 * @param [in] in Pointer to input vector.
 * @param [in] out Pointer to output vector.
 * @param [in] indices Pointer to output indices vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void radix_sort_fp16(GM_ADDR in, GM_ADDR out,
                                                      GM_ADDR indices,
                                                      GM_ADDR workspace,
                                                      GM_ADDR tiling) {
  /// Indicates the order of sorting
  constexpr bool descending = false;
  RadixSortTiling tiling_data;
  tiling::GetTilingData(&tiling_data, tiling);

  GM_ADDR const usrWorkspace = AscendC::GetUserWorkspace(workspace);
  GM_ADDR const lower = load_tril_matrix<int8_t>(tiling_data.matmul_size);

  // Arrays in workspace have to be aligned to their data type size. Therefore
  // we align the size of the radices array to 4 bytes, so that the
  // workspace that comes after starts at a valid address for int32_t.
  const uint32_t radices_size =
      scalar::AlignUp(tiling_data.num_elems * sizeof(uint8_t), sizeof(int32_t));
  const uint32_t split_ws_size = split::get_workspace_size<half>(
      tiling_data.num_elems, tiling_data.matmul_size);
  const uint32_t output_size = tiling_data.num_elems * sizeof(half);

  GM_ADDR const radices = usrWorkspace;
  GM_ADDR const split_workspace = radices + radices_size;
  GM_ADDR const out2 = split_workspace + split_ws_size;
  GM_ADDR const indices_ws = out2 + output_size;

  run_arithmetic_progression<int32_t, 0, 1, false /* ForceMixMode */>(
      indices, tiling_data.num_elems, tiling_data.vec_tile_size);
  SyncAll<false /*isAIVOnly*/>();

  run_radix_encode<false>(in, out, tiling_data.num_elems,
                          tiling_data.vec_tile_size);

  bool are_zeros_first = !descending;

  GM_ADDR iter_in = out;
  GM_ADDR iter_out = out2;
  GM_ADDR indices_in = indices;
  GM_ADDR indices_out = indices_ws;
  for (int i = 0; i < 16; i++) {
    SyncAll<true /*isAIVOnly*/>();

    const bool is_last_iter = i == 15;
    if (is_last_iter) {
      are_zeros_first = !are_zeros_first;
    }
    _radix_sort_iter(iter_in, radices, iter_out, indices_in, indices_out, lower,
                     split_workspace, tiling_data.num_elems,
                     tiling_data.matmul_size, tiling_data.vec_tile_size, i,
                     are_zeros_first);
    GM_ADDR tmp = iter_in;
    iter_in = iter_out;
    iter_out = tmp;

    tmp = indices_in;
    indices_in = indices_out;
    indices_out = tmp;
  }

  SyncAll<true /*isAIVOnly*/>();

  // Encode again to obtain the initial values (enc(enc(x)) = x)
  run_radix_encode<false>(out, out, tiling_data.num_elems,
                          tiling_data.vec_tile_size);
}

/**
 * @brief Run the `radix_sort_int16` kernel.
 *
 * @param [in] in Pointer to input vector.
 * @param [in] out Pointer to output vector.
 * @param [in] indices Pointer to output indices vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void radix_sort_int16(GM_ADDR in, GM_ADDR out,
                                                       GM_ADDR indices,
                                                       GM_ADDR workspace,
                                                       GM_ADDR tiling) {
  /// Indicates the order of sorting
  constexpr bool descending = false;
  RadixSortTiling tiling_data;
  tiling::GetTilingData(&tiling_data, tiling);

  GM_ADDR const lower = load_tril_matrix<int8_t>(tiling_data.matmul_size);

  // Arrays in workspace have to be aligned to their data type size. Therefore
  // we align the size of the radices array to 4 bytes, so that the
  // workspace that comes after starts at a valid address for int32_t.
  const uint32_t radices_size =
      scalar::AlignUp(tiling_data.num_elems * sizeof(uint8_t), sizeof(int32_t));
  const uint32_t split_ws_size = split::get_workspace_size<int16_t>(
      tiling_data.num_elems, tiling_data.matmul_size);
  const uint32_t output_size = tiling_data.num_elems * sizeof(uint16_t);

  GM_ADDR const radices = workspace;
  GM_ADDR const split_workspace = radices + radices_size;
  GM_ADDR const out2 = split_workspace + split_ws_size;
  GM_ADDR const indices_ws = out2 + output_size;

  run_arithmetic_progression<int32_t, 0, 1, false /* ForceMixMode */>(
      indices, tiling_data.num_elems, tiling_data.vec_tile_size);
  SyncAll<false /*isAIVOnly*/>();

  bool are_zeros_first = !descending;
  _radix_sort_iter(in, radices, out2, indices, indices_ws, lower,
                   split_workspace, tiling_data.num_elems,
                   tiling_data.matmul_size, tiling_data.vec_tile_size, 0,
                   are_zeros_first);

  SyncAll<true /*isAIVOnly*/>();

  GM_ADDR iter_in = out2;
  GM_ADDR iter_out = out;
  GM_ADDR indices_in = indices_ws;
  GM_ADDR indices_out = indices;
  for (int i = 1; i < 16; i++) {
    const bool is_last_iter = i == 15;
    if (is_last_iter) {
      are_zeros_first = !are_zeros_first;
    }
    _radix_sort_iter(iter_in, radices, iter_out, indices_in, indices_out, lower,
                     split_workspace, tiling_data.num_elems,
                     tiling_data.matmul_size, tiling_data.vec_tile_size, i,
                     are_zeros_first);

    SyncAll<true /*isAIVOnly*/>();

    GM_ADDR tmp = iter_in;
    iter_in = iter_out;
    iter_out = tmp;

    tmp = indices_in;
    indices_in = indices_out;
    indices_out = tmp;
  }
}
