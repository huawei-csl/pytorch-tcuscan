/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_scan_multi_core.cpp
 * @brief Kernel implementing a multi-core inclusive scan using cube-vector
 * approach.
 */

#include "kernel_utils.h"
#include "kernels/constants.h"
#include "kernels/kernel_scan_multi_core.h"
#include "lib/matmul_intf.h"
#include "tiling/tiling_scan_multi_core.h"

__aicore__ inline void CopyTiling(MultiCoreScanTiling *tiling,
                                  GM_ADDR tilingGM) {
  uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
  auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

  for (uint32_t i = 0; i < sizeof(MultiCoreScanTiling) / sizeof(uint32_t);
       i++, ptr++) {
    *ptr = *(tiling32 + i);
  }
}

template <typename InputT>
__aicore__ inline void _run_scan_multi_core(GM_ADDR input_vec,
                                            GM_ADDR output_vec,
                                            GM_ADDR workspace,
                                            GM_ADDR tilingGm) {
  using OutputT = kernel_utils::cube_unit::CubeOutType_t<InputT>;

  MultiCoreScanTiling tiling;
  CopyTiling(&tiling, tilingGm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t matmul_size = tiling.matmul_size;
  constexpr bool IsInclusive = true;

  GM_ADDR const usrWorkspace = AscendC::GetUserWorkspace(workspace);
  GM_ADDR const lower = load_tril_matrix<InputT>(matmul_size);

  // We consider the L2 cache maxed when the scan takes up around 50% of the
  // L2 total cache size -> maybe available L2 size can be a tiling parameter
  constexpr uint32_t available_l2_size = L2_SIZE / 2;
  const uint32_t fitting_len = scalar::AlignUp(
      available_l2_size / (sizeof(InputT) + sizeof(OutputT)), GM_ALIGNMENT);

  uint32_t remaining_len = vec_len;
  uint32_t offset = 0;
  uint32_t used_size =
      mc_scan::get_workspace_size<InputT, OutputT, IsInclusive>(remaining_len,
                                                                matmul_size) +
      vec_len * sizeof(InputT) + vec_len * sizeof(OutputT);

  OutputT starting_value = 0;

  while (used_size > available_l2_size && fitting_len < remaining_len) {
    run_scan_multi_core_kernel<InputT, IsInclusive>(
        input_vec + offset * sizeof(InputT), lower,
        output_vec + offset * sizeof(OutputT), workspace, fitting_len,
        matmul_size, starting_value);
    SyncAll<false>();

    remaining_len -= fitting_len;
    used_size = mc_scan::get_workspace_size<InputT, OutputT, IsInclusive>(
                    remaining_len, matmul_size) +
                vec_len * sizeof(InputT) + vec_len * sizeof(OutputT);
    starting_value = scalar::GetGMValue<OutputT>(
        output_vec + offset * sizeof(OutputT), fitting_len - 1, fitting_len);
    offset += fitting_len;
  }
  if (remaining_len) {
    run_scan_multi_core_kernel<InputT, IsInclusive>(
        input_vec + offset * sizeof(InputT), lower,
        output_vec + offset * sizeof(OutputT), workspace, remaining_len,
        matmul_size, starting_value);
  }
}

/**
 * @brief Run the multi core inclusive scan kernel with input dtype fp16
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void scan_multi_core_fp16(GM_ADDR input_vec,
                                                           GM_ADDR output_vec,
                                                           GM_ADDR workspace,
                                                           GM_ADDR tilingGm) {
  _run_scan_multi_core<half>(input_vec, output_vec, workspace, tilingGm);
}

/**
 * @brief Run the multi core inclusive scan kernel with input dtype int8
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void scan_multi_core_int8(GM_ADDR input_vec,
                                                           GM_ADDR output_vec,
                                                           GM_ADDR workspace,
                                                           GM_ADDR tilingGm) {
  _run_scan_multi_core<int8_t>(input_vec, output_vec, workspace, tilingGm);
}
