/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_scan_multi_core.cpp
 * @brief Kernel implementing a multi-core inclusive scan using cube-vector
 * approach.
 */

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

/**
 * @brief Run the multi core inclusive scan kernel.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void scan_multi_core(GM_ADDR input_vec,
                                                      GM_ADDR output_vec,
                                                      GM_ADDR workspace,
                                                      GM_ADDR tilingGm) {
  MultiCoreScanTiling tiling;
  CopyTiling(&tiling, tilingGm);

  // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha003/quickstart/quickstart/quickstart_18_0001.html?sub_id=%2Fzh%2FCANNCommunityEdition%2F80RC2alpha003%2Fdevguide%2Fopdevg%2Fascendcopdevg%2Fatlas_ascendc_10_0083.html
  GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

  // Select lower-triangular all-ones matrix staticly initialized on device
  // See `kernel_constants.h`
  GM_ADDR lower = load_tril_matrix<half>(tiling.matmul_size);

  run_scan_multi_core_kernel<half>(input_vec, lower, output_vec, usrWorkspace,
                                   tiling.num_elems, tiling.matmul_size, 0);

  /*

// We consider the L2 cache maxed when the scan takes up around 50% of the
// L2 total cache size -> maybe available L2 size can be a tiling parameter
constexpr uint32_t available_l2_size = L2_SIZE / 2;
const uint32_t fitting_len = scalar::AlignUp(
  available_l2_size / (sizeof(half) + sizeof(float)), GM_ALIGNMENT);

uint32_t remaining_len = vec_len;
uint32_t offset = 0;
uint32_t used_size = mc_scan::get_workspace_size<half, float, IsInclusive>(
                       remaining_len, matmul_size) +
                   vec_len * sizeof(half) + vec_len * sizeof(float);

float starting_value = 0;

while (used_size > available_l2_size && fitting_len < remaining_len) {
run_scan_multi_core_kernel<half, IsInclusive>(
    input_vec + offset * sizeof(half), upper_triangular,
    output_vec + offset * sizeof(float), workspace, fitting_len,
    matmul_size, starting_value);
SyncAll<false
      > ();

  remaining_len -= fitting_len;
  used_size = mc_scan::get_workspace_size<half, float, IsInclusive>(
                  remaining_len, matmul_size) +
              vec_len * sizeof(half) + vec_len * sizeof(float);
  starting_value = scalar::GetGMValue<float>(
      output_vec + offset * sizeof(float), fitting_len - 1, fitting_len);
  offset += fitting_len;
}
if (remaining_len) {
  run_scan_multi_core_kernel<half, IsInclusive>(
      input_vec + offset * sizeof(half), upper_triangular,
      output_vec + offset * sizeof(float), workspace, remaining_len,
      matmul_size, starting_value);
}
*/
}
