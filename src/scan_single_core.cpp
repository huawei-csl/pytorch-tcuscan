/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file scan_single_core.cpp
 * @brief Entrypoint for scan single core kernel operation.
 */

#include "kernels/constants.h"
#include "kernels/kernel_scan_single_core.h"
#include "lib/matmul_intf.h"
#include "tiling/tiling_scan_single_core.h"

__aicore__ inline void CopyTiling(SingleCoreScanTiling *tiling,
                                  GM_ADDR tilingGM) {
  uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
  auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

  for (uint32_t i = 0; i < sizeof(SingleCoreScanTiling) / sizeof(uint32_t);
       i++, ptr++) {
    *ptr = *(tiling32 + i);
  }
}

/**
 * @brief Run the `scan_single_core` kernel with int8 dtype.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */

extern "C" __global__ __aicore__ void scan_single_core_int8(GM_ADDR vec_in,
                                                            GM_ADDR vec_out,
                                                            GM_ADDR workspace,
                                                            GM_ADDR tilingGm) {
  SingleCoreScanTiling tiling;
  CopyTiling(&tiling, tilingGm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t matmul_size = tiling.matmul_size;

  GM_ADDR const usrWorkspace = AscendC::GetUserWorkspace(workspace);
  GM_ADDR const lower = load_tril_matrix<half>(matmul_size);

  run_scan_single_core<half>(vec_in, lower, vec_out, vec_len, matmul_size,
                             usrWorkspace);
}

/**
 * @brief Run the `scan_single_core` kernel with half/float16 dtype.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer to the output vector.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */

extern "C" __global__ __aicore__ void scan_single_core_fp16(GM_ADDR vec_in,
                                                            GM_ADDR vec_out,
                                                            GM_ADDR workspace,
                                                            GM_ADDR tilingGm) {
  SingleCoreScanTiling tiling;
  CopyTiling(&tiling, tilingGm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t matmul_size = tiling.matmul_size;

  GM_ADDR const usrWorkspace = AscendC::GetUserWorkspace(workspace);
  GM_ADDR const lower = load_tril_matrix<half>(matmul_size);

  run_scan_single_core<half>(vec_in, lower, vec_out, vec_len, matmul_size,
                             usrWorkspace);
}
