/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file row_scan.cpp
 * @brief Kernel implementing a multi-core inclusive row scan.
 */

#include "kernel_utils.h"
#include "kernels/constants.h"
#include "kernels/kernel_scan_multi_core.h"
#include "lib/matmul_intf.h"
#include "tiling/tiling_row_scan.h"

template <typename InputT>
__aicore__ inline void _run_row_scan(GM_ADDR input_vec, GM_ADDR output_vec,
                                     GM_ADDR tilingGm) {
  using OutputT = kernel_utils::cube_unit::CubeOutType_t<InputT>;

  RowScanTiling tiling;
  tiling::GetTilingData(&tiling, tilingGm);

  const uint32_t vec_len = tiling.num_elems;
  const uint32_t matmul_size = tiling.S;
  constexpr bool IsInclusive = true;

  GM_ADDR const lower = load_tril_matrix<InputT>(matmul_size);

  if ASCEND_IS_AIC {
    KernelRowScan<InputT> op_cube(matmul_size, matmul_size, vec_len);
    op_cube.Init(input_vec, lower, output_vec);
    op_cube.Process();
  }
}

/**
 * @brief Run the multi core inclusive row scan kernel with dtype fp16
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void row_scan_fp16(GM_ADDR input_vec,
                                                    GM_ADDR output_vec,
                                                    GM_ADDR workspace,
                                                    GM_ADDR tilingGm) {
  (void)workspace;
  _run_row_scan<half>(input_vec, output_vec, tilingGm);
}
