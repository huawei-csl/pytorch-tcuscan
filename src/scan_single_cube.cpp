/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file scan_single_cube.cpp
 * @brief Kernel implementing a single AI-core single-cube scan.
 */

#include "kernels/kernel_block_scan.h"
#include "kernels/kernel_complete_blocks_sc.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_scan_single_cube.h"

/**
 * @brief Run the single-cube scan kernel.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] upper Pointer to an upper-triangular all-ones matrix
 * of size \f$\textit{matmul_size} \times \textit{matmul_size}\f$.
 * @param [in] lower Pointer to a strict lower-triangular all-ones matrix
 * of size \f$\textit{matmul_size} \times \textit{matmul_size}\f$.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to a workspace buffer.
 * @param [in] tiling Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void scan_single_cube_fp16(
    GM_ADDR input_vec, GM_ADDR upper, GM_ADDR lower, GM_ADDR output_vec,
    GM_ADDR workspace, GM_ADDR tiling) {
  (void)workspace;
  using namespace tcuscan;
  ScanSingleCubeTiling tiling_data;
  tcuscan::GetTilingData(&tiling_data, tiling);

  if ASCEND_IS_AIC {
    KernelBlockScan<half, true> op(tiling_data.num_elems,
                                   tiling_data.matmul_size);
    op.Init(input_vec, upper, lower, output_vec);
    op.Process();
  }

  if ASCEND_IS_AIV {
    KernelCompleteBlocksSingleCore op(
        tiling_data.num_elems,
        tiling_data.matmul_size * tiling_data.matmul_size);
    op.Init(output_vec, output_vec);
    op.Process();
  }
}
