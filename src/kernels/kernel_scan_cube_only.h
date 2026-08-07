/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_scan_cube_only.h
 * @brief Kernel implementing a single AI-core cube-only scan.
 */

#include "kernels/kernel_block_scan.h"
#include "kernels/kernel_complete_blocks_sc.h"
#include "kernels/kernel_pad.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_scan_cube_only.h"

using namespace AscendC;

/**
 * @brief Run the scan cube only kernel.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] upper Pointer to an upper-triangular all-ones matrix
 * of size \f$\textit{matmul_size} \times \textit{matmul_size}\f$.
 * @param [in] lower Pointer to a strict lower-triangular all-ones matrix
 * of size \f$\textit{matmul_size} \times \textit{matmul_size}\f$.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] tiling Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void scan_cube_only(GM_ADDR input_vec,
                                                     GM_ADDR upper,
                                                     GM_ADDR lower,
                                                     GM_ADDR output_vec,
                                                     GM_ADDR tiling) {
  CubeOnlyScanTiling tiling_data;
  tiling::GetTilingData(&tiling_data, tiling);

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
