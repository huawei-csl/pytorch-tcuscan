/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_scan_multi_cube.h
 * @brief Kernel implementing multi-cube scan operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "kernel_block_scan.h"
#include "kernel_complete_blocks.h"
#include "kernel_reduce_tiles.h"
#include "kernel_utils.h"

using namespace AscendC;
using namespace kernel_utils;

/**
 * @brief Run the multi core scan kernel.
 *
 * @tparam IsInclusive Indicates whether the scan is inclusive or exclusive.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] lower Pointer to an lower-triangular sqaure matrix filled
 * with ones of size \f$\textit{matmul_size}\f$.
 * @param [in] upper_strict Pointer to a strict upper-triangular square matrix
 * filled with ones of size \f$\textit{matmul_size}\f$.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] vec_len Number of elements to process.
 * @param [in] matmul_size Size of the matmul tile used in the kernel.
 * @param [in] starting_value Starting value added to all entries of the
 * output vector.
 */
template <typename InputT, bool IsInclusive = true>
__aicore__ inline void run_scan_multi_cube_kernel(
    GM_ADDR input_vec, GM_ADDR lower, GM_ADDR upper_strict, GM_ADDR output_vec,
    GM_ADDR workspace, uint32_t vec_len, uint32_t matmul_size,
    cube_unit::CubeOutType_t<InputT> starting_value = 0) {
  using OutputT = cube_unit::CubeOutType_t<InputT>;
  const uint32_t align_size = matmul_size * matmul_size;
  GM_ADDR const sums = workspace;

  if ASCEND_IS_AIC {
    KernelBlockScan<InputT, false> op(vec_len, matmul_size);
    op.Init(input_vec, lower, upper_strict, output_vec);
    op.Process();
    PipeBarrier<PIPE_ALL>();
  }

  if ASCEND_IS_AIV {
    KernelReduceTiles<InputT, 1> op_reduce(align_size, vec_len);
    op_reduce.Init(input_vec, sums);
    op_reduce.Process();
  }

  SyncAll<false /*isAIVOnly*/>();

  if ASCEND_IS_AIV {
    KernelCompleteBlocks<float, true, 1> op(vec_len, align_size, align_size);
    op.Init(output_vec, sums, output_vec);
    op.Process();
  }
}
