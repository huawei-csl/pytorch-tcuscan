/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file seg_scan_single_core.cpp
 * @brief Launcher of seg_scan_single_core
 */

#include "kernels/constants.h"
#include "kernels/kernel_seg_scan_single_core.h"
#include "lib/matmul_intf.h"
#include "tiling/tiling_seg_scan_single_core.h"

/**
 * @brief Convert tiling struct to global memory.
 *
 * @param [in] tiling Input tiling struct.
 * @param [in] tilingGM Output global memory point to write tiling struct.
 */
__aicore__ inline void CopyTiling(SegScanSingleCoreTiling *tiling,
                                  GM_ADDR tilingGM) {
  uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
  auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

  for (uint32_t i = 0; i < sizeof(SegScanSingleCoreTiling) / sizeof(uint32_t);
       i++, ptr++) {
    *ptr = *(tiling32 + i);
  }
}

__aicore__ inline void _run_kernel(GM_ADDR input_vec, GM_ADDR input_flag,
                                   GM_ADDR u_s_half, GM_ADDR u_s_int8,
                                   GM_ADDR output_vec, uint32_t vec_len,
                                   uint32_t matmul_size, GM_ADDR workspace) {
  using InputT = half;
  using OutputT = float;
  using FlagT = int8_t;
  using FlagOutputT = int32_t;

  if (vec_len % (matmul_size * matmul_size) || vec_len % UB_ALIGNMENT) {
    const uint32_t align_size = matmul_size * matmul_size;
    const uint32_t padded_vec_len = scalar::AlignUp(vec_len, align_size);

    uint32_t ws_offset = 0;
    GM_ADDR const padded_input = workspace;
    ws_offset += padded_vec_len * sizeof(InputT);

    GM_ADDR const padded_input_rowwise_scan = workspace + ws_offset;
    ws_offset += padded_vec_len * sizeof(OutputT);

    GM_ADDR const padded_flag = workspace + ws_offset;
    ws_offset += padded_vec_len * sizeof(FlagT);

    GM_ADDR const padded_flag_rowwise_scan = workspace + ws_offset;

    run_pad_kernel<InputT, false>(input_vec, padded_input, vec_len, align_size);
    run_pad_kernel<FlagT, false>(input_flag, padded_flag, vec_len, align_size);

    sync::SyncGroup<sync::GroupSyncDirection::FULL>();

    if ASCEND_IS_AIC {
      KernelScan2PSingleCore<half, int8_t, /*SyncAfter*/ true> op(
          matmul_size, matmul_size, padded_vec_len);
      op.Init(padded_input, padded_flag, u_s_half, u_s_int8,
              padded_input_rowwise_scan, padded_flag_rowwise_scan);
      op.Process();
    }
    if ASCEND_IS_AIV {
      KernelSegScanRevertSpec<OutputT, FlagOutputT> op_vec(matmul_size,
                                                           vec_len);
      op_vec.Init(padded_input_rowwise_scan, padded_flag_rowwise_scan,
                  output_vec);
      op_vec.Process();
    }
  } else {
    // Scanned input flags should be written to the workspace

    GM_ADDR const ws_output = workspace;
    const uint32_t ws_offset = vec_len * sizeof(OutputT);
    GM_ADDR const ws_output_flag = workspace + ws_offset;

    sync::SyncGroup<sync::GroupSyncDirection::FULL>();

    if ASCEND_IS_AIC {
      KernelScan2PSingleCore<half, int8_t, /*SyncAfter*/ true> op(
          matmul_size, matmul_size, vec_len);
      op.Init(input_vec, input_flag, u_s_half, u_s_int8, ws_output,
              ws_output_flag);
      op.Process();
    }

    if ASCEND_IS_AIV {
      KernelSegScanRevertSpec<OutputT, FlagOutputT> op_vec(matmul_size,
                                                           vec_len);
      op_vec.Init(ws_output, ws_output_flag, output_vec);
      op_vec.Process();
    }
  }
}

/**
 * @brief Run the single core segmented scan kernel.
 *
 * @param [in] input_vec Pointer to an input data vector.
 * @param [in] input_flag Pointer to an input flag vector.
 * @param [in] output_vec Pointer to the output vector.
 * @param [in] workspace Pointer to the workspace struct.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void seg_scan_single_core(GM_ADDR input_vec,
                                                           GM_ADDR input_flag,
                                                           GM_ADDR output_vec,
                                                           GM_ADDR workspace,
                                                           GM_ADDR tilingGm) {
  SegScanSingleCoreTiling tiling;
  CopyTiling(&tiling, tilingGm);

  GM_ADDR const lower_half = load_tril_matrix<half>(tiling.matmul_size);
  GM_ADDR const lower_int8 = load_tril_matrix<int8_t>(tiling.matmul_size);

  GM_ADDR const usrWorkspace = AscendC::GetUserWorkspace(workspace);

  _run_kernel(input_vec, input_flag, lower_half, lower_int8, output_vec,
              tiling.vec_len, tiling.matmul_size, usrWorkspace);
}
