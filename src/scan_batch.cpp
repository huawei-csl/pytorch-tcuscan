/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file scan_batch.cpp
 * @brief Kernel implementing a multi-core inclusive scan using either
 * cube-vector or cube-only approach on a batched input.
 */

#include "kernels/constants.h"
#include "kernels/kernel_scan_batch.h"
#include "tiling/tiling_scan_batch.h"

/**
 * @brief Convert tiling struct to global memory.
 *
 * @param [in] tiling Input tiling struct.
 * @param [in] tilingGM Output global memory point to write tiling struct.
 */
__aicore__ inline void CopyTiling(ScanBatchTiling *tiling, GM_ADDR tilingGM) {
  uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
  auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

  for (uint32_t i = 0; i < sizeof(ScanBatchTiling) / sizeof(uint32_t);
       i++, ptr++) {
    *ptr = *(tiling32 + i);
  }
}

/**
 *  @brief Run the multi core inclusive scan kernel on a batched input.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to a workspace vector (empty vector!).
 * @param [in] tiling Pointer to the tiling structure.
 */
extern "C" __global__ __aicore__ void scan_batch(GM_ADDR input_vec,
                                                 GM_ADDR output_vec,
                                                 GM_ADDR workspace,
                                                 GM_ADDR tiling_gm) {
  (void)workspace;
  ScanBatchTiling tiling_data;
  CopyTiling(&tiling_data, tiling_gm);
  GM_ADDR const lower = load_tril_matrix<half>(tiling_data.matmul_size);

  ASCENDC_ASSERT(tiling_data.num_elems % GetFractalK<half>() == 0, {
    KERNEL_LOG(KERNEL_ERROR,
               "The length of the vectors (%d) must be "
               "divisible by the fractal size (%d).",
               tiling_data.num_elems, GetFractalK<half>());
  });
  ASCENDC_ASSERT(tiling_data.batch_size % tiling_data.vec_cube_ratio == 0, {
    KERNEL_LOG(KERNEL_ERROR,
               "The batch size (%d) must be "
               "divisible by the ratio between vector and cube cores (%d).",
               tiling_data.batch_size, tiling_data.vec_cube_ratio);
  });

  if (tiling_data.num_elems <= 128) {
    run_scan_cube_only_kernel<half>(input_vec, lower, output_vec,
                                    tiling_data.num_elems,
                                    tiling_data.batch_size);
  } else {
    run_scan_batch_kernel(input_vec, lower, output_vec, tiling_data.num_elems,
                          tiling_data.batch_size, tiling_data.matmul_size,
                          tiling_data.vec_cube_ratio);
  }
}
