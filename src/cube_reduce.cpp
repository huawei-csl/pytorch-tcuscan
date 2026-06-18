/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file cube_reduce.cpp
 * @brief Kernel implementing a multi-core MIX block reducer.
 */
#include "kernels/kernel_pto_cube_reduce.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_cube_reduce.h"

/**
 * @ingroup reduce
 * @brief Run `cube_reduce` kernel with dtype fp16.
 *
 * @param [in] vec_in Pointer to an input vector.
 * @param [in] vec_out Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void cube_reduce_fp16(GM_ADDR vec_in,
                                                       GM_ADDR vec_out,
                                                       GM_ADDR workspace,
                                                       GM_ADDR tiling_gm) {
  tcuscan::CubeReduceTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t matmul_size = tiling.matmul_size;

  using OutT = tcuscan::cube_unit::CubeOutType_t<half>;
  auto* in  = reinterpret_cast<__gm__ half*>(vec_in);
  auto* out = reinterpret_cast<__gm__ OutT*>(vec_out);
  switch (matmul_size) {
    case 16:
      tcuscan::run_pto_cube_reduce<half, 16>(in, out, workspace, vec_len);
      break;
    case 32:
      tcuscan::run_pto_cube_reduce<half, 32>(in, out, workspace, vec_len);
      break;
    case 64:
      tcuscan::run_pto_cube_reduce<half, 64>(in, out, workspace, vec_len);
      break;
    case 128:
      tcuscan::run_pto_cube_reduce<half, 128>(in, out, workspace, vec_len);
      break;
    default:
      break;
  }
}

/**
 * @ingroup reduce
 * @brief Run `cube_reduce` kernel with dtype int8.
 *
 * @param [in] vec_in Pointer to an input vector.
 * @param [in] vec_out Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void cube_reduce_int8(GM_ADDR vec_in,
                                                       GM_ADDR vec_out,
                                                       GM_ADDR workspace,
                                                       GM_ADDR tiling_gm) {
  tcuscan::CubeReduceTiling tiling;
  GetTilingData(&tiling, tiling_gm);

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t matmul_size = tiling.matmul_size;

  using OutT = tcuscan::cube_unit::CubeOutType_t<int8_t>;
  auto* in  = reinterpret_cast<__gm__ int8_t*>(vec_in);
  auto* out = reinterpret_cast<__gm__ OutT*>(vec_out);
  switch (matmul_size) {
    case 16:
      tcuscan::run_pto_cube_reduce<int8_t, 16>(in, out, workspace, vec_len);
      break;
    case 32:
      tcuscan::run_pto_cube_reduce<int8_t, 32>(in, out, workspace, vec_len);
      break;
    case 64:
      tcuscan::run_pto_cube_reduce<int8_t, 64>(in, out, workspace, vec_len);
      break;
    case 128:
      tcuscan::run_pto_cube_reduce<int8_t, 128>(in, out, workspace, vec_len);
      break;
    default:
      break;
  }
}
