#include "kernels/kernel_csr_gather.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_csr_gather.h"

extern "C" __global__ __aicore__ void csr_gather_fp16(
    GM_ADDR values_in, GM_ADDR cols_in, GM_ADDR x_in, GM_ADDR z_out,
    GM_ADDR workspace, GM_ADDR tiling_gm) {
  (void)workspace;
  tcuscan::CSRGatherTiling t;
  GetTilingData(&t, tiling_gm);

  tcuscan::run_csr_gather<half, false>(values_in, cols_in, x_in, z_out,
                                       t.num_elems, t.num_x_elems, t.tile_len);
}

extern "C" __global__ __aicore__ void csr_gather_int16(
    GM_ADDR values_in, GM_ADDR cols_in, GM_ADDR x_in, GM_ADDR z_out,
    GM_ADDR workspace, GM_ADDR tiling_gm) {
  (void)workspace;
  tcuscan::CSRGatherTiling t;
  GetTilingData(&t, tiling_gm);

  tcuscan::run_csr_gather<int16_t, false>(
      values_in, cols_in, x_in, z_out, t.num_elems, t.num_x_elems, t.tile_len);
}

extern "C" __global__ __aicore__ void csr_gather_fp32(
    GM_ADDR values_in, GM_ADDR cols_in, GM_ADDR x_in, GM_ADDR z_out,
    GM_ADDR workspace, GM_ADDR tiling_gm) {
  (void)workspace;
  tcuscan::CSRGatherTiling t;
  GetTilingData(&t, tiling_gm);

  tcuscan::run_csr_gather<float, false>(values_in, cols_in, x_in, z_out,
                                        t.num_elems, t.num_x_elems, t.tile_len);
}
