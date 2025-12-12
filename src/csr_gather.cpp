#include "kernels/kernel_csr_gather.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_csr_gather.h"

extern "C" __global__ __aicore__ void csr_gather(
    GM_ADDR values_in, GM_ADDR cols_in, GM_ADDR rows_in, GM_ADDR x_in,
    GM_ADDR z_out, GM_ADDR workspace, GM_ADDR tiling_gm) {
  (void)workspace;
  tcuscan::CSRGatherTiling t;
  GetTilingData(&t, tiling_gm);

  run_csr_gather<false>(values_in, cols_in, rows_in, x_in, z_out, t.num_elems,
                        t.num_row_ptr, t.num_x_elems, t.tile_len);
}
