#include "kernels/kernel_csr_gather.h"
#include "tiling/tiling_csr_gather.h"

__aicore__ inline void CopyTiling(CSRGatherTiling *tiling, GM_ADDR tilingGM) {
  uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
  auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

  for (uint32_t i = 0; i < sizeof(CSRGatherTiling) / sizeof(uint32_t);
       i++, ptr++) {
    *ptr = *(tiling32 + i);
  }
}

extern "C" __global__ __aicore__ void csr_gather(GM_ADDR values_in,
                                                 GM_ADDR cols_in, GM_ADDR x_in,
                                                 GM_ADDR z_out,
                                                 GM_ADDR workspace,
                                                 GM_ADDR tilingGm) {
  CSRGatherTiling tiling;
  CopyTiling(&tiling, tilingGm);

  run_csr_gather<false>(values_in, cols_in, x_in, z_out, tiling.num_elems,
                        tiling.num_x_elems, tiling.tile_len, workspace);
}