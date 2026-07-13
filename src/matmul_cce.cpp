#include "kernels/ascendc_kernel_operator.h"

#ifdef __DAV_C220_CUBE__
// Cube-specific CCE intrinsics need to be inside this MACRO

#include "kernels/kernel_matmul_cce.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_matmul_cce.h"

/**
 * @brief Copies tiling structure from global memory to registers.
 *
 * @tparam TilingT Structure representing kernel tiling parameters.
 * @param [in] tiling Pointer to the structure allocated in registers.
 * @param [in] tiling_global Pointer to the structure in global memory.
 */
template <typename TilingT>
__aicore__ inline void GetTilingData(TilingT* const tiling,
                                     GM_ADDR tiling_global) {
  uint32_t* const tiling_32b = reinterpret_cast<uint32_t*>(tiling);
  const __gm__ uint32_t* const tiling_global_32b =
      reinterpret_cast<__gm__ uint32_t*>(tiling_global);

  for (uint32_t i = 0; i < sizeof(TilingT) / sizeof(uint32_t); i++) {
    tiling_32b[i] = tiling_global_32b[i];
  }
}

extern "C" __global__ __aicore__ void matmul_cce(GM_ADDR x, GM_ADDR y,
                                                 GM_ADDR z, GM_ADDR workspace,
                                                 GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::MatMulCCETiling t;
  tcuscan::GetTilingData(&t, tiling_gm);

  const int M = t.M;
  const int N = t.K;
  const int K = t.K;

  run_matmul_cce(x, y, z, M, N, K);
}

#else  // __DAV_C220_VEC__

// Stub for vector compile path
extern "C" __global__ __aicore__ void matmul_cce(GM_ADDR a, GM_ADDR b,
                                                 GM_ADDR c, GM_ADDR workspace,
                                                 GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)a;
  (void)b;
  (void)c;
  (void)workspace;
  (void)tiling;
  pipe_barrier(PIPE_ALL);
}

#endif  // end of __DAV_C220_CUBE__ vs __DAV_C220_VEC__ switch
