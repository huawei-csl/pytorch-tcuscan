/**
 * @file torch_matmul_cce.h
 * @brief Torch wrapper for MatMul CCE kernel.
 * @date 2025-03-27
 *
 * @copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/tiling_matmul_cce.h"
#include "acl/acl.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "workspace.h"

extern "C" void launch_matmul_cce(uint32_t blockDim, void* stream, void* x,
                                  void* y, void* z, void* workspace,
                                  void* tiling_gm);

namespace tcuscan {

/**
 * @brief Matrix multiplication CCE kernel.
 *
 * @param a Input left matrix A.
 * @param b Input right matrix B.
 * @return at::Tensor Matrix product C=AB.
 */
at::Tensor matmul_cce(at::Tensor a, at::Tensor b) {
  uint32_t M = a.sizes()[0];
  uint32_t N = b.sizes()[0];  // b is passed as transposed
  uint32_t K = a.sizes()[1];  // == b.sizes()[1]
  const at::Device device = a.options().device();

  at::Tensor c = at::empty({M, N}, a.options());
  uint8_t* a_ptr = reinterpret_cast<uint8_t*>(a.storage().data_ptr().get());
  uint8_t* b_ptr = reinterpret_cast<uint8_t*>(b.storage().data_ptr().get());
  uint8_t* c_ptr = reinterpret_cast<uint8_t*>(c.storage().data_ptr().get());
  uint32_t block_dim = 20;  // 910B4, 20 cube cores

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

  MatMulCCETiling tiling{M, N, K};
  uint8_t* tiling_device = tcuscan::alloc_copy_tiling(tiling);

  const at::Tensor workspace_tensor = tcuscan::alloc_workspace(0, device);

  // Qualified: the enclosing host wrapper tcuscan::matmul_cce would otherwise
  // hide the kernel of the same name.
  ::launch_matmul_cce(block_dim, acl_stream, a_ptr, b_ptr, c_ptr,
                      const_cast<void*>(workspace_tensor.storage().data()),
                      tiling_device);
  aclrtSynchronizeStream(acl_stream);
  aclrtFree(tiling_device);

  return c.index({torch::indexing::Slice({torch::indexing::None, M}), "..."});
}

}  // namespace tcuscan
