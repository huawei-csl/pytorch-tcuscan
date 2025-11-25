/**
 * @file torch_gen_lower.h
 * @brief Torch wrapper for vector add.
 * @date 2025-03-27
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../host_utils.h"
#include "../tiling/tiling_gen_lower.h"
#include "aclrtlaunch_gen_lower_fp16.h"
// #include "aclrtlaunch_gen_lower_fp32.h"
#include "aclrtlaunch_gen_lower_int8.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace tcuscan {

/**
 * @brief Torch wrapper for generate lower triangular kernel.
 * @param matrix_size Size of the matrix to generate.
 * @param device The npu device to use.
 * @param dtype The desired scalar type for the matrix elements.
 * @return Returns tensor that contains the lower triangular matrix.
 */
at::Tensor run_gen_lower(uint32_t matrix_size, const at::Device& device,
                         const torch::Dtype dtype) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const uint32_t total_len = matrix_size * matrix_size;
  const at::Tensor z =
      at::empty({matrix_size, matrix_size},
                at::TensorOptions().dtype(dtype).device(device));
  const uint32_t tile_len = 8 * matrix_size;
  const uint32_t block_dim = host_utils::CeilDiv(total_len, tile_len);
  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const GenLowerTiling tiling{matrix_size, tile_len};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);
  if (dtype == at::kHalf) {
    ACLRT_LAUNCH_KERNEL(gen_lower_fp16)
    (block_dim, acl_stream, const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else if (dtype == at::kChar) {
    ACLRT_LAUNCH_KERNEL(gen_lower_int8)
    (block_dim, acl_stream, const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

}  // namespace tcuscan
