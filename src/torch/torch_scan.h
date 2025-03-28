/**
 * @file torch_scan.h
 * @brief Torch wrapper for scan kernels.
 * @date 2025-03-27
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/tiling_scan_batch.h"
#include "../tiling/tiling_scan_multi_core.h"
#include "../tiling/tiling_scan_single_core.h"
#include "aclrtlaunch_scan_batch.h"
#include "aclrtlaunch_scan_multi_core_fp16.h"
#include "aclrtlaunch_scan_multi_core_int8.h"
#include "aclrtlaunch_scan_single_core_fp16.h"
#include "aclrtlaunch_scan_single_core_int8.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "workspace.h"

namespace asc {

namespace scan {

/**
 * @brief Returns the prefix sum of an input 1D vector using one AI Core.
 *
 * @param x Input 1D vector.
 * @param S Matrix tiling parameter. Typical values: 32, 64, 128.
 * @return The prefix sum of `x`.
 */
at::Tensor run_scan_single_core(const at::Tensor &x, int S) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t totalLength = x.numel();

  // Outuput is always 32-bits (float or int32_t)
  const at::Tensor z = at::empty(
      {totalLength}, at::TensorOptions().dtype(at::kFloat).device(device));

  const SingleCoreScanTiling tiling{totalLength, matmul_size};
  uint8_t *tiling_device = allocCopyTiling(tiling);

  if (dtype == torch::kInt8) {
    const uint32_t user_workspace_size =
        workspace::sc_scan::GetWorkspaceSize<int8_t>(tiling);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);
    ACLRT_LAUNCH_KERNEL(scan_single_core_int8)
    (1 /* single core*/, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  } else {
    const uint32_t user_workspace_size =
        workspace::sc_scan::GetWorkspaceSize<int16_t>(tiling);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);
    ACLRT_LAUNCH_KERNEL(scan_single_core_fp16)
    (1 /* single core*/, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Returns the row-wise scan of a 2D input matrix `x`.
 *
 * @param x Input 2D matrix in row-major order.
 * @param S Tiling parameter. Typical values 32, 64, 128.
 * @return A 2D matrix that is the row-wise scan of `x`.
 */
at::Tensor run_scan_batch(const at::Tensor &x, int S) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const uint32_t block_size = x.size(0);  // For tiling/cube core parallelism

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t batch_size = x.size(0);
  const uint32_t vec_len = x.size(1);
  // Outuput is always 32-bits (float or int32_t)
  const at::Tensor z =
      at::empty({batch_size, vec_len},
                at::TensorOptions().dtype(at::kFloat).device(device));
  // Workspace is **always** required, even if it is an empty tensor
  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const ScanBatchTiling tiling{block_size, vec_len, batch_size, matmul_size,
                               2 /* vec-cube ratio */};
  uint8_t *tiling_device = allocCopyTiling(tiling);

  ACLRT_LAUNCH_KERNEL(scan_batch)
  (block_size, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tiling_device);

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Returns the prefix sum (scan) of a 1D vector `x`.
 *
 * @param x Input 1D vector.
 * @param S Tiling parameter. Typical values 32, 64, 128.
 * @return The prefix sum of `x`
 */
at::Tensor run_scan_multi_core(const at::Tensor &x, int S) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(SOC_VERSION);
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out =
      dtype == torch::kHalf ? torch::kFloat32 : torch::kInt32;

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t totalLength = x.numel();

  // Output is always 32-bits (float or int32_t)
  const at::Tensor z = at::empty(
      {totalLength}, at::TensorOptions().dtype(dtype_out).device(device));

  const uint32_t tile_elems = matmul_size * matmul_size;
  const size_t num_tiles = host_utils::CeilDiv(totalLength, tile_elems);

  uint32_t blockDim = ascendc_platform->GetCoreNum() / 2;
  while (num_tiles % blockDim != 0) {
    blockDim--;
  }
  if (blockDim <= 1) {
    blockDim = 1;
  }

  const MultiCoreScanTiling tiling{blockDim, totalLength, matmul_size};
  uint8_t *tiling_device = allocCopyTiling(tiling);

  if (dtype == torch::kHalf) {
    const uint32_t user_workspace_size =
        workspace::mc_scan::GetWorkspaceSize<int16_t>(
            tiling.num_elems, tiling.matmul_size, tiling.num_blocks);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);
    ACLRT_LAUNCH_KERNEL(scan_multi_core_fp16)
    (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  } else {
    const uint32_t user_workspace_size =
        workspace::mc_scan::GetWorkspaceSize<int8_t>(
            tiling.num_elems, tiling.matmul_size, tiling.num_blocks);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);

    ACLRT_LAUNCH_KERNEL(scan_multi_core_int8)
    (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

}  // namespace scan

}  // namespace asc
