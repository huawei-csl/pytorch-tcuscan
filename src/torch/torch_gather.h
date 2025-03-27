/**
 * @file torch_gather.h
 * @brief Torch wrapper for gather kernels.
 * @date 2025-03-27
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/tiling_csr_gather.h"
#include "../tiling/tiling_gather_spmv.h"
#include "../tiling/tiling_mc_gather.h"
#include "aclrtlaunch_csr_gather.h"
#include "aclrtlaunch_gather_spmv.h"
#include "aclrtlaunch_mc_gather.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "workspace.h"

namespace asc {

namespace gather {

at::Tensor run_mc_gather(const at::Tensor &values, const at::Tensor &idxs,
                         const uint32_t tile_len) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

  const at::Device device = values.options().device();
  const uint32_t tileLen = tile_len;
  const uint32_t blockDim = 20;

  uint32_t values_len = values.numel();
  uint32_t idx_len = idxs.numel();

  const at::Tensor z = at::empty(
      {idx_len}, at::TensorOptions().dtype(at::kFloat).device(device));
  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const McGatherTiling tiling{blockDim, idx_len, tileLen};
  uint8_t *tiling_device = allocCopyTiling(tiling);

  ACLRT_LAUNCH_KERNEL(mc_gather)
  (blockDim, acl_stream, const_cast<void *>(values.storage().data()),
   const_cast<void *>(idxs.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

at::Tensor run_csr_gather(const at::Tensor &values, const at::Tensor &cols,
                          const at::Tensor &x) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Tensor z = at::empty_like(values);
  const at::Device device = x.options().device();
  const uint32_t tileLen = 4 * 1024;

  uint32_t values_len = values.numel();

  uint32_t x_len = x.numel();

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const CSRGatherTiling tiling{values_len, x_len, tileLen};
  uint8_t *tiling_device = allocCopyTiling(tiling);

  uint32_t blockDim =
      static_cast<uint32_t>(host_utils::CeilDiv(values_len, (4 * tileLen)));
  blockDim = blockDim > 60 ? 40 : blockDim;

  if (blockDim <= 1) {
    blockDim = 1;
  }

  ACLRT_LAUNCH_KERNEL(csr_gather)
  (blockDim, acl_stream, const_cast<void *>(values.storage().data()),
   const_cast<void *>(cols.storage().data()),
   const_cast<void *>(x.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tiling_device);

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

at::Tensor run_gather_spmv(const at::Tensor &values, const at::Tensor &idxs,
                           const uint32_t tile_len) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

  const at::Device device = values.options().device();
  const uint32_t tileLen = tile_len;
  const uint32_t blockDim = 20;

  uint32_t values_len = values.numel();
  uint32_t idx_len = idxs.numel();

  const at::Tensor z =
      at::empty({idx_len},
                at::TensorOptions().dtype(values.scalar_type()).device(device));
  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const GatherSpmvTiling tiling{blockDim, idx_len, tileLen};
  uint8_t *tiling_device = allocCopyTiling(tiling);

  ACLRT_LAUNCH_KERNEL(gather_spmv)
  (blockDim, acl_stream, const_cast<void *>(values.storage().data()),
   const_cast<void *>(idxs.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}
}  // namespace gather

}  // namespace asc
