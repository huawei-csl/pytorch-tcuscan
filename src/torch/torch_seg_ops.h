/**
 * @file torch_seg_ops.h
 * @brief Torch wrapper for segmented operations.
 * @date 2025-03-27
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/tiling_seg_scan_mc_revert.h"
#include "../tiling/tiling_seg_scan_single_core.h"
#include "../tiling/tiling_seg_scan_vec_single_core.h"
#include "aclrtlaunch_seg_scan_mc_revert.h"
#include "aclrtlaunch_seg_scan_single_core.h"
#include "aclrtlaunch_seg_scan_vec_single_core.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "workspace.h"

namespace asc {

namespace seg_ops {

at::Tensor run_seg_scan_vec(const at::Tensor &x, const at::Tensor &f, int S) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();

  const uint32_t tile_len = static_cast<uint32_t>(S);
  const uint32_t totalLength = x.numel();

  const at::Tensor z = at::empty(
      {totalLength}, at::TensorOptions().dtype(at::kFloat).device(device));

  const SegScanVecSingleCoreTiling tiling{totalLength, tile_len};
  uint8_t *tiling_device = allocCopyTiling(tiling);

  const uint32_t user_workspace_size = 0;
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  ACLRT_LAUNCH_KERNEL(seg_scan_vec_single_core)
  (1 /* single core*/, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(f.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tiling_device);

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

at::Tensor run_seg_sum(const at::Tensor &x, const at::Tensor &f, int S) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();

  const at::Tensor scan_x = scan::run_scan_multi_core(x, S);
  const at::Tensor out_positions = scan::run_scan_multi_core(f, S);
  const at::Tensor compress_scan_x =
      compress::run_compress_pos(scan_x, f, out_positions, S);

  const at::Tensor prepend =
      at::empty({1}, at::TensorOptions().dtype(at::kFloat).device(device))
          .zero_();
  aclrtSynchronizeStream(acl_stream);

  const at::Tensor prep_compress_scan_x =
      torch::cat({prepend, compress_scan_x});

  const at::Tensor z = torch::diff(prep_compress_scan_x);

  return z;
}

at::Tensor run_seg_scan_mc_revert(const at::Tensor &x, const at::Tensor &f,
                                  const at::Tensor &diff) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();

  const uint32_t tileLen = 4 * 1024;
  const uint32_t totalLength = x.numel();

  uint32_t blockDim =
      static_cast<uint32_t>((totalLength + tileLen - 1) / tileLen);
  blockDim = blockDim > 40 ? 40 : blockDim;

  const at::Tensor z = at::empty(
      {totalLength}, at::TensorOptions().dtype(at::kFloat).device(device));

  const uint32_t diff_len = static_cast<uint32_t>(diff.numel());

  const SegScanMcRevertTiling tiling{blockDim, totalLength, diff_len, tileLen};
  uint8_t *tiling_device = allocCopyTiling(tiling);

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  ACLRT_LAUNCH_KERNEL(seg_scan_mc_revert)
  (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(f.storage().data()),
   const_cast<void *>(diff.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tiling_device);

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

at::Tensor run_seg_scan(const at::Tensor &x, const at::Tensor &f, int S) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t totalLength = x.numel();

  const at::Tensor z = at::empty(
      {totalLength}, at::TensorOptions().dtype(at::kFloat).device(device));

  const SegScanSingleCoreTiling tiling{totalLength, matmul_size};
  uint8_t *tiling_device = allocCopyTiling(tiling);

  const uint32_t user_workspace_size =
      workspace::seg_scan::GetWorkspaceSize<int16_t /* half */, int8_t>(
          totalLength, matmul_size);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  ACLRT_LAUNCH_KERNEL(seg_scan_single_core)
  (1 /* single core*/, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(f.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tiling_device);

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

}  // namespace seg_ops

}  // namespace asc
