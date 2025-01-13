/**
 * @file pybind11.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "aclrtlaunch_add_custom.h"
#include "aclrtlaunch_compress.h"
#include "aclrtlaunch_diff.h"
#include "aclrtlaunch_scan_multi_core.h"
#include "aclrtlaunch_seg_scan_single_core.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_compress.h"
#include "tiling/tiling_diff.h"
#include "tiling/tiling_scan_multi_core.h"
#include "tiling/tiling_seg_scan_single_core.h"
#include "tiling/tiling_vadd.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "workspace.h"

static const char *SOC_VERSION = "Ascend910B4";

namespace asc {

at::Tensor alloc_workspace(uint32_t user_workspace_size, at::Device device) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(SOC_VERSION);

  const uint32_t system_workspace_size =
      static_cast<uint32_t>(ascendc_platform->GetLibApiWorkSpaceSize());
  const uint32_t workspace_size = user_workspace_size + system_workspace_size;
  const at::Tensor workspace_tensor = at::empty(
      {workspace_size}, at::TensorOptions().dtype(at::kByte).device(device));

  return workspace_tensor;
}

at::Tensor run_add_custom(const at::Tensor &x, const at::Tensor &y) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Tensor z = at::empty_like(x);
  const at::Device device = x.options().device();
  const uint32_t blockDim = 8;
  const uint32_t tileLen = 128;
  uint32_t totalLength = 1;
  for (uint32_t size : x.sizes()) {
    totalLength *= size;
  }

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const VaddTiling tiling{totalLength, tileLen};

  constexpr size_t tilingSize = sizeof(MultiCoreScanTiling);
  const uint8_t *tilingHost = reinterpret_cast<const uint8_t *>(&tiling);
  uint8_t *tilingDevice;

  aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize,
              ACL_MEMCPY_HOST_TO_DEVICE);

  ACLRT_LAUNCH_KERNEL(add_custom)
  (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(y.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);

  aclrtFree(tilingDevice);

  return z;
}

at::Tensor run_diff(const at::Tensor &x) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Tensor z = at::empty_like(x);
  const at::Device device = x.options().device();
  const uint32_t tileLen = 256;
  uint32_t totalLength = 1;
  for (uint32_t size : x.sizes()) {
    totalLength *= size;
  }

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const DiffTiling tiling{totalLength, tileLen};
  constexpr size_t tilingSize = sizeof(DiffTiling);
  const uint8_t *tilingHost = reinterpret_cast<const uint8_t *>(&tiling);

  uint8_t *tilingDevice;
  aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize,
              ACL_MEMCPY_HOST_TO_DEVICE);

  const uint32_t blockDim = static_cast<uint32_t>(totalLength / (10 * tileLen));
  ACLRT_LAUNCH_KERNEL(diff)
  (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);

  aclrtFree(tilingDevice);

  return z;
}

at::Tensor run_seg_scan(const at::Tensor &x, const at::Tensor &f,
                        const at::Tensor &U_s_half,
                        const at::Tensor &U_s_int8) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();

  const int S = U_s_half.sizes()[0];

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  uint32_t totalLength = 1;
  for (uint32_t size : x.sizes()) {
    totalLength *= size;
  }

  const at::Tensor z = at::empty(
      {totalLength}, at::TensorOptions().dtype(at::kFloat).device(device));

  const SegScanSingleCoreTiling tiling{totalLength, matmul_size};

  constexpr size_t tilingSize = sizeof(SegScanSingleCoreTiling);
  const uint8_t *tilingHost = reinterpret_cast<const uint8_t *>(&tiling);

  uint8_t *tilingDevice;
  aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize,
              ACL_MEMCPY_HOST_TO_DEVICE);

  const uint32_t user_workspace_size =
      workspace::seg_scan::GetWorkspaceSize<int16_t, float, int8_t, int32_t>(
          totalLength, matmul_size);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  const uint32_t blockDim = 1;
  ACLRT_LAUNCH_KERNEL(seg_scan_single_core)
  (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(f.storage().data()),
   const_cast<void *>(U_s_half.storage().data()),
   const_cast<void *>(U_s_int8.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);

  aclrtFree(tilingDevice);

  return z;
}

at::Tensor run_scan_multi_core(const at::Tensor &x, const at::Tensor &U_s) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(SOC_VERSION);

  const int S = U_s.sizes()[0];

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  uint32_t totalLength = 1;
  for (uint32_t size : x.sizes()) {
    totalLength *= size;
  }

  const at::Tensor z = at::empty(
      {totalLength}, at::TensorOptions().dtype(at::kFloat).device(device));

  const uint32_t tile_elems = matmul_size * matmul_size;
  const size_t num_tiles = totalLength / tile_elems;

  uint32_t blockDim = ascendc_platform->GetCoreNum() / 2;
  while (num_tiles % blockDim != 0) {
    blockDim--;
  }
  if (blockDim <= 1) {
    blockDim = 8;
  }

  const MultiCoreScanTiling tiling{blockDim, totalLength, matmul_size};

  constexpr size_t tilingSize = sizeof(MultiCoreScanTiling);
  const uint8_t *tilingHost = reinterpret_cast<const uint8_t *>(&tiling);

  uint8_t *tilingDevice;
  aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize,
              ACL_MEMCPY_HOST_TO_DEVICE);

  const uint32_t user_workspace_size =
      workspace::mc_scan::GetWorkspaceSize<int16_t, float>(
          tiling.num_elems, tiling.matmul_size, tiling.num_blocks);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  ACLRT_LAUNCH_KERNEL(scan_multi_core)
  (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(U_s.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);

  aclrtFree(tilingDevice);

  return z;
}

at::Tensor run_compress(const at::Tensor &x, const at::Tensor &mask,
                        const at::Tensor &U_s) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const at::Tensor z = at::empty_like(x);

  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(SOC_VERSION);

  const int S = U_s.sizes()[0];

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  uint32_t totalLength = 1;
  for (uint32_t size : x.sizes()) {
    totalLength *= size;
  }

  const uint32_t tile_elems = matmul_size * matmul_size;
  const uint32_t vec_tile_size = tile_elems / 2;

  const size_t num_tiles = totalLength / tile_elems;

  uint32_t blockDim = ascendc_platform->GetCoreNum() / 2;
  while (num_tiles % blockDim != 0) {
    blockDim--;
  }
  if (blockDim <= 1) {
    blockDim = 8;
  }

  const CompressTiling tiling{totalLength, matmul_size, vec_tile_size};

  constexpr size_t tilingSize = sizeof(CompressTiling);
  const uint8_t *tilingHost = reinterpret_cast<const uint8_t *>(&tiling);

  uint8_t *tilingDevice;
  aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize,
              ACL_MEMCPY_HOST_TO_DEVICE);

  const uint32_t user_workspace_size = workspace::compress::GetWorkspaceSize(
      tiling.size, tiling.scan_tile_size, tiling.compress_tile_size);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  ACLRT_LAUNCH_KERNEL(compress)
  (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(mask.storage().data()),
   const_cast<void *>(U_s.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);

  aclrtFree(tilingDevice);

  return z;
}

}  // namespace asc

PYBIND11_MODULE(tcuscan_ops, m) {
  m.doc() = "TCUSCAN pybind11 interfaces";
  m.def("run_add_custom", &asc::run_add_custom, "AscendC Vector add");
  m.def("run_diff", &asc::run_diff, "AscendC Vector diff");
  m.def("run_seg_scan", &asc::run_seg_scan, "AscendC Segmented Scan");
  m.def("run_scan_multi_core", &asc::run_scan_multi_core, "AscendC MCSCAN");
  m.def("run_compress", &asc::run_compress, "AscendC MCSCAN");
}
