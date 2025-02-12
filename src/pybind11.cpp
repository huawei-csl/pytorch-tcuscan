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

#include "aclrtlaunch_compress_fp16.h"
#include "aclrtlaunch_compress_fp32.h"
#include "aclrtlaunch_compress_pos_fp16.h"
#include "aclrtlaunch_compress_pos_fp32.h"
#include "aclrtlaunch_copy_fp16.h"
#include "aclrtlaunch_copy_fp32.h"
#include "aclrtlaunch_csr_gather.h"
#include "aclrtlaunch_diff_fp16.h"
#include "aclrtlaunch_diff_fp32.h"
#include "aclrtlaunch_scan_multi_core_fp16.h"
#include "aclrtlaunch_scan_multi_core_int8.h"
#include "aclrtlaunch_scan_single_core_fp16.h"
#include "aclrtlaunch_scan_single_core_int8.h"
#include "aclrtlaunch_seg_scan_mc_revert.h"
#include "aclrtlaunch_seg_scan_single_core.h"
#include "aclrtlaunch_seg_scan_vec_single_core.h"
#include "aclrtlaunch_vadd_custom.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_compress.h"
#include "tiling/tiling_copy.h"
#include "tiling/tiling_csr_gather.h"
#include "tiling/tiling_diff.h"
#include "tiling/tiling_scan_multi_core.h"
#include "tiling/tiling_scan_single_core.h"
#include "tiling/tiling_seg_scan_mc_revert.h"
#include "tiling/tiling_seg_scan_single_core.h"
#include "tiling/tiling_seg_scan_vec_single_core.h"
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

size_t byte_size(const at::Tensor &x) {
  const auto dtype = x.options().dtype();

  if (dtype == torch::kHalf or dtype == torch::kInt16) {
    return 2;
  } else if (dtype == torch::kInt8) {
    return 1;
  } else {
    return 4;
  }
}

at::Tensor run_add(const at::Tensor &x, const at::Tensor &y) {
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

  constexpr size_t tilingSize = sizeof(VaddTiling);
  const uint8_t *tilingHost = reinterpret_cast<const uint8_t *>(&tiling);
  uint8_t *tilingDevice;

  aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize,
              ACL_MEMCPY_HOST_TO_DEVICE);

  ACLRT_LAUNCH_KERNEL(vadd_custom)
  (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(y.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);

  aclrtFree(tilingDevice);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

at::Tensor run_diff(const at::Tensor &x, int64_t max_size) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const auto dtype = x.options().dtype();
  const at::Device device = x.options().device();
  // Tiling length must be around 10,000
  const uint32_t tileLen = 20 * 1024 / byte_size(x);

  uint32_t totalLength;
  if (max_size > 0) {
    totalLength = max_size;
  } else {
    totalLength = 1;
    for (uint32_t size : x.sizes()) {
      totalLength *= size;
    }
  }

  const at::Tensor z =
      at::empty({totalLength}, at::TensorOptions().dtype(dtype).device(device));

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const DiffTiling tiling{totalLength, tileLen};
  constexpr size_t tilingSize = sizeof(DiffTiling);
  const uint8_t *tilingHost = reinterpret_cast<const uint8_t *>(&tiling);

  uint8_t *tilingDevice;
  aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize,
              ACL_MEMCPY_HOST_TO_DEVICE);

  uint32_t blockDim =
      static_cast<uint32_t>((totalLength + tileLen - 1) / tileLen);

  if (blockDim < 1) {
    blockDim = 1;
  }

  if (dtype == torch::kHalf) {
    ACLRT_LAUNCH_KERNEL(diff_fp16)
    (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);
  } else {
    ACLRT_LAUNCH_KERNEL(diff_fp32)
    (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);
  }

  aclrtFree(tilingDevice);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

at::Tensor run_seg_scan_mc_revert(const at::Tensor &x, const at::Tensor &f,
                                  const at::Tensor &diff) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const int32_t blockDim = 40;
  const uint32_t tileLen = 1024;

  uint32_t totalLength = 1;
  for (uint32_t size : x.sizes()) {
    totalLength *= size;
  }

  const at::Tensor z = at::empty(
      {totalLength}, at::TensorOptions().dtype(at::kFloat).device(device));

  const uint32_t diff_len = static_cast<uint32_t>(diff.numel());

  const SegScanMcRevertTiling tiling{blockDim, totalLength, diff_len, tileLen};

  constexpr size_t tilingSize = sizeof(SegScanMcRevertTiling);
  const uint8_t *tilingHost = reinterpret_cast<const uint8_t *>(&tiling);

  uint8_t *tilingDevice;
  aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize,
              ACL_MEMCPY_HOST_TO_DEVICE);
  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  ACLRT_LAUNCH_KERNEL(seg_scan_mc_revert)
  (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(f.storage().data()),
   const_cast<void *>(diff.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);

  aclrtFree(tilingDevice);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

at::Tensor run_seg_scan(const at::Tensor &x, const at::Tensor &f, int S) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();

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
      workspace::seg_scan::GetWorkspaceSize<int16_t /* half */, int8_t>(
          totalLength, matmul_size);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  ACLRT_LAUNCH_KERNEL(seg_scan_single_core)
  (1 /* single core*/, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(f.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);

  aclrtFree(tilingDevice);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

at::Tensor run_scan_multi_core(const at::Tensor &x, int S) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out =
      dtype == torch::kHalf ? torch::kFloat32 : torch::kInt32;
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(SOC_VERSION);

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  uint32_t totalLength = 1;
  for (uint32_t size : x.sizes()) {
    totalLength *= size;
  }

  // Outuput is always 32-bits (float or int32_t)
  const at::Tensor z = at::empty(
      {totalLength}, at::TensorOptions().dtype(dtype_out).device(device));

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

  if (dtype == torch::kHalf) {
    const uint32_t user_workspace_size =
        workspace::mc_scan::GetWorkspaceSize<int16_t>(
            tiling.num_elems, tiling.matmul_size, tiling.num_blocks);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);
    ACLRT_LAUNCH_KERNEL(scan_multi_core_fp16)
    (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);
  } else {
    const uint32_t user_workspace_size =
        workspace::mc_scan::GetWorkspaceSize<int8_t>(
            tiling.num_elems, tiling.matmul_size, tiling.num_blocks);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);

    ACLRT_LAUNCH_KERNEL(scan_multi_core_int8)
    (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);
  }

  aclrtFree(tilingDevice);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

at::Tensor run_compress(const at::Tensor &x, const at::Tensor &mask, int S) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(SOC_VERSION);

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  uint32_t totalLength = 1;
  for (uint32_t size : x.sizes()) {
    totalLength *= size;
  }

  const uint32_t tile_elems = matmul_size * matmul_size;
  const uint32_t vec_tile_size = tile_elems / 2;

  const uint32_t num_tiles = totalLength / tile_elems;

  uint32_t blockDim = ascendc_platform->GetCoreNum() / 2;
  while (num_tiles % blockDim != 0) {
    blockDim--;
  }
  if (blockDim <= 1) {
    blockDim = 1;
  }

  const at::Tensor z =
      at::empty({totalLength}, at::TensorOptions().dtype(dtype).device(device));

  const CompressTiling tiling{totalLength, matmul_size, vec_tile_size};

  constexpr size_t tilingSize = sizeof(CompressTiling);
  const uint8_t *tilingHost = reinterpret_cast<const uint8_t *>(&tiling);

  uint8_t *tilingDevice;
  aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize,
              ACL_MEMCPY_HOST_TO_DEVICE);

  const uint32_t user_workspace_size =
      workspace::compress::GetWorkspaceSize<int8_t>(tiling, blockDim);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  if (dtype == torch::kHalf or dtype == torch::kInt16) {
    ACLRT_LAUNCH_KERNEL(compress_fp16)
    (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(mask.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);
  } else {
    ACLRT_LAUNCH_KERNEL(compress_fp32)
    (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(mask.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);
  }

  aclrtFree(tilingDevice);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

at::Tensor run_csr_gather(const at::Tensor &values, const at::Tensor &cols,
                          const at::Tensor &x) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Tensor z = at::empty_like(values);
  const at::Device device = x.options().device();
  const uint32_t tileLen = 4 * 1024;

  uint32_t values_len = 1;
  for (uint32_t size : values.sizes()) {
    values_len *= size;
  }

  uint32_t x_len = 1;
  for (uint32_t size : x.sizes()) {
    x_len *= size;
  }

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const CSRGatherTiling tiling{values_len, x_len, tileLen};

  constexpr size_t tilingSize = sizeof(CSRGatherTiling);
  const uint8_t *tilingHost = reinterpret_cast<const uint8_t *>(&tiling);
  uint8_t *tilingDevice;

  aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize,
              ACL_MEMCPY_HOST_TO_DEVICE);

  uint32_t blockDim = static_cast<uint32_t>(values_len / (4 * tileLen));
  blockDim = blockDim > 60 ? 40 : blockDim;

  if (blockDim <= 1) {
    blockDim = 1;
  }

  ACLRT_LAUNCH_KERNEL(csr_gather)
  (blockDim, acl_stream, const_cast<void *>(values.storage().data()),
   const_cast<void *>(cols.storage().data()),
   const_cast<void *>(x.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);

  aclrtFree(tilingDevice);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

at::Tensor run_compress_pos(const at::Tensor &x, const at::Tensor &mask,
                            const at::Tensor &pos, int s) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(SOC_VERSION);

  const uint32_t matmul_size = static_cast<uint32_t>(s);
  uint32_t totalLength = 1;
  for (uint32_t size : x.sizes()) {
    totalLength *= size;
  }

  const uint32_t tile_elems = matmul_size * matmul_size;
  const uint32_t vec_tile_size = tile_elems / 2;

  const uint32_t num_tiles = totalLength / tile_elems;

  uint32_t blockDim = ascendc_platform->GetCoreNum() / 2;
  while (num_tiles % blockDim != 0) {
    blockDim--;
  }
  if (blockDim <= 1) {
    blockDim = 1;
  }

  // Last entry of pos tensor contains number of output elements.
  const at::Tensor z =
      at::empty({pos[totalLength - 1].item<int32_t>()},
                at::TensorOptions().dtype(dtype).device(device));

  const CompressTiling tiling{totalLength, matmul_size, vec_tile_size};

  constexpr size_t tilingSize = sizeof(CompressTiling);
  const uint8_t *tilingHost = reinterpret_cast<const uint8_t *>(&tiling);

  uint8_t *tilingDevice;
  aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize,
              ACL_MEMCPY_HOST_TO_DEVICE);

  const uint32_t user_workspace_size =
      workspace::compress::GetWorkspaceSize<int8_t>(tiling, blockDim);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  if (dtype == torch::kHalf or dtype == torch::kInt16) {
    ACLRT_LAUNCH_KERNEL(compress_pos_fp16)
    (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(mask.storage().data()),
     const_cast<void *>(pos.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);
  } else {
    ACLRT_LAUNCH_KERNEL(compress_pos_fp32)
    (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(mask.storage().data()),
     const_cast<void *>(pos.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);
  }

  aclrtFree(tilingDevice);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

at::Tensor run_seg_sum(const at::Tensor &x, const at::Tensor &f, int S) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(SOC_VERSION);
  const at::Device device = x.options().device();

  const at::Tensor scan_x = run_scan_multi_core(x, S);
  const at::Tensor out_positions = run_scan_multi_core(f, S);
  const at::Tensor compress_scan_x =
      run_compress_pos(scan_x, f, out_positions, S);

  const at::Tensor prepend =
      at::empty({1}, at::TensorOptions().dtype(at::kFloat).device(device))
          .zero_();
  aclrtSynchronizeStream(acl_stream);

  const at::Tensor prep_compress_scan_x =
      torch::cat({prepend, compress_scan_x});

  const at::Tensor z = torch::diff(prep_compress_scan_x);

  return z;
}

at::Tensor run_copy(const at::Tensor &x, int s) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(SOC_VERSION);

  // get total length
  uint32_t totalLength = 1;
  for (uint32_t size : x.sizes()) {
    totalLength *= size;
  }
  const at::Tensor z =
      at::empty({totalLength}, at::TensorOptions().dtype(dtype).device(device));

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const uint32_t num_block = 1;  // required for 1 core
  const uint32_t tile_size = static_cast<uint32_t>(s);
  const CopyTiling tiling{num_block, totalLength, tile_size};

  constexpr size_t tilingSize = sizeof(CopyTiling);
  const uint8_t *tilingHost = reinterpret_cast<const uint8_t *>(&tiling);

  uint8_t *tilingDevice;
  aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize,
              ACL_MEMCPY_HOST_TO_DEVICE);

  if (dtype == at::kFloat) {
    ACLRT_LAUNCH_KERNEL(copy_fp32)
    (1 /* single core*/, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);
  } else {
    ACLRT_LAUNCH_KERNEL(copy_fp16)
    (1 /* single core*/, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);
  }

  aclrtFree(tilingDevice);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

at::Tensor run_scan_single_core(const at::Tensor &x, int S) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(SOC_VERSION);

  // get total length
  const uint32_t matmul_size = static_cast<uint32_t>(S);
  uint32_t totalLength = 1;
  for (uint32_t size : x.sizes()) {
    totalLength *= size;
  }
  // Outuput is always 32-bits (float or int32_t)
  const at::Tensor z = at::empty(
      {totalLength}, at::TensorOptions().dtype(at::kFloat).device(device));

  const SingleCoreScanTiling tiling{totalLength, matmul_size};

  constexpr size_t tilingSize = sizeof(SingleCoreScanTiling);
  const uint8_t *tilingHost = reinterpret_cast<const uint8_t *>(&tiling);

  uint8_t *tilingDevice;
  aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize,
              ACL_MEMCPY_HOST_TO_DEVICE);

  if (dtype == torch::kInt8) {
    const uint32_t user_workspace_size =
        workspace::sc_scan::GetWorkspaceSize<int8_t>(tiling);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);
    ACLRT_LAUNCH_KERNEL(scan_single_core_int8)
    (1 /* single core*/, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);
  } else {
    const uint32_t user_workspace_size =
        workspace::sc_scan::GetWorkspaceSize<int16_t>(tiling);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);
    ACLRT_LAUNCH_KERNEL(scan_single_core_fp16)
    (1 /* single core*/, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);
  }

  aclrtFree(tilingDevice);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

at::Tensor run_seg_scan_vec(const at::Tensor &x, const at::Tensor &f, int S) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();

  const uint32_t tile_len = static_cast<uint32_t>(S);
  uint32_t totalLength = 1;
  for (uint32_t size : x.sizes()) {
    totalLength *= size;
  }

  const at::Tensor z = at::empty(
      {totalLength}, at::TensorOptions().dtype(at::kFloat).device(device));

  const SegScanVecSingleCoreTiling tiling{totalLength, tile_len};

  constexpr size_t tilingSize = sizeof(SegScanVecSingleCoreTiling);
  const uint8_t *tilingHost = reinterpret_cast<const uint8_t *>(&tiling);

  uint8_t *tilingDevice;
  aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize,
              ACL_MEMCPY_HOST_TO_DEVICE);

  const uint32_t user_workspace_size = 0;
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  ACLRT_LAUNCH_KERNEL(seg_scan_vec_single_core)
  (1 /* single core*/, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(f.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tilingDevice);

  aclrtFree(tilingDevice);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

}  // namespace asc

PYBIND11_MODULE(tcuscan_ops, m) {
  m.doc() = "TCUSCAN pybind11 interfaces";
  m.def("run_add", &asc::run_add, "AscendC Vector add");
  m.def("run_diff", &asc::run_diff, pybind11::arg(),
        pybind11::arg("max_size") = 0, "AscendC Vector diff");
  m.def("run_seg_scan", &asc::run_seg_scan, "AscendC Segmented Scan");
  m.def("run_scan_multi_core", &asc::run_scan_multi_core, "AscendC MCSCAN");
  m.def("run_csr_gather", &asc::run_csr_gather, "AscendC CSR gather");
  m.def("run_compress", &asc::run_compress, "AscendC compress");
  m.def("run_compress_pos", &asc::run_compress_pos,
        "AscendC compress with pre-computed output positions");
  m.def("run_seg_sum", &asc::run_seg_sum, "AscendC Segmented Sum");
  m.def("run_copy", &asc::run_copy, "AscendC Copy single core");
  m.def("run_scan_single_core", &asc::run_scan_single_core,
        "AscendC Scan Single Core");
  m.def("run_seg_scan_vec", &asc::run_seg_scan_vec,
        "AscendC Segmented Scan (vector-only)");
  m.def("run_seg_scan_mc_revert", &asc::run_seg_scan_mc_revert,
        "AscendC Vector Revert for MC Segmented Scan");
}
