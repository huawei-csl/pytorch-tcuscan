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
#include "aclrtlaunch_diff.h"
#include "aclrtlaunch_seg_scan_single_core.h"
#include "host_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace seg_scan {
template <typename InputVecT, typename OutputVecT, typename FlagVecT,
          typename FlagOutputVecT>
constexpr uint32_t GetWorkspaceSize(uint32_t vec_len, uint32_t matmul_size) {
  const uint32_t padded_vec_len =
      host_utils::AlignUp(vec_len, matmul_size * matmul_size);
  const uint32_t padded_input_size = padded_vec_len * sizeof(InputVecT);
  const uint32_t padded_input_rowwise_size =
      padded_vec_len * sizeof(OutputVecT);

  const uint32_t padded_flag_size = padded_vec_len * sizeof(FlagVecT);
  const uint32_t padded_rowwise_flag_size =
      padded_vec_len * sizeof(FlagOutputVecT);

  const uint32_t total_size = padded_input_size + padded_input_rowwise_size +
                              padded_flag_size + padded_rowwise_flag_size;
  return total_size;
}

}  // namespace seg_scan

namespace asc {

at::Tensor alloc_workspace(uint32_t user_workspace_size, at::Device device) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();

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

  ACLRT_LAUNCH_KERNEL(add_custom)
  (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(y.storage().data()),
   const_cast<void *>(z.storage().data()), totalLength, tileLen,
   const_cast<void *>(workspace_tensor.storage().data()));
  return z;
}

at::Tensor run_diff(const at::Tensor &x) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Tensor z = at::empty_like(x);
  const at::Device device = x.options().device();
  const uint32_t tileLen = 1024;
  uint32_t totalLength = 1;
  for (uint32_t size : x.sizes()) {
    totalLength *= size;
  }

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const uint32_t blockDim = static_cast<uint32_t>(totalLength / (10 * tileLen));
  ACLRT_LAUNCH_KERNEL(diff)
  (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(z.storage().data()), totalLength, tileLen,
   const_cast<void *>(workspace_tensor.storage().data()));
  return z;
}

at::Tensor run_seg_scan(const at::Tensor &x, const at::Tensor &f,
                        const at::Tensor &U_s_half,
                        const at::Tensor &U_s_int8) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();

  const int S = U_s_half.sizes()[0];

  const uint32_t tileLen = static_cast<uint32_t>(S);
  uint32_t totalLength = 1;
  for (uint32_t size : x.sizes()) {
    totalLength *= size;
  }

  const at::Tensor z = at::empty(
      {totalLength}, at::TensorOptions().dtype(at::kFloat).device(device));

  const uint32_t user_workspace_size =
      seg_scan::GetWorkspaceSize<int16_t, float, int8_t, int32_t>(totalLength,
                                                                  tileLen);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  const uint32_t blockDim = 1;
  ACLRT_LAUNCH_KERNEL(seg_scan_single_core)
  (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(f.storage().data()),
   const_cast<void *>(U_s_half.storage().data()),
   const_cast<void *>(U_s_int8.storage().data()),
   const_cast<void *>(z.storage().data()), totalLength, tileLen,
   const_cast<void *>(workspace_tensor.storage().data()));
  return z;
}
}  // namespace asc

PYBIND11_MODULE(tcuscan_ops, m) {
  m.doc() = "TCUSCAN pybind11 interfaces";
  m.def("run_add_custom", &asc::run_add_custom, "AscendC Vector add");
  m.def("run_diff", &asc::run_diff, "AscendC Vector diff");
  m.def("run_seg_scan", &asc::run_seg_scan, "AscendC Segmented Scan");
}
