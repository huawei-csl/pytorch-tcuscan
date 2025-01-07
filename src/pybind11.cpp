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
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace asc {
at::Tensor run_add_custom(const at::Tensor &x, const at::Tensor &y) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  at::Tensor z = at::empty_like(x);
  const uint32_t blockDim = 8;
  const uint32_t tileLen = 128;
  uint32_t totalLength = 1;
  for (uint32_t size : x.sizes()) {
    totalLength *= size;
  }
  ACLRT_LAUNCH_KERNEL(add_custom)
  (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(y.storage().data()),
   const_cast<void *>(z.storage().data()), totalLength, tileLen);
  return z;
}

at::Tensor run_diff(const at::Tensor &x) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  at::Tensor z = at::empty_like(x);
  const uint32_t tileLen = 1024;
  uint32_t totalLength = 1;
  for (uint32_t size : x.sizes()) {
    totalLength *= size;
  }
  const uint32_t blockDim = static_cast<uint32_t>(totalLength / (10 * tileLen));
  ACLRT_LAUNCH_KERNEL(diff)
  (blockDim, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(z.storage().data()), totalLength, tileLen);
  return z;
}
}  // namespace asc

PYBIND11_MODULE(tcuscan_ops, m) {
  m.doc() = "TCUSCAN pybind11 interfaces";  // optional module docstring
  m.def("run_add_custom", &asc::run_add_custom, "AscendC Vector add");
  m.def("run_diff", &asc::run_diff, "AscendC Vector diff");
}
