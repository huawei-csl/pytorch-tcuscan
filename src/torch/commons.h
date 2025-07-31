/**
 * @file commons.h
 * @brief Common utilities for torch integration of AsendC kernels.
 * @date 2025-03-27
 *
 * @copyright Copyright Huawei(c) 2025
 *
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

const static char *SOC_VERSION = "Ascend910B4";

namespace asc {
/**
 * @brief Allocates a torch tensor for AscendC kernel working space.
 *
 * @param user_workspace_size Workspace size in bytes
 * @param device Device on which the tensor is allocated.
 * @return at::Tensor The allocated workspace tensor.
 */
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

/**
 * @brief Allocates a torch tensor for AscendC kernel working space, zeroing
 * memory.
 *
 * @param user_workspace_size Workspace size in bytes
 * @param device Device on which the tensor is allocated.
 * @return at::Tensor The allocated workspace tensor.
 */
at::Tensor alloc_zeros_workspace(uint32_t user_workspace_size,
                                 at::Device device) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(SOC_VERSION);
  const uint32_t system_workspace_size =
      static_cast<uint32_t>(ascendc_platform->GetLibApiWorkSpaceSize());
  const uint32_t workspace_size = user_workspace_size + system_workspace_size;
  const at::Tensor workspace_tensor = at::zeros(
      {workspace_size}, at::TensorOptions().dtype(at::kByte).device(device));

  return workspace_tensor;
}

/**
 * @brief Returns number of bytes of given torch tensor.
 *
 * @param x Input torch tensor.
 * @return Number of bytes required for each element of `x`.
 */
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

/**
 * @brief Returns a device pointer on which the host tiling struct is copied at.
 *
 * Note: the user of this method must aclrtFree the returned pointer.
 *
 * @tparam T Struct of tiling
 * @param tiling_struct Input tiling
 * @return Device pointer where tiling struct is copied.
 */
template <typename T>
uint8_t *allocCopyTiling(const T &tiling_struct) {
  constexpr size_t tiling_size = sizeof(T);
  const uint8_t *tiling_host =
      reinterpret_cast<const uint8_t *>(&tiling_struct);

  uint8_t *tiling_device = nullptr;
  aclrtMalloc((void **)&tiling_device, tiling_size, ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMemcpy(tiling_device, tiling_size, tiling_host, tiling_size,
              ACL_MEMCPY_HOST_TO_DEVICE);

  return tiling_device;
}

}  // namespace asc
