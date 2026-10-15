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

namespace tcuscan {
/**
 * @brief Allocates a torch tensor for AscendC kernel working space.
 *
 * @param user_workspace_size Workspace size in bytes
 * @param device Device on which the tensor is allocated.
 * @return at::Tensor The allocated workspace tensor.
 */
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
      platform_ascendc::PlatformAscendCManager::GetInstance();
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
size_t byte_size(const at::Tensor& x) {
  const auto dtype = x.options().dtype();

  if (dtype == torch::kHalf or dtype == torch::kInt16) {
    return 2;
  } else if (dtype == torch::kInt8) {
    return 1;
  }
  return 4;
}

/**
 * @brief Copies the host tiling struct into a device tensor.
 *
 * The allocation comes from the caching allocator rather than from a per-call
 * `aclrtMalloc`/`aclrtFree` pair, and it is released when the returned tensor
 * goes out of scope.
 *
 * The caller must still keep the tensor alive until the kernel has finished
 * with it: the kernels are launched directly on the ACL stream, so the caching
 * allocator has no record of the dependency and would otherwise be free to
 * hand the block to the next allocation while the kernel is still reading it.
 *
 * @tparam T Struct of tiling.
 * @param tiling_struct Input tiling.
 * @param device Device on which the tensor is allocated.
 * @return Byte tensor holding a device copy of the tiling struct.
 */
template <typename T>
at::Tensor alloc_copy_tiling_tensor(const T& tiling_struct, at::Device device) {
  constexpr size_t TILING_SIZE = sizeof(T);
  const at::Tensor tiling_device =
      at::empty({static_cast<int64_t>(TILING_SIZE)},
                at::TensorOptions().dtype(at::kByte).device(device));
  aclrtMemcpy(tiling_device.data_ptr(), TILING_SIZE,
              reinterpret_cast<const uint8_t*>(&tiling_struct), TILING_SIZE,
              ACL_MEMCPY_HOST_TO_DEVICE);

  return tiling_device;
}

}  // namespace tcuscan
