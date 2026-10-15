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

#include <cstring>

#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/CachingHostAllocator.h"
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
 * The copy is issued on the current stream rather than with a synchronous
 * `aclrtMemcpy`. A synchronous copy is not stream-ordered, so it would write
 * the block as soon as the host reached it -- both before ops still sitting in
 * torch_npu's task queue had been placed on the stream, and potentially into a
 * block the caching allocator had just recycled from a kernel still running.
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

  // Taking the stream with `need_empty` drains torch_npu's host-side task
  // queue, so everything already issued against these tensors is on the stream
  // before the copy below is enqueued behind it.
  const c10_npu::NPUStream npu_stream =
      c10_npu::getCurrentNPUStream(device.index());
  aclrtStream stream = npu_stream.stream(true);

  // `tiling_struct` is the caller's stack temporary, so it cannot be the
  // source of an asynchronous copy. Stage it through pinned memory from
  // torch_npu's caching host allocator instead.
  at::DataPtr staging = at_npu::native::HostAlloc(TILING_SIZE);
  std::memcpy(staging.get(), &tiling_struct, TILING_SIZE);

  const aclError copy_status = aclrtMemcpyAsync(
      tiling_device.data_ptr(), TILING_SIZE, staging.get(), TILING_SIZE,
      ACL_MEMCPY_HOST_TO_DEVICE, stream);
  TORCH_CHECK(copy_status == ACL_SUCCESS,
              "alloc_copy_tiling_tensor: aclrtMemcpyAsync failed with ",
              copy_status, ".");

  // `staging` is freed as it goes out of scope below. Recording the stream
  // stops the host allocator reissuing that block before the copy reading it
  // has run.
  at_npu::native::CachingHostAllocator_recordEvent(
      staging.get(), staging.get_context(), npu_stream);

  return tiling_device;
}

}  // namespace tcuscan
