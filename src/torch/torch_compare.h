/**
 * @file torch_compare.h
 * @brief Torch wrapper for comparison kernels.
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/tiling_count_if.h"
#include "../tiling/tiling_greater_equal.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "workspace.h"

// AscendC kernel entry points launched below with `<<<>>>`; defined in
// compare.cpp.
extern "C" __global__ __aicore__ void count_if_fp16(__gm__ void* vec_in,
                                                    __gm__ void* vec_out,
                                                    __gm__ void* workspace,
                                                    __gm__ void* tiling_gm);
extern "C" __global__ __aicore__ void greater_equal_fp16(
    __gm__ void* vec_in, __gm__ void* vec_out, __gm__ void* workspace,
    __gm__ void* tiling_gm);

namespace tcuscan {

/**
 * @brief Vector count_if kernel
 *
 * @param [in] x Input data vector.
 * @param [in] pivot Input pivot
 * @param [in] tile_len Tile length that is assigned on each AIV core.
 * @param [in] compare_mode Comparison enum of \c AscendC::CompareScalar.
 * Default `less equal`
 * @return Returns a tensor containing the sum of `{ x_i <= pivot}`.
 */
at::Tensor run_count_if(const at::Tensor& x, float pivot, uint32_t tile_len,
                        uint8_t compare_mode) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();

  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t vec_len = x.numel();
  const uint32_t block_dim = std::min(host_utils::CeilDiv(vec_len, tile_len),
                                      ascendc_platform->GetCoreNumAiv());

  const at::Tensor z =
      at::zeros({1}, at::TensorOptions().dtype(torch::kInt32).device(device));

  const CountIfTiling tiling{block_dim, vec_len, tile_len, pivot, compare_mode};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == at::kHalf) {
    count_if_fp16<<<block_dim, nullptr, acl_stream>>>(
        const_cast<void*>(x.storage().data()),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else {
    /* Unsupported */
  }

  aclrtFree(tiling_device);

  return z;
}

/**
 * @brief Returns binary tensor `{ x_i >= pivot }` of type `int8_t`.
 *
 * @param [in] x Input data vector.
 * @param [in] pivot Input pivot
 * @param [in] tile_len Tile length that is assigned on each AIV core.
 * @return Copy of input vector `x`.
 */
at::Tensor run_greater_equal(const at::Tensor& x, float pivot,
                             uint32_t tile_len) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t num_elems = x.numel();

  const uint32_t block_dim = host_utils::CeilDiv(num_elems, tile_len);
  const at::Tensor z = at::empty(
      {num_elems}, at::TensorOptions().dtype(torch::kInt8).device(device));

  const GreaterEqualTiling tiling{block_dim, num_elems, tile_len, pivot};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == at::kHalf) {
    greater_equal_fp16<<<block_dim, nullptr, acl_stream>>>(
        const_cast<void*>(x.storage().data()),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else {
    /* Not supported */
  }

  aclrtFree(tiling_device);

  return z;
}
}  // namespace tcuscan
