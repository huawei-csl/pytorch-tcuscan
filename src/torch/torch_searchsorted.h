/**
 * @file torch_searchsorted.h
 * @brief Torch wrapper for the searchsorted kernel.
 * @date 2026-07-10
 *
 * @copyright Copyright Huawei (c) 2026
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/tiling_searchsorted.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

// AscendC kernel entry points launched below with `<<<>>>`; defined in
// searchsorted.cpp.
extern "C" __global__ __aicore__ void searchsorted_int32(__gm__ void* sorted,
                                                         __gm__ void* values,
                                                         __gm__ void* out,
                                                         __gm__ void* workspace,
                                                         __gm__ void* tilingGm);

namespace tcuscan {

/**
 * @brief Binary search (lower_bound) of int32 needle values into a sorted int32
 * array.
 *
 * Matches `torch.searchsorted(sorted, values, side='left', out_int32=True)`:
 * for each needle `values[i]`, returns the first index `j` such that
 * `sorted[j] >= values[i]`.
 *
 * @param sorted Sorted (ascending) int32 haystack tensor.
 * @param values int32 needle values.
 * @return int32 tensor of insertion indices, same length as `values`.
 */
at::Tensor run_searchsorted(const at::Tensor& sorted,
                            const at::Tensor& values) {
  TORCH_CHECK(sorted.scalar_type() == at::kInt,
              "run_searchsorted: sorted must be int32, got ",
              sorted.scalar_type());
  TORCH_CHECK(values.scalar_type() == at::kInt,
              "run_searchsorted: values must be int32, got ",
              values.scalar_type());

  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  const at::Device device = values.options().device();
  const uint32_t data_len = static_cast<uint32_t>(sorted.numel());
  const uint32_t num_values = static_cast<uint32_t>(values.numel());

  const at::Tensor out =
      at::empty({values.numel()},
                at::TensorOptions().dtype(torch::kInt32).device(device));
  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const SearchsortedTiling tiling{data_len, num_values};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  // The kernel distributes needles over cores in 32-byte-aligned output blocks
  // (8 int32 each); launch one core per block, capped at the cube-core count.
  constexpr uint32_t OUT_BLOCK = 32 / sizeof(int32_t);
  const uint32_t num_blocks = (num_values + OUT_BLOCK - 1) / OUT_BLOCK;
  const uint32_t num_aic = ascendc_platform->GetCoreNumAic();
  uint32_t block_dim = num_blocks < num_aic ? num_blocks : num_aic;
  if (block_dim < 1) {
    block_dim = 1;
  }

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  searchsorted_int32<<<block_dim, nullptr, acl_stream>>>(
      const_cast<void*>(sorted.storage().data()),
      const_cast<void*>(values.storage().data()),
      const_cast<void*>(out.storage().data()),
      const_cast<void*>(workspace_tensor.storage().data()), tiling_device);

  aclrtFree(tiling_device);

  return out;
}

}  // namespace tcuscan
