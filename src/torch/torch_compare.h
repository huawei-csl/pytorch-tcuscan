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

// Kernel launchers; documented at their definitions in src/*.cpp
/// @cond
extern "C" void launch_count_if_fp16(uint32_t blockDim, void* stream,
                                     void* vec_in, void* vec_out,
                                     void* workspace, void* tiling_gm);
extern "C" void launch_greater_equal_fp16(uint32_t blockDim, void* stream,
                                          void* vec_in, void* vec_out,
                                          void* workspace, void* tiling_gm);
extern "C" void launch_less_equal_fp16(uint32_t blockDim, void* stream,
                                       void* vec_in, void* vec_out,
                                       void* workspace, void* tiling_gm);
extern "C" void launch_greater_equal_count_fp16(uint32_t blockDim, void* stream,
                                                void* vec_in, void* counts_out,
                                                void* workspace,
                                                void* tiling_gm);
extern "C" void launch_greater_equal_count_fp32(uint32_t blockDim, void* stream,
                                                void* vec_in, void* counts_out,
                                                void* workspace,
                                                void* tiling_gm);
extern "C" void launch_greater_equal_ind_fp16(uint32_t blockDim, void* stream,
                                              void* vec_in, void* vec_out,
                                              void* indices_out,
                                              void* workspace, void* tiling_gm);
extern "C" void launch_greater_equal_ind_fp32(uint32_t blockDim, void* stream,
                                              void* vec_in, void* vec_out,
                                              void* indices_out,
                                              void* workspace, void* tiling_gm);
/// @endcond

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
  const at::Tensor tiling_device = alloc_copy_tiling_tensor(tiling, device);

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == at::kHalf) {
    launch_count_if_fp16(block_dim, acl_stream,
                         const_cast<void*>(x.storage().data()),
                         const_cast<void*>(z.storage().data()),
                         const_cast<void*>(workspace_tensor.storage().data()),
                         tiling_device.data_ptr());
  } else {
    /* Unsupported */
  }

  aclrtSynchronizeStream(acl_stream);

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
  const at::Tensor tiling_device = alloc_copy_tiling_tensor(tiling, device);

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == at::kHalf) {
    launch_greater_equal_fp16(
        block_dim, acl_stream, const_cast<void*>(x.storage().data()),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()),
        tiling_device.data_ptr());
  } else {
    /* Not supported */
  }

  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Returns binary tensor `{ x_i <= pivot }` of type `int8_t`.
 *
 * @param [in] x Input data vector.
 * @param [in] pivot Input pivot
 * @param [in] tile_len Tile length that is assigned on each AIV core.
 * @return Binary mask tensor with 1 where `x_i <= pivot`, 0 otherwise.
 */
at::Tensor run_less_equal(const at::Tensor& x, float pivot, uint32_t tile_len) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t num_elems = x.numel();

  const uint32_t block_dim = host_utils::CeilDiv(num_elems, tile_len);
  const at::Tensor z = at::empty(
      {num_elems}, at::TensorOptions().dtype(torch::kInt8).device(device));

  const GreaterEqualTiling tiling{block_dim, num_elems, tile_len, pivot};
  const at::Tensor tiling_device = alloc_copy_tiling_tensor(tiling, device);

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == at::kHalf) {
    launch_less_equal_fp16(block_dim, acl_stream,
                           const_cast<void*>(x.storage().data()),
                           const_cast<void*>(z.storage().data()),
                           const_cast<void*>(workspace_tensor.storage().data()),
                           tiling_device.data_ptr());
  } else {
    /* Not supported */
  }

  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Greater-equal filter that also returns the input indices.
 *
 * Computes `(x[x >= pivot], where(x >= pivot))` without materializing a dense
 * `int8` mask in global memory: the predicate is evaluated inside the
 * compaction kernel itself.
 *
 * Two launches are needed: the first counts, per AIV core, how many elements
 * satisfy the predicate (so that the host can size the outputs and each core
 * can derive its own output offset), the second does the compaction. Both are
 * launched with the same `blockDim` and the same `tile_len`, which is what
 * makes the per-core offsets consistent.
 *
 * @param [in] x Input data 1D vector.
 * @param [in] pivot Pivot to filter by.
 * @param [in] S Tiling parameter. Typical values: 32, 64, 128.
 * @return Tuple-2 of the selected values and their indices (int32).
 */
std::tuple<at::Tensor, at::Tensor> run_greater_equal_ind(const at::Tensor& x,
                                                         float pivot, int S) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t vec_len = x.numel();
  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t tile_len = matmul_size * matmul_size / 2;

  const uint32_t num_tiles = host_utils::CeilDiv(vec_len, tile_len);
  const uint32_t block_dim =
      std::max(1u, std::min(num_tiles, ascendc_platform->GetCoreNumAiv()));

  const GreaterEqualTiling tiling{block_dim, vec_len, tile_len, pivot};
  const at::Tensor tiling_device = alloc_copy_tiling_tensor(tiling, device);

  // The kernel counts the selected elements per core into the workspace
  // before compacting. Only the cores that take part write an entry, so the
  // buffer is zeroed; see `workspace_block_counts`.
  const uint32_t user_workspace_size = tcuscan::get_workspace_size(tiling);
  const at::Tensor workspace_tensor =
      alloc_zeros_workspace(user_workspace_size, device);

  // Both outputs are as long as there are elements at or above the pivot,
  // which only the device knows. Allocating that exactly would mean counting
  // on the host first and reading the total back, a blocking synchronization
  // that stalls the stream before the kernel is even launched. Allocate for
  // the worst case instead -- every element selected -- and return a view of
  // the part that was written.
  const at::Tensor z =
      at::empty({vec_len}, at::TensorOptions().dtype(dtype).device(device));
  const at::Tensor indices_out = at::empty(
      {vec_len}, at::TensorOptions().dtype(torch::kInt32).device(device));

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == torch::kHalf) {
    launch_greater_equal_ind_fp16(
        block_dim, acl_stream, const_cast<void*>(x.storage().data()),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(indices_out.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()),
        tiling_device.data_ptr());
  } else {
    launch_greater_equal_ind_fp32(
        block_dim, acl_stream, const_cast<void*>(x.storage().data()),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(indices_out.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()),
        tiling_device.data_ptr());
  }

  aclrtSynchronizeStream(acl_stream);

  const int32_t num_ges =
      workspace_block_counts(workspace_tensor, block_dim).sum().item<int32_t>();

  return std::make_tuple(z.slice(0, 0, num_ges),
                         indices_out.slice(0, 0, num_ges));
}
}  // namespace tcuscan
