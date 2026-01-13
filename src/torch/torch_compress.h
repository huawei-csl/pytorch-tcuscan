/**
 * @file torch_compress.h
 * @brief Torch wrapper for compaction/compress.
 * @date 2025-03-27
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/tiling_compress.h"
#include "../tiling/tiling_where.h"
#include "aclrtlaunch_compress_fp16.h"
#include "aclrtlaunch_compress_fp32.h"
#include "aclrtlaunch_compress_ind_fp16.h"
#include "aclrtlaunch_compress_ind_fp32.h"
#include "aclrtlaunch_compress_ind_no_arange_fp16.h"
#include "aclrtlaunch_compress_ind_no_arange_fp32.h"
#include "aclrtlaunch_compress_with_sums_fp16.h"
#include "aclrtlaunch_compress_with_sums_fp32.h"
#include "aclrtlaunch_where_fp16.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_compare.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_reduce.h"
#include "workspace.h"

namespace tcuscan {

/**
 * @brief Parallel vector compaction / compress.
 *
 * @param [in] x Input data 1D vector.
 * @param [in] mask Input boolean flag/mask vector of dtype int8.
 * @param [in] S Tiling parameter. Typical values: 32, 64, 128.
 * @return The subvector of `x`: `x[f == 1]`. Output length is `sum(f == 1)`.
 */
at::Tensor run_compress(const at::Tensor& x, const at::Tensor& mask, int S) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t total_length = x.numel();

  const uint32_t tile_len = matmul_size * matmul_size;

  const uint32_t num_tiles = host_utils::CeilDiv(total_length, tile_len);

  uint32_t block_dim = ascendc_platform->GetCoreNumAic();
  while (num_tiles % block_dim != 0) {
    block_dim--;
  }
  if (block_dim <= 1) {
    block_dim = 1;
  }

  const at::Tensor num_ones_per_block =
      run_reduce_tiles(mask, tile_len / 2, 2 * block_dim);
  const int32_t num_ones = num_ones_per_block.sum().item<int32_t>();

  const at::Tensor z =
      at::empty({num_ones}, at::TensorOptions().dtype(dtype).device(device));

  const CompressTiling tiling{block_dim, total_length, matmul_size};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  const uint32_t user_workspace_size = tcuscan::get_workspace_size(tiling);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  if (dtype == torch::kHalf or dtype == torch::kInt16) {
    ACLRT_LAUNCH_KERNEL(compress_with_sums_fp16)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(mask.storage().data()),
     const_cast<void*>(num_ones_per_block.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else {
    ACLRT_LAUNCH_KERNEL(compress_with_sums_fp32)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(mask.storage().data()),
     const_cast<void*>(num_ones_per_block.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Run the compress kernel given the output length.
 *
 * @param [in] x Input data 1D vector.
 * @param [in] mask Input boolean flag/mask vector of dtype int8.
 * @param [in] output_len Output length. `output_len` equals to `sum(mask)`
 * @param [in] S Tiling parameter. Typical values: 32, 64, 128.
 * @return The subvector of `x`: `x[f == 1]`. Output length is `sum(f == 1)`.
 */
at::Tensor run_compress_pos(const at::Tensor& x, const at::Tensor& mask,
                            uint32_t output_len, int S) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t total_length = x.numel();

  const uint32_t tile_len = matmul_size * matmul_size;

  const uint32_t num_tiles = total_length / tile_len;

  uint32_t block_dim = ascendc_platform->GetCoreNumAic();
  while (num_tiles % block_dim != 0) {
    block_dim--;
  }
  if (block_dim <= 1) {
    block_dim = 1;
  }

  const at::Tensor z =
      at::empty({output_len}, at::TensorOptions().dtype(dtype).device(device));

  const CompressTiling tiling{block_dim, total_length, matmul_size};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  const uint32_t user_workspace_size = tcuscan::get_workspace_size(tiling);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  if (dtype == torch::kHalf or dtype == torch::kInt16) {
    ACLRT_LAUNCH_KERNEL(compress_fp16)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(mask.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else {
    ACLRT_LAUNCH_KERNEL(compress_fp32)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(mask.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Parallel compaction / compress with indices.
 *
 * @param [in] x Input data vector.
 * @param [in] indices_in Input indices vector.
 * @param [in] mask Input boolean flag/mask vector of dtype int8.
 * @param [in] S Tiling parameter. Typical values: 32, 64, 128.
 * @return Tuple-2 of compacted data and corresponding indices.
 */
std::tuple<at::Tensor, at::Tensor> run_compress_ind(
    const at::Tensor& x, const at::Tensor& indices_in, const at::Tensor& mask,
    int S) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto indices_dtype = indices_in.options().dtype();

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t total_length = x.numel();

  const uint32_t tile_len = matmul_size * matmul_size;

  const uint32_t num_tiles = host_utils::CeilDiv(total_length, tile_len);

  uint32_t block_dim = ascendc_platform->GetCoreNumAic();
  while (num_tiles % block_dim != 0) {
    block_dim--;
  }
  if (block_dim <= 1) {
    block_dim = 1;
  }

  const at::Tensor num_ones_per_block =
      run_reduce_tiles(mask, tile_len / 2, 2 * block_dim);
  const int32_t num_ones = num_ones_per_block.sum().item<int32_t>();

  const at::Tensor z =
      at::empty({num_ones}, at::TensorOptions().dtype(dtype).device(device));

  const at::Tensor indices_out = at::empty(
      {num_ones}, at::TensorOptions().dtype(indices_dtype).device(device));

  const CompressTiling tiling{block_dim, total_length, matmul_size};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  const uint32_t user_workspace_size = tcuscan::get_workspace_size(tiling);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  if (dtype == torch::kHalf or dtype == torch::kInt16) {
    ACLRT_LAUNCH_KERNEL(compress_ind_fp16)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(indices_in.storage().data()),
     const_cast<void*>(mask.storage().data()),
     const_cast<void*>(num_ones_per_block.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(indices_out.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else {
    ACLRT_LAUNCH_KERNEL(compress_ind_fp32)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(indices_in.storage().data()),
     const_cast<void*>(mask.storage().data()),
     const_cast<void*>(num_ones_per_block.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(indices_out.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return std::make_tuple(z, indices_out);
}

/**
 * @brief Parallel compaction / compress with indices.
 *
 * @param [in] x Input data vector.
 * @param [in] mask Input boolean flag/mask vector of dtype int8.
 * @param [in] S Tiling parameter. Typical values: 32, 64, 128.
 * @return Tuple-2 of compacted data and corresponding indices.
 */
std::tuple<at::Tensor, at::Tensor> run_compress_ind_no_arange(
    const at::Tensor& x, const at::Tensor& mask, int S) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t total_length = x.numel();

  const uint32_t tile_len = matmul_size * matmul_size;

  const uint32_t num_tiles = host_utils::CeilDiv(total_length, tile_len);

  uint32_t block_dim = ascendc_platform->GetCoreNumAic();
  while (num_tiles % block_dim != 0) {
    block_dim--;
  }
  if (block_dim <= 1) {
    block_dim = 1;
  }

  const at::Tensor num_ones_per_block =
      run_reduce_tiles(mask, tile_len / 2, 2 * block_dim);
  const int32_t num_ones = num_ones_per_block.sum().item<int32_t>();

  const at::Tensor z =
      at::empty({num_ones}, at::TensorOptions().dtype(dtype).device(device));

  const at::Tensor indices_out = at::empty(
      {num_ones}, at::TensorOptions().dtype(torch::kInt32).device(device));

  const CompressTiling tiling{block_dim, total_length, matmul_size};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  if (dtype == torch::kHalf or dtype == torch::kInt16) {
    ACLRT_LAUNCH_KERNEL(compress_ind_no_arange_fp16)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(mask.storage().data()),
     const_cast<void*>(num_ones_per_block.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(indices_out.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else {
    ACLRT_LAUNCH_KERNEL(compress_ind_no_arange_fp32)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(mask.storage().data()),
     const_cast<void*>(num_ones_per_block.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(indices_out.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return std::make_tuple(z, indices_out);
}

/**
 * @brief Filter a tensor using `{ x_i >= pivot}`.
 *
 * @param [in] x Input data 1D vector.
 * @param [in] pivot Input pivot to filter by greter than.
 * @param [in] S Tiling parameter. Typical values: 32, 64, 128.
 * @return The subvector of `x`: `x[x >= pivot]`.
 */
at::Tensor run_filter_greater_equal(const at::Tensor& x, float pivot, int S) {
  const at::Tensor mask = run_greater_equal(x, pivot, S * S);
  return run_compress(x, mask, S);
}

/**
 * @brief Filter a tensor using `{ x_i <= pivot}`.
 *
 * @param [in] x Input data 1D vector.
 * @param [in] pivot Input pivot to filter by.
 * @param [in] S Tiling parameter. Typical values: 32, 64, 128.
 * @return The subvector of `x`: `x[x <= pivot]`.
 */
at::Tensor run_filter_less_equal(const at::Tensor& x, float pivot, int S) {
  const at::Tensor mask = (x <= pivot).to(torch::kInt8);
  return run_compress(x, mask, S);
}

/**
 * @brief Vector `where` kernel
 *
 * @param x Input data vector.
 * @param [in] pivot Input pivot to filter by.
 * @param [in] S Tiling parameter. Typical values: 32, 64, 128.
 * @return Indices where `{ x_i >= pivot}`.
 */
at::Tensor run_where(const at::Tensor& x, float pivot, int S) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t total_len = x.numel();

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t total_length = x.numel();

  const uint32_t tile_len = matmul_size * matmul_size;
  const uint32_t vec_tile_len = tile_len / 2;

  const uint32_t num_tiles = host_utils::CeilDiv(total_length, tile_len);

  uint32_t block_dim = ascendc_platform->GetCoreNumAic();
  while (num_tiles % block_dim != 0) {
    block_dim--;
  }
  if (block_dim <= 1) {
    block_dim = 1;
  }

  const at::Tensor mask = run_greater_equal(x, pivot, vec_tile_len);

  const at::Tensor num_ones_per_block =
      run_reduce_tiles(mask, vec_tile_len, 2 * block_dim);
  const int32_t num_ones = num_ones_per_block.sum().item<int32_t>();

  const at::Tensor z = at::empty(
      {num_ones}, at::TensorOptions().dtype(torch::kInt32).device(device));

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const WhereTiling tiling{block_dim, total_len, vec_tile_len, pivot};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  if (dtype == at::kHalf) {
    ACLRT_LAUNCH_KERNEL(where_fp16)
    (block_dim, acl_stream, const_cast<void*>(mask.storage().data()),
     const_cast<void*>(num_ones_per_block.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else {
    /* Unsupported */
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

}  // namespace tcuscan
