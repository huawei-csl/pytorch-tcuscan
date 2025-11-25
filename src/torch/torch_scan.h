/**
 * @file torch_scan.h
 * @brief Torch wrapper for scan kernels.
 * @date 2025-03-27
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/tiling_block_scan.h"
#include "../tiling/tiling_row_scan.h"
#include "../tiling/tiling_scan_batch.h"
#include "../tiling/tiling_scan_multi_core.h"
#include "../tiling/tiling_scan_multi_cube.h"
#include "../tiling/tiling_scan_single_core.h"
#include "aclrtlaunch_block_scan_fp16.h"
#include "aclrtlaunch_row_scan_fp16.h"
#include "aclrtlaunch_scan_batch_fp16.h"
#include "aclrtlaunch_scan_batch_fp32.h"
#include "aclrtlaunch_scan_multi_core_fp16.h"
#include "aclrtlaunch_scan_multi_core_fp16_no_l2.h"
#include "aclrtlaunch_scan_multi_core_int8.h"
#include "aclrtlaunch_scan_multi_core_int8_no_l2.h"
#include "aclrtlaunch_scan_multi_cube_fp16.h"
#include "aclrtlaunch_scan_single_core_fp16.h"
#include "aclrtlaunch_scan_single_core_fp32.h"
#include "aclrtlaunch_scan_single_core_int8.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_scan_cpu.h"
#include "workspace.h"

namespace tcuscan {

/**
 * @brief Returns the prefix sum of an input 1D vector using one AI Core.
 *
 * @param x Input 1D vector.
 * @param S Matrix tiling parameter. Typical values: 32, 64, 128.
 * @param starting_sum Starting sum for the scan_single core. Default is 0
 * @return The prefix sum of `x`.
 */
at::Tensor run_scan_single_core(const at::Tensor& x, int S,
                                double starting_sum = 0) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out = dtype == torch::kHalf || dtype == torch::kFloat32
                             ? torch::kFloat32
                             : torch::kInt32;

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t total_length = x.numel();

  // Outuput is always 32-bits (float or int32_t)
  const at::Tensor z = at::empty(
      {total_length}, at::TensorOptions().dtype(dtype_out).device(device));
  SingleCoreScanTiling tiling;
  tiling.num_elems = total_length;
  tiling.matmul_size = matmul_size;

  if (dtype == torch::kHalf || dtype == torch::kFloat32)
    tiling.running_sum.float_value = static_cast<float>(starting_sum);

  if (dtype == torch::kInt8) {
    tiling.running_sum.int_value = static_cast<int32_t>(starting_sum);
  }

  uint8_t* tiling_device = tcuscan::alloc_copy_tiling(tiling);

  if (dtype == torch::kInt8) {
    const uint32_t user_workspace_size =
        workspace::sc_scan::get_workspace_size<int8_t>(tiling);
    const at::Tensor workspace_tensor =
        alloc_zeros_workspace(user_workspace_size, device);
    ACLRT_LAUNCH_KERNEL(scan_single_core_int8)
    (1 /* single core*/, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else if (dtype == torch::kHalf) {
    const uint32_t user_workspace_size =
        workspace::sc_scan::get_workspace_size<int16_t>(tiling);
    const at::Tensor workspace_tensor =
        alloc_zeros_workspace(user_workspace_size, device);
    ACLRT_LAUNCH_KERNEL(scan_single_core_fp16)
    (1 /* single core*/, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else if (dtype == torch::kFloat32) {
    const uint32_t user_workspace_size =
        workspace::sc_scan::get_workspace_size<float>(tiling);
    const at::Tensor workspace_tensor =
        alloc_zeros_workspace(user_workspace_size, device);
    ACLRT_LAUNCH_KERNEL(scan_single_core_fp32)
    (1 /* single core*/, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else {
    /* Unsupported */
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Returns the row-wise scan of a 2D input matrix `x`.
 *
 * @param x Input 2D matrix in row-major order.
 * @param S Tiling parameter. Typical values 32, 64, 128.
 * @return A 2D matrix that is the row-wise scan of `x`.
 */
at::Tensor run_scan_batch(const at::Tensor& x, int S) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out = dtype == torch::kHalf || dtype == torch::kFloat32
                             ? torch::kFloat32
                             : torch::kInt32;

  const uint32_t matmul_size = static_cast<uint32_t>(S);

  uint32_t num_el = x.numel();
  const uint32_t batch_size = num_el / x.size(-1);
  const uint32_t vec_len = x.size(-1);

  const at::Tensor z =
      at::empty({batch_size, vec_len},
                at::TensorOptions().dtype(dtype_out).device(device));

  uint32_t block_dim = ascendc_platform->GetCoreNumAic();
  if (batch_size < block_dim) {
    block_dim = batch_size;
  }

  const ScanBatchTiling tiling{block_dim, vec_len, batch_size, matmul_size,
                               2 /* vec-cube ratio */};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  if (dtype == torch::kHalf) {
    const uint32_t user_workspace_size =
        workspace::scan_batch::get_workspace_size<int16_t>(tiling);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);
    ACLRT_LAUNCH_KERNEL(scan_batch_fp16)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else if (dtype == torch::kFloat32) {
    const uint32_t user_workspace_size =
        workspace::scan_batch::get_workspace_size<float>(tiling);
    const at::Tensor workspace_tensor =
        alloc_zeros_workspace(user_workspace_size, device);
    ACLRT_LAUNCH_KERNEL(scan_batch_fp32)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else {
    /* Unsupported */
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Returns the prefix sum (scan) of a 1D vector `x`.
 *
 * @param x Input 1D vector.
 * @param S Tiling parameter. Typical values 32, 64, 128.
 * @return The prefix sum of `x`
 */
at::Tensor run_scan_multi_core(const at::Tensor& x, int S) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out =
      dtype == torch::kHalf ? torch::kFloat32 : torch::kInt32;

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t total_length = x.numel();

  const at::Tensor z = at::empty(
      {total_length}, at::TensorOptions().dtype(dtype_out).device(device));

  const uint32_t tile_elems = matmul_size * matmul_size;
  const size_t num_tiles = host_utils::CeilDiv(total_length, tile_elems);

  uint32_t block_dim = ascendc_platform->GetCoreNumAic();
  if (num_tiles < block_dim) {
    block_dim = num_tiles;
  }

  uint64_t l2_cache_size;
  ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::L2,
                                   l2_cache_size);

  const MultiCoreScanTiling tiling{block_dim, total_length, matmul_size,
                                   l2_cache_size};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  if (dtype == torch::kHalf) {
    const uint32_t user_workspace_size =
        workspace::mc_scan::get_workspace_size<int16_t>(
            tiling.num_elems, tiling.matmul_size, tiling.num_blocks);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);
    ACLRT_LAUNCH_KERNEL(scan_multi_core_fp16)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else {
    const uint32_t user_workspace_size =
        workspace::mc_scan::get_workspace_size<int8_t>(
            tiling.num_elems, tiling.matmul_size, tiling.num_blocks);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);

    ACLRT_LAUNCH_KERNEL(scan_multi_core_int8)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Returns the prefix sum (scan) of a 1D vector `x` without L2 splitting
 * optimization.
 *
 * @param x Input 1D vector.
 * @param S Tiling parameter. Typical values 32, 64, 128.
 * @return The prefix sum of `x`
 */
at::Tensor run_scan_multi_core_no_l2(const at::Tensor& x, int S) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out =
      dtype == torch::kHalf ? torch::kFloat32 : torch::kInt32;

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t total_length = x.numel();

  const at::Tensor z = at::empty(
      {total_length}, at::TensorOptions().dtype(dtype_out).device(device));

  const uint32_t tile_elems = matmul_size * matmul_size;
  const size_t num_tiles = host_utils::CeilDiv(total_length, tile_elems);

  uint32_t block_dim = ascendc_platform->GetCoreNumAic();
  if (num_tiles < block_dim) {
    block_dim = num_tiles;
  }

  const MultiCoreScanTiling tiling{block_dim, total_length, matmul_size};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  if (dtype == torch::kHalf) {
    const uint32_t user_workspace_size =
        workspace::mc_scan::get_workspace_size<int16_t>(
            tiling.num_elems, tiling.matmul_size, tiling.num_blocks);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);
    ACLRT_LAUNCH_KERNEL(scan_multi_core_fp16_no_l2)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else {
    const uint32_t user_workspace_size =
        tcuscan::workspace::mc_scan::get_workspace_size<int8_t>(
            tiling.num_elems, tiling.matmul_size, tiling.num_blocks);
    const at::Tensor workspace_tensor =
        tcuscan::alloc_workspace(user_workspace_size, device);

    ACLRT_LAUNCH_KERNEL(scan_multi_core_int8_no_l2)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Returns the prefix sum (scan) of each block of length S of an 1D
 * vector `x`.
 *
 * @param x Input 1D vector.
 * @param S Tiling parameter. Typical values 32, 64, 128.
 * @return The prefix sum of each concecutive block of `x` (block length S).
 */
at::Tensor run_row_scan(const at::Tensor& x, int S) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out =
      dtype == torch::kHalf ? torch::kFloat32 : torch::kInt32;

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t M = x.size(0);

  const at::Tensor z =
      at::empty({M * S}, at::TensorOptions().dtype(dtype_out).device(device));

  const size_t num_tiles = M;

  uint32_t block_dim = ascendc_platform->GetCoreNumAic();
  if (num_tiles < block_dim) {
    block_dim = num_tiles;
  }

  const RowScanTiling tiling{M * matmul_size, matmul_size};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  if (dtype == torch::kHalf) {
    const at::Tensor workspace_tensor = alloc_workspace(0, device);
    ACLRT_LAUNCH_KERNEL(row_scan_fp16)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else {
    /* Unsupported*/
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Returns the prefix sum (scan) of each block of length S^2 of an 1D
 * vector `x`.
 *
 * @param x Input 1D vector.
 * @param upper Upper triangular all-ones matrix of size S.
 * @param lower_strict Strict lower triangular all-ones matrix of size S.
 * @return The prefix sum of each concecutive block of `x` (block length S^2).
 */
at::Tensor run_block_scan(const at::Tensor& x, const at::Tensor& upper,
                          const at::Tensor& lower_strict) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out =
      dtype == torch::kHalf ? torch::kFloat32 : torch::kInt32;

  const uint32_t s = static_cast<uint32_t>(upper.size(0));
  const uint32_t total_len = x.numel();

  const at::Tensor z = at::empty(
      {total_len}, at::TensorOptions().dtype(dtype_out).device(device));

  const uint32_t tile_elems = s * s;
  const size_t num_tiles = host_utils::CeilDiv(total_len, tile_elems);

  uint32_t block_dim = ascendc_platform->GetCoreNumAic();
  if (num_tiles < block_dim) {
    block_dim = num_tiles;
  }

  const BlockScanTiling tiling{total_len, s};
  uint8_t* tiling_device = tcuscan::alloc_copy_tiling(tiling);

  if (dtype == torch::kHalf) {
    const at::Tensor workspace_tensor = tcuscan::alloc_workspace(0, device);
    ACLRT_LAUNCH_KERNEL(block_scan_fp16)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(upper.storage().data()),
     const_cast<void*>(lower_strict.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else {
    /* Unsupported*/
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Returns the prefix sum (scan) by computing the block scan of length
 * S^2 using the cube unit.
 *
 * @param x Input 1D vector.
 * @param upper Upper triangular all-ones matrix of size S.
 * @param lower_strict Strict lower triangular all-ones matrix of size S.
 * @return The prefix sum of each concecutive block of `x` (block length S^2).
 */
at::Tensor run_scan_multi_cube(const at::Tensor& x, const at::Tensor& upper,
                               const at::Tensor& lower_strict) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out =
      dtype == torch::kHalf ? torch::kFloat32 : torch::kInt32;

  const uint32_t s = static_cast<uint32_t>(upper.size(0));
  const uint32_t total_len = x.numel();

  const at::Tensor z = at::empty(
      {total_len}, at::TensorOptions().dtype(dtype_out).device(device));

  const uint32_t tile_elems = s * s;
  const size_t num_tiles = host_utils::CeilDiv(total_len, tile_elems);

  uint32_t block_dim = ascendc_platform->GetCoreNumAic();
  if (num_tiles < block_dim) {
    block_dim = num_tiles;
  }

  uint64_t l2_cache_size;
  ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::L2,
                                   l2_cache_size);

  const ScanMultiCubeTiling tiling{block_dim, total_len, s, l2_cache_size};
  uint8_t* tiling_device = tcuscan::alloc_copy_tiling(tiling);

  if (dtype == torch::kHalf) {
    const uint32_t user_workspace_size =
        workspace::mc_scan::get_workspace_size<int16_t>(
            tiling.num_elems, tiling.matmul_size, tiling.num_blocks);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);
    ACLRT_LAUNCH_KERNEL(scan_multi_cube_fp16)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(upper.storage().data()),
     const_cast<void*>(lower_strict.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else {
    /* Unsupported*/
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

}  // namespace tcuscan
