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
#include "../tiling/tiling_scan_single_cube.h"
#include "../tiling/tiling_scan_vec_only.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_scan_cpu.h"
#include "workspace.h"

// Kernel launchers; documented at their definitions in src/*.cpp
/// @cond
extern "C" {
void launch_block_scan_fp16(uint32_t blockDim, void* stream, void* input_vec,
                            void* lower, void* upper_strict, void* output_vec,
                            void* workspace, void* tilingGm);
void launch_row_scan_fp16(uint32_t blockDim, void* stream, void* input_vec,
                          void* output_vec, void* workspace, void* tilingGm);
void launch_scan_batch_fp16(uint32_t blockDim, void* stream, void* input_vec,
                            void* output_vec, void* workspace, void* tiling);
void launch_scan_batch_fp32(uint32_t blockDim, void* stream, void* input_vec,
                            void* output_vec, void* workspace, void* tiling);
void launch_scan_multi_core_fp16(uint32_t blockDim, void* stream,
                                 void* input_vec, void* output_vec,
                                 void* workspace, void* tilingGm);
void launch_scan_multi_core_fp32(uint32_t blockDim, void* stream,
                                 void* input_vec, void* output_vec,
                                 void* workspace, void* tilingGm);
void launch_scan_multi_core_fp16_no_l2(uint32_t blockDim, void* stream,
                                       void* input_vec, void* output_vec,
                                       void* workspace, void* tilingGm);
void launch_scan_multi_core_fp32_no_l2(uint32_t blockDim, void* stream,
                                       void* input_vec, void* output_vec,
                                       void* workspace, void* tilingGm);
void launch_scan_multi_core_int8(uint32_t blockDim, void* stream,
                                 void* input_vec, void* output_vec,
                                 void* workspace, void* tilingGm);
void launch_scan_multi_core_int8_no_l2(uint32_t blockDim, void* stream,
                                       void* input_vec, void* output_vec,
                                       void* workspace, void* tilingGm);
void launch_scan_multi_cube_fp16(uint32_t blockDim, void* stream,
                                 void* input_vec, void* lower,
                                 void* upper_strict, void* output_vec,
                                 void* workspace, void* tiling_gm);
void launch_scan_single_core_fp16(uint32_t blockDim, void* stream, void* vec_in,
                                  void* vec_out, void* workspace,
                                  void* tilingGm);
void launch_scan_single_core_fp32(uint32_t blockDim, void* stream, void* vec_in,
                                  void* vec_out, void* workspace,
                                  void* tilingGm);
void launch_scan_single_core_int8(uint32_t blockDim, void* stream, void* vec_in,
                                  void* vec_out, void* workspace,
                                  void* tilingGm);
void launch_scan_single_cube_fp16(uint32_t blockDim, void* stream,
                                  void* input_vec, void* lower,
                                  void* upper_strict, void* output_vec,
                                  void* workspace, void* tilingGm);
void launch_scan_vec_only_fp16(uint32_t blockDim, void* stream, void* input_vec,
                               void* output_vec, void* workspace,
                               void* tilingGm);
}
/// @endcond

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
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out = dtype == torch::kHalf || dtype == torch::kFloat32
                             ? torch::kFloat32
                             : torch::kInt32;

  // The kernel reads its input as a flat row-major buffer, ignoring both
  // strides and the storage offset, so non-contiguous views have to be
  // materialized first. `contiguous()` is a no-op when they already are.
  const at::Tensor x_contig = x.contiguous();

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

  const at::Tensor tiling_device =
      tcuscan::alloc_copy_tiling_tensor(tiling, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
  if (dtype == torch::kInt8) {
    const uint32_t user_workspace_size =
        tcuscan::get_workspace_size<int8_t>(tiling);
    const at::Tensor workspace_tensor =
        alloc_zeros_workspace(user_workspace_size, device);
    launch_scan_single_core_int8(
        1 /* single core*/, acl_stream, x_contig.data_ptr(),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device.data_ptr());
  } else if (dtype == torch::kHalf) {
    const uint32_t user_workspace_size =
        tcuscan::get_workspace_size<int16_t>(tiling);
    const at::Tensor workspace_tensor =
        alloc_zeros_workspace(user_workspace_size, device);
    launch_scan_single_core_fp16(
        1 /* single core*/, acl_stream, x_contig.data_ptr(),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device.data_ptr());
  } else if (dtype == torch::kFloat32) {
    const uint32_t user_workspace_size =
        tcuscan::get_workspace_size<float>(tiling);
    const at::Tensor workspace_tensor =
        alloc_zeros_workspace(user_workspace_size, device);
    launch_scan_single_core_fp32(
        1 /* single core*/, acl_stream, x_contig.data_ptr(),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device.data_ptr());
  } else {
    /* Unsupported */
  }

  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Returns the inclusive scan of `x` along dimension `dim`.
 *
 * The kernel consumes its input as a flat row-major `batch_size x vec_len`
 * buffer, so it needs the scanned dimension to be the fastest running one and
 * ignores both the strides and the storage offset of `x`. Any other layout - a
 * transposed view, a strided slice, or a `dim` other than the last one - is
 * brought into that form here, and the result is mapped back onto the original
 * dimension order.
 *
 * @param x Input tensor with at least one dimension.
 * @param S Tiling parameter. Typical values 32, 64, 128.
 * @param dim Dimension to scan over. Negative values count from the back.
 * @return The inclusive scan of `x` along `dim`, with the shape of `x`. The
 * element type is always 32-bits wide. Scanning a dimension other than the
 * last one yields a transposed view, which is not row-major; call
 * `contiguous()` on the result if a row-major layout is needed.
 */
at::Tensor run_scan_batch(const at::Tensor& x, int S, int64_t dim = -1) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out = dtype == torch::kHalf || dtype == torch::kFloat32
                             ? torch::kFloat32
                             : torch::kInt32;
  TORCH_CHECK(x.dim() >= 1,
              "run_scan_batch: x must have at least one dimension");
  TORCH_CHECK(x.numel() > 0, "run_scan_batch: x must not be empty");
  TORCH_CHECK(dtype == torch::kHalf || dtype == torch::kFloat32,
              "run_scan_batch: x must be fp16 or fp32, got ", dtype);

  const int64_t last_dim = x.dim() - 1;
  const int64_t scan_dim = dim < 0 ? dim + x.dim() : dim;
  TORCH_CHECK(scan_dim >= 0 && scan_dim <= last_dim, "run_scan_batch: dim ",
              dim, " is out of range for a ", x.dim(), "-dimensional tensor");
  const bool needs_transpose = scan_dim != last_dim;

  // Move the scanned dimension last and materialize it as the fastest running
  // one. `transpose()` only rewrites the strides and `contiguous()` is a no-op
  // on a tensor that already has this layout, so scanning the last dimension
  // of a row-major input stays copy-free.
  const at::Tensor x_work = needs_transpose
                                ? x.transpose(scan_dim, last_dim).contiguous()
                                : x.contiguous();

  const uint32_t matmul_size = static_cast<uint32_t>(S);

  const int64_t vec_len = x_work.size(-1);
  const int64_t batch_size = x_work.numel() / vec_len;

  // The kernel hands a pair of vectors to every cube/vector core pair and
  // walks over `batch_size / VEC_CUBE_RATIO` of those pairs, so a trailing odd
  // vector would be left untouched. Pad the batch with a zero vector and drop
  // it again from the result.
  constexpr int64_t VEC_CUBE_RATIO = 2;
  const int64_t padded_batch =
      (batch_size + VEC_CUBE_RATIO - 1) / VEC_CUBE_RATIO * VEC_CUBE_RATIO;

  at::Tensor x_in = x_work.reshape({batch_size, vec_len});
  if (padded_batch != batch_size) {
    const at::Tensor padded =
        at::zeros({padded_batch, vec_len}, x_work.options());
    padded.narrow(0, 0, batch_size).copy_(x_in);
    x_in = padded;
  }

  const at::Tensor z =
      at::empty({padded_batch, vec_len},
                at::TensorOptions().dtype(dtype_out).device(device));

  uint32_t block_dim = ascendc_platform->GetCoreNumAic();
  if (padded_batch < static_cast<int64_t>(block_dim)) {
    block_dim = static_cast<uint32_t>(padded_batch);
  }

  const ScanBatchTiling tiling{block_dim, static_cast<uint32_t>(vec_len),
                               static_cast<uint32_t>(padded_batch), matmul_size,
                               static_cast<uint32_t>(VEC_CUBE_RATIO)};
  const at::Tensor tiling_device =
      alloc_copy_tiling_tensor(tiling, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == torch::kHalf) {
    const uint32_t user_workspace_size =
        tcuscan::get_workspace_size<int16_t>(tiling);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);
    launch_scan_batch_fp16(block_dim, acl_stream, x_in.data_ptr(), z.data_ptr(),
                           const_cast<void*>(workspace_tensor.storage().data()),
                           tiling_device.data_ptr());
  } else if (dtype == torch::kFloat32) {
    const uint32_t user_workspace_size =
        tcuscan::get_workspace_size<float>(tiling);
    const at::Tensor workspace_tensor =
        alloc_zeros_workspace(user_workspace_size, device);
    launch_scan_batch_fp32(block_dim, acl_stream, x_in.data_ptr(), z.data_ptr(),
                           const_cast<void*>(workspace_tensor.storage().data()),
                           tiling_device.data_ptr());
  } else {
    /* Unsupported */
  }

  aclrtSynchronizeStream(acl_stream);

  // Drop the batch padding and restore the shape and dimension order of `x`.
  // All three steps are views, so the layout of the result follows from the
  // request rather than being forced to row-major behind the caller's back.
  const at::Tensor result = z.narrow(0, 0, batch_size).reshape(x_work.sizes());
  return needs_transpose ? result.transpose(scan_dim, last_dim) : result;
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
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out = dtype == torch::kHalf || dtype == torch::kFloat32
                             ? torch::kFloat32
                             : torch::kInt32;
  TORCH_CHECK(dtype == torch::kHalf || dtype == torch::kFloat32 ||
                  dtype == torch::kInt8,
              "run_scan_multi_core: x must be fp16, fp32 or int8, got ", dtype);

  // The kernel reads its input as a flat row-major buffer, ignoring both
  // strides and the storage offset, so non-contiguous views have to be
  // materialized first. `contiguous()` is a no-op when they already are.
  const at::Tensor x_contig = x.contiguous();

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
  const at::Tensor tiling_device =
      alloc_copy_tiling_tensor(tiling, device);

  if (dtype == torch::kHalf) {
    const uint32_t user_workspace_size =
        tcuscan::get_workspace_size<int16_t>(tiling);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);
    launch_scan_multi_core_fp16(
        block_dim, acl_stream, x_contig.data_ptr(),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device.data_ptr());
  } else if (dtype == torch::kFloat32) {
    const uint32_t user_workspace_size =
        tcuscan::get_workspace_size<float>(tiling);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);

    launch_scan_multi_core_fp32(
        block_dim, acl_stream, x_contig.data_ptr(),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device.data_ptr());
  } else {
    const uint32_t user_workspace_size =
        tcuscan::get_workspace_size<int8_t>(tiling);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);

    launch_scan_multi_core_int8(
        block_dim, acl_stream, x_contig.data_ptr(),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device.data_ptr());
  }

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

  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out = dtype == torch::kHalf || dtype == torch::kFloat32
                             ? torch::kFloat32
                             : torch::kInt32;
  TORCH_CHECK(dtype == torch::kHalf || dtype == torch::kFloat32 ||
                  dtype == torch::kInt8,
              "run_scan_multi_core_no_l2: x must be fp16, fp32 or int8, got ",
              dtype);

  // The kernel reads its input as a flat row-major buffer, ignoring both
  // strides and the storage offset, so non-contiguous views have to be
  // materialized first. `contiguous()` is a no-op when they already are.
  const at::Tensor x_contig = x.contiguous();

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
  const at::Tensor tiling_device =
      alloc_copy_tiling_tensor(tiling, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
  if (dtype == torch::kHalf) {
    const uint32_t user_workspace_size =
        tcuscan::get_workspace_size<int16_t>(tiling);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);
    launch_scan_multi_core_fp16_no_l2(
        block_dim, acl_stream, x_contig.data_ptr(),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device.data_ptr());
  } else if (dtype == torch::kFloat32) {
    const uint32_t user_workspace_size =
        tcuscan::get_workspace_size<float>(tiling);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);
    launch_scan_multi_core_fp32_no_l2(
        block_dim, acl_stream, x_contig.data_ptr(),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device.data_ptr());
  } else {
    const uint32_t user_workspace_size =
        tcuscan::get_workspace_size<int8_t>(tiling);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);

    launch_scan_multi_core_int8_no_l2(
        block_dim, acl_stream, x_contig.data_ptr(),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device.data_ptr());
  }

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

  // The kernel reads its input as a flat row-major buffer, ignoring both
  // strides and the storage offset, so non-contiguous views have to be
  // materialized first. `contiguous()` is a no-op when they already are.
  const at::Tensor x_contig = x.contiguous();

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
  const at::Tensor tiling_device =
      alloc_copy_tiling_tensor(tiling, device);

  if (dtype == torch::kHalf) {
    const at::Tensor workspace_tensor = alloc_workspace(0, device);
    launch_row_scan_fp16(block_dim, acl_stream, x_contig.data_ptr(),
                         const_cast<void*>(z.storage().data()),
                         const_cast<void*>(workspace_tensor.storage().data()),
                         tiling_device.data_ptr());
  } else {
    /* Unsupported*/
  }

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

  // The kernel reads its input as a flat row-major buffer, ignoring both
  // strides and the storage offset, so non-contiguous views have to be
  // materialized first. `contiguous()` is a no-op when they already are.
  const at::Tensor x_contig = x.contiguous();
  const at::Tensor upper_contig = upper.contiguous();
  const at::Tensor lower_contig = lower_strict.contiguous();

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
  const at::Tensor tiling_device =
      tcuscan::alloc_copy_tiling_tensor(tiling, device);

  if (dtype == torch::kHalf) {
    const at::Tensor workspace_tensor = tcuscan::alloc_workspace(0, device);
    launch_block_scan_fp16(
        block_dim, acl_stream, x_contig.data_ptr(), upper_contig.data_ptr(),
        lower_contig.data_ptr(), const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device.data_ptr());
  } else {
    /* Unsupported*/
  }

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

  // The kernel reads its input as a flat row-major buffer, ignoring both
  // strides and the storage offset, so non-contiguous views have to be
  // materialized first. `contiguous()` is a no-op when they already are.
  const at::Tensor x_contig = x.contiguous();
  const at::Tensor upper_contig = upper.contiguous();
  const at::Tensor lower_contig = lower_strict.contiguous();

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
  const at::Tensor tiling_device =
      tcuscan::alloc_copy_tiling_tensor(tiling, device);

  if (dtype == torch::kHalf) {
    const uint32_t user_workspace_size =
        tcuscan::get_workspace_size<int16_t>(tiling);
    const at::Tensor workspace_tensor =
        alloc_workspace(user_workspace_size, device);
    launch_scan_multi_cube_fp16(
        block_dim, acl_stream, x_contig.data_ptr(), upper_contig.data_ptr(),
        lower_contig.data_ptr(), const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device.data_ptr());
  } else {
    /* Unsupported*/
  }

  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Returns the prefix sum of an input 1D vector using one AI Core.
 *
 * @param x Input 1D vector.
 * @param upper Upper triangular matrix.
 * @param lower_strict Strictly lower triangular matrix.
 * @return The prefix sum of `x`.
 */
at::Tensor run_scan_single_cube(const at::Tensor& x, const at::Tensor& upper,
                                const at::Tensor& lower_strict) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out = dtype == torch::kHalf || dtype == torch::kFloat32
                             ? torch::kFloat32
                             : torch::kInt32;

  // The kernel reads its input as a flat row-major buffer, ignoring both
  // strides and the storage offset, so non-contiguous views have to be
  // materialized first. `contiguous()` is a no-op when they already are.
  const at::Tensor x_contig = x.contiguous();
  const at::Tensor upper_contig = upper.contiguous();
  const at::Tensor lower_contig = lower_strict.contiguous();

  const uint32_t matmul_size = static_cast<uint32_t>(upper.size(0));
  const uint32_t total_length = x.numel();

  // Outuput is always 32-bits (float or int32_t)
  const at::Tensor z = at::empty(
      {total_length}, at::TensorOptions().dtype(dtype_out).device(device));
  const ScanSingleCubeTiling tiling{total_length, matmul_size};

  const at::Tensor tiling_device =
      tcuscan::alloc_copy_tiling_tensor(tiling, device);

  const at::Tensor workspace_tensor = alloc_zeros_workspace(0, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

  if (dtype == torch::kHalf) {
    launch_scan_single_cube_fp16(
        1 /* single core*/, acl_stream, x_contig.data_ptr(),
        upper_contig.data_ptr(), lower_contig.data_ptr(),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device.data_ptr());
  } else {
    /* Unsupported */
  }

  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Returns the prefix sum of an input 1D vector using one AI Core.
 *
 * @param x Input 1D vector.
 * @param S Matrix tiling parameter. Typical values: 32, 64, 128.
 * @return The prefix sum of `x`.
 */
at::Tensor run_scan_vec_only(const at::Tensor& x, int S) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out = dtype == torch::kHalf || dtype == torch::kFloat32
                             ? torch::kFloat32
                             : torch::kInt32;

  // The kernel reads its input as a flat row-major buffer, ignoring both
  // strides and the storage offset, so non-contiguous views have to be
  // materialized first. `contiguous()` is a no-op when they already are.
  const at::Tensor x_contig = x.contiguous();

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t total_length = x.numel();

  // Outuput is always 32-bits (float or int32_t)
  const at::Tensor z = at::empty(
      {total_length}, at::TensorOptions().dtype(dtype_out).device(device));
  ScanVecOnlyTiling tiling;
  tiling.num_elems = total_length;
  tiling.tile_width = matmul_size;
  tiling.tile_height = matmul_size;

  const at::Tensor tiling_device =
      tcuscan::alloc_copy_tiling_tensor(tiling, device);

  const at::Tensor workspace_tensor = alloc_zeros_workspace(0, device);
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == torch::kHalf) {
    launch_scan_vec_only_fp16(
        1 /* single core*/, acl_stream, x_contig.data_ptr(),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device.data_ptr());
  } else {
    /* Unsupported */
  }

  aclrtSynchronizeStream(acl_stream);

  return z;
}

}  // namespace tcuscan
