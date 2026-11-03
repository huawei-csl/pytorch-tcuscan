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
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_compare.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_reduce.h"
#include "workspace.h"

// Kernel launchers; documented at their definitions in src/*.cpp
/// @cond
extern "C" void launch_compress_fp16(uint32_t blockDim, void* stream, void* x,
                                     void* mask, void* z, void* workspace,
                                     void* tiling_gm);
extern "C" void launch_compress_fp32(uint32_t blockDim, void* stream, void* x,
                                     void* mask, void* z, void* workspace,
                                     void* tiling_gm);
extern "C" void launch_compress_ind_fp16(uint32_t blockDim, void* stream,
                                         void* vec_in, void* indices_in,
                                         void* mask, void* vec_out,
                                         void* indices_out, void* workspace,
                                         void* tiling_gm);
extern "C" void launch_compress_ind_fp32(uint32_t blockDim, void* stream,
                                         void* vec_in, void* indices_in,
                                         void* mask, void* vec_out,
                                         void* indices_out, void* workspace,
                                         void* tiling_gm);
extern "C" void launch_compress_ind_no_arange_fp16(
    uint32_t blockDim, void* stream, void* vec_in, void* mask, void* vec_out,
    void* indices_out, void* workspace, void* tiling_gm);
extern "C" void launch_compress_ind_no_arange_fp32(
    uint32_t blockDim, void* stream, void* vec_in, void* mask, void* vec_out,
    void* indices_out, void* workspace, void* tiling_gm);
extern "C" void launch_where_fp16(uint32_t blockDim, void* stream,
                                  void* mask_in, void* vec_out, void* workspace,
                                  void* tiling_gm);
/// @endcond

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

  // The output is as long as there are ones in `mask`, which only the device
  // knows. Allocating that exactly would mean reducing the mask on the host
  // first and reading the total back, a blocking synchronization that stalls
  // the stream before the kernel is even launched. Allocate for the worst
  // case instead -- every element selected -- and return a view of the part
  // that was written; the kernel reduces the mask itself.
  const at::Tensor z = at::empty(
      {total_length}, at::TensorOptions().dtype(dtype).device(device));

  const CompressTiling tiling{block_dim, total_length, matmul_size};
  const at::Tensor tiling_device = alloc_copy_tiling_tensor(tiling, device);

  const uint32_t user_workspace_size = tcuscan::get_workspace_size(tiling);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == torch::kHalf or dtype == torch::kInt16) {
    launch_compress_fp16(block_dim, acl_stream,
                         const_cast<void*>(x.storage().data()),
                         const_cast<void*>(mask.storage().data()),
                         const_cast<void*>(z.storage().data()),
                         const_cast<void*>(workspace_tensor.storage().data()),
                         tiling_device.data_ptr());
  } else {
    launch_compress_fp32(block_dim, acl_stream,
                         const_cast<void*>(x.storage().data()),
                         const_cast<void*>(mask.storage().data()),
                         const_cast<void*>(z.storage().data()),
                         const_cast<void*>(workspace_tensor.storage().data()),
                         tiling_device.data_ptr());
  }

  aclrtSynchronizeStream(acl_stream);

  // The kernel's own reduction left one count of ones per vector core at the
  // start of the workspace -- the same counts it turned into output offsets --
  // so the output length comes back without a second pass over the mask.
  const int32_t num_ones =
      workspace_block_counts(workspace_tensor, block_dim).sum().item<int32_t>();

  return z.slice(0, 0, num_ones);
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
  const at::Tensor tiling_device = alloc_copy_tiling_tensor(tiling, device);

  const uint32_t user_workspace_size = tcuscan::get_workspace_size(tiling);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == torch::kHalf or dtype == torch::kInt16) {
    launch_compress_fp16(block_dim, acl_stream,
                         const_cast<void*>(x.storage().data()),
                         const_cast<void*>(mask.storage().data()),
                         const_cast<void*>(z.storage().data()),
                         const_cast<void*>(workspace_tensor.storage().data()),
                         tiling_device.data_ptr());
  } else {
    launch_compress_fp32(block_dim, acl_stream,
                         const_cast<void*>(x.storage().data()),
                         const_cast<void*>(mask.storage().data()),
                         const_cast<void*>(z.storage().data()),
                         const_cast<void*>(workspace_tensor.storage().data()),
                         tiling_device.data_ptr());
  }

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

  // Both outputs are as long as there are ones in `mask`, a length only the
  // device knows; see `run_compress` for why they are allocated for the worst
  // case and sliced afterwards rather than sized up front.
  const at::Tensor z = at::empty(
      {total_length}, at::TensorOptions().dtype(dtype).device(device));

  const at::Tensor indices_out = at::empty(
      {total_length}, at::TensorOptions().dtype(indices_dtype).device(device));

  const CompressTiling tiling{block_dim, total_length, matmul_size};
  const at::Tensor tiling_device = alloc_copy_tiling_tensor(tiling, device);

  const uint32_t user_workspace_size = tcuscan::get_workspace_size(tiling);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == torch::kHalf or dtype == torch::kInt16) {
    launch_compress_ind_fp16(
        block_dim, acl_stream, const_cast<void*>(x.storage().data()),
        const_cast<void*>(indices_in.storage().data()),
        const_cast<void*>(mask.storage().data()),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(indices_out.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()),
        tiling_device.data_ptr());
  } else {
    launch_compress_ind_fp32(
        block_dim, acl_stream, const_cast<void*>(x.storage().data()),
        const_cast<void*>(indices_in.storage().data()),
        const_cast<void*>(mask.storage().data()),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(indices_out.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()),
        tiling_device.data_ptr());
  }

  aclrtSynchronizeStream(acl_stream);

  const int32_t num_ones =
      workspace_block_counts(workspace_tensor, block_dim).sum().item<int32_t>();

  return std::make_tuple(z.slice(0, 0, num_ones),
                         indices_out.slice(0, 0, num_ones));
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

  // Both outputs are as long as there are ones in `mask`, a length only the
  // device knows; see `run_compress` for why they are allocated for the worst
  // case and sliced afterwards rather than sized up front.
  const at::Tensor z = at::empty(
      {total_length}, at::TensorOptions().dtype(dtype).device(device));

  const at::Tensor indices_out = at::empty(
      {total_length}, at::TensorOptions().dtype(torch::kInt32).device(device));

  const CompressTiling tiling{block_dim, total_length, matmul_size};
  const at::Tensor tiling_device = alloc_copy_tiling_tensor(tiling, device);

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == torch::kHalf or dtype == torch::kInt16) {
    launch_compress_ind_no_arange_fp16(
        block_dim, acl_stream, const_cast<void*>(x.storage().data()),
        const_cast<void*>(mask.storage().data()),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(indices_out.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()),
        tiling_device.data_ptr());
  } else {
    launch_compress_ind_no_arange_fp32(
        block_dim, acl_stream, const_cast<void*>(x.storage().data()),
        const_cast<void*>(mask.storage().data()),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(indices_out.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()),
        tiling_device.data_ptr());
  }

  aclrtSynchronizeStream(acl_stream);

  const int32_t num_ones =
      workspace_block_counts(workspace_tensor, block_dim).sum().item<int32_t>();

  return std::make_tuple(z.slice(0, 0, num_ones),
                         indices_out.slice(0, 0, num_ones));
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
  const auto dtype = x.options().dtype();
  at::Tensor mask;
  if (dtype == at::kHalf) {
    mask = run_less_equal(x, pivot, static_cast<uint32_t>(S * S));
  } else {
    // For non-fp16 dtypes the PyTorch op runs on the NPU stream; synchronize
    // before launching ACLRT kernels in run_compress to avoid a data race on
    // the mask.
    mask = (x <= pivot).to(torch::kInt8);
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
    aclrtSynchronizeStream(acl_stream);

    aclrtSynchronizeStream(acl_stream);
  }
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
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  // The output length now comes out of the workspace the kernel filled in, so
  // there is no longer a sensible value to return when no kernel runs: the
  // unsupported dtype has to be refused rather than silently answered.
  TORCH_CHECK(dtype == at::kHalf,
              "run_where: only float16 input is supported, but got ",
              x.scalar_type(), ".");

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

  // One index per mask one, a count only the device knows; see `run_compress`
  // for why the output is allocated for the worst case and sliced afterwards
  // rather than sized up front.
  const at::Tensor z = at::empty(
      {total_length}, at::TensorOptions().dtype(torch::kInt32).device(device));

  const WhereTiling tiling{block_dim, total_len, vec_tile_len, pivot};
  const at::Tensor tiling_device = alloc_copy_tiling_tensor(tiling, device);

  const uint32_t user_workspace_size = tcuscan::get_workspace_size(tiling);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  launch_where_fp16(block_dim, acl_stream,
                    const_cast<void*>(mask.storage().data()),
                    const_cast<void*>(z.storage().data()),
                    const_cast<void*>(workspace_tensor.storage().data()),
                    tiling_device.data_ptr());

  aclrtSynchronizeStream(acl_stream);

  const int32_t num_ones =
      workspace_block_counts(workspace_tensor, block_dim).sum().item<int32_t>();

  return z.slice(0, 0, num_ones);
}

}  // namespace tcuscan
