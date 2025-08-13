/**
 * @file torch_split.h
 * @brief Torch wrapper for parallel split.
 * @date 2025-03-27
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/tiling_split.h"
#include "../tiling/tiling_topk.h"
#include "aclrtlaunch_split_ind_uint16.h"
#include "aclrtlaunch_split_uint16.h"
#include "aclrtlaunch_topk_fp16.h"
#include "aclrtlaunch_topk_int16.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "workspace.h"

namespace asc {

namespace split {

/**
 * @brief Returns the binary split with indices of x given a boolean mask
 * `mask`.
 *
 * @param x Input data vector.
 * @param mask Input boolean mask to use for splitting.
 * @param S Matrix tile parameter. Typical values 32,64,128.
 * @return Splitted input data vector.
 */
at::Tensor run_split(const at::Tensor &x, const at::Tensor &mask, int S) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t total_length = x.numel();

  const uint32_t tile_elems = matmul_size * matmul_size;
  const uint32_t vec_tile_size = tile_elems / 2;

  const uint32_t num_tiles = total_length / tile_elems;

  uint32_t block_dim = ascendc_platform->GetCoreNum() / 2;
  while (num_tiles % block_dim != 0) {
    block_dim--;
  }
  if (block_dim <= 1) {
    block_dim = 1;
  }

  const at::Tensor z = at::empty(
      {total_length}, at::TensorOptions().dtype(dtype).device(device));

  const SplitTiling tiling{block_dim, total_length, matmul_size, vec_tile_size};
  uint8_t *tiling_device = alloc_copy_tiling(tiling);

  const uint32_t user_workspace_size =
      workspace::split::get_workspace_size(tiling);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  if (dtype == torch::kHalf or dtype == torch::kInt16) {
    ACLRT_LAUNCH_KERNEL(split_uint16)
    (block_dim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(mask.storage().data()),
     const_cast<void *>(z.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Returns the binary split with indices of x given a boolean mask
 * `mask`. It permutes the input `indices_in` accordingly.
 *
 * @param x Input data vector.
 * @param mask Input boolean mask to use for splitting.
 * @param indices_in Input indices that is a permuation of [0,1,..., len(x)].
 * @param S Matrix tile parameter. Typical values 32,64,128.
 * @return Tuple of splitted data and indices.
 */
std::tuple<at::Tensor, at::Tensor> run_split_ind(const at::Tensor &x,
                                                 const at::Tensor &mask,
                                                 const at::Tensor &indices_in,
                                                 int S) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t total_length = x.numel();

  const uint32_t tile_elems = matmul_size * matmul_size;
  const uint32_t vec_tile_size = tile_elems / 2;

  const uint32_t num_tiles = total_length / tile_elems;

  uint32_t block_dim = ascendc_platform->GetCoreNum() / 2;
  while (num_tiles % block_dim != 0) {
    block_dim--;
  }
  if (block_dim <= 1) {
    block_dim = 1;
  }

  const at::Tensor vec_out = at::empty(
      {total_length}, at::TensorOptions().dtype(dtype).device(device));
  const at::Tensor indices_out = at::empty(
      {total_length}, at::TensorOptions().dtype(torch::kInt32).device(device));

  const SplitTiling tiling{block_dim, total_length, matmul_size, vec_tile_size};
  uint8_t *tiling_device = alloc_copy_tiling(tiling);

  const uint32_t user_workspace_size =
      workspace::split::get_workspace_size(tiling);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  if (dtype == torch::kHalf or dtype == torch::kInt16) {
    ACLRT_LAUNCH_KERNEL(split_ind_uint16)
    (block_dim, acl_stream, const_cast<void *>(x.storage().data()),
     const_cast<void *>(mask.storage().data()),
     const_cast<void *>(indices_in.storage().data()),
     const_cast<void *>(vec_out.storage().data()),
     const_cast<void *>(indices_out.storage().data()),
     const_cast<void *>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return std::make_tuple(vec_out, indices_out);
}

/**
 * @brief Top-K elements from an input vector of dtype int16.
 *
 * @param x Input 1D tensor
 * @param k Number of largest elements to return.
 * @param x_min Lower bound of all elements of `x`.
 * @param x_max Upper bound of all elements of `x`.
 * @param S Tiling parameter related to matrix multiplication.
 * @return Returns Tuple2. First tensor is top-K values and second the indices.
 */
std::tuple<at::Tensor, at::Tensor> run_topk_int16(const at::Tensor &x,
                                                  uint32_t k, int16_t x_min,
                                                  int16_t x_max, int S) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t total_length = x.numel();

  const uint32_t tile_elems = matmul_size * matmul_size;
  const uint32_t vec_tile_size = tile_elems / 2;

  const uint32_t num_tiles = total_length / tile_elems;

  uint32_t block_dim = ascendc_platform->GetCoreNum() / 2;
  while (num_tiles % block_dim != 0) {
    block_dim--;
  }
  if (block_dim <= 1) {
    block_dim = 1;
  }

  const at::Tensor vec_out =
      at::empty({k}, at::TensorOptions().dtype(dtype).device(device));
  const at::Tensor indices_out =
      at::empty({k}, at::TensorOptions().dtype(torch::kInt32).device(device));

  const uint32_t user_workspace_size =
      workspace::topk::get_workspace_size<int32_t>(total_length, matmul_size,
                                                   block_dim);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  TopKTiling tiling;
  tiling.num_elems = total_length;
  tiling.matmul_size = matmul_size;
  tiling.num_blocks = block_dim;
  tiling.vec_tile_size = vec_tile_size;
  tiling.x_min.value_i32 = static_cast<int32_t>(x_min);
  tiling.x_max.value_i32 = static_cast<int32_t>(x_max);
  tiling.k = k;

  uint8_t *tiling_device = alloc_copy_tiling(tiling);

  ACLRT_LAUNCH_KERNEL(topk_int16)
  (block_dim, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(vec_out.storage().data()),
   const_cast<void *>(indices_out.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tiling_device);

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return std::make_tuple(vec_out, indices_out);
}

/**
 * @brief Top-K elements from an input vector of dtype fp16.
 *
 * @param x Input 1D tensor
 * @param k Number of largest elements to return.
 * @param x_min Lower bound of all elements of `x`.
 * @param x_max Upper bound of all elements of `x`.
 * @param S Tiling parameter related to matrix multiplication.
 * @return Returns Tuple2. First tensor is top-K values and second the indices.
 */
std::tuple<at::Tensor, at::Tensor> run_topk_fp16(const at::Tensor &x,
                                                 uint32_t k, float x_min,
                                                 float x_max, int S) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t total_length = x.numel();

  const uint32_t tile_elems = matmul_size * matmul_size;
  const uint32_t vec_tile_size = tile_elems / 2;

  const uint32_t num_tiles = total_length / tile_elems;

  uint32_t block_dim = ascendc_platform->GetCoreNumAic();
  while (num_tiles % block_dim != 0) {
    block_dim--;
  }
  if (block_dim <= 1) {
    block_dim = 1;
  }

  const at::Tensor vec_out =
      at::empty({k}, at::TensorOptions().dtype(dtype).device(device));
  const at::Tensor indices_out =
      at::empty({k}, at::TensorOptions().dtype(torch::kInt32).device(device));

  const uint32_t user_workspace_size =
      workspace::topk::get_workspace_size<int32_t>(total_length, matmul_size,
                                                   block_dim);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  TopKTiling tiling;
  tiling.num_elems = total_length;
  tiling.matmul_size = matmul_size;
  tiling.num_blocks = block_dim;
  tiling.vec_tile_size = vec_tile_size;
  tiling.x_min.value_fp32 = x_min;
  tiling.x_max.value_fp32 = x_max;
  tiling.k = k;

  uint8_t *tiling_device = alloc_copy_tiling(tiling);

  ACLRT_LAUNCH_KERNEL(topk_fp16)
  (block_dim, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(vec_out.storage().data()),
   const_cast<void *>(indices_out.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tiling_device);

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return std::make_tuple(vec_out, indices_out);
}

}  // namespace split

}  // namespace asc
