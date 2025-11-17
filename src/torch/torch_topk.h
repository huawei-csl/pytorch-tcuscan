/**
 * @file torch_topk.h
 * @brief Torch wrapper for TopK pivot.
 * @date 2025-11-15
 *
 * @copyright Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights
 * reserved.
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/tiling_topk.h"
#include "../tiling/tiling_topk_pivot.h"
#include "aclrtlaunch_topk_fp16.h"
#include "aclrtlaunch_topk_int16.h"
#include "aclrtlaunch_topk_pivot_fp16.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "workspace.h"

namespace asc {

namespace topk {

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
std::tuple<at::Tensor, at::Tensor> run_topk_int16(const at::Tensor& x,
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

  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  ACLRT_LAUNCH_KERNEL(topk_int16)
  (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
   const_cast<void*>(vec_out.storage().data()),
   const_cast<void*>(indices_out.storage().data()),
   const_cast<void*>(workspace_tensor.storage().data()), tiling_device);

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
std::tuple<at::Tensor, at::Tensor> run_topk_fp16(const at::Tensor& x,
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

  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  ACLRT_LAUNCH_KERNEL(topk_fp16)
  (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
   const_cast<void*>(vec_out.storage().data()),
   const_cast<void*>(indices_out.storage().data()),
   const_cast<void*>(workspace_tensor.storage().data()), tiling_device);

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return std::make_tuple(vec_out, indices_out);
}

/**
 * @brief K-largest value estimator from an input vector of dtype fp16.
 *
 * @param x Input 1D tensor
 * @param k Input parameter k (of top-k)
 * @return Returns Tensor of length 16, where second element is estimator
 */
at::Tensor run_topk_pivot_fp16(const at::Tensor& x, uint32_t k) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t total_length = x.numel();
  const uint32_t tile_len = 128;
  const uint32_t num_tiles = host_utils::CeilDiv(total_length, tile_len);

  uint32_t block_dim = ascendc_platform->GetCoreNumAic();
  if (num_tiles < block_dim) {
    block_dim = num_tiles;
  }

  const at::Tensor vec_out =
      at::empty({16}, at::TensorOptions().dtype(dtype).device(device));

  const uint32_t user_workspace_size = 0;
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  const TopKPivotTiling tiling{block_dim, total_length, tile_len, k};

  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  ACLRT_LAUNCH_KERNEL(topk_pivot_fp16)
  (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
   const_cast<void*>(vec_out.storage().data()),
   const_cast<void*>(workspace_tensor.storage().data()), tiling_device);

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return vec_out;
}

}  // namespace topk

}  // namespace asc
