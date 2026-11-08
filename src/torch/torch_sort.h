/**
 * @file torch_sort.h
 * @brief Torch wrapper for sorting.
 * @date 2025-03-27
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/heuristics/heuristics_radix_sort.h"
#include "../tiling/tiling_radix_sort.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "workspace.h"

// Kernel launchers; documented at their definitions in src/*.cpp
/// @cond
extern "C" void launch_radix_sort_fp16(uint32_t blockDim, void* stream,
                                       void* in, void* out, void* indices,
                                       void* workspace, void* tiling);
extern "C" void launch_radix_sort_fp32(uint32_t blockDim, void* stream,
                                       void* in, void* out, void* indices,
                                       void* workspace, void* tiling);
extern "C" void launch_radix_sort_int16(uint32_t blockDim, void* stream,
                                        void* in, void* out, void* indices,
                                        void* workspace, void* tiling);
extern "C" void launch_radix_sort_int32(uint32_t blockDim, void* stream,
                                        void* in, void* out, void* indices,
                                        void* workspace, void* tiling);
/// @endcond

namespace tcuscan {

/**
 * @brief Radix sort on 1D vector with indices.
 *
 * Supported data types are `float16`, `float32`, `int16` and `int32`.
 *
 * @param x Input 1D vector.
 * @param S Tiling parameter. Typical values: 32,64, 128.
 * @return Returns 2-tuple of sorted values of `x` along with its indices.
 */
std::tuple<at::Tensor, at::Tensor> run_radix_sort(const at::Tensor& x, int S) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  TORCH_CHECK(dtype == torch::kHalf || dtype == torch::kFloat32 ||
                  dtype == torch::kInt16 || dtype == torch::kInt32,
              "run_radix_sort: unsupported dtype ", dtype,
              "; expected float16, float32, int16 or int32.");

  const uint32_t total_length = x.numel();
  const uint32_t matmul_size = static_cast<uint32_t>(S);

  const uint32_t tile_elems = matmul_size * matmul_size;
  const size_t num_tiles = total_length / tile_elems;

  // The kernels distribute tiles with `scalar::GetWorkDistribution()`, so
  // `num_tiles` does not have to be divisible by the number of blocks.
  // Requiring divisibility would drop the block count down to 1 whenever
  // `num_tiles` is a prime number.
  // The radix sort kernels are AIV-only, so `blockDim` counts vector cores.
  uint32_t block_dim = ascendc_platform->GetCoreNumAiv();
  if (num_tiles < block_dim) {
    block_dim = num_tiles;
  }
  if (block_dim <= 1) {
    block_dim = 1;
  }

  const at::Tensor vec_out = at::empty(
      {total_length}, at::TensorOptions().dtype(dtype).device(device));
  const at::Tensor indices_out = at::empty(
      {total_length}, at::TensorOptions().dtype(torch::kInt32).device(device));

  // The split keeps about 28 bytes of UB per element of a tile, so a 4-byte
  // key has to halve the tile a 2-byte one uses to stay inside the 192 KB.
  const uint32_t elem_size = static_cast<uint32_t>(x.element_size());
  const uint32_t vec_tile_size = tile_elems / (elem_size == 2 ? 2 : 4);

  RadixSortTiling tiling{block_dim, total_length, matmul_size, vec_tile_size};
  const at::Tensor tiling_device =
      tcuscan::alloc_copy_tiling_tensor(tiling, device);

  const uint32_t user_workspace_size =
      elem_size == 2 ? tcuscan::get_workspace_size<int16_t>(tiling)
                     : tcuscan::get_workspace_size<int32_t>(tiling);
  const at::Tensor workspace_tensor =
      tcuscan::alloc_workspace(user_workspace_size, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  // `data_ptr()` rather than `storage().data()`, so that a view with a
  // non-zero storage offset is read from the right place.
  const at::Tensor x_contig = x.contiguous();
  void* const in_ptr = x_contig.data_ptr();
  void* const out_ptr = vec_out.data_ptr();
  void* const indices_ptr = indices_out.data_ptr();
  void* const ws_ptr = workspace_tensor.data_ptr();

  if (dtype == torch::kHalf) {
    launch_radix_sort_fp16(block_dim, acl_stream, in_ptr, out_ptr, indices_ptr,
                           ws_ptr, tiling_device.data_ptr());
  } else if (dtype == torch::kFloat32) {
    launch_radix_sort_fp32(block_dim, acl_stream, in_ptr, out_ptr, indices_ptr,
                           ws_ptr, tiling_device.data_ptr());
  } else if (dtype == torch::kInt16) {
    launch_radix_sort_int16(block_dim, acl_stream, in_ptr, out_ptr, indices_ptr,
                            ws_ptr, tiling_device.data_ptr());
  } else {
    launch_radix_sort_int32(block_dim, acl_stream, in_ptr, out_ptr, indices_ptr,
                            ws_ptr, tiling_device.data_ptr());
  }

  aclrtSynchronizeStream(acl_stream);

  return std::make_tuple(vec_out, indices_out);
}

}  // namespace tcuscan
