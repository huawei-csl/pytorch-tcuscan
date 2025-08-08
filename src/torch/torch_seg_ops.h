/**
 * @file torch_seg_ops.h
 * @brief Torch wrapper for segmented operations.
 * @date 2025-03-27
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/tiling_seg_scan_mc_revert.h"
#include "../tiling/tiling_seg_scan_single_core.h"
#include "../tiling/tiling_seg_scan_vec_single_core.h"
#include "../tiling/tiling_seg_sum_single_core.h"
#include "aclrtlaunch_seg_scan_mc_revert.h"
#include "aclrtlaunch_seg_scan_single_core.h"
#include "aclrtlaunch_seg_scan_vec_single_core.h"
#include "aclrtlaunch_seg_sum_single_core_fp16.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "workspace.h"

namespace asc {

namespace seg_ops {

/**
 * @brief Segmented scan of input vector given a boolean flag vector
 * representation the segment starts. Uses only AIV cores.
 *
 * @param x Input data vector.
 * @param f Input boolean/mask flag vector. `f_i = 1` means a segment start at
 * index `i`.
 * @param S Tiling parameter. Typical values: 32, 64, 128.
 * @return Segmented scan of (x, f).
 */
at::Tensor run_seg_scan_vec(const at::Tensor &x, const at::Tensor &f, int S) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();

  const uint32_t tile_len = static_cast<uint32_t>(S);
  const uint32_t total_length = x.numel();

  const at::Tensor z = at::empty(
      {total_length}, at::TensorOptions().dtype(at::kFloat).device(device));

  const SegScanVecSingleCoreTiling tiling{total_length, tile_len};
  uint8_t *tiling_device = alloc_copy_tiling(tiling);

  const uint32_t user_workspace_size = 0;
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  ACLRT_LAUNCH_KERNEL(seg_scan_vec_single_core)
  (1 /* single core*/, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(f.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tiling_device);

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Segmented sum of input vector given a boolean flag vector
 * representation the segment starts.
 *
 * @param x Input data vector.
 * @param f Input boolean/mask flag vector. `f_i = 1` means a segment start at
 * index `i`.
 * @param S Tiling parameter. Typical values: 32, 64, 128.
 * @return Segmented sum of (x, f). Output length is `sum(f == 1)`.
 */
at::Tensor run_seg_sum(const at::Tensor &x, const at::Tensor &f, int S) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();

  const at::Tensor scan_x = scan::run_scan_multi_core(x, S);
  const at::Tensor out_positions = scan::run_scan_multi_core(f, S);
  const at::Tensor compress_scan_x =
      compress::run_compress_pos(scan_x, f, out_positions, S);

  const at::Tensor prepend =
      at::empty({1}, at::TensorOptions().dtype(at::kFloat).device(device))
          .zero_();
  aclrtSynchronizeStream(acl_stream);

  const at::Tensor prep_compress_scan_x =
      torch::cat({prepend, compress_scan_x});

  const at::Tensor z = torch::diff(prep_compress_scan_x);

  return z;
}

/**
 * @brief Internal undocumented method.
 *
 * @param x Input data vector.
 * @param f Input boolean/mask flag vector. `f_i = 1` means a segment start at
 * index `i`.
 * @param diff Vector differences.
 * @return Not documented.
 */
at::Tensor run_seg_scan_mc_revert(const at::Tensor &x, const at::Tensor &f,
                                  const at::Tensor &diff) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();

  const uint32_t tile_len = 4 * 1024;
  const uint32_t total_length = x.numel();

  uint32_t block_dim =
      static_cast<uint32_t>((total_length + tile_len - 1) / tile_len);
  block_dim = block_dim > 40 ? 40 : block_dim;

  const at::Tensor z = at::empty(
      {total_length}, at::TensorOptions().dtype(at::kFloat).device(device));

  const uint32_t diff_len = static_cast<uint32_t>(diff.numel());

  const SegScanMcRevertTiling tiling{block_dim, total_length, diff_len,
                                     tile_len};
  uint8_t *tiling_device = alloc_copy_tiling(tiling);

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  ACLRT_LAUNCH_KERNEL(seg_scan_mc_revert)
  (block_dim, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(f.storage().data()),
   const_cast<void *>(diff.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tiling_device);

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Segmented scan of input vector given a boolean flag vector
 * representation the segment starts.
 *
 * @param x Input data vector.
 * @param f Input boolean/mask flag vector. `f_i = 1` means a segment start at
 * index `i`.
 * @param S Tiling parameter. Typical values: 32, 64, 128.
 * @return Segmented scan of (x, f).
 */
at::Tensor run_seg_scan(const at::Tensor &x, const at::Tensor &f, int S) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t total_length = x.numel();

  const at::Tensor z = at::empty(
      {total_length}, at::TensorOptions().dtype(at::kFloat).device(device));

  const SegScanSingleCoreTiling tiling{total_length, matmul_size};
  uint8_t *tiling_device = alloc_copy_tiling(tiling);

  const uint32_t user_workspace_size =
      workspace::seg_scan::get_workspace_size<int16_t /* half */, int8_t>(
          total_length, matmul_size);
  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  ACLRT_LAUNCH_KERNEL(seg_scan_single_core)
  (1 /* single core*/, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(f.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tiling_device);

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Segmented sum single core
 *
 * @param x Input data vector.
 * @param indptr Input segment starts vector.
 * @param s Tiling parameter. Typical values: 32, 64, 128.
 * @return Segmented sum of (x, indptr).
 */
at::Tensor run_seg_sum_single_core(const at::Tensor &x,
                                   const at::Tensor &indptr, int s) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();

  constexpr uint32_t BLOCK_DIM = 1;  // single core
  const uint32_t matmul_size = static_cast<uint32_t>(s);
  const uint32_t total_length = x.numel();
  const uint32_t num_segments = indptr.numel();

  const at::Tensor z = at::empty(
      {num_segments}, at::TensorOptions().dtype(at::kFloat).device(device));

  const SegSumSingleCoreTiling tiling{total_length, num_segments, matmul_size};
  uint8_t *tiling_device = alloc_copy_tiling(tiling);

  const uint32_t user_workspace_size =
      workspace::seg_sum::get_workspace_size(tiling);

  const at::Tensor workspace_tensor =
      alloc_workspace(user_workspace_size, device);

  ACLRT_LAUNCH_KERNEL(seg_sum_single_core_fp16)
  (BLOCK_DIM, acl_stream, const_cast<void *>(x.storage().data()),
   const_cast<void *>(indptr.storage().data()),
   const_cast<void *>(z.storage().data()),
   const_cast<void *>(workspace_tensor.storage().data()), tiling_device);

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

}  // namespace seg_ops

}  // namespace asc
