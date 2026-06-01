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
#include "../tiling/tiling_seg_sum_multi_core.h"
#include "../tiling/tiling_seg_sum_single_core.h"
#include "../tiling/tiling_seg_sum_single_cube.h"
#include "aclrtlaunch_seg_scan_mc_revert.h"
#include "aclrtlaunch_seg_scan_single_core.h"
#include "aclrtlaunch_seg_scan_vec_single_core.h"
#include "aclrtlaunch_seg_sum_multi_core_fp16.h"
#include "aclrtlaunch_seg_sum_single_core_fp16.h"
#include "aclrtlaunch_seg_sum_single_core_int8.h"
#include "aclrtlaunch_seg_sum_single_cube_fp16.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "workspace.h"

namespace tcuscan {

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
at::Tensor run_seg_scan_vec(const at::Tensor& x, const at::Tensor& f, int S) {
  const at::Device device = x.options().device();

  const uint32_t tile_len = static_cast<uint32_t>(S);
  const uint32_t total_length = x.numel();

  const at::Tensor z = at::empty(
      {total_length}, at::TensorOptions().dtype(at::kFloat).device(device));

  const tcuscan::SegScanVecSingleCoreTiling tiling{total_length, tile_len};
  uint8_t* tiling_device = tcuscan::alloc_copy_tiling(tiling);

  const uint32_t user_workspace_size = 0;
  const at::Tensor workspace_tensor =
      tcuscan::alloc_workspace(user_workspace_size, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  ACLRT_LAUNCH_KERNEL(seg_scan_vec_single_core)
  (1 /* single core*/, acl_stream, const_cast<void*>(x.storage().data()),
   const_cast<void*>(f.storage().data()), const_cast<void*>(z.storage().data()),
   const_cast<void*>(workspace_tensor.storage().data()), tiling_device);

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
at::Tensor run_seg_sum(const at::Tensor& x, const at::Tensor& f, int S) {
  using tcuscan::run_compress_pos;
  using tcuscan::run_scan_multi_core;

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();

  const at::Tensor scan_x = run_scan_multi_core(x, S);
  const int64_t sum_f = torch::sum(f).item<int64_t>();
  const at::Tensor compress_scan_x =
      run_compress_pos(scan_x, f, static_cast<uint32_t>(sum_f), S);

  const at::Tensor prepend =
      at::zeros({1}, at::TensorOptions().dtype(at::kFloat).device(device));
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
at::Tensor run_seg_scan_mc_revert(const at::Tensor& x, const at::Tensor& f,
                                  const at::Tensor& diff) {
  const at::Device device = x.options().device();

  const uint32_t tile_len = 4 * 1024;
  const uint32_t total_length = x.numel();

  uint32_t block_dim =
      static_cast<uint32_t>((total_length + tile_len - 1) / tile_len);
  block_dim = block_dim > 40 ? 40 : block_dim;

  const at::Tensor z = at::empty(
      {total_length}, at::TensorOptions().dtype(at::kFloat).device(device));

  const uint32_t diff_len = static_cast<uint32_t>(diff.numel());

  const tcuscan::SegScanMcRevertTiling tiling{block_dim, total_length, diff_len,
                                              tile_len};
  uint8_t* tiling_device = tcuscan::alloc_copy_tiling(tiling);

  const at::Tensor workspace_tensor = tcuscan::alloc_workspace(0, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  ACLRT_LAUNCH_KERNEL(seg_scan_mc_revert)
  (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
   const_cast<void*>(f.storage().data()),
   const_cast<void*>(diff.storage().data()),
   const_cast<void*>(z.storage().data()),
   const_cast<void*>(workspace_tensor.storage().data()), tiling_device);

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
at::Tensor run_seg_scan(const at::Tensor& x, const at::Tensor& f, int S) {
  const at::Device device = x.options().device();

  const uint32_t matmul_size = static_cast<uint32_t>(S);
  const uint32_t total_length = x.numel();

  const at::Tensor z = at::empty(
      {total_length}, at::TensorOptions().dtype(at::kFloat).device(device));

  const tcuscan::SegScanSingleCoreTiling tiling{total_length, matmul_size};
  uint8_t* tiling_device = tcuscan::alloc_copy_tiling(tiling);

  const uint32_t user_workspace_size =
      tcuscan::get_workspace_size<int16_t /* half */, int8_t>(tiling);
  const at::Tensor workspace_tensor =
      tcuscan::alloc_workspace(user_workspace_size, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  ACLRT_LAUNCH_KERNEL(seg_scan_single_core)
  (1 /* single core*/, acl_stream, const_cast<void*>(x.storage().data()),
   const_cast<void*>(f.storage().data()), const_cast<void*>(z.storage().data()),
   const_cast<void*>(workspace_tensor.storage().data()), tiling_device);

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Segmented sum single core.
 *
 * Given an input data vector `x` and an segment index vector `indptr`, returns
 * the segmented sum of `(x, indptr)`.
 *
 * The segments in the `indptr` follows the `scipy.sparse.csr_matrix`
 * notation with the following adjustment: `indptr` does not start with `0` and
 * does not contain as last entry the value `len(x)`.
 *
 * @param [in] x Input data vector.
 * @param [in] indptr Input segment ending indices.
 * @param [in] s Tiling parameter. Typical values: 32, 64, 128.
 * @return Segmented sum. Output length is number of semgents and equals to
 * `len(indptr) + 1`
 */
at::Tensor run_seg_sum_single_core(const at::Tensor& x,
                                   const at::Tensor& indptr, int s) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out =
      dtype == torch::kHalf ? torch::kFloat32 : torch::kInt32;

  constexpr uint32_t BLOCK_DIM = 1;  // single core
  const uint32_t matmul_size = static_cast<uint32_t>(s);
  const uint32_t total_length = x.numel();
  // The indptr does not contain zero as first entry and len(x) as last entry.
  // Hence, the number of segments are +1.
  const uint32_t num_segments = indptr.numel() - 1;

  const at::Tensor z = at::empty(
      {num_segments}, at::TensorOptions().dtype(dtype_out).device(device));

  const tcuscan::SegSumSingleCoreTiling tiling{total_length, num_segments,
                                               matmul_size};
  uint8_t* tiling_device = tcuscan::alloc_copy_tiling(tiling);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  // Offset indptr by one element, since first element is always zero.
  void* indptr_data = static_cast<void*>(
      static_cast<uint8_t*>(const_cast<void*>(indptr.storage().data())) +
      indptr.element_size());

  if (dtype == torch::kHalf) {
    const uint32_t user_workspace_size =
        tcuscan::get_workspace_size<int16_t /* half */>(tiling);

    const at::Tensor workspace_tensor =
        tcuscan::alloc_workspace(user_workspace_size, device);

    ACLRT_LAUNCH_KERNEL(seg_sum_single_core_fp16)
    (BLOCK_DIM, acl_stream, const_cast<void*>(x.storage().data()), indptr_data,
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else if (dtype == torch::kInt8) {
    const uint32_t user_workspace_size =
        tcuscan::get_workspace_size<int8_t>(tiling);

    const at::Tensor workspace_tensor =
        tcuscan::alloc_workspace(user_workspace_size, device);
    ACLRT_LAUNCH_KERNEL(seg_sum_single_core_int8)
    (BLOCK_DIM, acl_stream, const_cast<void*>(x.storage().data()), indptr_data,
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Segmented sum single cube
 *
 * @param [in] x Input data vector.
 * @param [in] upper Upper triangular all-ones matrix of size S.
 * @param [in] lower_strict Strict lower triangular all-ones matrix of size S.
 * @param [in] indptr Input segment starts vector.
 * @param [in] s Tiling parameter. Typical values: 32, 64, 128.
 * @return Segmented sum of (x, indptr).
 */
at::Tensor run_seg_sum_single_cube(const at::Tensor& x, const at::Tensor& upper,
                                   const at::Tensor& lower_strict,
                                   const at::Tensor& indptr, int s) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out =
      dtype == torch::kHalf ? torch::kFloat32 : torch::kInt32;

  constexpr uint32_t BLOCK_DIM = 1;  // single core
  const uint32_t matmul_size = static_cast<uint32_t>(s);
  const uint32_t total_length = x.numel();
  const uint32_t num_segments = indptr.numel();

  const at::Tensor z = at::empty(
      {num_segments}, at::TensorOptions().dtype(dtype_out).device(device));

  const tcuscan::SegSumSingleCubeTiling tiling{total_length, num_segments,
                                               matmul_size};
  uint8_t* tiling_device = tcuscan::alloc_copy_tiling(tiling);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == torch::kHalf) {
    const uint32_t user_workspace_size =
        tcuscan::get_workspace_size<int16_t /* half */>(tiling);

    const at::Tensor workspace_tensor =
        tcuscan::alloc_workspace(user_workspace_size, device);

    ACLRT_LAUNCH_KERNEL(seg_sum_single_cube_fp16)
    (BLOCK_DIM, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(upper.storage().data()),
     const_cast<void*>(lower_strict.storage().data()),
     const_cast<void*>(indptr.storage().data()),
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
 * @brief Multi-core segmented sum
 *
 * @param [in] x Input data vector.
 * @param [in] indptr Input segment starts vector.
 * @param [in] s Tiling parameter. Typical values: 32, 64, 128.
 * @param [in] block_dim Number of blocks.
 * @return Segmented sum of (x, indptr).
 */
at::Tensor run_seg_sum_multi_core(const at::Tensor& x, const at::Tensor& indptr,
                                  int s, int block_dim) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out =
      dtype == torch::kHalf ? torch::kFloat32 : torch::kInt32;

  const uint32_t matmul_size = static_cast<uint32_t>(s);
  const uint32_t total_length = x.numel();
  const uint32_t num_segments = indptr.numel();

  const uint32_t ell = matmul_size * matmul_size;

  const at::Tensor sstart =
      torch::arange(0, std::min(block_dim * ell, total_length), ell,
                    at::TensorOptions().dtype(torch::kInt32).device(device));
  const at::Tensor bstart =
      torch::searchsorted(indptr, sstart, /*right=*/false);

  const at::Tensor z = at::empty(
      {num_segments}, at::TensorOptions().dtype(dtype_out).device(device));

  const tcuscan::SegSumSingleCoreTiling single_core_tiling{ell, num_segments,
                                                           matmul_size};

  const uint32_t singe_core_ws_size =
      tcuscan::get_workspace_size<int16_t /* half */>(single_core_tiling);

  const tcuscan::SegSumMultiCoreTiling tiling{total_length, num_segments,
                                              matmul_size, ell};
  uint8_t* tiling_device = tcuscan::alloc_copy_tiling(tiling);

  // Workspace is duplicated across AI cores, hence multiplied by block_dim
  const at::Tensor workspace_tensor =
      tcuscan::alloc_workspace(block_dim * singe_core_ws_size, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == torch::kHalf) {
    ACLRT_LAUNCH_KERNEL(seg_sum_multi_core_fp16)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(indptr.storage().data()),
     const_cast<void*>(bstart.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

}  // namespace tcuscan
