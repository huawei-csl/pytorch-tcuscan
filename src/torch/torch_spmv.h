/**
 * @file torch_spmv.h
 * @brief Torch C++ wrappers for SpMV (sparse matrix-vector multiplication).
 * @date 2025-03-27
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>

#include "../tiling/tiling_spmv.h"
#include "aclrtlaunch_spmv_v2_fp16.h"
#include "aclrtlaunch_spmv_v2_fp32.h"
#include "aclrtlaunch_spmv_v2_multi_cube_fp16.h"
#include "torch_gather.h"
#include "torch_scan.h"
#include "torch_seg_ops.h"

namespace tcuscan {

/**
 * @brief CSR sparse matrix - dense vector multiplication using the multi-cube
 * scan algorithm. See Segmented Operations using Matrix Multiplications
 * (https://arxiv.org/pdf/2506.23906)
 *
 * Computes A @ x where A is given in CSR format. The prefix scan is
 * accelerated by the cube unit using pre-computed triangular scan matrices.
 *
 * @param vals input non-zero values of the CSR matrix
 * @param indptr row pointer array of the CSR matrix (length rows + 1)
 * @param cols column index array of the CSR matrix
 * @param x dense vector to multiply: computes A @ x
 * @param upper pre-computed upper triangular scan matrix (SxS, float16)
 * @param lower_strict pre-computed strict lower triangular scan matrix (SxS,
 * float16)
 *
 * @note The gather_spmv tiling size is fixed at 128. Values above 512 cause
 * failures; no performance benefit was observed from tuning this parameter.
 *
 * @return Dense result vector of the SpMV product A @ x
 */
at::Tensor run_spmv_multi_cube(const at::Tensor& vals, const at::Tensor& indptr,
                               const at::Tensor& cols, const at::Tensor& x,
                               const at::Tensor& upper,
                               const at::Tensor& lower_strict) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

  const at::Tensor product = tcuscan::run_csr_gather(vals, cols, x);
  const at::Tensor scanned =
      tcuscan::run_scan_multi_cube(product, upper, lower_strict);
  const at::Tensor gathered = tcuscan::run_gather_spmv(scanned, indptr, 128);
  const at::Tensor z = torch::diff(gathered);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief CSR sparse matrix - dense vector multiplication using the multi-core
 * scan algorithm. See Segmented Operations using Matrix Multiplications
 * (https://arxiv.org/pdf/2506.23906)

 *
 * Computes A @ x where A is given in CSR format. The prefix scan is performed
 * on vector cores using the provided tile size @p s.
 *
 * @param vals input non-zero values of the CSR matrix
 * @param indptr row pointer array of the CSR matrix (length rows + 1)
 * @param cols column index array of the CSR matrix
 * @param x dense vector to multiply: computes A @ x
 * @param s tile size for the multi-core prefix scan
 *
 * @note The gather_spmv tiling size is fixed at 128. Values above 512 cause
 * failures; no performance benefit was observed from tuning this parameter.
 *
 * @return Dense result vector of the SpMV product A @ x
 */
at::Tensor run_spmv(const at::Tensor& vals, const at::Tensor& indptr,
                    const at::Tensor& cols, const at::Tensor& x, int s) {
  const auto dtype = vals.options().dtype();
  at::Tensor product;
  if (dtype == torch::kInt16) {
    product = tcuscan::run_csr_gather(vals, cols, x).to(torch::kInt8);
  } else {
    product = tcuscan::run_csr_gather(vals, cols, x);
  }
  const at::Tensor scanned = tcuscan::run_scan_multi_core(product, s);
  const at::Tensor gathered = tcuscan::run_gather_spmv(scanned, indptr, 128);
  const at::Tensor z = torch::diff(gathered);

  return z;
}

/**
 * @brief CSR sparse matrix - dense vector multiplication using the multi-core
 * segmented sum algorithm. See Segmented Operations using Matrix
 * Multiplications (https://arxiv.org/pdf/2506.23906)
 *
 * Computes A @ x where A is given in CSR format. Unlike run_spmv(), this
 * variant fuses the prefix-scan and gather steps into a single segmented sum
 * kernel (run_seg_sum_multi_core()), reducing intermediate tensor allocations.
 *
 * @param vals non-zero values of the CSR matrix
 * @param indptr row pointer array of the CSR matrix (length rows + 1)
 * @param cols column index array of the CSR matrix
 * @param x dense vector to multiply
 * @param s tile size for the multi-core segmented sum kernel
 * @param [in] segm_offsets Segment start index offset per block.
 * @return Dense result vector of the SpMV product A @ x
 */
at::Tensor run_spmv_v2(const at::Tensor& vals, const at::Tensor& indptr,
                       const at::Tensor& cols, const at::Tensor& x, int s,
                       c10::optional<at::Tensor> segm_offsets = c10::nullopt) {
  const auto vals_dtype = vals.options().dtype();
  const auto x_dtype = x.options().dtype();
  TORCH_CHECK((vals_dtype == torch::kHalf && x_dtype == torch::kHalf) ||
                  (vals_dtype == torch::kFloat && x_dtype == torch::kFloat),
              "run_spmv_v2: vals and x must both be fp16 or both be fp32, "
              "got vals=",
              vals_dtype, " x=", x_dtype);
  TORCH_CHECK(
      indptr.scalar_type() == at::kInt || indptr.scalar_type() == at::kUInt32,
      "run_spmv_v2: indptr must be int32 or uint32, got ",
      indptr.scalar_type());
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  const at::Device device = x.options().device();

  const uint32_t tile_len = static_cast<uint32_t>(s);
  const uint32_t nnz = static_cast<uint32_t>(vals.numel());
  const uint32_t x_len = static_cast<uint32_t>(x.numel());
  const uint32_t num_segments = static_cast<uint32_t>(indptr.numel() - 1);

  const uint32_t align_size = tile_len * tile_len;
  const uint32_t num_tiles = host_utils::CeilDiv(nnz, align_size);

  uint32_t block_dim = ascendc_platform->GetCoreNumAic();
  if (num_tiles < block_dim) {
    block_dim = num_tiles;
  }
  const uint32_t max_num_tiles_per_block =
      host_utils::CeilDiv(num_tiles, block_dim);
  const uint32_t block_len = max_num_tiles_per_block * align_size;

  at::Tensor segm_offsets_;
  if (segm_offsets.has_value()) {
    segm_offsets_ = segm_offsets.value();
  } else {
    const at::Tensor sstart = torch::clamp(
        torch::arange(
            0, block_dim + 1,
            torch::TensorOptions().dtype(torch::kInt32).device(device)) *
            block_len,
        c10::nullopt, static_cast<int32_t>(nnz));

    segm_offsets_ = torch::searchsorted(indptr.to(torch::kInt32), sstart,
                                        /*out_int32=*/true);
  }

  const at::Tensor z =
      at::zeros({num_segments},
                at::TensorOptions().dtype(torch::kFloat32).device(device));

  const tcuscan::SpMVTiling tiling{nnz, num_segments, x_len, tile_len,
                                   block_len};
  uint8_t* tiling_device = tcuscan::alloc_copy_tiling(tiling);

  const uint32_t padded_nnz = host_utils::AlignUp(nnz, align_size);
  const bool is_fp32 = (vals_dtype == torch::kFloat);

  // workspace: padded_nnz * sizeof(input_dtype) for CSR products
  //          + padded_nnz * sizeof(float) for cube scan output
  const uint32_t input_elem_size = is_fp32 ? sizeof(float) : sizeof(int16_t);
  const uint32_t workspace_size =
      padded_nnz * (input_elem_size + sizeof(float));
  const at::Tensor workspace_tensor =
      tcuscan::alloc_zeros_workspace(workspace_size, device);

  // Offset indptr by one element, since first element is always zero.
  void* indptr_data = static_cast<void*>(
      static_cast<uint8_t*>(const_cast<void*>(indptr.storage().data())) +
      indptr.element_size());

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (is_fp32) {
    ACLRT_LAUNCH_KERNEL(spmv_v2_fp32)
    (block_dim, acl_stream, const_cast<void*>(vals.storage().data()),
     const_cast<void*>(cols.storage().data()), const_cast<void*>(indptr_data),
     const_cast<void*>(x.storage().data()),
     const_cast<void*>(segm_offsets_.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else {
    ACLRT_LAUNCH_KERNEL(spmv_v2_fp16)
    (block_dim, acl_stream, const_cast<void*>(vals.storage().data()),
     const_cast<void*>(cols.storage().data()), const_cast<void*>(indptr_data),
     const_cast<void*>(x.storage().data()),
     const_cast<void*>(segm_offsets_.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief CSR sparse matrix - dense vector multiplication using the multi-cube
 * segmented sum algorithm. See Segmented Operations using Matrix
 * Multiplications (https://arxiv.org/pdf/2506.23906)
 *
 * Multi-cube variant of run_spmv_v2(): the CSR gather step is identical, but
 * the prefix-scan is computed with the multi-cube block-scan (distributing the
 * matrix tiles across all cube cores) and the segment reduction uses
 * atomic-add writes, mirroring run_seg_sum_multi_cube(). The tile size is taken
 * from the provided scan matrices (S = upper.size(0)).
 *
 * @param [in] vals non-zero values of the CSR matrix (fp16)
 * @param [in] indptr row pointer array of the CSR matrix (length rows + 1)
 * @param [in] cols column index array of the CSR matrix
 * @param [in] x dense vector to multiply (fp16)
 * @param [in] upper pre-computed upper triangular all-ones matrix (SxS, fp16)
 * @param [in] lower_strict pre-computed strict lower triangular all-ones matrix
 * (SxS, fp16)
 * @param [in] segm_offsets Segment start index offset per block. Optional; when
 * omitted it is computed internally from `indptr`.
 *
 * @note Only fp16 is supported: the multi-cube block scan is a half-only
 * kernel.
 *
 * @return Dense result vector of the SpMV product A @ x
 */
at::Tensor run_spmv_v2_multi_cube(
    const at::Tensor& vals, const at::Tensor& indptr, const at::Tensor& cols,
    const at::Tensor& x, const at::Tensor& upper,
    const at::Tensor& lower_strict,
    c10::optional<at::Tensor> segm_offsets = c10::nullopt) {
  const auto vals_dtype = vals.options().dtype();
  const auto x_dtype = x.options().dtype();
  TORCH_CHECK(vals_dtype == torch::kHalf && x_dtype == torch::kHalf,
              "run_spmv_v2_multi_cube: vals and x must both be fp16, got vals=",
              vals_dtype, " x=", x_dtype);
  TORCH_CHECK(indptr.scalar_type() == at::kInt,
              "run_spmv_v2_multi_cube: indptr must be int32, got ",
              indptr.scalar_type());
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  const at::Device device = x.options().device();

  const uint32_t tile_len = static_cast<uint32_t>(upper.size(0));
  const uint32_t nnz = static_cast<uint32_t>(vals.numel());
  const uint32_t x_len = static_cast<uint32_t>(x.numel());
  const uint32_t num_segments = static_cast<uint32_t>(indptr.numel() - 1);

  const uint32_t align_size = tile_len * tile_len;
  const uint32_t num_tiles = host_utils::CeilDiv(nnz, align_size);

  // The segmented sum runs on both vector cores of every AI Core group, so the
  // non-zeros are split into `vec_cores_per_cube * block_dim` blocks of
  // `block_len` non-zeros each, one per vector core.
  const uint32_t vec_cores_per_cube =
      ascendc_platform->GetCoreNumAiv() / ascendc_platform->GetCoreNumAic();
  uint32_t block_dim = ascendc_platform->GetCoreNumAic();
  const uint32_t max_block_dim =
      host_utils::CeilDiv(num_tiles, vec_cores_per_cube);
  if (max_block_dim < block_dim) {
    block_dim = max_block_dim;
  }
  const uint32_t num_vec_blocks = vec_cores_per_cube * block_dim;
  const uint32_t max_num_tiles_per_vec_block =
      host_utils::CeilDiv(num_tiles, num_vec_blocks);
  const uint32_t block_len = max_num_tiles_per_vec_block * align_size;

  at::Tensor segm_offsets_;
  if (segm_offsets.has_value()) {
    segm_offsets_ = segm_offsets.value();
  } else {
    const at::Tensor sstart = torch::clamp(
        torch::arange(
            0, num_vec_blocks + 1,
            torch::TensorOptions().dtype(torch::kInt32).device(device)) *
            block_len,
        c10::nullopt, static_cast<int32_t>(nnz));

    segm_offsets_ = torch::searchsorted(indptr, sstart,
                                        /*out_int32=*/true);
  }

  const at::Tensor z =
      at::zeros({num_segments},
                at::TensorOptions().dtype(torch::kFloat32).device(device));

  const tcuscan::SpMVTiling tiling{nnz, num_segments, x_len, tile_len,
                                   block_len};
  uint8_t* tiling_device = tcuscan::alloc_copy_tiling(tiling);

  const uint32_t padded_nnz = host_utils::AlignUp(nnz, align_size);

  // workspace: padded_nnz * sizeof(fp16) for CSR products
  //          + padded_nnz * sizeof(float) for cube scan output.
  // Zero-initialized so the [nnz, padded_nnz) tail acts as the zero-padding
  // consumed by the multi-cube block scan.
  const uint32_t workspace_size =
      padded_nnz * (sizeof(int16_t) + sizeof(float));
  const at::Tensor workspace_tensor =
      tcuscan::alloc_zeros_workspace(workspace_size, device);

  // Offset indptr by one element, since first element is always zero.
  void* indptr_data = static_cast<void*>(
      static_cast<uint8_t*>(const_cast<void*>(indptr.storage().data())) +
      indptr.element_size());

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  ACLRT_LAUNCH_KERNEL(spmv_v2_multi_cube_fp16)
  (block_dim, acl_stream, const_cast<void*>(vals.storage().data()),
   const_cast<void*>(cols.storage().data()),
   const_cast<void*>(upper.storage().data()),
   const_cast<void*>(lower_strict.storage().data()),
   const_cast<void*>(indptr_data), const_cast<void*>(x.storage().data()),
   const_cast<void*>(segm_offsets_.storage().data()),
   const_cast<void*>(z.storage().data()),
   const_cast<void*>(workspace_tensor.storage().data()), tiling_device);

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

}  // namespace tcuscan
