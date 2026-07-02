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
 * @param split_l2 If true, split the non-zeros into L2-cache-sized chunks
 * processed serially so each chunk's workspace stays L2-resident across the
 * gather / scan / segmented-sum stages. If false, process all non-zeros in a
 * single chunk (num_l2 = 1).
 * @param [in] segm_offsets Segment start index offset per block.
 * @return Dense result vector of the SpMV product A @ x
 */
inline at::Tensor run_spmv_v2_impl(
    const at::Tensor& vals, const at::Tensor& indptr, const at::Tensor& cols,
    const at::Tensor& x, int s, bool split_l2,
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

  const bool is_fp32 = (vals_dtype == torch::kFloat);
  const uint32_t input_elem_size = is_fp32 ? sizeof(float) : sizeof(int16_t);

  // Default (no L2 splitting): a single chunk covers all non-zeros.
  uint32_t block_len = max_num_tiles_per_block * align_size;
  uint32_t num_l2 = 1;

  if (split_l2) {
    uint64_t l2_size = 0;
    ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::L2,
                                     l2_size);

    // Per-element footprint that must stay L2-resident: the CSR products buffer
    // (input dtype) plus the fp32 row-scan output. The dense vector `x` is
    // excluded: it is accessed with random column indices and does not localize
    // per chunk.
    const uint64_t per_elem = input_elem_size + sizeof(float);
    // Chunk granularity so that each chunk tiles evenly across all cores and
    // block_len stays a multiple of align_size.
    const uint32_t chunk_granularity = block_dim * align_size;
    const uint64_t raw_fit = l2_size / per_elem;
    uint32_t fitting_len = static_cast<uint32_t>(
        (raw_fit / chunk_granularity) * chunk_granularity);
    if (fitting_len < chunk_granularity) {
      fitting_len = chunk_granularity;
    }

    const uint32_t candidate_num_l2 = host_utils::CeilDiv(nnz, fitting_len);
    if (candidate_num_l2 > 1) {
      num_l2 = candidate_num_l2;
      block_len = fitting_len / block_dim;
    }
    // Otherwise everything fits in L2: keep the single-chunk defaults.
  }

  at::Tensor segm_offsets_;
  if (segm_offsets.has_value()) {
    segm_offsets_ = segm_offsets.value();
  } else {
    // One boundary per (L2 chunk x block); global block id g indexes entry g.
    const at::Tensor sstart = torch::clamp(
        torch::arange(
            0, num_l2 * block_dim + 1,
            torch::TensorOptions().dtype(torch::kInt32).device(device)) *
            block_len,
        c10::nullopt, static_cast<int32_t>(nnz));

    segm_offsets_ = torch::searchsorted(indptr.to(torch::kInt32), sstart,
                                        /*out_int32=*/true);
  }

  const at::Tensor z =
      at::zeros({num_segments},
                at::TensorOptions().dtype(torch::kFloat32).device(device));

  const tcuscan::SpMVTiling tiling{nnz,       num_segments, x_len,   tile_len,
                                   block_len, num_l2,       block_dim};
  uint8_t* tiling_device = tcuscan::alloc_copy_tiling(tiling);

  const uint32_t padded_nnz = host_utils::AlignUp(nnz, align_size);

  // workspace: padded_nnz * sizeof(input_dtype) for CSR products
  //          + padded_nnz * sizeof(float) for cube scan output
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
 * @brief CSR SpMV (v2) with L2-cache splitting enabled.
 *
 * Splits the non-zeros into L2-cache-sized chunks (derived from the hardware L2
 * size) and processes them serially so each chunk's workspace stays resident in
 * L2 across the fused gather / scan / segmented-sum stages. Degrades to a single
 * chunk (identical to run_spmv_v2_no_l2) when the whole problem already fits in
 * L2. See run_spmv_v2_impl() for parameter details.
 */
inline at::Tensor run_spmv_v2(
    const at::Tensor& vals, const at::Tensor& indptr, const at::Tensor& cols,
    const at::Tensor& x, int s,
    c10::optional<at::Tensor> segm_offsets = c10::nullopt) {
  return run_spmv_v2_impl(vals, indptr, cols, x, s, /*split_l2=*/true,
                          segm_offsets);
}

/**
 * @brief CSR SpMV (v2) without L2-cache splitting (single chunk over all
 * non-zeros). Provided for benchmarking against run_spmv_v2().
 */
inline at::Tensor run_spmv_v2_no_l2(
    const at::Tensor& vals, const at::Tensor& indptr, const at::Tensor& cols,
    const at::Tensor& x, int s,
    c10::optional<at::Tensor> segm_offsets = c10::nullopt) {
  return run_spmv_v2_impl(vals, indptr, cols, x, s, /*split_l2=*/false,
                          segm_offsets);
}

}  // namespace tcuscan
