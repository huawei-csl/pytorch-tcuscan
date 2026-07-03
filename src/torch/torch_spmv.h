/**
 * @file torch_spmv.h
 * @brief Torch C++ wrappers for SpMV (sparse matrix-vector multiplication).
 * @date 2025-03-27
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>

#include <algorithm>
#include <limits>

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
 * @brief Per-block, per-L2-chunk slab length (in non-zeros).
 *
 * Must match the device derivation in run_spmv_v2() (src/spmv.cpp): all cores
 * cooperate on one L2 chunk of block_dim * block_len non-zeros, sized so the
 * CSR products buffer plus the fp32 row-scan output stay L2-resident. Falls
 * back to the full single-chunk slab when everything fits in L2.
 *
 * @param nnz Number of non-zeros.
 * @param block_dim Launch grid size (number of AI Core groups).
 * @param align_size Tile alignment granularity (tile_len * tile_len).
 * @param per_elem Working-set bytes per non-zero (input dtype + fp32 output).
 * @param l2_cache_size L2 cache size in bytes; a very large value disables
 * splitting.
 * @return Per-block slab length in non-zeros.
 */
inline uint32_t spmv_l2_block_len(uint32_t nnz, uint32_t block_dim,
                                  uint32_t align_size, uint64_t per_elem,
                                  uint64_t l2_cache_size) {
  const uint64_t total_tiles =
      host_utils::CeilDiv(static_cast<uint64_t>(nnz), align_size);
  const uint64_t l2_tiles = l2_cache_size / (per_elem * align_size);

  // Tiles per L2 chunk: capped by what fits in L2, but at least one.
  const uint64_t chunk_tiles =
      std::max<uint64_t>(std::min(total_tiles, l2_tiles), 1);

  // Spread the chunk across the cores; each core's slab is tile-aligned.
  return static_cast<uint32_t>(host_utils::CeilDiv(chunk_tiles, block_dim) *
                               align_size);
}

/**
 * @brief CSR sparse matrix - dense vector multiplication using the multi-core
 * segmented sum algorithm. See Segmented Operations using Matrix
 * Multiplications (https://arxiv.org/pdf/2506.23906).
 *
 * Computes A @ x where A is given in CSR format, fusing the prefix-scan and
 * gather steps into a single segmented sum kernel.
 *
 * @param vals non-zero values of the CSR matrix (fp16 or fp32)
 * @param indptr row pointer array of the CSR matrix (length rows + 1)
 * @param cols column index array of the CSR matrix
 * @param x dense vector to multiply
 * @param s tile size for the multi-core segmented sum kernel
 * @param split_l2 If true, query the hardware L2 size and split the non-zeros
 * into serial L2-resident chunks; if false, use a single chunk over all
 * non-zeros.
 * @param segm_offsets Optional precomputed segment start index per block.
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

  const bool is_fp32 = (vals_dtype == torch::kFloat);
  const uint32_t input_elem_size = is_fp32 ? sizeof(float) : sizeof(int16_t);
  const uint64_t per_elem = input_elem_size + sizeof(float);

  // L2 cache size drives the chunking, on both host and device. The no-L2
  // variant disables splitting by passing a value large enough that a single
  // chunk covers all non-zeros.
  uint64_t l2_cache_size = std::numeric_limits<uint64_t>::max();
  if (split_l2) {
    ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::L2,
                                     l2_cache_size);
  }

  const uint32_t block_len =
      spmv_l2_block_len(nnz, block_dim, align_size, per_elem, l2_cache_size);
  const uint32_t num_l2_iter = host_utils::CeilDiv(nnz, block_dim * block_len);

  at::Tensor segm_offsets_;
  if (segm_offsets.has_value()) {
    segm_offsets_ = segm_offsets.value();
  } else {
    // One boundary per (L2 chunk x block); global block id g indexes entry g.
    const at::Tensor sstart = torch::clamp(
        torch::arange(
            0, num_l2_iter * block_dim + 1,
            torch::TensorOptions().dtype(torch::kInt32).device(device)) *
            block_len,
        c10::nullopt, static_cast<int32_t>(nnz));

    segm_offsets_ = torch::searchsorted(indptr.to(torch::kInt32), sstart,
                                        /*out_int32=*/true);
  }

  const at::Tensor z =
      at::zeros({num_segments},
                at::TensorOptions().dtype(torch::kFloat32).device(device));

  const tcuscan::SpMVTiling tiling{nnz,      num_segments, x_len,
                                   tile_len, block_dim,    l2_cache_size};
  uint8_t* tiling_device = tcuscan::alloc_copy_tiling(tiling);

  const uint32_t padded_nnz = host_utils::AlignUp(nnz, align_size);

  const uint32_t workspace_size = padded_nnz * per_elem;
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
 * L2 across the fused gather / scan / segmented-sum stages. Degrades to a
 * single chunk (identical to run_spmv_v2_no_l2) when the whole problem already
 * fits in L2. See run_spmv_v2_impl() for parameter details.
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
