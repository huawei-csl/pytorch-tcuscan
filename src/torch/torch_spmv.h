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

namespace detail {

/**
 * @brief Combines an SpMV product into a caller-supplied output vector,
 * computing `y = z + beta * y` in place.
 *
 * Follows the BLAS convention: `beta == 0` overwrites @p y without reading it,
 * so an uninitialized (or NaN/Inf-carrying) output vector is valid input.
 *
 * @param [in,out] y Output vector, updated in place.
 * @param [in] z The (already `alpha`-scaled) SpMV product.
 * @param [in] beta Scaling factor applied to the incoming @p y.
 * @return Reference to @p y.
 */
inline at::Tensor& accumulate_into(at::Tensor& y, const at::Tensor& z,
                                   double beta) {
  if (beta == 0.0) {
    y.copy_(z);
  } else if (beta == 1.0) {
    y.add_(z);
  } else {
    y.mul_(beta).add_(z);
  }
  return y;
}

/**
 * @brief Validates a caller-supplied SpMV output vector `y`.
 *
 * @param [in] y Output vector to validate.
 * @param [in] num_segments Expected number of elements (matrix rows).
 * @param [in] device Device the SpMV runs on.
 * @param [in] fn Caller name, used in the error messages.
 */
inline void check_output_vector(const at::Tensor& y, uint32_t num_segments,
                                const at::Device& device, const char* fn) {
  TORCH_CHECK(y.scalar_type() == at::kFloat, fn, ": y must be float32, got ",
              y.scalar_type());
  TORCH_CHECK(y.dim() == 1, fn, ": y must be 1D, got ", y.dim(), "D");
  TORCH_CHECK(y.numel() == static_cast<int64_t>(num_segments), fn,
              ": y must have ", num_segments, " elements (one per row), got ",
              y.numel());
  TORCH_CHECK(y.is_contiguous(), fn, ": y must be contiguous");
  // The kernels address `y` through `storage().data()`, which ignores the
  // storage offset, so a contiguous view into a larger tensor would silently
  // write to the wrong place.
  TORCH_CHECK(y.storage_offset() == 0, fn,
              ": y must own its storage from offset 0 (got a view with offset ",
              y.storage_offset(), ")");
  TORCH_CHECK(y.device() == device, fn, ": y must live on ", device, ", got ",
              y.device());
}

}  // namespace detail

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
 * @param alpha Scaling factor of the SpMV product. Folded into the
 * `gather_spmv` stage, which is exact since `diff(alpha * g) == alpha *
 * diff(g)`.
 * @param beta Scaling factor of the incoming @p y. Ignored when @p y is not
 * supplied. Following the BLAS convention, `beta == 0` overwrites @p y without
 * reading it.
 * @param y Optional output vector (length = rows), updated in place and
 * returned. When omitted, a freshly allocated vector holding `alpha * A @ x`
 * is returned instead.
 *
 * @note The gather_spmv tiling size is fixed at 128. Values above 512 cause
 * failures; no performance benefit was observed from tuning this parameter.
 *
 * @return `y = alpha * A @ x + beta * y`
 */
at::Tensor run_spmv_multi_cube(const at::Tensor& vals, const at::Tensor& indptr,
                               const at::Tensor& cols, const at::Tensor& x,
                               const at::Tensor& upper,
                               const at::Tensor& lower_strict,
                               double alpha = 1.0, double beta = 0.0,
                               c10::optional<at::Tensor> y = c10::nullopt) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

  const at::Tensor product = tcuscan::run_csr_gather(vals, cols, x);
  const at::Tensor scanned =
      tcuscan::run_scan_multi_cube(product, upper, lower_strict);
  const at::Tensor gathered =
      tcuscan::run_gather_spmv(scanned, indptr, 128, alpha);
  const at::Tensor z = torch::diff(gathered);
  aclrtSynchronizeStream(acl_stream);

  if (!y.has_value()) {
    return z;
  }
  at::Tensor y_ = y.value();
  detail::check_output_vector(y_, static_cast<uint32_t>(indptr.numel() - 1),
                              x.options().device(), "run_spmv_multi_cube");
  return detail::accumulate_into(y_, z, beta);
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
 * @param alpha Scaling factor of the SpMV product. Folded into the
 * `gather_spmv` stage, which is exact since `diff(alpha * g) == alpha *
 * diff(g)`.
 * @param beta Scaling factor of the incoming @p y. Ignored when @p y is not
 * supplied. Following the BLAS convention, `beta == 0` overwrites @p y without
 * reading it.
 * @param y Optional output vector (length = rows), updated in place and
 * returned. When omitted, a freshly allocated vector holding `alpha * A @ x`
 * is returned instead.
 *
 * @note The gather_spmv tiling size is fixed at 128. Values above 512 cause
 * failures; no performance benefit was observed from tuning this parameter.
 *
 * @return `y = alpha * A @ x + beta * y`
 */
at::Tensor run_spmv(const at::Tensor& vals, const at::Tensor& indptr,
                    const at::Tensor& cols, const at::Tensor& x, int s,
                    double alpha = 1.0, double beta = 0.0,
                    c10::optional<at::Tensor> y = c10::nullopt) {
  const auto dtype = vals.options().dtype();
  at::Tensor product;
  if (dtype == torch::kInt16) {
    product = tcuscan::run_csr_gather(vals, cols, x).to(torch::kInt8);
  } else {
    product = tcuscan::run_csr_gather(vals, cols, x);
  }
  const at::Tensor scanned = tcuscan::run_scan_multi_core(product, s);
  const at::Tensor gathered =
      tcuscan::run_gather_spmv(scanned, indptr, 128, alpha);
  const at::Tensor z = torch::diff(gathered);

  if (!y.has_value()) {
    return z;
  }
  at::Tensor y_ = y.value();
  detail::check_output_vector(y_, static_cast<uint32_t>(indptr.numel() - 1),
                              x.options().device(), "run_spmv");
  return detail::accumulate_into(y_, z, beta);
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
 * @param alpha Scaling factor of the SpMV product, applied in-kernel by the
 * segment reduction as it writes each segment sum (fp32).
 * @param beta Scaling factor of the incoming @p y, applied in-kernel by a
 * pre-pass over @p y that runs before the segment reduction accumulates onto
 * it. Ignored when @p y is not supplied. Following the BLAS convention,
 * `beta == 0` overwrites @p y without reading it.
 * @param y Optional output vector (float32, length = rows), updated in place
 * and returned. When omitted, a freshly zeroed vector holding `alpha * A @ x`
 * is returned instead.
 * @return `y = alpha * A @ x + beta * y`
 */
at::Tensor run_spmv_v2(const at::Tensor& vals, const at::Tensor& indptr,
                       const at::Tensor& cols, const at::Tensor& x, int s,
                       c10::optional<at::Tensor> segm_offsets = c10::nullopt,
                       double alpha = 1.0, double beta = 0.0,
                       c10::optional<at::Tensor> y = c10::nullopt) {
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

  at::Tensor z;
  float beta_kernel;
  if (y.has_value()) {
    z = y.value();
    detail::check_output_vector(z, num_segments, device, "run_spmv_v2");
    beta_kernel = static_cast<float>(beta);
  } else {
    // Without a caller-supplied `y` the output starts at zero, so `beta * y`
    // vanishes for any beta. Pass 1 to skip the scaling pre-pass entirely.
    z = at::zeros({num_segments},
                  at::TensorOptions().dtype(torch::kFloat32).device(device));
    beta_kernel = 1.0f;
  }

  const tcuscan::SpMVTiling tiling{nnz,        num_segments,
                                   x_len,      tile_len,
                                   block_len,  static_cast<float>(alpha),
                                   beta_kernel};
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
 * @param alpha Scaling factor of the SpMV product, applied in-kernel by the
 * segment reduction as it writes each segment sum (fp32).
 * @param beta Scaling factor of the incoming @p y, applied in-kernel by a
 * pre-pass over @p y that runs before the segment reduction accumulates onto
 * it. Ignored when @p y is not supplied. Following the BLAS convention,
 * `beta == 0` overwrites @p y without reading it.
 * @param y Optional output vector (float32, length = rows), updated in place
 * and returned. When omitted, a freshly zeroed vector holding `alpha * A @ x`
 * is returned instead.
 *
 * @note Only fp16 is supported: the multi-cube block scan is a half-only
 * kernel.
 *
 * @return `y = alpha * A @ x + beta * y`
 */
at::Tensor run_spmv_v2_multi_cube(const at::Tensor& vals,
                                  const at::Tensor& indptr,
                                  const at::Tensor& cols, const at::Tensor& x,
                                  const at::Tensor& upper,
                                  const at::Tensor& lower_strict,
                                  double alpha = 1.0, double beta = 0.0,
                                  c10::optional<at::Tensor> y = c10::nullopt) {
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

  uint32_t block_dim = ascendc_platform->GetCoreNumAic();
  if (num_tiles < block_dim) {
    block_dim = num_tiles;
  }
  const uint32_t max_num_tiles_per_block =
      host_utils::CeilDiv(num_tiles, block_dim);
  const uint32_t block_len = max_num_tiles_per_block * align_size;

  at::Tensor z;
  float beta_kernel;
  if (y.has_value()) {
    z = y.value();
    detail::check_output_vector(z, num_segments, device,
                                "run_spmv_v2_multi_cube");
    beta_kernel = static_cast<float>(beta);
  } else {
    // Without a caller-supplied `y` the output starts at zero, so `beta * y`
    // vanishes for any beta. Pass 1 to skip the scaling pre-pass entirely.
    z = at::zeros({num_segments},
                  at::TensorOptions().dtype(torch::kFloat32).device(device));
    beta_kernel = 1.0f;
  }

  const tcuscan::SpMVTiling tiling{nnz,        num_segments,
                                   x_len,      tile_len,
                                   block_len,  static_cast<float>(alpha),
                                   beta_kernel};
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

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  ACLRT_LAUNCH_KERNEL(spmv_v2_multi_cube_fp16)
  (block_dim, acl_stream, const_cast<void*>(vals.storage().data()),
   const_cast<void*>(cols.storage().data()),
   const_cast<void*>(upper.storage().data()),
   const_cast<void*>(lower_strict.storage().data()),
   const_cast<void*>(indptr.storage().data()),
   const_cast<void*>(x.storage().data()), const_cast<void*>(z.storage().data()),
   const_cast<void*>(workspace_tensor.storage().data()), tiling_device);

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

}  // namespace tcuscan
