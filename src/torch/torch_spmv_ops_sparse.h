/**
 * @file torch_spmv_ops_sparse.h
 * @brief Torch C++ wrapper for the row-parallel CSR SpMV kernel migrated from
 * CANN ops-sparse.
 *
 * @copyright Copyright Huawei (c) 2026
 */
#pragma once

#include <pybind11/pybind11.h>

#include "../host_utils.h"
#include "../tiling/tiling_spmv_ops_sparse.h"
#include "aclrtlaunch_spmv_ops_sparse_fp16.h"
#include "aclrtlaunch_spmv_ops_sparse_fp32.h"
#include "commons.h"

namespace tcuscan {

/**
 * @brief CSR sparse matrix - dense vector multiplication using a row-parallel
 * kernel migrated from CANN ops-sparse.
 *
 * Computes the general SpMV \f$ y \leftarrow \alpha \cdot A x + \beta \cdot y
 * \f$, where \f$ A \f$ is given in CSR format. When @p trans is true the
 * transposed product \f$ A^\top x \f$ is computed instead (accumulated with
 * atomic adds).
 *
 * Each AI vector core is assigned a contiguous range of matrix rows. The
 * per-row pipeline gathers the matching entries of @p x, multiplies them by the
 * CSR values, reduces the products and combines with the (scaled) prior output.
 *
 * @param vals CSR non-zero values (length nnz). fp16 or fp32.
 * @param indptr CSR row-pointer array (length rows + 1). int32 or uint32.
 * @param cols CSR column-index array (length nnz). int32 or uint32.
 * @param x dense input vector. Same dtype as @p vals. Length must be
 *   `num_cols` (or `rows` when @p trans is true).
 * @param alpha scaling factor applied to the matrix-vector product.
 * @param beta scaling factor applied to the pre-existing output @p y.
 * @param trans whether to compute the transposed product.
 * @param num_cols number of columns of @p A. Required when @p trans is true;
 *   otherwise defaults to `x.numel()`.
 * @param y optional in/out output vector (used when @p beta != 0). When not
 *   provided a zero-initialized vector is allocated.
 *
 * @return Dense result vector: length `rows` (or `num_cols` when @p trans).
 */
at::Tensor run_spmv_ops_sparse(const at::Tensor& vals, const at::Tensor& indptr,
                               const at::Tensor& cols, const at::Tensor& x,
                               float alpha = 1.0f, float beta = 0.0f,
                               bool trans = false,
                               c10::optional<int64_t> num_cols = c10::nullopt,
                               c10::optional<at::Tensor> y = c10::nullopt) {
  const auto vals_dtype = vals.options().dtype();
  const auto x_dtype = x.options().dtype();
  TORCH_CHECK((vals_dtype == torch::kHalf && x_dtype == torch::kHalf) ||
                  (vals_dtype == torch::kFloat && x_dtype == torch::kFloat),
              "run_spmv_ops_sparse: vals and x must both be fp16 or both be "
              "fp32, got vals=",
              vals_dtype, " x=", x_dtype);
  TORCH_CHECK(
      indptr.scalar_type() == at::kInt || indptr.scalar_type() == at::kUInt32,
      "run_spmv_ops_sparse: indptr must be int32 or uint32, got ",
      indptr.scalar_type());
  TORCH_CHECK(
      cols.scalar_type() == at::kInt || cols.scalar_type() == at::kUInt32,
      "run_spmv_ops_sparse: cols must be int32 or uint32, got ",
      cols.scalar_type());

  // The transposed variant (SpmvKernelTrans) is migrated in the kernel header
  // but accumulates results with per-element atomic scatter to arbitrary,
  // sub-block-aligned column offsets, which is not yet stable on this platform.
  // It is gated here so the library never triggers a device fault; the standard
  // (non-transposed) SpMV below is fully validated for fp16 and fp32.
  TORCH_CHECK(!trans,
              "run_spmv_ops_sparse: trans=true is not yet supported "
              "(the transposed atomic-scatter kernel is unvalidated).");

  const at::Device device = x.options().device();

  TORCH_CHECK(indptr.numel() >= 1,
              "run_spmv_ops_sparse: indptr must have at least 1 element");
  const uint32_t total_rows_num = static_cast<uint32_t>(indptr.numel() - 1);
  uint32_t total_cols_num;
  if (num_cols.has_value()) {
    total_cols_num = static_cast<uint32_t>(num_cols.value());
  } else {
    TORCH_CHECK(
        !trans,
        "run_spmv_ops_sparse: num_cols must be provided when trans=true");
    total_cols_num = static_cast<uint32_t>(x.numel());
  }

  const int64_t x_expected = trans ? total_rows_num : total_cols_num;
  TORCH_CHECK(x.numel() == x_expected, "run_spmv_ops_sparse: x has ", x.numel(),
              " elements but expected ", x_expected);

  const int64_t y_len = trans ? total_cols_num : total_rows_num;

  at::Tensor y_out;
  if (y.has_value()) {
    y_out = y.value();
    TORCH_CHECK(y_out.numel() == y_len, "run_spmv_ops_sparse: y has ",
                y_out.numel(), " elements but expected ", y_len);
    TORCH_CHECK(y_out.options().dtype() == vals_dtype,
                "run_spmv_ops_sparse: y dtype must match vals dtype");
  } else {
    // beta * y requires a defined y: zero-initialize when the caller did not
    // pass one (covers the common beta == 0 case).
    y_out = at::zeros({y_len}, x.options());
  }

  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  uint32_t block_dim = ascendc_platform->GetCoreNumAiv();
  if (block_dim == 0) {
    block_dim = 1;
  }

  const tcuscan::SpMVOpsSparseTiling tiling{
      total_rows_num, total_cols_num, static_cast<float>(alpha),
      static_cast<float>(beta), trans ? 1u : 0u};
  uint8_t* tiling_device = tcuscan::alloc_copy_tiling(tiling);

  // No user workspace is needed; allocate the AscendC system workspace only.
  const at::Tensor workspace_tensor = tcuscan::alloc_workspace(0, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  const bool is_fp32 = (vals_dtype == torch::kFloat);
  if (is_fp32) {
    ACLRT_LAUNCH_KERNEL(spmv_ops_sparse_fp32)
    (block_dim, acl_stream, const_cast<void*>(indptr.storage().data()),
     const_cast<void*>(cols.storage().data()),
     const_cast<void*>(vals.storage().data()),
     const_cast<void*>(x.storage().data()),
     const_cast<void*>(y_out.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else {
    ACLRT_LAUNCH_KERNEL(spmv_ops_sparse_fp16)
    (block_dim, acl_stream, const_cast<void*>(indptr.storage().data()),
     const_cast<void*>(cols.storage().data()),
     const_cast<void*>(vals.storage().data()),
     const_cast<void*>(x.storage().data()),
     const_cast<void*>(y_out.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return y_out;
}

}  // namespace tcuscan
