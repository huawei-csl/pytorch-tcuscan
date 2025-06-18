/**
 * @file torch_spmv.h
 * @brief Torch wrapper for gather kernels.
 * @date 2025-03-27
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>

#include "torch_gather.h"
#include "torch_scan.h"

namespace asc {
namespace spmv {
/**
 * @brief
 *
 * @param vals: input values from CSR format
 * @param idx: row_ptr array from CSR format
 * @param cols: column array from CSR format
 * @param x: Vector to be multiplied to a matrix, A*x
 * @param S: tile size parameter for cube unit
 * @param upper: upper triangular matrix SxS of float16
 * @param lower_strict: lower triangular matrix SxS of float16
 *
 *
 * Note: gather_spmv also takes as input its own tiling size, i.e., 128. It can
 * be hardcoded in the run_gather_spmv as it happens for the run_csr_gather. No
 * performance improvements with scaling that param. Failures for >512
 * @return Returns the sparse (CSR) matrix product with vector
 *
 */

at::Tensor run_spmv_multi_cube(const at::Tensor &vals, const at::Tensor &idx,
                               const at::Tensor &cols, const at::Tensor &x,
                               int S, const at::Tensor &upper,
                               const at::Tensor &lower_strict) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

  const at::Tensor product = asc::gather::run_csr_gather(vals, cols, x);
  const at::Tensor scanned =
      asc::scan::run_scan_multi_cube(product, upper, lower_strict);
  const at::Tensor gathered = asc::gather::run_gather_spmv(scanned, idx, 128);
  const at::Tensor z = torch::diff(gathered);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief
 *
 * @param vals: input values from CSR format
 * @param idx: row_ptr array from CSR format
 * @param cols: column array from CSR format
 * @param x: Vector to be multiplied to a matrix, A*x
 * @param S: tile size parameter for cube unit
 *
 * Note: gather_spmv also takes as input its own tiling size, i.e., 128. It can
 * be hardcoded in the run_gather_spmv as it happens for the run_csr_gather. No
 * performance improvements with scaling that param. Failures for >512
 * @return Returns the sparse (CSR) matrix product with vector
 */
at::Tensor run_spmv(const at::Tensor &vals, const at::Tensor &idx,
                    const at::Tensor &cols, const at::Tensor &x, int S) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

  const at::Tensor product = asc::gather::run_csr_gather(vals, cols, x);
  const at::Tensor scanned = asc::scan::run_scan_multi_core(product, S);
  const at::Tensor gathered = asc::gather::run_gather_spmv(scanned, idx, 128);
  const at::Tensor z = torch::diff(gathered);
  aclrtSynchronizeStream(acl_stream);

  return z;
}
}  // namespace spmv
}  // namespace asc
