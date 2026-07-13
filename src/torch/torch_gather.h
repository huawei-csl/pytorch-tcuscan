/**
 * @file torch_gather.h
 * @brief Torch wrapper for gather kernels.
 * @date 2025-03-27
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/tiling_csr_gather.h"
#include "../tiling/tiling_gather_spmv.h"
#include "../tiling/tiling_mc_gather.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "workspace.h"

// AscendC kernel entry points launched below with `<<<>>>`; defined in
// csr_gather.cpp, gather_spmv.cpp, mc_gather.cpp.
extern "C" __global__ __aicore__ void csr_gather_fp16(
    __gm__ void* values_in, __gm__ void* cols_in, __gm__ void* x_in,
    __gm__ void* z_out, __gm__ void* workspace, __gm__ void* tiling_gm);
extern "C" __global__ __aicore__ void csr_gather_fp32(
    __gm__ void* values_in, __gm__ void* cols_in, __gm__ void* x_in,
    __gm__ void* z_out, __gm__ void* workspace, __gm__ void* tiling_gm);
extern "C" __global__ __aicore__ void csr_gather_int16(
    __gm__ void* values_in, __gm__ void* cols_in, __gm__ void* x_in,
    __gm__ void* z_out, __gm__ void* workspace, __gm__ void* tiling_gm);
extern "C" __global__ __aicore__ void gather_spmv(__gm__ void* values_in,
                                                  __gm__ void* cols_in,
                                                  __gm__ void* vec_out,
                                                  __gm__ void* workspace,
                                                  __gm__ void* tilingGm);
extern "C" __global__ __aicore__ void mc_gather_fp16(__gm__ void* values_in,
                                                     __gm__ void* indexes_in,
                                                     __gm__ void* z_out,
                                                     __gm__ void* workspace,
                                                     __gm__ void* tiling_gm);
extern "C" __global__ __aicore__ void mc_gather_fp32(__gm__ void* values_in,
                                                     __gm__ void* indexes_in,
                                                     __gm__ void* z_out,
                                                     __gm__ void* workspace,
                                                     __gm__ void* tiling_gm);

namespace tcuscan {

/**
 * @brief Multi-core gather of input 1D vector.
 *
 * @param values Input 1D vector.
 * @param idxs Input indices. Pre-condition: all indices are in-bounds of
 * `values`.
 * @param tile_len Tile length.
 * @return Gathered values.
 */
at::Tensor run_mc_gather(const at::Tensor& values, const at::Tensor& idxs,
                         const uint32_t tile_len) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();

  const auto dtype = values.options().dtype();
  const at::Device device = values.options().device();

  const uint32_t idx_len = idxs.numel();
  const uint32_t num_tiles = host_utils::CeilDiv(idx_len, tile_len);
  uint32_t block_dim = ascendc_platform->GetCoreNumAiv();
  if (num_tiles < block_dim) {
    block_dim = num_tiles;
  }

  const uint32_t values_len = values.numel();

  const at::Tensor z =
      at::empty({idx_len}, at::TensorOptions().dtype(dtype).device(device));
  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const McGatherTiling tiling{block_dim, values_len, idx_len, tile_len};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == torch::kHalf) {
    mc_gather_fp16<<<block_dim, nullptr, acl_stream>>>(
        const_cast<void*>(values.storage().data()),
        const_cast<void*>(idxs.storage().data()),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else {
    mc_gather_fp32<<<block_dim, nullptr, acl_stream>>>(
        const_cast<void*>(values.storage().data()),
        const_cast<void*>(idxs.storage().data()),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Special CSR gather method for SpMV.
 *
 * @param values Input values of CSR matrix.
 * @param cols Input columns of CSR matrix.
 * @param x Input vector
 * @return z[i] = values[i] * x[cols[i]] for i in range(len(cols)).
 */
at::Tensor run_csr_gather(const at::Tensor& values, const at::Tensor& cols,
                          const at::Tensor& x) {
  TORCH_CHECK(values.dim() == 1, "run_csr_gather: values must be 1D, got ",
              values.dim(), "D");
  TORCH_CHECK(cols.dim() == 1, "run_csr_gather: cols must be 1D, got ",
              cols.dim(), "D");
  TORCH_CHECK(x.dim() == 1, "run_csr_gather: x must be 1D, got ", x.dim(), "D");
  TORCH_CHECK(cols.numel() == values.numel(), "run_csr_gather: cols length (",
              cols.numel(), ") must match values length (", values.numel(),
              ")");
  const auto dtype = values.options().dtype();
  TORCH_CHECK(
      dtype == torch::kHalf || dtype == torch::kInt16 || dtype == torch::kFloat,
      "run_csr_gather: values dtype must be fp16, int16, or fp32, got ", dtype);
  TORCH_CHECK(x.options().dtype() == dtype, "run_csr_gather: x dtype (",
              x.options().dtype(), ") must match values dtype (", dtype, ")");
  TORCH_CHECK(cols.options().dtype() == torch::kInt32,
              "run_csr_gather: cols dtype must be int32, got ",
              cols.options().dtype());
  TORCH_CHECK(values.is_contiguous(),
              "run_csr_gather: values must be contiguous");
  TORCH_CHECK(cols.is_contiguous(), "run_csr_gather: cols must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "run_csr_gather: x must be contiguous");
  TORCH_CHECK(values.device() == cols.device(),
              "run_csr_gather: values and cols must be on the same device");
  TORCH_CHECK(values.device() == x.device(),
              "run_csr_gather: values and x must be on the same device");

  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  const uint32_t max_aiv_cores = ascendc_platform->GetCoreNumAiv();
  const at::Device device = x.options().device();

  const at::Tensor z = at::empty_like(values);
  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const uint32_t values_len = values.numel();
  const uint32_t x_len = x.numel();

  constexpr uint32_t TILE_LEN = 1024;

  const CSRGatherTiling tiling{values_len, x_len, TILE_LEN};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  uint32_t block_dim = host_utils::CeilDiv(values_len, TILE_LEN);
  block_dim = block_dim > max_aiv_cores ? max_aiv_cores : block_dim;

  if (block_dim <= 1) {
    block_dim = 1;
  }

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == torch::kHalf) {
    csr_gather_fp16<<<block_dim, nullptr, acl_stream>>>(
        const_cast<void*>(values.storage().data()),
        const_cast<void*>(cols.storage().data()),
        const_cast<void*>(x.storage().data()),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else if (dtype == torch::kInt16) {
    csr_gather_int16<<<block_dim, nullptr, acl_stream>>>(
        const_cast<void*>(values.storage().data()),
        const_cast<void*>(cols.storage().data()),
        const_cast<void*>(x.storage().data()),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else if (dtype == torch::kFloat) {
    csr_gather_fp32<<<block_dim, nullptr, acl_stream>>>(
        const_cast<void*>(values.storage().data()),
        const_cast<void*>(cols.storage().data()),
        const_cast<void*>(x.storage().data()),
        const_cast<void*>(z.storage().data()),
        const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Special gather for SpmV.
 *
 * @param values Input 1D vector.
 * @param idxs Input 1D indices vector.
 * @param tile_len Tile length.
 * @return Special gather for SpMV.
 */
at::Tensor run_gather_spmv(const at::Tensor& values, const at::Tensor& idxs,
                           const uint32_t tile_len) {
  const auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance();
  const uint32_t max_aiv_cores = ascendc_platform->GetCoreNumAiv();

  const at::Device device = values.options().device();

  uint32_t values_len = values.numel();
  uint32_t idx_len = idxs.numel();

  uint32_t block_dim = host_utils::CeilDiv(values_len, tile_len);
  block_dim = block_dim > max_aiv_cores ? max_aiv_cores : block_dim;

  const at::Tensor z = at::empty({idx_len}, values.options());
  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  const GatherSpmvTiling tiling{block_dim, values_len, idx_len, tile_len};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  gather_spmv<<<block_dim, nullptr, acl_stream>>>(
      const_cast<void*>(values.storage().data()),
      const_cast<void*>(idxs.storage().data()),
      const_cast<void*>(z.storage().data()),
      const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}
}  // namespace tcuscan
