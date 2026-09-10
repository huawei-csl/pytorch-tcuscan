/**
 * @file torch_scan_cpu.cpp
 * @brief Torch wrapper for CPU scan kernels.
 * @date 2025-10-12
 *
 * @copyright Copyright Huawei (c) 2025
 */
#include "torch_scan_cpu.h"

#include <omp.h>

#include <algorithm>
#include <vector>

namespace tcuscan {
namespace {

/**
 * @brief Inclusive scan inplace :a[i] := sum_{j=0..i} a[j]
 *
 * @tparam T Data type
 * @param a Input vector.
 * @param n Input vector length
 */
template <typename T>
void parallel_inclusive_scan_blocked(T* a, size_t n) {
  if (n == 0) return;

  const int max_threads = omp_get_max_threads();
  // heuristic block count: at most one block per thread, but allow more blocks
  // to improve load balance
  const std::size_t blocks =
      std::min<std::size_t>(std::max<std::size_t>(1, max_threads), n);
  const std::size_t block_size = (n + blocks - 1) / blocks;

  std::vector<T> block_totals(blocks);

// Phase 1: per-block inclusive scan and compute block totals
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
    for (std::size_t b = 0; b < blocks; ++b) {
      const std::size_t start = b * block_size;
      const std::size_t end = std::min(start + block_size, n);
      if (start >= end) {
        block_totals[b] = T(0);
        continue;
      }

      // sequential scan inside block (compiler can auto-vectorize)
      T sum = a[start];
      for (std::size_t i = start + 1; i < end; ++i) {
        sum += a[i];
        a[i] = sum;
      }
      block_totals[b] = sum;
    }
  }

  // Phase 2: scan block_totals (serial is fine because blocks << n)
  // block_prefix[b] = sum of totals for blocks < b
  std::vector<T> block_prefix(blocks);
  T running = T(0);
  for (std::size_t b = 0; b < blocks; ++b) {
    block_prefix[b] = running;
    running += block_totals[b];
  }

// Phase 3: add block_prefix offsets in parallel to each block (skip first block
// where prefix is 0)
#pragma omp parallel for schedule(static)
  for (std::size_t b = 0; b < blocks; ++b) {
    const T offset = block_prefix[b];
    if (offset == T(0)) continue;  // nothing to add for first block typically
    const std::size_t start = b * block_size;
    const std::size_t end = std::min(start + block_size, n);
    for (std::size_t i = start; i < end; ++i) {
      a[i] += offset;
    }
  }
}

}  // namespace

at::Tensor run_scan_cpu(const at::Tensor& x) {
  TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CPU);
  const auto dtype = x.options().dtype();
  const uint32_t x_len = x.numel();
  const auto dtype_out = dtype == torch::kHalf || dtype == torch::kFloat32
                             ? torch::kFloat32
                             : torch::kInt32;

  const at::Tensor z = x.clone().to(dtype_out);

  if (dtype == torch::kHalf) {
    parallel_inclusive_scan_blocked<float>(z.data_ptr<float>(), x_len);
  } else if (dtype == torch::kFloat32) {
    parallel_inclusive_scan_blocked<float>(z.data_ptr<float>(), x_len);
  } else if (dtype == torch::kInt8) {
    parallel_inclusive_scan_blocked<int32_t>(z.data_ptr<int32_t>(), x_len);
  }

  return z;
}

}  // namespace tcuscan
