/**
 * @file torch_scan_cpu.h
 * @brief Torch wrapper for CPU scan kernels.
 * @date 2025-10-12
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <torch/extension.h>

namespace tcuscan {

/**
 * @brief Returns the prefix sum of an input 1D vector.
 *
 * The implementation lives in torch_scan_cpu.cpp: it is OpenMP-based host code,
 * and the ASC compiler that builds the rest of the extension ships no OpenMP
 * runtime, so it has to stay in a plain C++ translation unit.
 *
 * @param x Input 1D vector.
 * @return The prefix sum of `x`.
 */
at::Tensor run_scan_cpu(const at::Tensor& x);

}  // namespace tcuscan
