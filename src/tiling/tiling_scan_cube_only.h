/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file tiling_scan_cube_only.h
 * @brief Tiling structure for the single AI-core cube-only scan.
 */

#pragma once

#include <cstdint>

namespace tcuscan {

#pragma pack(push, 8)
/**
 * @brief Cube-only scan tiling parameter structure.
 *
 * The AIC part scans blocks of length `matmul_size * matmul_size` via
 * MatMuls, while the AIV part completes the blocks in a single core.
 */
struct CubeOnlyScanTiling {
  /// @brief Total number of input elements.
  uint32_t num_elems;
  /// @brief Matmul size used. Each block has length `matmul_size *
  /// matmul_size`.
  uint32_t matmul_size;
};
#pragma pack(pop)

}  // namespace tcuscan
