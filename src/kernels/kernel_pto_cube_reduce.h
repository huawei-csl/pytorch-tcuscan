/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file kernel_pto_cube_reduce.h
 * @brief PTO-ISA port of KernelCubeReduce.
 *
 * Replaces AscendC TQue / TPipe / Copy* APIs with PTO tile intrinsics:
 *
 *   AscendC                          PTO-ISA
 *   ─────────────────────────────────────────────────────────────────────
 *   CopyGmToL1A + CopyL1ToL0A   →  TLOAD  (TileLeft)
 *   InitConstAllOnesL1 + CopyL1ToL0B →  TEXPANDS(tile_b, 1)
 *   cube_unit::Multiply (no acc) →  TMATMUL
 *   cube_unit::Multiply (accum)  →  TMATMUL_ACC  (in-place form)
 *   CopyCL0ToGlobal              →  TSTORE  (TileAcc → GM)
 *   CopyGmToVec + scalar loop    →  TLOAD (TileVec) + TCOLSUM + TROWSUM
 *   PipeBarrier<PIPE_ALL>        →  TSYNC(event)
 *   SetAtomicAdd / SetAtomicNone →  TSTORE<…, pto::AtomicType::AtomicAdd>
 *
 * Design note: PTO tile types require compile-time dimensions, so
 * `matmul_size` is promoted from a runtime parameter to the template
 * parameter `S`.  Instantiate with the same `S` values used in the
 * AscendC version (e.g. 16, 32, 64, 128).
 */
#pragma once

#include <type_traits>

#include "pto/pto-inst.hpp"
#include "tcuscan_utils.h"

namespace tcuscan {

// ─────────────────────────────────────────────────────────────────────────────
// KernelCubeReduce  (Cube unit path)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @ingroup reduce
 * @brief PTO-ISA block-reduction kernel using the Cube unit.
 *
 * Each Cube core reduces a contiguous slice of the input vector by treating
 * it as a sequence of S×S matrix tiles (A), multiplying each tile by a
 * persistent S×16 all-ones matrix (B), and accumulating the result into a
 * single S×16 accumulator tile (C).  After all tiles are consumed, C is
 * written to global memory.
 *
 * Because all 16 columns of the B matrix are ones, every column of the
 * resulting S×16 C tile holds the same partial row-sum — i.e. each row i
 * of C satisfies C[i][j] = sum_{k=0}^{S-1} A[i][k] for all j.  This is
 * the same FixPipe behaviour documented in the AscendC version.
 *
 * PTO tile mapping:
 *   TileA = TileLeft<InputT, S, S>   — input matrix tile in Left buffer
 *   TileB = TileRight<InputT, S, 16> — all-ones matrix tile in Right buffer
 *   TileC = TileAcc<OutputT, S, 16>  — accumulator tile
 *
 * @tparam InputT   Element type of the input vector (half or int8_t).
 * @tparam S        Matmul tile side length (compile-time constant).
 * @tparam SyncAfter  Emit a cross-core group sync after writing the result.
 */
template <typename InputT, uint32_t S, bool SyncAfter = false>
class KernelCubeReduce {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<InputT>;

  // ── PTO tile types ──────────────────────────────────────────────────────
  // TMATMUL constraint: TileLeft::Rows == TileAcc::Rows (S == S) ✓
  //                     TileLeft::Cols == TileRight::Rows (S == S) ✓
  //                     TileRight::Cols == TileAcc::Cols (16 == 16) ✓
  using TileA = pto::TileLeft<InputT, S, S>;
  using TileB = pto::TileRight<InputT, S, 16>;
  using TileC = pto::TileAcc<OutputT, S, 16>;

  // ── GlobalTensor descriptors ────────────────────────────────────────────
  // Input A: S×S tile in row-major (ND) layout.
  using GMShapeA  = pto::Shape<1, 1, 1, S, S>;
  using GMStrideA = pto::BaseShape2D<InputT, S, S, pto::Layout::ND>;
  using GTensorA  = pto::GlobalTensor<InputT, GMShapeA, GMStrideA, pto::Layout::ND>;

  // Output C: S×16 tile in row-major (ND) layout.
  using GMShapeC  = pto::Shape<1, 1, 1, S, 16>;
  using GMStrideC = pto::BaseShape2D<OutputT, S, 16, pto::Layout::ND>;
  using GTensorC  = pto::GlobalTensor<OutputT, GMShapeC, GMStrideC, pto::Layout::ND>;

 public:
  static constexpr uint32_t MAT_DIM_16      = 16;
  static constexpr uint32_t TILE_LEN        = S * S;
  static constexpr uint32_t OUTPUT_TILE_LEN = S * MAT_DIM_16;

  /**
   * @param vec_len  Number of elements in the input vector.
   *                 Must be divisible by S*S.
   */
  __aicore__ inline KernelCubeReduce(uint32_t vec_len)
      : block_num_(GetBlockNum()),
        vec_len_(vec_len),
        num_mat_iters_(vec_len / TILE_LEN),
        iters_per_block_(pto::scalar::CeilDiv(num_mat_iters_, block_num_)) {
    static_assert(tcuscan::cube_unit::IsCubeSupported<InputT>,
                  "Unsupported Cube dtype: use half or int8_t.");
  }

  /**
   * @param vec_in   Pointer to input vector in global memory.
   * @param vec_out  Pointer to output buffer in global memory.
   *                 Size: block_num * S * 16 elements of OutputT.
   */
  __aicore__ inline void Init(__gm__ InputT* vec_in, __gm__ OutputT* vec_out) {
    in_ptr_  = vec_in;
    out_ptr_ = vec_out;
  }

  /**
   * Main kernel body — call once per Cube core.
   */
  __aicore__ inline void Process() {
    const uint32_t id      = GetBlockIdx();
    const uint32_t offset0 = id * TILE_LEN * iters_per_block_;
    const uint32_t num_tiles =
        tcuscan::scalar::GetWorkDistribution(vec_len_, TILE_LEN, block_num_);

    if (num_tiles == 0) {
      if constexpr (SyncAfter) {
        pto::sync::SyncGroup<pto::sync::GroupSyncDirection::FULL>();
      }
      return;
    }

    // Fill B with all-ones once; held in the Right buffer for all iterations.
    // TEXPANDS broadcasts a scalar to every element of the tile.
    TileB tile_b;
    auto eb = TEXPANDS(tile_b, static_cast<InputT>(1));
    TSYNC(eb);

    // ── First tile: TMATMUL (initialises the accumulator C) ────────────────
    TileA tile_a;
    TileC tile_c;
    {
      GTensorA g_a(in_ptr_ + offset0);
      // TLOAD brings the S×S input tile from GM into the Left buffer.
      auto el = TLOAD(tile_a, g_a);
      TSYNC(el);
    }
    // C = A × B  (no prior accumulation)
    auto em0 = TMATMUL(tile_c, tile_a, tile_b);
    TSYNC(em0);

    // ── Remaining tiles: TMATMUL_ACC (in-place accumulate into C) ──────────
    for (uint32_t i = 1; i < num_tiles; i++) {
      GTensorA g_a(in_ptr_ + offset0 + i * TILE_LEN);
      auto el = TLOAD(tile_a, g_a);
      TSYNC(el);
      // C += A × B  (in-place form: cOutMatrix == cInMatrix == tile_c)
      auto em = TMATMUL_ACC(tile_c, tile_a, tile_b);
      TSYNC(em);
    }

    // ── Flush accumulator to global memory ─────────────────────────────────
    // TSTORE with TileType::Acc: destination layout must be ND or NZ.
    GTensorC g_c(out_ptr_ + id * OUTPUT_TILE_LEN);
    auto es = TSTORE(g_c, tile_c);
    TSYNC(es);

    if constexpr (SyncAfter) {
      // Signal Vector units that the Cube result is in global memory.
      pto::sync::SyncGroup<pto::sync::GroupSyncDirection::FULL>();
    }
  }

 private:
  __gm__ InputT*  in_ptr_  = nullptr;
  __gm__ OutputT* out_ptr_ = nullptr;

  const uint32_t block_num_;
  const uint32_t vec_len_;
  const uint32_t num_mat_iters_;
  const uint32_t iters_per_block_;
};

// ─────────────────────────────────────────────────────────────────────────────
// KernelCompleteCubeReduce  (Vector unit path)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @ingroup reduce
 * @brief PTO-ISA completion kernel (Vector unit): reduces each S×16 tile
 * produced by KernelCubeReduce to a single per-core scalar.
 *
 * Algorithm:
 *  1. TLOAD  — load the S×16 tile from global memory (Vec tile).
 *  2. TCOLSUM — reduce rows to obtain a 1×16 row vector where each element
 *               equals the column sum.  Because KernelCubeReduce fills B
 *               with all-ones, all 16 columns of the accumulator are
 *               identical, so every element of this 1×16 row equals the
 *               true block partial sum.
 *  3. TROWSUM — collapse the 1×16 row to a 1×1 scalar.  The result is
 *               16 × block_sum (since all 16 equal elements are summed).
 *  4. Scale   — multiply by 1/16 (float) or logical right-shift by 4
 *               (int32_t, valid because the value is always a multiple of 16)
 *               to recover the true block sum.
 *  5. TSTORE  — atomic-add the partial sum to the output scalar.
 *
 * @tparam T         Output element type from KernelCubeReduce (float/int32_t).
 * @tparam S         Matmul tile side length matching KernelCubeReduce.
 * @tparam SyncBefore  Wait for Cube unit group sync before reading input.
 */
template <typename T, uint32_t S, bool SyncBefore>
class KernelCompleteCubeReduce {
  static constexpr uint32_t MAT_DIM_16 = 16;
  static constexpr uint32_t TILE_LEN   = S * MAT_DIM_16;

  // ── PTO tile types ──────────────────────────────────────────────────────
  // Full S×16 tile loaded from GM (all 16 columns are identical by design).
  using TileIn  = pto::Tile<pto::TileType::Vec, T, S, 16>;
  // Scratch tile for TCOLSUM binary-tree reduction path.
  using TileTmp = pto::Tile<pto::TileType::Vec, T, S, 16>;
  // 1×16 row produced by TCOLSUM; each element == block partial sum.
  using TileSum = pto::Tile<pto::TileType::Vec, T, 1, 16>;
  // 1×1 scalar produced by TROWSUM then scaled; holds the final partial sum.
  using TileScalar = pto::Tile<pto::TileType::Vec, T, 1, 1>;

  // ── GlobalTensor descriptors ────────────────────────────────────────────
  using GMShapeIn   = pto::Shape<1, 1, 1, S, 16>;
  using GMStrideIn  = pto::BaseShape2D<T, S, 16, pto::Layout::ND>;
  using GTensorIn   = pto::GlobalTensor<T, GMShapeIn, GMStrideIn, pto::Layout::ND>;

  // One-element output slot per block.
  using GMShapeOut  = pto::Shape<1, 1, 1, 1, 1>;
  using GMStrideOut = pto::BaseShape2D<T, 1, 1, pto::Layout::ND>;
  using GTensorOut  = pto::GlobalTensor<T, GMShapeOut, GMStrideOut, pto::Layout::ND>;

 public:
  __aicore__ inline KernelCompleteCubeReduce() = default;

  /**
   * @param vec_in   Pointer to the cube-reduce output buffer (block_num × S × 16).
   * @param vec_out  Pointer to the final output buffer (block_num scalars).
   */
  __aicore__ inline void Init(__gm__ T* vec_in, __gm__ T* vec_out) {
    in_ptr_  = vec_in;
    out_ptr_ = vec_out;
  }

  /**
   * Main kernel body — call once per Vector core.
   */
  __aicore__ inline void Process() {
    const uint32_t id         = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t ratio      = static_cast<uint32_t>(GetTaskRation());
    const uint32_t output_idx = pto::scalar::FloorDiv(id, ratio);

    // ── Zero the output scalar before atomic accumulation ──────────────────
    GTensorOut g_out(out_ptr_ + output_idx);
    TileScalar tile_zero;
    auto ez = TEXPANDS(tile_zero, static_cast<T>(0));
    TSYNC(ez);
    // Plain (non-atomic) write to initialise: races avoided because only one
    // sub-block per output_idx issues this write before the sync below.
    auto es0 = TSTORE(g_out, tile_zero);
    TSYNC(es0);

    // ── Wait for KernelCubeReduce to finish writing its tile ───────────────
    if constexpr (SyncBefore) {
      pto::sync::SyncGroup<pto::sync::GroupSyncDirection::FULL>();
    }

    // ── Load the S×16 tile from global memory ─────────────────────────────
    const uint32_t gm_offset = output_idx * TILE_LEN;
    GTensorIn g_in(in_ptr_ + gm_offset);
    TileIn  tile_in;
    TileTmp tile_tmp;
    auto el = TLOAD(tile_in, g_in);
    TSYNC(el);

    // ── TCOLSUM: reduce S rows → 1 row of column sums ─────────────────────
    // tile_sum[0][j] = sum_i tile_in[i][j]  for j in [0, 16).
    // By the all-ones-B invariant: all 16 elements are equal (== block_sum).
    TileSum tile_sum;
    auto ec = TCOLSUM(tile_sum, tile_in, tile_tmp, /*isBinary=*/false);
    TSYNC(ec);

    // ── TROWSUM: collapse 1×16 → 1×1 (= 16 × block_sum) ──────────────────
    // All 16 elements are equal, so TROWSUM gives exactly 16 × block_sum.
    TileScalar tile_scalar;
    auto er = TROWSUM(tile_scalar, tile_sum);
    TSYNC(er);

    // ── Correct for the 16× factor ────────────────────────────────────────
    ScaleBy1Over16(tile_scalar);

    // ── Atomic-add the partial block sum to the output scalar ──────────────
    TSTORE<TileScalar, GTensorOut, pto::AtomicType::AtomicAdd>(g_out, tile_scalar);
  }

 private:
  // Divide a 1×1 tile by 16 to undo the TROWSUM over-count.
  // For float: multiply by 1/16 (exact in IEEE-754).
  // For int32_t: arithmetic right-shift by 4 (exact because the value is
  //              always a multiple of 16 — it is the 16-way replication of a
  //              single integer produced by the int8 matmul accumulator).
  __aicore__ inline void ScaleBy1Over16(TileScalar& t) {
    if constexpr (std::is_same_v<T, float>) {
      TMULS(t, t, static_cast<T>(1.0f / 16.0f));
    } else {
      static_assert(std::is_same_v<T, int32_t>,
                    "Unsupported OutputT: expected float or int32_t.");
      // Logical right-shift by 4 == integer division by 16.
      TSHRS(t, t, 4);
    }
  }

  __gm__ T* in_ptr_  = nullptr;
  __gm__ T* out_ptr_ = nullptr;
};

// ─────────────────────────────────────────────────────────────────────────────
// run_pto_cube_reduce  (entry point)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @ingroup reduce
 * @brief Top-level PTO-ISA entry point for the cube-reduce pipeline.
 *
 * Dispatches KernelCubeReduce to AIC cores and KernelCompleteCubeReduce to
 * AIV cores, separated by a cross-core group synchronisation.
 *
 * @tparam T          Input element type (half or int8_t).
 * @tparam S          Matmul tile side length (compile-time constant).
 *
 * @param vec_in      Pointer to the input vector in global memory.
 * @param vec_out     Pointer to the output scalars in global memory.
 *                    Size: block_num / GetTaskRation() elements of OutputT.
 * @param workspace   Scratch buffer; must hold at least
 *                    block_num × S × 16 × sizeof(OutputT) bytes.
 * @param vec_len     Number of input elements (must be divisible by S×S).
 */
template <typename T, uint32_t S>
__aicore__ inline void run_pto_cube_reduce(
    __gm__ T*    vec_in,
    __gm__ typename tcuscan::cube_unit::CubeOutType_t<T>* vec_out,
    __gm__ void* workspace,
    uint32_t     vec_len) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<T>;

  __gm__ OutputT* cube_reductions = static_cast<__gm__ OutputT*>(workspace);

  if ASCEND_IS_AIC {
    KernelCubeReduce<T, S, /*SyncAfter=*/true> op(vec_len);
    op.Init(vec_in, cube_reductions);
    op.Process();
  }

  if ASCEND_IS_AIV {
    KernelCompleteCubeReduce<OutputT, S, /*SyncBefore=*/true> op;
    op.Init(cube_reductions, vec_out);
    op.Process();
  }
}

}  // namespace tcuscan
