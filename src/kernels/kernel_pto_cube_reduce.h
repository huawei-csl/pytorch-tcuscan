/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file kernel_pto_cube_reduce.h
 * @brief PTO-ISA port of kernel_cube_reduce.h.
 *
 * AscendC -> PTO-ISA mapping (non-obvious cases)
 * ─────────────────────────────────────────────────────────────────────────
 *   AscendC                              PTO-ISA
 *   ──────────────────────────────────────────────────────────────────────
 *   GlobalTensor / SetGlobalBuffer     -> pto::GlobalTensor<T,Shape,Stride,ND>
 *   TPipe + TQue<A1/A2/B1/B2/CO1>     -> TileLeft / TileRight / TileAcc
 *   copy::CopyGmToL1A + CopyL1ToL0A   -> TLOAD (ND->NZ layout conversion
 *                                         handled internally by TLOAD for
 *                                         TileType::Mat / TileLeft)
 *   cube_unit::InitConstAllOnesL1      -> TEXPANDS(tile_b, InputT(1))
 *   cube_unit::Multiply<false>         -> TMATMUL(tile_c, tile_a, tile_b)
 *   cube_unit::Multiply<true>          -> TMATMUL_ACC(tile_c1, tile_c0, a, b)
 *   copy::CopyCL0ToGlobal + Fixpipe   -> TSTORE (TileType::Acc -> GM ND)
 *   PipeBarrier<PIPE_ALL>              -> TSYNC on the preceding event
 *   sync::SyncGroup<FULL>()            -> unchanged (kept as-is; not a
 *                                         PTO-ISA primitive)
 *   copy::CopyGmToVec + scalar loop   -> TLOAD into TileVec + TCOLSUM +
 *                                         TROWSUM + TMULS  (see AIV section)
 *   AscendC::SetAtomicAdd/CopyScalarToGm -> TSTORE<AtomicType::AtomicAdd>
 *
 * AscendC APIs with no direct 1:1 PTO-ISA equivalent:
 *   - TPipe / TQue double-buffering: replaced by TSYNC event chaining.
 *     PTO auto-mode handles UB assignment; no explicit TASSIGN required.
 *   - InitConstAllOnesL1 (initialises L1/L0 memory): replaced by
 *     TEXPANDS which directly broadcasts 1 into the TileRight buffer.
 *   - copy::CopyCL0ToGlobal (Fixpipe NZ2ND): replaced by TSTORE from
 *     TileAcc, which performs the same FixPipe conversion internally.
 *   - The AscendC AIV kernel reads S×16 elements via a scalar loop over
 *     column 0 (stride 16).  PTO replaces this with TCOLSUM (reduce rows
 *     -> 1×16) followed by TROWSUM (reduce columns -> 1×1), then TMULS by
 *     1/16 to cancel the 16-way column duplication that TMATMUL produces.
 *   - AscendC sub-block (AIV0 vs AIV1) split with SetAtomicAdd is kept
 *     structurally: each sub-block loads its rows_per_aiv rows, reduces
 *     them to a scalar, and accumulates with AtomicAdd.
 */
#pragma once

#include "pto/pto-inst.hpp"
#include "kernels/kernel_pad.h"
#include "tcuscan_utils.h"

namespace tcuscan {

// ─────────────────────────────────────────────────────────────────────────────
// pto_cube_reduce_aic
//
// Cube-unit path (AIC cores).  Mirrors KernelCubeReduce<InputT,SyncAfter>.
//
// Each cube core processes a contiguous slice of `num_tiles` S×S input tiles,
// multiplying every tile by an all-ones S×16 right-hand matrix.  The partial
// sums across tiles are accumulated in a single TileAcc (ping-pong two
// TileAcc objects).  The final S×16 accumulator is stored to global memory.
//
// Template parameters
//   InputT    - element type of the input vector (half / int8_t / ...)
//   S         - compile-time matmul size (must equal runtime matmul_size)
//   SyncAfter - if true, call SyncGroup<FULL>() after writing output
// ─────────────────────────────────────────────────────────────────────────────
template <typename InputT, uint32_t S, bool SyncAfter = false>
__aicore__ inline void pto_cube_reduce_aic(__gm__ InputT* vec_in,
                                           __gm__ tcuscan::cube_unit::CubeOutType_t<InputT>* cube_out,
                                           uint32_t vec_len) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<InputT>;

  // ── Tile type aliases ────────────────────────────────────────────────────
  // A-matrix: S×S input tile (left operand of TMATMUL).
  // TileLeft maps to AscendC's A2 (L0A) buffer after ND->NZ conversion.
  // TLOAD with TileType::Mat and ND global source performs ND->NZ internally.
  using TileA   = pto::TileLeft<InputT, S, S>;

  // B-matrix: S×16 all-ones tile (right operand), constant across iterations.
  // TileRight maps to AscendC's B2 (L0B) buffer.
  using TileB   = pto::TileRight<InputT, S, 16>;

  // C-matrix accumulator: S×16 float (or int32_t).
  // TileAcc maps to AscendC's CO1 (L0C) buffer.
  // Two instances are needed for ping-pong accumulation with TMATMUL_ACC.
  using TileC   = pto::TileAcc<OutputT, S, 16>;

  // ── GlobalTensor descriptors ─────────────────────────────────────────────
  // A-matrix input: S×S ND layout.
  // AscendC: global_a_.SetGlobalBuffer((__gm__ InputT*)vec_in, vec_len)
  //          + CopyGmToL1A + CopyL1ToL0A  ->  TLOAD(tile_a, GTensorA(...))
  using AShape   = pto::Shape<1, 1, 1, S, S>;
  using AStride  = pto::BaseShape2D<InputT, S, S, pto::Layout::ND>;
  using GTensorA = pto::GlobalTensor<InputT, AShape, AStride, pto::Layout::ND>;

  // C-matrix output: S×16 ND layout.
  // AscendC: global_c_.SetGlobalBuffer((__gm__ OutputT*)cube_out, ...)
  //          + copy::CopyCL0ToGlobal (Fixpipe NZ2ND) -> TSTORE(GTensorC, tile_c)
  using CShape   = pto::Shape<1, 1, 1, S, 16>;
  using CStride  = pto::BaseShape2D<OutputT, S, 16, pto::Layout::ND>;
  using GTensorC = pto::GlobalTensor<OutputT, CShape, CStride, pto::Layout::ND>;

  // ── Derived constants ────────────────────────────────────────────────────
  constexpr uint32_t tile_len = S * S;       // elements per A tile
  constexpr uint32_t out_len  = S * 16u;     // elements per C tile

  const uint32_t block_num  = GetBlockNum();
  const uint32_t id         = GetBlockIdx();

  // AscendC: num_mat_iters_ = FloorDiv(vec_len, tile_len)
  //          max_iters_per_block = CeilDiv(num_mat_iters_, block_num)
  const uint32_t num_mat_iters          = tcuscan::scalar::FloorDiv(vec_len, tile_len);
  const uint32_t max_iters_per_block    = tcuscan::scalar::CeilDiv(num_mat_iters, block_num);
  // Per-core base offset in *elements*.
  const uint32_t base_offset            = id * tile_len * max_iters_per_block;

  // AscendC: GetWorkDistribution(vec_len_, tile_len_, block_num_)
  const uint32_t num_tiles = tcuscan::scalar::GetWorkDistribution(vec_len, tile_len, block_num);

  if (num_tiles == 0) {
    // This core has no work.  If SyncAfter, still participate in the barrier.
    if constexpr (SyncAfter) {
      sync::SyncGroup<sync::GroupSyncDirection::FULL>();
    }
    return;
  }

  // ── Declare tile buffers ─────────────────────────────────────────────────
  // PTO auto-mode allocates UB/L0 space without explicit TASSIGN.
  TileA tile_a;
  TileB tile_b;
  TileC tile_c0, tile_c1;  // ping-pong accumulators

  // ── Fill B with all-ones (once, reused every iteration) ─────────────────
  // AscendC: LoadAllOnesToL0() -> cube_unit::InitConstAllOnesL1 + CopyL1ToL0B
  // PTO: TEXPANDS fills TileRight with the constant 1.
  // Event eb gates the subsequent TMATMUL to wait for the fill to complete.
  auto eb = TEXPANDS(tile_b, InputT(1));

  // ── First tile: no accumulation ──────────────────────────────────────────
  // AscendC: CubeIter<false>(ai_core_offset)
  //   -> CopyGmToL1A + CopyL1ToL0A -> TLOAD
  //   -> cube_unit::Multiply<false>  -> TMATMUL
  GTensorA ga0(vec_in + base_offset);
  auto el0 = TLOAD(tile_a, ga0);
  // TMATMUL waits on both the load event (el0) and the expand event (eb).
  // Result goes into tile_c0.  tile_c1 is the "current active" accumulator.
  auto em = TMATMUL(tile_c0, tile_a, tile_b, el0, eb);

  // ── Remaining tiles: accumulate into the ping-pong pair ─────────────────
  // AscendC: for idx in 1..num_tiles-1: CubeIter<true>(offset)
  //   -> CopyGmToL1A + CopyL1ToL0A -> TLOAD
  //   -> cube_unit::Multiply<true>  -> TMATMUL_ACC
  //
  // TMATMUL_ACC(c_out, c_in, a, b, events...)
  //   c_in  = previously computed accumulator
  //   c_out = freshly updated accumulator (ping-pong between tile_c0 / tile_c1)
  for (uint32_t idx = 1; idx < num_tiles; idx++) {
    const uint32_t offset = base_offset + idx * tile_len;
    GTensorA ga(vec_in + offset);
    auto el = TLOAD(tile_a, ga);

    if (idx % 2 == 1) {
      // c0 was the most-recent output; accumulate into c1.
      em = TMATMUL_ACC(tile_c1, tile_c0, tile_a, tile_b, em, el, eb);
    } else {
      // c1 was the most-recent output; accumulate into c0.
      em = TMATMUL_ACC(tile_c0, tile_c1, tile_a, tile_b, em, el, eb);
    }
  }

  // ── Determine final accumulator after ping-pong ──────────────────────────
  // If num_tiles is odd the last write went to tile_c0 (first iter) or the
  // iteration with idx%2==1 (odd idx) which writes to tile_c1.  Track:
  //   - idx=0 writes tile_c0  (1 tile total -> result in tile_c0)
  //   - idx=1 writes tile_c1  (2 tiles total -> result in tile_c1)
  //   - idx=2 writes tile_c0  (3 tiles total -> result in tile_c0)
  //   - ...  result is in tile_c0 when num_tiles is odd, tile_c1 when even.
  TileC& tile_c_final = (num_tiles % 2 == 1) ? tile_c0 : tile_c1;

  // ── Store S×16 accumulator to global memory ──────────────────────────────
  // AscendC: copy::CopyCL0ToGlobal(global_c_[id*out_len], co1_q_, S, 16)
  //   which calls Fixpipe with NZ2ND conversion.
  // PTO: TSTORE from TileAcc performs the equivalent FixPipe path.
  GTensorC gc(cube_out + id * out_len);
  auto es = TSTORE(gc, tile_c_final, em);

  // AscendC: PipeBarrier<PIPE_ALL>() after CopyCL0ToGlobal.
  // PTO: TSYNC on the store event ensures FixPipe has committed before the
  //      cross-core sync below releases the AIV cores.
  TSYNC(es);

  if constexpr (SyncAfter) {
    // AscendC: sync::SyncGroup<sync::GroupSyncDirection::FULL>()
    sync::SyncGroup<sync::GroupSyncDirection::FULL>();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// pto_cube_reduce_aiv
//
// Vector-unit path (AIV cores).  Mirrors KernelCompleteCubeReduce<OutputT,SyncBefore>.
//
// Reads the S×16 per-cube-core output produced by pto_cube_reduce_aic, reduces
// it to a single scalar per AI-core group using TCOLSUM+TROWSUM+TMULS, and
// accumulates the result into the final output using an atomic-add store.
//
// Template parameters
//   OutputT    - element type of the intermediate cube output (float / int32_t)
//   S          - compile-time matmul size
//   SyncBefore - if true, call SyncGroup<FULL>() before processing
// ─────────────────────────────────────────────────────────────────────────────
template <typename OutputT, uint32_t S, bool SyncBefore = false>
__aicore__ inline void pto_cube_reduce_aiv(__gm__ OutputT* vec_in,
                                           __gm__ OutputT* vec_out) {
  // ── Derived constants ────────────────────────────────────────────────────
  constexpr uint32_t tile_len = S * 16u;   // elements in one S×16 cube output tile

  const uint32_t id         = static_cast<uint32_t>(GetBlockIdx());
  const uint32_t aiv_id     = static_cast<uint32_t>(GetSubBlockIdx());
  const uint32_t ratio      = static_cast<uint32_t>(GetTaskRation());

  // AscendC: output_idx = FloorDiv(id, ratio)
  const uint32_t output_idx = tcuscan::scalar::FloorDiv(id, ratio);

  // ── Tile type aliases ────────────────────────────────────────────────────
  // Full S×16 tile loaded from the cube output.
  // AscendC: LocalTensor<OutputT> (UB VECIN) with size tile_len.
  using TileVec = pto::Tile<pto::TileType::Vec, OutputT, S, 16,
                            pto::BLayout::RowMajor>;

  // 1×16 tile: result of TCOLSUM (one value per column of the S×16 input).
  using TileColSum = pto::Tile<pto::TileType::Vec, OutputT, 1, 16,
                               pto::BLayout::RowMajor>;

  // Same shape as TileVec, used as TCOLSUM scratch (isBinary=false path).
  using TileTmpColSum = pto::Tile<pto::TileType::Vec, OutputT, S, 16,
                                  pto::BLayout::RowMajor>;

  // 1×1 tile: result of TROWSUM across the 1×16 col-sum row.
  // TROWSUM(dst[R×1], src[R×C], tmp) writes dst[i,0] = sum_j(src[i,j]).
  // With R=1, C=16 the result is a single scalar in a 1×1 tile.
  // RowMajor ND layout is used so that subsequent TMULS/TSHRS (which require
  // isRowMajor on both src and dst) can consume this tile directly.
  // A 1×1 tile has no layout distinction in practice (single element).
  using TileRowSum = pto::Tile<pto::TileType::Vec, OutputT, 1, 1,
                               pto::BLayout::RowMajor>;

  // Scratch tile for TROWSUM: must have at least as many columns as the source
  // to hold intermediate row-stride data.  Same shape/layout as TileColSum.
  using TileTmpRowSum = pto::Tile<pto::TileType::Vec, OutputT, 1, 16,
                                  pto::BLayout::RowMajor>;

  // 1×1 scalar result tile (TMULS / TSHRS output; also the AtomicAdd source).
  // Must be RowMajor for TMULS constraint on A2A3.
  using TileScalar = pto::Tile<pto::TileType::Vec, OutputT, 1, 1,
                               pto::BLayout::RowMajor>;

  // ── GlobalTensor descriptors ─────────────────────────────────────────────
  // Cube intermediate output (S×16 ND).
  // AscendC: global_input_.SetGlobalBuffer((__gm__ OutputT*)vec_in, ...)
  using InShape   = pto::Shape<1, 1, 1, S, 16>;
  using InStride  = pto::BaseShape2D<OutputT, S, 16, pto::Layout::ND>;
  using GTensorIn = pto::GlobalTensor<OutputT, InShape, InStride, pto::Layout::ND>;

  // Scalar output (1×1 ND, atomic-add destination).
  // AscendC: global_output_.SetGlobalBuffer((__gm__ OutputT*)vec_out, block_num)
  using OutShape   = pto::Shape<1, 1, 1, 1, 1>;
  using OutStride  = pto::BaseShape2D<OutputT, 1, 1, pto::Layout::ND>;
  using GTensorOut = pto::GlobalTensor<OutputT, OutShape, OutStride, pto::Layout::ND>;

  // ── WriteOutZero ─────────────────────────────────────────────────────────
  // AscendC: WriteOutZero() -> aiv_id==0 writes scalar 0 to output slot
  //          via copy::CopyScalarToGm with no atomic mode.
  // PTO: TEXPANDS fills a 1×1 tile with 0, TSTORE writes it (no AtomicAdd).
  if (aiv_id == 0) {
    TileScalar tile_zero;
    auto ef = TEXPANDS(tile_zero, OutputT(0));
    GTensorOut g_out_zero(vec_out + output_idx);
    auto es_zero = TSTORE(g_out_zero, tile_zero, ef);
    TSYNC(es_zero);
  }

  // AscendC: if constexpr (SyncBefore) { sync::SyncGroup<FULL>(); }
  if constexpr (SyncBefore) {
    sync::SyncGroup<sync::GroupSyncDirection::FULL>();
  }

  // ── ProcessTile ──────────────────────────────────────────────────────────
  // Each AIV sub-core processes rows_per_aiv rows of the S×16 cube output.
  // AscendC loads the full S×16 tile and then loops over `aiv_core_offset +
  // i*16` indices (stride 16 = MAT_DIM_16) to sum elements from column 0.
  //
  // PTO approach:
  //   1. TLOAD the rows_per_aiv rows assigned to this AIV sub-block into a
  //      tile with SetValidRow(rows_per_aiv).
  //   2. TCOLSUM: reduce rows -> 1×16 (sum each of the 16 columns).
  //   3. TROWSUM: reduce 1×16 -> 1×1 scalar (DN, Cols==1).  dst[0,0] holds
  //      the total sum of all 16 column entries from TCOLSUM.
  //   4. TMULS by 1/16 (float) or TSHRS by 4 (int32): correct for the 16-way
  //      column duplication inherent in the TMATMUL all-ones B matrix.
  //      The TMATMUL produces 16 identical columns; summing all 16 columns
  //      therefore overcounts by 16.
  //   5. TSTORE<AtomicAdd>: atomically accumulate into output[output_idx].

  const uint32_t rows_per_aiv      = S / ratio;
  // Byte offset of the first element this AIV sub-block owns within the S×16
  // cube output tile.
  // AscendC: aiv_core_offset = aiv_id * (tile_len / ratio)
  const uint32_t aiv_elem_offset   = output_idx * tile_len + aiv_id * rows_per_aiv * 16u;

  // ── Declare tiles ────────────────────────────────────────────────────────
  TileVec        tile_part;
  TileColSum     tile_col_sum;
  TileTmpColSum  tile_tmp_col;
  TileRowSum     tile_row_sum;
  TileTmpRowSum  tile_tmp_row;
  TileScalar     tile_result;

  // Restrict the valid row count to the sub-block's share.
  // AscendC: the scalar loop runs matmul_size_/ratio iterations.
  // PTO: tile_part.SetValidRow(rows_per_aiv) then TLOAD transfers only those rows.
  tile_part.SetValidRow(static_cast<int>(rows_per_aiv));

  // Load rows_per_aiv rows of the S×16 cube output into tile_part.
  // AscendC: copy::CopyGmToVec(vecin_q_, global_input_[gm_offset], tile_len)
  GTensorIn g_in(vec_in + aiv_elem_offset);
  auto el = TLOAD(tile_part, g_in);

  // TCOLSUM: sum all rows -> 1×16 vector.
  // dst.GetValidCol() == 16, src.GetValidRow() == rows_per_aiv.
  auto ec = TCOLSUM(tile_col_sum, tile_part, tile_tmp_col, /*isBinary=*/false, el);

  // TROWSUM: sum the 1×16 col-sum row -> 1×1 scalar tile (DN, Cols==1).
  // TROWSUM(dst[1×1], src[1×16], tmp[1×16]) writes dst[0,0] = sum_j(src[0,j]).
  // The 16 column sums from TCOLSUM are added together, giving the total sum
  // of all elements in tile_part — but overcounted by 16 (one per duplicate
  // column) because TMATMUL produced 16 identical columns.
  auto er = TROWSUM(tile_row_sum, tile_col_sum, tile_tmp_row, ec);

  // Scale by 1/16: cancels the 16-way column duplication from TMATMUL.
  // AscendC: sum += input_lt.GetValue(aiv_core_offset + i * 16)  [column 0 only]
  // tile_row_sum[0,0] holds sum(16 identical column totals); dividing by 16
  // yields the true partial sum for this AIV sub-block's rows.
  //
  // TMULS / TSHRS require src and dst to have the same ValidRow/ValidCol.
  // Both tile_row_sum and tile_result are 1×1, so this is satisfied.
  //
  // For float output (half input):  TMULS by 1.0f/16.0f
  // For int32_t output (int8 input): TSHRS by 4  (divide by 16 = shift right 4)
  if constexpr (std::is_same_v<OutputT, float>) {
    auto em = TMULS(tile_result, tile_row_sum, OutputT(1.0f / 16.0f), er);
    // AtomicAdd store: accumulate partial result from this AIV sub-block.
    // AscendC: AscendC::SetAtomicAdd<T>() + copy::CopyScalarToGm(global_output_[output_idx], ...)
    GTensorOut g_out(vec_out + output_idx);
    auto es = TSTORE<TileScalar, GTensorOut, pto::AtomicType::AtomicAdd>(g_out, tile_result, em);
    TSYNC(es);
  } else {
    // int32_t path: shift-right by 4 == divide by 16
    // AscendC: sum += input_lt.GetValue(index)  (only column-0 elements, no /16)
    // TSHRS scalar type must match DType (int32_t).
    auto em = TSHRS(tile_result, tile_row_sum, OutputT(4), er);
    GTensorOut g_out(vec_out + output_idx);
    auto es = TSTORE<TileScalar, GTensorOut, pto::AtomicType::AtomicAdd>(g_out, tile_result, em);
    TSYNC(es);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// run_pto_cube_reduce
//
// Top-level entry point.  Mirrors run_cube_reduce<T>.
//
// Handles optional padding (when vec_len is not aligned to S*S or UB), then
// dispatches pto_cube_reduce_aic on AIC cores and pto_cube_reduce_aiv on AIV
// cores.
//
// Template parameters
//   InputT - element type of the input vector (half / int8_t)
//   S      - compile-time matmul size (must be known at compile time for the
//             PTO tile types; the caller must instantiate for each concrete S)
// ─────────────────────────────────────────────────────────────────────────────
template <typename InputT, uint32_t S>
__aicore__ inline void run_pto_cube_reduce(__gm__ InputT* vec_in,
                                           __gm__ tcuscan::cube_unit::CubeOutType_t<InputT>* vec_out,
                                           GM_ADDR workspace,
                                           uint32_t vec_len) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<InputT>;

  constexpr uint32_t align_size = S * S;
  const uint32_t padded_vec_len = tcuscan::scalar::AlignUp(vec_len, align_size);

  // Intermediate cube output (S×16 per cube core) lives at the start of
  // workspace by default; moved past the padded input if padding is needed.
  // AscendC: padded_cube_reductions starts at workspace.
  GM_ADDR padded_cube_reductions = workspace;

  if (vec_len % align_size != 0 || vec_len % UB_ALIGNMENT != 0) {
    // ── Padding pass ───────────────────────────────────────────────────────
    // AscendC: run_pad_kernel<InputT,false>(vec_in, workspace, vec_len, align_size)
    // This is still an AscendC kernel; kept unchanged per the spec.
    GM_ADDR const padded_input = workspace;
    run_pad_kernel<InputT, false>(reinterpret_cast<GM_ADDR>(vec_in),
                                  padded_input, vec_len, align_size);

    // Sync after padding so that the cube kernel reads the padded data.
    // AscendC: sync::SyncGroup<FULL>()
    sync::SyncGroup<sync::GroupSyncDirection::FULL>();

    // Point vec_in at the padded copy and advance the cube-output pointer.
    vec_len = padded_vec_len;
    vec_in  = reinterpret_cast<__gm__ InputT*>(padded_input);
    // The intermediate cube output comes after the padded input.
    padded_cube_reductions = workspace + padded_vec_len * sizeof(InputT);
  }

  __gm__ OutputT* inter_out = reinterpret_cast<__gm__ OutputT*>(padded_cube_reductions);

  // ── Dispatch AIC path ────────────────────────────────────────────────────
  // AscendC: if ASCEND_IS_AIC { KernelCubeReduce<T,true> ... Process(); }
  if ASCEND_IS_AIC {
    pto_cube_reduce_aic<InputT, S, true /*SyncAfter*/>(vec_in, inter_out, vec_len);
  }

  // ── Dispatch AIV path ────────────────────────────────────────────────────
  // AscendC: if ASCEND_IS_AIV { KernelCompleteCubeReduce<OutputT,true> ... Process(); }
  if ASCEND_IS_AIV {
    pto_cube_reduce_aiv<OutputT, S, true /*SyncBefore*/>(inter_out, vec_out);
  }
}

}  // namespace tcuscan
