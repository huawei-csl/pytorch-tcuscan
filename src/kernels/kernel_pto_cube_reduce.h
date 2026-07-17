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
 *   copy::CopyGmToL1A                   -> TLOAD (GM -> L1 Mat tile; the
 *                                         ND->NZ conversion happens here)
 *   copy::CopyL1ToL0A                   -> TMOV (L1 Mat -> L0A Left)
 *   cube_unit::InitConstAllOnesL1      -> TEXPANDS(b_l1, InputT(1))
 *                                         (fills the L1 Mat tile with ones)
 *   cube_unit::CopyL1ToL0B (all-ones)  -> TMOV (L1 Mat -> L0B Right)
 *   cube_unit::Multiply<false>         -> TMATMUL(tile_c, tile_a, tile_b)
 *   cube_unit::Multiply<true>          -> TMATMUL_ACC(tile_c1, tile_c0, a, b)
 *   copy::CopyCL0ToGlobal + Fixpipe   -> TSTORE (TileAcc -> GM ND)
 *   PipeBarrier<PIPE_ALL>              -> TSYNC on the preceding event
 *   sync::SyncGroup<FULL>()            -> unchanged (kept as-is; not a
 *                                         PTO-ISA primitive)
 *   copy::CopyGmToVec + scalar loop   -> TLOAD into TileVec + TCOLSUM +
 *                                         TROWSUM + TMULS  (see AIV section)
 *   SetAtomicAdd + CopyScalarToGm     -> TSTORE<AtomicType::AtomicAdd>
 *
 * AscendC APIs with no direct 1:1 PTO-ISA equivalent:
 *   - TPipe / TQue double-buffering: replaced by TSYNC event chaining.
 *     PTO auto-mode handles L1/L0 assignment; no explicit TASSIGN required.
 *   - InitConstAllOnesL1 (initialises L1 memory): replaced by TEXPANDS,
 *     which broadcasts 1 into the L1 (Mat) buffer; a TMOV then copies it to
 *     the L0B (Right) operand. (TEXPANDS, like TLOAD, can only target a
 *     Vec/Mat tile, so the ones matrix is built in L1 and moved to L0B.)
 *   - copy::CopyCL0ToGlobal (Fixpipe NZ2ND): replaced by TSTORE from
 *     TileAcc, which performs the same FixPipe conversion internally.
 *   - The AscendC AIV kernel reads S×16 elements via a scalar loop over
 *     column 0 (stride 16).  PTO replaces this with TCOLSUM (reduce rows
 *     -> 1×16) followed by TROWSUM (reduce columns -> 1×1), then TMULS by
 *     1/16 to cancel the 16-way column duplication that TMATMUL produces.
 *     (TMATMUL with an all-ones B matrix produces 16 identical columns; the
 *     AscendC path only sums column 0, equivalent to dividing the full
 *     column-sum by 16.)
 *   - AscendC sub-block (AIV0 vs AIV1) split with SetAtomicAdd is kept
 *     structurally: each sub-block loads its rows_per_aiv rows, reduces
 *     them to a scalar, and accumulates with AtomicAdd.
 *   - queue::FreeFromQ<InputT>(b2_q_) at end of AIC Process: in PTO the
 *     tile lifetime is managed by the compiler; no explicit free is needed.
 */
#pragma once

// kernel_pad.h (and the AscendC headers it pulls in) must be included before
// pto/pto-inst.hpp: PTO-ISA headers do `using namespace std;` at global scope,
// which makes CANN's global-scope `struct conditional` in common_types.h
// ambiguous with std::conditional if parsed afterwards.
#include <type_traits>

#include "kernels/kernel_pad.h"
#include "pto/pto-inst.hpp"
#include "tcuscan_utils.h"

namespace tcuscan {

// ─────────────────────────────────────────────────────────────────────────────
// pto_cube_reduce_aic
//
// Cube-unit path (AIC cores).  Mirrors KernelCubeReduce<InputT,SyncAfter>.
//
// Each cube core processes a contiguous slice of `num_tiles` S×S input tiles,
// multiplying every tile by an all-ones S×16 right-hand matrix.  Partial sums
// are accumulated in a ping-pong pair of TileAcc buffers.  The final S×16
// accumulator is written to global memory via TSTORE (which performs the
// Fixpipe NZ2ND conversion internally).
//
// Template parameters:
//   InputT    - element type of the input vector (half / int8_t)
//   S         - compile-time matmul size (runtime matmul_size must equal S)
//   SyncAfter - if true, call SyncGroup<FULL>() after writing output
// ─────────────────────────────────────────────────────────────────────────────
template <typename InputT, uint32_t S, bool SyncAfter = false>
__aicore__ inline void pto_cube_reduce_aic(
    __gm__ InputT* vec_in,
    __gm__ tcuscan::cube_unit::CubeOutType_t<InputT>* cube_out,
    uint32_t vec_len) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<InputT>;

  // ── Tile type aliases ────────────────────────────────────────────────────
  // The cube (matmul) datapath is three-staged, exactly as in the AscendC
  // original: GM -> L1 -> L0. TLOAD can only target a Vec or Mat tile (its
  // destination must be TileType::Vec or TileType::Mat -- never Left/Right),
  // and TMATMUL can only read Left/Right operands, so an L1 (Mat) staging
  // tile plus a TMOV (L1 -> L0) step are mandatory. Loading straight from GM
  // into a TileLeft/TileRight is not expressible.

  // A-matrix L1 stage: S×S Mat tile (AscendC A1 / L1).
  // TLOAD from an ND GlobalTensor into this Mat tile performs the ND->NZ
  // conversion (replacing copy::CopyGmToL1A). ND->NZ requires the Mat tile's
  // SFractalSize == 512 and the source GlobalTensor's leading shape dims == 1.
  using TileL1A = pto::Tile<pto::TileType::Mat, InputT, S, S,
                            pto::BLayout::ColMajor, S, S,
                            pto::SLayout::RowMajor, 512>;
  // A-matrix L0A operand: S×S Left tile (AscendC A2 / L0A).
  // TMOV (Mat -> Left) copies the L1 stage here (replacing CopyL1ToL0A).
  using TileA = pto::TileLeft<InputT, S, S>;

  // B-matrix L1 stage: S×16 Mat tile of all-ones (AscendC B1 / L1).
  // TEXPANDS broadcasts constant 1 into this Mat tile (replacing
  // InitConstAllOnesL1). TEXPANDS also only targets Vec/Mat tiles, so the
  // all-ones matrix is likewise built in L1 and then moved to L0B.
  //
  // The staging tile mirrors the layout of its TMOV destination (TileB /
  // TileRight): BLayout::RowMajor + SLayout::ColMajor, matching how TileL1A
  // mirrors TileLeft. This is required for correctness on int8: a RowMajor
  // inner box packs C0 = alignedSize/sizeof(T) along the columns (32 for
  // int8), so a 16-wide tile would violate Cols % InnerCols == 0. The Right
  // operand's ColMajor inner box packs C0 along the rows instead, giving
  // InnerCols == fixedColSize == 16, so the 16-wide B tile is valid for every
  // element type.
  using TileL1B = pto::Tile<pto::TileType::Mat, InputT, S, 16,
                            pto::BLayout::RowMajor, S, 16,
                            pto::SLayout::ColMajor, 512>;
  // B-matrix L0B operand: S×16 Right tile (AscendC B2 / L0B).
  // TMOV (Mat -> Right) copies the L1 stage here (replacing CopyL1ToL0B).
  using TileB = pto::TileRight<InputT, S, 16>;

  // C-matrix accumulator: S×16 float (or int32_t for int8 input).
  // TileAcc maps to AscendC CO1 (L0C).  Two instances are needed for the
  // ping-pong accumulation pattern used by TMATMUL_ACC.
  using TileC = pto::TileAcc<OutputT, S, 16>;

  // ── GlobalTensor descriptors ─────────────────────────────────────────────
  // A-matrix input tile: S×S, ND layout.
  // AscendC: global_a_.SetGlobalBuffer((__gm__ InputT*)vec_in, vec_len)
  //   followed by CopyGmToL1A + CopyL1ToL0A -> TLOAD(tile_a, GTensorA(...))
  using AShape = pto::Shape<1, 1, 1, S, S>;
  using AStride = pto::BaseShape2D<InputT, S, S, pto::Layout::ND>;
  using GTensorA = pto::GlobalTensor<InputT, AShape, AStride, pto::Layout::ND>;

  // C-matrix output tile: S×16, ND layout.
  // AscendC: global_c_.SetGlobalBuffer((__gm__ OutputT*)cube_out, ...)
  //   + copy::CopyCL0ToGlobal (Fixpipe NZ2ND) -> TSTORE(GTensorC, tile_c)
  using CShape = pto::Shape<1, 1, 1, S, 16>;
  using CStride = pto::BaseShape2D<OutputT, S, 16, pto::Layout::ND>;
  using GTensorC = pto::GlobalTensor<OutputT, CShape, CStride, pto::Layout::ND>;

  // ── Derived constants ────────────────────────────────────────────────────
  constexpr uint32_t tile_len = S * S;   // elements per A tile
  constexpr uint32_t out_len = S * 16u;  // elements per C tile

  const uint32_t block_num = GetBlockNum();
  const uint32_t id = GetBlockIdx();

  // AscendC: num_mat_iters_ = FloorDiv(vec_len, tile_len)
  //          max_iters_per_block = CeilDiv(num_mat_iters_, block_num)
  const uint32_t num_mat_iters = tcuscan::scalar::FloorDiv(vec_len, tile_len);
  const uint32_t max_iters_per_block =
      tcuscan::scalar::CeilDiv(num_mat_iters, block_num);
  // Per-core element offset in the input vector.
  const uint32_t base_offset = id * tile_len * max_iters_per_block;

  // AscendC: GetWorkDistribution(vec_len_, tile_len_, block_num_)
  const uint32_t num_tiles =
      tcuscan::scalar::GetWorkDistribution(vec_len, tile_len, block_num);

  if (num_tiles == 0) {
    // This core has no work; still participate in the barrier if required.
    if constexpr (SyncAfter) {
      sync::SyncGroup<sync::GroupSyncDirection::FULL>();
    }
    return;
  }

  // ── Declare tile buffers ─────────────────────────────────────────────────
  // PTO auto-mode allocates L1/L0 space; no explicit TASSIGN is required.
  TileL1A a_l1;  // A staging buffer in L1
  TileL1B b_l1;  // B (all-ones) staging buffer in L1
  TileA tile_a;  // A operand in L0A
  TileB tile_b;  // B operand in L0B
  TileC tile_c0, tile_c1;  // ping-pong accumulators in L0C

  // ── Fill B with all-ones, reused across all iterations ───────────────────
  // AscendC: LoadAllOnesToL0() -> cube_unit::InitConstAllOnesL1 + CopyL1ToL0B
  // PTO: TEXPANDS broadcasts scalar 1 into the L1 (Mat) tile, then TMOV
  //      copies it L1 -> L0B (Mat -> Right).
  //
  // Pipe chain: TEXPANDS into a Mat tile runs on PIPE_MTE2 (Op::TEXPANDS_MAT),
  // TMOV (Mat -> Right) runs on PIPE_MTE1 (Op::TMOV_M2R), and TMATMUL runs on
  // PIPE_M. Each hop is a genuine cross-pipe dependency:
  //   TEXPANDS_MAT (MTE2) -> TMOV_M2R (MTE1) -> TMATMUL (M)
  // The MTE1->M edge only needs to be waited on by the very FIRST TMATMUL
  // below: once that instruction has synchronized against the L1->L0B move,
  // every later TMATMUL_ACC in the loop reads tile_b safely for free, because
  // PIPE_M executes its instructions strictly in issue order (same-pipe
  // dependencies are already hardware-ordered and must not be re-waited on --
  // Event<> even static_asserts SrcPipe != DstPipe, so an M->M "event" isn't
  // expressible).
  pto::Event<pto::Op::TEXPANDS_MAT, pto::Op::TMOV_M2R> eb_fill;
  eb_fill = TEXPANDS(b_l1, InputT(1));
  pto::Event<pto::Op::TMOV_M2R, pto::Op::TMATMUL> eb;
  eb = TMOV(tile_b, b_l1, eb_fill);

  // ── First tile: TMATMUL (no accumulation) ───────────────────────────────
  // AscendC: CubeIter<false>(ai_core_offset)
  //   -> CopyGmToL1A -> TLOAD(a_l1, ga0)            (GM -> L1, ND->NZ)
  //   -> CopyL1ToL0A -> TMOV(tile_a, a_l1)          (L1 -> L0A, Mat -> Left)
  //   -> cube_unit::Multiply<false> -> TMATMUL(tile_c0, tile_a, tile_b)
  GTensorA ga0(vec_in + base_offset);
  // TLOAD (MTE2) -> TMOV_M2L (MTE1) -> TMATMUL (M): two cross-pipe hops.
  pto::Event<pto::Op::TLOAD, pto::Op::TMOV_M2L> el0_load;
  el0_load = TLOAD(a_l1, ga0);
  pto::Event<pto::Op::TMOV_M2L, pto::Op::TMATMUL> el0;
  el0 = TMOV(tile_a, a_l1, el0_load);
  // TMATMUL waits on both the B move (eb) and the A move (el0).
  // Result is written to tile_c0.
  TMATMUL(tile_c0, tile_a, tile_b, el0, eb);

  // ── Remaining tiles: TMATMUL_ACC (accumulate) ────────────────────────────
  // AscendC: for idx in 1..num_tiles-1: CubeIter<true>(offset)
  //   -> CopyGmToL1A + CopyL1ToL0A -> TLOAD
  //   -> cube_unit::Multiply<true>  -> TMATMUL_ACC(c_out, c_in, a, b)
  //
  // Ping-pong rule:
  //   idx=0 -> tile_c0 written  (TMATMUL above)
  //   idx=1 -> tile_c1 written  (reads tile_c0)
  //   idx=2 -> tile_c0 written  (reads tile_c1)
  //   ...
  //
  // Each iteration re-stages A through L1 and L0A, so both hops of the
  // GM -> L1 -> L0A datapath need a fresh event every iteration:
  //   TLOAD (MTE2) writes a_l1, read by TMOV_M2L (MTE1);
  //   TMOV_M2L (MTE1) writes tile_a, read by TMATMUL_ACC (M).
  // The ping-pong reads of tile_c0/tile_c1 (written by the previous
  // TMATMUL/TMATMUL_ACC) and the repeated reads of tile_b are all consumed by
  // PIPE_M instructions that execute strictly after the PIPE_M instructions
  // that produced them, so no event is required (or expressible) for those
  // edges.
  for (uint32_t idx = 1; idx < num_tiles; idx++) {
    const uint32_t offset = base_offset + idx * tile_len;
    GTensorA ga(vec_in + offset);
    pto::Event<pto::Op::TLOAD, pto::Op::TMOV_M2L> el_load;
    el_load = TLOAD(a_l1, ga);
    pto::Event<pto::Op::TMOV_M2L, pto::Op::TMATMUL> el;
    el = TMOV(tile_a, a_l1, el_load);

    if (idx % 2 == 1) {
      // tile_c0 was the most-recent output; accumulate c0 into c1.
      TMATMUL_ACC(tile_c1, tile_c0, tile_a, tile_b, el);
    } else {
      // tile_c1 was the most-recent output; accumulate c1 into c0.
      TMATMUL_ACC(tile_c0, tile_c1, tile_a, tile_b, el);
    }
  }

  // ── Select the final accumulator after ping-pong ─────────────────────────
  // Iteration 0 writes tile_c0.
  // Odd idx  (1,3,...) writes tile_c1.
  // Even idx (2,4,...) writes tile_c0.
  // Final write is to tile_c0 when num_tiles is odd, tile_c1 when even.
  TileC& tile_c_final = (num_tiles % 2 == 1) ? tile_c0 : tile_c1;

  // ── Store S×16 accumulator to global memory ──────────────────────────────
  // AscendC: copy::CopyCL0ToGlobal(global_c_[id*out_len], co1_q_, S, 16)
  //   which invokes Fixpipe with NZ2ND conversion.
  // PTO: TSTORE from TileAcc performs the equivalent FixPipe path.
  //
  // TMATMUL/TMATMUL_ACC execute on PIPE_M, TSTORE (Acc->GM) executes on
  // PIPE_FIX, so this is a genuine cross-pipe dependency. em is recorded
  // explicitly here (rather than captured from a TMATMUL_ACC call inside the
  // loop above) because the M-pipe write we must wait for is whichever
  // TMATMUL/TMATMUL_ACC executed last; recording the event here, after all
  // PIPE_M instructions have been issued, is equivalent, since PIPE_M
  // processes its instructions (including the flag set by em.Record()) in
  // strict program order.
  GTensorC gc(cube_out + id * out_len);
  pto::Event<pto::Op::TMATMUL, pto::Op::TSTORE_ACC> em;
  em.Record();
  TSTORE(gc, tile_c_final, em);

  // AscendC: PipeBarrier<PIPE_ALL>() after CopyCL0ToGlobal.
  // PTO: no further explicit wait is required here. When SyncAfter is set,
  // sync::SyncGroup<FULL>()'s AIC branch issues its own
  // ffts_cross_core_sync() on PIPE_FIX -- the same pipe as the TSTORE above
  // -- so the store is already ordered before the barrier by same-pipe (FIX)
  // program order; no cross-pipe Event can (or needs to) bridge PIPE_FIX to
  // itself.
  if constexpr (SyncAfter) {
    // AscendC: sync::SyncGroup<sync::GroupSyncDirection::FULL>()
    sync::SyncGroup<sync::GroupSyncDirection::FULL>();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// pto_cube_reduce_aiv
//
// Vector-unit path (AIV cores).  Mirrors
// KernelCompleteCubeReduce<OutputT,SyncBefore>.
//
// Reads the S×16 per-cube-core output produced by pto_cube_reduce_aic and
// reduces it to a single scalar per AI-core group using TCOLSUM+TROWSUM+TMULS,
// then accumulates into the final output using an atomic-add TSTORE.
//
// Template parameters:
//   OutputT    - element type of the intermediate cube output (float / int32_t)
//   S          - compile-time matmul size
//   SyncBefore - if true, call SyncGroup<FULL>() before processing
// ─────────────────────────────────────────────────────────────────────────────
template <typename OutputT, uint32_t S, bool SyncBefore = false>
__aicore__ inline void pto_cube_reduce_aiv(__gm__ OutputT* vec_in,
                                           __gm__ OutputT* vec_out) {
  using namespace pto;
  // ── Derived constants ────────────────────────────────────────────────────
  constexpr uint32_t tile_len = S * 16u;  // elements in one S×16 cube output

  const uint32_t id = static_cast<uint32_t>(GetBlockIdx());
  const uint32_t aiv_id = static_cast<uint32_t>(GetSubBlockIdx());
  const uint32_t ratio = static_cast<uint32_t>(GetTaskRation());

  // AscendC: output_idx = FloorDiv(id, ratio)
  const uint32_t output_idx = tcuscan::scalar::FloorDiv(id, ratio);

  // ── Tile type aliases ────────────────────────────────────────────────────
  // rows_per_aiv rows of the S×16 cube output for this AIV sub-block.
  // AscendC: the scalar loop runs matmul_size_/ratio iterations (one per row
  // group).  The PTO tile is declared with the max row count S; the runtime
  // valid-row count is set to rows_per_aiv via SetValidRow below.
  // RowValid_ must be pto::DYNAMIC: SetValidRow() static_asserts on it
  // (pto_tile.hpp requires ValidRow == DYNAMIC to allow a runtime override).
  using TileVec = pto::Tile<pto::TileType::Vec, OutputT, S, 16,
                            pto::BLayout::RowMajor, pto::DYNAMIC>;

  // TCOLSUM: sum rows -> 1×16 vector.
  //   dst shape: 1 row × 16 cols, RowMajor ND.
  //   tmp shape: same as src (S×16), RowMajor ND (required by TCOLSUM).
  using TileColSum =
      pto::Tile<pto::TileType::Vec, OutputT, 1, 16, pto::BLayout::RowMajor>;
  using TileTmpColSum =
      pto::Tile<pto::TileType::Vec, OutputT, S, 16, pto::BLayout::RowMajor>;

  // TROWSUM: sum the 1×16 result -> 1×1 scalar.
  //   dst shape: 1 row x 1 valid col, RowMajor.  The underlying column
  //   capacity is padded to 16 (matching TileColSum/TileTmpRowSum) because
  //   the RowMajor+NoneBox static_assert in pto_tile.hpp requires
  //   Cols * sizeof(DType) to be 32-byte aligned; a literal 1x1 tile
  //   (4 bytes for float) can never satisfy that regardless of TileType::Vec.
  //   ValidCol is set to 1 so ops only touch the logical scalar element.
  //   tmp shape: matches src (1×16), RowMajor.
  using TileRowSum = pto::Tile<pto::TileType::Vec, OutputT, 1, 16,
                               pto::BLayout::RowMajor, 1, 1>;
  using TileTmpRowSum =
      pto::Tile<pto::TileType::Vec, OutputT, 1, 16, pto::BLayout::RowMajor>;

  // Scalar result tile: 1 row x 1 valid col, RowMajor (TMULS/TSHRS src and
  // dst requirement).  Column capacity padded to 16 for the same 32-byte
  // alignment reason as TileRowSum above.
  using TileScalar = pto::Tile<pto::TileType::Vec, OutputT, 1, 16,
                               pto::BLayout::RowMajor, 1, 1>;

  // ── GlobalTensor descriptors ─────────────────────────────────────────────
  // Cube intermediate output (S×16 ND).
  // AscendC: global_input_.SetGlobalBuffer((__gm__ OutputT*)vec_in, ...)
  //   + copy::CopyGmToVec -> TLOAD(tile_part, g_in)
  using InShape = pto::Shape<1, 1, 1, S, 16>;
  using InStride = pto::BaseShape2D<OutputT, S, 16, pto::Layout::ND>;
  using GTensorIn =
      pto::GlobalTensor<OutputT, InShape, InStride, pto::Layout::ND>;

  // Scalar output (1×1 ND, atomic-add destination).
  // AscendC: global_output_.SetGlobalBuffer((__gm__ OutputT*)vec_out,
  // block_num)
  using OutShape = pto::Shape<1, 1, 1, 1, 1>;
  using OutStride = pto::BaseShape2D<OutputT, 1, 1, pto::Layout::ND>;
  using GTensorOut =
      pto::GlobalTensor<OutputT, OutShape, OutStride, pto::Layout::ND>;

  // ── WriteOutZero ─────────────────────────────────────────────────────────
  // AscendC: WriteOutZero() -> aiv_id==0 writes scalar 0 to output slot
  //   via copy::CopyScalarToGm (no atomic mode).
  // PTO: TEXPANDS fills a 1×1 tile with 0; TSTORE writes it plain (no
  // AtomicAdd). The zero initialises the output slot so that subsequent
  // AtomicAdd stores accumulate onto a known-zero base.
  if (aiv_id == 0) {
    TileScalar tile_zero;
    // TEXPANDS executes on PIPE_V, TSTORE (Vec->GM) executes on PIPE_MTE3
    // (TSTORE_VEC), so the V->MTE3 handoff is a genuine cross-pipe
    // dependency.
    pto::Event<pto::Op::TEXPANDS, pto::Op::TSTORE_VEC> ef;
    ef = TEXPANDS(tile_zero, OutputT(0));
    GTensorOut g_out_zero(vec_out + output_idx);
    TSTORE(g_out_zero, tile_zero, ef);
    // No further explicit wait is required here: when SyncBefore is set (see
    // below), sync::SyncGroup<FULL>()'s AIV branch issues its own
    // ffts_cross_core_sync() on PIPE_MTE3 -- the same pipe as the TSTORE
    // above -- so the store is already ordered before the barrier by
    // same-pipe (MTE3) program order.
  }

  // AscendC: if constexpr (SyncBefore) { sync::SyncGroup<FULL>(); }
  if constexpr (SyncBefore) {
    sync::SyncGroup<sync::GroupSyncDirection::FULL>();
  }

  // ── ProcessTile ──────────────────────────────────────────────────────────
  // Each AIV sub-core owns rows_per_aiv consecutive rows of the S×16 cube
  // output tile.  The sub-block's portion starts at element aiv_elem_offset
  // in the global intermediate buffer.
  //
  // AscendC logic:
  //   aiv_core_offset = aiv_id * (tile_len / ratio)
  //   for i in 0..matmul_size_/ratio-1:
  //     sum += input_lt.GetValue(aiv_core_offset + i * 16)  // column 0 only
  //
  // PTO approach:
  //   1. TLOAD the rows_per_aiv×16 sub-tile.
  //   2. TCOLSUM: sum rows -> 1×16.
  //   3. TROWSUM: sum 1×16 -> 1×1.
  //   4. TMULS by 1/16 (float) or TSHRS by 4 (int32): cancel the 16-way
  //      column duplication.  TMATMUL produces 16 identical columns; the
  //      AscendC code only reads column 0, so the correct scalar is the
  //      column-sum divided by 16.
  //   5. TSTORE<AtomicAdd>: accumulate partial results from both AIV
  //      sub-blocks into output[output_idx].

  const uint32_t rows_per_aiv = S / ratio;
  // Byte offset in the global intermediate buffer for this sub-block's rows.
  // AscendC: gm_offset = output_idx * tile_len;
  //          aiv_core_offset = aiv_id * (tile_len / ratio)
  //   -> absolute element index = output_idx*tile_len + aiv_id*rows_per_aiv*16
  const uint32_t aiv_elem_offset =
      output_idx * tile_len + aiv_id * rows_per_aiv * 16u;

  // ── Declare tiles ────────────────────────────────────────────────────────
  TileVec tile_part;
  TileColSum tile_col_sum;
  TileTmpColSum tile_tmp_col;
  TileRowSum tile_row_sum;
  TileTmpRowSum tile_tmp_row;
  TileScalar tile_result;

  // Set the runtime valid-row count to rows_per_aiv for tile_part.
  // TCOLSUM reads src.GetValidRow() rows; tiles with smaller valid counts
  // than the static Rows are valid in PTO.
  // AscendC: the scalar loop runs matmul_size_/ratio iterations.
  tile_part.SetValidRow(static_cast<int>(rows_per_aiv));

  // Load rows_per_aiv rows of the S×16 cube output into tile_part.
  // AscendC: copy::CopyGmToVec(vecin_q_, global_input_[gm_offset], tile_len)
  //   followed by LocalTensor::GetValue accesses.
  //
  // TLOAD executes on PIPE_MTE2, TCOLSUM executes on PIPE_V, so the
  // MTE2->V handoff is a genuine cross-pipe dependency.
  GTensorIn g_in(vec_in + aiv_elem_offset);
  pto::Event<pto::Op::TLOAD, pto::Op::TCOLSUM> el;
  el = TLOAD(tile_part, g_in);

  // TCOLSUM: sum rows -> 1×16 column-sum vector.
  // The result tile_col_sum[0,j] = sum over all rows of tile_part[i,j].
  // isBinary=false uses sequential accumulation into dst.
  // tmp must match src's element type and ND layout (same as TileVec).
  TCOLSUM(tile_col_sum, tile_part, tile_tmp_col, /*isBinary=*/false, el);

  // TROWSUM: sum the 1×16 col-sum row -> 1×1 scalar.
  // tile_row_sum[0,0] = sum_j(tile_col_sum[0,j]).
  // This gives the sum of all 16 (identical) column totals; we correct for
  // the 16-way duplication below.
  // tmp must be the same shape as src (1×16, RowMajor).
  // A2A3 constraint: src.GetValidRow() == dst.GetValidRow() == 1.
  //
  // TCOLSUM and TROWSUM both execute on PIPE_V, so no event is needed (or
  // even expressible: Event<> static_asserts SrcPipe != DstPipe) between
  // them -- PIPE_V executes its instructions strictly in issue order.
  TROWSUM(tile_row_sum, tile_col_sum, tile_tmp_row);

  // Divide by 16 to recover the true partial sum:
  //   AscendC loops over column 0 only (matmul_size_/ratio elements),
  //   which equals (sum of all 16 identical columns) / 16.
  //
  // TMULS (float path): multiply by 1/16.
  //   A2A3 constraints: TileType::Vec, RowMajor, src.ValidRow==dst.ValidRow.
  // TSHRS (int32 path): arithmetic right-shift by 4 == divide by 16.
  //   A2A3 constraints: int32_t, TileType::Vec, ValidRow/Col must match.
  //   Scalar type is typename TileDataDst::DType = int32_t(4).
  //
  // TMULS/TSHRS also execute on PIPE_V (same pipe as TROWSUM above -- again
  // no event needed there), but their result feeds the final TSTORE, which
  // executes on PIPE_MTE3 (TSTORE_VEC): *that* transition is a genuine
  // cross-pipe dependency and needs an explicit Event.
  if constexpr (std::is_same_v<OutputT, float>) {
    pto::Event<pto::Op::TMULS, pto::Op::TSTORE_VEC> em;
    em = TMULS(tile_result, tile_row_sum, OutputT(1.0f / 16.0f));

    // AtomicAdd store: accumulate partial result from this AIV sub-block.
    // AscendC: AscendC::SetAtomicAdd<T>()
    //   + copy::CopyScalarToGm(global_output_[output_idx], res_out_q_, sum)
    //   + AscendC::SetAtomicNone()
    GTensorOut g_out(vec_out + output_idx);
    pto::TSTORE<TileScalar, GTensorOut, pto::AtomicType::AtomicAdd>(
        g_out, tile_result, em);
  } else {
    // int32_t path (int8_t input): shift-right by 4 == divide by 16.
    pto::Event<pto::Op::TSHRS, pto::Op::TSTORE_VEC> em;
    em = TSHRS(tile_result, tile_row_sum, OutputT(4));
    GTensorOut g_out(vec_out + output_idx);
    pto::TSTORE<TileScalar, GTensorOut, pto::AtomicType::AtomicAdd>(
        g_out, tile_result, em);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// run_pto_cube_reduce
//
// Top-level entry point.  Mirrors run_cube_reduce<T>.
//
// Handles optional padding (when vec_len is not aligned to S*S or
// UB_ALIGNMENT), then dispatches pto_cube_reduce_aic on AIC cores and
// pto_cube_reduce_aiv on AIV cores.
//
// Template parameters:
//   InputT - element type of the input vector (half / int8_t)
//   S      - compile-time matmul size; the caller must instantiate for each
//             concrete S value since PTO tile types require compile-time
//             shapes.
// ─────────────────────────────────────────────────────────────────────────────
template <typename InputT, uint32_t S>
__aicore__ inline void run_pto_cube_reduce(
    __gm__ InputT* vec_in,
    __gm__ tcuscan::cube_unit::CubeOutType_t<InputT>* vec_out,
    GM_ADDR workspace, uint32_t vec_len) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<InputT>;

  constexpr uint32_t align_size = S * S;
  const uint32_t padded_vec_len = tcuscan::scalar::AlignUp(vec_len, align_size);

  // Intermediate cube output (S×16 per cube core) lives at the start of
  // workspace by default; relocated past the padded input when padding is
  // needed.
  // AscendC: padded_cube_reductions starts at workspace.
  GM_ADDR padded_cube_reductions = workspace;

  if (vec_len % align_size != 0 || vec_len % UB_ALIGNMENT != 0) {
    // ── Padding pass ───────────────────────────────────────────────────────
    // AscendC: run_pad_kernel<InputT,false>(vec_in, workspace, vec_len,
    // align_size) This is an AscendC kernel; kept unchanged as the PTO spec
    // allows mixing.
    GM_ADDR const padded_input = workspace;
    run_pad_kernel<InputT, false>(reinterpret_cast<GM_ADDR>(vec_in),
                                  padded_input, vec_len, align_size);

    // Sync after padding so that the cube kernel reads the padded data.
    // AscendC: sync::SyncGroup<FULL>()
    sync::SyncGroup<sync::GroupSyncDirection::FULL>();

    // Point vec_in at the padded copy; advance the cube-output pointer past it.
    vec_len = padded_vec_len;
    vec_in = reinterpret_cast<__gm__ InputT*>(padded_input);
    padded_cube_reductions = workspace + padded_vec_len * sizeof(InputT);
  }

  __gm__ OutputT* inter_out =
      reinterpret_cast<__gm__ OutputT*>(padded_cube_reductions);

  // ── Dispatch AIC path ────────────────────────────────────────────────────
  // AscendC: if ASCEND_IS_AIC { KernelCubeReduce<T,true> op; op.Process(); }
  if ASCEND_IS_AIC {
    pto_cube_reduce_aic<InputT, S, true /*SyncAfter*/>(vec_in, inter_out,
                                                       vec_len);
  }

  // ── Dispatch AIV path ────────────────────────────────────────────────────
  // AscendC: if ASCEND_IS_AIV { KernelCompleteCubeReduce<OutputT,true> op;
  // op.Process(); }
  if ASCEND_IS_AIV {
    pto_cube_reduce_aiv<OutputT, S, true /*SyncBefore*/>(inter_out, vec_out);
  }
}

}  // namespace tcuscan
