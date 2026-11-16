/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file kernel_seg_sum_vec_revert.h
 * @brief Kernel implementing a revert speculation of segmented sum cube
 * operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Corrects the (speculative block scan) output of the `KernelRowScan` so
 * that the segmented sum of a vector is returned.
 *
 *
 * ### Example:
 * Given, vec_in = [1,2,3,4,5,6,7,8,9,10] and segm_ind_in = [0, 4, 7]
 * returns vec_out = (1+2+3+4, 5+6, 7+8+9+10) = (10, 11, 34)
 *
 * ### Algorithm
 *
 * The input is the output of `KernelRowScan`, i.e. an inclusive scan that is
 * local to each run of @p tile_len consecutive elements ("sub-tile"). Writing
 * \f$T(i)\f$ for the true prefix sum of the original (unscanned) vector over
 * \f$[0, i)\f$, the value stored at position \f$i\f$ is
 * \f$T(i+1) - T(\lfloor i/L \rfloor L)\f$ with \f$L = \textit{tile\_len}\f$.
 *
 * The sum of the segment ending at \f$e_j\f$ (exclusive) is
 * \f$T(e_j) - T(e_{j-1})\f$. Over a group of consecutive sub-tiles starting at
 * input position \f$B\f$, define
 *
 * \f[ u_j = T(e_j) - T(B) = P[r_j] + \textit{scan}[e_j - 1] \f]
 *
 * where \f$r_j\f$ is the index of the sub-tile holding \f$e_j - 1\f$ and
 * \f$P[r]\f$ is the sum of the totals of the sub-tiles preceding \f$r\f$ in the
 * group. Every segment that ends inside the group then has sum
 * \f$u_j - u_{j-1}\f$, a plain `diff` of gathered vectors, computed on the
 * vector unit. Only the first segment of each group needs the carry
 * \f$T(B) - T(e_{j-1})\f$ from the preceding group, which is kept in a scalar.
 *
 * The two parts of \f$u\f$ are gathered and differenced separately, as
 * \f$(\textit{scan}_j - \textit{scan}_{j-1}) + (P[r_j] - P[r_{j-1}])\f$.
 * Consecutive segments usually end in the same sub-tile, and then the second
 * term is one number minus itself and vanishes exactly, so the result is as
 * accurate as the within-sub-tile subtraction the scalar walk would have done,
 * however long the group is.
 *
 * Groups whose segment ends are too few to fill a batch skip the prefix table
 * and take `ScalarGroupWalk`, which carries a running accumulator across
 * sub-tiles exactly as the pre-vectorization kernel did. Group length itself
 * is picked from the input's segment density, so that a densely segmented
 * input gets full batches while a sparse one pays no per-group cost at all.
 *
 * @tparam T Input data type. Supports `float` and `int32_t`.
 * @tparam SyncBefore If true, the `KernelSegSumVecRevert` (this kernel) waits
 * for the cube unit to send a synchronization signal after each matrix tile is
 * ready. Matrix tile has length `tile_len * tile_len`.
 * @tparam UseAtomicWrite If true, the output is written using atomic-add
 * semantics.
 */
template <typename T, bool SyncBefore = false, bool UseAtomicWrite = false>
class KernelSegSumVecRevert {
  constexpr static uint32_t BUFFER_NUM = 2;

  /// @brief Maximum number of segment ends handled by one vectorized batch.
  constexpr static uint32_t SEG_BATCH_LEN = 512;
  /// @brief Vector instructions are issued on counts rounded up to this many
  /// elements, so that mask-producing ops (`Compare`) always see whole mask
  /// words. The lanes past the real count are clamped to a valid gather
  /// position and never written out.
  constexpr static uint32_t VEC_ROUND = 64;
  /// @brief Number of segment ends a group aims to hold. Groups are sized
  /// from the input's segment density so that a batch is worth vectorizing,
  /// while a sparsely segmented input gets groups large enough that the
  /// per-group bookkeeping disappears against the per-sub-tile work.
  constexpr static uint32_t GROUP_TARGET_SEGMENTS = SEG_BATCH_LEN;
  /// @brief Batches with at most this many segments are handled by the scalar
  /// path, for which the fixed cost of the vector path does not pay off. Such
  /// batches accumulate into the pending output tile, so a long run of them
  /// still stores at full width.
  constexpr static uint32_t SCALAR_BATCH_THRESHOLD = 32;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Input vector length.
   * @param [in] num_segments Number of segments.
   * @param [in] tile_len Tile length.
   * @param [in] vec_start_offset Start offset of input data vector. Segment
   * values will be offset accordindly. Default value is `0`.
   * @param [in] alpha Scaling factor applied to every segment sum on write.
   * Default value is `1` (no scaling). Scaling at the output stage (instead of
   * scaling the input vector) keeps the running scan accumulator unscaled and
   * distributes over the atomic-add writes of overlapping blocks.
   */
  __aicore__ inline KernelSegSumVecRevert(uint32_t vec_len,
                                          uint32_t num_segments,
                                          uint32_t tile_len,
                                          uint32_t vec_start_offset = 0,
                                          T alpha = 1)
      : vec_len_(vec_len),
        num_segments_(num_segments),
        tile_len_(tile_len),
        matrix_tile_len_(tile_len * tile_len),
        num_tiles_(scalar::CeilDiv(vec_len_, matrix_tile_len_)),
        vec_start_offset_(vec_start_offset),
        alpha_(alpha),
        log2_tile_len_(scalar::Log2Exact(tile_len)),
        // The vectorized path addresses sub-tiles with a shift, so it requires
        // a power-of-two tile. `KernelRowScan` only ever runs with matmul sizes
        // 16/32/64/128, but fall back to the scalar walk if that ever changes.
        is_vectorizable_(log2_tile_len_ != scalar::LOG2_NOT_EXACT),
        group_len_(
            ComputeGroupLen(tile_len, matrix_tile_len_, vec_len, num_segments)),
        seg_buf_len_(scalar::AlignUp(SEG_BATCH_LEN + 1, VEC_ROUND)) {
    constexpr bool IS_DT_SUPPORTED =
        std::is_same_v<T, float> || std::is_same_v<T, int32_t>;
    static_assert(IS_DT_SUPPORTED, "Unsupported data type.");
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_in Pointer to the input vector in global memory.
   * @param [in] segm_ind_in Pointer to the segment indices vector in global
   * memory.
   * @param [in] vec_out Pointer to the output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR segm_ind_in,
                              GM_ADDR vec_out) {
    global_in_.SetGlobalBuffer((__gm__ T*)vec_in + vec_start_offset_, vec_len_);
    global_segm_in_.SetGlobalBuffer((__gm__ uint32_t*)segm_ind_in,
                                    num_segments_);
    global_out_.SetGlobalBuffer((__gm__ T*)vec_out, num_segments_ + 1);

    pipe_.InitBuffer(in_q_, BUFFER_NUM, matrix_tile_len_ * sizeof(T));

    if (is_vectorizable_) {
      pipe_.InitBuffer(segm_q_, BUFFER_NUM, SEG_BATCH_LEN * sizeof(uint32_t));
      pipe_.InitBuffer(out_q_, 1, SEG_BATCH_LEN * sizeof(T));

      pipe_.InitBuffer(sidx_buf_, seg_buf_len_ * sizeof(int32_t));
      pipe_.InitBuffer(segb_buf_, seg_buf_len_ * sizeof(uint32_t));
      pipe_.InitBuffer(pos_buf_, seg_buf_len_ * sizeof(int32_t));
      pipe_.InitBuffer(posc_buf_, seg_buf_len_ * sizeof(int32_t));
      pipe_.InitBuffer(sel_buf_, seg_buf_len_ * sizeof(uint8_t));
      pipe_.InitBuffer(gval_buf_, seg_buf_len_ * sizeof(T));
      pipe_.InitBuffer(pval_buf_, seg_buf_len_ * sizeof(T));
      pipe_.InitBuffer(u_buf_, seg_buf_len_ * sizeof(T));
      pipe_.InitBuffer(fwd_buf_, seg_buf_len_ * sizeof(T));
      pipe_.InitBuffer(fidx_buf_, seg_buf_len_ * sizeof(int32_t));
      // One entry per sub-tile of a group, plus the group total.
      pipe_.InitBuffer(ptab_buf_, (group_len_ / tile_len_ + 1) * sizeof(T));
    } else {
      pipe_.InitBuffer(segm_q_, BUFFER_NUM, tile_len_ * sizeof(uint32_t));
      pipe_.InitBuffer(out_q_, 1, tile_len_ * sizeof(T));
    }
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    if (GetSubBlockIdx() == 0) {
      if (is_vectorizable_) {
        PipelineProcessVectorized();
      } else {
        PipelineProcessWithCube();
      }
    } else {
      SyncWithCubeNoop();
    }
  }

 private:
  /**
   * @brief Number of input elements handled by one vectorized group.
   *
   * A multiple of @p tile_len, never larger than one matrix tile.
   */
  __aicore__ static inline uint32_t ComputeGroupLen(uint32_t tile_len,
                                                    uint32_t matrix_tile_len,
                                                    uint32_t vec_len,
                                                    uint32_t num_segments) {
    if (num_segments == 0 || vec_len == 0) {
      return matrix_tile_len;
    }
    // Inputs averaging many elements per segment get the largest group, which
    // costs one dispatch probe per matrix tile and nothing else; dense inputs
    // get a group just long enough to fill a batch.
    const uint32_t elems_per_segment = scalar::CeilDiv(vec_len, num_segments);
    uint32_t group_len = matrix_tile_len;
    if (elems_per_segment <= matrix_tile_len / GROUP_TARGET_SEGMENTS) {
      group_len =
          scalar::AlignUp(GROUP_TARGET_SEGMENTS * elems_per_segment, tile_len);
    }
    if (group_len > matrix_tile_len) {
      group_len = matrix_tile_len;
    }
    if (group_len < tile_len) {
      group_len = tile_len;
    }
    return group_len;
  }

  __aicore__ inline void SyncWithCubeNoop() {
    for (uint32_t tile_idx = 0; tile_idx < num_tiles_; tile_idx++) {
      if constexpr (SyncBefore) {
        sync::SyncGroup<sync::GroupSyncDirection::FULL>();
      }
    }
  }

  /**
   * @brief Load the next tile of segment ends into @p segm_lt_.
   *
   * Tiles start at multiples of `SEG_BATCH_LEN` so that the global memory
   * reads stay aligned.
   */
  __aicore__ inline void LoadSegTile(uint32_t base) {
    if (seg_tile_valid_) {
      segm_q_.template FreeTensor<uint32_t>(segm_lt_);
    }
    seg_tile_base_ = base;
    seg_tile_count_ =
        scalar::NextTileLen(SEG_BATCH_LEN, seg_tile_base_, num_segments_);
    copy::CopyGmToVec(segm_q_, global_segm_in_[seg_tile_base_],
                      seg_tile_count_);
    segm_lt_ = segm_q_.template DeQue<uint32_t>();
    // The segment ends are read with `GetValue`, which `DeQue` alone does not
    // order against the copy-in.
    sync::ScalarWaitForMte2();
    seg_tile_valid_ = true;
  }

  /**
   * @brief Segment end at global segment index @p idx, in input-local
   * coordinates. Requires the holding tile to be loaded.
   */
  __aicore__ inline uint32_t SegEndAt(uint32_t idx) {
    return static_cast<uint32_t>(segm_lt_.GetValue(idx - seg_tile_base_) -
                                 vec_start_offset_);
  }

  /**
   * @brief Number of segment ends, starting at @p from, that are strictly below
   * @p limit and held by the currently loaded segment tile.
   *
   * Binary search over the sorted ends; `O(log)` scalar reads per batch.
   */
  __aicore__ inline uint32_t CountEndsBelow(uint32_t from, uint32_t limit) {
    uint32_t lo = from;
    uint32_t hi = seg_tile_base_ + seg_tile_count_;
    if (lo >= hi || SegEndAt(lo) >= limit) {
      return 0;
    }
    // Invariant: SegEndAt(lo) < limit, and hi is past the last end below it.
    while (hi - lo > 1) {
      const uint32_t mid = lo + (hi - lo) / 2;
      if (SegEndAt(mid) < limit) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    return lo + 1 - from;
  }

  /**
   * @brief Whether a group holds more than `SCALAR_BATCH_THRESHOLD` segment
   * ends, and so is worth the prefix table and the vector path.
   *
   * Answers with a single probe rather than a count, because this runs once
   * per group whether or not the group has any segments at all, and a sparse
   * input must not pay for it.
   *
   * @param [in] from Index of the next unprocessed segment.
   * @param [in] limit Input position the group ends at.
   */
  __aicore__ inline bool IsGroupDense(uint32_t from, uint32_t limit) {
    const uint32_t probe = from + SCALAR_BATCH_THRESHOLD;
    if (probe >= num_segments_) {
      return false;
    }
    // The probe has to be in the loaded tile to be read. Re-anchoring on the
    // cursor re-reads fewer than `SCALAR_BATCH_THRESHOLD` entries, since the
    // probe is that far ahead of it.
    if (probe >= seg_tile_base_ + seg_tile_count_) {
      LoadSegTile(from);
      if (probe >= seg_tile_base_ + seg_tile_count_) {
        return false;
      }
    }
    return SegEndAt(probe) < limit;
  }

  /**
   * @brief The two parts of \f$u(e) = T(e) - T(B)\f$ for a segment end @p e
   * inside the group starting at \f$B\f$, computed with scalar reads.
   *
   * Kept apart rather than summed so that the caller can difference them
   * separately; see `ProcessBatchVectorized`.
   *
   * @param [in] vec_in_lt Matrix tile resident in Unified Buffer.
   * @param [in] ptab_lt Sub-tile prefix table of the group.
   * @param [in] group_ub_offset Offset of the group inside @p vec_in_lt.
   * @param [in] end_in_group Segment end relative to the group start.
   * @param [out] g The segment's own sub-tile-local scan value.
   * @param [out] p The prefix of the sub-tiles preceding it in the group.
   */
  __aicore__ inline void ScalarGP(const LocalTensor<T>& vec_in_lt,
                                  const LocalTensor<T>& ptab_lt,
                                  uint32_t group_ub_offset,
                                  uint32_t end_in_group, T& g, T& p) {
    if (end_in_group == 0) {
      g = static_cast<T>(0);
      p = static_cast<T>(0);
      return;
    }
    const uint32_t pos = end_in_group - 1;
    p = ptab_lt.GetValue(pos >> log2_tile_len_);
    g = vec_in_lt.GetValue(group_ub_offset + pos);
  }

  /**
   * @brief Set the elements of @p data whose @p sel bit is 0 to zero, in place.
   *
   * `Select` only operates on `half`/`float`; for `int32_t` the data is
   * reinterpreted as `float` since the selection is a bitwise copy and a zeroed
   * `float` shares the bit pattern of a zeroed `int32_t`.
   */
  __aicore__ inline void ZeroWhereNotSelected(const LocalTensor<T>& data,
                                              const LocalTensor<uint8_t>& sel,
                                              uint32_t count) {
    if constexpr (std::is_same_v<T, int32_t>) {
      const LocalTensor<float> data_f = data.template ReinterpretCast<float>();
      Select(data_f, sel, data_f, static_cast<float>(0),
             SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    } else {
      Select(data, sel, data, static_cast<T>(0),
             SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    }
  }

  /**
   * @brief Build the sub-tile prefix table \f$P\f$ of one group.
   *
   * `P[r]` is the sum of the totals of the group's sub-tiles before `r`, so
   * `P[0] == 0` and `P[num_sub_tiles]` is the group total. The sub-tile total
   * is the last element of its local scan.
   *
   * @return The group total, i.e. `P[num_sub_tiles]`.
   */
  __aicore__ inline T BuildPrefixTable(const LocalTensor<T>& vec_in_lt,
                                       const LocalTensor<T>& ptab_lt,
                                       uint32_t group_ub_offset,
                                       uint32_t group_elems) {
    const uint32_t num_sub_tiles = scalar::CeilDiv(group_elems, tile_len_);
    T running = 0;
    ptab_lt.SetValue(0, running);
    for (uint32_t r = 0; r < num_sub_tiles; r++) {
      const uint32_t sub_tile_end =
          scalar::Min<uint32_t>((r + 1) * tile_len_, group_elems);
      running += vec_in_lt.GetValue(group_ub_offset + sub_tile_end - 1);
      ptab_lt.SetValue(r + 1, running);
    }
    return running;
  }

  /**
   * @brief Emit the sums of @p count segments ending inside the current group,
   * using the vector unit.
   *
   * Gathers the two parts of `u` for the batch, differences each with the
   * `Gather`-based shift, and writes the result straight to its place in the
   * output. The first element of the difference is fixed up from
   * @p prev_g / @p prev_p, which describe the segment preceding the batch and
   * are the only values the vector path cannot supply; at the start of a group
   * that predecessor lies outside it and @p carry stands in for it instead.
   *
   * @param [in,out] prev_g,prev_p The preceding segment's parts, updated to
   * the last segment of this batch.
   * @param [in] first_in_group Whether the predecessor is outside this group,
   * in which case @p carry describes it.
   */
  __aicore__ inline void ProcessBatchVectorized(
      const LocalTensor<T>& vec_in_lt, const LocalTensor<T>& ptab_lt,
      uint32_t group_ub_offset, uint32_t group_start, uint32_t group_elems,
      uint32_t first_seg, uint32_t count, bool first_in_group, T carry,
      T& prev_g, T& prev_p) {
    // One lead-in lane holds the preceding segment, whose difference partner is
    // supplied below; the gathered value itself is never used.
    const uint32_t num_lanes = count + 1;
    const uint32_t vec_count = scalar::AlignUp(num_lanes, VEC_ROUND);

    // This batch stores straight from its own output tile, so anything still
    // pending has to reach global memory first to keep the output in order.
    FlushPending();

    // `ptab_lt` was just filled with `SetValue`; the gather below reads it.
    sync::VecWaitForScalar();

    const LocalTensor<int32_t> sidx_lt = sidx_buf_.template Get<int32_t>();
    const LocalTensor<uint32_t> segb_lt = segb_buf_.template Get<uint32_t>();
    const LocalTensor<int32_t> pos_lt = pos_buf_.template Get<int32_t>();
    const LocalTensor<int32_t> posc_lt = posc_buf_.template Get<int32_t>();
    const LocalTensor<uint8_t> sel_lt = sel_buf_.template Get<uint8_t>();
    const LocalTensor<T> gval_lt = gval_buf_.template Get<T>();
    const LocalTensor<T> pval_lt = pval_buf_.template Get<T>();
    const LocalTensor<T> u_lt = u_buf_.template Get<T>();
    const LocalTensor<T> fwd_lt = fwd_buf_.template Get<T>();

    // Gather the batch's segment ends preceded by one lead-in lane, i.e. the
    // local indices [lo - 1, lo + count) of the loaded segment tile. `Maxs`
    // keeps the lead-in in bounds when the batch starts the tile; `Mins` keeps
    // the rounding lanes inside the loaded data.
    const int32_t lo = static_cast<int32_t>(first_seg - seg_tile_base_);
    ArithProgression<int32_t>(
        sidx_lt, (lo - 1) * static_cast<int32_t>(sizeof(uint32_t)),
        static_cast<int32_t>(sizeof(uint32_t)), vec_count);
    Maxs(sidx_lt, sidx_lt, 0, vec_count);
    Mins(sidx_lt, sidx_lt,
         static_cast<int32_t>((seg_tile_count_ - 1) * sizeof(uint32_t)),
         vec_count);
    AscendC::Gather(segb_lt, segm_lt_,
                    sidx_lt.template ReinterpretCast<uint32_t>(),
                    static_cast<uint32_t>(0), vec_count);

    // Position of the last element of each segment, relative to the group.
    // A segment ending exactly on the group start yields -1 and contributes 0.
    const int32_t shift =
        static_cast<int32_t>(vec_start_offset_ + group_start) + 1;
    Adds(pos_lt, segb_lt.template ReinterpretCast<int32_t>(), -shift,
         vec_count);
    Maxs(posc_lt, pos_lt, 0, vec_count);
    Mins(posc_lt, posc_lt, static_cast<int32_t>(group_elems - 1), vec_count);
    // A lane is live exactly when clamping did not move it.
    Compare(sel_lt, pos_lt, posc_lt, CMPMODE::EQ, vec_count);

    // scan[e - 1], zeroed on the lanes clamped above.
    Adds(sidx_lt, posc_lt, static_cast<int32_t>(group_ub_offset), vec_count);
    Muls(sidx_lt, sidx_lt, static_cast<int32_t>(sizeof(T)), vec_count);
    AscendC::Gather(gval_lt, vec_in_lt,
                    sidx_lt.template ReinterpretCast<uint32_t>(),
                    static_cast<uint32_t>(0), vec_count);
    ZeroWhereNotSelected(gval_lt, sel_lt, vec_count);

    // P[r], the prefix of the sub-tiles preceding each segment's own sub-tile.
    ShiftRight(sidx_lt.template ReinterpretCast<uint32_t>(),
               posc_lt.template ReinterpretCast<uint32_t>(), log2_tile_len_,
               vec_count);
    Muls(sidx_lt, sidx_lt, static_cast<int32_t>(sizeof(T)), vec_count);
    AscendC::Gather(pval_lt, ptab_lt,
                    sidx_lt.template ReinterpretCast<uint32_t>(),
                    static_cast<uint32_t>(0), vec_count);

    // A segment's sum is the difference of consecutive `u = p + g`. The two
    // parts are differenced separately: consecutive segments usually share a
    // sub-tile, and then `p` is the same number twice and cancels exactly,
    // which keeps the result as accurate as a within-sub-tile subtraction no
    // matter how long the group is. `fwd_lt[t]` holds entry `t + 1`,
    // following the shift-by-gather idiom of `KernelDiff`.
    LocalTensor<T> out_lt = out_q_.template AllocTensor<T>();
    AscendC::Gather(fwd_lt, gval_lt,
                    fidx_lt_.template ReinterpretCast<uint32_t>(),
                    static_cast<uint32_t>(0), count);
    Sub(out_lt, fwd_lt, gval_lt, count);
    AscendC::Gather(fwd_lt, pval_lt,
                    fidx_lt_.template ReinterpretCast<uint32_t>(),
                    static_cast<uint32_t>(0), count);
    Sub(u_lt, fwd_lt, pval_lt, count);
    Add(out_lt, out_lt, u_lt, count);
    if (alpha_ != static_cast<T>(1)) {
      Muls(out_lt, out_lt, alpha_, count);
    }

    // The lead-in lane carries no usable value, so the batch's first
    // difference is recomputed on the scalar unit against the predecessor the
    // caller tracked (as `KernelDiff` does for its boundary element).
    sync::ScalarWaitForVec();
    T first_g, first_p;
    ScalarGP(vec_in_lt, ptab_lt, group_ub_offset,
             SegEndAt(first_seg) - group_start, first_g, first_p);
    out_lt.SetValue(0, alpha_ * (first_in_group ? (carry + first_p + first_g)
                                                : ((first_g - prev_g) +
                                                   (first_p - prev_p))));

    out_q_.template EnQue<T>(out_lt);
    WriteOut(count);

    ScalarGP(vec_in_lt, ptab_lt, group_ub_offset,
             SegEndAt(first_seg + count - 1) - group_start, prev_g, prev_p);
  }

  /**
   * @brief Emit the sums of @p count segments with scalar reads.
   *
   * Used for batches too small to amortize the vector path. The sums are
   * accumulated in the pending output tile rather than written straight out,
   * so that a long run of small batches still produces full-width stores.
   *
   * @param [in,out] prev_g,prev_p The preceding segment's parts, updated to
   * the last segment of this batch.
   * @param [in] first_in_group Whether the predecessor is outside this group,
   * in which case @p carry describes it.
   */
  __aicore__ inline void ProcessBatchScalar(
      const LocalTensor<T>& vec_in_lt, const LocalTensor<T>& ptab_lt,
      uint32_t group_ub_offset, uint32_t group_start, uint32_t first_seg,
      uint32_t count, bool first_in_group, T carry, T& prev_g, T& prev_p) {
    for (uint32_t t = 0; t < count; t++) {
      T g, p;
      ScalarGP(vec_in_lt, ptab_lt, group_ub_offset,
               SegEndAt(first_seg + t) - group_start, g, p);
      AppendPending(alpha_ * (first_in_group && t == 0
                                  ? (carry + p + g)
                                  : ((g - prev_g) + (p - prev_p))));
      prev_g = g;
      prev_p = p;
    }
  }

  /**
   * @brief Emit every segment sum of one group with a running accumulator.
   *
   * Walks the group's sub-tiles in order, carrying
   * \f$\textit{acc} = T(\textit{current position}) - T(e_{\textit{last}})\f$
   * across them, so no prefix table is needed and the only input reads are the
   * one sub-tile total per sub-tile that the arithmetic requires regardless.
   * This is the cheaper path when a group holds few segment ends.
   *
   * @param [in] vec_in_lt Matrix tile resident in Unified Buffer.
   * @param [in] group_ub_offset Offset of the group inside @p vec_in_lt.
   * @param [in] group_start Input position of the group start.
   * @param [in] group_elems Number of input elements in the group.
   * @param [in,out] seg_cursor Index of the next segment; advanced past every
   * segment ending inside the group.
   * @param [in] carry \f$T(\textit{group\_start}) - T(e_{\textit{last}})\f$.
   * @return The same quantity taken at the group end.
   */
  __aicore__ inline T ScalarGroupWalk(const LocalTensor<T>& vec_in_lt,
                                      uint32_t group_ub_offset,
                                      uint32_t group_start,
                                      uint32_t group_elems,
                                      uint32_t& seg_cursor, T carry) {
    T acc = carry;
    uint32_t sub_start = 0;
    // The end of the segment at the cursor, held across sub-tiles so that a
    // sub-tile containing no segment costs one comparison rather than a read.
    uint32_t end = NextEnd(seg_cursor);

    while (sub_start < group_elems) {
      const uint32_t sub_len =
          scalar::NextTileLen(tile_len_, sub_start, group_elems);
      const uint32_t sub_end = group_start + sub_start + sub_len;

      while (end < sub_end) {
        // The last value of the segment, zero when it ends on the sub-tile
        // boundary and so contributes nothing beyond the running accumulator.
        const T delta =
            (end == group_start + sub_start)
                ? static_cast<T>(0)
                : vec_in_lt.GetValue(group_ub_offset + (end - group_start) - 1);
        AppendPending(alpha_ * (acc + delta));
        acc = -delta;
        seg_cursor++;
        end = NextEnd(seg_cursor);
      }

      acc += vec_in_lt.GetValue(group_ub_offset + sub_start + sub_len - 1);
      sub_start += sub_len;
    }
    return acc;
  }

  /**
   * @brief End of the segment at @p cursor, loading its tile if needed.
   *
   * Returns a value past every input position once the segments run out, so
   * that callers can stop on the comparison they already make.
   */
  __aicore__ inline uint32_t NextEnd(uint32_t cursor) {
    if (cursor >= num_segments_) {
      return UINT32_MAX;
    }
    if (cursor >= seg_tile_base_ + seg_tile_count_) {
      LoadSegTile(cursor);
    }
    return SegEndAt(cursor);
  }

  /**
   * @brief Append one segment sum to the pending output tile, flushing it once
   * full.
   */
  __aicore__ inline void AppendPending(T value) {
    if (!pend_valid_) {
      pend_lt_ = out_q_.template AllocTensor<T>();
      pend_count_ = 0;
      pend_valid_ = true;
    }
    pend_lt_.SetValue(pend_count_, value);
    pend_count_++;
    if (pend_count_ >= SEG_BATCH_LEN) {
      FlushPending();
    }
  }

  /**
   * @brief Write out whatever the pending output tile holds.
   *
   * Must run before anything else claims the output queue, and before the
   * kernel returns.
   */
  __aicore__ inline void FlushPending() {
    if (!pend_valid_) {
      return;
    }
    if (pend_count_ == 0) {
      out_q_.template FreeTensor<T>(pend_lt_);
    } else {
      out_q_.template EnQue<T>(pend_lt_);
      WriteOut(pend_count_);
    }
    pend_count_ = 0;
    pend_valid_ = false;
  }

  /**
   * @brief Write @p count already-enqueued segment sums at the output cursor.
   */
  __aicore__ inline void WriteOut(uint32_t count) {
    if constexpr (UseAtomicWrite) {
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::SetAtomicAdd<T>();
    }
    copy::CopyVecToGm(global_out_[out_offset_], out_q_, count);
    if constexpr (UseAtomicWrite) {
      AscendC::SetAtomicNone();
    }
    out_offset_ += count;
  }

  /**
   * @brief Run the kernel with the vectorized segment walk.
   */
  __aicore__ inline void PipelineProcessVectorized() {
    // Constant byte offsets of the shift-by-one gather, built once.
    fidx_lt_ = fidx_buf_.template Get<int32_t>();
    ArithProgression<int32_t>(fidx_lt_, static_cast<int32_t>(sizeof(T)),
                              static_cast<int32_t>(sizeof(T)), seg_buf_len_);

    const LocalTensor<T> ptab_lt = ptab_buf_.template Get<T>();

    if (num_segments_ > 0) {
      LoadSegTile(0);
    }

    T carry = 0;
    uint32_t seg_cursor = 0;
    uint32_t global_in_offset = 0;

    for (uint32_t matrix_tile_idx = 0; matrix_tile_idx < num_tiles_;
         matrix_tile_idx++) {
      if constexpr (SyncBefore) {
        sync::SyncGroup<sync::GroupSyncDirection::FULL>();
      }

      const uint32_t num_elems_to_copy =
          scalar::NextTileLen(matrix_tile_len_, global_in_offset, vec_len_);
      copy::CopyGmToVec(in_q_, global_in_[global_in_offset], num_elems_to_copy);
      LocalTensor<T> vec_in_lt = in_q_.template DeQue<T>();
      // Sub-tile totals are read with `GetValue`; order the scalar unit
      // against the copy-in.
      sync::ScalarWaitForMte2();

      for (uint32_t group_ub_offset = 0; group_ub_offset < num_elems_to_copy;
           group_ub_offset += group_len_) {
        const uint32_t group_elems =
            scalar::NextTileLen(group_len_, group_ub_offset, num_elems_to_copy);
        const uint32_t group_start = global_in_offset + group_ub_offset;
        const uint32_t group_end = group_start + group_elems;

        if (seg_cursor < num_segments_ &&
            seg_cursor >= seg_tile_base_ + seg_tile_count_) {
          LoadSegTile(seg_cursor);
        }

        // Groups holding few segments go down a walk that needs no prefix
        // table, which keeps the cost of a sparsely-segmented input at the
        // one sub-tile total per sub-tile that the arithmetic needs anyway.
        if (!IsGroupDense(seg_cursor, group_end)) {
          carry = ScalarGroupWalk(vec_in_lt, group_ub_offset, group_start,
                                  group_elems, seg_cursor, carry);
          continue;
        }

        const T group_total =
            BuildPrefixTable(vec_in_lt, ptab_lt, group_ub_offset, group_elems);

        // The parts of `u` for the segment preceding the one being emitted, in
        // this group's coordinates. Until the group's first segment is out,
        // the predecessor lies before the group and `carry` stands for it.
        T prev_g = 0;
        T prev_p = 0;
        bool first_in_group = true;

        while (seg_cursor < num_segments_) {
          if (seg_cursor >= seg_tile_base_ + seg_tile_count_) {
            LoadSegTile(seg_cursor);
          }
          const uint32_t count = CountEndsBelow(seg_cursor, group_end);
          if (count == 0) {
            break;
          }

          if (count <= SCALAR_BATCH_THRESHOLD) {
            ProcessBatchScalar(vec_in_lt, ptab_lt, group_ub_offset, group_start,
                               seg_cursor, count, first_in_group, carry, prev_g,
                               prev_p);
          } else {
            ProcessBatchVectorized(vec_in_lt, ptab_lt, group_ub_offset,
                                   group_start, group_elems, seg_cursor, count,
                                   first_in_group, carry, prev_g, prev_p);
          }
          first_in_group = false;
          seg_cursor += count;

          // A batch stops either at the group end or at the end of the loaded
          // segment tile; only the latter leaves work for another batch.
          if (seg_cursor < seg_tile_base_ + seg_tile_count_) {
            break;
          }
        }

        // `carry = T(group_end) - T(e_last)`. A group that emitted nothing
        // just lengthens the span the incoming carry already covers.
        if (!first_in_group) {
          carry = group_total - (prev_p + prev_g);
        } else {
          carry += group_total;
        }
      }

      in_q_.template FreeTensor<T>(vec_in_lt);
      global_in_offset += num_elems_to_copy;
    }

    if (seg_tile_valid_) {
      segm_q_.template FreeTensor<uint32_t>(segm_lt_);
      seg_tile_valid_ = false;
    }

    // The trailing segment holds whatever is left after the last emitted end.
    AppendPending(alpha_ * carry);
    FlushPending();
  }

  /**
   * @brief Returns a local tensor that contains the next segment indices tile
   * (of length up to `tile_len_`)
   *
   * @return Segments index vector of tile length at most `tile_len_`.
   */
  __aicore__ inline LocalTensor<uint32_t> LoadNextSegmentTile() {
    const uint32_t next_tile_len =
        scalar::NextTileLen(tile_len_, segments_offset_, num_segments_);
    copy::CopyGmToVec(segm_q_, global_segm_in_[segments_offset_],
                      next_tile_len);

    return segm_q_.template DeQue<uint32_t>();
  }

  /**
   * @brief Prepare for output writing of the current value on the given
   * index.
   *
   * @param vec_out_lt Tile of output containing the segmented sums.
   * @param index Tile index to write on.
   * @param value Value to write.
   */
  __aicore__ inline void SafeOutWrite(LocalTensor<T>& vec_out_lt,
                                      uint32_t& index, const T value) {
    vec_out_lt.SetValue(index, value);
    index++;
    // Write tile to GM and "re-allocate" the output tile.
    if (index >= tile_len_) {
      out_q_.template EnQue<T>(vec_out_lt);
      if constexpr (UseAtomicWrite) {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SetAtomicAdd<T>();
      }
      copy::CopyVecToGm(global_out_[out_offset_], out_q_, tile_len_);
      if constexpr (UseAtomicWrite) {
        AscendC::SetAtomicNone();
      }
      out_offset_ += tile_len_;
      vec_out_lt = out_q_.template AllocTensor<T>();

      // "Local-coordinates" of local tensor index must be zeroed
      index = 0;
    }
  }

  /**
   * @brief Returns the index corresponding to the next segment end.
   *
   * Method follows the iterator pattern. At termination, returns `vec_len_`.
   *
   * @param [in] segm_ind_lt Tile containing the segment-ending indices.
   * @param [in] index Current "local" index within the @p segm_ind_lt tile.
   * @return Returns the next segment end.
   */
  __aicore__ inline uint32_t NextSegmEndIndex(
      LocalTensor<uint32_t>& segm_ind_lt, uint32_t& index) {
    index++;

    if (segments_offset_ + index >= num_segments_) {
      // Run out-of-segments, set the segment end to maximum
      // vector length.
      return vec_len_;
    } else if (index >= tile_len_) {
      // Run out-of-segments in the segment tiles. Just load the
      // next segment tile and update counters/indices
      segm_q_.template FreeTensor<uint32_t>(segm_ind_lt);
      segments_offset_ += tile_len_;
      segm_ind_lt = LoadNextSegmentTile();

      // "Local-coordinates" segment index must be zeroed and keep
      // track of the output vector location.
      index = 0;
      return static_cast<uint32_t>(segm_ind_lt.GetValue(0) - vec_start_offset_);
    } else {
      // Read next segment
      return static_cast<uint32_t>(segm_ind_lt.GetValue(index) -
                                   vec_start_offset_);
    }
  }

  /**
   * @brief Run the kernel with a scalar walk over the segments.
   *
   * Fallback for tile lengths that are not a power of two.
   */
  __aicore__ inline void PipelineProcessWithCube() {
    LocalTensor<uint32_t> segm_ind_lt = LoadNextSegmentTile();
    LocalTensor<T> vec_out_lt = out_q_.template AllocTensor<T>();

    T accumulation = 0;
    uint32_t global_in_offset = 0;
    uint32_t out_idx = 0;
    uint32_t segm_idx = 0;
    uint32_t segm_end = static_cast<uint32_t>(segm_ind_lt.GetValue(segm_idx) -
                                              vec_start_offset_);

    for (uint32_t matrix_tile_idx = 0; matrix_tile_idx < num_tiles_;
         matrix_tile_idx++) {
      if constexpr (SyncBefore) {
        sync::SyncGroup<sync::GroupSyncDirection::FULL>();
      }

      const uint32_t num_elems_to_copy =
          scalar::NextTileLen(matrix_tile_len_, global_in_offset, vec_len_);
      copy::CopyGmToVec(in_q_, global_in_[global_in_offset], num_elems_to_copy);
      LocalTensor<T> vec_in_lt = in_q_.template DeQue<T>();

      uint32_t matrix_tile_offset = 0;
      uint32_t num_elems_to_process =
          scalar::NextTileLen(tile_len_, matrix_tile_offset, num_elems_to_copy);

      while (num_elems_to_process > 0) {
        while (segm_end < global_in_offset + num_elems_to_process) {
          // Index of the segment end relative to the matrix tile
          const uint32_t matrix_tile_segm_end =
              matrix_tile_offset + segm_end - global_in_offset;

          // Last segment value. Zero if lies on tile boundary.
          const bool is_on_end_of_tile = segm_end == global_in_offset;
          const T delta = is_on_end_of_tile
                              ? 0
                              : vec_in_lt.GetValue(matrix_tile_segm_end - 1);
          SafeOutWrite(vec_out_lt, out_idx, alpha_ * (accumulation + delta));

          // Keep track of the value of the last segment's end to
          // subtract it from the next segment (recall scan
          // speculation within tile)
          accumulation = -delta;

          // Mutates both segm_ind_lt and segm_idx
          segm_end = NextSegmEndIndex(segm_ind_lt, segm_idx);
        }

        global_in_offset += num_elems_to_process;
        matrix_tile_offset += num_elems_to_process;

        accumulation += vec_in_lt.GetValue(matrix_tile_offset - 1);

        // Number of element to process in the next iteration
        num_elems_to_process = scalar::NextTileLen(
            tile_len_, matrix_tile_offset, num_elems_to_copy);
      }

      in_q_.template FreeTensor<T>(vec_in_lt);
    }

    segm_q_.template FreeTensor<uint32_t>(segm_ind_lt);

    // The last segment contains the remaining accumulation values.
    vec_out_lt.SetValue(out_idx, alpha_ * accumulation);

    const uint32_t tail_len = out_idx + 1;
    out_q_.template EnQue<T>(vec_out_lt);
    if constexpr (UseAtomicWrite) {
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::SetAtomicAdd<T>();
    }
    copy::CopyVecToGm(global_out_[out_offset_], out_q_, tail_len);
    if constexpr (UseAtomicWrite) {
      AscendC::SetAtomicNone();
    }
  }

  TPipe pipe_;

  TQue<QuePosition::VECIN, BUFFER_NUM> in_q_;
  TQue<QuePosition::VECIN, BUFFER_NUM> segm_q_;
  TQue<QuePosition::VECOUT, 1> out_q_;

  TBuf<QuePosition::VECCALC> sidx_buf_;
  TBuf<QuePosition::VECCALC> segb_buf_;
  TBuf<QuePosition::VECCALC> pos_buf_;
  TBuf<QuePosition::VECCALC> posc_buf_;
  TBuf<QuePosition::VECCALC> sel_buf_;
  TBuf<QuePosition::VECCALC> gval_buf_;
  TBuf<QuePosition::VECCALC> pval_buf_;
  TBuf<QuePosition::VECCALC> u_buf_;
  TBuf<QuePosition::VECCALC> fwd_buf_;
  TBuf<QuePosition::VECCALC> fidx_buf_;
  TBuf<QuePosition::VECCALC> ptab_buf_;

  GlobalTensor<T> global_in_;
  GlobalTensor<uint32_t> global_segm_in_;
  GlobalTensor<T> global_out_;

  const uint32_t vec_len_;
  const uint32_t num_segments_;
  const uint32_t tile_len_;
  const uint32_t matrix_tile_len_;
  const uint32_t num_tiles_;
  const uint32_t vec_start_offset_;
  const T alpha_;
  const uint32_t log2_tile_len_;
  const bool is_vectorizable_;
  const uint32_t group_len_;
  const uint32_t seg_buf_len_;

  // Vectorized path state.
  LocalTensor<int32_t> fidx_lt_;
  // Output tile holding the sums produced by scalar batches, stored once full.
  LocalTensor<T> pend_lt_;
  uint32_t pend_count_ = 0;
  bool pend_valid_ = false;
  LocalTensor<uint32_t> segm_lt_;
  uint32_t seg_tile_base_ = 0;
  uint32_t seg_tile_count_ = 0;
  bool seg_tile_valid_ = false;

  // Scalar fallback state.
  uint32_t segments_offset_ = 0;

  uint32_t out_offset_ = 0;
};

}  // namespace tcuscan
