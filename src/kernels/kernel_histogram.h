/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * @file kernel_histogram.h
 * @brief Kernel implementing a Vector histogram kernel operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Returns the counts of `torch.histogram(x, bins=num_bins)`.
 *
 * The input is read as a flat buffer, so the shape of the tensor it came from
 * is irrelevant: `torch.histogram` histograms every element regardless of
 * layout, and the host hands over a contiguous, flattened view.
 *
 * Bin `j` covers `[edge_j, edge_{j+1})` and the last bin is closed on both
 * ends. Elements outside `[hist_min, hist_max]` are not counted at all, so
 * the counts sum to the input length only when the range spans the input.
 *
 * The kernel also writes out the edges it binned against. They are the second
 * half of what `torch.histogram` returns, and the host cannot always derive
 * them: on the default path the range is a device-side reduction it never
 * reads back. Emitting them from here also makes them exact rather than a
 * reconstruction, and costs nothing - the edges are already computed.
 *
 * @tparam T Input data type. Supported types are `float16`.
 */
template <typename T>
class KernelHistogram {
  constexpr static uint32_t BUFFER_NUM = 2;
  /// @brief Number of range values read from global memory: `{min, max}`.
  constexpr static uint32_t NUM_BOUNDS = 2;

  using PackedMaskT = uint8_t;
  using GatherMaskT =
      typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Dimension of the input vectors.
   * @param [in] tile_len Tile length.
   * @param [in] num_bins Number of histogram bins.
   * @param [in] hist_min Lower end of the histogram range; elements below it
   * are not counted. Ignored if `bounds_in_gm`.
   * @param [in] hist_max Upper end of the histogram range; elements above it
   * are not counted. Ignored if `bounds_in_gm`.
   * @param [in] bounds_in_gm Whether `hist_min` and `hist_max` must be read
   * from global memory instead; see `LoadBounds`. The range is then the
   * input's own min/max, so it spans every element.
   */
  __aicore__ inline KernelHistogram(uint32_t vec_len, uint32_t tile_len,
                                    uint32_t num_bins, float hist_min,
                                    float hist_max, bool bounds_in_gm)
      : vec_core_num_(GetBlockNum() * GetTaskRation()),
        vec_len_(vec_len),
        tile_len_(tile_len),
        num_bins_(num_bins),
        bounds_in_gm_(bounds_in_gm),
        hist_min_(hist_min),
        hist_max_(hist_max),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_len_)),
        max_num_tiles_per_block_(scalar::CeilDiv(num_tiles_, vec_core_num_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_in Pointer to the input vector in global memory.
   * @param [in] bounds Pointer to the `{min, max}` pair in global memory. Only
   * read when the kernel was constructed with `bounds_in_gm`; may be null
   * otherwise.
   * @param [in] vec_out Pointer to the output vector in global memory.
   * @param [in] bin_edges Pointer to the `num_bins + 1` bin edges in global
   * memory, written by the first block.
   */
  __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR bounds, GM_ADDR vec_out,
                              GM_ADDR bin_edges) {
    global_in_.SetGlobalBuffer((__gm__ T*)vec_in, vec_len_);
    global_out_.SetGlobalBuffer((__gm__ int32_t*)vec_out, num_bins_);
    global_edges_.SetGlobalBuffer((__gm__ float*)bin_edges, num_bins_ + 1);

    pipe_.InitBuffer(in_q_, BUFFER_NUM, tile_len_ * sizeof(T));
    pipe_.InitBuffer(out_q_, 1, num_bins_ * sizeof(int32_t));
    pipe_.InitBuffer(
        edges_q_, 1,
        scalar::AlignUp((num_bins_ + 1) * sizeof(float), UB_ALIGNMENT));

    pipe_.InitBuffer(out_buf_, num_bins_ * sizeof(int32_t));
    pipe_.InitBuffer(work_buf_, tile_len_ * sizeof(T));
    pipe_.InitBuffer(packed_mask_buf_, tile_len_ * sizeof(PackedMaskT) / 8);

    if (bounds_in_gm_) {
      global_bounds_.SetGlobalBuffer((__gm__ T*)bounds, NUM_BOUNDS);
    }
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    if (bounds_in_gm_) {
      LoadBounds();
    }

    // Every block holds the same range and would compute the same edges, so
    // one of them writes them out and the rest stay off that buffer.
    if (GetBlockIdx() == 0) {
      WriteBinEdges();
    }

    uint32_t global_offset =
        GetBlockIdx() * tile_len_ * max_num_tiles_per_block_;
    const uint32_t num_tiles_to_process =
        scalar::GetWorkDistribution(vec_len_, tile_len_, vec_core_num_);

    const LocalTensor<int32_t> vec_out_lt = out_q_.AllocTensor<int32_t>();
    AscendC::Duplicate<int32_t>(vec_out_lt, 0, num_bins_);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_to_process; tile_idx++) {
      const bool full_tile = global_offset + tile_len_ <= vec_len_;
      const uint32_t num_elems_to_process =
          full_tile ? tile_len_ : vec_len_ - global_offset;

      copy::CopyGmToVec(in_q_, global_in_[global_offset], num_elems_to_process);
      ProcessTile(vec_out_lt, num_elems_to_process);
      global_offset += tile_len_;
    }
    out_q_.EnQue<int32_t>(vec_out_lt);
    AtomicAddWritePartialHist();
  }

 private:
  /**
   * @brief Read the histogram range from global memory.
   *
   * The range spans the input, so it is the result of a reduction over `x`.
   * Having the host read that reduction back costs a blocking device-to-host
   * synchronization, which is far more expensive than this kernel; instead the
   * caller leaves the two values in global memory and every core loads them
   * here. The load is two elements wide and lands in L2 after the first core
   * touches it.
   */
  __aicore__ inline void LoadBounds() {
    // Read the two values with scalar loads straight from global memory,
    // exactly as the tiling struct is read. Staging them through UB would mean
    // a DMA whose completion the scalar unit cannot observe by dequeuing
    // alone: `DeQue` on a VECIN queue waits on MTE2_V, which gates the vector
    // pipe, while `GetValue` is a scalar read free to run ahead of the copy.
    hist_min_ = static_cast<float>(global_bounds_.GetValue(0));
    hist_max_ = static_cast<float>(global_bounds_.GetValue(1));

    // A constant input spans an empty range, which would make the step zero
    // and every bin edge equal. `torch.histogram` widens such a range to
    // `[min - 0.5, max + 0.5]`, which lands every element in the middle bin.
    // Only this kernel can apply it on the default path: the host never reads
    // the reduction back, so it cannot tell that the range degenerated.
    if (hist_min_ == hist_max_) {
      hist_min_ -= 0.5F;
      hist_max_ += 0.5F;
    }
  }

  __aicore__ inline void ProcessTile(const LocalTensor<int32_t>& vec_out_lt,
                                     uint32_t size) {
    LocalTensor<T> vec_in_lt = in_q_.DeQue<T>();
    LocalTensor<int32_t> partial_out_lt = out_buf_.Get<int32_t>();

    const float step = (hist_max_ - hist_min_) / num_bins_;

    // V->S: the previous tile's `Add` reads `partial_out_lt`, which the scalar
    // stores below overwrite.
    sync::ScalarWaitForVec();

    // Elements outside the range are dropped rather than folded into the end
    // bins, as `torch.histogram` does. On the default path the range is the
    // input's own min/max (see `LoadBounds`), so nothing can fall outside it
    // and neither out-of-range count has to be computed.
    int32_t current_size = bounds_in_gm_
                               ? static_cast<int32_t>(size)
                               : CountAtMost(vec_in_lt, hist_max_, size);

    // The loop below writes indices `num_bins_ - 1 .. 1` and the statement
    // after it writes index 0, so every entry of `partial_out_lt` is assigned
    // and no zero-fill is needed.
    for (uint32_t i = 1; i < num_bins_; i++) {
      const int32_t less_than_elems =
          CountBelow(vec_in_lt, hist_max_ - i * step, size);
      partial_out_lt.SetValue(num_bins_ - i, current_size - less_than_elems);
      current_size = less_than_elems;
    }

    // The lower edge of bin 0 is `hist_min_` itself rather than
    // `hist_max_ - num_bins_ * step`, which the rounding of `step` would place
    // a hair to either side of it; `WriteBinEdges` reports the same edge.
    //
    // Those two are the only edges the loop does not derive from one another,
    // so they are also the only pair that can come out of order: a range much
    // narrower than one float32 ulp of its own magnitude leaves `step` below
    // the rounding granularity, and `hist_max_ - (num_bins_ - 1) * step` can
    // then land just under `hist_min_`. The count of what lies between them is
    // meaningless at that point, but it must not be negative.
    const int32_t below_range =
        bounds_in_gm_ ? 0 : CountBelow(vec_in_lt, hist_min_, size);
    const int32_t first_bin = current_size - below_range;
    partial_out_lt.SetValue(0, first_bin > 0 ? first_bin : 0);

    // S->V: the scalar stores above feed the vector `Add` below.
    sync::VecWaitForScalar();

    Add(vec_out_lt, vec_out_lt, partial_out_lt, num_bins_);

    in_q_.FreeTensor<T>(vec_in_lt);
  }

  /**
   * @brief Count the tile elements lying strictly below `edge`.
   *
   * @param [in] vec_lt Input vector tile.
   * @param [in] edge Bin edge, in the float32 precision the range is kept in.
   * @param [in] size Input tile size to consider; first `size` elements.
   */
  __aicore__ inline int32_t CountBelow(LocalTensor<T>& vec_lt, float edge,
                                       uint32_t size) {
    const T pivot = static_cast<T>(edge);
    // `pivot` is `edge` rounded to nearest, so it is not always a faithful
    // pivot. When it rounds *down*, an input exactly equal to it still belongs
    // below the true edge, but `< pivot` excludes it and pushes it one bin up;
    // comparing with `<=` restores it. When it rounds up (or is exact),
    // `< pivot` already matches the true edge, because no value of type `T`
    // lies between the two. Either way the comparison below agrees with an
    // exact one for every representable input.
    const CMPMODE cmp_mode =
        static_cast<float>(pivot) < edge ? CMPMODE::LE : CMPMODE::LT;

    return CountCompare(vec_lt, pivot, cmp_mode, size);
  }

  /**
   * @brief Count the tile elements that are at most `edge`.
   *
   * Mirror of `CountBelow` for the closed upper end of the range: an element
   * equal to the largest edge belongs to the last bin rather than being
   * dropped as out of range.
   *
   * @param [in] vec_lt Input vector tile.
   * @param [in] edge Bin edge, in the float32 precision the range is kept in.
   * @param [in] size Input tile size to consider; first `size` elements.
   */
  __aicore__ inline int32_t CountAtMost(LocalTensor<T>& vec_lt, float edge,
                                        uint32_t size) {
    const T pivot = static_cast<T>(edge);
    // When `pivot` rounds *up* past the true edge, an input equal to it lies
    // above the edge and must not be counted, so the comparison tightens to
    // `<`. When it rounds down (or is exact), `<= pivot` already matches the
    // exact comparison, because no value of type `T` lies between the two.
    const CMPMODE cmp_mode =
        static_cast<float>(pivot) > edge ? CMPMODE::LT : CMPMODE::LE;

    return CountCompare(vec_lt, pivot, cmp_mode, size);
  }

  /**
   * @brief Count the cardinality of the set {x_i `cmp_mode` pivot}, where
   * \f$ x_i \f$ are the first `size` elements of the tile.
   *
   * @param [in] vec_lt Input vector tile.
   * @param [in] pivot Pivot to compare the input elements against.
   * @param [in] cmp_mode Comparison against `pivot`; `CMPMODE::LT` or
   * `CMPMODE::LE`, chosen by the caller so that the comparison in type `T`
   * matches the exact one against the unrounded bin edge.
   * @param [in] size Input tile size to consider; first `size` elements.
   */
  __aicore__ inline int32_t CountCompare(LocalTensor<T>& vec_lt, T pivot,
                                         CMPMODE cmp_mode, uint32_t size) {
    LocalTensor<T> work_lt = work_buf_.Get<T>();

    const LocalTensor<uint8_t> packed_mask_8b = packed_mask_buf_.Get<uint8_t>();
    AscendC::Duplicate(packed_mask_8b.template ReinterpretCast<int32_t>(), 0,
                       packed_mask_8b.GetSize() / 4);
    // V->V on `packed_mask_8b`: the vector pipe alone needs ordering here.
    AscendC::PipeBarrier<PIPE_V>();
    CompareScalar(packed_mask_8b, vec_lt, pivot, cmp_mode, tile_len_);
    // V->V: `CompareScalar` writes the mask that `GatherMask` reads below.
    AscendC::PipeBarrier<PIPE_V>();

    const LocalTensor<GatherMaskT> mask_lt =
        packed_mask_buf_.Get<GatherMaskT>();

    uint64_t num_gathered_elems = 0;
    GatherMask(work_lt, vec_lt, mask_lt, true, size, {1, 1, 8, 8},
               num_gathered_elems);
    // V->S: `GatherMask` reports the selected count from the vector pipe; the
    // scalar unit reads it on the next line.
    sync::ScalarWaitForVec();

    return static_cast<int32_t>(num_gathered_elems);
  }

 private:
  /**
   * @brief Write out the `num_bins_ + 1` edges that bound the bins.
   *
   * The expressions are the ones `ProcessTile` compares against, so the edges
   * reported are exactly the ones the counts were decided by, down to the
   * last bit of the float32 rounding.
   */
  __aicore__ inline void WriteBinEdges() {
    const LocalTensor<float> edges_lt = edges_q_.AllocTensor<float>();
    const float step = (hist_max_ - hist_min_) / num_bins_;

    edges_lt.SetValue(0, hist_min_);
    for (uint32_t i = 1; i <= num_bins_; i++) {
      edges_lt.SetValue(i, hist_max_ - (num_bins_ - i) * step);
    }

    // The stores above are scalar and the copy below runs on MTE3, which
    // `EnQue` alone does not order against them. This runs once per launch,
    // so the blanket barrier costs nothing worth saving.
    AscendC::PipeBarrier<PIPE_ALL>();

    edges_q_.EnQue<float>(edges_lt);
    copy::CopyVecToGm(global_edges_, edges_q_, num_bins_ + 1);
  }

  __aicore__ inline void AtomicAddWritePartialHist() {
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SetAtomicAdd<int32_t>();
    copy::CopyVecToGm(global_out_, out_q_, num_bins_);
    // `SetAtomicNone` is a scalar config write that takes effect as soon as it
    // retires, while the store above is still queued on MTE3. Without this
    // barrier the mode can be cleared before the store executes, turning this
    // core's accumulate into an overwrite and dropping the other cores' counts.
    AscendC::PipeBarrier<PIPE_MTE3>();
    AscendC::SetAtomicNone();
  }

  TPipe pipe_;

  TQue<QuePosition::VECIN, BUFFER_NUM> in_q_;
  TQue<QuePosition::VECOUT, 1> out_q_;
  TQue<QuePosition::VECOUT, 1> edges_q_;

  TBuf<QuePosition::VECCALC> work_buf_;
  TBuf<QuePosition::VECCALC> out_buf_;
  TBuf<QuePosition::VECCALC> packed_mask_buf_;

  GlobalTensor<T> global_in_;
  GlobalTensor<T> global_bounds_;
  GlobalTensor<int32_t> global_out_;
  GlobalTensor<float> global_edges_;

  const uint32_t vec_core_num_;
  const uint32_t vec_len_;
  const uint32_t tile_len_;
  const uint32_t num_bins_;
  const bool bounds_in_gm_;
  float hist_min_;
  float hist_max_;
  const uint32_t num_tiles_;
  const uint32_t max_num_tiles_per_block_;
};

/**
 * @brief Run the `histogram` kernel.
 *
 * @tparam T Input data type.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] bounds Pointer to the `{min, max}` histogram range in global
 * memory. Only read if `bounds_in_gm`; may be null otherwise.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] bin_edges Pointer to the `num_bins + 1` bin edges.
 * @param [in] vec_len Dimension of the input vector.
 * @param [in] tile_len Tile length.
 * @param [in] num_bins Number of histogram bins.
 * @param [in] hist_min Lower end of the histogram range; elements below it are
 * not counted. Ignored if `bounds_in_gm`.
 * @param [in] hist_max Upper end of the histogram range; elements above it are
 * not counted. Ignored if `bounds_in_gm`.
 * @param [in] bounds_in_gm Whether to take the range from `bounds` rather than
 * from `hist_min` / `hist_max`.
 */
template <typename T>
__aicore__ inline void run_histogram(GM_ADDR vec_in, GM_ADDR bounds,
                                     GM_ADDR vec_out, GM_ADDR bin_edges,
                                     uint32_t vec_len, uint32_t tile_len,
                                     uint32_t num_bins, float hist_min,
                                     float hist_max, bool bounds_in_gm) {
  if ASCEND_IS_AIV {
    KernelHistogram<T> op(vec_len, tile_len, num_bins, hist_min, hist_max,
                          bounds_in_gm);
    op.Init(vec_in, bounds, vec_out, bin_edges);
    op.Process();
  }
}

}  // namespace tcuscan
