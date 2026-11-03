/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file kernel_cube_reduce.h
 * @brief Kernel implementing a block reduction kernel that uses Cube and Vector
 * units.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "kernels/kernel_pad.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @ingroup reduce
 * @brief Performs block reductions using matrix multiplications. Each Cube core
 * reduces the input vector independently by multiplying its input block (viewed
 * as a matrix with `S := matmul_size` columns) with an all-ones \f$S\times 16
 * \f$ matrix. Each Cube core outputs an \f$ S \times 16 \f$ so that the sum of
 * any column equals to the input reduction.
 *
 * The kernels support a AI core group synchronization after writing to the
 * output.
 *
 * Limitation: the algorithm outputs 16 identical column due to FixPipe
 * limitations.
 *
 * @tparam InputT Data type of the input vector.
 * @tparam SyncAfter If true, synchronize vector units with cube unit after
 * writing reductions into global memory.
 */
template <typename InputT, bool SyncAfter = false>
class KernelCubeReduce {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<InputT>;

 public:
  /// @brief Redundant cube unit reduction dimension.
  constexpr static uint32_t MAT_DIM_16 = 16;

  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Number of elements in an input vector.
   * @param [in] matmul_size Matrix multiplication size.
   */
  __aicore__ inline KernelCubeReduce(uint32_t vec_len, uint32_t matmul_size)
      : block_num_(GetBlockNum()),
        vec_len_(vec_len),
        matmul_size_(matmul_size),
        tile_len_(matmul_size_ * matmul_size_),
        all_ones_len_(matmul_size_ * MAT_DIM_16),
        num_mat_iters_(scalar::FloorDiv(vec_len, tile_len_)),
        max_num_mat_iters_per_block_(
            scalar::CeilDiv(num_mat_iters_, block_num_)) {
    static_assert(tcuscan::cube_unit::IsCubeSupported<InputT>,
                  "Unsupported input Cube dtype. Please use half or int8_t.");
    ASCENDC_ASSERT((vec_len % tile_len_ == 0), {
      KERNEL_LOG(KERNEL_ERROR,
                 "The length of the input vector (%d) must be "
                 "divisible by the tile size (%d)",
                 vec_len, tile_len_);
    });
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_in Pointer to input vector in global memory.
   * @param [in] vec_out Pointer to output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR vec_out) {
    global_a_.SetGlobalBuffer((__gm__ InputT*)vec_in, vec_len_);
    global_c_.SetGlobalBuffer((__gm__ OutputT*)vec_out,
                              block_num_ * all_ones_len_);

    pipe_.InitBuffer(a1_q_, 1, tile_len_ * sizeof(InputT));
    pipe_.InitBuffer(a2_q_, 1, tile_len_ * sizeof(InputT));
    pipe_.InitBuffer(b1_q_, 1, all_ones_len_ * sizeof(InputT));
    pipe_.InitBuffer(b2_q_, 1, all_ones_len_ * sizeof(InputT));

    pipe_.InitBuffer(co1_q_, 1, tile_len_ * sizeof(OutputT));
  }

  /**
   * @brief Run the kernel - process all tiles.
   */
  __aicore__ inline void Process() {
    const uint32_t id = GetBlockIdx();
    const uint32_t ai_core_offset =
        id * tile_len_ * max_num_mat_iters_per_block_;

    const uint32_t num_tiles_to_process =
        tcuscan::scalar::GetWorkDistribution(vec_len_, tile_len_, block_num_);

    if (num_tiles_to_process == 0) {
      if constexpr (SyncAfter) {
        sync::SyncGroup<sync::GroupSyncDirection::FULL>();
      }
      return;
    }

    // Load all-ones matrix B only once from global memory into L0B.
    LoadAllOnesToL0();
    // First iteration: perform no L0C accumulation.
    CubeIter<false /* accumulate */>(ai_core_offset);
    for (uint32_t idx = 1; idx < num_tiles_to_process; idx++) {
      const uint32_t offset = ai_core_offset + idx * tile_len_;
      CubeIter<true /* accumulate */>(offset);
    }

    const uint32_t output_tile_len = matmul_size_ * MAT_DIM_16;
    copy::CopyCL0ToGlobal(global_c_[id * output_tile_len], co1_q_, matmul_size_,
                          MAT_DIM_16);
    // Ensure Fixpipe has committed the L0C result before the AIVs are released
    // by the cross-core sync below.
    PipeBarrier<PIPE_ALL>();

    if constexpr (SyncAfter) {
      sync::SyncGroup<sync::GroupSyncDirection::FULL>();
    }
    queue::FreeFromQ<InputT>(b2_q_);
  }

 private:
  /**
   * @brief Loads an input matrix tile and performs a matrix multiplication
   * with preloaded B=`all-ones matrix`. Input tile has \f$ S \times 16\f$
   * size.
   *
   * Note: Matrix B in L0B is not "freed" after operation.
   *
   * @tparam AccumulateL0C If true, accumulate on L0C buffer.
   *
   * @param [in] global_offset Input global memory offset to load tile from.
   */
  template <bool AccumulateL0C = false>
  __aicore__ inline void CubeIter(uint32_t global_offset) {
    LoadAToL0(global_offset);
    cube_unit::Multiply<InputT, AccumulateL0C /* accumulate_c */,
                        true /* free_a*/, false /* free_b */>(
        a2_q_, b2_q_, co1_q_, matmul_size_, MAT_DIM_16, matmul_size_);
    AscendC::PipeBarrier<PIPE_ALL>();
  }

  /**
   * @brief Load a tile of input matrix A into L0A. Input tile has size \f$ S
   * \times S\f$.
   */
  __aicore__ inline void LoadAToL0(uint32_t global_offset) {
    copy::CopyGmToL1A(a1_q_, global_a_[global_offset], m_blocks_, k_blocks_);
    copy::CopyL1ToL0A<InputT, true>(a2_q_, a1_q_, m_blocks_, k_blocks_);
  }

  /**
   * @brief Load all-ones matrix B into L0B.
   */
  __aicore__ inline void LoadAllOnesToL0() {
    LocalTensor<InputT> all_ones_lt = b1_q_.AllocTensor<InputT>();
    cube_unit::InitConstAllOnesL1<InputT, TPosition::B1>(all_ones_lt,
                                                         all_ones_len_);
    b1_q_.EnQue<InputT>(all_ones_lt);
    copy::CopyL1ToL0B<InputT, true>(b2_q_, b1_q_, k_blocks_, n_blocks_);
  }

  TPipe pipe_;

  TQue<QuePosition::A1, 1> a1_q_;
  TQue<QuePosition::A2, 1> a2_q_;
  TQue<QuePosition::B1, 1> b1_q_;
  TQue<QuePosition::B2, 1> b2_q_;

  TQue<QuePosition::CO1, 1> co1_q_;

  GlobalTensor<InputT> global_a_;
  GlobalTensor<OutputT> global_c_;

  const uint32_t block_num_;
  const uint32_t vec_len_;
  const uint32_t matmul_size_;
  const uint32_t tile_len_;
  const uint32_t all_ones_len_;
  const uint32_t num_mat_iters_;
  const uint32_t max_num_mat_iters_per_block_;

  // Fractal sizes: 16x32 for int8 and 16x16 for fp16
  constexpr static uint32_t M_CUBE_BLOCK_SIZE = tcuscan::GetFractalMN<InputT>();
  constexpr static uint32_t K_CUBE_BLOCK_SIZE = tcuscan::GetFractalK<InputT>();

  const uint32_t m_blocks_ = matmul_size_ / M_CUBE_BLOCK_SIZE;
  const uint32_t n_blocks_ = MAT_DIM_16 / M_CUBE_BLOCK_SIZE;
  const uint32_t k_blocks_ = matmul_size_ / K_CUBE_BLOCK_SIZE;
};

/**
 * @ingroup reduce
 * @brief Given the multi-core output of `KernelCubeReduce` (which is an \f$ S
 * \times 16 \f$ matrix per block, `S := matmul_size`), reduces the first matrix
 * column into a single scalar sum-reduction per AI core.
 *
 * The kernel expects that the cube cores use a single `sync::SyncGroup()` after
 * completion. No synchronization between AI cores is required.
 *
 * Example of 2 AI Cores with matrix size s=128 in 910B. Input length is
 * 2*128=258 elements and output length is (# of AI
 * cores) * GetTaskRation() = 2 * 2 = 4. Output values are the block reduction
 * of the input with block length s / GetTaskRation() = 128 / 2 = 64.
 *
 * @tparam T Data type of the input vector.
 * @tparam SyncBefore If true, synchronize with cube unit before processing.
 */
template <typename T, bool SyncBefore>
class KernelCompleteCubeReduce {
  constexpr static uint32_t MAT_DIM_16 = KernelCubeReduce<T>::MAT_DIM_16;
  constexpr static int32_t MIN_VEC_SIZE = UB_ALIGNMENT / sizeof(T);

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] matmul_size Matrix size used in `KernelCubeReduce`.
   */
  __aicore__ inline KernelCompleteCubeReduce(uint32_t matmul_size)
      : block_num_(GetBlockNum()),
        matmul_size_(matmul_size),
        tile_len_(matmul_size_ * MAT_DIM_16) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_in Pointer to input buffer in global memory.
   * @param [in] partials Pointer to scratch for one partial sum per block.
   * @param [in] vec_out Pointer to the single-element output.
   */
  __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR partials,
                              GM_ADDR vec_out) {
    global_input_.SetGlobalBuffer((__gm__ T*)vec_in, GetBlockNum() * tile_len_);
    global_partials_.SetGlobalBuffer((__gm__ T*)partials, block_num_);
    global_output_.SetGlobalBuffer((__gm__ T*)vec_out, 1);

    pipe_.InitBuffer(vecin_q_, 1, tile_len_ * sizeof(T));
    pipe_.InitBuffer(res_out_q_, 1, MIN_VEC_SIZE * sizeof(T));
    pipe_.InitBuffer(partials_q_, 1,
                     scalar::AlignUp(block_num_ * sizeof(T), UB_ALIGNMENT));
  }

  /**
   * @brief The first AIV core of each AI core zeroes its block's partial sum,
   * which its two AIV cores then atomically add into.
   */
  __aicore__ inline void WriteOutZero() {
    const uint32_t aiv_id = static_cast<uint32_t>(GetSubBlockIdx());
    // Only the first AIV core of each AI core, writes zero to the output.
    if (aiv_id == 0) {
      const uint32_t block_idx =
          scalar::FloorDiv(GetBlockIdx(), GetTaskRation());

      // Write zero since later atomic-add is used.
      copy::CopyScalarToGm(global_partials_[block_idx], res_out_q_,
                           static_cast<T>(0));
    }
  }

  /**
   * @brief Run the kernel.
   *
   */
  __aicore__ inline void Process() {
    WriteOutZero();
    // Only AIV 0 of each AI core writes that zero, while both AIVs atomically
    // add their half-column in `ProcessTile`. `SyncGroup` below pairs each AIV
    // with the *cube* core, not the two AIVs with each other, so nothing else
    // orders the store against the partner's add: the partner writes no zero,
    // so it can reach its add while AIV 0's store is still queued on MTE3, and
    // the zero then erases it. That drops one AIV's half of the block's sum --
    // the failure the `int8` sizes of `test_cube_reduce` were hitting.
    SyncAll<true /*isAIVOnly*/>();

    // A synchronization barrier is expected by the Cube unit.
    if constexpr (SyncBefore) {
      sync::SyncGroup<sync::GroupSyncDirection::FULL>();
    }
    ProcessTile();

    // Every block has now left its own partial sum behind; one core adds them
    // up into the single output element.
    SyncAll<true /*isAIVOnly*/>();
    CombinePartials();
  }

  /**
   * @brief Sum the per-block partials into the one-element output.
   *
   * A single core does this rather than every block atomically adding into the
   * output directly: a four-byte atomic add issued by many *different* cores
   * to one address loses updates on this part, where the two AIVs of a single
   * core accumulating into their own block's slot is reliable. There are only
   * `block_num_` values to add, so the serial pass costs nothing.
   */
  __aicore__ inline void CombinePartials() {
    if (GetBlockIdx() != 0 || GetSubBlockIdx() != 0) {
      return;
    }

    copy::CopyGmToVec(partials_q_, global_partials_, block_num_);
    LocalTensor<T> partials_lt = partials_q_.template DeQue<T>();
    // `DeQue` orders the vector unit against the copy, not the scalar unit
    // that reads the values below.
    sync::ScalarWaitForMte2();

    T total = 0;
    for (uint32_t i = 0; i < block_num_; i++) {
      total += partials_lt.GetValue(i);
    }
    partials_q_.FreeTensor(partials_lt);

    copy::CopyScalarToGm(global_output_[0], res_out_q_, total);
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void ProcessTile() {
    const uint32_t id = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t aiv_id = static_cast<uint32_t>(GetSubBlockIdx());
    const uint32_t ratio = static_cast<uint32_t>(GetTaskRation());

    // Which block's matrix tile this core reduces; the result of every one
    // of them lands in the same output element.
    const uint32_t input_idx = scalar::FloorDiv(id, ratio);
    const uint32_t gm_offset = input_idx * tile_len_;

    // ReduceSum input matrix tile into a scalar.
    copy::CopyGmToVec(vecin_q_, global_input_[gm_offset], tile_len_);
    LocalTensor<T> input_lt = vecin_q_.DeQue<T>();
    // `DeQue` only orders the vector unit against the copy above. The column
    // is summed below with scalar `GetValue` reads, so the scalar unit has to
    // be ordered against MTE2 explicitly or it may read stale UB contents.
    sync::ScalarWaitForMte2();

    const uint32_t aiv_core_offset = aiv_id * (tile_len_ / ratio);
    // AIV 0 sums the first half of the first column of `input_lt`
    // (viewed as `matmul_size_` x `16` matrix). AIV1 sums the second half.
    T sum = 0;
    for (uint32_t i = 0; i < matmul_size_ / ratio; i++) {
      const uint32_t index = aiv_core_offset + i * MAT_DIM_16;
      sum += input_lt.GetValue(index);
    }

    PipeBarrier<PIPE_ALL>();
    AscendC::SetAtomicAdd<T>();
    copy::CopyScalarToGm(global_partials_[input_idx], res_out_q_, sum);
    // `SetAtomicNone` is a scalar config write that takes effect as soon as it
    // retires, while the store above is still queued on MTE3. Without this
    // barrier the mode can be cleared before the store executes, turning this
    // AIV's accumulate into an overwrite and dropping the partner AIV's half
    // of the column sum.
    AscendC::PipeBarrier<PIPE_MTE3>();
    AscendC::SetAtomicNone();
    vecin_q_.FreeTensor(input_lt);
  }

 private:
  TPipe pipe_;

  TQue<QuePosition::VECIN, 2> vecin_q_;
  TQue<QuePosition::VECOUT, 1> res_out_q_;

  TQue<QuePosition::VECIN, 1> partials_q_;

  GlobalTensor<T> global_input_;
  GlobalTensor<T> global_partials_;
  GlobalTensor<T> global_output_;

  const uint32_t block_num_;
  const uint32_t matmul_size_;
  const uint32_t tile_len_;
};

/**
 * @ingroup reduce
 * @brief Run the `cube_reduce` kernel.
 *
 * @tparam T Input data type.
 *
 * @param [in] vec_in Pointer to an input vector.
 * @param [in] vec_out Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] vec_len Number of elements to process.
 * @param [in] matmul_size Size of the matmul tile used in the kernel.
 */
template <typename T>
__aicore__ inline void run_cube_reduce(GM_ADDR vec_in, GM_ADDR vec_out,
                                       GM_ADDR workspace, uint32_t vec_len,
                                       uint16_t matmul_size) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<T>;

  const uint32_t align_size = matmul_size * matmul_size;
  const uint32_t padded_vec_len = scalar::AlignUp(vec_len, align_size);

  GM_ADDR padded_cube_reductions = workspace;

  if (vec_len % (matmul_size * matmul_size) || vec_len % UB_ALIGNMENT) {
    GM_ADDR const padded_input = workspace;

    run_pad_kernel<T, false>(vec_in, padded_input, vec_len, align_size);

    // Every vector core pads its own tiles and every cube core then reads all
    // of them, so this barrier has to span the device. `SyncGroup` alone pairs
    // a cube core with its own two AIVs alone, which leaves it free to read
    // tiles another group has not written yet -- it would reduce whatever the
    // workspace happened to hold.
    AscendC::SyncAll<false /*isAIVOnly*/>();
    sync::SyncGroup<sync::GroupSyncDirection::FULL>();
    PipeBarrier<PIPE_ALL>();

    // Cube kernel must read the padded input from workspace
    vec_len = padded_vec_len;
    vec_in = padded_input;
    padded_cube_reductions = workspace + padded_vec_len * sizeof(T);
  }

  if ASCEND_IS_AIC {
    KernelCubeReduce<T, true /* Sync */> op_vec(vec_len, matmul_size);
    op_vec.Init(vec_in, padded_cube_reductions);
    op_vec.Process();
  }

  if ASCEND_IS_AIV {
    // One partial sum per block, parked after the cube stage's output.
    GM_ADDR const block_partials =
        padded_cube_reductions + GetBlockNum() * matmul_size *
                                     KernelCubeReduce<T>::MAT_DIM_16 *
                                     sizeof(OutputT);

    KernelCompleteCubeReduce<OutputT, true /* Sync */> op_vec(matmul_size);
    op_vec.Init(padded_cube_reductions, block_partials, vec_out);
    op_vec.Process();
  }
}

}  // namespace tcuscan
