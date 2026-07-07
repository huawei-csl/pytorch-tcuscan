#pragma once
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * @file kernel_spmv_ops_sparse.h
 * @brief Row-parallel CSR sparse matrix-vector multiplication (SpMV) kernel.
 *
 * Migrated from the CANN `ops-sparse` reference implementation
 * (src/spmv/arch22/kernels/spmv_kernel.h) and adapted to the tcuscan codebase
 * conventions. Computes the general SpMV
 * \f$ y \leftarrow \alpha \cdot A x + \beta \cdot y \f$ where \f$ A \f$ is
 * given in CSR format, together with the transposed variant \f$ A^\top x \f$.
 */

#include <type_traits>

#include "ascendc_kernel_operator.h"

using namespace AscendC;

namespace tcuscan {

/// @brief Number of bytes per UB data block; DataCopy/DataCopyPad operate at
/// this granularity, so every UB buffer must be sized to a multiple of it.
constexpr uint32_t SPMV_UB_BLOCK = 32;

/**
 * @brief Round a byte count up to a whole UB data block.
 *
 * DataCopyPad reads/writes a UB tensor in 32-byte blocks even for a single
 * element, so allocating an exact `count * sizeof(T)` buffer would let the DMA
 * write past the end of the allocation and corrupt neighbouring UB buffers.
 */
__aicore__ inline uint32_t UbBlockBytes(uint32_t bytes) {
  return (bytes + SPMV_UB_BLOCK - 1) & ~(SPMV_UB_BLOCK - 1);
}

/**
 * @brief CSR SpMV AscendC operator using a row-parallel strategy.
 *
 * Each AI Core processes a contiguous range of whole rows. Within a row the
 * computation follows the CopyIn -> Compute -> CopyOut three-stage pipeline.
 *
 * @tparam CompT Compute data type (e.g. `float` or `int32_t`).
 * @tparam ValT Storage data type of the CSR values / dense vector `x`.
 * @tparam OutT Storage data type of the output vector `y`.
 */
template <typename CompT, typename ValT = CompT, typename OutT = CompT>
class SpmvKernel {
  constexpr static uint32_t BUFFER_NUM = 1;

 public:
  __aicore__ inline SpmvKernel(){};

  __aicore__ inline void Init(GM_ADDR csrRowPtr, GM_ADDR csrColInd,
                              GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
                              uint32_t totalRowsNum, uint32_t totalColsNum,
                              CompT alpha, CompT beta);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn(int32_t currentRow, int32_t validNum);
  __aicore__ inline void CopyOut(int32_t currentRow);
  __aicore__ inline void Compute(int32_t currentRow, int32_t validNum);

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueColIdx,
      inQueueVals;
  AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
  AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> workQueueX,
      workQueueReduce, floatQueueReduce;
  AscendC::GlobalTensor<int32_t> csrRowPtrGm;
  AscendC::GlobalTensor<int32_t> csrColIndGm;
  AscendC::GlobalTensor<ValT> csrValGm;
  AscendC::GlobalTensor<ValT> xVecGm;
  AscendC::GlobalTensor<OutT> yVecGm;

  uint32_t totalRowsNum;
  uint32_t totalColsNum;
  uint32_t startRow;
  uint32_t blockRowNum;
  uint32_t blockLength;
  uint32_t tileLength;
  uint32_t startValIdx;
  uint32_t endValIdx;
  CompT alpha;
  CompT beta;
};

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::Init(
    GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec,
    GM_ADDR yVec, uint32_t totalRowsNum, uint32_t totalColsNum, CompT alpha,
    CompT beta) {
  this->totalRowsNum = totalRowsNum;
  this->totalColsNum = totalColsNum;
  this->alpha = alpha;
  this->beta = beta;

  uint32_t blockNum = AscendC::GetBlockNum();
  if (blockNum == 0) {
    this->blockRowNum = 0;
    return;
  }

  // Distribute rows evenly: the first `tempRowNum` cores take one extra row.
  uint32_t tempRowNum = totalRowsNum % blockNum;
  if (AscendC::GetBlockIdx() < tempRowNum) {
    this->blockRowNum = this->totalRowsNum / AscendC::GetBlockNum() + 1;
    this->startRow = this->blockRowNum * AscendC::GetBlockIdx();
  } else {
    this->blockRowNum = this->totalRowsNum / AscendC::GetBlockNum();
    this->startRow = this->blockRowNum * AscendC::GetBlockIdx() + tempRowNum;
  }

  if (this->blockRowNum == 0) {
    return;
  }

  // Slice out this core's rowPtr segment plus its vals/colIdx segment.
  csrRowPtrGm.SetGlobalBuffer((__gm__ int32_t *)csrRowPtr + this->startRow,
                              this->blockRowNum + 1);

  this->startValIdx = csrRowPtrGm(0);
  this->endValIdx = csrRowPtrGm(this->blockRowNum);
  if (endValIdx >= startValIdx) {
    this->blockLength = endValIdx - startValIdx;
  } else {
    this->blockLength = 0;
  }

  csrColIndGm.SetGlobalBuffer((__gm__ int32_t *)csrColInd + startValIdx,
                              this->blockLength);
  csrValGm.SetGlobalBuffer((__gm__ ValT *)csrVal + startValIdx,
                           this->blockLength);
  xVecGm.SetGlobalBuffer((__gm__ ValT *)xVec, this->totalColsNum);
  yVecGm.SetGlobalBuffer((__gm__ OutT *)yVec + this->startRow,
                         this->blockRowNum);

  // Compute the longest row handled by this core (tile-size basis).
  this->tileLength = 0;
  for (int i = 0; i < this->blockRowNum; i++) {
    uint32_t rowLength = csrRowPtrGm(i + 1) - csrRowPtrGm(i);
    if (rowLength > this->tileLength) this->tileLength = rowLength;
  }
  if (this->tileLength == 0) {
    this->tileLength = 8;  // All-empty rows still need a minimal alignment to
                           // avoid InitBuffer(0).
  }

  pipe.InitBuffer(inQueueColIdx, BUFFER_NUM,
                  UbBlockBytes(this->tileLength * sizeof(int32_t)));
  pipe.InitBuffer(inQueueVals, BUFFER_NUM,
                  UbBlockBytes(this->tileLength * sizeof(CompT)));
  pipe.InitBuffer(workQueueX, BUFFER_NUM,
                  UbBlockBytes(this->tileLength * sizeof(CompT)));
  pipe.InitBuffer(workQueueReduce, BUFFER_NUM,
                  UbBlockBytes(this->tileLength * sizeof(float)));
  pipe.InitBuffer(floatQueueReduce, BUFFER_NUM,
                  UbBlockBytes(this->tileLength * sizeof(float)));
  pipe.InitBuffer(outQueueY, BUFFER_NUM, SPMV_UB_BLOCK);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::CopyIn(int32_t currentRow,
                                                             int32_t validNum) {
  AscendC::LocalTensor<int32_t> colIdxLocal =
      inQueueColIdx.AllocTensor<int32_t>();
  AscendC::LocalTensor<CompT> valsLocal = inQueueVals.AllocTensor<CompT>();
  if (validNum > 0) {
    AscendC::DataCopyExtParams copyParamsColIdx{
        1, (uint32_t)(validNum * sizeof(int32_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> padParamsColIdx{false, 0, 0, 0};

    uint32_t offset = csrRowPtrGm(currentRow) - this->startValIdx;
    AscendC::DataCopyPad(colIdxLocal, csrColIndGm[offset], copyParamsColIdx,
                         padParamsColIdx);

    if constexpr (std::is_same_v<ValT, CompT>) {
      AscendC::DataCopyExtParams cpV{1, (uint32_t)(validNum * sizeof(ValT)), 0,
                                     0, 0};
      AscendC::DataCopyPadExtParams<ValT> padV{false, 0, 0, 0};
      AscendC::DataCopyPad(valsLocal, csrValGm[offset], cpV, padV);
    } else {
      // Cast from storage type ValT to compute type CompT.
      AscendC::LocalTensor<ValT> valsTmp = workQueueX.AllocTensor<ValT>();
      AscendC::DataCopyExtParams cpV{1, (uint32_t)(validNum * sizeof(ValT)), 0,
                                     0, 0};
      AscendC::DataCopyPadExtParams<ValT> padV{false, 0, 0, 0};
      AscendC::DataCopyPad(valsTmp, csrValGm[offset], cpV, padV);
      // MTE2->Vector: the DMA into valsTmp must complete before the Cast reads
      // it, otherwise the cast consumes uninitialized UB (observed as NaNs).
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::Cast<CompT, ValT>(valsLocal, valsTmp,
                                 AscendC::RoundMode::CAST_NONE, validNum);
      AscendC::PipeBarrier<PIPE_V>();
      workQueueX.FreeTensor(valsTmp);
    }
  }
  inQueueColIdx.EnQue(colIdxLocal);
  inQueueVals.EnQue(valsLocal);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::CopyOut(
    int32_t currentRow) {
  // outQueueY carries the fully-combined output element produced by Compute().
  AscendC::LocalTensor<OutT> yOut = outQueueY.DeQue<OutT>();
  AscendC::DataCopyExtParams copyParams{1, (uint32_t)sizeof(OutT), 0, 0, 0};
  AscendC::DataCopyPad(yVecGm[currentRow], yOut, copyParams);
  outQueueY.FreeTensor(yOut);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::Compute(
    int32_t currentRow, int32_t validNum) {
  AscendC::LocalTensor<int32_t> colIdxLocal = inQueueColIdx.DeQue<int32_t>();
  AscendC::LocalTensor<CompT> valsLocal = inQueueVals.DeQue<CompT>();

  // Dot product of the current row with the gathered x entries.
  CompT dot = static_cast<CompT>(0);
  if (validNum > 0) {
    AscendC::LocalTensor<CompT> xLocal = workQueueX.AllocTensor<CompT>();
    // The scalar unit reads colIdxLocal, which was filled by MTE2 in CopyIn.
    // The queue DeQue only inserts an MTE2->Vector event, so an explicit
    // barrier is needed before the scalar gather to keep the scalar unit from
    // racing the DMA (which manifests as a UB/D-cache bus fault on device).
    AscendC::PipeBarrier<PIPE_ALL>();
    for (int i = 0; i < validNum; i++) {
      xLocal.SetValue(i, static_cast<CompT>(xVecGm(colIdxLocal(i))));
    }
    // Scalar->Vector: xLocal writes must be visible before the vector product.
    AscendC::PipeBarrier<PIPE_ALL>();
    // Element-wise product: xLocal *= valsLocal.
    AscendC::Mul(xLocal, xLocal, valsLocal, validNum);
    AscendC::PipeBarrier<PIPE_V>();

    // Reduce in float for accuracy.
    AscendC::LocalTensor<float> sharedTmpBuffer =
        workQueueReduce.AllocTensor<float>();
    AscendC::LocalTensor<float> floatTmpBuffer =
        floatQueueReduce.AllocTensor<float>();
    if constexpr (std::is_same_v<CompT, float>) {
      AscendC::ReduceSum<float>(floatTmpBuffer, xLocal, sharedTmpBuffer,
                                validNum);
    } else {
      AscendC::Cast<float, CompT>(floatTmpBuffer, xLocal,
                                  AscendC::RoundMode::CAST_ROUND, validNum);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::ReduceSum<float>(floatTmpBuffer, floatTmpBuffer, sharedTmpBuffer,
                                validNum);
    }
    // Vector->Scalar: the reduced sum must be visible before it is read out.
    AscendC::PipeBarrier<PIPE_ALL>();
    dot = static_cast<CompT>(floatTmpBuffer.GetValue(0));

    workQueueReduce.FreeTensor(sharedTmpBuffer);
    floatQueueReduce.FreeTensor(floatTmpBuffer);
    workQueueX.FreeTensor(xLocal);
  }

  // z = alpha * (A row . x) + beta * y[row]. The single-element combine is done
  // with scalars to keep outQueueY a clean single-producer/single-consumer
  // buffer (a queue round-trip of y across CopyIn/Compute/CopyOut proved racy).
  const CompT yPrev = static_cast<CompT>(yVecGm(currentRow));
  const CompT z = this->alpha * dot + this->beta * yPrev;

  AscendC::LocalTensor<OutT> yOut = outQueueY.AllocTensor<OutT>();
  yOut.SetValue(0, static_cast<OutT>(z));
  // Scalar->MTE3: the value must be written before CopyOut stores it.
  AscendC::PipeBarrier<PIPE_ALL>();
  outQueueY.EnQue<OutT>(yOut);

  inQueueColIdx.FreeTensor(colIdxLocal);
  inQueueVals.FreeTensor(valsLocal);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernel<CompT, ValT, OutT>::Process() {
  if (this->blockRowNum == 0) return;

  for (int32_t m = 0; m < this->blockRowNum; m++) {
    int32_t validNum = csrRowPtrGm(m + 1) - csrRowPtrGm(m);
    CopyIn(m, validNum);
    Compute(m, validNum);
    CopyOut(m);
  }
}

/**
 * @brief Transposed CSR SpMV AscendC operator (\f$ y \leftarrow \alpha \cdot
 * A^\top x + \beta \cdot y \f$).
 *
 * Uses the same row-parallel distribution as @ref SpmvKernel: each AI Core
 * processes whole rows. However, each non-zero of a row now contributes to a
 * distinct column of `y`, so accumulation across rows onto the same `y[j]` is
 * made race-free with atomic adds.
 *
 * @tparam CompT Compute data type (e.g. `float` or `int32_t`).
 * @tparam ValT Storage data type of the CSR values / dense vector `x`.
 * @tparam OutT Storage data type of the output vector `y`.
 */
template <typename CompT, typename ValT = CompT, typename OutT = CompT>
class SpmvKernelTrans {
  constexpr static uint32_t BUFFER_NUM = 1;

 public:
  __aicore__ inline SpmvKernelTrans(){};

  __aicore__ inline void Init(GM_ADDR csrRowPtr, GM_ADDR csrColInd,
                              GM_ADDR csrVal, GM_ADDR xVec, GM_ADDR yVec,
                              uint32_t totalRowsNum, uint32_t totalColsNum,
                              CompT alpha, CompT beta);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void ScaleBeta();
  __aicore__ inline void CopyIn(int32_t currentRow, int32_t validNum);
  __aicore__ inline void Compute(int32_t currentRow, int32_t validNum);
  __aicore__ inline void CopyOut(int32_t currentRow, int32_t validNum);

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueColIdx,
      inQueueVals;
  AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
  AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> workQueueX,
      workQueueReduce, floatQueueReduce;
  AscendC::GlobalTensor<int32_t> csrRowPtrGm;
  AscendC::GlobalTensor<int32_t> csrColIndGm;
  AscendC::GlobalTensor<ValT> csrValGm;
  AscendC::GlobalTensor<ValT> xVecGm;
  AscendC::GlobalTensor<OutT> yVecGm;

  uint32_t totalRowsNum;
  uint32_t totalColsNum;
  uint32_t startRow;
  uint32_t blockRowNum;
  uint32_t blockLength;
  uint32_t tileLength;
  uint32_t startValIdx;
  uint32_t endValIdx;
  CompT alpha;
  CompT beta;

  // beta-scaling stage: the y segment this core is responsible for (read).
  uint32_t yBlockStart;
  uint32_t yBlockCount;
};

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::Init(
    GM_ADDR csrRowPtr, GM_ADDR csrColInd, GM_ADDR csrVal, GM_ADDR xVec,
    GM_ADDR yVec, uint32_t totalRowsNum, uint32_t totalColsNum, CompT alpha,
    CompT beta) {
  this->totalRowsNum = totalRowsNum;
  this->totalColsNum = totalColsNum;
  this->alpha = alpha;
  this->beta = beta;

  uint32_t blockNum = AscendC::GetBlockNum();
  if (blockNum > 0) {
    // ===== Row distribution (identical to SpmvKernel). =====
    uint32_t tempRowNum = totalRowsNum % blockNum;
    if (AscendC::GetBlockIdx() < tempRowNum) {
      this->blockRowNum = this->totalRowsNum / blockNum + 1;
      this->startRow = this->blockRowNum * AscendC::GetBlockIdx();
    } else {
      this->blockRowNum = this->totalRowsNum / blockNum;
      this->startRow = this->blockRowNum * AscendC::GetBlockIdx() + tempRowNum;
    }

    // ===== y-column distribution (beta-scaling stage: split totalColsNum y
    // elements evenly across the cores). =====
    uint32_t tempColNum = totalColsNum % blockNum;
    if (AscendC::GetBlockIdx() < tempColNum) {
      this->yBlockCount = this->totalColsNum / blockNum + 1;
      this->yBlockStart = this->yBlockCount * AscendC::GetBlockIdx();
    } else {
      this->yBlockCount = this->totalColsNum / blockNum;
      this->yBlockStart =
          this->yBlockCount * AscendC::GetBlockIdx() + tempColNum;
    }
  } else {
    this->blockRowNum = 0;
    this->yBlockCount = 0;
  }

  // xVecGm/yVecGm must be initialized on every core (ScaleBeta runs on all).
  xVecGm.SetGlobalBuffer((__gm__ ValT *)xVec, this->totalRowsNum);
  yVecGm.SetGlobalBuffer((__gm__ OutT *)yVec, this->totalColsNum);

  if (this->blockRowNum > 0) {
    csrRowPtrGm.SetGlobalBuffer((__gm__ int32_t *)csrRowPtr + this->startRow,
                                this->blockRowNum + 1);

    this->startValIdx = csrRowPtrGm(0);
    this->endValIdx = csrRowPtrGm(this->blockRowNum);
    if (endValIdx >= startValIdx) {
      this->blockLength = endValIdx - startValIdx;
    } else {
      this->blockLength = 0;
    }

    csrColIndGm.SetGlobalBuffer((__gm__ int32_t *)csrColInd + startValIdx,
                                this->blockLength);
    csrValGm.SetGlobalBuffer((__gm__ ValT *)csrVal + startValIdx,
                             this->blockLength);

    this->tileLength = 0;
    for (int i = 0; i < this->blockRowNum; i++) {
      uint32_t rowLength = csrRowPtrGm(i + 1) - csrRowPtrGm(i);
      if (rowLength > this->tileLength) this->tileLength = rowLength;
    }
    if (this->tileLength == 0) {
      this->tileLength = 8;
    }
  } else {
    this->tileLength = 8;
  }

  // UB buffers must be initialized on every core (all cores set up the pipe).
  pipe.InitBuffer(inQueueColIdx, BUFFER_NUM,
                  UbBlockBytes(this->tileLength * sizeof(int32_t)));
  pipe.InitBuffer(inQueueVals, BUFFER_NUM,
                  UbBlockBytes(this->tileLength * sizeof(CompT)));
  pipe.InitBuffer(workQueueX, BUFFER_NUM,
                  UbBlockBytes(this->tileLength * sizeof(CompT)));
  pipe.InitBuffer(workQueueReduce, BUFFER_NUM,
                  UbBlockBytes(this->tileLength * sizeof(float)));
  pipe.InitBuffer(floatQueueReduce, BUFFER_NUM,
                  UbBlockBytes(this->tileLength * sizeof(float)));
  pipe.InitBuffer(outQueueY, BUFFER_NUM, SPMV_UB_BLOCK);
}

/**
 * @brief beta-scaling stage: each core multiplies its own y segment by beta.
 *
 * Since y segments do not overlap across cores no atomic op is needed. When
 * beta == 0 the host has already zero-initialized y, so this returns early.
 * When beta == 1 no scaling is required and it returns immediately.
 */
template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::ScaleBeta() {
  if (this->yBlockCount == 0) return;

  // beta == 1: no scaling required.
  if (this->beta == static_cast<CompT>(1)) return;

  // Process y[ yBlockStart .. yBlockStart+yBlockCount ) in tileLength chunks.
  for (uint32_t offset = 0; offset < this->yBlockCount;
       offset += this->tileLength) {
    uint32_t chunkSize = this->tileLength;
    if (offset + chunkSize > this->yBlockCount) {
      chunkSize = this->yBlockCount - offset;
    }

    if constexpr (std::is_same_v<CompT, OutT>) {
      AscendC::LocalTensor<CompT> yChunkLocal = workQueueX.AllocTensor<CompT>();

      AscendC::DataCopyExtParams cp{1, (uint32_t)(chunkSize * sizeof(CompT)), 0,
                                    0, 0};
      AscendC::DataCopyPadExtParams<CompT> padParams{false, 0, 0, 0};
      AscendC::DataCopyPad(yChunkLocal, yVecGm[this->yBlockStart + offset], cp,
                           padParams);
      AscendC::PipeBarrier<PIPE_ALL>();

      AscendC::Muls(yChunkLocal, yChunkLocal, this->beta, chunkSize);
      AscendC::PipeBarrier<PIPE_ALL>();

      AscendC::DataCopyExtParams cp2{1, (uint32_t)(chunkSize * sizeof(CompT)),
                                     0, 0, 0};
      AscendC::DataCopyPad(yVecGm[this->yBlockStart + offset], yChunkLocal,
                           cp2);
      AscendC::PipeBarrier<PIPE_V>();

      workQueueX.FreeTensor(yChunkLocal);
    } else {
      // OutT != CompT (e.g. half output with float compute):
      // read y as OutT -> cast to CompT -> *beta -> cast back -> write y.
      AscendC::LocalTensor<OutT> yChunkOut = workQueueX.AllocTensor<OutT>();
      AscendC::LocalTensor<CompT> yChunkComp =
          floatQueueReduce.AllocTensor<CompT>();

      AscendC::DataCopyExtParams cp{1, (uint32_t)(chunkSize * sizeof(OutT)), 0,
                                    0, 0};
      AscendC::DataCopyPadExtParams<OutT> padParams{false, 0, 0, 0};
      AscendC::DataCopyPad(yChunkOut, yVecGm[this->yBlockStart + offset], cp,
                           padParams);
      AscendC::PipeBarrier<PIPE_ALL>();

      AscendC::Cast<CompT, OutT>(yChunkComp, yChunkOut,
                                 AscendC::RoundMode::CAST_ROUND, chunkSize);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Muls(yChunkComp, yChunkComp, this->beta, chunkSize);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast<OutT, CompT>(yChunkOut, yChunkComp,
                                 AscendC::RoundMode::CAST_ROUND, chunkSize);
      AscendC::PipeBarrier<PIPE_ALL>();

      AscendC::DataCopyExtParams cp2{1, (uint32_t)(chunkSize * sizeof(OutT)), 0,
                                     0, 0};
      AscendC::DataCopyPad(yVecGm[this->yBlockStart + offset], yChunkOut, cp2);
      AscendC::PipeBarrier<PIPE_V>();

      floatQueueReduce.FreeTensor(yChunkComp);
      workQueueX.FreeTensor(yChunkOut);
    }
  }
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::CopyIn(
    int32_t currentRow, int32_t validNum) {
  AscendC::LocalTensor<int32_t> colIdxLocal =
      inQueueColIdx.AllocTensor<int32_t>();
  AscendC::LocalTensor<CompT> valsLocal = inQueueVals.AllocTensor<CompT>();
  if (validNum > 0) {
    AscendC::DataCopyExtParams copyParamsColIdx{
        1, (uint32_t)(validNum * sizeof(int32_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> padParamsColIdx{false, 0, 0, 0};

    uint32_t offset = csrRowPtrGm(currentRow) - this->startValIdx;
    AscendC::DataCopyPad(colIdxLocal, csrColIndGm[offset], copyParamsColIdx,
                         padParamsColIdx);

    if constexpr (std::is_same_v<ValT, CompT>) {
      AscendC::DataCopyExtParams cpV{1, (uint32_t)(validNum * sizeof(ValT)), 0,
                                     0, 0};
      AscendC::DataCopyPadExtParams<ValT> padV{false, 0, 0, 0};
      AscendC::DataCopyPad(valsLocal, csrValGm[offset], cpV, padV);
    } else {
      AscendC::LocalTensor<ValT> valsTmp = workQueueX.AllocTensor<ValT>();
      AscendC::DataCopyExtParams cpV{1, (uint32_t)(validNum * sizeof(ValT)), 0,
                                     0, 0};
      AscendC::DataCopyPadExtParams<ValT> padV{false, 0, 0, 0};
      AscendC::DataCopyPad(valsTmp, csrValGm[offset], cpV, padV);
      // MTE2->Vector: the DMA into valsTmp must complete before the Cast.
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::Cast<CompT, ValT>(valsLocal, valsTmp,
                                 AscendC::RoundMode::CAST_NONE, validNum);
      AscendC::PipeBarrier<PIPE_V>();
      workQueueX.FreeTensor(valsTmp);
    }
  }
  inQueueColIdx.EnQue(colIdxLocal);
  inQueueVals.EnQue(valsLocal);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::Compute(
    int32_t currentRow, int32_t validNum) {
  AscendC::LocalTensor<CompT> valsLocal = inQueueVals.DeQue<CompT>();
  AscendC::LocalTensor<CompT> contribLocal = workQueueX.AllocTensor<CompT>();

  if (validNum > 0) {
    CompT xVal;
    if constexpr (std::is_same_v<ValT, CompT>) {
      xVal = xVecGm(this->startRow + currentRow);
    } else {
      AscendC::LocalTensor<ValT> xTmp = outQueueY.AllocTensor<ValT>();
      AscendC::LocalTensor<CompT> cTmp = floatQueueReduce.AllocTensor<CompT>();
      xTmp.SetValue(0, xVecGm(this->startRow + currentRow));
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::Cast<CompT, ValT>(cTmp, xTmp, AscendC::RoundMode::CAST_NONE, 1);
      AscendC::PipeBarrier<PIPE_ALL>();
      xVal = cTmp.GetValue(0);
      outQueueY.FreeTensor(xTmp);
      floatQueueReduce.FreeTensor(cTmp);
    }

    AscendC::Muls(contribLocal, valsLocal, xVal, validNum);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(contribLocal, contribLocal, this->alpha, validNum);
    AscendC::PipeBarrier<PIPE_V>();
  }
  workQueueX.EnQue(contribLocal);
  inQueueVals.FreeTensor(valsLocal);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::CopyOut(
    int32_t currentRow, int32_t validNum) {
  AscendC::LocalTensor<int32_t> colIdxLocal = inQueueColIdx.DeQue<int32_t>();
  AscendC::LocalTensor<CompT> contribLocal = workQueueX.DeQue<CompT>();

  if (validNum > 0) {
    // The scalar unit reads colIdxLocal (filled by MTE2 in CopyIn) and
    // contribLocal (produced by the vector unit in Compute). Insert an explicit
    // barrier before the scalar loop to avoid racing those producers, which
    // manifests as a UB/D-cache bus fault on device.
    AscendC::PipeBarrier<PIPE_ALL>();
    if constexpr (std::is_same_v<CompT, OutT>) {
      AscendC::LocalTensor<CompT> atomicElem = outQueueY.AllocTensor<CompT>();
      for (int j = 0; j < validNum; j++) {
        atomicElem.SetValue(0, contribLocal(j));
        // Scalar->MTE3 (and MTE3->Scalar for the next iteration): the store
        // must observe the just-written value, and must finish before it is
        // reused.
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams cp{1, (uint32_t)sizeof(CompT), 0, 0, 0};
        AscendC::SetAtomicAdd<CompT>();
        AscendC::DataCopyPad(yVecGm[colIdxLocal(j)], atomicElem, cp);
        AscendC::SetAtomicNone();
      }
      outQueueY.FreeTensor(atomicElem);
    } else {
      // CompT != OutT: cast contrib -> OutT and atomic-add directly into y.
      AscendC::LocalTensor<OutT> oCast = floatQueueReduce.AllocTensor<OutT>();

      AscendC::Cast<OutT, CompT>(oCast, contribLocal,
                                 AscendC::RoundMode::CAST_ROUND, validNum);
      // Vector->Scalar sync: oCast is read element-wise by the scalar unit.
      AscendC::PipeBarrier<PIPE_ALL>();

      AscendC::LocalTensor<OutT> atomicElem = outQueueY.AllocTensor<OutT>();
      for (int j = 0; j < validNum; j++) {
        atomicElem.SetValue(0, oCast(j));
        // Scalar->MTE3 (and MTE3->Scalar for the next iteration).
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams cp{1, (uint32_t)sizeof(OutT), 0, 0, 0};
        AscendC::SetAtomicAdd<OutT>();
        AscendC::DataCopyPad(yVecGm[colIdxLocal(j)], atomicElem, cp);
        AscendC::SetAtomicNone();
      }
      outQueueY.FreeTensor(atomicElem);
      floatQueueReduce.FreeTensor(oCast);
    }
  }

  inQueueColIdx.FreeTensor(colIdxLocal);
  workQueueX.FreeTensor(contribLocal);
}

template <typename CompT, typename ValT, typename OutT>
__aicore__ inline void SpmvKernelTrans<CompT, ValT, OutT>::Process() {
  // Stage 1: beta scaling (cores work on disjoint segments, no atomics needed).
  ScaleBeta();
  AscendC::SyncAll<true /*isAIVOnly*/>();

  // Stage 2: process rows, atomically accumulating each row's contributions.
  for (int32_t m = 0; m < this->blockRowNum; m++) {
    int32_t validNum = csrRowPtrGm(m + 1) - csrRowPtrGm(m);
    CopyIn(m, validNum);
    Compute(m, validNum);
    CopyOut(m, validNum);
  }
  AscendC::SyncAll<true /*isAIVOnly*/>();
}

/**
 * @brief Run the `spmv_ops_sparse` kernel.
 *
 * Computes \f$ y \leftarrow \alpha \cdot A x + \beta \cdot y \f$ (or the
 * transposed product when @p trans is true) for a CSR matrix \f$ A \f$.
 *
 * @tparam CompT Compute data type.
 * @tparam ValT Storage type of the CSR values and dense vector `x`.
 * @tparam OutT Storage type of the output vector `y`.
 *
 * @param [in] csrRowPtr CSR row-pointer array (length rows + 1).
 * @param [in] csrColInd CSR column-index array (length nnz).
 * @param [in] csrVal CSR non-zero values (length nnz).
 * @param [in] xVec Dense input vector.
 * @param [in,out] yVec Dense output vector (also read for the beta term).
 * @param [in] totalRowsNum Number of rows of the CSR matrix.
 * @param [in] totalColsNum Number of columns of the CSR matrix.
 * @param [in] alpha Scaling factor for the matrix-vector product.
 * @param [in] beta Scaling factor for the pre-existing output vector.
 * @param [in] trans Whether to compute the transposed product.
 */
template <typename CompT, typename ValT = CompT, typename OutT = CompT>
__aicore__ inline void run_spmv_ops_sparse(GM_ADDR csrRowPtr, GM_ADDR csrColInd,
                                           GM_ADDR csrVal, GM_ADDR xVec,
                                           GM_ADDR yVec, uint32_t totalRowsNum,
                                           uint32_t totalColsNum, CompT alpha,
                                           CompT beta, bool trans) {
  if (trans) {
    SpmvKernelTrans<CompT, ValT, OutT> op;
    op.Init(csrRowPtr, csrColInd, csrVal, xVec, yVec, totalRowsNum,
            totalColsNum, alpha, beta);
    op.Process();
  } else {
    SpmvKernel<CompT, ValT, OutT> op;
    op.Init(csrRowPtr, csrColInd, csrVal, xVec, yVec, totalRowsNum,
            totalColsNum, alpha, beta);
    op.Process();
  }
}

}  // namespace tcuscan
