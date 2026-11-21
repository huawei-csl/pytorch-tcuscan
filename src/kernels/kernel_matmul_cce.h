#pragma once

#include "ascendc_kernel_operator.h"

#ifdef __DAV_C220_CUBE__
// Cube-specific CCE intrinsics need to be inside this MACRO

#define CUBE_SIZE 256
#define CUBE_BLOCK_SIZE 16

#define L2_SWIZZLE

inline __aicore__ void CopyToL12D(__cbuf__ __fp16 *L1, __gm__ __fp16 *gm,
                                  int row, int col, int height, int width,
                                  int step) {
  auto offset = (row * step + col);
  copy_gm_to_cbuf_multi_nd2nz_b16(
      L1, gm + offset, 0, 1 /* nd2nzParams.ndNum */,
      height /* nd2nzParams.nValue */, width /* nd2nzParams.dValue */,
      0 /* nd2nzParams.srcNdMatrixStride */, step /* nd2nzParams.srcDValue */,
      height /* nd2nzParams.dstNzC0Stride */, 1 /* nd2nzParams.dstNzNStride */,
      0 /* nd2nzParams.dstNzMatrixStride */);
}

inline __aicore__ void CopyToL0ACol(__ca__ __fp16 *l0, __cbuf__ __fp16 *cbuf,
                                    int m_factor, int k_factor_start,
                                    int k_factor_num) {
  int startIndex = 0;
  int repeatTimes = m_factor;
  int dstGap = k_factor_num - 1;
  int srcStride = 1;
  int src_offset = m_factor * CUBE_SIZE;
  int sid = 0;
  for (int k = k_factor_start; k < k_factor_start + k_factor_num; ++k) {
    load_cbuf_to_ca(l0 + (k - k_factor_start) * CUBE_SIZE,
                    cbuf + k * src_offset, startIndex, repeatTimes, srcStride,
                    dstGap, sid, 0 /*transpose*/, inc);
  }
}

inline __aicore__ void CopyToL0BTCol(__cb__ __fp16 *l0, __cbuf__ __fp16 *cbuf,
                                     int n_factor, int k_factor_start,
                                     int k_factor_num) {
  int src_step = n_factor * CUBE_SIZE;
  int startIndex = 0;
  int repeatTimes = k_factor_num * n_factor;
  int dstGap = 0;
  int srcStride = 1;
  int sid = 0;
  load_cbuf_to_cb(l0, cbuf + k_factor_start * src_step, startIndex, repeatTimes,
                  srcStride, dstGap, sid, 0 /*transpose*/, inc);
}

inline __aicore__ void Mmad(__cc__ float *cc, __ca__ __fp16 *ca,
                            __cb__ __fp16 *cb, int m, int n, int k,
                            bool init_val, unsigned char unit_flag = 0) {
  mad(cc, ca, cb, m, k, n, unit_flag, false /* kDirection Align */, 0,
      init_val);
}

inline __aicore__ void CopyToGm(__gm__ __fp16 *z, __cc__ float *cc, int mOffset,
                                int nOffset, int m, int n, int N,
                                unsigned char unit_flag = 0) {
  uint64_t config = 0x1;
  set_nd_para(config);
  int cOffset = mOffset * N + nOffset;
  copy_matrix_cc_to_gm(
      z + cOffset, cc, 0 /* fixpipeInfo.sid */, n, m,
      N /* fixpipeInfo.dstStride */, m /* fixpipeInfo.srcStride */,
      unit_flag /* fixpipeInfo.unitFlag */, QuantMode_t::F322F16,
      0 /* static_cast<uint8_t>(fixpipeInfo.reluEn) */,
      0 /* fixpipeInfo.channelSplit */, 1 /* fixpipeInfo.nz2ndEn */);
}

// transforms `m_idx` and `n_idx` for L2 locality
inline __aicore__ void SwizzleBlockIdx(int32_t loop_idx, int32_t m_loop,
                                       int32_t n_loop, int32_t swizzl_direction,
                                       int32_t swizzl_count, int64_t &m_idx,
                                       int64_t &n_idx) {
  uint32_t in_batch_idx = loop_idx % (m_loop * n_loop);
  if (swizzl_direction == 0) {  // Zn
    uint32_t tile_block_loop = (m_loop + swizzl_count - 1) / swizzl_count;
    uint32_t tile_block_idx = in_batch_idx / (swizzl_count * n_loop);
    uint32_t in_tile_block_idx = in_batch_idx % (swizzl_count * n_loop);

    uint32_t n_row = swizzl_count;
    if (tile_block_idx == tile_block_loop - 1) {
      n_row = m_loop - swizzl_count * tile_block_idx;
    }
    m_idx = tile_block_idx * swizzl_count + in_tile_block_idx % n_row;
    n_idx = in_tile_block_idx / n_row;
    if (tile_block_idx % 2 != 0) {
      n_idx = n_loop - n_idx - 1;
    }
  } else if (swizzl_direction == 1) {  // Nz
    uint32_t tile_block_loop = (n_loop + swizzl_count - 1) / swizzl_count;
    uint32_t tile_block_idx = in_batch_idx / (swizzl_count * m_loop);
    uint32_t in_tile_block_idx = in_batch_idx % (swizzl_count * m_loop);

    uint32_t n_col = swizzl_count;
    if (tile_block_idx == tile_block_loop - 1) {
      n_col = n_loop - swizzl_count * tile_block_idx;
    }
    m_idx = in_tile_block_idx / n_col;
    n_idx = tile_block_idx * swizzl_count + in_tile_block_idx % n_col;
    if (tile_block_idx % 2 != 0) {
      m_idx = m_loop - m_idx - 1;
    }
  }
}

inline __aicore__ void run_matmul_cce(GM_ADDR x, GM_ADDR y, GM_ADDR z, int M,
                                      int N, int K) {
  int32_t swizzlCount = 3;       // tune me
  int32_t swizzleDirection = 1;  // tune me
  uint64_t l2cacheoffset = 0;

  int mTileSize = 128;  // assume M dimension is multiple of 128
  int mTileNum = M / mTileSize;
  int mTileFactor = mTileSize / CUBE_BLOCK_SIZE;

  int nTileSize = 256;  // assume N dimension is multiple of 256
  int nTileFactor = nTileSize / CUBE_BLOCK_SIZE;

  constexpr int kTileSize = 256;  // assume N dimension is multiple of 512
  constexpr int kTileFactor = kTileSize / CUBE_BLOCK_SIZE;
  constexpr int kHtileSize = kTileSize >> 1;
  constexpr int kDtileSize = kTileSize << 1;
  int kDtileNum = K / kDtileSize;

  constexpr int kHtileFactor = kHtileSize / CUBE_BLOCK_SIZE;
  static_assert(kTileSize % 4 == 0);
  constexpr int kQtileSize = kTileSize >> 2;
  constexpr int kQtileFactor = kQtileSize / CUBE_BLOCK_SIZE;

  __gm__ __fp16 *x_gm_addr = ((__gm__ __fp16 *)x);
  int l1ATileSize = mTileSize * kDtileSize;
  int l1ATileBytes = l1ATileSize * sizeof(__fp16);
  __cbuf__ __fp16 *A_cbuf_addr0 =
      reinterpret_cast<__cbuf__ __fp16 *>((uintptr_t)0);
  __cbuf__ __fp16 *A_cbuf_addr1 =
      reinterpret_cast<__cbuf__ __fp16 *>((uintptr_t)l1ATileBytes);
  __cbuf__ __fp16 *A_cbuf_addr[2] = {A_cbuf_addr0, A_cbuf_addr1};

  int l0ATileSize = mTileSize * kQtileSize;
  int l0ATileBytes = l0ATileSize * sizeof(__fp16);
  __ca__ __fp16 *ca_addr0 = reinterpret_cast<__ca__ __fp16 *>((uintptr_t)0);
  __ca__ __fp16 *ca_addr1 =
      reinterpret_cast<__ca__ __fp16 *>((uintptr_t)l0ATileBytes);

  __gm__ __fp16 *y_gm_addr = ((__gm__ __fp16 *)y);
  if (mTileNum == 1) {
    y_gm_addr = ((__gm__ __fp16 *)((uint64_t)y + l2cacheoffset));
  }
  int l1BStart = l1ATileBytes * 2;
  int l1BTileSize = nTileSize * kTileSize;
  int l1BTileBytes = l1BTileSize * sizeof(__fp16);
  __cbuf__ __fp16 *B_cbuf_addr0 =
      reinterpret_cast<__cbuf__ __fp16 *>((uintptr_t)l1BStart);
  __cbuf__ __fp16 *B_cbuf_addr1 =
      reinterpret_cast<__cbuf__ __fp16 *>((uintptr_t)(l1BStart + l1BTileBytes));

  int l0BTileSize = nTileSize * kQtileSize;
  int l0BTileBytes = l0BTileSize * sizeof(__fp16);
  __cb__ __fp16 *cb_addr0 = reinterpret_cast<__cb__ __fp16 *>((uintptr_t)0);
  __cb__ __fp16 *cb_addr1 =
      reinterpret_cast<__cb__ __fp16 *>((uintptr_t)l0BTileBytes);

  __gm__ __fp16 *z_gm_addr = ((__gm__ __fp16 *)z);
  __cc__ float *cc_addr = reinterpret_cast<__cc__ float *>((uintptr_t)0);

  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  int curr = 0;
  bool firstIter = true;

  int nLoop = N / nTileSize;
  if (N % nTileSize != 0) {
    nLoop = nLoop + 1;
  }
  int mLoop = M / mTileSize;
  int coreLoop = nLoop * mLoop;

  for (int32_t loop_idx = 0; loop_idx < coreLoop; loop_idx++) {
    if (loop_idx % block_num != block_idx) {
      continue;
    }
    int64_t midx = loop_idx / nLoop;
    int64_t nidx = loop_idx % nLoop;

#ifdef L2_SWIZZLE
    SwizzleBlockIdx(loop_idx, mLoop, nLoop, swizzleDirection, swizzlCount, midx,
                    nidx);
#endif

    nTileSize = 256;
    int nOffset = nidx * nTileSize;
    if (nOffset + nTileSize > N) {
      nTileSize = 128;
    }
    nTileFactor = nTileSize / CUBE_BLOCK_SIZE;
    int mOffset = midx * mTileSize;

    if (!firstIter) {
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    }

    int kStart = 0;

    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0 + curr);
    CopyToL12D(A_cbuf_addr[curr], x_gm_addr, mOffset, kStart, mTileSize,
               kDtileSize, K);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0 + curr);
    for (int kIdx = 0; kIdx < kDtileNum; ++kIdx) {
      bool firstKIter = (kIdx == 0);
      int next = 1 - curr;

      int kOffset = kIdx * kDtileSize;
      int kOffsetNext = kOffset + kDtileSize;

      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);

      CopyToL12D(B_cbuf_addr0, y_gm_addr, nOffset, kOffset + 0 * kTileSize,
                 nTileSize, kTileSize, K);

      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);

      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0 + curr);
      CopyToL0ACol(ca_addr0, A_cbuf_addr[curr], mTileFactor,
                   0 * kTileFactor + 0 * kHtileFactor + 0 * kQtileFactor,
                   kQtileFactor);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
      CopyToL0BTCol(cb_addr0, B_cbuf_addr0, nTileFactor,
                    0 * kHtileFactor + 0 * kQtileFactor, kQtileFactor);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      Mmad(cc_addr, ca_addr0, cb_addr0, mTileSize, nTileSize, kQtileSize,
           firstKIter /* init_val */, 0 /* unit_flat */);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);

      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      CopyToL0ACol(ca_addr1, A_cbuf_addr[curr], mTileFactor,
                   0 * kTileFactor + 0 * kHtileFactor + 1 * kQtileFactor,
                   kQtileFactor);
      CopyToL0BTCol(cb_addr1, B_cbuf_addr0, nTileFactor,
                    0 * kHtileFactor + 1 * kQtileFactor, kQtileFactor);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      Mmad(cc_addr, ca_addr1, cb_addr1, mTileSize, nTileSize, kQtileSize,
           0 /* init_val */, 0 /* unit_flat */);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      CopyToL0ACol(ca_addr0, A_cbuf_addr[curr], mTileFactor,
                   0 * kTileFactor + 1 * kHtileFactor + 0 * kQtileFactor,
                   kQtileFactor);
      CopyToL0BTCol(cb_addr0, B_cbuf_addr0, nTileFactor,
                    1 * kHtileFactor + 0 * kQtileFactor, kQtileFactor);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      Mmad(cc_addr, ca_addr0, cb_addr0, mTileSize, nTileSize, kQtileSize,
           0 /* init_val */, 0 /* unit_flat */);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);

      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      CopyToL0ACol(ca_addr1, A_cbuf_addr[curr], mTileFactor,
                   0 * kTileFactor + 1 * kHtileFactor + 1 * kQtileFactor,
                   kQtileFactor);
      CopyToL0BTCol(cb_addr1, B_cbuf_addr0, nTileFactor,
                    1 * kHtileFactor + 1 * kQtileFactor, kQtileFactor);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      Mmad(cc_addr, ca_addr1, cb_addr1, mTileSize, nTileSize, kQtileSize,
           0 /* init_val */, 0 /* unit_flat */);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);

      CopyToL12D(B_cbuf_addr1, y_gm_addr, nOffset, kOffset + kTileSize,
                 nTileSize, kTileSize, K);

      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);

      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      CopyToL0ACol(ca_addr0, A_cbuf_addr[curr], mTileFactor,
                   1 * kTileFactor + 0 * kHtileFactor + 0 * kQtileFactor,
                   kQtileFactor);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
      CopyToL0BTCol(cb_addr0, B_cbuf_addr1, nTileFactor,
                    0 * kHtileFactor + 0 * kQtileFactor, kQtileFactor);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      Mmad(cc_addr, ca_addr0, cb_addr0, mTileSize, nTileSize, kQtileSize,
           0 /* init_val */, 0 /* unit_flat */);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);

      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      CopyToL0ACol(ca_addr1, A_cbuf_addr[curr], mTileFactor,
                   1 * kTileFactor + 0 * kHtileFactor + 1 * kQtileFactor,
                   kQtileFactor);
      CopyToL0BTCol(cb_addr1, B_cbuf_addr1, nTileFactor,
                    0 * kHtileFactor + 1 * kQtileFactor, kQtileFactor);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      Mmad(cc_addr, ca_addr1, cb_addr1, mTileSize, nTileSize, kQtileSize,
           0 /* init_val */, 0 /* unit_flat */);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      CopyToL0ACol(ca_addr0, A_cbuf_addr[curr], mTileFactor,
                   1 * kTileFactor + 1 * kHtileFactor + 0 * kQtileFactor,
                   kQtileFactor);
      CopyToL0BTCol(cb_addr0, B_cbuf_addr1, nTileFactor,
                    1 * kHtileFactor + 0 * kQtileFactor, kQtileFactor);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      Mmad(cc_addr, ca_addr0, cb_addr0, mTileSize, nTileSize, kQtileSize,
           0 /* init_val */, 0 /* unit_flat */);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);

      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      CopyToL0ACol(ca_addr1, A_cbuf_addr[curr], mTileFactor,
                   1 * kTileFactor + 1 * kHtileFactor + 1 * kQtileFactor,
                   kQtileFactor);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0 + curr);
      CopyToL0BTCol(cb_addr1, B_cbuf_addr1, nTileFactor,
                    1 * kHtileFactor + 1 * kQtileFactor, kQtileFactor);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      Mmad(cc_addr, ca_addr1, cb_addr1, mTileSize, nTileSize, kQtileSize,
           0 /* init_val */, 0 /* unit_flat */);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

      if (kIdx + 1 < kDtileNum) {
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0 + next);
        CopyToL12D(A_cbuf_addr[next], x_gm_addr, mOffset,
                   kOffsetNext /* + kDtileSize */, mTileSize, kDtileSize, K);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0 + next);
      }
      curr ^= 1;
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    CopyToGm(z_gm_addr, cc_addr, mOffset, nOffset, mTileSize, nTileSize, N,
             0 /* unit_flag */);

    if (loop_idx + block_num < coreLoop) {
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    }

    firstIter = false;
  }  // M *N
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
}

#elif defined(__DAV_C310_CUBE__)
// dav-c310 (A5) cube path.
//
// The dav-c220 implementation above is written directly against the CCE
// intrinsics (`load_cbuf_to_ca`, `copy_matrix_cc_to_gm`, manual L1/L0 base
// addresses and `set_flag`/`wait_flag` pipelines). Every one of those either
// does not exist on A5 or took a different signature there, so the kernel
// used to compile down to the vector stub and produced garbage. This is a
// straightforward re-implementation on top of the AscendC API, which keeps
// the same semantics (C = A @ B^T, fp16 in and out) without the hand-rolled
// double buffering and L2 swizzle of the dav-c220 version.

#include "tcuscan_utils.h"

/**
 * @brief Copy a `rows` x `cols` tile out of a row-major global matrix into an
 * L1 tensor, converting ND to NZ on the fly.
 *
 * Unlike `tcuscan::copy::CopyGmToL1A` this takes the leading dimension of the
 * source separately, so it can read a tile out of a wider matrix.
 *
 * @param [in] dst Destination tensor in L1.
 * @param [in] src Source global tensor, positioned at the tile origin.
 * @param [in] rows Number of rows of the tile.
 * @param [in] cols Number of columns of the tile.
 * @param [in] src_leading_dim Number of columns of the full source matrix.
 */
inline __aicore__ void MatmulCceCopyTileToL1(
    const AscendC::LocalTensor<half> &dst,
    const AscendC::GlobalTensor<half> &src, uint16_t rows, uint16_t cols,
    uint16_t src_leading_dim) {
  AscendC::Nd2NzParams params;
  params.ndNum = 1;
  params.nValue = rows;
  params.dValue = cols;
  params.srcNdMatrixStride = 0;
  params.srcDValue = src_leading_dim;
  params.dstNzC0Stride = rows;
  params.dstNzNStride = 1;
  params.dstNzMatrixStride = 0;
  AscendC::DataCopy(dst, src, params);
}

/**
 * @brief Write an `m_size` x `n_size` fp32 tile from L0C into a row-major fp16
 * global matrix, at a row stride of `dst_leading_dim`.
 *
 * @param [in] dst Destination global tensor, positioned at the tile origin.
 * @param [in] src Source tensor in L0C.
 * @param [in] m_size Number of rows of the tile.
 * @param [in] n_size Number of columns of the tile.
 * @param [in] dst_leading_dim Number of columns of the full destination.
 */
inline __aicore__ void MatmulCceCopyTileToGm(
    const AscendC::GlobalTensor<half> &dst,
    const AscendC::LocalTensor<float> &src, uint16_t m_size, uint16_t n_size,
    uint32_t dst_leading_dim) {
  constexpr uint16_t fractal = 16;
  AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> params;
  params.nSize = n_size;
  params.mSize = m_size;
  params.srcStride = (m_size + fractal - 1) / fractal * fractal;
  params.dstStride = dst_leading_dim;
  params.quantPre = QuantMode_t::F322F16;
  params.params = AscendC::Nz2NdParams(/*ndNum=*/1, /*srcNdStride=*/0,
                                       /*dstNdStride=*/0);
  AscendC::Fixpipe<half, float, AscendC::CFG_ROW_MAJOR>(dst, src, params);
}

/**
 * @brief Compute C = A @ B^T on the cube unit.
 *
 * As in the dav-c220 version, `M`, `N` and `K` are assumed to be multiples of
 * the 128-element tile size.
 *
 * @param [in] x Pointer to A, a row-major `M` x `K` fp16 matrix.
 * @param [in] y Pointer to B, a row-major `N` x `K` fp16 matrix.
 * @param [in] z Pointer to C, a row-major `M` x `N` fp16 matrix.
 * @param [in] M Number of rows of A.
 * @param [in] N Number of rows of B.
 * @param [in] K Number of columns of A and B.
 */
inline __aicore__ void run_matmul_cce(GM_ADDR x, GM_ADDR y, GM_ADDR z, int M,
                                      int N, int K) {
  using tcuscan::GetFractalMN;

  constexpr uint16_t kFractal = 16;
  // 128x128 fp32 accumulator is 64 KB of L0C; the A and B tiles are 32 KB
  // each in L1 and in L0A/L0B.
  constexpr uint16_t kMTile = 128;
  constexpr uint16_t kNTile = 128;
  constexpr uint16_t kKTile = 128;

  const uint32_t m_loop = tcuscan::scalar::CeilDiv<uint32_t>(M, kMTile);
  const uint32_t n_loop = tcuscan::scalar::CeilDiv<uint32_t>(N, kNTile);
  const uint32_t k_loop = tcuscan::scalar::CeilDiv<uint32_t>(K, kKTile);

  AscendC::GlobalTensor<half> global_a;
  AscendC::GlobalTensor<half> global_b;
  AscendC::GlobalTensor<half> global_c;
  global_a.SetGlobalBuffer((__gm__ half *)x);
  global_b.SetGlobalBuffer((__gm__ half *)y);
  global_c.SetGlobalBuffer((__gm__ half *)z);

  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::A1, 1> a1_q;
  AscendC::TQue<AscendC::QuePosition::B1, 1> b1_q;
  AscendC::TQue<AscendC::QuePosition::A2, 1> a2_q;
  AscendC::TQue<AscendC::QuePosition::B2, 1> b2_q;
  AscendC::TQue<AscendC::QuePosition::CO1, 1> co1_q;

  pipe.InitBuffer(a1_q, 1, kMTile * kKTile * sizeof(half));
  pipe.InitBuffer(b1_q, 1, kNTile * kKTile * sizeof(half));
  pipe.InitBuffer(a2_q, 1, kMTile * kKTile * sizeof(half));
  pipe.InitBuffer(b2_q, 1, kNTile * kKTile * sizeof(half));
  pipe.InitBuffer(co1_q, 1, kMTile * kNTile * sizeof(float));

  const uint32_t num_tiles = m_loop * n_loop;
  const uint32_t block_idx = AscendC::GetBlockIdx();
  const uint32_t block_num = AscendC::GetBlockNum();

  for (uint32_t tile = block_idx; tile < num_tiles; tile += block_num) {
    const uint32_t m_idx = tile / n_loop;
    const uint32_t n_idx = tile % n_loop;
    const uint32_t m_off = m_idx * kMTile;
    const uint32_t n_off = n_idx * kNTile;

    for (uint32_t k_idx = 0; k_idx < k_loop; ++k_idx) {
      const uint32_t k_off = k_idx * kKTile;

      // A tile: rows [m_off, m_off + kMTile) of the M x K matrix.
      const AscendC::LocalTensor<half> a1_lt = a1_q.AllocTensor<half>();
      MatmulCceCopyTileToL1(a1_lt, global_a[m_off * K + k_off], kMTile, kKTile,
                            static_cast<uint16_t>(K));
      a1_q.EnQue(a1_lt);
      tcuscan::copy::CopyL1ToL0A<half, true>(a2_q, a1_q, kMTile / kFractal,
                                             kKTile / kFractal);

      // B tile: rows [n_off, n_off + kNTile) of the N x K matrix. Because B is
      // stored transposed with respect to the cube's B operand, the ND -> NZ
      // conversion already yields the ZN layout `Mmad` expects, so the L1 ->
      // L0B step is a plain contiguous copy.
      const AscendC::LocalTensor<half> b1_lt = b1_q.AllocTensor<half>();
      MatmulCceCopyTileToL1(b1_lt, global_b[n_off * K + k_off], kNTile, kKTile,
                            static_cast<uint16_t>(K));
      b1_q.EnQue(b1_lt);
      {
        AscendC::LocalTensor<half> src = b1_q.DeQue<half>();
        const AscendC::LocalTensor<half> dst = b2_q.AllocTensor<half>();
        AscendC::LoadData2dParams params;
        params.repeatTimes =
            (kNTile / kFractal) * (kKTile / kFractal);
        params.srcStride = 1;
        params.ifTranspose = false;
        AscendC::LoadData(dst, src, params);
        b1_q.FreeTensor(src);
        b2_q.EnQue(dst);
      }

      if (k_idx == 0) {
        tcuscan::cube_unit::Multiply<half, false, true, true>(
            a2_q, b2_q, co1_q, kMTile, kNTile, kKTile);
      } else {
        tcuscan::cube_unit::Multiply<half, true, true, true>(
            a2_q, b2_q, co1_q, kMTile, kNTile, kKTile);
      }
      AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::LocalTensor<float> co1_lt = co1_q.DeQue<float>();
    MatmulCceCopyTileToGm(global_c[m_off * N + n_off], co1_lt, kMTile, kNTile,
                          static_cast<uint32_t>(N));
    co1_q.FreeTensor(co1_lt);
    AscendC::PipeBarrier<PIPE_ALL>();
  }
}

#else  // __DAV_C220_VEC__ / __DAV_C310_VEC__

// Stub for vector compile path
inline __aicore__ void run_matmul_cce(GM_ADDR a, GM_ADDR b, GM_ADDR c, int M,
                                      int N, int K) {
  pipe_barrier(PIPE_ALL);
}

#endif  // end of cube vs vector compile path switch
