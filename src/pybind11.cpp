/**
 * @file pybind11.cpp
 *
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "torch/torch_compare.h"
#include "torch/torch_compress.h"
#include "torch/torch_copy.h"
#include "torch/torch_diff.h"
#include "torch/torch_gather.h"
#include "torch/torch_gen_lower.h"
#include "torch/torch_histogram.h"
#include "torch/torch_linalg.h"
#include "torch/torch_matmul_cce.h"
#include "torch/torch_pad.h"
#include "torch/torch_reduce.h"
#include "torch/torch_scan.h"
#include "torch/torch_seg_ops.h"
#include "torch/torch_sort.h"
#include "torch/torch_split.h"
#include "torch/torch_spmv.h"
#include "torch/torch_topk.h"
#include "torch/torch_vadd.h"

/**
 * @brief Pybind11 module.
 */
PYBIND11_MODULE(tcuscan_ops, m) {
  m.doc() = "TCUSCAN AscendC operators";
  m.def("run_add", &tcuscan::run_add, pybind11::arg("x"), pybind11::arg("y"),
        "Vector add");
  m.def("run_diff", &tcuscan::run_diff, pybind11::arg("x"),
        pybind11::arg("max_size") = 0, "Vector differentiation");
  m.def("run_seg_scan", &tcuscan::run_seg_scan, pybind11::arg("x"),
        pybind11::arg("f"), pybind11::arg("S"), "Segmented Scan");
  m.def("run_scan_multi_core", &tcuscan::run_scan_multi_core,
        pybind11::arg("x"), pybind11::arg("S"), "Multi-core Scan");
  m.def("run_scan_multi_core_no_l2", &tcuscan::run_scan_multi_core_no_l2,
        pybind11::arg("x"), pybind11::arg("S"),
        "Multi-core Scan (no L2 cache optimization)");
  m.def("run_csr_gather", &tcuscan::run_csr_gather, pybind11::arg("values"),
        pybind11::arg("cols"), pybind11::arg("x"), "CSR gather (for SpMV)");
  m.def("run_compress", &tcuscan::run_compress, pybind11::arg("x"),
        pybind11::arg("mask"), pybind11::arg("S"), "Compaction/compress");
  m.def("run_filter_greater_eq", &tcuscan::run_filter_greater_equal,
        pybind11::arg("x"), pybind11::arg("pivot"), pybind11::arg("S"),
        "Filter by great-equal pivot.");
  m.def("run_filter_less_eq", &tcuscan::run_filter_less_equal,
        pybind11::arg("x"), pybind11::arg("pivot"), pybind11::arg("S"),
        "Filter by less-equal pivot.");
  m.def("run_where", &tcuscan::run_where, pybind11::arg("x"),
        pybind11::arg("pivot"), pybind11::arg("S"), "torch.where operator.");
  m.def("run_compress_ind", &tcuscan::run_compress_ind, pybind11::arg("x"),
        pybind11::arg("indices_in"), pybind11::arg("mask"), pybind11::arg("S"),
        "Compaction that returns indices");
  m.def("run_compress_ind_no_arange", &tcuscan::run_compress_ind_no_arange,
        pybind11::arg("x"), pybind11::arg("mask"), pybind11::arg("S"),
        "Compaction that returns indices (no input indices)");
  m.def("run_compress_pos", &tcuscan::run_compress_pos, pybind11::arg("x"),
        pybind11::arg("mask"), pybind11::arg("output_len"), pybind11::arg("S"),
        "Compaction/compress with pre-computed output positions");
  m.def("run_seg_sum", &tcuscan::run_seg_sum, pybind11::arg("x"),
        pybind11::arg("f"), pybind11::arg("S"), "Segmented Sum");
  m.def("run_seg_sum_single_core", &tcuscan::run_seg_sum_single_core,
        pybind11::arg("x"), pybind11::arg("indptr"), pybind11::arg("s"),
        "Segmented Sum (single-core)");
  m.def("run_seg_sum_multi_core", &tcuscan::run_seg_sum_multi_core,
        pybind11::arg("x"), pybind11::arg("indptr"), pybind11::arg("s"),
        pybind11::arg("segm_offsets") = pybind11::none(),
        "Segmented Sum (multi-core)");
  m.def("run_seg_sum_single_cube", &tcuscan::run_seg_sum_single_cube,
        pybind11::arg("x"), pybind11::arg("upper"),
        pybind11::arg("lower_strict"), pybind11::arg("indptr"),
        pybind11::arg("s"), "Segmented Sum (single-cube)");
  m.def("run_spmv", &tcuscan::run_spmv, pybind11::arg("vals"),
        pybind11::arg("indptr"), pybind11::arg("cols"), pybind11::arg("x"),
        pybind11::arg("s"), "Sparse Matrix-Vector Multiplication");
  m.def("run_spmv_v2", &tcuscan::run_spmv_v2, pybind11::arg("vals"),
        pybind11::arg("indptr"), pybind11::arg("cols"), pybind11::arg("x"),
        pybind11::arg("s"),
        "Sparse Matrix-Vector Multiplication Using Segmented Sum");
  m.def("run_spmv_multi_cube", &tcuscan::run_spmv_multi_cube,
        pybind11::arg("vals"), pybind11::arg("indptr"), pybind11::arg("cols"),
        pybind11::arg("x"), pybind11::arg("upper"),
        pybind11::arg("lower_strict"),
        "Sparse Matrix-Vector Multiplication Using Multi-cube Scan");
  m.def("run_copy", &tcuscan::run_copy, pybind11::arg("x"), pybind11::arg("s"),
        "Copy single core");
  m.def("run_scan_batch", &tcuscan::run_scan_batch, pybind11::arg("x"),
        pybind11::arg("S"), "Scan Batch");
  m.def("run_scan_single_core", &tcuscan::run_scan_single_core,
        pybind11::arg("x"), pybind11::arg("S"),
        pybind11::arg("starting_sum") = 0, "Scan Single Core");
  m.def("run_seg_scan_vec", &tcuscan::run_seg_scan_vec, pybind11::arg("x"),
        pybind11::arg("f"), pybind11::arg("S"), "Segmented Scan (vector-only)");
  m.def("run_seg_scan_mc_revert", &tcuscan::run_seg_scan_mc_revert,
        pybind11::arg("x"), pybind11::arg("f"), pybind11::arg("diff"),
        "Vector Revert for MC Segmented Scan");
  m.def("run_topk_int16", &tcuscan::run_topk_int16, pybind11::arg("x"),
        pybind11::arg("k"), pybind11::arg("x_min"), pybind11::arg("x_max"),
        pybind11::arg("S"), "TopK using parallel splits (int16)");
  m.def("run_topk_fp16", &tcuscan::run_topk_fp16, pybind11::arg("x"),
        pybind11::arg("k"), pybind11::arg("x_min"), pybind11::arg("x_max"),
        pybind11::arg("S"), "TopK using parallel splits (fp16)");
  m.def("run_topk_pivot_fp16", &tcuscan::run_topk_pivot_fp16,
        pybind11::arg("x"), pybind11::arg("k"),
        pybind11::arg("num_samples") = 32, "K-largest value estimator (fp16)");
  m.def("run_split", &tcuscan::run_split, pybind11::arg("x"),
        pybind11::arg("mask"), pybind11::arg("S"), "Split (16-bits)");
  m.def("run_split_ind", &tcuscan::run_split_ind, pybind11::arg("x"),
        pybind11::arg("mask"), pybind11::arg("indices_in"), pybind11::arg("S"),
        "Split with indices (16-bits)");
  m.def("run_mc_gather", &tcuscan::run_mc_gather, pybind11::arg("values"),
        pybind11::arg("idxs"), pybind11::arg("tile_len"),
        "Vector Multi Core Gather");
  m.def("run_gather_spmv", &tcuscan::run_gather_spmv, pybind11::arg("values"),
        pybind11::arg("idxs"), pybind11::arg("tile_len"),
        "Vector Multi Core Gather SPMV");
  m.def("run_radix_sort", &tcuscan::run_radix_sort, pybind11::arg("x"),
        pybind11::arg("S"), "Radix sort using cube units");
  m.def("run_matmul_cce", &tcuscan::matmul_cce, pybind11::arg("a"),
        pybind11::arg("b"),
        "Matrix multiplication CCE kernel (B dims must be a multiple of 512)");
  m.def("run_row_scan", &tcuscan::run_row_scan, pybind11::arg("x"),
        pybind11::arg("S"), "Matrix multiplication row scan kernel");
  m.def("run_gen_lower", &tcuscan::run_gen_lower, pybind11::arg("matrix_size"),
        pybind11::arg("device"), pybind11::arg("dtype"),
        "Generate lower triangular matrix");
  m.def("run_reduce_tiles", &tcuscan::run_reduce_tiles, pybind11::arg("x"),
        pybind11::arg("tile_len"), pybind11::arg("num_blocks"),
        "Sum-reduce over tiles");
  m.def("run_cube_reduce", &tcuscan::run_cube_reduce, pybind11::arg("x"),
        pybind11::arg("num_blocks"), "Block reduction using AIC/AIV cores.");
  m.def("run_complete_rows", &tcuscan::run_complete_rows, pybind11::arg("x"),
        pybind11::arg("sums"), pybind11::arg("tile_width"),
        pybind11::arg("tile_height"), "Down-sweep (second) phase of MCSCAN");
  m.def("run_complete_blocks", &tcuscan::run_complete_blocks,
        pybind11::arg("x"), pybind11::arg("sums"), pybind11::arg("tile_length"),
        "Block-wise down-sweep (second) phase of block scan");
  m.def("run_block_scan", &tcuscan::run_block_scan, pybind11::arg("x"),
        pybind11::arg("upper"), pybind11::arg("lower_strict"),
        "Block scan on blocks of length S^2");
  m.def("run_simple_pad", &tcuscan::run_simple_pad, pybind11::arg("x"),
        pybind11::arg("align_len"),
        "Padding of an input tensor from length vec_len up to align_len");
  m.def("run_scan_multi_cube", &tcuscan::run_scan_multi_cube,
        pybind11::arg("x"), pybind11::arg("upper"),
        pybind11::arg("lower_strict"), "Multi-cube scan");
  m.def("run_scan_cpu", &tcuscan::run_scan_cpu, pybind11::arg("x"),
        "Scan on CPUs");
  m.def("run_tri_inv_col_sweep", &tcuscan::run_tri_inv_col_sweep,
        pybind11::arg("x"), "Unit upper triangular matrix inverses (fp16)");
  m.def("run_tri_inv_cube_col_sweep", &tcuscan::run_tri_inv_cube_col_sweep,
        pybind11::arg("x"), "Triangular matrix inverse using AIV/AICs(fp16)");
  m.def("run_triu_inv_rec_unroll", &tcuscan::run_triu_inv_rec_unroll,
        pybind11::arg("x"), "Upper triangular inverse");
  m.def("run_count_if", &tcuscan::run_count_if, pybind11::arg("x"),
        pybind11::arg("pivot"), pybind11::arg("tile_len"),
        pybind11::arg("compare_mode"), "Count if");
  m.def("run_histogram", &tcuscan::run_histogram, pybind11::arg("x"),
        pybind11::arg("num_bins"), "Histogram");
}
