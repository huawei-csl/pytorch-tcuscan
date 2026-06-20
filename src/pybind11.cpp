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
  m.def("run_add", &tcuscan::run_add, "Vector add");
  m.def("run_diff", &tcuscan::run_diff, pybind11::arg(),
        pybind11::arg("max_size") = 0, "Vector diff");
  m.def("run_seg_scan", &tcuscan::run_seg_scan, "Segmented Scan");
  m.def("run_scan_multi_core", &tcuscan::run_scan_multi_core,
        "Multi-core Scan");
  m.def("run_scan_multi_core_no_l2", &tcuscan::run_scan_multi_core_no_l2,
        "Multi-core Scan (no L2 cache optimization)");
  m.def("run_csr_gather", &tcuscan::run_csr_gather, "CSR gather");
  m.def("run_compress", &tcuscan::run_compress, "Compaction/compress");
  m.def("run_filter_greater_eq", &tcuscan::run_filter_greater_equal,
        "Filter by great-equal pivot.");
  m.def("run_filter_less_eq", &tcuscan::run_filter_less_equal,
        "Filter by less-equal pivot.");
  m.def("run_where", &tcuscan::run_where, "torch.where operator.");
  m.def("run_compress_ind", &tcuscan::run_compress_ind,
        "Compaction that returns indices");
  m.def("run_compress_ind_no_arange", &tcuscan::run_compress_ind_no_arange,
        "Compaction that returns indices (no input indices)");
  m.def("run_compress_pos", &tcuscan::run_compress_pos,
        "Compaction/compress with pre-computed output positions");
  m.def("run_seg_sum", &tcuscan::run_seg_sum, "Segmented Sum");
  m.def("run_seg_sum_single_core", &tcuscan::run_seg_sum_single_core,
        "Segmented Sum (single-core)");
  m.def("run_seg_sum_multi_core", &tcuscan::run_seg_sum_multi_core,
        pybind11::arg("x"), pybind11::arg("indptr"), pybind11::arg("s"),
        pybind11::arg("segm_offsets") = pybind11::none(),
        "Segmented Sum (multi-core)");
  m.def("run_seg_sum_single_cube", &tcuscan::run_seg_sum_single_cube,
        "Segmented Sum (single-cube)");
  m.def("run_spmv", &tcuscan::run_spmv, "Sparse Matrix-Vector Multiplication");
  m.def("run_spmv_v2", &tcuscan::run_spmv_v2,
        "Sparse Matrix-Vector Multiplication Using Segmented Sum");
  m.def("run_spmv_multi_cube", &tcuscan::run_spmv_multi_cube,
        "Sparse Matrix-Vector Multiplication Using Multi-cube Scan");
  m.def("run_copy", &tcuscan::run_copy, "Copy single core");
  m.def("run_scan_batch", &tcuscan::run_scan_batch, "Scan Batch");
  m.def("run_scan_single_core", &tcuscan::run_scan_single_core,
        pybind11::arg("x"), pybind11::arg("S"),
        pybind11::arg("starting_sum") = 0, "Scan Single Core");
  m.def("run_seg_scan_vec", &tcuscan::run_seg_scan_vec,
        "Segmented Scan (vector-only)");
  m.def("run_seg_scan_mc_revert", &tcuscan::run_seg_scan_mc_revert,
        "Vector Revert for MC Segmented Scan");
  m.def("run_topk_int16", &tcuscan::run_topk_int16,
        "TopK using parallel splits (int16)");
  m.def("run_topk_fp16", &tcuscan::run_topk_fp16,
        "TopK using parallel splits (fp16)");
  m.def("run_topk_pivot_fp16", &tcuscan::run_topk_pivot_fp16,
        pybind11::arg("x"), pybind11::arg("k"),
        pybind11::arg("num_samples") = 32, "K-largest value estimator (fp16)");
  m.def("run_split", &tcuscan::run_split, "Split (16-bits)");
  m.def("run_split_ind", &tcuscan::run_split_ind,
        "Split with indices (16-bits)");
  m.def("run_mc_gather", &tcuscan::run_mc_gather, "Vector Multi Core Gather");
  m.def("run_gather_spmv", &tcuscan::run_gather_spmv,
        "Vector Multi Core Gather SPMV");
  m.def("run_radix_sort", &tcuscan::run_radix_sort,
        "Radix sort using cube units");
  m.def("run_matmul_cce", &tcuscan::matmul_cce,
        "Matrix multiplication CCE kernel (B dims must be a multiple of 512)");
  m.def("run_row_scan", &tcuscan::run_row_scan,
        "Matrix multiplication row scan kernel");
  m.def("run_gen_lower", &tcuscan::run_gen_lower,
        "Generate lower triangular matrix");
  m.def("run_reduce_tiles", &tcuscan::run_reduce_tiles,
        "Sum-reduce over tiles");
  m.def("run_cube_reduce", &tcuscan::run_cube_reduce,
        "Block reduction using AIC/AIV cores.");
  m.def("run_complete_rows", &tcuscan::run_complete_rows,
        "Down-sweep (second) phase of MCSCAN");
  m.def("run_complete_blocks", &tcuscan::run_complete_blocks,
        "Block-wise down-sweep (second) phase of block scan");
  m.def("run_block_scan", &tcuscan::run_block_scan,
        "Block scan on blocks of length S^2");
  m.def("run_simple_pad", &tcuscan::run_simple_pad,
        "Padding of an input tensor from length vec_len up to align_len");
  m.def("run_scan_multi_cube", &tcuscan::run_scan_multi_cube,
        "Multi-cube scan");
  m.def("run_scan_cpu", &tcuscan::run_scan_cpu, "Scan on CPUs");
  m.def("run_tri_inv_col_sweep", &tcuscan::run_tri_inv_col_sweep,
        "Unit upper triangular matrix inverses (fp16)");
  m.def("run_tri_inv_cube_col_sweep", &tcuscan::run_tri_inv_cube_col_sweep,
        "Triangular matrix inverse using AIV/AICs(fp16)");
  m.def("run_triu_inv_rec_unroll", &tcuscan::run_triu_inv_rec_unroll,
        "Upper triangular inverse");
  m.def("run_count_if", &tcuscan::run_count_if, "Count if");
  m.def("run_histogram", &tcuscan::run_histogram,
        "Histogram with number of bins.");
}
