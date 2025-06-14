/**
 * @file pybind11.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "torch/torch_compress.h"
#include "torch/torch_copy.h"
#include "torch/torch_diff.h"
#include "torch/torch_gather.h"
#include "torch/torch_gen_lower.h"
#include "torch/torch_matmul_cce.h"
#include "torch/torch_pad.h"
#include "torch/torch_reduce.h"
#include "torch/torch_scan.h"
#include "torch/torch_seg_ops.h"
#include "torch/torch_sort.h"
#include "torch/torch_split.h"
#include "torch/torch_spmv.h"
#include "torch/torch_vadd.h"

/**
 * @brief Pybind11 module.
 */
PYBIND11_MODULE(tcuscan_ops, m) {
  m.doc() = "TCUSCAN AscendC operators";
  m.def("run_add", &asc::add::run_add, "Vector add");
  m.def("run_diff", &asc::diff::run_diff, pybind11::arg(),
        pybind11::arg("max_size") = 0, "Vector diff");
  m.def("run_seg_scan", &asc::seg_ops::run_seg_scan, "Segmented Scan");
  m.def("run_scan_multi_core", &asc::scan::run_scan_multi_core,
        "Multi-core Scan");
  m.def("run_scan_multi_core_no_l2", &asc::scan::run_scan_multi_core_no_l2,
        "Multi-core Scan (no L2 cache optimization)");
  m.def("run_csr_gather", &asc::gather::run_csr_gather, "CSR gather");
  m.def("run_compress", &asc::compress::run_compress, "Compaction/compress");
  m.def("run_compress_pos", &asc::compress::run_compress_pos,
        "Compaction/compress with pre-computed output positions");
  m.def("run_seg_sum", &asc::seg_ops::run_seg_sum, "Segmented Sum");
  m.def("run_spmv", &asc::spmv::run_spmv,
        "Sparse Matrix-Vector Multiplication");
  m.def("run_copy", &asc::copy::run_copy, "Copy single core");
  m.def("run_scan_batch", &asc::scan::run_scan_batch, "Scan Batch");
  m.def("run_scan_single_core", &asc::scan::run_scan_single_core,
        pybind11::arg("x"), pybind11::arg("S"),
        pybind11::arg("starting_sum") = 0, "Scan Single Core");
  m.def("run_seg_scan_vec", &asc::seg_ops::run_seg_scan_vec,
        "Segmented Scan (vector-only)");
  m.def("run_seg_scan_mc_revert", &asc::seg_ops::run_seg_scan_mc_revert,
        "Vector Revert for MC Segmented Scan");
  m.def("run_topk_int16", &asc::split::run_topk_int16,
        "TopK using parallel splits (int16)");
  m.def("run_topk_fp16", &asc::split::run_topk_fp16,
        "TopK using parallel splits (fp16)");
  m.def("run_split", &asc::split::run_split, "Split (16-bits)");
  m.def("run_split_ind", &asc::split::run_split_ind,
        "Split with indices (16-bits)");
  m.def("run_mc_gather", &asc::gather::run_mc_gather,
        "Vector Multi Core Gather");
  m.def("run_gather_spmv", &asc::gather::run_gather_spmv,
        "Vector Multi Core Gather SPMV");
  m.def("run_radix_sort", &asc::sort::run_radix_sort,
        "Radix sort using cube units");
  m.def("run_matmul_cce", &asc::matmul::matmul_cce,
        "Matrix multiplication CCE kernel (B dimensions must be a multiple of "
        "512)");
  m.def("run_row_scan", &asc::scan::run_row_scan,
        "Matrix multiplication row scan kernel");
  m.def("run_gen_lower", &asc::gen::run_gen_lower,
        "Generate lower triangular matrix");
  m.def("run_reduce_tiles", &asc::reduce::run_reduce_tiles,
        "Sum-reduce over tiles");
  m.def("run_complete_rows", &asc::reduce::run_complete_rows,
        "Down-sweep (second) phase of MCSCAN");
  m.def("run_complete_blocks", &asc::reduce::run_complete_blocks,
        "Block-wise down-sweep (second) phase of block scan");
  m.def("run_block_scan", &asc::scan::run_block_scan,
        "Block scan on blocks of length S^2");
  m.def("run_simple_pad", &asc::pad::run_simple_pad,
        "Padding of an input tensor from length vec_len up to align_len");
}
