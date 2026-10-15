# pytorch-tcuscan

AscendC TCUSCAN operators (scan, sort, split, top-k, compress, etc) using the Cube unit.

## Performance highlights

TCUSCAN expresses prefix-sum-style work as matrix multiplications so that it runs on the
Cube (matrix) units. That unlocks a compute resource these operators normally leave idle
and keeps the Vector units free for the surrounding element-wise work, which turns
several latency-bound PyTorch operators into bandwidth-bound ones.

The numbers below come from this repository's own profiling harness
(`scripts/profiling/profile_tcuscan_ops.py`: 10 warm-up and 100 timed device-side
iterations per point), measured on:

| | |
| --- | --- |
| Accelerator | Ascend 910B2, 24 Cube / 48 Vector cores, 192 MB L2 |
| CANN | 9.1.0 |
| `torch-npu` | 2.10.0 |
| Build | Release, `SOC_VERSION=Ascend910B4` |

Inputs are `fp16` vectors swept up to ~10^8 elements. *Speedup* is the baseline time
divided by the TCUSCAN time at the largest size measured, with the median over the whole
sweep in parentheses. For scale, a device-to-device `copy` of the same tensors peaks at
**916 GB/s** here, which is the practical roofline for the bandwidth column.

| Operator | Baseline | Speedup | Peak bandwidth | Reproduce with |
| --- | --- | --- | --- | --- |
| `run_scan_multi_core` | `torch.cumsum` | **189×** (median 123×) | 415 GB/s | `make profile_mcscan` |
| `run_compress` | `torch.masked_select` | **39–44×** (median 28×) | 404 GB/s | `make profile_compress_paper` |
| `run_histogram`, 32 bins | `torch.histc` | **49×** (median 47×) | 112 GB/s | `make profile_histogram` |
| `run_radix_sort` | `torch.sort` | **8.6×** (median 8.1×) | 26 GB/s | `make profile_fp16_radix_sort` |
| `run_scan_batch` | `torch.cumsum(dim=1)` | **3.3×** (median 2.9×) | 458 GB/s | `make profile_scan_batch` |
| `run_topk_int16`, k=256 | `torch.topk` | **1.6×** | — | `make profile_int16_topk` |

How to read that table:

- **Scan.** `torch.cumsum` does not parallelise a *single* long 1-D scan: it takes 273 ms
  for 10^8 `fp16` elements (2 GB/s) where `run_scan_multi_core` takes 1.4 ms (415 GB/s).
  The 189× therefore says as much about the baseline as about the kernel. The batch-scan
  row is the fairer throughput comparison: with 65 536-wide rows `torch.cumsum(dim=1)`
  *can* parallelise across the batch and reaches 89 GB/s, and TCUSCAN is still 3.3×
  faster at 458 GB/s, i.e. half the copy roofline.
- **Compress** and **histogram** are timed on exactly the same inputs as their baselines
  (the same device-resident mask, the same tensor); only the compaction / binning call is
  measured. Both PyTorch baselines are slow for this access pattern (7–10 GB/s and
  2.3 GB/s). The histogram comparison is conservative: `run_histogram` derives its bin
  edges from the data, while `torch.histc` is handed explicit bounds and skips that pass.
- **Radix sort** is multi-pass, so its *effective* bandwidth is low by construction and
  the speedup is the meaningful number. It leads `torch.sort` across the entire sweep,
  from 1.1× at 3·10^5 elements to 8–10× from 10^6 upwards.
- **Top-k** is where the Cube-based approach does not pay off for `fp16`:
  `run_topk_fp16` is **0.54×** of `torch.topk` at k=2048, matching the negative result
  reported in the paper for k ≤ 4096. The `int16` path is ahead at 1.6× (71.7 ms versus
  115.9 ms at 10^8), but note that `torch.topk` is itself ~35× slower on `int16` than on
  `fp16`, so part of that margin is a weak baseline rather than a fast kernel.

Small inputs are not the target: every kernel above carries a fixed launch and
synchronisation cost, so `run_scan_batch` only overtakes its baseline above ~10^7
elements, and the other operators reach their quoted ratios from ~10^6 elements upwards.

### Sparse matrix–vector multiplication

`run_spmv` and `run_spmv_v2` build SpMV on the segmented sum, and `run_spmv_multi_cube`
on the multi-cube scan. Measured over the 28 SuiteSparse matrices of the ALENEX
evaluation set (`make profile_fp16_spmv_alenex`, `fp16`, `S=128`), effective bandwidth
tracks the number of non-zeros:

| Matrix | rows | nnz | `run_spmv` | `run_spmv_v2` | `run_spmv_multi_cube` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `enron` | 69 244 | 0.28 M | 3.6 GB/s | 2.7 GB/s | 4.0 GB/s |
| `water_tank` | 60 740 | 2.0 M | 19.6 GB/s | 22.9 GB/s | 19.8 GB/s |
| `cant` | 62 451 | 4.0 M | 35.4 GB/s | 46.3 GB/s | 35.6 GB/s |
| `pdb1HYS` | 36 417 | 4.3 M | 38.5 GB/s | 52.5 GB/s | 38.9 GB/s |
| `crankseg_1` | 52 804 | 10.6 M | 77.4 GB/s | 88.2 GB/s | 80.5 GB/s |
| `crankseg_2` | 63 838 | 14.1 M | 96.8 GB/s | 92.0 GB/s | **98.5 GB/s** |

Across the full set the median is ~32 GB/s and the peak 98.5 GB/s. Matrices below roughly
10^6 non-zeros are dominated by launch and gather overhead rather than by bandwidth, so
the three variants are within noise of each other there. Above that, the segmented-sum
variant (`run_spmv_v2`) is fastest on 13 of the 21 larger matrices, while
`run_spmv_multi_cube` takes the largest one in the set.

### Reference

The algorithms behind the scan, sort, compaction and sampling kernels are described in:

> Bartłomiej Wróblewski, Gioele Gottardo, Anastasios Zouzias.
> *Parallel Scan on Ascend AI Accelerators*. arXiv:2505.15112, 2025.
> <https://arxiv.org/abs/2505.15112>

The notebooks under `nbs/paper-plots/` regenerate the figures of that paper from the CSV
files these `make profile_*` targets produce. Absolute numbers there were measured on an
Ascend 910B4 and differ from the table above.

## Support matrix

| Component | Supported |
| --- | --- |
| Hardware | Atlas A2 / A3 training series (`SOC_VERSION` defaults to `Ascend910B4`) |
| CANN toolkit | 9.0.0 |
| `torch-npu` | 2.10.0 |
| Python | 3.10 – 3.12 |
| OS | Linux (aarch64, x86_64) |

Other SoCs, CANN releases, and `torch-npu` versions are untested and may fail to build or
produce incorrect results. Select a different chip at configure time with
`SOC_VERSION=<soc> pip install -v .`, and point at a non-default CANN install with
`ASCEND_CANN_PACKAGE_PATH` (default `/usr/local/Ascend/ascend-toolkit/latest`).

## TCUSCAN kernels

All kernels are exported from the `tcuscan_ops` module (see `src/pybind11.cpp`).
Most kernels operate on 1D tensors and take a tiling parameter `S` (typical values: 32, 64, 128).

The **input dtypes** column lists the dtypes of the *value* tensor that the wrapper in
`src/torch/` actually dispatches on. Auxiliary tensors follow fixed conventions: boolean
masks and segment flags are `int8`, and index tensors (`cols`, `indptr`, `indices`) are `int32`.

Reducing kernels promote their accumulator: `fp16 -> fp32` and integer inputs `int8 -> int32`.

### Scan (prefix sum)

| Operator | Input dtypes | Description |
| --- | --- | --- |
| `run_scan_single_core` | `fp16`, `fp32`, `int8` | Prefix sum of a 1D vector on a single AI Core |
| `run_scan_multi_core` | `fp16`, `fp32`, `int8` | Multi-core prefix sum |
| `run_scan_multi_core_no_l2` | `fp16`, `fp32`, `int8` | Multi-core prefix sum without the L2 cache splitting optimization |
| `run_scan_multi_cube` | `fp16` | Multi-cube prefix sum, built on top of the block scan |
| `run_scan_batch` | `fp16`, `fp32` | Row-wise prefix sum of a 2D matrix (one cube core per row) |
| `run_row_scan` | `fp16` | Prefix sum of each consecutive block of length `S` |
| `run_block_scan` | `fp16` | Prefix sum of each consecutive block of length `S^2` |
| `run_scan_cpu` | `fp16`, `fp32`, `int8` | CPU reference implementation |

`run_scan_multi_core` and `run_scan_multi_core_no_l2` accept `fp32` input and return `fp32`
output (no accumulator promotion). The `fp32` path uses dedicated
`scan_multi_core_fp32` / `scan_multi_core_fp32_no_l2` kernels.

Both wrappers reject any other dtype with a `TORCH_CHECK` error rather than silently
falling through to the `int8` kernel.

### Segmented scan / segmented sum

| Operator | Input dtypes | Description |
| --- | --- | --- |
| `run_seg_scan` | `fp16` | Segmented scan driven by a boolean flag vector |
| `run_seg_scan_vec` | `fp16` | Segmented scan, vector (AIV) cores only |
| `run_seg_scan_mc_revert` | `fp32` | Vector revert phase of the multi-core segmented scan |
| `run_seg_sum` | `fp16` | Segmented sum driven by a boolean flag vector |
| `run_seg_sum_single_core` | `fp16`, `int8` | Segmented sum from a CSR-style `indptr` (single core) |
| `run_seg_sum_multi_core` | `fp16`, `int8` | Segmented sum from `indptr` across vector cores |
| `run_seg_sum_single_cube` | `fp16` | Segmented sum on a single cube unit |
| `run_seg_sum_multi_cube` | `fp16` | Segmented sum across cube units |


### Sparse matrix–vector multiplication (SpMV)

| Operator | Input dtypes | Description |
| --- | --- | --- |
| `run_spmv` | `fp16`, `int16` | CSR SpMV |
| `run_spmv_v2` | `fp16`, `fp32` | CSR SpMV built on the segmented sum |
| `run_spmv_multi_cube` | `fp16` | CSR SpMV built on the multi-cube scan |
| `run_csr_gather` | `fp16`, `int16`, `fp32` | `z[i] = values[i] * x[cols[i]]`, the gather+scale phase of SpMV |
| `run_gather_spmv` | `fp32` | Specialized multi-core gather for SpMV (fixed tiling of 128) |
| `run_mc_gather` | `fp16`, `fp32` | General multi-core gather of a 1D vector |

For `run_spmv_v2`, `vals` and `x` must have the *same* dtype, and `indptr` must be
`int32`/`uint32`.

All four SpMV entry points (`run_spmv`, `run_spmv_v2`, `run_spmv_multi_cube`,
`run_spmv_v2_multi_cube`) take the BLAS-style scalars `alpha`, `beta` and an optional
output vector `y`, computing

```python
y = tcuscan_ops.run_spmv_v2(vals, indptr, cols, x, s, alpha=2.0, beta=1.0, y=y)  # y = 2*A@x + y
```

### Sort, top-k, and split

| Operator | Input dtypes | Description |
| --- | --- | --- |
| `run_radix_sort` | `fp16`, `int16` | Radix sort of a 1D vector, returns values and `int32` indices |
| `run_topk_fp16` / `run_topk_int16` | `fp16` / `int16` | Top-K elements via parallel splits |
| `run_topk_pivot_fp16` | `fp16` | K-largest value estimator by sampling (`num_samples`, default 32) |
| `run_split` | `fp16`, `int16` | Binary split of a vector given a boolean mask |
| `run_split_ind` | `fp16`, `int16` | Binary split returning both values and indices |

`run_split`/`run_split_ind` dispatch to a single 16-bit kernel (`split_uint16`), so `fp16` and
`int16` share the same code path.

### Compaction, filtering, and counting

| Operator | Input dtypes | Description |
| --- | --- | --- |
| `run_compress` | `fp16`, `int16`, `fp32` | Stream compaction given a boolean mask |
| `run_compress_pos` | `fp16`, `int16`, `fp32` | Compaction with pre-computed output positions |
| `run_compress_ind` | `fp16`, `int16`, `fp32` | Compaction returning the surviving indices |
| `run_compress_ind_no_arange` | `fp16`, `int16`, `fp32` | Same, without materializing the input index vector |
| `run_filter_greater_eq` | `fp16` | Filter by `x_i >= pivot` |
| `run_filter_less_eq` | `fp16`, `int16`, `fp32` | Filter by `x_i <= pivot` |
| `run_where` | `fp16` | `torch.where`-style selection |
| `run_count_if` | `fp16` | Count elements satisfying a pivot predicate |

The `run_compress*` kernels dispatch on element *width*: `fp16`/`int16` use the 16-bit kernel,
everything else falls through to the 32-bit (`*_fp32`) kernel. `run_filter_greater_eq` is
`fp16`-only because it builds its mask with the `fp16`-only `greater_equal` kernel, whereas
`run_filter_less_eq` builds the mask in ATen and so inherits the wider `run_compress` support.

### Reductions and histogram

| Operator | Input dtypes | Description |
| --- | --- | --- |
| `run_reduce_tiles` | `fp16`, `int8` | Sum-reduction over each tile of a 1D vector |
| `run_cube_reduce` | `fp16`, `int8` | Block reduction using both AIC and AIV cores |
| `run_complete_rows` | `fp32`, `int64` | Down-sweep (second) phase of MCSCAN |
| `run_complete_blocks` | `fp32`, `int32` | Block-wise down-sweep (second) phase of the block scan |
| `run_histogram` | `fp16` | Histogram over a given number of bins (returns `int32`) |

`run_complete_rows` keys on `int64` (`torch::kLong`) while dispatching the `int32` kernel;
`run_complete_blocks` keys on `int32`. The two are not interchangeable.

### Linear algebra

| Operator | Input dtypes | Description |
| --- | --- | --- |
| `run_gen_lower` | `fp16`, `int8` | Generate a lower triangular all-ones matrix |
| `run_tri_inv_col_sweep` | `fp16`, `fp32` | Unit upper triangular matrix inverse, column sweep |
| `run_tri_inv_cube_col_sweep` | `fp16` | Triangular matrix inverse using AIV/AIC cores (returns `fp32`) |
| `run_triu_inv_rec_unroll` | `fp16` | Upper triangular inverse, recursive unrolled (returns `fp32`) |

`run_gen_lower` takes the dtype as an explicit argument rather than inferring it from an input.

### Elementwise and utility

| Operator | Input dtypes | Description |
| --- | --- | --- |
| `run_add` | `fp16` | Vector add |
| `run_diff` | `fp16`, `fp32` | Vector first difference |
| `run_copy` | `fp16`, `fp32` | Memory copy on a single AI Core |
| `run_simple_pad` | `fp16` | Pad a tensor from `vec_len` up to `align_len` |

`run_add`, `run_simple_pad` and `run_matmul_cce` launch a single kernel with no dtype
dispatch at all, so they assume `fp16` input.

## Getting started

```bash
export CMAKE_GENERATOR="Unix Makefiles"
pip install -v https://github.com/huawei-csl/pytorch-tcuscan.git --extra-index-url https://download.pytorch.org/whl/cpu
```

Or clone the repository and run:
```bash
export CMAKE_GENERATOR="Unix Makefiles"
pip install -v . --extra-index-url https://download.pytorch.org/whl/cpu
```

Then, inside Python, type `import tcuscan`.

# Development
Integrating TCUSCAN kernels to pytorch npu.

## Build, test, and install
Currently there is no explicit "pythonic" installation of the package.
All the artifacts are created using:
```bash
make build
```
This will create two shared objects: `build/lib/libkernels.so` and `tcuscan_ops.*.so`.
Unit tests can be executed with
```bash
make test
```
### Debug builds

`build.sh` configures `-DCMAKE_BUILD_TYPE=Release` (`-O3 -DNDEBUG`). For an unoptimised
host build with `-g` and active `assert()`s, pass `Debug` instead:

```bash
source "${ASCEND_HOME_PATH}"/bin/setenv.bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:$(python -c 'import torch, pybind11; print(torch.utils.cmake_prefix_path + ":" + pybind11.get_cmake_dir())')"

rm -rf build
cmake -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DSOC_VERSION=Ascend910B4 \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}"
cmake --build build -j
```

`CMAKE_BUILD_TYPE` is cached, so always reconfigure from a clean `build/` when switching.
This affects host-side C++ only; for kernel debugging use the standalone `Makefile`
targets and on-device printing.

The artifacts can currently be installed inside a local python environment (e.g. conda) by manually moving them to the appropriate folders.
This can be done with:
```bash
make install_in_local_conda_env
```
which will manually create a new python environment, build the package, and move the shared objects inside the environment.
After this, the package can be imported as follows:
```python
import torch
import torch_npu
import tcuscan_ops
```

## Integrating a new kernel
To integrate a new kernel you need three main steps.

### 1: Define your kernel API

Let's say the kernel is `${MYKERNEL}` with the following files:

```bash
pytorch-tcuscan/src/kernels/kernel_${MYKERNEL}.h
pytorch-tcuscan/src/${MYKERNEL}.cpp
pytorch-tcuscan/src/tiling/tiling_${MYKERNEL}.h
```
The `${MYKERNEL}.cpp` file needs to be also added inside `CMakeLists.txt`

### 2: Implement entrypoint in pybind
In the `src/pybind11.cpp` file we need to implement a wrapper function to call the kernel.
This function lies inside the `asc` namespace, and added inside the `PYBIND11_MODULE`
```cpp
namespace asc{
    // ...
  at::Tensor run_my_kernel(const at::Tensor &x, int S) {
    //...
  }
} // namespace asc
PYBIND11_MODULE(tcuscan_ops, m) {
  m.doc() = "TCUSCAN AscendC operators";
  //...
  m.def("run_my_kernel", &asc::run_my_kernel, "MYKERNEL execution");
}
```

### 3: Implement tests and add them in CI
The final step is to build a test file `tests/test_${MYKERNEL}.py`, and add a job in `.gitlab-ci.yml` to run the tests in CI.

### 4: Important notes / nitpicks
- In `pybind11.cpp`, every `run_*` function must allocate a workspace tensor, even if it is an empty tensor. This tensor must be passed as an argument to the kernel.
- In `test/test_*.py`, we always need to import `torch_npu`, which might be unused. Make sure to add a comment `import torch_npu # noqa` so that the prospector passes the CI
- When the input is a 2d-tensor, e.g., as in `scan_batch`, we usually set `block_size` (i.e. the number of cube cores) equal to the batch size.
- All the files related to a kernel (test, tiling, header, cpp,...) must all have the same name. E.g., if the kernel name is `scan_fp16`, we need to name the files `tiling_scan_fp16.h`, `kernel_scan_fp16.h`, `scan_fp16.cpp`, `test_scan_fp16.py`, etc.


## Contributors

Below is a list of contributors (alphabetical order by surname):
- Gioele Gottardo
- Aleksandros Sobczyk
- Giuseppe Sorrentino
- Bartłomiej Wróblewski
- Anastasios Zouzias
