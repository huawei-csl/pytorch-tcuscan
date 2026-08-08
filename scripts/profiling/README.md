# Torch Profiling scripts

Benchmarking / profiling scripts for the `tcuscan_ops` operators across Ascend/NPU,
GPU/CUDA and CPU backends. Each script measures per-op latency (with warm-up and
timed iterations) and writes results to a `bench_results_*.csv` file plus a
`torch_profiler*.log` file.

## Scripts

| Script | Backend | Description |
| ------ | ------- | ----------- |
| `profile_tcuscan_ops.py` | Ascend/NPU | Full operator suite (scan, segmented sum/scan, sort, top-k/top-p, gather, SpMV, triangular inverse, histogram, ...). |
| `profile_tcuscan_ops_gpu.py` | GPU/CUDA | GPU/CUDA subset of the operators. |
| `profile_tcuscan_ops_cpu.py` | CPU | CPU reference implementations. |
| `profile_sparse_matrices.py` | Ascend/NPU | SpMV / segmented-scan profiling on real SuiteSparse (`.mtx`) matrices. |
| `profile_random_matrices.py` | Ascend/NPU | SpMV / segmented-scan profiling on randomly generated sparse matrices. |

> The GPU/CPU scripts share most of the logic with the NPU script and could be
> merged into a single file; they are kept separate for convenience.

## Usage

Select the operator to benchmark with `--bench` and the data type with `--dtype`.

```bash
# NPU: benchmark the multi-cube scan in fp16
python profile_tcuscan_ops.py --bench mcscan --dtype fp16

# GPU: benchmark sort in int32 (pick the device/name via env vars)
DEVICE_TYPE=cuda GPU_NAME=A100 python profile_tcuscan_ops_gpu.py --bench sort --dtype int32

# CPU: benchmark scan in fp32
python profile_tcuscan_ops_cpu.py --bench scan --dtype fp32

# NPU: SpMV on a SuiteSparse matrix
python profile_sparse_matrices.py --bench spmv --dtype fp16 --matrixpath /path/to/matrix.mtx

# NPU: SpMV on a random sparse matrix (power-law nnz distribution)
python profile_random_matrices.py --bench spmv --dtype fp16 --density 0.01 --prob PowerLaw
```

Run any script with `--help` to see the full list of `--bench` choices, which
differ per backend.

## Common arguments

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--bench` | — | Operator to benchmark (choices vary per script). |
| `--dtype` | — | `int8`, `int16`, `int32`, `fp16` or `fp32` (support varies per op). |
| `--s` | `64` | Tile/segment size. |
| `--k` | `256` (`4096` for CPU) | `k` parameter for top-k / block ops. |
| `--max_size` | `1e8` | Max input size to sweep up to. |
| `--num_cores` | `20` | Number of cores/cubes to use. |
| `--density` | varies | Sparsity density (sparse/random-matrix scripts). |
| `--matrixpath` | — | Path to a `.mtx` matrix (`profile_sparse_matrices.py`). |
| `--alpha`, `--prob` | `2`, `Uniform` | Random-matrix distribution controls (`profile_random_matrices.py`). |

## Environment variables

| Variable | Default | Used by |
| -------- | ------- | ------- |
| `DEVICE_TYPE` | `npu` (`cuda` for GPU script) | Selects the compute backend. |
| `GPU_NAME` | `A100` | Labels the output CSV for the GPU script. |
