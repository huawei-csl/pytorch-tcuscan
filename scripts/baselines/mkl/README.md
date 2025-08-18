# Benchmark for mkl spmv
Benchmark for the spmv routines of intel mkl library, using the `sparse_dot_mkl` python wrappers.

## Tuning performance
`MKL` supports standard directives such as `OMP_NUM_THREADS` and `MKL_NUM_THREADS`.
The specific effect of each environment variable might differ depending on the system, linked libraries, etc.
Before performing the actual performance benchmark try to make some tests and see how they affect the behaviour.

## Environment
The following command will create a conda environment with all the dependencies:
```
conda env create -f environment.yml
```

The `environment.yml` was generated as follows:
```
conda create -n baseline-mkl python=3.12
conda activate baseline-mkl
conda install -c conda-forge sparse_dot_mkl
conda env export > environment.yml
```
i.e., `sparse_dot_mkl` is the only real dependency, which is at version `0.9.8` at the time of this writing.
