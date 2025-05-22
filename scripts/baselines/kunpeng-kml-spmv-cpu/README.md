## SpMV using Kunpeng KML Sparse BLAS library

Sparse matrix-vector (SpMV) using Kunpeng's KML library as part of [Kunpeng Boostkit](https://www.hikunpeng.com/en/developer/boostkit).

Our SpMV code is based on the [devkitdemo](https://github.com/kunpengcompute/devkitdemo/tree/main/Development_framework/hpc-sdk/examples/kml/spblas).

### Requirements:

1. You must first download the [Sparse Suite matrices](https://sparse.tamu.edu/) used in the test. Update the `SPARSE_SUITE_HOME` in the Makefile accordingly.
2. You must install `boostkit-kml-1.7.0.aarch64.deb` using `dpkg -i boostkit-kml-X.Y.Z.aarch64.deb`, see [GitHub](https://github.com/kunpengcompute/devkitdemo.git)

#### Compile
```bash
make compile
```

#### Run

```bash
make run
```
