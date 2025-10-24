# pytorch-tcuscan

AscendC TCUSCAN operators (scan, sort, split, top-k, compress, etc) using the Cube unit.


## Getting started

```bash
export CMAKE_GENERATOR="Unix Makefiles"
pip install -v ssh://git@szv-open.codehub.huawei.com:2222/innersource/tcuscan_G/pytorch-tcuscan.git
```

Or clone the repository and run:
```bash
export CMAKE_GENERATOR="Unix Makefiles"
pip install -v
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
To integrate a new kernel from the `ascend-scan-910b` repo we need three main steps.

### 1: Copy kernel header files
The first step is to copy the kernel header files from the `ascend-scan-910b` repo.
Let's say the kernel is `${MYKERNEL}`
We need to copy:
```bash
cp ascendc-scan-910b/src/kernels/kernel_${MYKERNEL}.h pytorch-tcuscan/src/kernels/kernel_${MYKERNEL}.h
cp ascendc-scan-910b/src/kernels/kernel_${MYKERNEL}.cpp pytorch-tcuscan/src/${MYKERNEL}.cpp
cp ascendc-scan-910b/src/kernels/tiling_${MYKERNEL}.h pytorch-tcuscan/src/tiling/tiling_${MYKERNEL}.h
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
