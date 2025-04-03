## How To Run CU-Sparse

### Step 1: Download the target sparse suite matrices

The provided scripts automatically evaluate with cusparse **all** the downloaded matrices. Thus install the needed ones using the dedicated code:

see - https://rnd-gitlab-eu.huawei.com/a00565472/pytorch-tcuscan/-/merge_requests/120

### Step 2: Enable NVCC

NVCC must be enable by doing the proper exports. For simplicity:

```bash
source ./exports.sh
```

### Step 3: Compile and run

For this step, you can just use the dedicated automation


```bash
make compile_and_run
```

This will run with a default metrix specified in the Makefile

### Step 4: Evaluate Cusparse

To run cusparse on **all** the downloaded matrices, run the following command:

```bash
./automatic_test.sh
```
