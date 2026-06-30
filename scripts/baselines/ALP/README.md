## ALP

This baseline is about evaluating ALP on Kunpeng 920.

### Step 1
Copy the script in the root of ALP repository https://github.com/Algebraic-Programming/ALP

### Step 2

edit _include/graphblas/base/config.hpp_ modifying:
- SIMD::SIZE (https://github.com/Algebraic-Programming/ALP/blob/develop/include/graphblas/base/config.hpp#L123) - this value should represents the number, in bytes, of the SIMD vector registers on the target processor. On Kunpeng920, we do have 128-bit SIMD operations, thus 16 Bytes

- L1 Cache Size (https://github.com/Algebraic-Programming/ALP/blob/develop/include/graphblas/base/config.hpp#L237) - this value is the size of L1 Cache, that is 64KB on the kunpeng 920

### Step 3

Edit the trigger script to add the path where matrices are stored and the list of matrices to use. Notice that, the scripts assumes all the mtx file in a single folder. To run it:

```
./trigger.sh >> spmv_test.log
```

**Note**: OMP environmental variables should be set, for the OMP test - OMP_NUM_THREADS, OMP_PLACES, OMP_PROC_BIND
