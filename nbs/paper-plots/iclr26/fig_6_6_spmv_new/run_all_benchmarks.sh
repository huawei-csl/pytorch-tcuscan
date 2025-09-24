#!/bin/bash
THIS_DIRECTORY="$(pwd)"

pushd ../../../../ || exit
# make profile_fp16_spmv_real_multi_cube_for_iclr26

pushd scripts/baselines/eigen-spmv-cpu || exit
./execute_tests_iclr26.sh
popd || exit

pushd scripts/baselines/mkl || exit
./execute_tests_iclr26.sh
popd || exit


# Come back to current directory
popd || exit


DATETIME="$(date '+%Y-%m-%d_%H:%M')"
BENCHMARKS_DIR="${THIS_DIRECTORY}/benchmarks_${DATETIME}"
echo "************************************************************************"
echo "* Running benchmarks on ${BENCHMARKS_DIR}} with timestamp ${DATETIME}"
echo "************************************************************************"

mkdir -p "${BENCHMARKS_DIR}"
mv ../../../../bench_*.csv "${BENCHMARKS_DIR}"
mv ../../../../scripts/baselines/eigen-spmv-cpu/bench_*.csv "${BENCHMARKS_DIR}"
mv ../../../../scripts/baselines/mkl/bench_*.csv "${BENCHMARKS_DIR}"
