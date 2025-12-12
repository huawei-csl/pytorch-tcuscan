.PHONY: all clean setup_ci setup_once setup_once_aarch64 build test profile

DENSITY?=0.001
LOCAL_SPARSE_MATRIX_NAME?='Williams/pdb1HYS/pdb1HYS'
BASE_SPARSE_MATRIX_PATH?=/scratch/TCUSCAN/sparse-suite-matrices/ssgetpy-downloaded-matrices
FULL_SPARSE_MATRIX_PATH=${BASE_SPARSE_MATRIX_PATH}/${LOCAL_SPARSE_MATRIX_NAME}
PROFILING_SCRIPTS_PATH=scripts/profiling/
CONDA_ENV_NAME="pytorch_tcuscan"

TORCH_NPU_URL=https://gitcode.com/Ascend/pytorch/releases/download
PT_WHEEL_NAME=torch_npu-2.6.0.post2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
PT_WHEEL_URL=${TORCH_NPU_URL}/v7.1.0.2-pytorch2.6.0/${PT_WHEEL_NAME}


PT_WHEEL_AARCH_NAME=torch_npu-2.5.1.post3-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
PT_WHEEL_AARCH_URL=${TORCH_NPU_URL}/v7.1.0.2-pytorch2.5.1/${PT_WHEEL_AARCH_NAME}


ASCEND_DEVICE=Ascend910B4
DEVICE_TYPE?=npu

all: build test

clean:
	rm -rf build/ tests/__pycache__/
	rm -f *.so

setup_ci:
	pip3 install --cache-dir=/scratch/TCUSCAN/wheels/ -r requirements.txt

install_in_local_conda_env: create_conda_env
	conda run -n ${CONDA_ENV_NAME} ./build.sh -v ${ASCEND_DEVICE}
	mv tcuscan_ops.cpython-*.so $(shell conda env list | grep ${CONDA_ENV_NAME} | awk '{print $$2}')/lib/python3.10/site-packages/
	mv build/lib/libkernels.so $(shell conda env list | grep ${CONDA_ENV_NAME} | awk '{print $$2}')/lib/
	rm -rf build/

create_conda_env:
	conda create -y -n ${CONDA_ENV_NAME} python=3.10
	# PyTorch Ascend requires cmake >= 3.18
	conda install -y cmake -n ${CONDA_ENV_NAME}
	conda run -n ${CONDA_ENV_NAME} pip3 install -r requirements.txt
	wget -nc ${PT_WHEEL_URL}
	conda run -n ${CONDA_ENV_NAME} pip3 install ${PT_WHEEL_NAME} --index-url https://download.pytorch.org/whl/cpu

setup_once:
	pip3 install -r requirements.txt
	wget -nc ${PT_WHEEL_URL}
	pip3 install --force-reinstall ${PT_WHEEL_NAME} --index-url https://download.pytorch.org/whl/cpu

# For 910B2 experiments, you need to update the L2_SIZE (constexpr) and SOC_VERSION (const static char*) in the code
setup_once_aarch64:
	pip3 install -r requirements.txt
	wget -nc ${PT_WHEEL_AARCH_URL}
	pip3 install --force-reinstall  ${PT_WHEEL_AARCH_NAME}

clang_tidy: build_tidy
	python3 ./scripts/ci/run-clang-tidy.py -j 1 -p build/ src/

build: build.sh
	./build.sh -v ${ASCEND_DEVICE}

pypackage:
	./build-pypackage.sh -v ${ASCEND_DEVICE}

build_tidy: build-tidy.sh
	./build-tidy.sh -v ${ASCEND_DEVICE}

docs:
	doxygen doxygen/Doxyfile

test:
	python3 -m pytest tests/

test_%:
	python3 -m pytest tests/test_$*.py -v

###
# Profiling kernels
###

profile_%: profile_fp32_$* profile_fp16_$* profile_int16_$*

profile_int8_%:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench $* --dtype int8

profile_int16_%:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench $* --dtype int16

profile_int32_%:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench $* --dtype int32

profile_fp16_%:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench $* --dtype fp16

profile_fp32_%:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench $* --dtype fp32

profile_all_s_fp16_%:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench $* --s 32 --dtype fp16 --iter-step-multiplier 4
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench $* --s 64 --dtype fp16 --iter-step-multiplier 2
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench $* --s 128 --dtype fp16


profile_cpu_fp32_%:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops_cpu.py --bench $* --dtype fp32

profile_cpu_fp16_%:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops_cpu.py --bench $* --dtype fp16

profile_cpu: profile_cpu_fp16_copy profile_cpu_fp16_scan profile_cpu_fp16_sort profile_cpu_fp16_masked_select profile_cpu_fp16_topk

profile_cuda:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops_gpu.py --bench copy --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops_gpu.py --bench scan --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops_gpu.py --bench sort --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops_gpu.py --bench masked_select --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops_gpu.py --bench topp --dtype fp16

profile_baselines:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench copy --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench scan --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench sort --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench masked_select --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench topp --dtype fp16

profile_mcscan_int8:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench copy --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan --s 128 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan --s 128 --dtype int8


profile_mcscan: profile_all_s_fp16_mcscan
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench copy --dtype fp16

profile_mcscan_no_l2: profile_mcscan
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan_no_l2 --s 32 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan_no_l2 --s 64 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan_no_l2 --s 128 --dtype fp16

profile_row_scan: profile_all_s_fp16_row_scan
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench cast --dtype fp16

profile_block_scan: profile_all_s_fp16_block_scan
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench cast --dtype fp16

profile_down_sweep:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench copy --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench complete_rows --s 128 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench complete_blocks --s 128 --k 2 --dtype fp32

profile_complete_blocks:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench copy --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench complete_blocks --s 128 --k 4 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench complete_blocks --s 128 --k 8 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench complete_blocks --s 128 --k 16 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench complete_blocks --s 128 --k 32 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench complete_blocks --s 128 --k 64 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench complete_blocks --s 128 --k 128 --dtype fp32

profile_scan_multi_cube:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench copy --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench cast --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan --s 128 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench scan_multi_cube --s 128 --dtype fp16

profile_scscan: profile_all_s_fp16_scscan

profile_compress_paper:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench compress --s 128 --density 0.1 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench compress --s 128 --density 0.1 --dtype fp32

profile_scan_batch:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench scan_batch --dtype fp16 --min-iter-index 16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench scan_batch_tcuscan --s 16 --dtype fp16 --iter-step-multiplier 8
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench scan_batch_tcuscan --s 32 --dtype fp16 --iter-step-multiplier 4
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench scan_batch_tcuscan --s 64 --dtype fp16 --iter-step-multiplier 2
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench scan_batch_tcuscan --s 128 --dtype fp16

profile_fp16_compress:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench compress --s 32  --density ${DENSITY} --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench compress --s 64  --density ${DENSITY} --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench compress --s 128 --density ${DENSITY} --dtype fp16

profile_fp32_compress:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench compress --s 32  --density ${DENSITY} --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench compress --s 64  --density ${DENSITY} --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench compress --s 128 --density ${DENSITY} --dtype fp32

profile_fp16_seg_scan_sc:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_sc --dtype fp16 --s 32 --density ${DENSITY} --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_sc --dtype fp16 --s 64 --density ${DENSITY} --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_sc --dtype fp16 --s 128 --density ${DENSITY} --num_cores 1

profile_fp16_segmented_sum:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench segmented_sum --density ${DENSITY} --s 32 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench segmented_sum --density ${DENSITY} --s 64 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench segmented_sum --density ${DENSITY} --s 128 --dtype fp16

profile_fp16_sc_segmented_sum:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench sc_segmented_sum --density ${DENSITY} --s 32 --dtype fp16 --num_cores 1 --max_size 1000000
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench sc_segmented_sum --density ${DENSITY} --s 64 --dtype fp16 --num_cores 1 --max_size 2000000
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench sc_segmented_sum --density ${DENSITY} --s 128 --dtype fp16 --num_cores 1 --max_size 4000000

profile_fp16_cube_segmented_sum:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench cube_segmented_sum --density ${DENSITY} --s 32 --dtype fp16 --num_cores 1 --max_size 1000000
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench cube_segmented_sum --density ${DENSITY} --s 64 --dtype fp16 --num_cores 1 --max_size 2000000
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench cube_segmented_sum --density ${DENSITY} --s 128 --dtype fp16 --num_cores 1 --max_size 4000000

profile_fp32_custom_copy:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench custom_copy --dtype fp32 --num_cores 1

profile_fp16_custom_copy:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench custom_copy --dtype fp16 --num_cores 1

profile_fp16_vec_seg_scan:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench vec_seg_scan_sc --dtype fp16 --density ${DENSITY}  --s 32  --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench vec_seg_scan_sc --dtype fp16 --density ${DENSITY}  --s 64  --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench vec_seg_scan_sc --dtype fp16 --density ${DENSITY}  --s 128 --num_cores 1

profile_diffs: profile_fp16_diff profile_fp16_diff_cann profile_fp16_diffp_cann profile_fp32_diff_cann

profile_fp16_segmented_scan_real:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${FULL_SPARSE_MATRIX_PATH} --s 32 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${FULL_SPARSE_MATRIX_PATH} --s 64 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${FULL_SPARSE_MATRIX_PATH} --s 128 --num_cores 1

profile_fp16_vec_segmented_scan_real:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${FULL_SPARSE_MATRIX_PATH} --s 32 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${FULL_SPARSE_MATRIX_PATH} --s 64 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${FULL_SPARSE_MATRIX_PATH} --s 128 --num_cores 1

profile_fp16_compress_real:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench compress --s 32  --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench compress --s 64  --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench compress --s 128 --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp16

profile_fp32_compress_real:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench compress --s 32  --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench compress --s 64  --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench compress --s 128 --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp32

profile_fp16_diff_real:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench diff --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp16

profile_fp32_revert_mcscan:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_mc_revert --dtype fp32 --num_cores 20 --s 128 --density ${DENSITY}

profile_radix_sort: profile_fp16_radix_sort profile_int16_radix_sort

profile_fp16_radix_sort:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench sort --dtype fp16 --s 128 --num_cores 20
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench radix_sort --dtype fp16 --s 32 --num_cores 20 --iter-step-multiplier 4
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench radix_sort --dtype fp16 --s 64 --num_cores 20 --iter-step-multiplier 2
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench radix_sort --dtype fp16 --s 128 --num_cores 20

profile_int16_radix_sort:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench sort --dtype int16 --s 128 --num_cores 20
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench radix_sort --dtype int16 --s 32 --num_cores 20 --iter-step-multiplier 4
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench radix_sort --dtype int16 --s 64 --num_cores 20 --iter-step-multiplier 2
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench radix_sort --dtype int16 --s 128 --num_cores 20

profile_int16_topk:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench topk --dtype int16 --s 128 --num_cores 20
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tcuscan_topk --dtype int16 --s 32 --num_cores 20
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tcuscan_topk --dtype int16 --s 64 --num_cores 20
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tcuscan_topk --dtype int16 --s 128 --num_cores 20

profile_fp16_topk:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench topk --dtype fp16 --s 128 --num_cores 20 --k 2048
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tcuscan_topk --dtype fp16 --s 32 --num_cores 20 --iter-step-multiplier 4 --k 2048
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tcuscan_topk --dtype fp16 --s 64 --num_cores 20 --iter-step-multiplier 2 --k 2048
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tcuscan_topk --dtype fp16 --s 128 --num_cores 20 --k 2048

profile_topp:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench topp --dtype fp16 --s 128 --num_cores 20 --max_size 16000000 --iter-step-divider 8
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tcuscan_topp --dtype fp16 --s 32 --num_cores 20 --max_size 16000000 --iter-step-divider 2
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tcuscan_topp --dtype fp16 --s 64 --num_cores 20 --max_size 16000000 --iter-step-divider 4
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tcuscan_topp --dtype fp16 --s 128 --num_cores 20 --max_size 16000000 --iter-step-divider 8

profile_cube_reduce:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench reduce_tiles --s 128 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench cube_reduce --s 128 --dtype fp16

include Makefile.spaa.mk Makefile.spmv.mk Makefile.tri_inv.mk
