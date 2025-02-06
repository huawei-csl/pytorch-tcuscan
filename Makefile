.PHONY: all clean setup_ci setup_once setup_once_aarch64 build test profile

DENSITY?=0.05
LOCAL_SPARSE_MATRIX_NAME?='Boeing/bcsstk35/bcsstk35'
BASE_SPARSE_MATRIX_PATH?=${HOME}/.ssgetpy/MM/
FULL_SPARSE_MATRIX_PATH=${BASE_SPARSE_MATRIX_PATH}${LOCAL_SPARSE_MATRIX_NAME}

all: build test

clean:
	rm -rf build/ tests/__pycache__/
	rm -f *.so

setup_ci:
	pip3 install --cache-dir=/scratch/TCUSCAN/wheels/ -r requirements.txt

setup_once:
	pip3 install -r requirements.txt
	wget https://gitee.com/ascend/pytorch/releases/download/v6.0.0-pytorch2.4.0/torch_npu-2.4.0.post2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
	pip3 install torch_npu-2.4.0.post2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --index-url https://download.pytorch.org/whl/cpu

setup_once_aarch64:
	pip3 install -r requirements.txt
	wget https://gitee.com/ascend/pytorch/releases/download/v6.0.0-pytorch2.4.0/torch_npu-2.4.0.post2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
	pip3 install torch_npu-2.4.0.post2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --index-url https://download.pytorch.org/whl/cpu

build: build.sh src/vadd.cpp src/diff.cpp src/seg_scan_single_core.cpp src/pybind11.cpp
	./build.sh -v ASCEND910B4

test:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/  python3 -m pytest tests/

test_%:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/  python3 -m pytest tests/test_$*.py

profile_%: profile_fp32_$* profile_fp16_$* profile_int16_$*

profile_fp16_%:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench $* --dtype fp16

profile_fp32_%:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench $* --dtype fp32

profile_int16_%:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench $* --dtype int16

profile_mcscan:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench mcscan --s 32 --dtype fp16
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench mcscan --s 64 --dtype fp16
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench mcscan --s 128 --dtype fp16

profile_scscan:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench scscan --s 32 --dtype fp16
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench scscan --s 64 --dtype fp16
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench scscan --s 128 --dtype fp16

profile_fp16_compress:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench compress --s 32  --density ${DENSITY} --dtype fp16
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench compress --s 64  --density ${DENSITY} --dtype fp16
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench compress --s 128 --density ${DENSITY} --dtype fp16

profile_fp32_compress:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench compress --s 32  --density ${DENSITY} --dtype fp32
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench compress --s 64  --density ${DENSITY} --dtype fp32
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench compress --s 128 --density ${DENSITY} --dtype fp32

profile_fp16_seg_scan_sc:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench seg_scan_sc --dtype fp16 --s 32 --density ${DENSITY} --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench seg_scan_sc --dtype fp16 --s 64 --density ${DENSITY} --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench seg_scan_sc --dtype fp16 --s 128 --density ${DENSITY} --num_cores 1

profile_fp16_segmented_sum:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench segmented_sum --density ${DENSITY} --s 32 --dtype fp16
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench segmented_sum --density ${DENSITY} --s 64 --dtype fp16
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench segmented_sum --density ${DENSITY} --s 128 --dtype fp16

profile_fp32_custom_copy:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench custom_copy --dtype fp32 --num_cores 1


profile_fp16_custom_copy:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench custom_copy --dtype fp16 --num_cores 1

profile_fp16_vec_seg_scan:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench vec_seg_scan_sc --dtype fp16 --density ${DENSITY}  --s 32  --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench vec_seg_scan_sc --dtype fp16 --density ${DENSITY}  --s 64  --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench vec_seg_scan_sc --dtype fp16 --density ${DENSITY}  --s 128 --num_cores 1

profile_diffs: profile_fp16_diff profile_fp16_diff_cann profile_fp16_diffp_cann profile_fp32_diff_cann

custom_test:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/  python3 -m pytest tests/test_diff_cann.py -v

profile_fp16_segmented_scan_real:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${FULL_SPARSE_MATRIX_PATH} --s 32 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${FULL_SPARSE_MATRIX_PATH} --s 64 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${FULL_SPARSE_MATRIX_PATH} --s 128 --num_cores 1

profile_fp16_vec_segmented_scan_real:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${FULL_SPARSE_MATRIX_PATH} --s 32 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${FULL_SPARSE_MATRIX_PATH} --s 64 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${FULL_SPARSE_MATRIX_PATH} --s 128 --num_cores 1

profile_fp16_compress_real:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench compress --s 32  --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp16
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench compress --s 64  --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp16
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench compress --s 128 --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp16

profile_fp32_compress_real:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench compress --s 32  --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp32
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench compress --s 64  --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp32
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench compress --s 128 --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp32

profile_mcscan_real:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench mcscan --s 32  --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp16
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench mcscan --s 64  --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp16
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench mcscan --s 128 --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp16

profile_fp16_diff_real:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench diff --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp16

profile_fp32_diff_real:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench diff --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp16

paper_fig_6:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench mcscan --s 128 --dtype fp16
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench compress --s 128 --density 0.05 --dtype fp32
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench diff_cann --s 128 --dtype fp32
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_tcuscan_ops.py --bench copy --dtype fp16

paper_fig_5:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_random_matrices.py --bench custom_copy --dtype fp16 --distr Uniform --num_cores 1 --s 4096
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --distr Uniform --s 128 --density 0.01 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --distr Uniform --s 128 --density 0.001 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --distr Uniform --s 128 --density 0.0001 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_random_matrices.py --bench custom_copy --dtype fp16 --distr Uniform --num_cores 1 --s 128
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --distr PowerLaw --s 128 --density 0.01 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --distr PowerLaw --s 128 --density 0.001 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --distr PowerLaw --s 128 --density 0.0001 --num_cores 1

paper_fig_3:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Sandia/ASIC_680k/ASIC_680k --s 128 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Williams/mc2depi/mc2depi --s 128 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Williams/mac_econ_fwd500/mac_econ_fwd500 --s 128 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Williams/pdb1HYS/pdb1HYS --s 128 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Freescale/circuit5M/circuit5M --s 128 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Williams/pdb1HYS/pdb1HYS --s 128 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Freescale/circuit5M/circuit5M --s 128 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Williams/mac_econ_fwd500/mac_econ_fwd500 --s 128 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Williams/mc2depi/mc2depi --s 128 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Sandia/ASIC_680k/ASIC_680k --s 128 --num_cores 1





powerlaw_fig_5:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --distr PowerLaw --s 128 --density 0.01 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --distr PowerLaw --s 128 --density 0.001 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --distr PowerLaw --s 128 --density 0.0001 --num_cores 1

uniform_fig_5:
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --distr Uniform --s 128 --density 0.01 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --distr Uniform --s 128 --density 0.001 --num_cores 1
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/ DEVICE_TYPE=npu python3 profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --distr Uniform --s 128 --density 0.0001 --num_cores 1
