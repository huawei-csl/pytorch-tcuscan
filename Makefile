.PHONY: all clean setup_ci setup_once setup_once_aarch64 build test profile

DENSITY?=0.001
LOCAL_SPARSE_MATRIX_NAME?='Boeing/bcsstk35/bcsstk35'
BASE_SPARSE_MATRIX_PATH?=${HOME}/.ssgetpy/MM/
FULL_SPARSE_MATRIX_PATH=${BASE_SPARSE_MATRIX_PATH}${LOCAL_SPARSE_MATRIX_NAME}
PROFILING_SCRIPTS_PATH=./scripts/profiling/

DEVICE_TYPE?=npu
LD_LIBRARY_PATH := ${LD_LIBRARY_PATH}:$(shell pwd)/build/lib/

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

build: build.sh src/vadd.cpp src/diff.cpp src/seg_scan_single_core.cpp src/seg_scan_mc_revert.cpp src/pybind11.cpp
	./build.sh -v ASCEND910B4

docs:
	doxygen doxygen/Doxyfile

test:
	python3 -m pytest tests/

test_%:
	python3 -m pytest tests/test_$*.py

profile_%: profile_fp32_$* profile_fp16_$* profile_int16_$*

profile_fp16_%:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench $* --dtype fp16

profile_fp32_%:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench $* --dtype fp32

profile_int16_%:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench $* --dtype int16

profile_int32_%:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench $* --dtype int32

profile_all_s_fp16_%:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench $* --s 32 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench $* --s 64 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench $* --s 128 --dtype fp16

profile_mcscan:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench copy --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan --s 32 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan --s 64 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan --s 128 --dtype fp16

profile_scscan: profile_all_s_fp16_scscan

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

profile_mcscan_real:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench mcscan --s 32  --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench mcscan --s 64  --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench mcscan --s 128 --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp16

profile_fp16_diff_real:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench diff --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp16

profile_fp32_diff_real:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench diff --matrixpath ${FULL_SPARSE_MATRIX_PATH} --dtype fp16

profile_fp32_revert_mcscan:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_mc_revert --dtype fp32 --num_cores 20 --s 128 --density ${DENSITY}

profile_fp16_radixsort:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench radixsort --dtype fp16 --s 32 --num_cores 20
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench radixsort --dtype fp16 --s 64 --num_cores 20
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench radixsort --dtype fp16 --s 128 --num_cores 20


paper_fig_5:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench custom_copy --dtype fp16 --prob Uniform --num_cores 1 --s 4096
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --prob Uniform --s 128 --density 0.01 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --prob Uniform --s 128 --density 0.001 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --prob Uniform --s 128 --density 0.0001 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --prob PowerLaw --s 128 --density 0.01 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --prob PowerLaw --s 128 --density 0.001 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --prob PowerLaw --s 128 --density 0.0001 --num_cores 1

paper_fig_3:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Sandia/ASIC_680k/ASIC_680k --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Williams/mc2depi/mc2depi --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Williams/mac_econ_fwd500/mac_econ_fwd500 --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Williams/pdb1HYS/pdb1HYS --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Freescale/circuit5M/circuit5M --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Williams/pdb1HYS/pdb1HYS --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Freescale/circuit5M/circuit5M --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Williams/mac_econ_fwd500/mac_econ_fwd500 --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Williams/mc2depi/mc2depi --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Sandia/ASIC_680k/ASIC_680k --s 128 --num_cores 1
profile_fp16_csr_gather:

	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench csr_gather --dtype fp16 --num_cores 40

fig_3_matmul:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Williams/pdb1HYS/pdb1HYS --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Freescale/circuit5M/circuit5M --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Williams/mac_econ_fwd500/mac_econ_fwd500 --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Williams/mc2depi/mc2depi --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Sandia/ASIC_680k/ASIC_680k --s 128 --num_cores 1

fig_3_veconly:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Sandia/ASIC_680k/ASIC_680k --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Williams/mc2depi/mc2depi --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Williams/mac_econ_fwd500/mac_econ_fwd500 --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Williams/pdb1HYS/pdb1HYS --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}Freescale/circuit5M/circuit5M --s 128 --num_cores 1

paper_fig_3: fig_3_matmul fig_3_veconly

powerlaw_fig_5:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --prob PowerLaw --s 128 --density 0.01 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --prob PowerLaw --s 128 --density 0.001 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --prob PowerLaw --s 128 --density 0.0001 --num_cores 1

uniform_fig_5:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --prob Uniform --s 128 --density 0.01 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --prob Uniform --s 128 --density 0.001 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench seg_scan_sc --dtype fp16 --prob Uniform --s 128 --density 0.0001 --num_cores 1

paper_fig_5: powerlaw_fig_5 uniform_fig_5
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench custom_copy --dtype fp16 --prob Uniform --num_cores 1 --s 4096

paper_fig_6_segsum:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan --s 128 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench compress --s 128 --density ${DENSITY} --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench diff_cann --s 128 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench copy --dtype fp16

paper_fig_6_segscan:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan --s 128 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan --s 128 --dtype int8
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_mc_revert --s 128 --dtype fp32 --density ${DENSITY}

paper_fig_6: paper_fig_6_segsum paper_fig_6_segscan


paper_fig_7:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench copy --s 128 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan --s 128 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan --s 128 --dtype int8
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench compress --s 128 --density 0.01 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench compress --s 128 --density 0.001 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan --s 128 --dtype int8
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench diff_cann --s 128 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_mc_revert --s 128 --dtype fp32 --density 0.001
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_mc_revert --s 128 --dtype fp32 --density 0.01

paper_fig_4b:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_sc --dtype fp16 --s 128 --density 0.0 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_sc --dtype fp16 --s 128 --density 0.0001 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_sc --dtype fp16 --s 128 --density 0.001 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_sc --dtype fp16 --s 128 --density 0.003 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_sc --dtype fp16 --s 128 --density 0.004 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_sc --dtype fp16 --s 128 --density 0.01 --num_cores 1
