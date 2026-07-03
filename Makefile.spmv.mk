SPARSE_SUITE_HOME:=/scratch/TCUSCAN/sparse-suite-matrices/ssgetpy-downloaded-matrices/
# List of selected sparse matrices
SPARSE_MATRICES=pdb1HYS/pdb1HYS rma10/rma10 conf5_4-8x8-05/conf5_4-8x8-05 conf5_4-8x8-10/conf5_4-8x8-10 mip1/mip1 cant/cant
# Sparse suite matrices used in ALENEX
ALENEX_MATRICES=vsp_bcsstk30_500sep_10in_1Kout/vsp_bcsstk30_500sep_10in_1Kout kron_g500-logn16/kron_g500-logn16 enron/enron \
					water_tank/water_tank mip1/mip1 gupta2/gupta2 bcircuit/bcircuit TSOPF_FS_b300_c2/TSOPF_FS_b300_c2 \
					nasasrb/nasasrb qa8fm/qa8fm g7jac200/g7jac200 pct20stif/pct20stif c-67b/c-67b H2O/H2O Ga3As3H12/Ga3As3H12 \
					me2010/me2010 k1_san/k1_san crankseg_2/crankseg_2 laminar_duct3D/laminar_duct3D pdb1HYS/pdb1HYS \
					pkustk04/pkustk04 crankseg_1/crankseg_1 struct3/struct3 c-70/c-70 Chebyshev4/Chebyshev4 GaAsH6/GaAsH6 \
					srb1/srb1 cant/cant
ICLR_MATRICES=b-2_r-4_w-1_g-2_n-65536/b-2_r-4_w-1_g-2_n-65536 \
				b-4_r-2_w-1_g-2_n-65536/b-4_r-2_w-1_g-2_n-65536 \
				b-16_r-4_w-1_g-2_n-65536/b-16_r-4_w-1_g-2_n-65536 \
				b-16_r-2_w-1_g-2_n-65536/b-16_r-2_w-1_g-2_n-65536 \
				b-2_r-2_w-1_g-2_n-65536/b-2_r-2_w-1_g-2_n-65536 \
				b-64_r-4_w-1_g-2_n-65536/b-64_r-4_w-1_g-2_n-65536 \
				b-4_r-4_w-1_g-2_n-65536/b-4_r-4_w-1_g-2_n-65536 \
				b-4_r-8_w-1_g-2_n-65536/b-4_r-8_w-1_g-2_n-65536 \
				b-16_r-8_w-1_g-2_n-65536/b-16_r-8_w-1_g-2_n-65536 \
				b-4_r-6_w-1_g-2_n-65536/b-4_r-6_w-1_g-2_n-65536 \
				b-16_r-6_w-1_g-2_n-65536/b-16_r-6_w-1_g-2_n-65536 \
				b-64_r-2_w-1_g-2_n-65536/b-64_r-2_w-1_g-2_n-65536 \
				b-2_r-8_w-1_g-2_n-65536/b-2_r-8_w-1_g-2_n-65536 \
				b-64_r-8_w-1_g-2_n-65536/b-64_r-8_w-1_g-2_n-65536 \
				b-64_r-6_w-1_g-2_n-65536/b-64_r-6_w-1_g-2_n-65536 \
				b-2_r-6_w-1_g-2_n-65536/b-2_r-6_w-1_g-2_n-65536


##################################
# SpMV related makefile targets  #
##################################

profile_fp32_gather_spmv_matrix:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench gather_spmv --dtype fp32 --prob Uniform --num_cores 20 --s 128 --density 0.001
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench gather_spmv --dtype fp32 --prob Uniform --num_cores 20 --s 256 --density 0.001
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench gather_spmv --dtype fp32 --prob Uniform --num_cores 20 --s 512 --density 0.001
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench gather_spmv --dtype fp32 --prob Uniform --num_cores 20 --s 128 --density 0.0001
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench gather_spmv --dtype fp32 --prob Uniform --num_cores 20 --s 256 --density 0.0001
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench gather_spmv --dtype fp32 --prob Uniform --num_cores 20 --s 512 --density 0.0001

profile_fp32_gather_spmv:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench gather_spmv --dtype fp32 --num_cores 20 --s 128
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench gather_spmv --dtype fp32 --num_cores 20 --s 256
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench gather_spmv --dtype fp32 --num_cores 20 --s 512

profile_fp16_spmv:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench spmv --dtype fp16 --prob Uniform --num_cores 20 --s 128 --density 0.001

profile_fp16_spmv_uniform:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench spmv --dtype fp16 --prob Uniform --num_cores 20 --s 128 --density 0.1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench spmv --dtype fp16 --prob Uniform --num_cores 20 --s 128 --density 0.01
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench spmv --dtype fp16 --prob Uniform --num_cores 20 --s 128 --density 0.001
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench spmv --dtype fp16 --prob Uniform --num_cores 20 --s 128 --density 0.0001

profile_fp16_spmv_powerlaw:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench spmv --dtype fp16 --prob PowerLaw --num_cores 20 --s 128 --density 0.1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench spmv --dtype fp16 --prob PowerLaw --num_cores 20 --s 128 --density 0.01
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench spmv --dtype fp16 --prob PowerLaw --num_cores 20 --s 128 --density 0.001
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench spmv --dtype fp16 --prob PowerLaw --num_cores 20 --s 128 --density 0.0001

profile_fp16_spmv_random: profile_fp16_spmv_powerlaw profile_fp16_spmv_uniform

profile_fp16_spmv_real:
	$(foreach MATRIX,$(SPARSE_MATRICES), python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv --s 128  --matrixpath ${SPARSE_SUITE_HOME}/${MATRIX} --dtype fp16;)

profile_fp16_spmv_real_multi_cube:
	$(foreach MATRIX,$(SPARSE_MATRICES), python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128  --matrixpath ${SPARSE_SUITE_HOME}/${MATRIX} --dtype fp16;)

profile_fp16_spmv_v2_real:
	$(foreach MATRIX,$(SPARSE_MATRICES), python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_v2 --s 128  --matrixpath ${SPARSE_SUITE_HOME}/${MATRIX} --dtype fp16;)

profile_fp16_spmv_versions: profile_fp16_spmv_real profile_fp16_spmv_real_multi_cube profile_fp16_spmv_v2_real

profile_fp16_spmv_real_multi_cube_for_alenex26: # Suite-Sparse matrix ids: [2225, 2602, 2444, 1227, 541, 1867, 1238, 1399, 1357, 1385, 845, 566, 1589, 850, 807] + ...
	$(foreach MATRIX,$(ALENEX_MATRICES), python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128  --matrixpath ${SPARSE_SUITE_HOME}/${MATRIX} --dtype fp16;)

profile_fp16_spmv_real_multi_cube_for_iclr26:
	$(foreach MATRIX,$(ALENEX_MATRICES), python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128  --matrixpath /scratch/asobczyk/datasets/suite-sparse-alenex26/${MATRIX} --dtype fp16;)
	$(foreach MATRIX,$(ICLR_MATRICES), python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128  --matrixpath /scratch/asobczyk/datasets/big-bird-artificial-matrices/${MATRIX} --dtype fp16;)

profile_fp16_csr_gather_real:
	$(foreach MATRIX,$(SPARSE_MATRICES), python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench csr_gather --s 128 --dtype fp16 --matrixpath ${SPARSE_SUITE_HOME}/${MATRIX};)

profile_mcscan_real:
	$(foreach MATRIX,$(SPARSE_MATRICES), python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench mcscan --s 128 --dtype fp16 --matrixpath ${SPARSE_SUITE_HOME}/${MATRIX};)

profile_fp32_gather_spmv_real:
	$(foreach MATRIX,$(SPARSE_MATRICES), python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench gather_spmv --s 128 --dtype fp32 --matrixpath ${SPARSE_SUITE_HOME}/${MATRIX};)

profile_fp32_diff_real:
	$(foreach MATRIX,$(SPARSE_MATRICES), python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench diff --s 128 --dtype fp32 --matrixpath ${SPARSE_SUITE_HOME}/${MATRIX};)

breakdown_spmv: profile_fp16_spmv_real profile_fp16_csr_gather_real profile_mcscan_real profile_fp32_gather_spmv_real profile_fp32_diff_real

profile_spmv_all_ss:
	export SPARSE_SUITE_HOME=${SPARSE_SUITE_HOME} && python3 ${PROFILING_SCRIPTS_PATH}/profile_ascend_spmv.py --bench spmv_multi_cube --s 128  --dtype fp16
