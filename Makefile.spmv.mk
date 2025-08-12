##################################
# SpMV related makefile targets  #
##################################

profile_fp32_gather_spmv_matrix:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench gather_spmv --dtype fp32 --prob Uniform --num_cores 20 --s 128 --density 0.001
	python3 ${PROFILING_SCRIPTS_PATH}/profile_random_matrices.py --bench gather_spmv --dtype fp32 --prob Uniform --num_cores 20 --s 256 --density 0.001
	python3 ${PROFILING_SCRIPTS_PATH
	}/profile_random_matrices.py --bench gather_spmv --dtype fp32 --prob Uniform --num_cores 20 --s 512 --density 0.001
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
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}pdb1HYS/pdb1HYS --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}rma10/rma10 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}conf5_4-8x8-05/conf5_4-8x8-05 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}conf5_4-8x8-10/conf5_4-8x8-10 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}mip1/mip1 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}cant/cant --dtype fp16

profile_fp16_spmv_real_multi_cube:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}pdb1HYS/pdb1HYS --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}rma10/rma10 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}conf5_4-8x8-05/conf5_4-8x8-05 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}conf5_4-8x8-10/conf5_4-8x8-10 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}mip1/mip1 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}cant/cant --dtype fp16

profile_fp16_spmv_real_multi_cube_for_alenex26: # Suite-Sparse matrix ids: [2225, 2602, 2444, 1227, 541, 1867, 1238, 1399, 1357, 1385, 845, 566, 1589, 850, 807] + ...
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/vsp_bcsstk30_500sep_10in_1Kout/vsp_bcsstk30_500sep_10in_1Kout
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/kron_g500-logn16/kron_g500-logn16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/enron/enron
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/water_tank/water_tank
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/mip1/mip1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/gupta2/gupta2
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/bcircuit/bcircuit
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/TSOPF_FS_b300_c2/TSOPF_FS_b300_c2
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/nasasrb/nasasrb
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/qa8fm/qa8fm
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/g7jac200/g7jac200
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/pct20stif/pct20stif
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/c-67b/c-67b
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/H2O/H2O
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/Ga3As3H12/Ga3As3H12
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/me2010/me2010
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/k1_san/k1_san
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/crankseg_2/crankseg_2
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/laminar_duct3D/laminar_duct3D
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/pdb1HYS/pdb1HYS
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/pkustk04/pkustk04
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/crankseg_1/crankseg_1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/struct3/struct3
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/c-70/c-70
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/Chebyshev4/Chebyshev4
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/GaAsH6/GaAsH6
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/srb1/srb1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128 --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}/cant/cant


# 	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}mip1/mip1 --dtype fp16
# 	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}cant/cant --dtype fp16
# 	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}pdb1HYS/pdb1HYS --dtype fp16
# 	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench spmv_multi_cube --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}rma10/rma10 --dtype fp16


profile_fp16_csr_gather_real:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench csr_gather  --matrixpath ${BASE_SPARSE_MATRIX_PATH}pdb1HYS/pdb1HYS --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench csr_gather  --matrixpath ${BASE_SPARSE_MATRIX_PATH}rma10/rma10 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench csr_gather  --matrixpath ${BASE_SPARSE_MATRIX_PATH}conf5_4-8x8-05/conf5_4-8x8-05 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench csr_gather  --matrixpath ${BASE_SPARSE_MATRIX_PATH}conf5_4-8x8-10/conf5_4-8x8-10 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench csr_gather  --matrixpath ${BASE_SPARSE_MATRIX_PATH}mip1/mip1 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench csr_gather  --matrixpath ${BASE_SPARSE_MATRIX_PATH}cant/cant --dtype fp16

profile_mcscan_real:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench mcscan --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}pdb1HYS/pdb1HYS --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench mcscan --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}rma10/rma10 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench mcscan --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}conf5_4-8x8-05/conf5_4-8x8-05 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench mcscan --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}conf5_4-8x8-10/conf5_4-8x8-10 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench mcscan --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}mip1/mip1 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench mcscan --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}cant/cant --dtype fp16

profile_fp32_gather_spmv_real:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench gather_spmv --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}pdb1HYS/pdb1HYS --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench gather_spmv --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}rma10/rma10 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench gather_spmv --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}conf5_4-8x8-05/conf5_4-8x8-05 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench gather_spmv --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}conf5_4-8x8-10/conf5_4-8x8-10 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench gather_spmv --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}mip1/mip1 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench gather_spmv --s 128  --matrixpath ${BASE_SPARSE_MATRIX_PATH}cant/cant --dtype fp32

profile_fp32_diff_real:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench diff --matrixpath ${BASE_SPARSE_MATRIX_PATH}pdb1HYS/pdb1HYS --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench diff --matrixpath ${BASE_SPARSE_MATRIX_PATH}rma10/rma10 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench diff --matrixpath ${BASE_SPARSE_MATRIX_PATH}conf5_4-8x8-05/conf5_4-8x8-05 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench diff --matrixpath ${BASE_SPARSE_MATRIX_PATH}conf5_4-8x8-10/conf5_4-8x8-10 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench diff --matrixpath ${BASE_SPARSE_MATRIX_PATH}mip1/mip1 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench diff --matrixpath ${BASE_SPARSE_MATRIX_PATH}cant/cant --dtype fp32

breakdown_spmv: profile_fp16_spmv_real profile_fp16_csr_gather_real profile_mcscan_real profile_fp32_gather_spmv_real profile_fp32_diff_real
