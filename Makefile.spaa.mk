############################
# SPAA 2025 paper figures  #
############################

paper_fig_3:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}ASIC_680k/ASIC_680k --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}mc2depi/mc2depi --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}mac_econ_fwd500/mac_econ_fwd500 --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}pdb1HYS/pdb1HYS --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}circuit5M/circuit5M --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}pdb1HYS/pdb1HYS --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}circuit5M/circuit5M --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}mac_econ_fwd500/mac_econ_fwd500 --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}mc2depi/mc2depi --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}ASIC_680k/ASIC_680k --s 128 --num_cores 1

profile_csr_gather:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench copy --dtype fp16 --num_cores 20
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench csr_gather --dtype fp16 --num_cores 20

profile_mcgather:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench copy --dtype fp16 --num_cores 20
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcgather --dtype fp16 --num_cores 20


fig_3_matmul:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}pdb1HYS/pdb1HYS --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}circuit5M/circuit5M --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}mac_econ_fwd500/mac_econ_fwd500 --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}mc2depi/mc2depi --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}ASIC_680k/ASIC_680k --s 128 --num_cores 1

fig_3_veconly:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}ASIC_680k/ASIC_680k --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}mc2depi/mc2depi --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}mac_econ_fwd500/mac_econ_fwd500 --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}pdb1HYS/pdb1HYS --s 128 --num_cores 1
	python3 ${PROFILING_SCRIPTS_PATH}/profile_sparse_matrices.py --bench vec_seg_scan_sc --dtype fp16 --matrixpath ${BASE_SPARSE_MATRIX_PATH}circuit5M/circuit5M --s 128 --num_cores 1

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
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan --s 128 --dtype fp16 --min-iter-index 4 --iter-step-multiplier 2
	#python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench scan_multi_cube --s 128 --dtype fp16 --min-iter-index 4 --iter-step-multiplier 2
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench compress --s 128 --density ${DENSITY} --dtype fp32 --min-iter-index 4 --iter-step-multiplier 2
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench diff_cann --s 128 --dtype fp32 --min-iter-index 4 --iter-step-multiplier 2
	# python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench copy --dtype fp16 --min-iter-index 4 --iter-step-multiplier 2

paper_fig_6_segscan:
	# python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan --s 128 --dtype fp16 --min-iter-index 4 --iter-step-multiplier 2
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan --s 128 --dtype int8 --min-iter-index 4 --iter-step-multiplier 2
	#python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench scan_multi_cube --s 128 --dtype fp16 --min-iter-index 4 --iter-step-multiplier 2
	#python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench scan_multi_cube --s 128 --dtype int8 --min-iter-index 4 --iter-step-multiplier 2
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_mc_revert --s 128 --dtype fp32 --density ${DENSITY} --min-iter-index 4 --iter-step-multiplier 2

paper_fig_6: paper_fig_6_segsum paper_fig_6_segscan


paper_fig_7:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench copy --s 128 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan --s 128 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan --s 128 --dtype int8
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench compress --s 128 --density 0.01 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench compress --s 128 --density 0.001 --dtype fp32
	# python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench compress --s 128 --density 0.005 --dtype fp32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench mcscan --s 128 --dtype int8
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench diff_cann --s 128 --dtype fp32
	# python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_mc_revert --s 128 --dtype fp32 --density 0.005
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_mc_revert --s 128 --dtype fp32 --density 0.001
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_mc_revert --s 128 --dtype fp32 --density 0.01

paper_fig_4b:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_sc --dtype fp16 --s 128 --density 0.0 --num_cores 1 --iter-step-multiplier 32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_sc --dtype fp16 --s 128 --density 0.0001 --num_cores 1 --iter-step-multiplier 32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_sc --dtype fp16 --s 128 --density 0.001 --num_cores 1 --iter-step-multiplier 32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_sc --dtype fp16 --s 128 --density 0.003 --num_cores 1 --iter-step-multiplier 32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_sc --dtype fp16 --s 128 --density 0.005 --num_cores 1 --iter-step-multiplier 32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_sc --dtype fp16 --s 128 --density 0.008 --num_cores 1 --iter-step-multiplier 32
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench seg_scan_sc --dtype fp16 --s 128 --density 0.01 --num_cores 1 --iter-step-multiplier 32
