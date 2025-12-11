#############################
# Triangular Matrix inverse #
#############################

profile_tri_inv:
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tri_inv_cube_col_sweep --s 16 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tri_inv_cube_col_sweep --s 32 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tri_inv_cube_col_sweep --s 64 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tri_inv_cube_col_sweep --s 128 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tri_inv_col_sweep --s 16 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tri_inv_col_sweep --s 32 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tri_inv_col_sweep --s 64 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tri_inv_col_sweep --s 128 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tri_inv_baseline --s 16 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tri_inv_baseline --s 32 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tri_inv_baseline --s 64 --dtype fp16
	python3 ${PROFILING_SCRIPTS_PATH}/profile_tcuscan_ops.py --bench tri_inv_baseline --s 128 --dtype fp16
