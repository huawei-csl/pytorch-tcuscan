.PHONY: codegen deploy rollback clean

# Choose your kernel name below
# Convention: use came_case, i.e., awe_some
TCUSCAN_KERNEL=awesome_name
CPP_NAMESPACE=tcuscan

codegen:
	python3 gen_new_kernel.py --name ${TCUSCAN_KERNEL} --cpp_namespace ${CPP_NAMESPACE}

deploy: codegen
	mv kernel_${TCUSCAN_KERNEL}.h ../../src/kernels/
	mv tiling_${TCUSCAN_KERNEL}.h ../../src/tiling/
	mv kernel_${TCUSCAN_KERNEL}.cpp ../../src/${TCUSCAN_KERNEL}.cpp
	mv torch_${TCUSCAN_KERNEL}.h ../../src/torch/
	mv test_${TCUSCAN_KERNEL}.py ../../tests/
	@echo "******************************************************"
	@echo "  You need to include 'src/${TCUSCAN_KERNEL}.cpp' in the"
	@echo "  list of source files at 'ascendc_library(kernels SHARED ...)' "
	@echo "  of the CMakeLists.txt file."
	@echo "******************************************************"

rollback:
	rm ../../src/kernels/kernel_${TCUSCAN_KERNEL}.h
	rm ../../src/tiling/tiling_${TCUSCAN_KERNEL}.h
	rm ../../src/${TCUSCAN_KERNEL}.cpp
	rm ../../src/torch/torch_${TCUSCAN_KERNEL}.h
	rm ../../tests/test_${TCUSCAN_KERNEL}.py

clean:
	rm -rf kernel_${TCUSCAN_KERNEL}.h tiling_${TCUSCAN_KERNEL}.h ${TCUSCAN_KERNEL}.cpp  test_${TCUSCAN_KERNEL}.py
