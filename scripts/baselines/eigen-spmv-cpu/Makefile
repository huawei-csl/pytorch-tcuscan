.PHONY: compile run clean

EXECUTABLE=spmv_sparsesuite
SS_MATRIX:=Bova/rma10/rma10.mtx
SPARSE_SUITE_HOME=/scratch/TCUSCAN/sparse-suite-matrices/ssgetpy-downloaded-matrices/

compile: clean
	mkdir build && cd build && cmake .. && make

run: compile
	./build/${EXECUTABLE} ${SPARSE_SUITE_HOME}/${SS_MATRIX}

# Run over all Sparse Suite matrices using ssgetpy query.
all_ss: compile
	export SPARSE_SUITE_HOME=${SPARSE_SUITE_HOME} && python3 execute_all_sparse_suite.py

clean:
	rm -rf build ${EXECUTABLE}
