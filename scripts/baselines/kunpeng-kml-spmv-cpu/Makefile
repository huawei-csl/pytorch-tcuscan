.PHONY: compile run clean

EXECUTABLE=kml_spmv_sparsesuite
SS_MATRIX:=gyro_m/gyro_m.mtx
SPARSE_SUITE_HOME:=/scratch/TCUSCAN/sparse-suite-matrices

compile: clean
	mkdir build && cd build && cmake .. && make

run: ./build/${EXECUTABLE}
	./build/${EXECUTABLE} ${SPARSE_SUITE_HOME}/${SS_MATRIX}


# Run over all Sparse Suite matrices using ssgetpy query.
all_ss: compile
	export SPARSE_SUITE_HOME=${SPARSE_SUITE_HOME} && python3 execute_all_sparse_suite.py

clean:
	rm -rf build ${EXECUTABLE}
