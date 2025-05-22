.PHONY: compile run clean

EXECUTABLE=kml_spmv_sparsesuite
SS_MATRIX:=gyro_m/gyro_m.mtx
SPARSE_SUITE_HOME:=/scratch/TCUSCAN/sparse-suite-matrices

compile: clean
	mkdir build && cd build && cmake .. && make

run: ./build/${EXECUTABLE}
	./build/${EXECUTABLE} ${SPARSE_SUITE_HOME}/${SS_MATRIX}

clean:
	rm -rf build ${EXECUTABLE}
