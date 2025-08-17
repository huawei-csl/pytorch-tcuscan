.PHONY: compile run clean

EXECUTABLE=spmv_sparsesuite
SS_MATRIX:=Bova/rma10/rma10.mtx
SPARSE_SUITE_HOME:=${HOME}/.ssgetpy/MM

compile: clean
	mkdir build && cd build && cmake .. && make

run: compile
	./build/${EXECUTABLE} ${SPARSE_SUITE_HOME}/${SS_MATRIX}

run-all: compile
	./execute_tests.sh

clean:
	rm -rf build ${EXECUTABLE}
