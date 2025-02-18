CPPFILE?=./sparsesuite/baseline_sparsematrix_cpu.cpp
DENSITY?=0.05
LOCAL_SPARSE_MATRIX_NAME?='Boeing/bcsstk35/bcsstk35.mtx'
BASE_SPARSE_MATRIX_PATH?=${HOME}/.ssgetpy/MM/
MTXFILE?=${BASE_SPARSE_MATRIX_PATH}${LOCAL_SPARSE_MATRIX_NAME}

compile:
	g++ -O3 ${CPPFILE} -o test_baseline


clean:
	rm test_baseline

run: compile
	./test_baseline ${MTXFILE}
