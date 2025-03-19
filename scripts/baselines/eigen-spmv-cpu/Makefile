

SRC:=example
EXECUTABLE=spmv_${SRC}
FILENAME:=Williams/pdb1HYS/pdb1HYS.mtx
BASE:=${HOME}/.ssgetpy/MM/

compile:
	g++ -O3 -Wall -I /scratch/gsorrentino/tcuscan/baseline/pytorch-tcuscan/scripts/baselines/eigen-spmv-cpu/eigen/eigen-master/ ${SRC}.cpp -o ${EXECUTABLE}

run: compile
	./${EXECUTABLE} ${BASE}${FILENAME}

clean:
	rm ${EXECUTABLE}
