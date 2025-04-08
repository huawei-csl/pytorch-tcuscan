.PHONY: all clean


SPARSE_SUITE_HOME=/scratch/TCUSCAN/sparse-suite-matrices/ssgetpy-downloaded-matrices/

all:
	pip install ssgetpy
	export SPARSE_SUITE_HOME=${SPARSE_SUITE_HOME} && python3 download-sparse-suite-matrices.py


clean:
	rm sparse_suite_matrices.csv
