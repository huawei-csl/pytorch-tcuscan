#!/usr/bin/env python3

# Python script to run experiments on all sparse suite matrices given a ssgetpy query.

import os
import subprocess

import ssgetpy

SS_HOME = os.environ.get("SPARSE_SUITE_HOME", f"{os.getenv('HOME')}/.ssgetpy/")


def run_exp(sparse_suite_mat_loc: str, omp_num_threads: int):
    """Run the a command line script given the Sparse Suite Matrix location on disk and the number of OMP threads."""
    env_setup = f"export OMP_PLACES=cores && export OMP_NUM_THREADS={omp_num_threads}"
    return subprocess.call(
        f"{env_setup} && ./build/spmv_sparsesuite {sparse_suite_mat_loc}", shell=True
    )


def main():

    omp_num_threads = 32

    matrices = ssgetpy.search(
        nzbounds=(100000, 100000000), colbounds=(10000, 70000), limit=900
    )

    for matrix in matrices:
        print(f"name: {matrix.name}")
        print(f"rows: {matrix.rows} | {matrix.cols}")
        matrix_id = matrix.id
        matrix = ssgetpy.fetch(matrix_id, location=SS_HOME)[0]
        print(f"Downloading SS Matrix: {matrix.name}")
        file_location, _ = matrix.download(extract=True, destpath=SS_HOME)
        print(f"Downloaded SS Matrix: {matrix.name} @ {file_location}")
        mat_full_path = os.path.join(file_location, f"{matrix.name}.mtx")
        try:
            _ = run_exp(mat_full_path, omp_num_threads)
        except subprocess.CalledProcessError as result:
            print("error code", result.returncode, result.output)
            break


if __name__ == "__main__":
    main()
