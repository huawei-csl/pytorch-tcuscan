#!/usr/bin/env python3

# Python script to download Sparse Suite matrices using ssgetpy


import os

import ssgetpy

SS_HOME = os.environ.get("SPARSE_SUITE_HOME", f"{os.getenv('HOME')}/.ssgetpy/")

# SuiteSparse matrix IDs for each matrix set defined in Makefile.spmv.mk.
# Keep these in sync with the matrix-name lists there.

# SPARSE_MATRICES (the default set used by e.g. profile_fp16_spmv_versions)
DEFAULT_MATRIX_IDS = [
    2373,  # pdb1HYS
    374,  # rma10
    1598,  # conf5_4-8x8-05
    1599,  # conf5_4-8x8-10
    1385,  # mip1
    2375,  # cant
]

# ALENEX_MATRICES (used by profile_fp16_spmv_real_multi_cube_for_alenex26)
ALENEX_MATRIX_IDS = [
    2633,  # vsp_bcsstk30_500sep_10in_1Kout
    2489,  # kron_g500-logn16
    2444,  # enron
    1880,  # water_tank
    1385,  # mip1
    537,  # gupta2
    541,  # bcircuit
    2225,  # TSOPF_FS_b300_c2
    761,  # nasasrb
    845,  # qa8fm
    566,  # g7jac200
    368,  # pct20stif
    1589,  # c-67b
    1357,  # H2O
    1352,  # Ga3As3H12
    2602,  # me2010
    1238,  # k1_san
    1258,  # crankseg_2
    1399,  # laminar_duct3D
    2373,  # pdb1HYS
    850,  # pkustk04
    1257,  # crankseg_1
    807,  # struct3
    1227,  # c-70
    1867,  # Chebyshev4
    1354,  # GaAsH6
    1282,  # srb1
    2375,  # cant
]

# NOTE: ICLR_MATRICES (b-*_r-*_w-*_g-*_n-65536) are artificial/generated
# matrices, not SuiteSparse downloads, so they have no ssgetpy IDs.

MATRIX_SETS = {
    "default": DEFAULT_MATRIX_IDS,
    "alenex": ALENEX_MATRIX_IDS,
    "all": DEFAULT_MATRIX_IDS + ALENEX_MATRIX_IDS,
}

# Which set to download; override with e.g. MATRIX_SET=alenex.
_MATRIX_SET = os.environ.get("MATRIX_SET", "default")
# De-duplicate while preserving order (some ids appear in multiple sets).
_SPARSE_SUITE_MATRIX_IDS = list(dict.fromkeys(MATRIX_SETS[_MATRIX_SET]))


def main():

    # Example 1: download matrices with nnz bounds.
    # matrices = ssgetpy.search(nzbounds=(100000, 10000000), limit=5)

    # Example 2: downloading from a group.
    # matrices = ssgetpy.search(group="QCD", limit=10)

    # Example 3: Download list of matrices by ids
    downloaded_mats = []
    print(
        f"Downloading matrix set '{_MATRIX_SET}' ({len(_SPARSE_SUITE_MATRIX_IDS)} matrices)"
    )
    print(f"Writing to {SS_HOME}")
    for matrix_id in _SPARSE_SUITE_MATRIX_IDS:
        matrix = ssgetpy.fetch(matrix_id, location=SS_HOME)[0]
        print(f"Downloading SS Matrix: {matrix.name}")
        file_location, _ = matrix.download(extract=True, destpath=SS_HOME)
        print(f"Downloaded SS Matrix: {matrix.name} @ {file_location}")
        # Sparsity = fraction of zero entries in the dense matrix.
        total = matrix.rows * matrix.cols
        sparsity = 1.0 - (matrix.nnz / total) if total else 0.0
        downloaded_mats.append(
            {
                "name": matrix.name,
                "loc": f"{file_location}/{matrix.name}.mtx",
                "nrows": matrix.rows,
                "ncols": matrix.cols,
                "nnz": matrix.nnz,
                "sparsity": sparsity,
            }
        )

    with open("sparse_suite_matrices.csv", "w", encoding="utf-8") as fd:
        fd.write("name,nrows,ncols,nnz,sparsity,location\n")
        for item in downloaded_mats:
            print(item)
            fd.write(
                f"{item['name']},{item['nrows']},{item['ncols']},"
                f"{item['nnz']},{item['sparsity']:.6g},{item['loc']}\n"
            )


if __name__ == "__main__":
    main()
