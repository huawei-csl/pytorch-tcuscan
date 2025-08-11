#!/usr/bin/env python3

# Python script to download Sparse Suite matrices using ssgetpy


import os

import ssgetpy

SS_HOME = os.environ.get("SPARSE_SUITE_HOME", f"{os.getenv('HOME')}/.ssgetpy/")

_SPARSE_SUITE_MATRIX_IDS = [
    # 2225,
    # 2602,
    # 2444,
    # 1227,
    # 541,
    # 1867,
    # 1238,
    # 1399,
    # 1357,
    # 1385,
    # 845,
    # 566,
    # 1589,
    # 850,
    # 807,
    1258,
    1257,
    1352,
    2489,
    537,
    2633,
    1354,
    1282,
    368,
    761,
    1880,
]


def main():

    # Example 1: download matrices with nnz bounds.
    # matrices = ssgetpy.search(nzbounds=(100000, 10000000), limit=5)

    # Example 2: downloading from a group.
    # matrices = ssgetpy.search(group="QCD", limit=10)

    # Example 3: Download list of matrices by ids
    downloaded_mats = []
    print(f"Writing to {SS_HOME}")
    for matrix_id in _SPARSE_SUITE_MATRIX_IDS:
        matrix = ssgetpy.fetch(matrix_id, location=SS_HOME)[0]
        print(f"Downloading SS Matrix: {matrix.name}")
        file_location, _ = matrix.download(extract=True, destpath=SS_HOME)
        print(f"Downloaded SS Matrix: {matrix.name} @ {file_location}")
        downloaded_mats.append(
            {"name": matrix.name, "loc": f"{file_location}/{matrix.name}.mtx"}
        )

    with open("sparse_suite_matrices.csv", "w", encoding="utf-8") as fd:
        fd.write("name,location\n")
        for item in downloaded_mats:
            print(item)
            fd.write(f"{item['name']},{item['loc']}\n")


if __name__ == "__main__":
    main()
