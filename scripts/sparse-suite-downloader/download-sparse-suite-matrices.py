#!/usr/bin/env python3

# Python script to download Sparse Suite matrices using ssgetpy


import os

import ssgetpy

SS_HOME = os.environ.get("SPARSE_SUITE_HOME", f"{os.getenv('HOME')}/.ssgetpy/")

_SPARSE_SUITE_MATRIX_IDS = [
    1599,
    369,
    2380,
    1385,
    2377,
    2379,
    2276,
    1362,
    1353,
    2442,
    2443,
    1279,
    2376,
    544,
    2373,
    2374,
    2375,
    2378,
    1321,
    374,
    1599,
    1419,
    1598,
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
