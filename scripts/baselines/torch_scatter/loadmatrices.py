import numpy as np
import ssgetpy
from scipy.io import mmread
from scipy.sparse import csr_matrix
from tqdm import tqdm


def convert_to_segments(A_ssget):
    "Converts a sparse matrix from ssget into a segmented sum/scan input (x, f)"

    file_location, _ = A_ssget.download(extract=True)
    name = A_ssget.name

    A = mmread(f"{file_location}/{name}.mtx")
    B = csr_matrix(A)

    # Data vector
    x = B.data

    # Flags vector
    f = np.zeros(B.nnz + 1)
    f[B.indptr] = 1

    return x, f


if __name__ == "__main__":

    matrices = ssgetpy.search(
        rowbounds=(36417, 36417),
        colbounds=(36417, 36417),
        nzbounds=(100000, 10000000),
        limit=2000,
    )

    spmat_dict = {}
    for mat in tqdm(matrices):
        name = mat.name
        x, f = convert_to_segments(mat)
        ratio = sum(f) / len(f)
