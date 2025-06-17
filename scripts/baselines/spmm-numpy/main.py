import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import random as sp_random


def get_block_triu_matrix_from_A(A):
    blocks = []
    offset = 0
    for left, right in zip(A.indptr, A.indptr[1:]):
        width = right - left
        s, e = offset, offset + width
        U_s = np.diag(A.data[s:e]) @ np.triu(np.ones(width))
        blocks.append(U_s)
        offset += width
    Z = block_diag(*blocks)
    return Z


if __name__ == "__main__":
    m = 16
    n = 531
    A = sp_random(m, m, density=0.25, format="csr")
    s = A.nnz
    B = np.random.rand(m, n)
    print(f"A.indptr : {A.indptr}")

    DU = get_block_triu_matrix_from_A(A)
    S = A.indptr[1:] - 1
    B_prime = B[A.indices, :]

    BDUS = B_prime.T.dot(DU)[:, S]
    print(np.linalg.norm(A.dot(B) - BDUS.T, "fro"))
