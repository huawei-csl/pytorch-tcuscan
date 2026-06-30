import time

import numpy as np
from scipy.sparse import random as sp_random
from sparse_dot_mkl import dot_product_mkl

if __name__ == "__main__":
    n = 4000
    A = sp_random(n, n, density=0.01, format="csr")
    b = np.random.rand(
        n,
    )

    t = time.time()
    x_mkl = dot_product_mkl(A, b)
    t_mkl = time.time() - t

    t = time.time()
    x_scipy = A.dot(b)
    t_scipy = time.time() - t

    print(f"Error:      {np.linalg.norm(x_mkl - x_scipy):.2e}")
    print(f"Time mkl:   {t_mkl:.2e}")
    print(f"Time scipy: {t_scipy:.2e}")
