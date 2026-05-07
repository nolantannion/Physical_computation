import numpy as np
from numba import njit


@njit()
def banded(Aa, va, up, down):
    A = np.copy(Aa)
    v = np.copy(va)
    N = len(v)

    for m in range(N):
        div = A[up, m]
        v[m] /= div
        for k in range(1, down+1):
            if m+k < N:
                v[m+k] -= A[up+k, m]*v[m]
        for i in range(up):
            j = m + up - i
            if j < N:
                A[i, j] /= div
                for k in range(1, down+1):
                    if j < N:
                        A[i+k, j] -= A[up+k, m]*A[i, j]

    for m in range(N-2, -1, -1):
        for i in range(up):
            j = m + up - i
            if j < N:
                v[m] -= A[i, j]*v[j]

    return v
