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


def paso_psi(psi, b1, b2, A):
    '''
    Funcion que calcula un paso temporal usando Crank-Nicolson

    INPUTS:
    - psi: funcion de onda 1D 
    - b1, b2: coeficientes del termino independiente
    - A: matriz tridiagonal para resolver con los coeficientes

    (Requiere de la funcion banded)

    RETURNS:
    - psi_next: funcion de onda el el siguiente paso temporal 
    '''

    psi_in = psi[1:-1]

    # Construccion del vector RHS
    v = b1 * psi_in.copy()

    v[1:] += b2 * psi_in[:-1]
    v[:-1] += b2 * psi_in[1:]

    # Resolver sistema lineal
    psi_new = banded(A.copy(), v, 1, 1)

    # Reconstruccion con fronteras
    psi_next = np.zeros_like(psi, dtype=complex)

    psi_next[1:-1] = psi_new

    return psi_next