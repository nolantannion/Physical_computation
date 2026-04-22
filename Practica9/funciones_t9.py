import numpy as np


def gs_poisson(V, rho, a=1.0):
    """
    Realiza 1 barrido Gauss-Seidel sobre la matriz V.
    """
    N, M = V.shape
    delta = 0.0

    for i in range(1, N-1):
        for j in range(1, M-1):
            V_new = 0.25 * (
                V[i+1, j] + V[i-1, j] +
                V[i, j+1] + V[i, j-1] +
                a**2 * rho[i, j]
            )
            
            diff = abs(V_new - V[i, j])
            if diff > delta:
                delta = diff

            V[i, j] = V_new

    return V, delta


def jacobi_poisson(V, rho, a=1.0):
    """
    Un paso de Jacobi vectorizado.
    """
    V_new = V.copy()

    V_new[1:-1, 1:-1] = 0.25 * (
        V[2:, 1:-1] + V[:-2, 1:-1] +
        V[1:-1, 2:] + V[1:-1, :-2] +
        a**2 * rho[1:-1, 1:-1]
    )

    delta = np.max(np.abs(V_new - V))
    return V_new, delta



def gss_poisson(V, rho, a=1.0, omega=0.5):
    N, M = V.shape
    delta = 0.0

    for i in range(1, N-1):
        for j in range(1, M-1):
            old = V[i, j]

            V_gs = 0.25 * (
                V[i+1, j] + V[i-1, j] +
                V[i, j+1] + V[i, j-1] +
                a**2 * rho[i, j]
            )

            # SOR
            V[i, j] = old + omega * (V_gs - old)

            diff = abs(V[i, j] - old)
            if diff > delta:
                delta = diff

    return V, delta