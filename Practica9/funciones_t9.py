import numpy as np
from numba import njit

@njit
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


@njit
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


@njit
def gs_cil(V, a, r_min):
    N = len(V)
    delta = 0.0

    for i in range(1, N-1):
        r_i = r_min + i * a

        coef_p = 1.0 + a/(2.0*r_i)
        coef_m = 1.0 - a/(2.0*r_i)

        V_new = 0.5 * (
            coef_p * V[i+1] +
            coef_m * V[i-1]
        )

        diff = abs(V_new - V[i])
        if diff > delta:
            delta = diff

        V[i] = V_new

    return V, delta


def jacobi_cil(V, a, r_min):
    ''' Resolucion de la ecuacion de laplace radial usando el metodo de jacobi ante simetria 
    polar y sobre z. Recibe un array unidimensional V, el paso entre puntos y el radio minimo en el sistema '''
    V_new = V.copy()

    # índices interiores
    i = np.arange(1, len(V)-1)
    r_i = r_min + i * a

    coef_p = 1.0 + a/(2.0*r_i)
    coef_m = 1.0 - a/(2.0*r_i)

    V_new[1:-1] = 0.5 * (
        coef_p * V[2:] +
        coef_m * V[:-2]
    )

    delta = np.max(np.abs(V_new - V))
    return V_new, delta



@njit
def gs_cil_2d(V, rho, a=1.0, r_min=1.0):
    """
    Gauss-Seidel para Poisson en coordenadas cilíndricas (r,z).
    
    V   : matriz (Nr, Nz)
    rho : matriz fuente (Nr, Nz)
    a   : paso (dr = dz)
    r_min : radio mínimo (evita r=0)
    """
    Nr, Nz = V.shape
    delta = 0.0

    for i in range(1, Nr-1):
        r_i = r_min + i * a

        cp = 1.0 + a/(2.0*r_i)
        cm = 1.0 - a/(2.0*r_i)

        for j in range(1, Nz-1):

            V_new = 0.25 * (
                cp * V[i+1, j] +
                cm * V[i-1, j] +
                V[i, j+1] + V[i, j-1] +
                a*a * rho[i, j]
            )

            diff = abs(V_new - V[i, j])
            if diff > delta:
                delta = diff

            V[i, j] = V_new

    return V, delta


@njit
def gs_poisson_1d(V, rho, a=1.0):
    ''' Un paso en la resolucion de poisson en 1D usando gauss seidel. 
    Recibe un array 1d V, el temrino inhomogenero rho tambien unidimensional y el paso entre puntos a.
    Devuelve un array con la solucion en el siguiente paso'''

    N = len(V)
    delta = 0.0

    for i in range(1, N-1):
            V_new = 0.5 * (
                V[i+1] + V[i-1] +
                a**2 * rho[i]
            )
            
            diff = abs(V_new - V[i])
            if diff > delta:
                delta = diff

            V[i] = V_new

    return V, delta




@njit
def gss_cil_1d(V, a, r_min, omega=0.9):
    """
    Gauss-Seidel + SOR para Laplace radial en coordenadas cilíndricas (1D).

    V      : array 1D (potencial)
    a      : paso espacial (dr)
    r_min  : radio mínimo
    omega  : parámetro SOR
    """
    N = len(V)
    delta = 0.0

    for i in range(1, N-1):
        r_i = r_min + i * a

        # coeficientes cilíndricos
        cp = 1.0 + a/(2.0*r_i)
        cm = 1.0 - a/(2.0*r_i)

        old = V[i]

        V_gs = 0.5 * (
            cp * V[i+1] +
            cm * V[i-1]
        )

        # SOR
        V[i] = old + omega * (V_gs - old)

        diff = abs(V[i] - old)
        if diff > delta:
            delta = diff

    return V, delta



# Funcion reutilizada del tema 5
def loc_max(s):
    '''
    INPUT
    s: array del que queremos hallar el maximo

    RETURN
    ind: array con los indices de los maximos
    M: array con los valores de los maximos locales 

    '''
    ind = np.where((s[1:-1] > s[:-2]) & (s[1:-1] > s[2:]))[0] + 1
    M = s[ind]
    return ind, M