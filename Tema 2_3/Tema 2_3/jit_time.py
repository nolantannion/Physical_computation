from numba import jit
import time

@jit(nopython = True)
def calcular_pi_rapido(n_terms):
    """Calcula usando serie de Leibniz."""
    suma = 0.0
    for k in range(n_terms):
        suma += (-1)**k / (2*k + 1)
    
    return 4 * suma


def calcular_pi_terms(n_terms):
    """Calcula usando serie de Leibniz."""
    suma = 0.0
    for k in range(n_terms):
        suma += (-1)**k / (2*k + 1)
    
    return 4 * suma





n = 10_000_000
start = time.time()
pi_lento = calcular_pi_terms(n) # Versi´on original
t_python = time.time() - start
start = time.time()
pi_rapido = calcular_pi_rapido(n) # Versi´on Numba
t_numba = time.time() - start
print(f"Python: {t_python:.2e}, Numba: {t_numba:.2e}")
print(f"Aceleraci´on: {t_python/t_numba:.0e}")
