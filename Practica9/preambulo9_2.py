import numpy as np
import matplotlib.pyplot as plt
from funciones_t9 import jacobi_poisson, gss_poisson, gs_poisson
from time import time

# Parámetros
N = 100         # tamaño de la malla
a = 1.0         # paso de red
tol = 1e-6      # El tiempo de calculo aumenta mucho

# Inicialización
V = np.zeros((N, N))
rho = np.zeros((N, N))

V[-1,:] = 1.0

# Iteración Gauss-Seidel
delta = 1.0
t0  = time()
while delta > tol:
    V, delta = gss_poisson(V = V, rho = rho, omega=0.9)
tf = time() - t0
print(f'GS: Tiempo de calculo {tf}')


# Iteración Gauss-Seidel
V2 = np.zeros_like(V)
delta = 1.0
t0  = time()
while delta > tol:
    V2, delta = jacobi_poisson(V = V2, rho = rho)
tf = time() - t0
print(f'Jacobi: Tiempo de calculo {tf}')

# Plot
plt.imshow(V, cmap='inferno', origin= 'lower')
plt.colorbar(label='Potencial')
plt.title('Solución de Poisson')
plt.show()