import numpy as np
import matplotlib.pyplot as plt
from funciones_t9 import jacobi_poisson
from time import time

# Parámetros
N = 100         # tamaño de la malla
a = 1.0          # paso de red
tol = 1e-6 

# Inicialización
V = np.zeros((N, N))
rho = np.zeros((N, N))

# Definir dos cuadrados de carga
rho[20:40, 20:40] = -1.0
rho[60:80, 60:80] = 1.0

# Iteración de jacobi
delta = 1.0
t0  = time()
while delta > tol:
    V, delta = jacobi_poisson(V = V, rho = rho)

tf = time() - t0
print(f'Tiempo de calculo {tf}')

# Plot
plt.imshow(V, cmap='inferno', origin= 'lower')
plt.colorbar(label='Potencial')
plt.title('Solución de Poisson')
plt.show()