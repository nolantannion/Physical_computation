import numpy as np
import matplotlib.pyplot as plt
from funciones_t9 import gs_poisson_1d, loc_max

# Parametros fisicos
N = 200
t0 = 0
tf = 10
t = np.linspace(t0, tf, N)
a = t[1] - t[0]

# Inicialización
x = np.zeros(N)
g = 9.81*np.ones_like(x)

# Iteración GS
tol = 1e-6
delta = 1.0

while delta > tol:
    x, delta = gs_poisson_1d(x, g, a=a)

# Solucion exacta 
x_exact = 0.5 * 9.81 * t * (tf - t)

# Magnitudes
vaprox = (x[1] - x[0]) / a
i, xmax = loc_max(x)

print(f'Velocidad incial estimada: {vaprox} m/s')
print(f'Altura maxima: {xmax[0]}')

# Representacion 
fig, ax = plt.subplots()

ax.plot(t, x, label="Numérico")
ax.plot(t, x_exact, '--', label="Exacta")

ax.set_xlabel("t (s)")
ax.set_ylabel("x (m)")
ax.grid()
ax.axis('tight')
ax.set_title('Elevación según t')

ax.legend(loc='lower center')


plt.savefig('Tiro_parab.png', dpi = 500)
plt.show()