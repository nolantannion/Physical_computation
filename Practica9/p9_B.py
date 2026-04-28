import numpy as np
import matplotlib.pyplot as plt
from funciones_t9 import gs_cil

# Parámetros físicos
r0 = 0.01   
rf = 0.04   
Vi = 2.5    
Ve = 0.0    

# Malla
N = 200
r = np.linspace(r0, rf, N)
a = r[1] - r[0]

# Inicialización
V = np.zeros(N)
V[0] = Vi
V[-1] = Ve

# Iteración GS
tol = 1e-6
delta = 1.0

while delta > tol:
    V, delta = gs_cil(V, a, r0)

# Comparamos con la solucion analitica
V_sol = Vi * np.log(r/rf) / np.log(r0/rf)

# Representacion
fig, ax = plt.subplots()
ax.plot(r, V, label="Numérico")
ax.plot(r, V_sol, '--', label="Analítico")
ax.set_xlabel("r (m)")
ax.set_ylabel("V (V)")
ax.grid()
ax.axis('tight')
ax.vlines([r0,rf], -10,10, colors= 'red', linestyles= 'dashed', label = 'Frontera')
ax.legend(loc = 'upper center')
plt.savefig('Potencial_cil.png', dpi = 500)
plt.show()