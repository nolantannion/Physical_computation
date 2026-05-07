'''
Alba Ruesga Alonso, Sofia Martín Alañón, Nolan Tannion Rodríguez
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from funciones_t11 import banded


# ============================================================
# PARÁMETROS FÍSICOS Y COMPUTACIONALES
# ============================================================

# Caja unidimensional
L = 1.0

# Discretizacion espacial
dx = 5e-4
N = int(L/dx)

# Discretizacion temporal
dt = 5e-7
h = dt

# Número de iteraciones
pasos = 4000

# Constantes físicas 
hbar = 1.0
m = 1.0

# Paquete Gaussiano 
x0 = 0.25 * L   # Posicion inicial
sigma = 0.03    # Anchura del paquete
k0 = 700    # Momento inicial

# Barrera V
xc = 0.6 * L    # Centro de la barrera
w = 0.05    # Anchura
V0 = 1e5    # Altura del potencial

# MALLA ESPACIAL
x = np.linspace(0, L, N)

# FUNCIÓN DE ONDA INICIAL

psi = np.exp(-(x - x0)**2 / (2 * sigma**2))* np.exp(1j * k0 * x)

# Condiciones de contorno
psi[0] = 0
psi[-1] = 0

# POTENCIAL RECTANGULAR
V = np.zeros(N)

V[(x > xc) & (x < xc + w)] = V0

# Solo puntos interiores
V_in = V[1:-1]

# CRANK-NICOLSON
m_in = N - 2

# Coeficiente cinético
s = 1j * hbar * h / (4 * m * dx**2)

# Coeficiente potencial
r = 1j * h * V_in / (2 * hbar)

# Diagonales
a1 = 1 + 2*s + r
a2 = -s

b1 = 1 - 2*s - r
b2 = s

# MATRIZ A
A = np.zeros((3, m_in), dtype=complex)

for i in range(m_in):

    A[1, i] = a1[i]

    if i > 0:
        A[2, i-1] = a2

    if i < m_in - 1:
        A[0, i+1] = a2

# PASO TEMPORAL
def paso_psi(psi):

    psi_in = psi[1:-1]

    # Construcción del vector RHS
    v = b1 * psi_in.copy()

    v[1:] += b2 * psi_in[:-1]
    v[:-1] += b2 * psi_in[1:]

    # Resolver sistema lineal
    psi_new = banded(A.copy(), v, 1, 1)

    # Reconstrucción con fronteras
    psi_next = np.zeros_like(psi, dtype=complex)

    psi_next[1:-1] = psi_new

    return psi_next

# EVOLUCIÓN TEMPORAL
psi_sol = np.zeros((N, pasos), dtype=complex)
psi_sol[:, 0] = psi

for n in range(1, pasos):

    psi_sol[:, n] = paso_psi(psi_sol[:, n-1])

# ============================================================
# MÓDULO
# ============================================================

modpsi = abs(psi_sol)

# ============================================================
# ANIMACIÓN
# ============================================================

fig, ax = plt.subplots()

linea, = ax.plot(x, modpsi[:, 0])
ax.vlines([xc,xc+w], 0, 1, color = 'r')

ax.set_xlim(0, L)
ax.set_ylim(0, np.max(modpsi)*1.1)

ax.set_xlabel("x")
ax.set_ylabel(r"$|\psi(x,t)|$")
ax.set_title("Evolución temporal")

def update(frame):

    linea.set_ydata(modpsi[:, frame])

    return linea,

anim = FuncAnimation(
    fig,
    update,
    frames=np.arange(0, pasos, 10),
    blit=True,
    interval = 10
)

plt.show()