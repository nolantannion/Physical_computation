import numpy as np
from funciones_t11 import banded
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# Parámetros físicos
# =========================
hbar = 1.0546e-34
m = 9.109e-31

L = 1e-8
N = 1000
a = L / N
h = 1e-18

# =========================
# Condición inicial
# =========================
x = np.linspace(0, L, N)

x0 = L/2
sigma = 1e-10
kappa = 5e10

psi = np.exp(-(x-x0)**2/(2*sigma**2)) * np.exp(1j*kappa*x)

# condiciones de contorno
psi[0] = psi[-1] = 0

# puntos interiores
m_in = N - 2

# =========================
# Coeficientes CN
# =========================
s = 1j * hbar * h / (4 * m * a**2)

a1 = 1 + 2*s
a2 = -s

b1 = 1 - 2*s
b2 = s

# =========================
# Matriz A (banded)
# =========================
A = np.zeros((3, m_in), dtype=complex)

A[1, :] = a1
A[0, 1:] = a2
A[2, :-1] = a2

# =========================
# Paso temporal
# =========================
def paso_psi(psi):
    psi_in = psi[1:-1]

    # RHS vectorizado (mucho más rápido)
    v = b1 * psi_in.copy()
    v[1:] += b2 * psi_in[:-1]
    v[:-1] += b2 * psi_in[1:]

    # resolver sistema
    psi_new = banded(A, v, 1, 1)

    # reconstruir
    psi_next = np.zeros_like(psi, dtype=complex)
    psi_next[1:-1] = psi_new
    # bordes ya son 0

    return psi_next

# =========================
# Evolución temporal
# =========================
pasos = 200

psi_sol = np.zeros((N, pasos), dtype=complex)
psi_sol[:, 0] = psi

for n in range(1, pasos):
    psi_sol[:, n] = paso_psi(psi_sol[:, n-1])

# =========================
# Figura
# =========================
fig, ax = plt.subplots()
line, = ax.plot(x, np.real(psi_sol[:, 0]), lw=2)

ax.set_xlim(0, L)
ax.set_ylim(-1, 1)
ax.set_xlabel("x")
ax.set_ylabel(r"Re($\psi$)")
ax.set_title("Evolución temporal (Crank-Nicolson)")

# =========================
# Animación
# =========================
def update(frame):
    line.set_ydata(np.real(psi_sol[:, frame]))
    return line,

ani = FuncAnimation(
    fig,
    update,
    frames=pasos,
    interval=30,
    blit=True
)

plt.show()