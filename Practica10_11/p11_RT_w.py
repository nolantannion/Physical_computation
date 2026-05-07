'''
Alba Ruesga Alonso, Sofia Martín Alañón, Nolan Tannion Rodríguez
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from funciones_t11 import paso_psi

'''
Script para calcular la variacion de R y T en funcion del grosor de la barrera
'''

# =========================================================
# Parámetros y constantes
# =========================================================

L = 1.0

dx = 5e-4
N = int(L / dx)

dt = 5e-7
pasos = 2000

tiempo = np.arange(pasos) * dt

hbar = 1.0
m = 1.0

# =========================================================
# Paquete gaussiano
# =========================================================

x0 = 0.25 * L
sigma = 0.03
k0 = 700

# =========================================================
# Barrera
# =========================================================

xc = 0.6 * L

# Distintos grosores
WW = [0.01, 0.03, 0.05, 0.1]

# Altura fija
V0 = 350*k0

# =========================================================
# Malla espacial
# =========================================================

x = np.linspace(0, L, N)

# =========================================================
# Función de onda inicial
# =========================================================

psi0 = (
    np.exp(-(x - x0)**2 / (2 * sigma**2))
    * np.exp(1j * k0 * x)
)

norm = np.sqrt(trapezoid(np.abs(psi0)**2, x))
psi0 /= norm

psi0[0] = 0
psi0[-1] = 0

# =========================================================
# Puntos interiores
# =========================================================

m_in = N - 2

# =========================================================
# Figura
# =========================================================

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# =========================================================
# Bucle sobre anchuras
# =========================================================

for w in WW:

    # ---------------------------------------------
    # Potencial
    # ---------------------------------------------

    V = np.zeros(N)

    zona_barrera = (x >= xc) & (x <= xc + w)

    V[zona_barrera] = V0

    V_in = V[1:-1]

    # ---------------------------------------------
    # Regiones T y R
    # ---------------------------------------------

    transm = x > (xc + w)
    refl = x < xc

    # ---------------------------------------------
    # Coeficientes Crank-Nicolson
    # ---------------------------------------------

    s = 1j * hbar * dt / (4 * m * dx**2)
    r = 1j * dt * V_in / (2 * hbar)

    a1 = 1 + 2*s + r
    a2 = -s

    b1 = 1 - 2*s - r
    b2 = s

    # ---------------------------------------------
    # Matriz tridiagonal
    # ---------------------------------------------

    A = np.zeros((3, m_in), dtype=complex)

    A[1, :] = a1
    A[0, 1:] = a2
    A[2, :-1] = a2

    # ---------------------------------------------
    # Evolución temporal
    # ---------------------------------------------

    psi_sol = np.zeros((N, pasos), dtype=complex)

    psi_sol[:, 0] = psi0.copy()

    for n in range(1, pasos):

        psi_sol[:, n] = paso_psi(psi_sol[:, n-1], b1, b2, A)

    # ---------------------------------------------
    # T y R
    # ---------------------------------------------

    T = np.zeros(pasos)
    R = np.zeros(pasos)

    for n in range(pasos):

        prob = np.abs(psi_sol[:, n])**2

        T[n] = trapezoid(prob[transm], x[transm])

        R[n] = trapezoid(prob[refl], x[refl])

    # ---------------------------------------------
    # Gráficas
    # ---------------------------------------------

    ax[0].plot(tiempo, T, label=fr'$w = {w:.3f}$')

    ax[1].plot(tiempo, R, label=fr'$w = {w:.3f}$')

# =========================================================
# Configuración
# =========================================================

ax[0].set_title('Transmitancia')
ax[0].set_xlabel('Tiempo')
ax[0].set_ylabel('T(t)')
ax[0].grid()
ax[0].legend()

ax[1].set_title('Reflectancia')
ax[1].set_xlabel('Tiempo')
ax[1].set_ylabel('R(t)')
ax[1].grid()
ax[1].legend()

plt.tight_layout()

plt.savefig('TR_distintos_grosores.png', dpi=300)

plt.show()