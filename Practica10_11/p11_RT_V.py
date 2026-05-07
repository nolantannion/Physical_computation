'''
Alba Ruesga Alonso, Sofia Martín Alañón, Nolan Tannion Rodríguez
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from funciones_t11 import paso_psi

# =========================================================
# Parametros y constantes
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
# Barrera de potencial
# =========================================================

xc = 0.6 * L
w = 0.05

VV = [10*k0, 100*k0, 250*k0, k0**2]

# Malla espacial
x = np.linspace(0, L, N)

# Regiones
transm = x > (xc + w)
refl = x < xc

# =========================================================
# Funcion de onda inicial normalizada
# =========================================================
psi0 = (np.exp(-(x - x0)**2 / (2 * sigma**2))* np.exp(1j * k0 * x))

norm = np.sqrt(trapezoid(np.abs(psi0)**2, x))
psi0 /= norm

psi0[0] = 0
psi0[-1] = 0

# Puntos interiores
m_in = N - 2


# Figura
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# =========================================================
# Bucle sobre potenciales
# =========================================================

for V0 in VV:


    # Potencial
    V = np.zeros(N)

    zona_barrera = (x >= xc) & (x <= xc + w)
    V[zona_barrera] = V0

    V_in = V[1:-1]

    # Coeficientes Crank-Nicolson
    s = 1j * hbar * dt / (4 * m * dx**2)
    r = 1j * dt * V_in / (2 * hbar)

    a1 = 1 + 2*s + r
    a2 = -s

    b1 = 1 - 2*s - r
    b2 = s

    # Matriz tridiagonal
    A = np.zeros((3, m_in), dtype=complex)
    A[1, :] = a1
    A[0, 1:] = a2
    A[2, :-1] = a2



    # Evolución temporal
    psi_sol = np.zeros((N, pasos), dtype=complex)
    psi_sol[:, 0] = psi0.copy()

    for n in range(1, pasos):
        psi_sol[:, n] = paso_psi(psi_sol[:, n-1], b1=b1, b2=b2, A = A)

    # Calculo de T y R 
    T = np.zeros(pasos)
    R = np.zeros(pasos)

    # Evolucionamos la cantidad de pasos deseada almacenando T y R
    for n in range(pasos):
        prob = np.abs(psi_sol[:, n])**2

        T[n] = trapezoid(prob[transm], x[transm])
        R[n] = trapezoid(prob[refl], x[refl])

    # Graficamos
    ax[0].plot(tiempo, T, label=fr'$V_0 = {V0:.1e}$')
    ax[1].plot(tiempo, R, label=fr'$V_0 = {V0:.1e}$')



# Parametros de la representacion
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
plt.savefig('TR_distintos_potenciales.png', dpi=300)

plt.show()


