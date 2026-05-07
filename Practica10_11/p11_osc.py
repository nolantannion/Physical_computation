'''
Alba Ruesga Alonso, Sofia Martín Alañón, Nolan Tannion Rodríguez
'''

'''
Evolucion temporal de un paquete gaussiano ante un potencial armonico.
El potencial esta reescalado para facilitar la visualizacion.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import trapezoid
from funciones_t11 import paso_psi

# ============================================================
# PARAMETROS
# ============================================================

L = 1.0

dx = 5e-4
N = int(L/dx)

dt = 5e-7
pasos = 5000

hbar = 1.0
m = 1.0

# ============================================================
# PAQUETE GAUSSIANO
# ============================================================

x0 = 0.5 * L
sigma = 0.03
k0 = 500

# ============================================================
# POTENCIAL ARMONICO
# ============================================================

omega = 3000

x = np.linspace(0, L, N)

V = 0.5 * omega**2 * (x - L/2)**2


# ============================================================
# FUNCION DE ONDA INICIAL
# ============================================================

psi = np.exp(-(x-x0)**2 / (2 * sigma**2))*np.exp(1j * k0 * x)

norm = np.sqrt(trapezoid(np.abs(psi)**2, x))
psi /= norm

psi[0] = 0
psi[-1] = 0

# ============================================================
# SOLO PUNTOS INTERIORES
# ============================================================

V_in = V[1:-1]
m_in = N - 2

# ============================================================
# COEFICIENTES CRANK-NICOLSON
# ============================================================

s = 1j * hbar * dt / (4 * m * dx**2)
r = 1j * dt * V_in / (2 * hbar)

a1 = 1 + 2*s + r
a2 = -s

b1 = 1 - 2*s - r
b2 = s

# ============================================================
# MATRIZ TRIDIAGONAL
# ============================================================

A = np.zeros((3, m_in), dtype=complex)

A[1, :] = a1
A[0, 1:] = a2
A[2, :-1] = a2

# ============================================================
# EVOLUCION TEMPORAL
# ============================================================
psi_sol = np.zeros((N, pasos), dtype=complex)
psi_sol[:, 0] = psi

for n in range(1, pasos):
    psi_sol[:, n] = paso_psi( psi_sol[:, n-1], b1, b2, A)

# Densidad de probabilidad
prob = np.abs(psi_sol)**2

# ============================================================
# ANIMACION
# ============================================================

fig, ax = plt.subplots(figsize=(9,5))

# Funcion de onda
linea_psi, = ax.plot( x, prob[:,0], lw=2, label=r'$|\psi(x,t)|^2$')

# Potencial escalado para visualizarlo junto a |psi|2
VP = V/np.max(V)*np.max(prob)
linea_V, = ax.plot( x, VP, '--', lw=2, label='Potencial')

ax.set_xlim(0, L)
ax.set_ylim(0, np.max(prob)*1.2)

ax.set_xlabel('x')
ax.set_ylabel(r'$|\psi|^2$')

ax.set_title('Evolución temporal')

ax.legend()

# Actualizacion
def update(frame):
    linea_psi.set_ydata(prob[:, frame])

    return linea_psi,

anim = FuncAnimation(fig, update, frames=np.arange(0, pasos, 5), interval=10, blit=True)
plt.show()

# Seleccion de fotogramas
im0, im1 = 0, 800
figura, eje = plt.subplots()

eje.plot(x, prob[:,im0], label = 'Distribución inicial')
eje.plot(x, prob[:,im1], label = f't = {im1*dt:.2e}')
eje.plot(x,VP, linestyle = '--', label = 'Potencial')

eje.set_xlabel('x')
eje.set_ylabel(r'$|\psi|^2$')
eje.set_title('Evolución de distribución Gaussiana')

eje.legend()

plt.savefig('Oscilador.png', dpi = 300)
plt.show()
