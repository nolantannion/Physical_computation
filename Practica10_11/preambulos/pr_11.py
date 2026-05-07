'''
Alba Ruesga Alonso, Sofia Martín Alañón, Nolan Tannion Rodríguez
'''


'''
La funcion de onda se va atenuando por el tamaño finito de la transformada a diferencia de usar Crank-Nicolson donde la onda se conserva.
'''


import numpy as np
from funciones_t11 import banded
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from dcst import dst, idst

# Parámetros físicos
hbar = 1.0546e-34
m = 9.109e-31

L = 1e-8
N = 1000
a = L / N
h = 1e-18


# Condición inicial
x = np.linspace(0, L, N-1)

x0 = L/2
sigma = 1e-10
kappa = 5e10

psi = np.exp(-(x-x0)**2/(2*sigma**2)) * np.exp(1j*kappa*x)

# Calculamos los coeficientes
alfa = dst(psi.real)
eta = dst(psi.imag)

bk = alfa + 1j*eta


k = np.arange(1,N, 1)
omega = np.pi**2 * hbar * k**2 / (2*m*L**2)

dt = 1e-18
t = np.arange(0, 1e-15 + dt, dt)

# Calculamos la evolucion temporal
psit = np.zeros((len(x), len(t)), dtype=complex)
for indice, tiempo in enumerate(t):

    # Evolución temporal de coeficientes
    bk_t = bk * np.exp(-1j * omega * tiempo)

    # Reconstrucción en espacio real
    psi_t = idst(bk_t.real) + 1j * idst(bk_t.imag)

    psit[:, indice] = psi_t

# Animacion
fig, ax = plt.subplots()
line, = ax.plot(x, np.real(psit[:, 0]), lw=2)

def update(frame):
    line.set_ydata(np.real(psit[:, frame]))
    return line,

ani = FuncAnimation(
    fig,
    update,
    frames=np.arange(0,len(t), 2),
    interval=30,
    blit=True
)



plt.show()




