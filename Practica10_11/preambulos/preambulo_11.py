'''
Alba Ruesga Alonso, Sofia Martín Alañón, Nolan Tannion Rodríguez
'''
import numpy as np
from funciones_t11 import banded
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Parámetros físicos
hbar = 1.0546e-34
m = 9.109e-31

L = 1e-8
N = 1000
a = L / N
h = 1e-18



# Condición inicial
x = np.linspace(0, L, N)

x0 = L/2
sigma = 1e-10
kappa = 5e10

psi = np.exp(-(x-x0)**2/(2*sigma**2)) * np.exp(1j*kappa*x)

# condiciones de contorno
psi[0] = psi[-1] = 0


# solo puntos interiores
psi_in = psi[1:-1]
m_in = N - 2



# Coeficientes CN
s = 1j * hbar * h / (4 * m * a**2)

a1 = 1 + 2*s
a2 = -s

b1 = 1 - 2*s
b2 = s


# Matriz A (banded)
A = np.zeros((3, m_in), dtype=complex)

for i in range(m_in):
    A[1,i] = a1
    if i > 0:
        A[2,i-1] = a2
    if i < m_in-1:
        A[0,i+1] = a2


# Funcion que hace un paso temporal 
# construir el lado derecho: v = B psi
v = np.zeros(m_in, dtype=complex)

def paso_psi(psi):
    psi_in = psi[1:-1]

    v = b1 * psi_in.copy()
    v[1:] += b2 * psi_in[:-1]
    v[:-1] += b2 * psi_in[1:]

    psi_new = banded(A, v, 1, 1)

    psi_next = np.zeros_like(psi, dtype=complex)
    psi_next[1:-1] = psi_new

    return psi_next


pasos = 2000
num = 1


psi_sol = np.zeros([N, pasos], dtype= complex)
psi_sol[:,0] = psi



# Actualizamos la cantidad de pasos
while num < pasos:
    psi_sol[:, num] = paso_psi(psi_sol[:, num - 1])
    num += 1

# Modulo de la funcion
modpsi = np.absolute(psi_sol)


# Figura
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
line, = ax.plot(x, np.real(psi), np.imag(psi), lw=2)

# ax.set_xlim(0, L)
# ax.set_ylim(-1, 1)
# ax.set_xlabel("x")
# ax.set_ylabel(r"Re($\psi$)")
ax.set_title("Evolución temporal (Crank-Nicolson)")
ax.set_xlabel('x (m)')
ax.set_ylabel(r'Re{$\psi$}')
ax.set_zlabel(r'Im{$\psi$}')
ax.grid(False)

# line, = ax.plot(x, np.real(psi_sol[:, 0]), np.imag(psi_sol[:, 0]),
#                 color='cyan', lw=1.5)

ax.set_xlim(x.min(), x.max())
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# ax.xaxis.pane.fill = False
# ax.yaxis.pane.fill = False
# ax.zaxis.pane.fill = False



# Función de animación
def update(frame):
    line.set_data_3d(x, np.real(psi_sol[:, frame]), np.imag(psi_sol[:,frame]))
    # ax.view_init(elev=30, azim=frame * 0.5)

    return line,

# Animación
ani = FuncAnimation(fig,update,frames=pasos,interval=10,blit=True)
plt.show()