import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve, factorized

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from numba import njit


# Parametros
hbar = 1.0546e-34
m = 9.109e-31

# Longitud de la caja
Lx, Ly = 1e-8, 1e-8
Nx, Ny = 200, 200

# Tamaño de la discretizacion
dx = Lx / Nx
dy = Ly / Ny

# Paso temporal
h = 2e-17

# Arrays sobre la longitud de cada lado
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)

# Malla del espacio
X, Y = np.meshgrid(x, y, indexing='ij')

# Condicion inicial 
sigma = 3e-10
psi = np.exp(-(X**2 + Y**2)/(2*sigma**2)).astype(complex)

# Condiciones de contorno
psi[0,:] = psi[-1,:] = 0
psi[:,0] = psi[:,-1] = 0


# Laplaciano 2D 
Ix = eye(Nx)
Iy = eye(Ny)

ex = np.ones(Nx)
Tx = diags([ex[:-1], -2*ex, ex[:-1]], offsets=[-1, 0, 1]) / dx**2

ey = np.ones(Ny)
Ty = diags([ey[:-1], -2*ey, ey[:-1]], offsets=[-1, 0, 1]) / dy**2

Laplacian = kron(Iy, Tx) + kron(Ty, Ix)

# Elementos totales
Ntot = Nx * Ny
I = eye(Ntot)


# Potencial
omega = 5e14
V = 0.5 * m * omega**2 * (X**2 + Y**2)

# Convertimos a una matriz diagonal
V_vec = V.reshape(Ntot)
V_mat = diags(V_vec)


# Coeficientes CN
alpha = 1j * hbar * h / (4 * m)


beta = 1j * h / (2 * hbar)

A = I + alpha * Laplacian + beta * V_mat
B = I - alpha * Laplacian - beta * V_mat

# Paso Crank–Nicolson
A = A.tocsc()
B = B.tocsr()

# Paso de Crank-Nicolson 
solve_A = factorized(A)


def paso_CN(psi):
    psi_vec = psi.ravel()
    v = B @ psi_vec
    psi_new = solve_A(v)
    psi_new = psi_new.reshape((Nx, Ny))

    psi_new[0,:] = psi_new[-1,:] = 0
    psi_new[:,0] = psi_new[:,-1] = 0

    return psi_new


# Un paso
pasos = 2500
psit = np.zeros([Nx,Ny,pasos], dtype= complex)
psit[:,:,0] = psi # Condicion incial
i = 1   # indice del paso temporal

# Calculamos la evolucion a lo largo del numero de pasos
while i < pasos:
    psit[:,:,i] = paso_CN(psit[:,:, i-1])
    i+= 1


# Figura
fig, ax = plt.subplots()

# Representamos como un mapa de calor
img = ax.imshow(np.abs(psi), extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(r"$|\psi(x,y,t)|$")

# Colorbar 
cbar = plt.colorbar(img, ax=ax)

# Funcion de actualizacion
def update(frame):

    img.set_data(np.abs(psit[:,:, frame]))
    return (img,)

# Ejecutamos la animacion
ani = FuncAnimation(fig, update, frames=range(0, pasos, 5), interval=10, blit=True)
plt.show()