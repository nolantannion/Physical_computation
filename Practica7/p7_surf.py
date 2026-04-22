import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from time  import time 
from numba import  njit
from mpl_toolkits.mplot3d import Axes3D 
'''
Representacion de la superficie M-H-T
'''

# Parámetros fisicos y computacionales
N = 8
J = 1.0
kB = 1.0
mu = 1.0

pasos = 5*N*N

##############################
# FUNCIONES
##############################

# Funcion para calcular la energia total
def energia(espin, H):
    vecinos = (
        np.roll(espin, 1, axis=0) +
        np.roll(espin, -1, axis=0) +
        np.roll(espin, 1, axis=1) +
        np.roll(espin, -1, axis=1)
    )
    return -0.5 * J * np.sum(espin * vecinos) - mu * H * np.sum(espin)


# Funcion para calcular la variacion de energia
@njit
def delta_E(espin, i, j, H, T):
    N = len(espin)
    s = espin[i, j]
    vecinos = (
        espin[(i+1)%N, j] +
        espin[(i-1)%N, j] +
        espin[i, (j+1)%N] +
        espin[i, (j-1)%N]
    )
    return 2 * s * (J * vecinos + mu * H)

# Funcion para actualizar segun temperatura y cambio de energia 
@njit
def barrido(espin, H, T):
    N = len(espin)
    for i in range(N):
        for j in range(N):
            dE = delta_E(espin, i, j, H, T)
            if dE < 0 or np.random.random() < np.exp(-dE/(kB*T)):
                espin[i, j] *= -1

# Magnetización por espin
@njit
def magnetizacion(espin):
    return np.sum(espin) / espin.size




# Generamos la malla de espines
espin = np.random.choice([-1.0, 1.0], size=(N, N))
estado0 = espin.copy()

TT = np.linspace(1.0, 3.0, 20)
Hu = np.linspace(-2, 2, 80)
Hd = np.linspace(2, -2, 80)

Md = np.zeros((len(TT), len(Hd)))
Mu = np.zeros((len(TT), len(Hu)))



for i, T in enumerate(TT):

    # subida
    for j, H in enumerate(Hu):
        for _ in range(pasos):
            barrido(espin, H, T)
        Mu[i, j] = magnetizacion(espin)

    # bajada 
    for j, H in enumerate(Hd):
        for _ in range(pasos):
            barrido(espin, H, T)
        Md[i, j] = magnetizacion(espin)


Xa, Y = np.meshgrid(Hu, TT)
Xd, _    = np.meshgrid(Hd, TT)

fig, ax = plt.subplots( subplot_kw= {'projection': '3d'})

ax.plot_surface(Xa, Y, Mu, alpha=0.7, label = 'Subida')
ax.plot_surface(Xd, Y, Md, alpha=0.7, label = 'Bajada')

ax.set_xlabel("H")
ax.set_ylabel("T")
ax.set_zlabel("M")

ax.view_init(elev = 30, azim = -110, roll = 0)

ax.legend()

fig.savefig('Superficie.png', dpi = 300)
plt.show()

