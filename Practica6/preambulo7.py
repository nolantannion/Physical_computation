from numpy import zeros, exp
import numpy as np
from random import random
import matplotlib.pyplot as plt

# Parámetros
N = 7
J = 1.0
kB = 1.0

temps = np.linspace(1.0, 10.0, 50)
n_pasos = 500

np.random.seed(8)

N = 7
espin = np.random.choice([-1, 1], size=(N, N))
estado0 = espin.copy()

# Energía condiciones periódicas usando roll
def energia(espin):
    vecinos = (
        np.roll(espin, 1, axis=0) +
        np.roll(espin, -1, axis=0) +
        np.roll(espin, 1, axis=1) +
        np.roll(espin, -1, axis=1)
    )

    E = -0.5 * J * np.sum(espin * vecinos) / N**2
    return E

# Funcion para calcular la variacion de energia
def delta_E(espin, i, j):
    N = len(espin)
    s = espin[i, j]
    vecinos = (
        espin[(i+1)%N, j] +
        espin[(i-1)%N, j] +
        espin[i, (j+1)%N] +
        espin[i, (j-1)%N]
    )
    return 2 * J * s * vecinos

# Funcion para actualizar segun temperatura y cambio de energia 
def barrido(espin, T):
    N = len(espin)
    for i in range(N):
        for j in range(N):
            dE = delta_E(espin, i, j)
            if dE < 0 or random() < exp(-dE/(kB*T)):
                espin[i, j] *= -1

# Magnetización
def magnetizacion(espin):
    return np.sum(espin) / espin.size

# Simulación
E_T, M_T = [], []



for i,T in enumerate(temps):
    
    E_acc, M_acc = 0, 0
    for _ in range(n_pasos):
        barrido(espin, T)
        E_acc += energia(espin)
        M_acc += magnetizacion(espin)

    E_T.append(E_acc / n_pasos)
    M_T.append(M_acc / n_pasos)

# Gráficas
plt.plot(temps, E_T, '-o')
plt.xlabel("T"); plt.ylabel("E")
plt.title('Energía')
plt.figure()
plt.plot(temps, M_T, '-o')
plt.xlabel("T"); plt.ylabel("M")
plt.title('Magnetización')
plt.show()


# Figura que compara el estado inicial y el final a una T
fig, ax = plt.subplots(nrows= 1, ncols= 2, figsize = (8,6))
ax[0].imshow(estado0)
ax[0].set_title('Estado inicial')
ax[1].set_title(f'Estado final: T = {temps[-1]}')
# Segunda grafica del espin
ax[1].imshow(espin)

plt.show()