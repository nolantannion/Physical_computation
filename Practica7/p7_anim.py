import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from matplotlib.animation import FuncAnimation


'''
Graficas de la magnetizacion y eneergia, ademas de simulacion de la evolucion de los espines 
para una mayor temperatura que permite una mayor cantidad de saltos.
'''
# Semilla de numeros aleatorios fija
np.random.seed(8)

# Parámetros fisicos y computacionales
N = 8
J = 1.0
kB = 1.0
mu = 1.0

T = 10
print('Temperatura: ', T)

# Array de subida, bajada y la union
H_up = np.arange(-10, 11, 0.5)
H_down = np.arange(10, -11, -0.5)
HH = np.concatenate([H_up, H_down])

# Pasos de montecarlo en función de H
pasos_mont = 5* N**2




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
def delta_E(espin, i, j, H):
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
@njit()
def barrido(espin, H):
    N = len(espin)
    for i in range(N):
        for j in range(N):
            dE = delta_E(espin, i, j, H)
            if dE < 0 or np.random.random() < np.exp(-dE/(kB*T)):
                espin[i, j] *= -1

# Magnetización por espin
@njit
def magnetizacion(espin):
    return np.sum(espin) / espin.size


@njit
def delta_E_numba(espin, i, j, H, J, mu):
    N = espin.shape[0]
    s = espin[i, j]
    vecinos = (
        espin[(i+1) % N, j] +
        espin[(i-1) % N, j] +
        espin[i, (j+1) % N] +
        espin[i, (j-1) % N]
    )
    return 2.0 * s * (J * vecinos + mu * H)



# Generamos la malla de espines
espin = np.random.choice([-1.0, 1.0], size=(N, N))
estado0 = espin.copy()



# Simulación
E_T, M_T = [], []

Espines_T = [espin.copy()]

for i,h in enumerate(HH):

    # Medida
    M_acc = 0
    E_acc = 0

    for _ in range(pasos_mont):
        barrido(espin, h)
        M_acc += magnetizacion(espin)
        E_acc += energia(espin, h)
    

    M_T.append(M_acc / pasos_mont)
    E_T.append(E_acc / pasos_mont)

    Espines_T.append(espin.copy())

# Gráficas
plt.figure()
plt.plot(HH, M_T, '-o')
plt.xlabel("H")
plt.ylabel("M")
plt.title('Magnetización')
plt.grid()

plt.figure()
plt.plot(HH, E_T, '-o')
plt.xlabel("H")
plt.ylabel("E")
plt.title('Energía')
plt.grid()


# Simulacion
fig, ax = plt.subplots()

im = ax.imshow(Espines_T[0], cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)

def update(frame):
    im.set_array(Espines_T[frame])
    ax.set_title(f"H = {HH[frame]}")
    return [im]

ani = FuncAnimation(
    fig,
    update,
    frames=len(Espines_T),
    interval=100,
    blit=False
)

plt.show()

