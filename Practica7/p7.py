import numpy as np
import matplotlib.pyplot as plt
from numba import njit

'''
Resolucion de la practica base. Ciclo de histeresis calculado con distintos pasos
'''

# Semilla de números aleatorios fija
# np.random.seed(8)

# Parámetros físicos y computacionales
N = 8
J = 1.0
kB = 1.0
mu = 1.0
T = 0.25  

# Configuración del campo  
H_up = np.arange(-10, 11, 0.5)
H_down = np.arange(10, -11, -0.5)
HH = np.concatenate([H_up, H_down])

# Lista de cantidad de pasos
pasos_mc = [1, 10, 100, 1000, 10000]

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

# Magnetización por espín
@njit
def magnetizacion(espin):
    return np.sum(espin) / espin.size


# Simulacion
plt.figure(figsize=(10, 6))

for pasos in pasos_mc:
    # Inicializamos con saturación positiva cada ciclo
    espin = np.ones((N, N))
    
    M_T = []

    for h in HH:
        M_acc = 0
        
        # Evolución y promediado para cada valor de H
        for _ in range(pasos):
            barrido(espin, h)
            M_acc += magnetizacion(espin)
        
        # Guardamos el promedio de magnetización para este campo H
        M_T.append(M_acc / pasos)

    plt.plot(HH, M_T, '-o', label=f'Pasos = {pasos}')



plt.xlabel("Campo H")
plt.ylabel("Magnetización ")
plt.title(f'Histeresis en función del tiempo')
plt.legend()
plt.grid()

plt.savefig("Pasos.png", dpi = 500)
plt.show()