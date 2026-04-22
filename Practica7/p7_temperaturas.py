import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from time  import time 
from numba import  njit

'''
Comparacion de los comportamientos de la magnetizacion y la energia tras la variacion de H a 
distintas temperaturas. Se usa njit para acelerar la ejecución
'''
# np.random.seed(7)

# Parámetros fisicos y computacionales
N = 8
J = 1.0
kB = 1.0
mu = 1.0

TT = np.array([0.1, 0.25, 1.3,5])

# Array de subida, bajada y la union
H_up = np.arange(-10, 11, 1)
H_down = np.arange(10, -11, -1)
HH = np.concatenate([H_up, H_down])

# Pasos de montecarlo en función de H
pasos_mont = 8*N**2

# Semilla de numeros aleatorios fija



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



# Figuras y ejes para las graficas
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

# Bucle que recorre las temperaturas definidas
for temperatura in TT:
    # Arrays de energias y magnetizaciones
    E_T, M_T = [], []

    # Recorremos los valores de H
    for i,h in enumerate(HH):

        # Fijamos a 0 los valores y barremos para medirlos en cada iteracion
        M_acc = 0
        E_acc = 0

        # Barremos para los pasos de montecarlo definidos para esa H
        for _ in range(pasos_mont):
            barrido(espin, h, temperatura)
            M_acc += magnetizacion(espin)
            E_acc += energia(espin, h)

        # Añadimos las magnitudes a los arrays
        M_T.append(M_acc / pasos_mont)
        E_T.append(E_acc / pasos_mont)

    # Gráficas
    ax1.plot(HH, M_T, '-o', label = f'T: {temperatura}')
    ax2.plot(HH, E_T, '-o', label = f'T: {temperatura}')

    # Separar ramas
    n = len(H_up)

    H_up_vals = HH[:n]
    M_up_vals = M_T[:n]

    H_down_vals = HH[n:]
    M_down_vals = M_T[n:]

    # Integrales (trapecios)
    A_up = trapezoid(M_up_vals, x=H_up_vals)
    A_down = trapezoid(M_down_vals, x=H_down_vals)

    # Área del ciclo (valor absoluto)
    A_histeresis = abs(A_up + A_down)


    print(f"T: {temperatura}; \t Integral cerrada: { A_histeresis} ")
print('El area de la integral decrece con T.')


ax1.set_xlabel("H")
ax1.set_ylabel("M")
ax1.set_title('Magnetización')
ax1.grid()

ax2.set_xlabel("H")
ax2.set_ylabel("E")
ax2.set_title('Energía')
ax2.grid()

ax1.legend()
ax2.legend()

# fig1.savefig('Magnetizacion', dpi = 500)
# fig2.savefig('Energias.png', dpi = 500)

plt.show()
