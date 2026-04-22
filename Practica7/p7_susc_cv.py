import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from time  import time 
from numba import  njit

'''
Calculo de la susceptibilidad y calor especifico usando las variaciones de otras
magnitudes obtenidas mediante el algoritmo de metropolis
'''
# np.random.seed(7)

# Parámetros fisicos y computacionales
N = 16
J = 1.0
kB = 1.0
mu = 1.0

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



# Susceptividad y calor especifico
# Definimos una media movil para reducir el ruido de las fluctuaciones
def media_movil(x, w):
    return np.convolve(x, np.ones(w)/w, mode='same')

# Campo nulo
H = 0.0

# Rango de temperaturas (cerca de Tc - 2.27 en 2D)
TT = np.linspace(1.4, 3.5, 60)

# Parámetros Monte Carlo
n_est = 5*N*N       # estabilizacion
n_med  = 12*N*N       # medidas

# Arrays con el calor y la susceptibilidad para cada temperatura
Chi_T = []
Cv_T  = []

for T in TT:

    # estado inicial
    espin = np.random.choice([-1.0, 1.0], size=(N, N))

    # Estabilizacion
    for _ in range(n_est):
        barrido(espin, H, T)

    # Medidas 
    M_acc = 0.0
    E_acc = 0.0
    M2_acc = 0.0
    E2_acc = 0.0

    for _ in range(n_med):

        # decorrelación
        for _ in range(2):
            barrido(espin, H, T)

        M = magnetizacion(espin)
        E = energia(espin, H) / N**2

        # Usamos el valor absoluto en M por la simetria en H = 0
        M_acc  += abs(M)
        M2_acc += M*M
        E_acc  += E
        E2_acc += E*E

    # promedios
    M_pr  = M_acc / n_med
    M2_pr = M2_acc / n_med
    E_pr  = E_acc / n_med
    E2_pr = E2_acc / n_med

    # Magnitudes
    chi = (1/ (kB * T)) * (M2_pr - M_pr**2)
    cv  = (1.0 / (kB * T**2)) * (E2_pr - E_pr**2)

    Chi_T.append(chi)
    Cv_T.append(cv)

# Calculamos la media movil sobre los datos
v = 5
Susc = media_movil(Chi_T, w=v) 
Calorv = media_movil(Cv_T, w= v)


Tc = 2.269 # Temperatura critica del modelo de ising 2d

# Graficas

fig, ax = plt.subplots(1,2, figsize=(10,4))

ax[0].plot(TT, Susc, '-', label = 'Suceptibilidad')
ax[0].set_xlabel("T")
ax[0].set_ylabel("χ")
ax[0].set_title("Susceptibilidad")


ax[1].plot(TT, Calorv, '-', label = 'Calor especifico')
ax[1].set_xlabel("T")
ax[1].set_ylabel("C")
ax[1].set_title("Calor específico")



for a in ax:
    a.axis('tight')
    a.grid()
    a.vlines(x = Tc, ymin = 0, ymax = 1e3, color = 'red', label = r'$T_c$')
    a.legend()


fig.savefig('Susceptibilidad.png', dpi = 500)


plt.show()