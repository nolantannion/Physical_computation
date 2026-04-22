import numpy as np
import matplotlib.pyplot as plt
from numba import njit

'''
Calculo de la susceptibilidad y capacidad calorifica en una red de 3 dimensiones
'''

# Parámetros
N = 6
J = 1.0
kB = 1.0
mu = 1.0


# Funciones
def energia_3D(espin, H):
    vecinos = (
        np.roll(espin, 1, axis=0) + np.roll(espin, -1, axis=0) +
        np.roll(espin, 1, axis=1) + np.roll(espin, -1, axis=1) +
        np.roll(espin, 1, axis=2) + np.roll(espin, -1, axis=2)
    )
    return -0.5 * J * np.sum(espin * vecinos) - mu * H * np.sum(espin)

@njit
def delta_E_3D(espin, i, j, k, H):
    N = espin.shape[0]
    s = espin[i, j, k]

    vecinos = (
        espin[(i+1)%N, j, k] + espin[(i-1)%N, j, k] +
        espin[i, (j+1)%N, k] + espin[i, (j-1)%N, k] +
        espin[i, j, (k+1)%N] + espin[i, j, (k-1)%N]
    )

    return 2 * s * (J * vecinos + mu * H)

# Barrido Monte Carlo
@njit
def barrido_3D(espin, H, T):
    N = espin.shape[0]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                dE = delta_E_3D(espin, i, j, k, H)
                if dE < 0 or np.random.random() < np.exp(-dE/(kB*T)):
                    espin[i, j, k] *= -1

# Magnetización por espín
@njit
def magnetizacion_3D(espin):
    return np.sum(espin) / espin.size

# Media móvil
def media_movil(x, w):
    return np.convolve(x, np.ones(w)/w, mode='same')

# ------------------ SIMULACIÓN ------------------

H = 0.0
TT = np.linspace(3.5, 6.0, 60)   # alrededor de Tc ~ 4.51

n_est = 5*N**3
n_med = 10*N**3

Chi_T = []
Cv_T  = []

for T in TT:

    espin = np.random.choice([-1.0, 1.0], size=(N, N, N))

    # --- termalización ---
    for _ in range(n_est):
        barrido_3D(espin, H, T)

    # --- medidas ---
    M_acc = 0.0
    M2_acc = 0.0
    E_acc = 0.0
    E2_acc = 0.0

    for _ in range(n_med):

        for _ in range(2):
            barrido_3D(espin, H, T)

        M = magnetizacion_3D(espin)
        M_abs = abs(M)

        E = energia_3D(espin, H) / (N**3)

        M_acc  += M_abs
        M2_acc += M*M
        E_acc  += E
        E2_acc += E*E

    # promedios
    M_pr  = M_acc / n_med
    M2_pr = M2_acc / n_med
    E_pr = E_acc / n_med
    E2_pr = E2_acc / n_med

    # Cantidades por espin
    chi = (1/(kB*T)) * (M2_pr - M_pr**2)
    cv  = (1/(kB*T**2)) * (E2_pr - E_pr**2)

    Chi_T.append(chi)
    Cv_T.append(cv)

# Suavizado
v = 7
Susc = media_movil(Chi_T, w=v)
Calorv = media_movil(Cv_T, w=v)

Tc = 4.51

# ------------------ GRÁFICAS ------------------

fig, ax = plt.subplots(1,2, figsize=(10,4))

ax[0].plot(TT, Susc, '-', label='Susceptibilidad')
ax[0].set_xlabel("T")
ax[0].set_ylabel("χ")
ax[0].set_title("Susceptibilidad")

ax[1].plot(TT, Calorv, '-', label='Calor específico')
ax[1].set_xlabel("T")
ax[1].set_ylabel("C")
ax[1].set_title("Calor Específico")

for a in ax:
    a.grid()
    a.vlines(x=Tc, ymin=0, ymax=max(max(Susc), max(Calorv)), color='red', label=r'$T_c$')
    a.legend()

fig.savefig('Susc3D.png', dpi = 500)
plt.show()