'''
Alba Ruesga Alonso, Sofia Martín Alañón, Nolan Tannion Rodríguez
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import trapezoid
from funciones_t11 import paso_psi


# Parametros y constantes

# Lado de la caja unidimensional
L = 1.0

# Discretizacion espacial
dx = 5e-4
N = int(L/dx)

# Discretizacion temporal
dt = 5e-7
h = dt 
pasos = 6000    # pasos temporales

# Constantes fisicas 
hbar = 1.0
m = 1.0

# Paquete Gaussiano 
x0 = 0.25 * L   # Posicion inicial
sigma = 0.03    # Anchura del paquete
k0 = 700    # Momento inicial

# Barrera V
xc = 0.6 * L    # Centro de la barrera
w = 0.05    # Anchura
V0 = 200*k0   # Altura del potencial (ligeramente inferior al recomendado con motivo de visualizacion)

# Malla espacial
x = np.linspace(0, L, N)

# Regiones del espacio
z_t = x > (xc)
z_r = x < (xc + w)

# Funcion de onda incial normalizada
psi = np.exp(-(x - x0)**2 / (2 * sigma**2))* np.exp(1j * k0 * x)
norm = np.sqrt(trapezoid(np.abs(psi)**2, x))
psi /= norm

# Condiciones de contorno
psi[0] = 0
psi[-1] = 0

# Potencial rectangular
V = np.zeros(N)

V[z_t & z_r] = V0

# Solo puntos interiores
V_in = V[1:-1]
m_in = N - 2


# Coeficientes, incorporando el potencial
s = 1j * hbar * dt / (4 * m * dx**2)
r = 1j * dt * V_in / (2 * hbar)

# Diagonales
a1 = 1 + 2*s + r
a2 = -s

b1 = 1 - 2*s - r
b2 = s

# Matriz A
A = np.zeros((3, m_in), dtype=complex)
A[1, :] = a1
A[0, 1:] = a2
A[2, :-1] = a2


# Evolucion temporal
psi_sol = np.zeros((N, pasos), dtype=complex)
psi_sol[:, 0] = psi

for n in range(1, pasos):
    psi_sol[:, n] = paso_psi(psi_sol[:, n-1], b1, b2, A)

# Modulo de la funcion en cada posicion y tiempo
modpsi = np.abs(psi_sol)



# Calculamos R y T
# Arrays para almacenar resultados
T = np.zeros(pasos)
R = np.zeros(pasos)
P_total = np.zeros(pasos)

# Regiones de transmision y reflexion
transm = x > (xc + w)
refl = x < xc

# Cálculo temporal
prob = np.zeros_like(psi_sol, dtype= float)
for n in range(pasos):

    # Densidad de probabilidad
    prob[:,n] = np.abs(psi_sol[:, n])**2

    # Transmitancia
    T[n] = trapezoid( prob[transm,n], x[transm])

    # Reflectancia
    R[n] = trapezoid(prob[refl,n], x[refl])

    # Probabilidad total
    P_total[n] = trapezoid(prob[:,n], x)

# Representacion de T y R
tiempo = np.arange(pasos) * dt

plt.figure(figsize=(8,5))

plt.plot(tiempo, T, label='Transmitancia')
plt.plot(tiempo, R, label='Reflectancia')
plt.plot(tiempo, T + R, '--', label='T + R')
plt.plot(tiempo, P_total, ':', label='Probabilidad total')

plt.xlabel('Tiempo')
plt.ylabel('Probabilidad')

plt.title('T(t) y R(t)')

plt.legend()
plt.grid()

plt.savefig('Evolucion de T y R')
plt.show()




# Animacion de la funcion de onda
fig, ax = plt.subplots()

linea, = ax.plot(x, prob[:, 0])
ax.vlines([xc,xc+w], 0, 100, color = 'r')

ax.set_xlim(0, L)
ax.set_ylim(0, np.max(prob)*1.1)

ax.set_xlabel("x")
ax.set_ylabel(r"$|\psi(x,t)|^2$")
ax.set_title("Evolución temporal")

def update(frame):
    linea.set_ydata(prob[:, frame])

    return linea,

anim = FuncAnimation( fig, update, frames=np.arange(0, pasos, 10), blit=True, interval = 10)

plt.show()