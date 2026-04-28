import numpy as np
import matplotlib.pyplot as plt

from funciones_t10 import laplaciano_1d

# parametros fisicos y computacionales
L = 20.0
D = 0.1
tau = 365.0

A = 10.0
B = 12.0

N = 100
h = L / N
dt = 0.1

# tiempo
t0 = 0.0
tf = 10 * tau
tt = np.arange(t0, tf + dt, dt)
nt = len(tt)

# Inicializamos
T = np.ones((N+1, nt)) * 10.0
T[-1, :] = 11.0  # fondo fijo todo el rato

# bucle para resolver 
for i in range(1, nt):
    T_prev = T[:, i-1]

    # SOLO interior (clave)
    lap = laplaciano_1d(T_prev, h)
    T[1:-1, i] = T_prev[1:-1] + dt * D * lap[1:-1]

    # condiciones de contorno
    T[0, i]  = A + B*np.sin(2*np.pi*tt[i]/tau)
    T[-1, i] = 11.0

# tiempos del ultimo año
t1 = 9*tau
t2 = 9*tau + tau/4
t3 = 9*tau + tau/2
t4 = 9*tau + 3*tau/4

i1 = np.argmin(np.abs(tt - t1))
i2 = np.argmin(np.abs(tt - t2))
i3 = np.argmin(np.abs(tt - t3))
i4 = np.argmin(np.abs(tt - t4))

T1 = T[:, i1]
T2 = T[:, i2]
T3 = T[:, i3]
T4 = T[:, i4]

# graficamos 
x = np.linspace(0, L, N+1)

plt.plot(x, T1, label="Primavera")
plt.plot(x, T2, label="Verano")
plt.plot(x, T3, label="Otoño")
plt.plot(x, T4, label="Invierno")

plt.xlabel("Profundidad (m)")
plt.ylabel("Temperatura (°C)")
plt.title("Temperatura en la corteza terrestre")
plt.legend()
plt.grid()
plt.show()