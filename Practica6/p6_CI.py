import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

'''
Script que calcula la evolucion temporal de la difusion en 1D ante condiciones de contorno inhomogeneas
utilizando un ''flujo'' de caminantes aleatorios y resolviendo discretizando con diferencias finitas.

Se compara con la solucion estacionaria de la difusion
'''

# Parametros 
D = 5
dx = 0.5
Nx = 100

dt = dx**2 / (2*D)

nw = 2000
tf = 200
Nt = int(tf/dt)

# Posiciones de los caminantes, todos en 0
pos = np.zeros(nw)

# Arrays con el tiempo y las temperaturas que vamos a seleccionar
t = np.arange(0, tf, dt)
T = np.zeros((int(Nx/4), Nt))


# Array para diferencias finitas 
Td = np.zeros((Nx, Nt))

# condiciones y de contorno 
Td[0, :] = 100
Td[-1, :] = 0


# Rango y array ''temporal'' para los promedios
rango = 10
temp = np.zeros((int(Nx/4), rango))

# --- SIMULACIÓN ---
for i in range(1,Nt):

    # Movimiento +-1
    mov = np.random.choice([-1, 1], size=pos.size)
    pos = pos + mov

    #Eliminamos fuera del borde derecho 
    inside = pos < Nx
    pos = pos[inside]

    # Impedimos la salida por el borde izquierdo 
    fuera = np.where(pos < 0)
    pos[fuera] = 1

    # Metemos de nuevo caminantes en 0 
    nn = nw - pos.size
    if nn > 0:
        new_walkers = np.zeros(nn, dtype=int)
        pos = np.concatenate([pos, new_walkers])

    # ''Temperatura''
    h, _ = np.histogram(pos, bins=int(Nx/4),  range=(0, Nx))


    # T normalizada a 100
    T_inst = h / np.max(h) * 100

    # Guardamos en array temporal
    temp[:, i % rango] = T_inst

    # Promedio temporal cuando hay suficientes datos
    if i >= rango:
        T[:, i] = np.mean(temp, axis=1)

    # 4 primeros pasos
    else:
        T[:, i] = T_inst


    # Diferencias finitas
    for j in range(1, Nx-1):
        Td[j, i] = (
            Td[j, i-1]
            + D * dt / dx**2 * (
                Td[j+1, i-1]
                - 2*Td[j, i-1]
                + Td[j-1, i-1]
            )
        )

    # mantenemos contornos
    Td[0, i] = 100
    Td[-1, i] = 0



# --- ANIMACIÓN ---
fig, ax = plt.subplots()

x = np.arange(0, Nx, 4)
xf = np.arange(0, Nx, 1)

T0 = 100
Tl = 0
Ta = T0 + (Tl - T0) * x / Nx

linea, = ax.plot([], [], label='Temperatura')
ax.plot(x, Ta, 'k--', label='Solución estacionaria')

ax.set_xlabel('x')
ax.set_ylabel('T')
ax.set_title('Difusión con frontera inhomogenea')


linea2, = ax.plot([], [], label='Dif. finitas')



def update(frame):
    linea.set_data(x, T[:, frame])
    linea2.set_data(xf, Td[:, frame])
    return (linea, linea2)

anim = FuncAnimation(fig, update, frames=np.arange(0,Nt,2), interval=5, blit=True)

ax.legend()
plt.show()


foto, ejef  = plt.subplots()
ejef.plot(x, Ta, label = 'Solución estacionaria')
ejef.plot(x, T[:,-1], label = 'Caminantes aleatorios')
ejef.plot(xf,Td[:,-1], label = 'Diferencias finitas')

ejef.legend()
ejef.set_xlabel('x')
ejef.set_ylabel('T')
ejef.set_title('Difusión con frontera inhomogenea')

foto.savefig('Dif_i.png', dpi = 500)





