import numpy as np
import matplotlib.pyplot as plt
import random as rd
from matplotlib.animation import FuncAnimation

# Parametros
N = 1000 # numero de pasos
nw = 100 # Numero de caminantes
dr = 0.2  # Distancia de paso radial

dist = np.zeros([nw,N], dtype= float)
angulo = np.zeros([nw,N], dtype= float)

x = np.zeros_like(dist)
y = np.zeros_like(dist)


for i in range(N):
    for k in range(nw):
        angle = np.deg2rad(rd.randint(0,359)) # Angulo aleatorio en radianes
        dice = rd.random()
        if dice < 0.5:
            dist[k,i] = dist[k,i-1] + dr
            x[k,i] = x[k,i-1] + dr*np.cos(angle)
            y[k,i] = y[k,i-1] + dr*np.sin(angle)
        else:
            dist[k,i] = dist[k,i-1] - dr
            x[k,i] = x[k,i-1] - dr*np.cos(angle)
            y[k,i] = y[k,i-1] - dr*np.sin(angle)


# Calculamos los promedios
xp = np.sum(x,0)
yp = np.sum(y,0)

x2p = np.sum(x**2,0)/nw
y2p = np.sum(y**2,0)/nw

r2 = (x**2 + y**2)
rp = np.sum(np.sqrt(r2),0)
r2p = np.sum(r2,0)/nw

n = np.arange(0,N)

figura, eje = plt.subplots(nrows=1, ncols=2, figsize = (6,6))
eje[0].plot(n, x2p, label = r'$\sigma _x^2$')
eje[0].plot(n, y2p, label = r'$\sigma _y^2$')
eje[0].plot(n, r2p, label = r'$\sigma _r^2$')

eje[0].set_xlabel('Pasos')
eje[0].set_ylabel(r'$ \langle Dist^2 \rangle $')
eje[0].set_title('Desplazamiento cuadrático')
eje[0].legend()
eje[0].grid()

eje[1].plot(n, xp, label = r'$\sigma _x$')
eje[1].plot(n, yp, label = r'$\sigma _y$')
eje[1].plot(n, rp, label = r'$\sigma _r$')
eje[1].set_xlabel('Pasos')
eje[1].set_ylabel(r'$ \langle Dist \rangle $')
eje[1].set_title('Promedio')
eje[1].legend()
eje[1].grid()




figura1, eje1 = plt.subplots(figsize = (6,6))
for i in range(nw):
    eje1.plot(x[i,:], y[i,:])


eje1.set_xlabel('x')
eje1.set_ylabel(r'y')
eje1.set_title('Trayectoria')
eje1.legend()
eje1.grid()

plt.show()


