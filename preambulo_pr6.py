import numpy as np
import matplotlib.pyplot as plt
import random as rd
from matplotlib.collections import LineCollection


# Parametros
N = 1000
nw = 100

dx = 0.1
dy = 0.1

xs = np.zeros([nw,N])

for i in range(1,N):
    for j in range(nw):

        dadox, dadoy = rd.random(), rd.random()
        
        if dadox > 0.5:
            xs[j,i] = xs[j,i-1] + dx
        else:
            xs[j,i] = xs[j,i-1] - dx



# Calculamos la suma cuadratica sobre en la dimension que recorre los pasos
x2 = np.sum(xs**2,0)

# Lista con los valores de 0 a N
n = np.arange(0,N)

# Promedio 
x2n = x2 / nw



# Representación del desplazamiento cuadratico promedio
figura, eje = plt.subplots(figsize = (6,6))
eje.plot(n, x2n)


eje.set_xlabel('N pasos')
eje.set_ylabel(r'$\langle x^2 _n \rangle _m$')
eje.set_title('Distancia cuarática promedio')

# Representacion de la trayectoria
fig, ax = plt.subplots(figsize = (6,6))
ax.set_xlabel('Número de pasos')
ax.set_ylabel('Distancia x')
ax.set_title(f'{nw} caminantes aleatorios')

for i in range(nw):
    ax.plot(n, xs[i,:], label = f'{i+1}')

# Si no hay muchos caminantes muestra la leyenda
if nw <8:
    ax.legend()


plt.show()