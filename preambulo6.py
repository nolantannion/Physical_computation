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
ys = np.zeros_like(xs)


for i in range(1,N):
    for j in range(nw):

        dadox, dadoy = rd.random(), rd.random()
        
        if dadox > 0.5:
            xs[j,i] = xs[j,i-1] + dx
        else:
            xs[j,i] = xs[j,i-1] - dx

        if dadoy > 0.5:
            ys[j,i] = ys[j,i-1] + dy
        else:
            ys[j,i] = ys[j,i-1] - dy


# Calculamos la suma cuadratica sobre en la dimension que recorre los pasos
x2 = np.sum(xs**2,0)
y2 = np.sum(ys**2,0)
rr = (x2+y2)/nw

x21 = np.sqrt(x2)/nw
y21 = np.sqrt(y2)/nw

n = np.arange(0,N)

figura, eje = plt.subplots(figsize = (6,6))
eje.plot(n, rr, label = r'$\sigma _{r2}$')
eje.plot(n, x2, label = r'$\sigma _{x2}$')
eje.plot(n, y2, label = r'$\sigma _{y2}$')

eje.legend()
eje.set_xlabel('N pasos')
eje.set_ylabel('Desviación')


fig, ax = plt.subplots(figsize=(6,6))


for i in range(nw):

    points = np.array([xs[i], ys[i]]).T
    segments = np.stack([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap='viridis')
    lc.set_array(np.arange(N-1))

    ax.add_collection(lc)

    ax.scatter(xs[i,0],ys[i,0])#, label =  'p0')
    ax.scatter(xs[i,-1],ys[i,-1])#, label = 'pf')

ax.autoscale(segments.all)
ax.grid()
ax.axis('tight')
ax.legend()
plt.show()