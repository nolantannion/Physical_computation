import numpy as np
import matplotlib.pyplot as plt 
from math import e

# Constantes 
atau = 1
btau = [0.1,1,e,10]

# Creamos un array que pueda contener otros arrays con distintos tamaños
tsol = []
nasol = []
nbsol = []

# Intervalo temporal (en segundos)

t0 = 0 
tf = 20
dt = 1e-3

na0 = 1
nb0 = 1

# Función que define la dinámica del sistema (susceptible a modificación)
def sistema(na,nb, atau, btau):
    
    dna = -na/atau
    dnb = na/atau - nb/btau
    
    return dna, dnb

# Función que aplica euler
def euler(dt,t0,tf,na0, nb0, atau, btau):  
    N = int((tf - t0)/dt)
    t = np.linspace(t0, tf, N+1)
    na = np.zeros(N+1)
    nb = np.zeros(N+1)
    
    na[0] = na0
    nb[0] = nb0
    
    for n in range(0,N):
        
        tem = sistema(na[n],nb[n], atau, btau)
        na[n+1] = na[n] + tem[0]*dt
        nb[n+1] = nb[n] + tem[1]*dt
        
    return t, na, nb



# Calculamos las solucionesy representamos
fig, ax = plt.subplots(nrows=1, ncols=2)    
for i,bt in enumerate(btau):
    
    t, na, nb = euler(dt,t0,tf,na0,nb0, atau, bt)
    
    tsol.append(t)
    nasol.append(na)
    nbsol.append(nb)
    
    # Representación de la trayectoria frente al tiempo
    
    ax[1].plot(t,nb, label = rf'$\tau _B$: {bt:.1f}')

ax[0].set_title('Na frente al tiempo')
ax[1].set_title('Nb frente al tiempo')

ax[0].plot(t,na, label = rf'$\tau _A: {atau}$')

# Ajustes de la gráfica 
for eje in ax:
    
    eje.set_xlabel('Tiempo (s)')
    eje.set_ylabel('N ')
    eje.axis('tight')
    eje.legend()
    eje.grid()

plt.show()




