import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

# Inicialización de parámetros y listas
g = 9.8
t0, tf = 0, 10
dt = [10,1,0.1,1e-2,1e-4]


#Definimos una función que aplica el método de euler con numba
@jit(nopython = True)
def eulerf(dt):
    N = int((tf - t0)/dt)
    t = np.linspace(t0, tf, N+1)
    v = np.zeros(N+1)
    
    for n in range(0,N):
        v[n+1] = v[n] - g*dt
    
    return t, v

#Definimos una función que aplica el método de euler sin numba
def euler(dt):  
    N = int((tf - t0)/dt)
    t = np.linspace(t0, tf, N+1)
    v = np.zeros(N+1)
    
    for n in range(0,N):
        v[n+1] = v[n] - g*dt
    
    return t, v


def solucion_exacta(t):
    
    return -g*t


for dtt in dt:

    sol = euler(dtt)
    sol_ex = solucion_exacta(sol[0])
    plt.scatter(sol[0],sol[1], alpha = 0.5, label= f'dt: {dtt}')
    plt.scatter(sol[0], sol_ex, label = 'Solución analítica')
    
    plt.title('Caída libre')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad (m/s)')
    plt.legend()
    plt.grid()
    plt.show()
    
    print(f'dt: {dtt}')
    print(f'Tiempos: {sol[0]}')
    print(f'Velocidades: {sol[1]} \n')

    


    



