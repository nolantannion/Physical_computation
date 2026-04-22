import numpy as np 
import matplotlib.pyplot as plt
from math import gamma

prec = 1e-4

# Limites y dimensiones del hipervolumen
R = 2
D = 12



# Funcion que calcula el volumen en D dimensiones dado el radio y la tolerancia
def calcular_vD(R,D, tol):

    print(f'\n{D} Dimensiones')

    # Volumen teórico
    vt = (np.pi**(D/2) / gamma(D/2 + 1)) * R**D
    print(f'Volumen teorico: {vt:.5f}')

    Hv = (2*R)**D   # Hiper volumen del recinto

    err = 10*tol    # valor inicial del error mayor que la tolerancia

    # Puntos totales y en el interior
    N = 0
    Nd = 0

    while err > tol:
        N += 1

        p = 2*R*np.random.random(D) - R # puntos aleatorios entre R y -R en cada dimension

        r2 = np.sum(p**2)   # radio cuadratica de nuestro punto aleatorio

        if r2 < R**2:   # condicion de pertenencia al interior
            Nd += 1
        
        V = Hv * Nd/N   # Volumen en esta iteracion

        err = np.abs(V-vt)/vt  # Calcular el nuevo error relativo

    print(f'Volumen calculado: {V:.5f}')
    print(f'Puntos totales: {N} \nPuntos dentro: {Nd}')
    print(f'Valores pseudoaleatorios generados: {N*D}')


for i in range(2, D+1):
    calcular_vD(R = R,D = i,tol=prec)