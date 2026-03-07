# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 10:29:35 2026

@author: physicist
"""

from math import pi, sqrt
from numpy import zeros
from vpython import sphere, rate, color, canvas, vector

# Definición de parámetros de la simulación
GMSol = 4 * pi**2        # G * Masa del Sol en Unidades Astronómicas
dt = 0.002              # Paso equivalente a menos de un día: 0.0027
T = 76                  # Periodo en AU (1 año)
tiempoTotal = 1 * T     # Tiempo total del cálculo
NN = tiempoTotal / dt   # Número de pasos
N = int(NN)


# Creación de vectores de posición y velocidad
x =  zeros(N, float)
y =  zeros(N, float)
r =  zeros(N, float)    # Radio vector del planeta
vx = zeros(N, float)
vy = zeros(N, float)
t  = zeros(N, float)

# Condiciones Iniciales expresadas en unidades astronómicas y años.
t[0] = 0
x[0] = 0.59
y[0] = 0
r[0] = sqrt(x[0]**2 + y[0]**2)
vx[0] = 0
vy[0] = 2.2 * pi

# Bucle principal utilizando método de Euler-Cromer
for i in range(N-1):
    vx[i+1] = vx[i] - GMSol * (x[i]/ r[i]**3) * dt
    vy[i+1] = vy[i] - GMSol * (y[i]/ r[i]**3) * dt
    x[i+1]  = x[i] + vx[i+1] * dt
    y[i+1]  = y[i] + vy[i+1] * dt
    r[i+1] = sqrt(x[i+1]**2 + y[i+1]**2)
    t[i+1] = t[i] + dt

#Representación del movimiento planetario.
canvas(title = "Órbita de la Tierra", x = 500, y = 100, width = 700, height = 700, center = vector(0, 0, 0.4), 
        forward = vector(0, 0, -1), background = vector(0, 0, 0))
        
sol = sphere(pos = vector(0.0, 0, 0), color = color.yellow, radius = 0.2)
haley = sphere(pos = vector(x[0], y[0], 0), color = color.cyan, radius = 0.05)

#Bucle que representa el movimiento y nos proporciona el tiempo transcurrido.
for i in range(N-1):
    rate(50)
    px = x[i]
    py = y[i]
    sphere(pos = vector(px, py, 0), color = color.red, radius = 0.01)
    haley.pos = vector(px, py, 0)
    print(t[i])
    
