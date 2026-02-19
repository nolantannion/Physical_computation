# Práctica 3
'''
En esta ampliación se estudia el efecto de considerar el drag 
únicamente en el eje x de manera simplificada y considerarlo en todas las direcciones
comprobando como afecta a la trayectoria y a la energía
'''

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from funciones_pr3 import rho_const, euler, arrastre_completo, arrastrex


# Constantes y arrays
v0 = 70 # Velocidad incial (m/s)
S0wm = 0.25 # Parámetro del efecto Magnus (s^-1)
rho0 = 1.27 # Densidad del aire a temperatura ambiente y nivel del mar (kg/m^3)
m = 0.04593 # Masa maxima de una pelota de golf (kg)
r_bola = 0.0213 # Radio de una bola de golf (m)
A = np.pi* r_bola**2 # Área frontal de una pelota de golf
g = 9.81 # Aceleración de la gravedad (m/s^2)


# Integración
t0 = 0 
h = 1e-3
tmax = 100


# Función que define la dinámica del sistema
# vars tiene la estructura [x,vx,y,vy,z,vz]
# Usaremos la aproximación isoterma al ser y<<10 km
def sistema_completo(vars, rho, sign):
    
    # Creamos el array que almacena las derivadas de cada variable
    dvars = np.zeros_like(vars)
    
    # Calculamos el módulo de la velocidad
    modv = np.sqrt(vars[1]**2 + vars[3]**2 + vars[5]**2)
    
    C = arrastre_completo(var=vars)
    # Calculamos la derivada de cada variable
    dvars[0] = vars[1]
    dvars[2] = vars[3]
    dvars[4] = vars[5]
    
    # No incluimos el arrastre en x e y
    arrastre = -(1/2)*rho*A/m *modv 
    dvars[1] = arrastre*vars[1]*C[0]
    dvars[3] = -g + arrastre*vars[3]*C[1]
    dvars[5] = S0wm*vars[1] * sign + arrastre*vars[5]*C[2]
    
    
    return dvars


def sistema_x(vars, rho, sign):
    
    # Creamos el array que almacena las derivadas de cada variable
    dvars = np.zeros_like(vars)
    
    # Calculamos el módulo de la velocidad
    modv = np.sqrt(vars[1]**2 + vars[3]**2 + vars[5]**2)
    
    C = arrastrex(var=vars)
    # Calculamos la derivada de cada variable
    dvars[0] = vars[1]
    dvars[2] = vars[3]
    dvars[4] = vars[5]
    
    # No incluimos el arrastre en x e y
    arrastre = -(1/2)*rho*A/m *modv 
    dvars[1] = arrastre*vars[1]*C[0]
    dvars[3] = -g + arrastre*vars[3]*C[1]
    dvars[5] = S0wm*vars[1] * sign + arrastre*vars[5]*C[2]
    
    
    return dvars

# Parámetro de rotación y leyenda
spin = [1,-1]
leyendas = ['Hook','Slice']

# Dirección del golpeo de la bola
ang = np.deg2rad(15)
direccion = [np.cos(ang),np.sin(ang), 0]

# Estado inicial
estado = [0, v0*np.cos(ang),0,v0*np.sin(ang),0,0]

# Figura con los ejes en 3D
fig, ax = plt.subplots(subplot_kw= dict(projection = '3d'), figsize = (8,6))

# Cálculo y representación de la trayectoria para cada dirección de giro y cada modelado del rozamiento
for i, sign in enumerate(spin):

    t, sol = euler(sistema_x,t0,tmax, h, estado, sign, rho_const)
    trk, solrk = euler(sistema_completo,t0,tmax, h, estado, sign, rho_const)

    # Estudiamos como afecta el tipo de integrador que usemos en la precisión
    ax.plot(sol[:,0], sol[:,4], sol[:,2], label = f'Arrastre en x,{leyendas[i]}') 
    ax.plot(solrk[:,0], solrk[:,4], solrk[:,2], label = f'Arraste completo, {leyendas[i]}') 

    # Calculamos la distancia entre puntos de aterrizaje
    diff  = abs(np.sqrt(sol[-1,0]**2 + sol[-1,2]**2 + sol[-1,4]**2) - np.sqrt(solrk[-1,0]**2 + solrk[-1,2]**2 + solrk[-1,4]**2))
    print('Distancia entre aterrizajes: ',diff)


    print(f'Rozamiento en x {leyendas[i]}') 
    print(f'Desplazamiento maximo en |x|: {max(abs(sol[:,0]))}')
    print(f'Desplazamiento maximo en |y|: {max(abs(sol[:,2]))}')
    print(f'Desplazamiento maximo en |z|: {max(abs(sol[:,4]))} \n')

    print(f'Rozamiento completo {leyendas[i]}') 
    print(f'Desplazamiento maximo en |x|: {max(abs(solrk[:,0]))}')
    print(f'Desplazamiento maximo en |y|: {max(abs(solrk[:,2]))}')
    print(f'Desplazamiento maximo en |z|: {max(abs(solrk[:,4]))} \n')


fig.legend()
ax.set_xlabel('Eje x (m)')
ax.set_ylabel('Eje z (m)')
ax.set_zlabel('Eje y (m)')
ax.set_title('Rozamientos')


# Calculamos la energía para el slice y la representamos aprovechando que son movimientos simétricos
figura, eje  = plt.subplots( figsize = (8,6))
v2 = sol[:,1]**2 + sol[:,3]**2 + sol[:,5]**2

Ep = m*g*sol[:,2]
Ec = (1/2)*m*v2
ET = Ec + Ep

v2rk = solrk[:,1]**2 + solrk[:,3]**2 + solrk[:,5]**2

Eprk = m*g*solrk[:,2]
Ecrk = (1/2)*m*v2rk
ETrk = Ecrk + Eprk

eje.plot(t,ET, label = 'Arrastre en x')
eje.plot(trk,ETrk,  label = 'Arrastre Completo')
eje.set_xlabel('tiempo (s)')
eje.set_ylabel('Energía (J)')
figura.legend()
figura.suptitle('Comparación de variación de energía')
plt.grid()
fig.savefig('Arrastres_3d.png', dpi = 300)
figura.savefig('Comparacion_energia.png', dpi = 500)
plt.show()

print('Variacion de E: ',abs(ETrk[-1] - ET[-1]))
