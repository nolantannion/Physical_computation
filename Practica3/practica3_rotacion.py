# Práctica 3
'''
En esta ampliación de la práctica 3 se estudia los cambios de la trayectoria
que aparecen en una rotación que no esté contenida en un único eje
 
'''
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from funciones_pr3 import rho_const, euler, arrastre_completo


# Constantes y arrays
v0 = 70 # Velocidad incial (m/s)
S0wm = 0.25 # Parámetro del efecto Magnus (s^-1)
rho0 = 1.27 # Densidad del aire a temperatura ambiente y nivel del mar (kg/m^3)
m = 0.04593 # Masa maxima de una pelota de golf (kg)
r_bola = 0.0213 # Radio de una bola de golf (m)
A = np.pi* r_bola**2 # Área frontal de una pelota de golf
g = 9.81 # Aceleración de la gravedad (m/s^2)

# Rotación en varias direcciones
rotacion = [S0wm* np.cos(45), S0wm*np.sin(45), 0]


# Integración
t0 = 0 
h = 1e-3
tmax = 100


# Función que define la dinámica del sistema
# vars tiene la estructura [x,vx,y,vy,z,vz]
# Usaremos la aproximación isoterma al ser y<<10 km
def sistema_cross(vars, rho, sign):
    
    # Creamos el array que almacena las derivadas de cada variable
    dvars = np.zeros_like(vars)
    
    # Calculamos el módulo de la velocidad
    modv = np.sqrt(vars[1]**2 + vars[3]**2 + vars[5]**2)
    vel = [vars[1],vars[3],vars[5]]
    
    C = arrastre_completo(var=vars)
    # No incluimos el arrastre en z e y
    arrastre = -(1/2)*rho*A/m *modv 

    Fmag = -np.cross(rotacion,vel) *sign
    # Calculamos la derivada de cada variable
    dvars[0] = vars[1]
    dvars[2] = vars[3]
    dvars[4] = vars[5]
    
    
    dvars[1] = arrastre*vars[1]*C[0] + Fmag[0] 
    dvars[3] = -g + arrastre*vars[3]*C[1] + Fmag[1]
    dvars[5] = arrastre*vars[5]*C[2] + Fmag[2]
    
    
    return dvars


def sistema(vars, rho, sign):
    
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




# Incluimos la dirección de giro
spin = [1,-1]
leyendas = ['Hook','Slice']

# Dirección del golpeo de la bola
ang = np.deg2rad(15)
direccion = [np.cos(ang),np.sin(ang), 0]

# Estado inicial
estado = [0, v0*np.cos(ang),0,v0*np.sin(ang),0,0]

# Figura con los ejes en 3D
fig, ax = plt.subplots(nrows= 1, ncols= 1, subplot_kw= dict(projection = '3d'), figsize = (8,6))
fig1, ax1 = plt.subplots(nrows= 1, ncols= 3, figsize = (10,6))

# Cálculo y representación de la trayectoria para cada dirección de giro
for i, sign in enumerate(spin):

    t, sol = euler(sistema,t0,tmax, h, estado, sign, rho_const)
    trk, solrk = euler(sistema_cross,t0,tmax, h, estado, sign, rho_const)

    # Estudiamos como afecta el tipo de integrador que usemos en la precisión
    ax.plot(sol[:,0], sol[:,4], sol[:,2], label = f'Ideal {leyendas[i]}') 
    ax.plot(solrk[:,0], solrk[:,4], solrk[:,2], label = f'Desviado {leyendas[i]}') 

    # Representamos la diferen
    ax1[0].plot(t,sol[:,0], label = f'Ideal {leyendas[i]}')
    ax1[0].plot(trk,solrk[:,0], label = f'Desviado {leyendas[i]}')

    ax1[1].plot(t,sol[:,2], label =f'Ideal {leyendas[i]}')
    ax1[1].plot(trk,solrk[:,2], label = f'Desviado {leyendas[i]}')

    ax1[2].plot(t,sol[:,4], label =f'Ideal {leyendas[i]}')
    ax1[2].plot(trk,solrk[:,4], label = f'Desviado {leyendas[i]}')

    ax1[0].set_xlabel('tiempo (s)')
    ax1[1].set_xlabel('tiempo (s)')
    ax1[2].set_xlabel('tiempo (s)')

    ax1[0].set_ylabel('x (m)')
    ax1[1].set_ylabel('y (m)')
    ax1[2].set_ylabel('z (m)')

    ax1[0].legend()
    ax1[1].legend()
    ax1[2].legend()

    # Calculamos la distancia entre puntos de aterrizaje
    diff  = abs(np.sqrt(sol[-1,0]**2 + sol[-1,2]**2 + sol[-1,4]**2) - np.sqrt(solrk[-1,0]**2 + solrk[-1,2]**2 + solrk[-1,4]**2))
    print('Distancia entre aterrizajes: ',diff)



    print(f'Ideal {leyendas[i]}') 
    print(f'Desplazamiento maximo en |x|: {max(abs(sol[:,0]))}')
    print(f'Desplazamiento maximo en |y|: {max(abs(sol[:,2]))}')
    print(f'Desplazamiento maximo en |z|: {max(abs(sol[:,4]))} \n')

    print(f'Desviado {leyendas[i]}') 
    print(f'Desplazamiento maximo en |x|: {max(abs(solrk[:,0]))}')
    print(f'Desplazamiento maximo en |y|: {max(abs(solrk[:,2]))}')
    print(f'Desplazamiento maximo en |z|: {max(abs(solrk[:,4]))} \n')


fig.legend()
fig1.suptitle('Trayectoria según la rotación')
ax.set_xlabel('Eje x (m)')
ax.set_ylabel('Eje z (m)')
ax.set_zlabel('Eje y (m)')
ax.set_title('Comparación de rotaciones')
# fig.savefig('Rotacion_3d.png', dpi = 500)
# fig1.savefig('Rotacion_trayectorias.png', dpi = 500)

plt.show()

