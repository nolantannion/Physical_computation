# Práctica 3
'''
Para la realizacion de la practica base se considera una densidad del aire constante
y se desprecia el arrastre sufrido en las direcciones z e y.
'''

import numpy as np 
import matplotlib.pyplot as plt
from funciones_pr3 import euler, arrastrex, rho_const


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


def sistema(vars, rho, sign):
    '''
    Función que define la dinámica del sistema.

    INPUTS
    
    - vars: array con estructura [x,vx,y,vy,z,vz]
    - rho: función para calcular la densidad en función de la altura 
    - sign: array de valor  +1 o -1 que indica el sentido de rotación de la pelota

    RETURNS
    - dvars: array que contiene la derivada de cada variable de vars
    '''
    
    # Creamos el array que almacena las derivadas de cada variable
    dvars = np.zeros_like(vars)
    
    # Modulo de la velocidad 
    modv = np.sqrt(vars[1]**2 + vars[3]**2 + vars[5]**2)

    # Coeficiente de arrastre como función de v
    C = arrastrex(var=vars)
    
    # Calculamos la derivada de cada variable de posición
    dvars[0] = vars[1]
    dvars[2] = vars[3]
    dvars[4] = vars[5]
    
    arrastre = -(1/2)*rho*A/m *modv 
    dvars[1] = arrastre*vars[1]*C[0]
    dvars[3] = -g + arrastre*vars[3]*C[1]
    dvars[5] = S0wm*vars[1] * sign + arrastre*vars[5]*C[2]
    
    
    return dvars



# Array para la direccion de giro y leyenda
spin = [1,-1]
leyendas = ['Hook','Slice']


# Angulo del golpeo de la bola
ang = np.deg2rad(15)
direccion = [np.cos(ang),np.sin(ang), 0]


# Condiciones iniciales
estado = [0, v0*np.cos(ang),0,v0*np.sin(ang),0,0]


# Figura con los ejes en 3D y una figura 2D
ax = plt.figure().add_subplot(projection  = '3d')
plt.figure()

# Cálculo y representación de la trayectoria para cada dirección de giro
for i, sign in enumerate(spin):

    t, sol = euler(sistema,t0,tmax, h, estado, sign, rho_const)

    # Representamos la trayectoria en funcion de la rotacion
    ax.plot(sol[:,0], sol[:,4], sol[:,2], label = f'{leyendas[i]}') # Cambiamos el orden del plot para mejorar la visualizacion

    # Mostramos en pantalla informacion del maximo alcance en cada direccion
    print(f'{leyendas[i]}') 
    print(f'Desplazamiento maximo en |x|: {max(abs(sol[:,0]))}')
    print(f'Desplazamiento maximo en |y|: {max(abs(sol[:,2]))}')
    print(f'Desplazamiento maximo en |z|: {max(abs(sol[:,4]))} \n')
    
    # Desplazamiento en z
    plt.plot(t,sol[:,4], label = f'{leyendas[i]}')


# Parametros para la estetica de la representacion
ax.legend()
ax.set_xlabel('Eje x (m)')
ax.set_ylabel('Eje z (m)')
ax.set_zlabel('Eje y (m)')
ax.set_title('Comparación de trayectorias Hook y Slice')

# plt.savefig('Trayectoria_3d.png', dpi = 500)


# Parametros para el desplazamiento lateral en z 

plt.xlabel('Tiempo (s)')
plt.ylabel('Desplazamiento  en z (m)')
plt.title('Desplazamiento lateral ')
plt.legend()
plt.grid()
plt.savefig('z_base.png', dpi = 500)
plt.show()



