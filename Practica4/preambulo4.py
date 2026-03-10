import numpy as np
import matplotlib.pyplot as plt
from funciones_t4 import euler_cromer
'''
Caso con resonancia en la fuerza impulsora y rozamiento bajo
'''

# Variables y constantes
g = 9.81    # Aceleracion de la gravedad
l = 1   # Longitud del pendulo
roz = 0.5   # Coeficiente de rozamiento
omega_forz = np.sqrt(9.81)  # Frecuencia de la fuerza impulsora
Af = 0.5  # Amplitud de la fuerza impulsora
fi = 0  # Desfase de la fuerza impulsora


# vars tiene el formato [theta, omega]
def sistema_ideal(t,vars):
    '''
    INPUT
    - t: tiempo 
    - vars: variables a derivar

    OUTPUT
    - dvars: derivadas de las variables contendias en vars
    '''
    # Creamos un array del mismo tamaño que las variables
    dvars = np.zeros_like(vars)

    # Derivadas de cada variable
    dvars[0] = vars[1]
    dvars[1] = -g/l * vars[0]

    return dvars

def sistema_rozamiento(t,vars):
    '''
    INPUT
    - t: tiempo 
    - vars: variables a derivar

    OUTPUT
    - dvars: derivadas de las variables contendias en vars
    '''
    # Creamos un array del mismo tamaño que las variables
    dvars = np.zeros_like(vars)

    # Derivadas de cada variable
    dvars[0] = vars[1]
    dvars[1] = -g/l * vars[0] - roz*vars[1]

    return dvars

def sistema_roz_forzado(t,vars):
    '''
    INPUT
    - t: tiempo 
    - vars: variables a derivar

    OUTPUT
    - dvars: derivadas de las variables contendias en vars
    '''
    # Creamos un array del mismo tamaño que las variables
    dvars = np.zeros_like(vars)

    # Derivadas de cada variable
    dvars[0] = vars[1]
    dvars[1] = -g/l * np.sin(vars[0]) - roz*vars[1] + Af*np.sin(omega_forz*t)

    return dvars


# Integracion
t0 = 0
tmax = 15
dt = 1e-3

# Condiciones iniciales
theta0 = np.deg2rad(45)
omega0 = 0
estado0 = [theta0, omega0]

# Solucion ideal, con rozamiento y rozamiento forzado
ts, sols = euler_cromer(sistema=sistema_ideal, t0 = t0, tf = tmax, h=dt, estado0= estado0)
tr, solr = euler_cromer(sistema=sistema_rozamiento, t0 = t0, tf = tmax, h=dt, estado0= estado0)
trf, solrf = euler_cromer(sistema=sistema_roz_forzado, t0 = t0, tf = tmax, h=dt, estado0= estado0)

# Panel de figuras
fig, ax = plt.subplots(nrows=1, ncols=3, figsize= (8,6))
ax[0].plot(ts,sols[:,0], label = 'Ideal')
ax[1].plot(tr,solr[:,0], label = 'Rozamiento')
ax[2].plot(trf,solrf[:,0], label = 'Rozamiento y forzado')


ax[0].grid()
ax[0].legend()
ax[0].set_xlabel('tiempo (s)')
ax[0].set_ylabel(r'$\theta$ (rad)')

ax[1].grid()
ax[1].legend()
ax[1].set_xlabel('tiempo (s)')
ax[1].set_ylabel(r'$\theta$ (rad)')

ax[2].grid()
ax[2].legend()
ax[2].set_xlabel('tiempo (s)')
ax[2].set_ylabel(r'$\theta$ (rad)')

# Todos los movimientos en un grafico
fig1, ax1 = plt.subplots()
ax1.set_xlabel('tiempo (s)')
ax1.set_ylabel(r'$\theta$ (rad)')

ax1.plot(ts,sols[:,0], label = 'Ideal')
ax1.plot(tr,solr[:,0], label = 'Rozamiento')
ax1.plot(trf,solrf[:,0], label = 'Rozamiento y forzado')
plt.grid()
plt.legend()




plt.show()