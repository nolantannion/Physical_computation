import numpy as np
import matplotlib.pyplot as plt
from funciones_t4 import euler_cromer, euler_cromerr


# Variables y constantes
g = 9.81    # Aceleracion de la gravedad
l = 1   # Longitud del pendulo
roz = 0.5   # Coeficiente de rozamiento
omega_forz = 2/3  # Frecuencia de la fuerza impulsora
Af = 0.5  # Amplitud de la fuerza impulsora
fi = 0  # Desfase de la fuerza impulsora

rozamientos = [0.5, 5, 12]


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

def sistema_rozamiento(t,vars, roz):
    '''
    INPUT
    - t: tiempo 
    - vars: variables a derivar
    - roz: rozamiento del sistema

    OUTPUT
    - dvars: derivadas de las variables contendias en vars
    '''
    # Creamos un array del mismo tamaño que las variables
    dvars = np.zeros_like(vars)

    # Derivadas de cada variable
    dvars[0] = vars[1]
    dvars[1] = -g/l * vars[0] - roz*vars[1]

    return dvars

def sistema_roz_forzado(t,vars, roz):
    '''
    INPUT
    - t: tiempo 
    - vars: variables a derivar
    - roz: rozamiento del sistema

    OUTPUT
    - dvars: derivadas de las variables contendias en vars
    '''
    # Creamos un array del mismo tamaño que las variables
    dvars = np.zeros_like(vars)

    # Derivadas de cada variable
    dvars[0] = vars[1]
    dvars[1] = -g/l * vars[0] - roz*vars[1] + Af*np.sin(omega_forz*t)

    return dvars


# Integracion
t0 = 0
tmax = 15
dt = 1e-3

# Condiciones iniciales
theta0 = np.deg2rad(3)
omega0 = 0
estado0 = [theta0, omega0]

# Solucion ideal, con rozamiento y rozamiento forzado

# Panel de figuras
fig, ax = plt.subplots(nrows=1, ncols=3, figsize= (10,6))

ts, sols = euler_cromer(sistema=sistema_ideal, t0 = t0, tf = tmax, h=dt, estado0= estado0)
ax[0].plot(ts,sols[:,0], label = 'Ideal')
ax[0].grid()
ax[0].legend()
ax[0].set_xlabel('tiempo (s)')
ax[0].set_ylabel(r'$\theta$ (rad)')

for rozamiento in rozamientos: 
    tr, solr = euler_cromerr(sistema=sistema_rozamiento, t0 = t0, tf = tmax, h=dt, estado0= estado0, roz = rozamiento)
    trf, solrf = euler_cromerr(sistema=sistema_roz_forzado, t0 = t0, tf = tmax, h=dt, estado0= estado0, roz = rozamiento)

    ax[1].plot(tr,solr[:,0], label = f'r,r = {rozamiento} ')
    ax[2].plot(trf,solrf[:,0], label = f'rf, r = {rozamiento} ')

    ax[1].grid()
    ax[1].legend()
    ax[1].set_xlabel('tiempo (s)')
    ax[1].set_ylabel(r'$\theta$ (rad)')

    ax[2].grid()
    ax[2].legend()
    ax[2].set_xlabel('tiempo (s)')
    ax[2].set_ylabel(r'$\theta$ (rad)')


plt.show()