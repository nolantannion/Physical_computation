import numpy as np
import matplotlib.pyplot as plt
from funciones_t4 import euler_cromerr
from matplotlib.animation import FuncAnimation
from numpy.fft import rfft
'''
Incluye una representacion del espacio de fases de cada pendulo
'''

# Variables y constantes
g = 9.81    # Aceleracion de la gravedad
l = 9.81   # Longitud del pendulo
roz = 1/2   # Coeficiente de rozamiento
omega_forz = 2/3  # Frecuencia de la fuerza impulsora
Af = 1.2  # Amplitud de la fuerza impulsora
fi = 0  # Desfase de la fuerza impulsora

# Parametros del sistema con rozamiento
parametros = [Af,omega_forz, roz]

def sistema(t,vars, params):
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
    dvars[1] = -g/l * np.sin(vars[0]) - params[2]*vars[1] + params[0]*np.sin(params[1]*t)

    return dvars


# Parametros de integracion
t0 = 0
tmax = 100
# Se puede variar dt pero se incluye como un parametro fijo en la funcion
# con valor 1e-3
# dt = 1e-3

# Calculos para el pendulo 1
theta01 = 0.20
omega01 = 0.0
estado1 = [theta01, omega01]

t1, sol1 = euler_cromerr(sistema=sistema, t0 = t0, tf = tmax, estado0= estado1, params= parametros, h = 1e-2)

theta1 = sol1[:,0]
omega1 = sol1[:,1]
theta1norm = (theta1 + np.pi) % (2*np.pi) - np.pi

# Indice del 20% del array
N =int( 0.2* len(t1))


thetasol = [theta1norm]
omegasol = [sol1[:,1]]
# Calculo de los otros pendulos y representacion
rozamientos = np.linspace(0.48,0.52,3)
for rozm in rozamientos:

    theta02 = 0.3
    omega02 = 0.0
    estado2 = [theta02, omega02]
    parametros[2] = rozm
    t2, sol2 = euler_cromerr(sistema=sistema, t0 = t0, tf = tmax, estado0= estado2, params= parametros, h = 1e-2)

    # Array con la solucion de cada variable
    theta2 = sol2[:,0]
    omega2 = sol2[:,1]

    # Normalizamos los angulos
    theta2norm = (theta2 + np.pi) % (2*np.pi) - np.pi

    # Añadimos la solucion a nuestro array de soluciones
    omegasol.append(omega2)
    thetasol.append(theta2norm)


# Creamos un array que contenga las leyendas
leyendas = ['Pendulo 1']
for roz in rozamientos:
    leyendas.append(f'r: {roz}')

# Representamos el espacio de fases para cada caso
fig, ax = plt.subplots(nrows=2, ncols=2, figsize = (10,6))
figura, eje = plt.subplots(figsize = (7,7))
fig.subplots_adjust(wspace=0.4, hspace=0.4)
j = 0
k = 0
for i in range(len(thetasol)):
    theta = thetasol[i]
    omega = omegasol[i]

    if j>1:
        j = 0
        k = 1

    eje.scatter(theta,omega, s = 0.4,  label = f'{leyendas[i]}')

    # Representamos el espacio de fases usando un gradiente de color para
    # indicar la evolucion temporal
    e_fases = ax[j,k].scatter(theta, omega, s=0.4, label = f'{leyendas[i]}', c = t1, cmap = 'viridis')
    ax[j,k].legend()
    ax[j,k].set_xlabel(r"$\theta$")
    ax[j,k].set_ylabel(r"$\dot{\theta}$")
    fig.colorbar(e_fases, ax = ax[j,k], label = 't') # Barra que muestra el valor del gradiente
    j += 1


fig.suptitle('Espacio de fases')
eje.set_xlim(-0.5,0.5)
eje.axis('tight')
eje.set_xlabel(r"$\theta$")
eje.set_ylabel(r"$\dot{\theta}$")
eje.set_title('Diagrama de Fases')
eje.legend()

# Almacenamos las figuras y las mostramos
#figura.savefig('Esp_de_fases.png', dpi = 300)
#fig.savefig('sube_de_fases.png', dpi = 500)

plt.show()